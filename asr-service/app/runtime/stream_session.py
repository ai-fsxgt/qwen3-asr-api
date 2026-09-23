"""标准模式实时预览：在线 VAD 断句 + 内存 ASR 解码。

实时链路只生成文字预览，不执行说话人分离和声纹识别。
所有阻塞推理经线程池执行，避免阻塞 WebSocket 事件循环。
"""
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import app.config as cfg
from app.engines.streaming_vad_engine import StreamingVADEngine
from app.runtime.noise_gate import NoiseFloorTracker, rms_dbfs, should_gate
from app.utils.audio_resampler import pcm_bytes_to_array, resample_to_16k
from app.utils.result_parser import extract_text, extract_words
from app.utils.validation import (
    ENERGY_FLOOR_RANGE,
    MAX_END_SILENCE_RANGE,
    MAX_SEGMENT_SEC_RANGE,
    SNR_MIN_RANGE,
    coerce_num_in_range,
    parse_bool,
)

logger = logging.getLogger(__name__)

_TARGET_SR = 16000
_MIN_AUDIO_FS = 8000
_MAX_AUDIO_FS = 96000
_IDLE_KEEP_MS = 5000


class AudioBuffer:
    """累积 16kHz float32 单声道样本，按绝对毫秒切片并释放已消费部分。"""

    def __init__(self, sr: int = _TARGET_SR):
        self.sr = sr
        self._chunks: list[np.ndarray] = []
        self._total = 0
        self._base_ms = 0

    def append(self, arr: np.ndarray):
        if arr is not None and arr.size:
            chunk = np.asarray(arr, dtype=np.float32)
            self._chunks.append(chunk)
            self._total += chunk.size

    def _merged(self) -> np.ndarray:
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
        return self._chunks[0] if self._chunks else np.zeros(0, dtype=np.float32)

    def _ms_to_idx(self, ms: int) -> int:
        idx = int((ms - self._base_ms) * self.sr / 1000)
        return max(0, min(idx, self._total))

    @property
    def base_ms(self) -> int:
        return self._base_ms

    @property
    def end_ms(self) -> int:
        return self._base_ms + int(self._total * 1000 / self.sr)

    def slice_ms(self, start_ms, end_ms) -> np.ndarray:
        if start_ms is None:
            start_ms = self._base_ms
        start = self._ms_to_idx(int(start_ms))
        end = self._ms_to_idx(int(end_ms))
        return self._merged()[start:end]

    def drop_until_ms(self, ms: int):
        idx = self._ms_to_idx(int(ms))
        if idx <= 0:
            return
        buf = self._merged()[idx:]
        self._chunks = [buf] if buf.size else []
        self._total = int(buf.size)
        self._base_ms += int(idx * 1000 / self.sr)


class StreamSession:
    """单个 WebSocket 会话：缓冲音频、VAD 断句并输出文字 final。"""

    def __init__(
        self,
        sid,
        svad: StreamingVADEngine,
        asr,
        punc,
        executor,
        asr_sem,
        *,
        language=None,
        max_segment_sec=30,
        enable_words=False,
        noise_filter=False,
        energy_floor_dbfs=-50.0,
        snr_min_db=6.0,
        priority_gate=None,
    ):
        self.sid = sid
        self._svad = svad
        self._asr = asr
        self._punc = punc
        self._executor = executor
        self._asr_sem = asr_sem
        self._default_max_segment_sec = max_segment_sec
        self._max_segment_sec = max_segment_sec
        self._enable_words = enable_words
        self._default_noise_filter = noise_filter
        self._default_energy_floor_dbfs = energy_floor_dbfs
        self._default_snr_min_db = snr_min_db
        self._noise_filter = noise_filter
        self._energy_floor_dbfs = energy_floor_dbfs
        self._snr_min_db = snr_min_db
        self._noise_tracker = None
        self._priority_gate = priority_gate

        self._max_end_silence_ms = cfg.VAD_MAX_SILENCE
        self._with_punc = True
        self._with_words = True
        self.audio_fs = _TARGET_SR
        self.language = language
        self.wav_name = sid
        self.vad_cache = None
        self.seg_id = 0
        self.seg_start_ms = None
        self.buffer = None
        self._frame_count = 0

    def configure(self, cfg_msg: dict):
        cfg_msg = cfg_msg or {}
        raw_fs = cfg_msg.get("audio_fs", _TARGET_SR)
        try:
            audio_fs = int(raw_fs)
        except (TypeError, ValueError):
            raise ValueError(f"audio_fs 非法: {raw_fs!r}")
        if not (_MIN_AUDIO_FS <= audio_fs <= _MAX_AUDIO_FS):
            raise ValueError(
                f"audio_fs 必须在 [{_MIN_AUDIO_FS}, {_MAX_AUDIO_FS}] 范围内，收到 {audio_fs}"
            )

        self.audio_fs = audio_fs
        if cfg_msg.get("language") is not None:
            self.language = cfg_msg.get("language")
        self.wav_name = cfg_msg.get("wav_name", self.sid)
        self.vad_cache = self._svad.new_cache()
        self.seg_id = 0
        self.seg_start_ms = None
        self.buffer = AudioBuffer(sr=_TARGET_SR)
        self._frame_count = 0
        self._max_segment_sec = self._default_max_segment_sec
        self._max_end_silence_ms = cfg.VAD_MAX_SILENCE
        self._with_punc = True
        self._with_words = True
        self._noise_filter = self._default_noise_filter
        self._energy_floor_dbfs = self._default_energy_floor_dbfs
        self._snr_min_db = self._default_snr_min_db
        self._apply_session_override(cfg_msg)
        self._noise_tracker = NoiseFloorTracker() if self._noise_filter else None
        warnings = self._collect_ignored_params(cfg_msg)
        logger.info(
            f"[stream] 会话配置 sid={self.sid[:8]} audio_fs={self.audio_fs} "
            f"language={self.language} wav={self.wav_name} 忽略项={warnings or '无'}"
        )
        return warnings

    def _apply_session_override(self, cfg_msg: dict):
        self._noise_filter = parse_bool(
            cfg_msg.get("noise_filter"), self._noise_filter, "noise_filter"
        )
        energy_floor = cfg_msg.get("energy_floor_dbfs")
        if energy_floor is not None:
            self._energy_floor_dbfs = coerce_num_in_range(
                energy_floor, ENERGY_FLOOR_RANGE, "energy_floor_dbfs"
            )
        snr_min = cfg_msg.get("snr_min_db")
        if snr_min is not None:
            self._snr_min_db = coerce_num_in_range(
                snr_min, SNR_MIN_RANGE, "snr_min_db"
            )
        end_silence = cfg_msg.get("max_end_silence_ms")
        if end_silence is not None:
            self._max_end_silence_ms = coerce_num_in_range(
                end_silence,
                MAX_END_SILENCE_RANGE,
                "max_end_silence_ms",
                cast=int,
            )
        max_segment = cfg_msg.get("max_segment_sec")
        if max_segment is not None:
            self._max_segment_sec = coerce_num_in_range(
                max_segment,
                MAX_SEGMENT_SEC_RANGE,
                "max_segment_sec",
                cast=int,
            )
        self._with_punc = parse_bool(
            cfg_msg.get("with_punc"), self._with_punc, "with_punc"
        )
        self._with_words = parse_bool(
            cfg_msg.get("with_words"), self._with_words, "with_words"
        )

    def _collect_ignored_params(self, cfg_msg: dict) -> list[str]:
        ignored = []
        if cfg_msg.get("with_words") is True and not self._enable_words:
            ignored.append("with_words")
        if cfg_msg.get("with_punc") is True and self._punc is None:
            ignored.append("with_punc")
        return ignored

    async def _in_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def feed_audio(self, pcm_bytes):
        """喂入一块 PCM16，按句异步产出 final 信封。"""
        if self.buffer is None:
            self.configure({})
        arr = pcm_bytes_to_array(pcm_bytes)
        if arr.size == 0:
            return
        if self.audio_fs != _TARGET_SR:
            arr = await self._in_thread(resample_to_16k, arr, self.audio_fs)
        self.buffer.append(arr)

        self._frame_count += 1
        events = await self._in_thread(
            self._svad.process_chunk,
            arr,
            self.vad_cache,
            False,
            self._max_end_silence_ms,
        )
        if events:
            logger.debug(
                f"[stream] frame#{self._frame_count} 收到{arr.size}样本 VAD事件={events}"
            )
        elif self._frame_count % 16 == 0:
            logger.debug(
                f"[stream] frame#{self._frame_count} 心跳 "
                f"buffer={self.buffer.end_ms - self.buffer.base_ms}ms "
                f"end_ms={self.buffer.end_ms} seg_start={self.seg_start_ms}"
            )
        for event in events:
            async for message in self._on_event(event):
                yield message

        if (
            self.seg_start_ms is not None
            and self.buffer.end_ms - self.seg_start_ms
            >= self._max_segment_sec * 1000
        ):
            end_ms = self.buffer.end_ms
            logger.info(
                f"[stream] 长句兜底切分 seg_start={self.seg_start_ms} end={end_ms}"
            )
            async for message in self._emit_final(self.seg_start_ms, end_ms):
                yield message
            self.seg_start_ms = end_ms
            self.buffer.drop_until_ms(end_ms)

        if self.seg_start_ms is None:
            if self._noise_tracker is not None:
                self._noise_tracker.update(arr)
            keep_from = self.buffer.end_ms - _IDLE_KEEP_MS
            if keep_from > self.buffer.base_ms:
                self.buffer.drop_until_ms(keep_from)

    async def flush(self):
        """收到 stop 时冲刷未闭合句。"""
        if self.buffer is None:
            return
        logger.debug(
            f"[stream] flush sid={self.sid[:8]} 总帧数={self._frame_count} "
            f"seg_start={self.seg_start_ms}"
        )
        try:
            events = await self._in_thread(
                self._svad.process_chunk,
                np.zeros(0, dtype=np.float32),
                self.vad_cache,
                True,
                self._max_end_silence_ms,
            )
        except Exception as exc:
            logger.warning(f"[stream] VAD final 冲刷失败，继续冲刷剩余缓冲: {exc}")
            events = []
        for event in events:
            async for message in self._on_event(event):
                yield message
        if self.seg_start_ms is not None:
            async for message in self._emit_final(
                self.seg_start_ms, self.buffer.end_ms
            ):
                yield message
            self.seg_start_ms = None

    async def _on_event(self, event):
        event_type = event["type"]
        if event_type == "start":
            self.seg_start_ms = event["start"]
            return
        if event_type not in ("end", "complete"):
            return
        start_ms = event["start"] if event_type == "complete" else self.seg_start_ms
        if start_ms is None:
            start_ms = self.buffer.base_ms
        async for message in self._emit_final(start_ms, event["end"]):
            yield message
        self.seg_start_ms = None
        self.buffer.drop_until_ms(event["end"])

    async def _emit_final(self, start_ms, end_ms):
        segment = self.buffer.slice_ms(start_ms, end_ms)
        if segment is None or segment.size == 0:
            logger.debug(f"[stream] 跳过空段 start={start_ms} end={end_ms}")
            return
        if self._noise_tracker is not None:
            segment_dbfs = rms_dbfs(segment)
            floor = self._noise_tracker.floor_dbfs
            gated, reason = should_gate(
                segment_dbfs,
                floor,
                energy_floor_dbfs=self._energy_floor_dbfs,
                snr_min_db=self._snr_min_db,
            )
            if gated:
                logger.debug(
                    f"[stream] 远场/噪声丢弃 段[{int(start_ms)},{int(end_ms)}]"
                    f"={int(end_ms - start_ms)}ms dbfs={segment_dbfs:.1f} "
                    f"floor={'—' if floor is None else f'{floor:.1f}'} 门={reason}"
                )
                return

        started_at = time.monotonic()
        if self._priority_gate is None:
            async with self._asr_sem:
                result = await self._in_thread(
                    self._asr.transcribe_array,
                    segment,
                    _TARGET_SR,
                    self.language,
                )
        else:
            with self._priority_gate.realtime_section():
                async with self._asr_sem:
                    result = await self._in_thread(
                        self._asr.transcribe_array,
                        segment,
                        _TARGET_SR,
                        self.language,
                    )
        decode_ms = (time.monotonic() - started_at) * 1000
        text = extract_text(result)
        if self._punc is not None and self._with_punc and text.strip():
            try:
                text = await self._in_thread(self._punc.restore, text)
            except Exception as exc:
                logger.warning(f"标点恢复失败，使用原始文本: {exc}")

        logger.info(
            f"[stream] final#{self.seg_id} 段[{int(start_ms)},{int(end_ms)}]"
            f"={int(end_ms - start_ms)}ms 样本={segment.size} "
            f"解码={decode_ms:.0f}ms 文本长度={len(text)}"
        )
        message = {
            "type": "final",
            "seg_id": self.seg_id,
            "text": text,
            "start": int(start_ms),
            "end": int(end_ms),
        }
        if self._enable_words and self._with_words:
            words = extract_words(result, int(start_ms) / 1000.0)
            if words:
                message["words"] = words
        self.seg_id += 1
        yield message


class VadOfflineBackend:
    """标准模式实时预览后端。"""

    mode = "standard"
    backend = "vad-offline"

    def __init__(
        self,
        asr,
        vad,
        punc=None,
        *,
        max_sessions=4,
        asr_concurrency=1,
        max_segment_sec=30,
        vad_chunk_ms=200,
        noise_filter=False,
        energy_floor_dbfs=-50.0,
        snr_min_db=6.0,
        priority_gate=None,
    ):
        self._svad = StreamingVADEngine(vad, chunk_ms=vad_chunk_ms)
        self._asr = asr
        self._punc = punc
        self._max_sessions = max_sessions
        self._max_segment_sec = max_segment_sec
        self._noise_filter = noise_filter
        self._energy_floor_dbfs = energy_floor_dbfs
        self._snr_min_db = snr_min_db
        self._priority_gate = priority_gate
        self._enable_words = bool(getattr(asr, "align_enabled", False))
        self._asr_sem = asyncio.Semaphore(asr_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max(2, asr_concurrency + 2),
            thread_name_prefix="stream-asr",
        )
        self._active = 0
        self._count_lock = threading.Lock()
        self.capabilities = {
            "partial_results": False,
            "word_timestamps": self._enable_words,
            "languages_auto": True,
            "speaker_labels": False,
            "speaker_identification": False,
            "noise_filter_tunable": True,
            "speaker_tunable": False,
            "endpoint_tunable": True,
            "output_toggles": True,
        }

    async def acquire(self) -> bool:
        with self._count_lock:
            if self._active >= self._max_sessions:
                return False
            self._active += 1
        return True

    def create_session(self, sid) -> StreamSession:
        return StreamSession(
            sid,
            self._svad,
            self._asr,
            self._punc,
            self._executor,
            self._asr_sem,
            max_segment_sec=self._max_segment_sec,
            enable_words=self._enable_words,
            noise_filter=self._noise_filter,
            energy_floor_dbfs=self._energy_floor_dbfs,
            snr_min_db=self._snr_min_db,
            priority_gate=self._priority_gate,
        )

    def release(self, session):
        if session is not None:
            session.buffer = None
            session.vad_cache = None
            session._noise_tracker = None
        with self._count_lock:
            self._active = max(0, self._active - 1)

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
