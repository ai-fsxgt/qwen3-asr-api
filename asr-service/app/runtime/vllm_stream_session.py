"""vLLM 模式实时文字预览：能量端点断句 + 原生流式解码。

实时链路只输出 partial 和 final 文本，不执行说话人分离和声纹识别。
"""
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import app.config as cfg
from app.runtime.noise_gate import rms_dbfs
from app.utils.audio_resampler import pcm_bytes_to_array, resample_to_16k
from app.utils.validation import coerce_num_in_range

logger = logging.getLogger(__name__)

_TARGET_SR = 16000
_MIN_AUDIO_FS = 8000
_MAX_AUDIO_FS = 96000
CHUNK_SIZE_SEC_RANGE = (0.5, 5.0)

_UNSUPPORTED_KEYS = (
    "with_words",
    "with_punc",
    "noise_filter",
    "energy_floor_dbfs",
    "snr_min_db",
    "max_end_silence_ms",
    "max_segment_sec",
    "identify_speakers",
    "diarize",
    "speaker_threshold",
    "speaker_min_seg_ms",
    "speaker_max",
    "speaker_id_threshold",
    "speaker_id_margin",
)


def _is_decoder_prompt_overflow(exc) -> bool:
    message = str(exc).lower()
    return "decoder prompt" in message and "maximum model length" in message


class EnergyEndpointer:
    """按帧能量判定语音开始，并在尾静音达到阈值时结束当前句。"""

    def __init__(self, *, energy_floor_dbfs=-45.0, end_silence_ms=800):
        self._floor = energy_floor_dbfs
        self._end_silence_ms = end_silence_ms
        self.reset()

    def reset(self):
        self._in_speech = False
        self._silence_ms = 0
        self._time_ms = 0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def is_active(self, arr) -> bool:
        return rms_dbfs(arr) >= self._floor

    def process(self, arr, frame_ms=None) -> list:
        duration_ms = (
            int(frame_ms)
            if frame_ms is not None
            else int(arr.size * 1000 / _TARGET_SR)
        )
        active = self.is_active(arr)
        events = []
        start_ms = self._time_ms
        self._time_ms += duration_ms
        if active:
            self._silence_ms = 0
            if not self._in_speech:
                self._in_speech = True
                events.append({"type": "start", "start": start_ms})
        elif self._in_speech:
            self._silence_ms += duration_ms
            if self._silence_ms >= self._end_silence_ms:
                self._in_speech = False
                events.append({"type": "end", "end": self._time_ms})
        return events


class VllmStreamSession:
    """单个 WebSocket 会话：句内输出 partial，句尾输出 final。"""

    def __init__(
        self,
        sid,
        engine,
        endpointer: EnergyEndpointer,
        executor,
        infer_sem,
        *,
        language=None,
        max_utterance_sec=20,
        max_state_sec=120,
        priority_gate=None,
    ):
        self.sid = sid
        self._engine = engine
        self._endpointer = endpointer
        self._executor = executor
        self._sem = infer_sem
        self._priority_gate = priority_gate
        self.language = language
        self._max_utt_samples = max(1, int(max_utterance_sec * _TARGET_SR))
        self._max_state_samples = max(1, int(max_state_sec * _TARGET_SR))

        self.audio_fs = _TARGET_SR
        self._chunk_size_sec = None
        self.state = None
        self.seg_id = 0
        self._seg_start_ms = None
        self._total_ms = 0
        self._utt_samples = 0
        self._state_samples = 0
        self._last_partial = ""
        self._generation = 0

    def configure(self, cfg_msg: dict) -> list:
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
        chunk_size_sec = cfg_msg.get("chunk_size_sec")
        if chunk_size_sec is not None:
            self._chunk_size_sec = coerce_num_in_range(
                chunk_size_sec,
                CHUNK_SIZE_SEC_RANGE,
                "chunk_size_sec",
            )

        self._generation += 1
        self._endpointer.reset()
        self.state = None
        self.seg_id = 0
        self._seg_start_ms = None
        self._total_ms = 0
        self._utt_samples = 0
        self._state_samples = 0
        self._last_partial = ""
        warnings = [key for key in _UNSUPPORTED_KEYS if cfg_msg.get(key) is not None]
        logger.info(
            f"[vllm-stream] 会话配置 sid={self.sid[:8]} audio_fs={self.audio_fs} "
            f"language={self.language} chunk={self._chunk_size_sec or '默认'} "
            f"hard_cut={self._max_utt_samples / _TARGET_SR:.0f}s "
            f"state_cut={self._max_state_samples / _TARGET_SR:.0f}s "
            f"忽略项={warnings or '无'}"
        )
        return warnings

    async def _in_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def _infer(self, fn, *args):
        if self._priority_gate is None:
            async with self._sem:
                return await self._in_thread(fn, *args)
        with self._priority_gate.realtime_section():
            async with self._sem:
                return await self._in_thread(fn, *args)

    def _begin_ui_segment(self, start_ms):
        self._seg_start_ms = start_ms
        self._utt_samples = 0
        self._last_partial = ""

    async def _begin_segment(self, start_ms):
        if self.state is None:
            self.state = await self._in_thread(
                self._engine.new_state,
                self.language,
                self._chunk_size_sec,
            )
            self._state_samples = 0
        self._begin_ui_segment(start_ms)

    def _clear_ui_segment(self):
        self._seg_start_ms = None
        self._utt_samples = 0
        self._last_partial = ""

    def _clear_state(self, *, discard_segment=False):
        self.state = None
        self._state_samples = 0
        if discard_segment:
            self._clear_ui_segment()

    def _make_final(self, text, start_ms, end_ms):
        message = {
            "type": "final",
            "seg_id": self.seg_id,
            "text": text or "",
            "start": int(start_ms or 0),
            "end": int(end_ms),
        }
        self.seg_id += 1
        self._clear_ui_segment()
        return message

    def _overflow_result(self, state, exc):
        logger.warning(
            f"[vllm-stream] SDK state 上下文超限，使用最后有效结果收尾 "
            f"sid={self.sid[:8]} state={self._state_samples / _TARGET_SR:.1f}s: {exc}"
        )
        return getattr(state, "text", ""), getattr(state, "language", None)

    async def _finish_state(self, state):
        try:
            return await self._infer(self._engine.finish, state)
        except Exception as exc:
            if not _is_decoder_prompt_overflow(exc):
                raise
            return self._overflow_result(state, exc)

    async def _finish_segment(self, end_ms, *, overflow=None):
        generation = self._generation
        state = self.state
        try:
            result = (
                await self._finish_state(state)
                if overflow is None
                else self._overflow_result(state, overflow)
            )
        except Exception:
            if generation == self._generation:
                self._clear_state(discard_segment=True)
            raise
        if generation != self._generation:
            return None
        text, _language = result
        message = None
        if self._seg_start_ms is None:
            logger.info(
                f"[vllm-stream] 关闭 SDK state，无前端分段输出 "
                f"sid={self.sid[:8]} end={end_ms}ms"
            )
        else:
            message = self._make_final(text, self._seg_start_ms, end_ms)
        self._clear_state()
        return message

    async def _feed_piece(self, arr):
        text, _language = await self._infer(self._engine.feed, arr, self.state)
        self._state_samples += arr.size
        if self._seg_start_ms is None:
            return None
        self._utt_samples += arr.size
        if text and text != self._last_partial:
            self._last_partial = text
            return {"type": "partial", "seg_id": self.seg_id, "text": text}
        return None

    def _next_piece_samples(self, remaining_samples):
        limits = [remaining_samples]
        if self.state is None:
            limits.append(self._max_state_samples)
        else:
            limits.append(max(1, self._max_state_samples - self._state_samples))
        if self._seg_start_ms is None:
            limits.append(self._max_utt_samples)
        else:
            limits.append(max(1, self._max_utt_samples - self._utt_samples))
        return max(1, min(limits))

    async def _ensure_active_state(self, start_ms):
        if self.state is None:
            await self._begin_segment(start_ms)
        elif self._seg_start_ms is None:
            self._begin_ui_segment(start_ms)

    async def _maybe_cut_after_piece(self, end_ms):
        if self.state is None:
            return None
        if self._state_samples >= self._max_state_samples:
            logger.info(
                f"[vllm-stream] SDK state 到期重建 sid={self.sid[:8]} end={end_ms}ms"
            )
            return await self._finish_segment(end_ms)
        if (
            self._seg_start_ms is not None
            and self._utt_samples >= self._max_utt_samples
        ):
            logger.info(
                f"[vllm-stream] 连续语音硬分段 sid={self.sid[:8]} end={end_ms}ms"
            )
            return await self._finish_segment(end_ms)
        return None

    async def feed_audio(self, pcm_bytes):
        arr = pcm_bytes_to_array(pcm_bytes)
        if arr.size == 0:
            return
        if self.audio_fs != _TARGET_SR:
            arr = await self._in_thread(resample_to_16k, arr, self.audio_fs)

        offset = 0
        while offset < arr.size:
            piece_size = self._next_piece_samples(arr.size - offset)
            piece = arr[offset: offset + piece_size]
            duration_ms = int(piece.size * 1000 / _TARGET_SR)
            active = self._endpointer.is_active(piece)
            events = self._endpointer.process(piece, frame_ms=duration_ms)

            start_events = [event for event in events if event["type"] == "start"]
            if start_events:
                await self._ensure_active_state(start_events[0]["start"])
            elif active and self._seg_start_ms is None:
                await self._ensure_active_state(self._total_ms)

            if self.state is not None:
                generation = self._generation
                try:
                    partial = await self._feed_piece(piece)
                except Exception as exc:
                    if generation != self._generation:
                        return
                    if not _is_decoder_prompt_overflow(exc):
                        self._clear_state(discard_segment=True)
                        raise
                    final = await self._finish_segment(self._total_ms, overflow=exc)
                    if final is not None:
                        yield final
                    if active and generation == self._generation:
                        await self._ensure_active_state(self._total_ms)
                        try:
                            partial = await self._feed_piece(piece)
                        except Exception as retry_exc:
                            self._clear_state(discard_segment=True)
                            if not _is_decoder_prompt_overflow(retry_exc):
                                raise
                            logger.error(
                                f"[vllm-stream] 全新 SDK state 仍上下文超限，"
                                f"丢弃当前音频块 sid={self.sid[:8]}: {retry_exc}"
                            )
                        else:
                            if partial is not None:
                                yield partial
                else:
                    if partial is not None:
                        yield partial

            self._total_ms += duration_ms
            offset += piece.size

            if self.state is not None and any(
                event["type"] == "end" for event in events
            ):
                final = await self._finish_segment(self._total_ms)
                if final is not None:
                    yield final
                continue

            final = await self._maybe_cut_after_piece(self._total_ms)
            if final is not None:
                yield final

    async def flush(self):
        if self.state is not None:
            final = await self._finish_segment(self._total_ms)
            if final is not None:
                yield final

    def close(self):
        self._generation += 1
        self._clear_state(discard_segment=True)


class VllmStreamBackend:
    """vLLM 模式实时文字预览后端。"""

    mode = "vllm"
    backend = "vllm-native"

    def __init__(
        self,
        engine,
        *,
        max_sessions=16,
        concurrency=1,
        max_utterance_sec=20,
        max_state_sec=120,
        energy_floor_dbfs=-45.0,
        end_silence_ms=800,
        priority_gate=None,
    ):
        self._engine = engine
        self._max_sessions = max_sessions
        self._max_utterance_sec = max_utterance_sec
        self._max_state_sec = max_state_sec
        self._energy_floor_dbfs = energy_floor_dbfs
        self._end_silence_ms = end_silence_ms
        self._priority_gate = priority_gate
        self._sem = asyncio.Semaphore(concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max(2, concurrency + 1),
            thread_name_prefix="vllm-asr",
        )
        self._active = 0
        self._count_lock = threading.Lock()
        self.capabilities = {
            "partial_results": True,
            "word_timestamps": False,
            "languages_auto": True,
            "speaker_labels": False,
            "speaker_identification": False,
            "noise_filter_tunable": False,
            "speaker_tunable": False,
            "endpoint_tunable": False,
            "output_toggles": False,
        }

    async def acquire(self) -> bool:
        with self._count_lock:
            if self._active >= self._max_sessions:
                return False
            self._active += 1
        return True

    def create_session(self, sid) -> VllmStreamSession:
        endpointer = EnergyEndpointer(
            energy_floor_dbfs=self._energy_floor_dbfs,
            end_silence_ms=self._end_silence_ms,
        )
        return VllmStreamSession(
            sid,
            self._engine,
            endpointer,
            self._executor,
            self._sem,
            max_utterance_sec=self._max_utterance_sec,
            max_state_sec=self._max_state_sec,
            priority_gate=self._priority_gate,
        )

    def release(self, session):
        if session is not None:
            session.close()
        with self._count_lock:
            self._active = max(0, self._active - 1)

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
