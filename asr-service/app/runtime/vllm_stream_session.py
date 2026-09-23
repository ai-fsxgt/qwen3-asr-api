"""vLLM 流式活动后端（路线 A）：能量端点断句 + vLLM 原生流式解码。

实现与 stream_session.py 同一 StreamBackend / session 鸭子接口（被 ws_routes.py 消费）：
    Backend: .mode / .backend / .capabilities ；async acquire() -> bool ；
             create_session(sid) -> session ；release(session) ；shutdown()
    session: configure(dict) -> warnings:list ；feed_audio(bytes) -> async-iter[dict] ；
             flush() -> async-iter[dict]
产出信封 dict：{type:"partial",seg_id,text} /
{type:"final",seg_id,text,start,end,speaker?,speaker_name?}（无 words）。

本模块不 import vLLM/qwen_asr（经 VLLMASREngine 鸭子调用），亦不 import stream_session
（其顶层依赖 funasr，vLLM 环境不含）。时间戳用累计样本计数；仅在实时说话人
功能启用时，为当前 UI 分段保留有界 PCM 快照。
"""
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import app.config as cfg
from app.runtime.energy_vad import EnergyVAD
from app.runtime.speaker_cluster import OnlineSpeakerClusterer, speaker_label
from app.runtime.speaker_identity import RealtimeIdentityTracker
from app.utils.audio_resampler import pcm_bytes_to_array, resample_to_16k
from app.runtime.noise_gate import rms_dbfs
from app.utils.validation import (
    SPK_ID_MARGIN_RANGE,
    SPK_ID_THRESHOLD_RANGE,
    SPK_MAX_RANGE,
    SPK_MIN_SEG_RANGE,
    SPK_THRESHOLD_RANGE,
    coerce_num_in_range,
    parse_bool,
)

logger = logging.getLogger(__name__)

_TARGET_SR = 16000
_MIN_AUDIO_FS = 8000
_MAX_AUDIO_FS = 96000
CHUNK_SIZE_SEC_RANGE = (0.5, 5.0)      # 与 vllm_asr_engine.clamp_chunk_size_sec 对齐
_NEW_CLUSTER_MIN_EVIDENCE_MS = 2250    # 两个完整重叠窗覆盖的唯一语音时长
_PENDING_SPEAKER_MAX_GAP_SEGMENTS = 4

_SPEAKER_VAD = EnergyVAD(
    energy_floor_dbfs=cfg.VLLM_ENERGY_FLOOR_DBFS,
)


def _is_decoder_prompt_overflow(exc) -> bool:
    message = str(exc).lower()
    return "decoder prompt" in message and "maximum model length" in message


# 客户端可能下发、但 vllm 模式仍不支持的参数 → 软提示忽略（不报错）
_UNSUPPORTED_KEYS = (
    "with_words", "with_punc",
    "noise_filter", "energy_floor_dbfs", "snr_min_db",
    "max_end_silence_ms", "max_segment_sec",
)


class _SegmentAudioBuffer:
    """当前 UI 分段的有界 PCM 块。

    append 只做一次精确大小拷贝，避免切片长期持有客户端大帧；合并延迟到声纹线程。
    last_active_samples 用于裁掉端点判停前的尾静音，内部短暂停顿仍完整保留。
    """

    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._last_active_samples = 0

    def append(self, arr: np.ndarray, *, active: bool):
        if arr is None or arr.size == 0:
            return
        chunk = np.asarray(arr, dtype=np.float32).copy()
        self._chunks.append(chunk)
        self._samples += int(chunk.size)
        if active:
            self._last_active_samples = self._samples

    def to_array(self) -> np.ndarray:
        if self._last_active_samples <= 0:
            return np.zeros(0, dtype=np.float32)
        merged = self._chunks[0] if len(self._chunks) == 1 else np.concatenate(self._chunks)
        return np.ascontiguousarray(merged[:self._last_active_samples])


class _SpeakerEvidence(list):
    """标记由真实声纹引擎生成的窗口证据，区别于测试替身的段级结果。"""


def _embed_speaker_segment(speaker, audio: _SegmentAudioBuffer):
    """按离线同款 VAD 滑窗提取实时说话人证据。"""
    wav = audio.to_array()
    if wav.size == 0:
        return None

    # 自定义/测试引擎若只实现段级接口，仍按一个完整段调用。
    if not hasattr(speaker, "embed_realtime_windows"):
        embedding = speaker.embed_realtime_segment(wav)
        duration_ms = int(wav.size * 1000 / _TARGET_SR)
        return [(embedding, duration_ms, wav)]

    from app.engines.speaker_embedding_engine import make_windows

    speech_spans = _SPEAKER_VAD.detect_array(wav, _TARGET_SR)
    windows = [
        window
        for start_ms, end_ms in speech_spans
        for window in make_windows(start_ms / 1000.0, end_ms / 1000.0)
    ]
    if not windows:
        return None

    embeddings = speaker.embed_realtime_windows(wav, windows)
    evidence = []
    covered_until = 0.0
    for (start, end), embedding in zip(windows, embeddings, strict=True):
        # 重叠窗仅把新增覆盖的音频累计为身份识别证据，避免同一语音重复计时。
        evidence_start = max(start, covered_until)
        covered_until = max(covered_until, end)
        start_sample = max(int(evidence_start * _TARGET_SR), 0)
        end_sample = min(int(end * _TARGET_SR), wav.size)
        identity_audio = np.ascontiguousarray(wav[start_sample:end_sample])
        evidence.append((
            embedding,
            int((end - start) * 1000),
            identity_audio,
        ))
    return _SpeakerEvidence(evidence)


def _map_speaker_identity(
    service,
    label,
    embedding,
    duration_sec,
    threshold,
    margin,
):
    """在线程池中执行同步声纹库查询，返回完整匹配结果。"""
    mapping = service.map_clusters(
        [{"label": label, "centroid": embedding, "dur_sec": duration_sec}],
        id_threshold=threshold,
        id_margin=margin,
    )
    if not mapping or not mapping[0].get("speaker_id"):
        return None
    return mapping[0]


class EnergyEndpointer:
    """按帧能量端点：能量越门限→句开始；尾静音累计≥end_silence_ms→句结束。

    每帧 = 一次 process(arr) 的整段 RMS（端点粒度 = 客户端推流块时长）。
    输入须为 float32 [-1,1)（满量程参考）；与 FSMN-VAD 等价产出 start/end 事件。
    """

    def __init__(self, *, energy_floor_dbfs=-45.0, end_silence_ms=800):
        self._floor = energy_floor_dbfs
        self._end_sil = end_silence_ms
        self.reset()

    def reset(self):
        self._in_speech = False
        self._silence_ms = 0
        self._t_ms = 0          # 会话内累计时间（ms）

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def is_active(self, arr) -> bool:
        return rms_dbfs(arr) >= self._floor

    def process(self, arr, frame_ms=None) -> list:
        """返回事件：[{'type':'start','start':ms}] / [{'type':'end','end':ms}] / []。"""
        dur = int(frame_ms) if frame_ms is not None else int(arr.size * 1000 / _TARGET_SR)
        active = self.is_active(arr)
        events, t0 = [], self._t_ms
        self._t_ms += dur
        if active:
            self._silence_ms = 0
            if not self._in_speech:
                self._in_speech = True
                events.append({"type": "start", "start": t0})
        elif self._in_speech:
            self._silence_ms += dur
            if self._silence_ms >= self._end_sil:
                self._in_speech = False
                events.append({"type": "end", "end": self._t_ms})
        return events


class VllmStreamSession:
    """单个 WS 会话：能量端点断句 + vLLM 流式解码 → 句内 partial / 句尾 final。"""

    def __init__(self, sid, engine, endpointer: EnergyEndpointer, executor, infer_sem,
                 *, language=None, max_utterance_sec=20, max_state_sec=120,
                 priority_gate=None, speaker=None, speaker_service=None,
                 speaker_executor=None):
        self.sid = sid
        self._engine = engine
        self._endpointer = endpointer
        self._executor = executor
        self._speaker_executor = speaker_executor or executor
        self._sem = infer_sem                  # asyncio.Semaphore：限同时解码会话数
        self._priority_gate = priority_gate
        self._speaker = speaker                # None = 实时说话人关闭
        self._speaker_service = speaker_service
        self.language = language
        self._max_utt_samples = max(1, int(max_utterance_sec * _TARGET_SR))
        self._max_state_samples = max(1, int(max_state_sec * _TARGET_SR))
        self._reset_speaker_options()
        # 会话态
        self.audio_fs = _TARGET_SR
        self._chunk_size_sec = None            # None=用引擎默认；configure 可按会话覆盖
        self.state = None                      # 当前 SDK 流式状态（None=无活动语音）
        self.seg_id = 0
        self._seg_start_ms = None
        self._total_ms = 0                     # 会话累计音频时长（ms）
        self._utt_samples = 0                  # 当前硬分段已喂样本数
        self._state_samples = 0                # 当前 SDK state 已喂样本数
        self._last_partial = ""
        self._segment_audio = None             # 当前硬分段 PCM（仅 diarize 开启时）
        self._spk_cluster = None               # configure() 时创建，会话级不回溯聚类
        self._spk_identity_tracker = RealtimeIdentityTracker(
            cfg.SPEAKER_REALTIME_CONFIRMATIONS
        )
        self._raw_output_labels: dict[str, str] = {}
        self._identity_output_labels: dict[str, str] = {}
        self._output_label_identities: dict[str, str] = {}
        self._used_output_labels: set[str] = set()
        self._pending_speaker_groups: list[dict] = []
        self._generation = 0                   # configure/close 后使在途说话人结果失效

    def _reset_speaker_options(self):
        self._spk_threshold = cfg.SPEAKER_THRESHOLD
        self._spk_min_seg_ms = cfg.SPEAKER_MIN_SEG_MS
        self._spk_max = cfg.SPEAKER_MAX
        self._spk_id_threshold = cfg.SPEAKER_ID_THRESHOLD
        self._spk_id_margin = cfg.SPEAKER_ID_MARGIN
        self._with_diarize = True
        self._identify = False

    def _apply_speaker_options(self, cfg_msg: dict):
        st = cfg_msg.get("speaker_threshold")
        if st is not None:
            self._spk_threshold = coerce_num_in_range(
                st, SPK_THRESHOLD_RANGE, "speaker_threshold")
        ms = cfg_msg.get("speaker_min_seg_ms")
        if ms is not None:
            self._spk_min_seg_ms = coerce_num_in_range(
                ms, SPK_MIN_SEG_RANGE, "speaker_min_seg_ms", cast=int)
        sx = cfg_msg.get("speaker_max")
        if sx is not None:
            self._spk_max = coerce_num_in_range(
                sx, SPK_MAX_RANGE, "speaker_max", cast=int)
        it = cfg_msg.get("speaker_id_threshold")
        if it is not None:
            self._spk_id_threshold = coerce_num_in_range(
                it, SPK_ID_THRESHOLD_RANGE, "speaker_id_threshold")
        im = cfg_msg.get("speaker_id_margin")
        if im is not None:
            self._spk_id_margin = coerce_num_in_range(
                im, SPK_ID_MARGIN_RANGE, "speaker_id_margin")
        self._with_diarize = parse_bool(
            cfg_msg.get("diarize"), self._with_diarize, "diarize")
        self._identify = parse_bool(
            cfg_msg.get("identify_speakers"), False, "identify_speakers")

    def _collect_ignored_params(self, cfg_msg: dict) -> list[str]:
        ignored = [k for k in _UNSUPPORTED_KEYS if cfg_msg.get(k) is not None]
        speaker_ready = self._speaker is not None
        for key in ("speaker_threshold", "speaker_min_seg_ms", "speaker_max"):
            if cfg_msg.get(key) is not None and not speaker_ready:
                ignored.append(key)
        if cfg_msg.get("diarize") is True and not speaker_ready:
            ignored.append("diarize")

        identification_ready = (
            speaker_ready
            and self._speaker_service is not None
            and self._with_diarize
        )
        for key in ("speaker_id_threshold", "speaker_id_margin"):
            if cfg_msg.get(key) is not None and not identification_ready:
                ignored.append(key)
        if cfg_msg.get("identify_speakers") is True and not identification_ready:
            ignored.append("identify_speakers")
        return ignored

    def configure(self, cfg_msg: dict) -> list:
        cfg_msg = cfg_msg or {}
        raw_fs = cfg_msg.get("audio_fs", _TARGET_SR)
        try:
            audio_fs = int(raw_fs)
        except (TypeError, ValueError):
            raise ValueError(f"audio_fs 非法: {raw_fs!r}")
        if not (_MIN_AUDIO_FS <= audio_fs <= _MAX_AUDIO_FS):
            raise ValueError(
                f"audio_fs 必须在 [{_MIN_AUDIO_FS}, {_MAX_AUDIO_FS}] 范围内，收到 {audio_fs}")
        self.audio_fs = audio_fs
        if cfg_msg.get("language") is not None:
            self.language = cfg_msg.get("language")
        css = cfg_msg.get("chunk_size_sec")
        if css is not None:                    # 会话级覆盖（D6），越界抛 ValueError → invalid_config
            self._chunk_size_sec = coerce_num_in_range(css, CHUNK_SIZE_SEC_RANGE, "chunk_size_sec")
        self._reset_speaker_options()
        self._apply_speaker_options(cfg_msg)
        # 重置会话态
        self._generation += 1
        self._endpointer.reset()
        self.state = None
        self.seg_id = 0
        self._seg_start_ms = None
        self._total_ms = 0
        self._utt_samples = 0
        self._state_samples = 0
        self._last_partial = ""
        self._segment_audio = None
        if self._speaker is not None and self._with_diarize:
            self._spk_cluster = OnlineSpeakerClusterer(
                threshold=self._spk_threshold,
                max_speakers=self._spk_max,
                min_seg_ms=self._spk_min_seg_ms,
            )
        else:
            self._spk_cluster = None
        self._spk_identity_tracker = RealtimeIdentityTracker(
            cfg.SPEAKER_REALTIME_CONFIRMATIONS
        )
        self._raw_output_labels = {}
        self._identity_output_labels = {}
        self._output_label_identities = {}
        self._used_output_labels = set()
        self._pending_speaker_groups = []
        warnings = self._collect_ignored_params(cfg_msg)
        logger.info(f"[vllm-stream] 会话配置 sid={self.sid[:8]} audio_fs={self.audio_fs} "
                    f"language={self.language} chunk={self._chunk_size_sec or '默认'} "
                    f"hard_cut={self._max_utt_samples / _TARGET_SR:.0f}s "
                    f"state_cut={self._max_state_samples / _TARGET_SR:.0f}s "
                    f"speaker={'开' if self._spk_cluster is not None else '关'} "
                    f"identify={'开' if self._identify and self._speaker_service is not None else '关'} "
                    f"忽略项={warnings or '无'}")
        return warnings

    async def _in_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def _in_speaker_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._speaker_executor, fn, *args)

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
        self._segment_audio = (
            _SegmentAudioBuffer() if self._spk_cluster is not None else None)

    async def _begin_segment(self, start_ms):
        if self.state is None:
            self.state = await self._in_thread(self._engine.new_state, self.language, self._chunk_size_sec)
            self._state_samples = 0
        self._begin_ui_segment(start_ms)

    def _clear_ui_segment(self):
        self._seg_start_ms = None
        self._utt_samples = 0
        self._last_partial = ""
        self._segment_audio = None

    def _clear_state(self, *, discard_segment=False):
        self.state = None
        self._state_samples = 0
        if discard_segment:
            self._clear_ui_segment()

    def _make_ui_final(self, full_text, start_ms, end_ms):
        msg = {"type": "final", "seg_id": self.seg_id, "text": full_text or "",
               "start": int(start_ms or 0), "end": int(end_ms)}
        self.seg_id += 1
        self._last_partial = ""
        return msg

    def _emit_ui_final(self, full_text, end_ms):
        audio = self._segment_audio
        generation = self._generation
        msg = self._make_ui_final(full_text, self._seg_start_ms, end_ms)
        self._clear_ui_segment()
        return msg, audio, generation

    async def _embed_segment(self, audio):
        if audio is None or self._speaker is None:
            return None
        try:
            return await self._in_speaker_thread(
                _embed_speaker_segment, self._speaker, audio)
        except Exception as exc:
            logger.warning(f"[vllm-stream] 说话人 embedding 失败，本段不标注: {exc}")
            return None

    async def _lookup_speaker_name(self, label, cluster, generation):
        service = self._speaker_service
        if service is None:
            return None
        identity_windows = cluster.take_identity_audio_windows(
            label,
            cfg.SPEAKER_IDENTIFY_MIN_SEC,
            _TARGET_SR,
            cfg.SPEAKER_REALTIME_CONFIRMATIONS,
        )
        if not identity_windows:
            return self._spk_identity_tracker.confirmed_identity(label)
        try:
            for identity_audio in identity_windows:
                embedding = await self._in_speaker_thread(
                    self._speaker.embed_realtime_identity_segment,
                    identity_audio,
                )
                duration_sec = len(identity_audio) / _TARGET_SR
                match = await self._in_speaker_thread(
                    _map_speaker_identity,
                    service,
                    label,
                    embedding,
                    duration_sec,
                    self._spk_id_threshold,
                    self._spk_id_margin,
                )
                if (
                    generation != self._generation
                    or cluster is not self._spk_cluster
                ):
                    return None
                self._spk_identity_tracker.observe(label, match)
        except Exception as exc:
            logger.warning(f"[vllm-stream] 声纹识别失败，本段仅保留匿名标签: {exc}")
            return None
        return self._spk_identity_tracker.confirmed_identity(label)

    def _next_output_label(self) -> str | None:
        for index in range(self._spk_max):
            label = speaker_label(index)
            if label not in self._used_output_labels:
                self._used_output_labels.add(label)
                return label
        return None

    def _display_label(
        self,
        raw_label: str,
        speaker_id: str | None = None,
    ) -> str:
        label = self._raw_output_labels.get(raw_label)
        if label is None:
            if raw_label in self._used_output_labels:
                label = self._next_output_label() or raw_label
            else:
                label = raw_label
                self._used_output_labels.add(label)
            self._raw_output_labels[raw_label] = label
        if speaker_id is None:
            return label

        identity_label = self._identity_output_labels.get(speaker_id)
        if identity_label is not None:
            return identity_label
        owner = self._output_label_identities.get(label)
        if owner is None or owner == speaker_id:
            identity_label = label
        else:
            identity_label = self._next_output_label() or label
        self._identity_output_labels[speaker_id] = identity_label
        self._output_label_identities[identity_label] = speaker_id
        return identity_label

    def _queue_pending_speaker_evidence(self, item, segment_id: int) -> dict:
        embedding, duration_ms, _identity_audio = item
        similarities = [
            float(group["centroid"] @ embedding)
            for group in self._pending_speaker_groups
        ]
        if similarities and max(similarities) >= self._spk_threshold:
            index = int(np.argmax(similarities))
            group = self._pending_speaker_groups[index]
            merged = (
                group["centroid"] * group["duration_ms"]
                + embedding * duration_ms
            )
            group["duration_ms"] += duration_ms
            group["centroid"] = merged / np.linalg.norm(merged)
        else:
            group = {
                "centroid": np.asarray(embedding).copy(),
                "duration_ms": duration_ms,
                "items": [],
                "last_segment_id": segment_id,
            }
            if len(self._pending_speaker_groups) >= self._spk_max:
                self._pending_speaker_groups.pop(0)
            self._pending_speaker_groups.append(group)
        group["items"].append((segment_id, item))
        group["last_segment_id"] = segment_id
        return group

    async def _annotate_final(self, msg, embedded, generation):
        if msg is None or generation != self._generation:
            return None
        if embedded is None:
            return msg
        cluster = self._spk_cluster
        if cluster is None:
            return msg
        self._pending_speaker_groups = [
            group
            for group in self._pending_speaker_groups
            if msg["seg_id"] - group["last_segment_id"]
            <= _PENDING_SPEAKER_MAX_GAP_SEGMENTS
        ]
        votes: dict[str, int] = {}
        touched_groups = []
        try:
            for embedding, duration_ms, identity_audio in embedded:
                if (
                    isinstance(embedded, _SpeakerEvidence)
                    and cluster.closest_label(embedding) is None
                ):
                    group = self._queue_pending_speaker_evidence(
                        (embedding, duration_ms, identity_audio),
                        msg["seg_id"],
                    )
                    if not any(existing is group for existing in touched_groups):
                        touched_groups.append(group)
                    continue
                label = cluster.assign(
                    embedding,
                    duration_ms,
                    identity_audio=(
                        identity_audio
                        if self._identify and self._speaker_service is not None
                        else None
                    ),
                )
                if label is not None:
                    votes[label] = votes.get(label, 0) + duration_ms

            for group in touched_groups:
                evidence_ms = sum(
                    len(item[2]) * 1000 / _TARGET_SR
                    for _segment_id, item in group["items"]
                )
                if (
                    len(group["items"]) < 2
                    or evidence_ms < max(
                        self._spk_min_seg_ms,
                        _NEW_CLUSTER_MIN_EVIDENCE_MS,
                    )
                ):
                    continue
                label = cluster.create_cluster(group["centroid"])
                if label is None:
                    continue
                for segment_id, item in group["items"]:
                    embedding, duration_ms, identity_audio = item
                    cluster.assign_to(
                        label,
                        embedding,
                        duration_ms,
                        identity_audio=(
                            identity_audio
                            if self._identify and self._speaker_service is not None
                            else None
                        ),
                    )
                    if segment_id == msg["seg_id"]:
                        votes[label] = votes.get(label, 0) + duration_ms
                self._pending_speaker_groups = [
                    pending
                    for pending in self._pending_speaker_groups
                    if pending is not group
                ]
        except Exception as exc:
            logger.warning(f"[vllm-stream] 说话人归簇失败，本段不标注: {exc}")
            return msg
        if not votes:
            return msg
        label = max(votes, key=votes.get)
        identity = None
        if self._identify and self._speaker_service is not None:
            try:
                identity = await self._lookup_speaker_name(
                    label,
                    cluster,
                    generation,
                )
            except Exception as exc:
                logger.warning(f"[vllm-stream] 声纹识别失败，本段仅保留匿名标签: {exc}")
                identity = None
            if generation != self._generation or cluster is not self._spk_cluster:
                return None
        speaker_id, name = identity if identity is not None else (None, None)
        msg["speaker"] = self._display_label(label, speaker_id)
        if name is not None:
            msg["speaker_name"] = name
        return msg

    def _overflow_result(self, state, exc):
        logger.warning(
            f"[vllm-stream] SDK state 上下文超限，使用最后有效结果收尾 "
            f"sid={self.sid[:8]} state={self._state_samples / _TARGET_SR:.1f}s: {exc}")
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
        audio = self._segment_audio
        state = self.state
        try:
            if overflow is None:
                finish_result, embedded = await asyncio.gather(
                    self._finish_state(state),
                    self._embed_segment(audio),
                )
            else:
                finish_result = self._overflow_result(state, overflow)
                embedded = await self._embed_segment(audio)
        except Exception:
            if generation == self._generation:
                self._clear_state(discard_segment=True)
            raise
        if generation != self._generation:
            return None
        text, _ = finish_result
        if self._seg_start_ms is None:
            logger.info(f"[vllm-stream] 关闭 SDK state，无前端分段输出 sid={self.sid[:8]} end={end_ms}ms")
            final = None
        else:
            final = self._emit_ui_final(text, end_ms)
        self._clear_state()
        if final is None:
            return None
        msg, _, final_generation = final
        return await self._annotate_final(msg, embedded, final_generation)

    async def _feed_piece(self, arr, *, active):
        text, _ = await self._infer(self._engine.feed, arr, self.state)
        self._state_samples += arr.size
        if self._seg_start_ms is None:
            return None
        if self._segment_audio is not None:
            self._segment_audio.append(arr, active=active)
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
            logger.info(f"[vllm-stream] SDK state 到期重建 sid={self.sid[:8]} end={end_ms}ms")
            return await self._finish_segment(end_ms)
        if self._seg_start_ms is not None and self._utt_samples >= self._max_utt_samples:
            logger.info(f"[vllm-stream] 连续语音硬分段 sid={self.sid[:8]} end={end_ms}ms")
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
            n = self._next_piece_samples(arr.size - offset)
            piece = arr[offset: offset + n]
            dur_ms = int(piece.size * 1000 / _TARGET_SR)
            active = self._endpointer.is_active(piece)
            events = self._endpointer.process(piece, frame_ms=dur_ms)

            start_events = [e for e in events if e["type"] == "start"]
            if start_events:
                await self._ensure_active_state(start_events[0]["start"])
            elif active and self._seg_start_ms is None:
                await self._ensure_active_state(self._total_ms)

            if self.state is not None:
                generation = self._generation
                try:
                    partial = await self._feed_piece(piece, active=active)
                except Exception as exc:
                    if generation != self._generation:
                        return
                    if not _is_decoder_prompt_overflow(exc):
                        self._clear_state(discard_segment=True)
                        raise
                    final = await self._finish_segment(
                        self._total_ms, overflow=exc)
                    if final is not None:
                        yield final
                    # 丢弃已污染的旧 state，把当前块在全新 state 中只重放一次。
                    if active and generation == self._generation:
                        await self._ensure_active_state(self._total_ms)
                        try:
                            partial = await self._feed_piece(
                                piece, active=True)
                        except Exception as retry_exc:
                            self._clear_state(discard_segment=True)
                            if not _is_decoder_prompt_overflow(retry_exc):
                                raise
                            logger.error(
                                f"[vllm-stream] 全新 SDK state 仍上下文超限，"
                                f"丢弃当前音频块 sid={self.sid[:8]}: {retry_exc}")
                        else:
                            if partial is not None:
                                yield partial
                else:
                    if partial is not None:
                        yield partial

            self._total_ms += dur_ms
            offset += piece.size

            if self.state is not None and any(e["type"] == "end" for e in events):
                final = await self._finish_segment(self._total_ms)
                if final is not None:
                    yield final
                continue

            cut_msg = await self._maybe_cut_after_piece(self._total_ms)
            if cut_msg is not None:
                yield cut_msg

    async def flush(self):
        """收到 stop：冲刷未闭合句出 final。"""
        if self.state is not None:
            final = await self._finish_segment(self._total_ms)
            if final is not None:
                yield final

    def close(self):
        """释放会话域状态，并使尚未返回的说话人任务失效。"""
        self._generation += 1
        self._clear_state(discard_segment=True)
        self._spk_cluster = None
        self._spk_identity_tracker.clear()


class VllmStreamBackend:
    """路线 A 活动后端：能量端点 + vLLM 原生流式。实现 StreamBackend 接口。

    与 VadOfflineBackend（stream_session.py）结构同构，差异仅 session 类型与 capabilities。
    """

    mode = "vllm"
    backend = "vllm-native"

    def __init__(self, engine, *, max_sessions=16, concurrency=1, max_utterance_sec=20,
                 max_state_sec=120, energy_floor_dbfs=-45.0, end_silence_ms=800,
                 priority_gate=None, speaker=None, speaker_service=None):
        self._engine = engine
        self._speaker = speaker
        self._speaker_service = speaker_service
        self._max_sessions = max_sessions
        self._max_utterance_sec = max_utterance_sec
        self._max_state_sec = max_state_sec
        self._energy_floor_dbfs = energy_floor_dbfs
        self._end_silence_ms = end_silence_ms
        self._priority_gate = priority_gate
        # generate 由引擎 _infer_lock 串行；此信号量限同时在飞解码的会话数（默认 1）
        self._sem = asyncio.Semaphore(concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max(2, concurrency + 1), thread_name_prefix="vllm-asr")
        self._speaker_executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="vllm-speaker")
            if speaker is not None else None
        )
        self._active = 0
        self._count_lock = threading.Lock()
        self.capabilities = {
            "partial_results": True,
            "word_timestamps": False,
            "languages_auto": True,
            "speaker_labels": speaker is not None,
            "speaker_identification": (
                speaker is not None and speaker_service is not None),
            "noise_filter_tunable": False,
            "speaker_tunable": speaker is not None,
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
            energy_floor_dbfs=self._energy_floor_dbfs, end_silence_ms=self._end_silence_ms)
        return VllmStreamSession(
            sid, self._engine, endpointer, self._executor, self._sem,
            max_utterance_sec=self._max_utterance_sec, max_state_sec=self._max_state_sec,
            priority_gate=self._priority_gate, speaker=self._speaker,
            speaker_service=self._speaker_service,
            speaker_executor=self._speaker_executor)

    def release(self, session):
        try:
            if session is not None:
                session.close()
        finally:
            with self._count_lock:
                self._active = max(0, self._active - 1)

    def shutdown(self):
        if self._speaker_executor is not None:
            self._speaker_executor.shutdown(wait=True, cancel_futures=True)
        self._executor.shutdown(wait=False, cancel_futures=True)
