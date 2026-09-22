"""声纹库编排层：登记/认领/识别流程。

管理路径统一执行格式转换 → VAD → 质量门槛 → embedding → 入库/比对。
全部方法【同步】，路由层经 asyncio.to_thread 下沉调用（对齐 routes.py 体例）。
错误约定：
- 质量门槛/输入问题 → ValueError（中文信息，路由转 400 直接透传 detail）；
- 库故障 → SpeakerStoreError 原样上抛（路由转 500）；
- 转写联动入口（map_clusters / map_and_enroll_clusters）【永不抛错】——
  识别/登记失败一律优雅退回匿名（speaker_id/name=None），不影响转写结果。
"""
import logging
import os
import shutil
import threading
import uuid

import numpy as np
import soundfile as sf

import app.config as cfg
from app.engines.speaker_embedding_engine import (
    make_identity_windows,
    make_windows,
)
from app.pipeline.audio_preprocessor import convert_to_wav
from app.runtime.speaker_cluster import cluster_offline
from app.runtime.speaker_store import (
    SpeakerIdentification,
    SpeakerStore,
    SpeakerStoreError,
)

logger = logging.getLogger(__name__)

QUALITY_HINT_MIN_TEMPLATES = 3   # 建议模板数（不足仅提示，不阻断）


class SpeakerService:

    def __init__(self, store: SpeakerStore, embed_engine, vad_engine):
        self.store = store
        self._engine = embed_engine
        self._vad = vad_engine
        # 数据库变更与留存音频操作必须保持同一顺序，避免并发删除后重新落盘。
        self._speaker_data_lock = threading.Lock()

    # ─── 内部：单文件 → 模板向量 ───

    def _embed_file(
        self,
        file_path: str,
        *,
        keep_wav: bool = False,
        purpose: str = "登记",
        min_sec: float | None = None,
    ) -> tuple[np.ndarray, float, str | None]:
        """转换 → VAD → 时长门槛 → 单人校验 → 窗均值模板。

        返回 (L2 归一模板向量, 有效语音秒数, 留存的 wav 路径或 None)。
        keep_wav=False 或失败路径上一律清理临时 wav。
        """
        os.makedirs(cfg.UPLOADS_DIR, exist_ok=True)
        wav_path = os.path.join(cfg.UPLOADS_DIR, f"spk_{uuid.uuid4().hex}.wav")
        ok = False
        try:
            convert_to_wav(file_path, wav_path)
            segments = self._vad.detect(wav_path)
            dur = sum(e - s for s, e in segments) / 1000.0
            required_sec = (
                cfg.SPEAKER_ENROLL_MIN_SEC if min_sec is None else min_sec
            )
            if dur < required_sec:
                raise ValueError(
                    f"{purpose}样本有效语音不足（{dur:.1f}s < {required_sec}s），"
                    "请提供更长的清晰人声"
                )
            wav, sample_rate = sf.read(wav_path, dtype="float32")
            windows = [w for s, e in segments
                       for w in make_windows(s / 1000.0, e / 1000.0)]
            embs = self._engine.embed_windows(wav, windows)
            labels = cluster_offline(embs, max_speakers=cfg.SPEAKER_MAX)
            if len(np.unique(labels)) > 1:
                raise ValueError(f"{purpose}样本含多个说话人，请提供单人清晰录音")
            speech_chunks = [
                wav[
                    max(int(start_ms * sample_rate / 1000), 0):
                    min(int(end_ms * sample_rate / 1000), len(wav))
                ]
                for start_ms, end_ms in segments
            ]
            speech = np.ascontiguousarray(np.concatenate(speech_chunks))
            identity_windows = make_identity_windows(
                0.0,
                len(speech) / sample_rate,
            )
            if not identity_windows:
                identity_windows = [(0.0, len(speech) / sample_rate)]
            identity_embeddings = self._engine.embed_windows(
                speech,
                identity_windows,
            )
            template = identity_embeddings.mean(axis=0)
            norm = float(np.linalg.norm(template))
            if norm > 0:
                template = template / norm
            ok = True
            return template.astype(np.float32), dur, (wav_path if keep_wav else None)
        finally:
            if (not ok or not keep_wav) and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError as e:
                    logger.warning(f"声纹临时文件清理失败: {e}")

    def _audio_dir(self, speaker_id: str) -> str:
        return os.path.join(cfg.BASE_DIR, "data", "speaker_audio", speaker_id)

    @staticmethod
    def _cleanup_wavs(paths: list[str]) -> None:
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"声纹临时文件清理失败: {e}")

    @staticmethod
    def _quality_hint(template_count: int) -> str | None:
        if template_count >= QUALITY_HINT_MIN_TEMPLATES:
            return None
        return (
            f"建议提供 ≥{QUALITY_HINT_MIN_TEMPLATES} 个不同场景的样本以提升识别稳健性"
        )

    @classmethod
    def _with_quality_hint(cls, result: dict) -> dict:
        response = dict(result)
        if "quality_hint" not in response:
            quality_hint = cls._quality_hint(response["templates"])
            if quality_hint is not None:
                response["quality_hint"] = quality_hint
        return response

    def _replace_retained_audio(
        self,
        speaker_id: str,
        kept_wavs: list[str],
        template_ids: list[int],
    ) -> None:
        """模板已原子替换后同步留存音频；失败仅记录，不反转已提交的认领。"""
        audio_dir = self._audio_dir(speaker_id)
        try:
            if os.path.isdir(audio_dir):
                shutil.rmtree(audio_dir)
            if kept_wavs:
                os.makedirs(audio_dir, exist_ok=True)
                for template_id, wav_path in zip(
                    template_ids, kept_wavs, strict=True
                ):
                    shutil.move(
                        wav_path,
                        os.path.join(audio_dir, f"{template_id}.wav"),
                    )
        except (OSError, ValueError) as e:
            logger.error(f"说话人已认领，但留存音频替换失败: {e}")
        finally:
            self._cleanup_wavs(kept_wavs)

    # ─── 管理路径（路由经 to_thread 调用）───

    def enroll(self, name: str, note: str | None, file_paths: list[str],
               consent: bool) -> dict:
        """登记新说话人。全部样本过质量门槛后单事务入库。"""
        if consent is not True:
            raise ValueError("登记必须携带 consent=true（确认已获得数据主体同意）")
        if not file_paths:
            raise ValueError("至少需要 1 个音频样本")
        if len(file_paths) > SpeakerStore.MAX_TEMPLATES:
            raise ValueError(f"样本数超过模板上限 {SpeakerStore.MAX_TEMPLATES}")

        keep = cfg.SPEAKER_STORE_AUDIO
        vectors, durs, kept_wavs = [], [], []
        try:
            for p in file_paths:
                vec, dur, kept = self._embed_file(p, keep_wav=keep)
                vectors.append(vec)
                durs.append(dur)
                if kept:
                    kept_wavs.append(kept)
            with self._speaker_data_lock:
                speaker_id = self.store.enroll_speaker(
                    name, note, vectors, durs, consent=True, source="manual")
                if kept_wavs:
                    audio_dir = self._audio_dir(speaker_id)
                    os.makedirs(audio_dir, exist_ok=True)
                    for i, wav_path in enumerate(kept_wavs):
                        shutil.move(
                            wav_path,
                            os.path.join(audio_dir, f"{i:02d}.wav"),
                        )
        except Exception:
            self._cleanup_wavs(kept_wavs)
            raise
        resp = {"speaker_id": speaker_id, "name": name, "templates": len(vectors)}
        return self._with_quality_hint(resp)

    def claim(
        self,
        speaker_id: str,
        name: str,
        note: str | None,
        file_paths: list[str],
        consent: bool,
        claim_key: str,
    ) -> dict:
        """认领自动登记说话人，保留 speaker_id 并用管理员样本替换模板。"""
        if consent is not True:
            raise ValueError("认领必须携带 consent=true（确认已获得数据主体同意）")
        name = name.strip()
        if not name:
            raise ValueError("说话人名称不能为空")
        claim_key = claim_key.strip()
        if not claim_key:
            raise ValueError("claim_key 不能为空")
        if len(claim_key) > SpeakerStore.MAX_CLAIM_KEY_LENGTH:
            raise ValueError(
                f"claim_key 长度不能超过 "
                f"{SpeakerStore.MAX_CLAIM_KEY_LENGTH} 个字符"
            )
        if not file_paths:
            raise ValueError("至少需要 1 个音频样本")
        if len(file_paths) > SpeakerStore.MAX_TEMPLATES:
            raise ValueError(f"样本数超过模板上限 {SpeakerStore.MAX_TEMPLATES}")

        replay = self.store.resolve_claim(speaker_id, claim_key)
        if replay is not None:
            return self._with_quality_hint(replay)

        keep = cfg.SPEAKER_STORE_AUDIO
        vectors, durs, kept_wavs = [], [], []
        try:
            for file_path in file_paths:
                vector, dur, kept = self._embed_file(
                    file_path,
                    keep_wav=keep,
                    purpose="认领",
                )
                vectors.append(vector)
                durs.append(dur)
                if kept:
                    kept_wavs.append(kept)
            with self._speaker_data_lock:
                result, created = self.store.claim_auto_speaker(
                    speaker_id,
                    name,
                    note,
                    vectors,
                    durs,
                    claim_key,
                    self._quality_hint(len(vectors)),
                )
                if created:
                    self._replace_retained_audio(
                        speaker_id,
                        kept_wavs,
                        result["template_ids"],
                    )
        except Exception:
            self._cleanup_wavs(kept_wavs)
            raise

        if not created:
            self._cleanup_wavs(kept_wavs)
        return self._with_quality_hint(result)

    def add_template(self, speaker_id: str, file_path: str) -> dict:
        """为既有说话人追加模板（质心自动重算）。"""
        vec, dur, kept = self._embed_file(file_path, keep_wav=cfg.SPEAKER_STORE_AUDIO,
                                          purpose="追加")
        try:
            with self._speaker_data_lock:
                self.store.add_template(speaker_id, vec, dur)
                if kept:
                    audio_dir = self._audio_dir(speaker_id)
                    os.makedirs(audio_dir, exist_ok=True)
                    shutil.move(
                        kept,
                        os.path.join(
                            audio_dir,
                            f"t_{uuid.uuid4().hex[:8]}.wav",
                        ),
                    )
        except Exception:
            if kept and os.path.exists(kept):
                os.remove(kept)
            raise
        info = self.store.get_speaker(speaker_id)
        return {"speaker_id": speaker_id, "templates": len(info["templates"]) if info else 0}

    def delete_speaker(self, speaker_id: str) -> None:
        """删除数据库记录并清理留存音频目录。"""
        with self._speaker_data_lock:
            self.store.delete_speaker(speaker_id)
            audio_dir = self._audio_dir(speaker_id)
            if os.path.isdir(audio_dir):
                shutil.rmtree(audio_dir, ignore_errors=True)

    def identify_file(self, file_path: str) -> dict:
        """单文件 1:N 识别。"""
        vec, dur, _ = self._embed_file(
            file_path,
            keep_wav=False,
            purpose="识别",
            min_sec=cfg.SPEAKER_IDENTIFY_MIN_SEC,
        )
        decision = self.store.identify_detailed(
            vec,
            threshold=cfg.SPEAKER_ID_THRESHOLD,
            margin=cfg.SPEAKER_ID_MARGIN,
        )
        self._audit_identification(decision, via="file", duration_sec=dur)
        hit = decision.as_match(include_source=True)
        if hit is None:
            return {
                "matched": False,
                "reason": decision.reason,
                "score": decision.top1_score,
                "top2_score": decision.top2_score,
                "margin": decision.margin,
            }
        return {
            "matched": True,
            **hit,
            "reason": decision.reason,
            "top2_score": decision.top2_score,
            "margin": decision.margin,
        }

    # ─── 转写联动路径（永不抛错，失败退回匿名）───

    @staticmethod
    def _anon(label: str) -> dict:
        return {"label": label, "speaker_id": None, "name": None, "score": None}

    def _audit_identification(
        self,
        decision: SpeakerIdentification,
        *,
        via: str,
        duration_sec: float,
        label: str | None = None,
        threshold: float | None = None,
        required_margin: float | None = None,
        evidence_detail: dict | None = None,
    ) -> None:
        detail = {
            "matched": decision.matched,
            "reason": decision.reason,
            "top1_score": decision.top1_score,
            "top2_score": decision.top2_score,
            "margin": decision.margin,
            "threshold": (
                cfg.SPEAKER_ID_THRESHOLD if threshold is None else threshold
            ),
            "required_margin": (
                cfg.SPEAKER_ID_MARGIN
                if required_margin is None
                else required_margin
            ),
            "duration_sec": duration_sec,
            "label": label,
            "via": via,
            "match_basis": "template",
        }
        if evidence_detail is not None:
            detail.update(evidence_detail)
        self.store.audit(
            "identify",
            decision.speaker_id if decision.matched else None,
            detail,
        )

    @staticmethod
    def _aggregate_identity_decisions(
        decisions: list[SpeakerIdentification],
        durations: list[float],
    ) -> tuple[SpeakerIdentification, dict]:
        matched = [
            (decision, duration)
            for decision, duration in zip(decisions, durations, strict=True)
            if decision.matched
        ]
        matched_duration = sum(duration for _decision, duration in matched)
        total_duration = sum(durations)
        detail = {
            "evidence_count": len(decisions),
            "matched_evidence_count": len(matched),
            "matched_duration_sec": matched_duration,
            "evidence_duration_sec": total_duration,
            "support_ratio": (
                matched_duration / total_duration if total_duration > 0 else 0.0
            ),
        }
        for attribute, prefix in (
            ("top1_score", "evidence_top1"),
            ("margin", "evidence_margin"),
        ):
            values = []
            for decision in decisions:
                value = getattr(decision, attribute)
                if value is not None:
                    values.append(float(value))
            if values:
                detail.update({
                    f"{prefix}_min": min(values),
                    f"{prefix}_median": float(np.median(values)),
                    f"{prefix}_max": max(values),
                })
        reason_counts = {}
        for decision in decisions:
            reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
        if reason_counts:
            detail["evidence_reasons"] = reason_counts
        if not decisions:
            return SpeakerIdentification(False, "insufficient_duration"), detail
        if not matched:
            strongest = max(
                decisions,
                key=lambda decision: (
                    decision.top1_score
                    if decision.top1_score is not None else float("-inf")
                ),
            )
            return strongest, detail

        speaker_ids = {decision.speaker_id for decision, _duration in matched}
        representative = max(
            (decision for decision, _duration in matched),
            key=lambda decision: (
                decision.top1_score
                if decision.top1_score is not None else float("-inf")
            ),
        )
        if len(speaker_ids) > 1:
            return SpeakerIdentification(
                False,
                "inconsistent_identity",
                top1_score=representative.top1_score,
                top2_score=representative.top2_score,
                margin=representative.margin,
            ), detail
        if matched_duration * 2 <= total_duration:
            return SpeakerIdentification(
                False,
                "insufficient_identity_support",
                top1_score=representative.top1_score,
                top2_score=representative.top2_score,
                margin=representative.margin,
            ), detail

        def weighted_average(attribute: str) -> float | None:
            values = [
                (getattr(decision, attribute), duration)
                for decision, duration in matched
                if getattr(decision, attribute) is not None
            ]
            if not values:
                return None
            weight = sum(duration for _value, duration in values)
            return sum(value * duration for value, duration in values) / weight

        return SpeakerIdentification(
            True,
            "matched",
            top1_score=weighted_average("top1_score"),
            top2_score=weighted_average("top2_score"),
            margin=weighted_average("margin"),
            speaker_id=representative.speaker_id,
            name=representative.name,
            source=representative.source,
        ), detail

    def _identify_offline_evidence(
        self,
        clusters: list[dict],
        *,
        threshold: float,
        margin: float,
        cancelled=None,
    ) -> dict[
        str,
        tuple[SpeakerIdentification, dict, np.ndarray | None, list[dict]],
    ]:
        prepared = []
        vectors = []
        for cluster in clusters:
            if cancelled and cancelled():
                raise InterruptedError("说话人处理已取消")
            waveform = np.asarray(
                cluster["identity_waveform"],
                dtype=np.float32,
            ).reshape(-1)
            sample_rate = int(cluster["identity_sample_rate"])
            source_intervals = cluster["identity_intervals"]
            duration_sec = float(cluster["identity_duration_sec"])
            groups = []
            identity_centroid = None
            if duration_sec >= cfg.SPEAKER_IDENTIFY_MIN_SEC:
                windows = make_identity_windows(
                    0.0,
                    duration_sec,
                    win=cfg.SPEAKER_IDENTIFY_MIN_SEC,
                    step=cfg.SPEAKER_IDENTIFY_MIN_SEC / 2,
                )
                window_intervals = [
                    self._source_intervals_for_window(
                        source_intervals,
                        start,
                        end,
                    )
                    for start, end in windows
                ]
                embeddings = self._engine.embed_identity_clips(
                    (
                        self._audio_from_source_intervals(
                            waveform,
                            intervals,
                            sample_rate,
                        )
                        for intervals in window_intervals
                    ),
                    cancelled=cancelled,
                )
                groups = [
                    (
                        embedding,
                        end - start,
                        intervals,
                    )
                    for embedding, (start, end), intervals in zip(
                        embeddings,
                        windows,
                        window_intervals,
                        strict=True,
                    )
                ]
                identity_centroid = sum(
                    vector * duration
                    for vector, duration, _intervals in groups
                )
                norm = float(np.linalg.norm(identity_centroid))
                if norm > 0:
                    identity_centroid = identity_centroid / norm
            prepared.append((
                cluster.get("label", "?"),
                groups,
                identity_centroid,
            ))
            vectors.extend(
                vector for vector, _duration, _intervals in groups
            )

        decisions = self.store.identify_many_detailed(
            vectors,
            threshold=threshold,
            margin=margin,
        ) if vectors else []
        results = {}
        offset = 0
        for label, groups, identity_centroid in prepared:
            count = len(groups)
            group_decisions = decisions[offset:offset + count]
            durations = [
                duration for _vector, duration, _intervals in groups
            ]
            decision, detail = self._aggregate_identity_decisions(
                group_decisions,
                durations,
            )
            anchors = []
            for index, (
                group_decision,
                (_vector, _duration, intervals),
            ) in enumerate(zip(group_decisions, groups, strict=True)):
                if not group_decision.matched:
                    continue
                adjacent_decisions = group_decisions[
                    max(0, index - 1):index
                ] + group_decisions[index + 1:index + 2]
                if not any(
                    adjacent.matched
                    and adjacent.speaker_id == group_decision.speaker_id
                    for adjacent in adjacent_decisions
                ):
                    continue
                anchors.append({
                    "speaker_id": group_decision.speaker_id,
                    "name": group_decision.name,
                    "score": group_decision.top1_score,
                    "intervals": intervals,
                })
            results[label] = (
                decision,
                detail,
                identity_centroid,
                anchors,
            )
            offset += count
        return results

    @staticmethod
    def _audio_from_source_intervals(
        waveform: np.ndarray,
        intervals: list[tuple[float, float]],
        sample_rate: int,
    ) -> np.ndarray:
        """从原始波形按时间区间组装单个身份识别窗口。"""
        chunks = [
            waveform[
                max(round(start * sample_rate), 0):
                min(round(end * sample_rate), waveform.size)
            ]
            for start, end in intervals
        ]
        chunks = [chunk for chunk in chunks if chunk.size]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks)

    @staticmethod
    def _source_intervals_for_window(
        source_intervals: list[tuple[float, float]],
        window_start: float,
        window_end: float,
    ) -> list[tuple[float, float]]:
        """Map a window on concatenated speech back to original timeline."""
        mapped = []
        cursor = 0.0
        for source_start, source_end in source_intervals:
            duration = source_end - source_start
            overlap_start = max(window_start, cursor)
            overlap_end = min(window_end, cursor + duration)
            if overlap_end > overlap_start:
                mapped.append((
                    source_start + overlap_start - cursor,
                    source_start + overlap_end - cursor,
                ))
            cursor += duration
            if cursor >= window_end:
                break
        return mapped

    def _identify_cluster(
        self,
        cluster: dict,
        *,
        threshold: float,
        margin: float,
        via: str,
    ) -> dict | None:
        label = cluster.get("label", "?")
        duration_sec = float(cluster.get("dur_sec") or 0.0)
        if duration_sec < cfg.SPEAKER_IDENTIFY_MIN_SEC:
            decision = SpeakerIdentification(False, "insufficient_duration")
        else:
            decision = self.store.identify_detailed(
                cluster["centroid"],
                threshold=threshold,
                margin=margin,
            )
        self._audit_identification(
            decision,
            via=via,
            duration_sec=duration_sec,
            label=label,
            threshold=threshold,
            required_margin=margin,
            evidence_detail=(
                {
                    "evidence_count": 1,
                    "matched_evidence_count": int(decision.matched),
                    "matched_duration_sec": (
                        duration_sec if decision.matched else 0.0
                    ),
                    "evidence_duration_sec": duration_sec,
                    "support_ratio": 1.0 if decision.matched else 0.0,
                }
                if via == "stream" else None
            ),
        )
        return decision.as_match()

    def _identify_or_enroll_cluster(
        self,
        cluster: dict,
        *,
        threshold: float,
        margin: float,
        cancelled=None,
    ) -> dict | None:
        """先无锁识别，未命中时由存储层跨实例串行复查并登记。"""
        label = cluster.get("label", "?")
        duration_sec = float(cluster.get("dur_sec") or 0.0)
        decision = self.store.identify_detailed(
            cluster["centroid"],
            threshold=threshold,
            margin=margin,
        )
        if cancelled and cancelled():
            return None
        hit = decision.as_match()
        if hit is not None:
            self._audit_identification(
                decision,
                via="offline",
                duration_sec=duration_sec,
                label=label,
                threshold=threshold,
                required_margin=margin,
            )
            return {"label": label, **hit}

        auto_enroll_options = {
            "threshold": threshold,
            "margin": margin,
        }
        if cancelled is not None:
            auto_enroll_options["cancelled"] = cancelled
        enrollment = self.store.identify_or_enroll_auto(
            cluster["centroid"],
            duration_sec,
            **auto_enroll_options,
        )
        if enrollment is None:
            return None
        self._audit_identification(
            enrollment.identification,
            via="offline",
            duration_sec=duration_sec,
            label=label,
            threshold=threshold,
            required_margin=margin,
        )
        if not enrollment.created:
            hit = enrollment.identification.as_match()
            if hit is None:
                return self._anon(label)
            return {"label": label, **hit}
        return {
            "label": label,
            "speaker_id": enrollment.speaker_id,
            "name": enrollment.name,
            "score": None,
            "auto_enrolled": True,
        }

    def map_clusters(self, clusters: list[dict], *,
                     id_threshold: float | None = None,
                     id_margin: float | None = None) -> list[dict]:
        """实时联动入口（仅识别，不自动登记）。

        clusters = [{"label", "centroid", "dur_sec"}]；返回每簇
        {"label","speaker_id","name","score"}，
        未命中/任何异常 → 该簇 speaker_id/name=None（识别失败永远优雅退回匿名）。
        id_threshold/id_margin 缺省=服务端 cfg（支持按会话覆盖）。
        """
        thr = cfg.SPEAKER_ID_THRESHOLD if id_threshold is None else id_threshold
        mgn = cfg.SPEAKER_ID_MARGIN if id_margin is None else id_margin
        out = []
        for c in clusters:
            label = c.get("label", "?")
            try:
                hit = self._identify_cluster(
                    c,
                    threshold=thr,
                    margin=mgn,
                    via="stream",
                )
                out.append({"label": label, **hit} if hit else self._anon(label))
            except Exception as e:
                logger.warning(f"簇识别失败，退回匿名: {e}")
                out.append(self._anon(label))
        return out

    def map_and_enroll_clusters(self, clusters: list[dict], *,
                                id_threshold: float | None = None,
                                id_margin: float | None = None,
                                cancelled=None) -> list[dict]:
        """离线联动入口（识别 + 自动登记）。

        clusters = [{"label","centroid","dur_sec","identity_waveform",
                     "identity_intervals"}]。未命中且开启
        speaker_auto_enroll 且簇语音总时长过门槛 → 以「说话人_NN」占位名登记
        （source='auto'；开启自动登记 = 部署方声明已获数据主体同意，consent 同责）。
        已命中的说话人不自动追加模板（防投毒）。登记失败退回匿名，不影响转写。
        自动登记由 PostgreSQL 事务锁跨服务串行化，避免并发任务重复建档。
        id_threshold/id_margin 缺省=服务端 cfg（支持按请求覆盖）。
        cancelled 触发后停止处理剩余簇，返回已完成的映射。
        """
        thr = cfg.SPEAKER_ID_THRESHOLD if id_threshold is None else id_threshold
        mgn = cfg.SPEAKER_ID_MARGIN if id_margin is None else id_margin
        if cancelled and cancelled():
            return []
        try:
            evidence_results = self._identify_offline_evidence(
                clusters,
                threshold=thr,
                margin=mgn,
                cancelled=cancelled,
            )
        except InterruptedError:
            return []
        except Exception as e:
            logger.warning(f"离线声纹批量识别失败，退回匿名: {e}")
            return [self._anon(cluster.get("label", "?")) for cluster in clusters]
        if cancelled and cancelled():
            return []
        out = []
        for c in clusters:
            if cancelled and cancelled():
                break
            label = c.get("label", "?")
            try:
                with self._speaker_data_lock:
                    if cancelled and cancelled():
                        break
                    dur = float(c.get("identity_duration_sec") or 0.0)
                    (
                        decision,
                        detail,
                        identity_centroid,
                        identity_anchors,
                    ) = evidence_results[label]
                    self._audit_identification(
                        decision,
                        via="offline",
                        duration_sec=dur,
                        label=label,
                        threshold=thr,
                        required_margin=mgn,
                        evidence_detail=detail,
                    )
                    hit = decision.as_match()
                    if (
                        cfg.SPEAKER_AUTO_ENROLL
                        and identity_centroid is not None
                        and dur >= cfg.SPEAKER_IDENTIFY_MIN_SEC
                        and dur >= cfg.SPEAKER_AUTO_ENROLL_MIN_SEC
                    ):
                        if hit:
                            mapping = {"label": label, **hit}
                        elif (
                            decision.reason in {
                                "ambiguous_margin",
                                "inconsistent_identity",
                                "insufficient_identity_support",
                            }
                        ):
                            mapping = self._anon(label)
                        else:
                            enroll_cluster = {
                                **c,
                                "centroid": identity_centroid,
                                "dur_sec": dur,
                            }
                            mapping = self._identify_or_enroll_cluster(
                                enroll_cluster,
                                threshold=thr,
                                margin=mgn,
                                cancelled=cancelled,
                            )
                            if mapping is None:
                                break
                    else:
                        if cancelled and cancelled():
                            break
                        if hit:
                            mapping = {"label": label, **hit}
                        else:
                            mapping = self._anon(label)
                    mapping["_identity_anchors"] = identity_anchors
                out.append(mapping)
            except Exception as e:
                logger.warning(f"簇识别/自动登记失败，退回匿名: {e}")
                out.append(self._anon(label))
        return out
