"""说话人聚类（在线增量 + 离线全局）。

在线：质心 + 余弦阈值增量归簇（实时转写会话级，无回溯不改历史标签）。
离线：AHC（<40 窗）/ 谱聚类（≥40 窗）+ 小簇复核 + 近簇合并。
     聚类策略改写自 3D-Speaker speakerlab/process/cluster.py（Apache 2.0）：
     AHC 用 scipy linkage 替代 fastcluster；谱聚类保留未归一化拉普拉斯 + eigen-gap 定簇数。

聚类阈值沿用现有默认值，切换 ReDimNet2 后需结合真实数据重新标定。

所有输入 embedding 均假定 L2 归一化（SpeakerEmbeddingEngine 出口保证），余弦 = 点积。
scipy / scikit-learn 仅在离线聚类函数内延迟导入：依赖缺失只影响说话人分离，不拖垮主链路。
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ─── 离线聚类参数 ───

AHC_SIM_THR = 0.40        # AHC 合并阈：平均余弦相似度 ≥ 此值并簇
CLUSTER_LINE = 40         # 窗数 < 此值走 AHC，否则谱聚类
SPECTRAL_PVAL = 0.012     # 谱聚类 p-pruning 保留比例
SPECTRAL_MIN_PNUM = 6     # p-pruning 每行至少保留的邻居数
MIN_CLUSTER_SIZE = 4      # 簇大小 ≤ 此值进入小簇相似度复核
MINOR_MERGE_COS = 0.80    # 小簇仅在与大簇高度相似时合并，避免吞掉短时真实说话人
MER_COS = 0.8             # 簇质心相似度 > 此值时合并
MAX_NUM_SPKS = 15         # 谱聚类簇数搜索上界的硬上限


def speaker_label(idx: int) -> str:
    """0..25 → A..Z；≥26 → Z1/Z2…（防御性，正常不会到）。"""
    if idx < 26:
        return chr(ord("A") + idx)
    return f"Z{idx - 25}"


def _label_index(label: str) -> int:
    """speaker_label 的逆映射；非法标签返回 -1。"""
    if len(label) == 1 and "A" <= label <= "Z":
        return ord(label) - ord("A")
    if len(label) > 1 and label[0] == "Z" and label[1:].isdigit():
        return 25 + int(label[1:])
    return -1


class OnlineSpeakerClusterer:
    """会话级在线增量聚类：质心 + 余弦阈值。无回溯（不改历史标签）。"""

    def __init__(self, threshold: float = 0.5, max_speakers: int = 8,
                 min_seg_ms: int = 1500):
        self._thr = threshold
        self._max = max_speakers
        self._min_seg_ms = min_seg_ms
        self._centroids: list[np.ndarray] = []   # 每说话人运行质心（L2 归一）
        self._durations_ms: list[int] = []
        self._identity_audio: list[np.ndarray] = []

    def _add_identity_evidence(
        self,
        index: int,
        audio: np.ndarray | None,
    ) -> None:
        if audio is None:
            return
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        self._identity_audio[index] = np.concatenate(
            (self._identity_audio[index], chunk)
        )

    def assign(
        self,
        emb: np.ndarray,
        dur_ms: int,
        *,
        identity_audio: np.ndarray | None = None,
    ) -> str | None:
        """归簇一个段级 embedding（L2 归一 [192]），返回 A/B/C… 或 None（无法判定）。"""
        dur_ms = max(int(dur_ms), 0)
        sims = [float(c @ emb) for c in self._centroids]
        best = max(sims, default=-1.0)
        idx = sims.index(best) if sims else -1

        if dur_ms < self._min_seg_ms:
            # 短段 embedding 不可靠（spike EXP7）：只挂靠，不建簇不更新质心
            if best >= self._thr:
                self._add_identity_evidence(idx, identity_audio)
                return speaker_label(idx)
            return None
        if best >= self._thr:
            previous_ms = self._durations_ms[idx]
            merged = self._centroids[idx] * previous_ms + emb * dur_ms
            self._centroids[idx] = merged / np.linalg.norm(merged)
            self._durations_ms[idx] += dur_ms
            self._add_identity_evidence(idx, identity_audio)
            return speaker_label(idx)
        if len(self._centroids) >= self._max:
            # 低置信度且已达上限：保持匿名，避免强塞最近簇污染身份结果。
            return None
        self._centroids.append(np.asarray(emb, dtype=np.float32).copy())
        self._durations_ms.append(dur_ms)
        self._identity_audio.append(np.zeros(0, dtype=np.float32))
        self._add_identity_evidence(
            len(self._centroids) - 1,
            identity_audio,
        )
        return speaker_label(len(self._centroids) - 1)

    def closest_label(self, emb: np.ndarray) -> str | None:
        """返回达到阈值的最近簇，不改变簇状态。"""
        similarities = [float(centroid @ emb) for centroid in self._centroids]
        if not similarities:
            return None
        index = int(np.argmax(similarities))
        if similarities[index] < self._thr:
            return None
        return speaker_label(index)

    def create_cluster(self, emb: np.ndarray) -> str | None:
        """用指定质心创建新簇，观测证据由调用方随后累计。"""
        if len(self._centroids) >= self._max:
            return None
        centroid = np.asarray(emb, dtype=np.float32).copy()
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid /= norm
        self._centroids.append(centroid)
        self._durations_ms.append(0)
        self._identity_audio.append(np.zeros(0, dtype=np.float32))
        return speaker_label(len(self._centroids) - 1)

    def assign_to(
        self,
        label: str,
        emb: np.ndarray,
        dur_ms: int,
        *,
        identity_audio: np.ndarray | None = None,
    ) -> bool:
        """把观测证据更新到指定簇。"""
        index = _label_index(label)
        if not 0 <= index < len(self._centroids):
            return False
        duration_ms = max(int(dur_ms), 0)
        previous_ms = self._durations_ms[index]
        merged = self._centroids[index] * previous_ms + emb * duration_ms
        norm = float(np.linalg.norm(merged))
        if norm > 0:
            self._centroids[index] = merged / norm
        self._durations_ms[index] += duration_ms
        self._add_identity_evidence(index, identity_audio)
        return True

    @property
    def centroids(self) -> list[np.ndarray]:
        return list(self._centroids)

    def centroid_of(self, label: str) -> np.ndarray | None:
        """按 A/B/C… 标签取质心（声纹库 V 系列衔接面）。"""
        idx = _label_index(label)
        if 0 <= idx < len(self._centroids):
            return self._centroids[idx]
        return None

    def duration_sec_of(self, label: str) -> float:
        """返回参与质心计算的累计语音时长，供实时识别质量门控。"""
        idx = _label_index(label)
        if 0 <= idx < len(self._durations_ms):
            return self._durations_ms[idx] / 1000.0
        return 0.0

    def take_identity_audio_windows(
        self,
        label: str,
        min_duration_sec: float,
        sample_rate: int = 16000,
        max_windows: int = 1,
    ) -> list[np.ndarray]:
        """取最近的身份语音滑窗，并保留尾部重叠供下一批复用。"""
        idx = _label_index(label)
        if not 0 <= idx < len(self._identity_audio):
            return []
        required_samples = int(min_duration_sec * sample_rate)
        audio = self._identity_audio[idx]
        if audio.size < required_samples:
            return []
        step_samples = int(min_duration_sec * sample_rate / 2)
        available = 1 + (audio.size - required_samples) // step_samples
        count = min(available, max(int(max_windows), 1))
        latest_start = audio.size - required_samples
        first_start = latest_start - (count - 1) * step_samples
        clips = [
            np.ascontiguousarray(
                audio[
                    first_start + index * step_samples:
                    first_start + index * step_samples + required_samples
                ]
            )
            for index in range(count)
        ]
        retained_samples = required_samples - step_samples
        self._identity_audio[idx] = (
            audio[-retained_samples:].copy()
            if retained_samples
            else np.zeros(0, dtype=np.float32)
        )
        return clips


# ─── 离线全局聚类 ───

def cluster_offline(embeddings: np.ndarray, max_speakers: int = 8) -> np.ndarray:
    """全局聚类入口。输入 L2 归一 [N,192]，返回等长整型标签数组。

    标签已按簇首次出现顺序重排（0 = 最先开口的人），调用方映射 A/B/C… 即稳定可测。
    """
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    if n < CLUSTER_LINE:
        labels = _ahc(embeddings)
    else:
        labels = _spectral(embeddings, max_spks=min(max_speakers, MAX_NUM_SPKS))

    labels = _filter_minor_clusters(labels, embeddings)
    labels = _merge_close_clusters(labels, embeddings)
    return _reorder_by_first_appearance(labels)


def _ahc(embs: np.ndarray) -> np.ndarray:
    """AHC average linkage：相似度 ≥ AHC_SIM_THR 并簇（= 余弦距离切点 1-AHC_SIM_THR）。"""
    from scipy.cluster.hierarchy import linkage, fcluster
    z = linkage(embs, method="average", metric="cosine")
    return fcluster(z, t=1.0 - AHC_SIM_THR, criterion="distance") - 1


def _spectral(embs: np.ndarray, max_spks: int,
              pval: float = SPECTRAL_PVAL, min_pnum: int = SPECTRAL_MIN_PNUM) -> np.ndarray:
    """谱聚类：余弦亲和阵 + p-pruning + 未归一化拉普拉斯 eigen-gap 定簇数 + k-means 收尾。"""
    import scipy.sparse.linalg as sla
    from sklearn.cluster import KMeans

    sim = embs @ embs.T
    n = sim.shape[0]
    # p-pruning：每行只保留最大的 max(n*pval, min_pnum) 个相似度，其余置 0
    # （argpartition 向量化：每行最小的 n_prune 个位置，等价逐行 argsort 前缀）
    n_prune = min(int((1 - pval) * n), n - min_pnum)
    pruned = sim.copy()
    if n_prune > 0:
        idx = np.argpartition(pruned, n_prune - 1, axis=1)[:, :n_prune]
        np.put_along_axis(pruned, idx, 0, axis=1)
    sym = 0.5 * (pruned + pruned.T)
    np.fill_diagonal(sym, 0)
    laplacian = -sym
    laplacian[np.diag_indices(n)] = np.abs(sym).sum(axis=1)

    k = min(max_spks + 1, n)
    try:
        lambdas, vecs = sla.eigsh(laplacian, k=k, which="SM")
    except sla.ArpackError as e:
        # 近 rank-1 拉普拉斯（如单人长音频的近同 embedding）ARPACK 可能不收敛：
        # 降级单簇，不让单点异常吞掉整个文件的说话人标签
        logger.warning(f"谱聚类特征分解未收敛，降级单说话人: {e}")
        return np.zeros(n, dtype=int)
    gaps = np.diff(lambdas[:max_spks + 1])
    num_spks = int(np.argmax(gaps)) + 1 if len(gaps) else 1

    # k-means 固定随机种子，保证同输入同输出（测试与排查友好）
    return KMeans(n_clusters=num_spks, n_init=10, random_state=42).fit_predict(vecs[:, :num_spks])


def _filter_minor_clusters(labels: np.ndarray, embs: np.ndarray,
                           min_cluster_size: int = MIN_CLUSTER_SIZE,
                           merge_cos: float = MINOR_MERGE_COS) -> np.ndarray:
    """仅把与大簇高度相似的小簇整体并入，保留短时真实说话人。

    偏离上游：全员皆小簇（短音频，如 <8s 双人对话）时上游坍缩为单簇
    （np.zeros_like），此处改为保持原标签——过滤的本意是抑噪，无主簇时不应抹平真簇。
    """
    labels = labels.copy()
    uniq, counts = np.unique(labels, return_counts=True)
    major = uniq[counts > min_cluster_size]
    if len(major) == len(uniq) or len(major) == 0:
        return labels
    centers = np.stack([embs[labels == cluster].mean(axis=0) for cluster in major])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    for minor in uniq[counts <= min_cluster_size]:
        center = embs[labels == minor].mean(axis=0)
        center /= np.linalg.norm(center)
        similarities = centers @ center
        nearest = int(np.argmax(similarities))
        if similarities[nearest] >= merge_cos:
            labels[labels == minor] = major[nearest]
    return labels


def _merge_close_clusters(labels: np.ndarray, embs: np.ndarray,
                          cos_thr: float = MER_COS) -> np.ndarray:
    """迭代合并质心相似度 > cos_thr 的最近簇对（对齐 CommonClustering.merge_by_cos）。"""
    labels = labels.copy()
    while True:
        uniq = np.unique(labels)
        if len(uniq) == 1:
            break
        centers = np.stack([embs[labels == c].mean(axis=0) for c in uniq])
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        affinity = np.triu(centers @ centers.T, 1)
        i, j = np.unravel_index(int(np.argmax(affinity)), affinity.shape)
        if affinity[i, j] < cos_thr:
            break
        labels[labels == uniq[j]] = uniq[i]
    return labels


def _reorder_by_first_appearance(labels: np.ndarray) -> np.ndarray:
    """按簇首次出现顺序重映射为 0,1,2…（A 永远是先开口的人）。"""
    mapping: dict[int, int] = {}
    out = np.empty_like(labels)
    for i, lab in enumerate(labels):
        if lab not in mapping:
            mapping[lab] = len(mapping)
        out[i] = mapping[lab]
    return out


class DiarizationResult:
    """离线分离结果：窗标签 → ASR 段重叠加权投票；clusters 为声纹库 V 系列衔接面。

    输入约定：windows 按时间顺序、labels 已首现重排（cluster_offline 出口保证）。
    """

    def __init__(self, windows: list[tuple[float, float]],
                 labels: np.ndarray, embeddings: np.ndarray,
                 *, waveform: np.ndarray | None = None,
                 sample_rate: int = 16000):
        self._windows = list(windows)
        self._labels = np.asarray(labels, dtype=int)
        self._embs = np.asarray(embeddings)
        self._waveform = (
            None
            if waveform is None
            else np.asarray(waveform, dtype=np.float32).reshape(-1)
        )
        self._sample_rate = int(sample_rate)

    def label_for(self, start: float, end: float) -> str | None:
        """段 [start,end]（秒）与各窗的重叠时长加权投票；无重叠窗 → None。"""
        votes: dict[int, float] = {}
        for (wst, wed), lab in zip(self._windows, self._labels):
            overlap = min(end, wed) - max(start, wst)
            if overlap > 0:
                votes[int(lab)] = votes.get(int(lab), 0.0) + overlap
        if not votes:
            return None
        return speaker_label(max(votes, key=votes.get))  # 平票取先开口者（dict 插入序）

    @property
    def labels_in_order(self) -> list[str]:
        """全部说话人标签，按首次开口顺序（如 ["A","B"]）。"""
        if len(self._labels) == 0:
            return []
        return [speaker_label(i) for i in range(int(self._labels.max()) + 1)]

    @property
    def clusters(self) -> list[dict]:
        """返回每簇质心、语音总时长和身份识别音频索引。

        dur_sec 为簇内窗时间区间的并集长度（滑窗重叠不重复计）；
        identity_duration_sec 是实际分配的身份识别语音时长。
        """
        out = []
        n_clusters = int(self._labels.max()) + 1 if len(self._labels) else 0
        if n_clusters == 0:
            return out
        identity_intervals = (
            self._identity_intervals_by_cluster(n_clusters)
            if self._waveform is not None
            else None
        )
        for i in range(n_clusters):
            idx = np.where(self._labels == i)[0]
            centroid = self._embs[idx].mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid = centroid / norm
            cluster = {
                "label": speaker_label(i),
                "centroid": centroid,
                "dur_sec": _union_duration([self._windows[j] for j in idx]),
            }
            if identity_intervals is not None:
                intervals = identity_intervals[i]
                cluster["identity_waveform"] = self._waveform
                cluster["identity_sample_rate"] = self._sample_rate
                cluster["identity_intervals"] = intervals
                identity_samples = sum(
                    max(
                        round((end - start) * self._sample_rate),
                        0,
                    )
                    for start, end in intervals
                )
                cluster["identity_duration_sec"] = (
                    identity_samples / self._sample_rate
                )
            out.append(cluster)
        return out

    def _identity_intervals_by_cluster(
        self,
        n_clusters: int,
    ) -> list[list[tuple[float, float]]]:
        """把重叠窗口的新增覆盖区间唯一分配给对应簇。"""
        if self._waveform is None:
            raise RuntimeError("离线身份识别缺少原始音频")
        intervals: list[list[tuple[float, float]]] = [
            [] for _ in range(n_clusters)
        ]
        covered_until = float("-inf")
        for (start, end), label in zip(
            self._windows,
            self._labels,
            strict=True,
        ):
            new_duration = end - max(start, covered_until)
            slice_start = max(start, covered_until)
            covered_until = max(covered_until, end)
            if new_duration <= 0:
                continue
            start_sample = max(int(slice_start * self._sample_rate), 0)
            end_sample = min(
                int(end * self._sample_rate),
                self._waveform.size,
            )
            if end_sample > start_sample:
                label_index = int(label)
                intervals[label_index].append((
                    start_sample / self._sample_rate,
                    end_sample / self._sample_rate,
                ))
        return intervals


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    """Return the union duration of sorted or unsorted time intervals."""
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total
