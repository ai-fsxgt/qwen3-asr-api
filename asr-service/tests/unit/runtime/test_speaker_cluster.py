"""app/runtime/speaker_cluster.py 单元测试（合成 embedding，不触模型）。

覆盖离线 AHC/谱聚类分支、阈值语义、小簇并入、近簇合并和首现重排。
只测逻辑，不测真实音频精度。
"""
import numpy as np

from app.runtime.speaker_cluster import (
    CLUSTER_LINE,
    _merge_close_clusters,
    cluster_offline,
    speaker_label,
)

RNG = np.random.default_rng(42)
DIM = 192


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def around(center: np.ndarray, scale: float = 0.01) -> np.ndarray:
    v = center + RNG.normal(scale=scale, size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def mix(a: np.ndarray, b: np.ndarray, wa: float, wb: float) -> np.ndarray:
    v = wa * a + wb * b
    return v / np.linalg.norm(v)


# ─── speaker_label ───

def test_speaker_label_alphabet_and_overflow():
    assert speaker_label(0) == "A"
    assert speaker_label(25) == "Z"
    assert speaker_label(26) == "Z1"
    assert speaker_label(27) == "Z2"


# ─── cluster_offline：分支与边界 ───

def test_offline_empty_and_single():
    assert cluster_offline(np.zeros((0, DIM))).shape == (0,)
    assert cluster_offline(unit(0)[np.newaxis]).tolist() == [0]


def test_offline_ahc_three_clusters_first_appearance_order():
    embs = np.stack(
        [around(unit(0)) for _ in range(6)]
        + [around(unit(1)) for _ in range(6)]
        + [around(unit(2)) for _ in range(5)]
    )
    assert len(embs) < CLUSTER_LINE  # AHC 分支
    labels = cluster_offline(embs)
    assert labels.tolist() == [0] * 6 + [1] * 6 + [2] * 5  # 首现重排：A=先开口


def test_offline_interleaved_speakers():
    embs = np.stack([around(unit(i % 2)) for i in range(12)])
    labels = cluster_offline(embs)
    assert labels.tolist() == [i % 2 for i in range(12)]


def test_offline_ahc_threshold_semantics():
    # spike 勘误锁定：相似度 ≥0.40 并簇，<0.40 分簇（非"距离 0.4=相似度 0.6"）
    close = np.stack([around(unit(0)) for _ in range(6)]
                     + [around(mix(unit(0), unit(1), 1.0, 1.0)) for _ in range(6)])
    assert len(np.unique(cluster_offline(close))) == 1  # 簇间相似度 ≈0.71 → 并

    far = np.stack([around(unit(0)) for _ in range(6)]
                   + [around(mix(unit(0), unit(1), 0.2, 0.98)) for _ in range(6)])
    assert len(np.unique(cluster_offline(far))) == 2  # 簇间相似度 ≈0.2 → 分


def test_offline_dissimilar_minor_cluster_is_preserved():
    embs = np.stack(
        [around(unit(0)) for _ in range(10)]
        + [around(unit(1)) for _ in range(10)]
        + [around(unit(2)) for _ in range(3)]  # 小簇与两个大簇都不相似
    )
    labels = cluster_offline(embs)
    assert len(np.unique(labels)) == 3
    assert set(labels[20:]) == {2}  # 不强制并入无关说话人


def test_offline_spectral_branch_two_clusters():
    embs = np.stack(
        [around(unit(0)) for _ in range(30)] + [around(unit(1)) for _ in range(30)]
    )
    assert len(embs) >= CLUSTER_LINE  # 谱聚类分支
    labels = cluster_offline(embs)
    assert labels.tolist() == [0] * 30 + [1] * 30


def test_offline_spectral_single_speaker():
    embs = np.stack([around(unit(0)) for _ in range(45)])
    labels = cluster_offline(embs)
    assert labels.tolist() == [0] * 45


def test_offline_spectral_arpack_failure_degrades_single(monkeypatch):
    """ARPACK 不收敛（病态拉普拉斯）→ 降级单说话人，不向上抛异常吞掉整文件标签。"""
    import scipy.sparse.linalg as sla

    def boom(*args, **kwargs):
        raise sla.ArpackNoConvergence("ARPACK 未收敛", np.array([]), np.array([]))

    monkeypatch.setattr(sla, "eigsh", boom)
    embs = np.stack([around(unit(0)) for _ in range(45)])
    labels = cluster_offline(embs)
    assert labels.tolist() == [0] * 45


# ─── 后处理 ───

def test_merge_close_clusters_by_centroid_similarity():
    a = [around(unit(0)) for _ in range(5)]
    b = [around(mix(unit(0), unit(1), 0.97, 0.24)) for _ in range(5)]  # 质心相似度 ≈0.97
    labels = _merge_close_clusters(np.array([0] * 5 + [1] * 5), np.stack(a + b))
    assert len(np.unique(labels)) == 1

    c = [around(unit(1)) for _ in range(5)]  # 质心相似度 ≈0
    labels2 = _merge_close_clusters(np.array([0] * 5 + [1] * 5), np.stack(a + c))
    assert len(np.unique(labels2)) == 2
