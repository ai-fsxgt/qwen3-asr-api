"""app/engines/speaker_embedding_engine.py 单元测试（mock 模型 forward，不加载真权重）。

覆盖：make_windows 边界、共享推理锁、embed_windows 批切分与 L2 归一、
离线任务等待实时转写及取消。精度类断言不在此。
"""
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from app.engines.speaker_embedding_engine import (
    SpeakerEmbeddingEngine,
    make_windows,
)
from app.runtime.realtime_priority import RealtimePriorityGate


# ─── make_windows ───

def test_make_windows_regular_sliding():
    assert make_windows(0.0, 3.0) == [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0)]


def test_make_windows_exact_one_window():
    assert make_windows(0.0, 1.5) == [(0.0, 1.5)]


def test_make_windows_tail_window_length_in_bounds():
    ws = make_windows(0.0, 1.6)
    assert ws == [(0.0, 1.5), (0.75, 1.6)]
    for st, ed in ws:
        assert 0.75 < ed - st <= 1.5


def test_make_windows_short_segment_patch():
    # 上游 chunk() 对 ≤0.75s 段产生 0 窗；本实现补丁为整段 1 窗
    assert make_windows(0.0, 0.5) == [(0.0, 0.5)]


def test_make_windows_with_offset():
    assert make_windows(10.0, 13.0) == [(10.0, 11.5), (10.75, 12.25), (11.5, 13.0)]


def test_make_windows_zero_length_segment():
    assert make_windows(2.0, 2.0) == []


# ─── 推理锁 ───

def test_uses_injected_inference_lock():
    inference_lock = threading.Lock()
    eng = SpeakerEmbeddingEngine(inference_lock=inference_lock)
    assert eng._infer_lock is inference_lock


# ─── embed_windows ───

def _engine_with_mock(dim=192, *, priority_gate=None):
    eng = SpeakerEmbeddingEngine(priority_gate=priority_gate)
    model = MagicMock(side_effect=lambda feats: torch.ones(feats.shape[0], dim) * 3.0)
    eng._model = model
    return eng, model


def test_embed_windows_requires_loaded():
    eng = SpeakerEmbeddingEngine()
    with pytest.raises(RuntimeError):
        eng.embed_windows(np.zeros(16000, dtype=np.float32), [(0.0, 1.0)])


def test_embed_windows_empty_windows():
    eng, _ = _engine_with_mock()
    out = eng.embed_windows(np.zeros(16000, dtype=np.float32), [])
    assert out.shape == (0, 192)


def test_embed_windows_batching_and_l2_norm():
    eng, model = _engine_with_mock()
    wav = np.random.default_rng(0).normal(size=16000 * 2).astype(np.float32)
    windows = [(0.0, 1.5)] * 130  # ReDimNet2 小批推理：130 窗 → 33 批
    out = eng.embed_windows(wav, windows)
    assert out.shape == (130, 192)
    assert model.call_count == 33
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embed_identity_clips_consumes_one_model_batch_at_a_time():
    eng, model = _engine_with_mock()

    def clips():
        for index in range(9):
            if index == 4:
                assert model.call_count == 1
            yield np.full(16000 * 5, index, dtype=np.float32)

    out = eng.embed_identity_clips(clips())

    assert out.shape == (9, 192)
    assert model.call_count == 3


def test_embed_windows_short_clip_circle_padded():
    eng, model = _engine_with_mock()
    wav = np.random.default_rng(1).normal(size=16000).astype(np.float32)
    # 混合长短窗：补齐后批内帧数一致才能 stack 成功
    out = eng.embed_windows(wav, [(0.0, 1.0), (0.0, 0.3)])
    assert out.shape == (2, 192)
    feats = model.call_args[0][0]
    assert feats.shape[0] == 2



def test_priority_gate_preserves_embedding_values():
    def fake_model(waveforms):
        return waveforms[:, :1].repeat(1, 192)

    plain = SpeakerEmbeddingEngine()
    prioritized = SpeakerEmbeddingEngine(priority_gate=RealtimePriorityGate())
    plain._model = fake_model
    prioritized._model = fake_model
    wav = np.linspace(0.1, 1.0, 16000, dtype=np.float32)
    windows = [(0.0, 1.0)] * 130

    expected = plain.embed_windows(wav, windows)
    actual = prioritized.embed_windows(wav, windows)

    assert np.array_equal(actual, expected)



def test_offline_embedding_stops_when_cancelled_while_waiting(monkeypatch):
    gate = RealtimePriorityGate()
    eng, model = _engine_with_mock(priority_gate=gate)
    cancelled = threading.Event()
    waiting = threading.Event()
    errors = []
    wait_realtime_clear = gate.wait_realtime_clear

    def wait_and_record(*args, **kwargs):
        waiting.set()
        return wait_realtime_clear(*args, **kwargs)

    monkeypatch.setattr(gate, "wait_realtime_clear", wait_and_record)

    def run():
        try:
            eng.embed_windows(
                np.zeros(16000, dtype=np.float32),
                [(0.0, 1.0)],
                cancelled=cancelled.is_set,
            )
        except Exception as exc:
            errors.append(exc)

    with gate.realtime_section():
        worker = threading.Thread(target=run)
        worker.start()
        assert waiting.wait(timeout=1)
        assert worker.is_alive()

        cancelled.set()
        worker.join(timeout=1)
        assert not worker.is_alive()

    assert len(errors) == 1
    assert isinstance(errors[0], InterruptedError)
    model.assert_not_called()


def test_model_tag_constant():
    # V 系列衔接面：模板兼容性标识必须存在且非空
    assert (
        SpeakerEmbeddingEngine.MODEL_TAG
        == "redimnet2_b6_vb2_vox2_cnc2_v0_lm@v1"
    )
