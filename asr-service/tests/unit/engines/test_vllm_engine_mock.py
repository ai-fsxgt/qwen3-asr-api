"""VLLMASREngine 封装单元测试（mock Qwen3ASRModel，不依赖 vLLM / GPU）。

绕过 load()（不 import qwen_asr），直接注入 mock _model 验证三段式调用序、
chunk_size_sec 钳制与会话级覆盖。standard venv 即可运行。
"""
from types import SimpleNamespace

import numpy as np

from app.engines.vllm_asr_engine import (
    VLLMASREngine, clamp_chunk_size_sec, CHUNK_SIZE_SEC_MIN, CHUNK_SIZE_SEC_MAX,
)


def test_clamp_chunk_size_sec():
    assert clamp_chunk_size_sec(0.1) == CHUNK_SIZE_SEC_MIN
    assert clamp_chunk_size_sec(99) == CHUNK_SIZE_SEC_MAX
    assert clamp_chunk_size_sec(1.5) == 1.5


class _MockModel:
    def __init__(self):
        self.calls = []

    def init_streaming_state(self, language=None, chunk_size_sec=None,
                             unfixed_chunk_num=None, unfixed_token_num=None):
        self.calls.append(("init", language, chunk_size_sec, unfixed_chunk_num, unfixed_token_num))
        return SimpleNamespace(text="", language=language or "")

    def streaming_transcribe(self, pcm, state):
        self.calls.append(("feed", int(pcm.size)))
        state.text += "x"
        state.language = "Chinese"
        return state

    def finish_streaming_transcribe(self, state):
        self.calls.append(("finish",))
        state.text += "。"
        return state


class _SamplingMockModel(_MockModel):
    def __init__(self):
        super().__init__()
        self.sampling_params = SimpleNamespace(max_tokens=4096)
        self.seen_max_tokens = []

    def streaming_transcribe(self, pcm, state):
        self.seen_max_tokens.append(self.sampling_params.max_tokens)
        return super().streaming_transcribe(pcm, state)

    def finish_streaming_transcribe(self, state):
        self.seen_max_tokens.append(self.sampling_params.max_tokens)
        return super().finish_streaming_transcribe(state)

    def transcribe(self, **kwargs):
        self.seen_max_tokens.append(self.sampling_params.max_tokens)
        return []


class _FailingSamplingMockModel(_SamplingMockModel):
    def streaming_transcribe(self, pcm, state):
        self.seen_max_tokens.append(self.sampling_params.max_tokens)
        raise ValueError("decode failed")


def _engine_with_model(**kw):
    eng = VLLMASREngine(**kw)
    eng._model = _MockModel()
    return eng


def test_engine_clamps_init_chunk_size():
    eng = VLLMASREngine(chunk_size_sec=99)
    assert eng.chunk_size_sec == CHUNK_SIZE_SEC_MAX


def test_new_state_uses_engine_chunk_size():
    eng = _engine_with_model(chunk_size_sec=2.0)
    eng.new_state(language="Chinese")
    assert eng._model.calls[0] == ("init", "Chinese", 2.0, 2, 5)


def test_new_state_chunk_size_override_clamped():
    eng = _engine_with_model(chunk_size_sec=1.0)
    eng.new_state(chunk_size_sec=99)              # 越界 → 钳到上限
    assert eng._model.calls[0][2] == CHUNK_SIZE_SEC_MAX


def test_feed_and_finish_sequence():
    eng = _engine_with_model()
    st = eng.new_state()
    t1, _ = eng.feed(np.zeros(1600, np.float32), st)
    t2, _ = eng.feed(np.zeros(1600, np.float32), st)
    tf, lang = eng.finish(st)
    assert (t1, t2, tf) == ("x", "xx", "xx。")
    assert lang == "Chinese"
    assert [c[0] for c in eng._model.calls] == ["init", "feed", "feed", "finish"]


def test_streaming_uses_bounded_tokens_without_affecting_offline():
    eng = VLLMASREngine(stream_max_new_tokens=256)
    eng._model = _SamplingMockModel()
    eng._stream_sampling_params = SimpleNamespace(max_tokens=256)
    original_params = eng._model.sampling_params
    state = eng.new_state()

    eng.feed(np.zeros(1600, np.float32), state)
    eng.finish(state)
    eng.transcribe(np.zeros(1600, np.float32))

    assert eng._model.seen_max_tokens == [256, 256, 4096]
    assert eng._model.sampling_params is original_params


def test_streaming_restores_sampling_params_after_failure():
    eng = VLLMASREngine(stream_max_new_tokens=256)
    eng._model = _FailingSamplingMockModel()
    eng._stream_sampling_params = SimpleNamespace(max_tokens=256)
    original_params = eng._model.sampling_params
    state = eng.new_state()

    try:
        eng.feed(np.zeros(1600, np.float32), state)
    except ValueError as exc:
        assert str(exc) == "decode failed"
    else:
        raise AssertionError("expected decode failure")

    assert eng._model.seen_max_tokens == [256]
    assert eng._model.sampling_params is original_params


def test_is_loaded():
    eng = VLLMASREngine()
    assert eng.is_loaded is False
    eng._model = _MockModel()
    assert eng.is_loaded is True


def test_align_device_normalized():
    """对齐器设备规范化为小写；缺省/None → cuda（OOM 时可经 --vllm-align-device cpu 移出 GPU）。"""
    assert VLLMASREngine()._align_device == "cuda"
    assert VLLMASREngine(align_device="CPU")._align_device == "cpu"
    assert VLLMASREngine(align_device=None)._align_device == "cuda"


def test_infer_batch_size_default_bounded():
    """对齐/ASR 批大小默认有界（4，非 -1）：防长音频把全部 180s 块一次对齐致 OOM。"""
    assert VLLMASREngine()._infer_batch_size == 4
    assert VLLMASREngine(infer_batch_size=1)._infer_batch_size == 1
