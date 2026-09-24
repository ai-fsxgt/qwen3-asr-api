"""StreamSession 客户端按会话覆盖与忽略项提示。"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from app.runtime.stream_session import StreamSession


def _make_session(*, punc=None, enable_words=False):
    return StreamSession(
        "sid",
        MagicMock(),
        MagicMock(),
        punc,
        ThreadPoolExecutor(max_workers=1),
        asyncio.Semaphore(1),
        enable_words=enable_words,
    )


def test_override_segment_fields():
    session = _make_session()
    session.configure({"max_end_silence_ms": 500, "max_segment_sec": 20})
    assert session._max_end_silence_ms == 500
    assert session._max_segment_sec == 20


def test_reconfigure_resets_session_options():
    import app.config as cfg

    session = _make_session()
    session.configure({
        "max_end_silence_ms": 500,
        "max_segment_sec": 20,
        "with_punc": False,
        "with_words": False,
    })
    session.configure({})

    assert session._max_end_silence_ms == cfg.VAD_MAX_SILENCE
    assert session._max_segment_sec == session._default_max_segment_sec
    assert session._with_punc is True
    assert session._with_words is True


@pytest.mark.parametrize("message", [
    {"max_end_silence_ms": 5000},
    {"max_segment_sec": 0},
    {"with_punc": "yes"},
])
def test_invalid_override_raises(message):
    session = _make_session()
    with pytest.raises(ValueError):
        session.configure(message)


def test_legacy_speaker_options_are_ignored():
    session = _make_session()
    warnings = session.configure({
        "diarize": True,
        "identify_speakers": True,
        "speaker_threshold": 0.6,
        "speaker_min_seg_ms": 800,
        "speaker_max": 5,
        "speaker_id_threshold": 0.5,
        "speaker_id_margin": 0.2,
    })
    assert warnings == []


def test_warns_when_output_feature_not_enabled():
    session = _make_session()
    warnings = session.configure({"with_words": True, "with_punc": True})
    assert set(warnings) == {"with_words", "with_punc"}


def test_no_warnings_when_output_features_available():
    session = _make_session(punc=MagicMock(), enable_words=True)
    warnings = session.configure({"with_words": True, "with_punc": True})
    assert warnings == []


def test_disable_toggle_never_warns():
    session = _make_session()
    warnings = session.configure({"with_punc": False, "with_words": False})
    assert warnings == []
    assert session._with_punc is False
    assert session._with_words is False
