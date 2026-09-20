from app.runtime.speaker_identity import RealtimeIdentityTracker


def match(speaker_id="speaker-1", name="张三"):
    return {"speaker_id": speaker_id, "name": name, "score": 0.8}


def test_name_is_exposed_after_two_consistent_matches():
    tracker = RealtimeIdentityTracker(required_confirmations=2)
    assert tracker.observe("A", match()) is None
    assert tracker.observe("A", match()) == "张三"


def test_single_conflict_does_not_replace_confirmed_identity():
    tracker = RealtimeIdentityTracker(required_confirmations=2)
    tracker.observe("A", match())
    tracker.observe("A", match())

    assert tracker.observe("A", match("speaker-2", "李四")) == "张三"
    assert tracker.observe("A", match("speaker-2", "李四")) == "李四"


def test_two_unknown_results_revoke_confirmed_identity():
    tracker = RealtimeIdentityTracker(required_confirmations=2)
    tracker.observe("A", match())
    tracker.observe("A", match())

    assert tracker.observe("A", None) == "张三"
    assert tracker.observe("A", None) is None


def test_labels_have_independent_state():
    tracker = RealtimeIdentityTracker(required_confirmations=2)
    tracker.observe("A", match("speaker-1", "张三"))
    tracker.observe("B", match("speaker-2", "李四"))
    assert tracker.observe("A", match("speaker-1", "张三")) == "张三"
    assert tracker.observe("B", match("speaker-2", "李四")) == "李四"
