"""app/runtime/speaker_service.py 单元测试（内存 Store + fake 引擎/VAD）。

覆盖：enroll 质量门槛（时长/多人/consent）、模板均值入库、identify_file、
map_clusters 异常兜底、自动登记分支全集（过门槛/时长不足/开关关/序号递增/失败退回匿名）、
临时文件清理、留存音频。阈值只测逻辑分支（V0 标定铁律）。
"""
import os
import threading
import types
import uuid

import numpy as np
import pytest

import app.config as cfg
from app.runtime.speaker_service import SpeakerService
from app.runtime.speaker_store import (
    SpeakerAutoEnrollment,
    SpeakerClaimConflictError,
    SpeakerIdentification,
    SpeakerStoreError,
)

DIM = 192


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


class FakeEngine:
    """窗起点 < split_at 秒 → unit(vec_idx)，否则 unit(vec_idx+1)（制造多人样本）。"""

    def __init__(self, vec_idx=0, split_at=None):
        self.vec_idx = vec_idx
        self.split_at = split_at

    def embed_windows(self, wav, windows, *, cancelled=None):
        sample_value = float(np.asarray(wav).reshape(-1)[0])
        base_idx = (
            int(round(sample_value)) - 1
            if sample_value >= 1.0
            else self.vec_idx
        )
        out = []
        for st, _ in windows:
            i = base_idx + (
                1 if self.split_at is not None and st >= self.split_at else 0
            )
            out.append(unit(i))
        return np.stack(out)


class FakeSpeakerStore:
    """只实现 SpeakerService 所需契约，避免单元测试依赖外部数据库。"""

    def __init__(self):
        self._speakers = {}
        self._claims = {}
        self._next_template_id = 1
        self._auto_name_seq = 0
        self._closed = False
        self._lock = threading.RLock()

    def _ensure_open(self):
        if self._closed:
            raise SpeakerStoreError("声纹库连接池已关闭")

    @staticmethod
    def _centroid(vectors):
        centroid = np.stack(vectors).mean(axis=0)
        return centroid / np.linalg.norm(centroid)

    def _new_templates(self, vectors, durations):
        templates = []
        for vector, duration in zip(vectors, durations, strict=True):
            templates.append({
                "id": self._next_template_id,
                "vector": np.asarray(vector, dtype=np.float32),
                "dur_sec": float(duration),
            })
            self._next_template_id += 1
        return templates

    def enroll_speaker(
        self,
        name,
        note,
        vectors,
        durations,
        consent,
        source="manual",
    ):
        self._ensure_open()
        if consent is not True:
            raise SpeakerStoreError("登记必须携带 consent=true")
        with self._lock:
            speaker_id = uuid.uuid4().hex
            self._speakers[speaker_id] = {
                "id": speaker_id,
                "name": name,
                "note": note,
                "source": source,
                "centroid": self._centroid(vectors),
                "templates": self._new_templates(vectors, durations),
            }
            return speaker_id

    def get_speaker(self, speaker_id):
        self._ensure_open()
        with self._lock:
            row = self._speakers.get(speaker_id)
            if row is None:
                return None
            return {
                "id": row["id"],
                "name": row["name"],
                "note": row["note"],
                "source": row["source"],
                "templates": [
                    {
                        "id": template["id"],
                        "dur_sec": template["dur_sec"],
                    }
                    for template in row["templates"]
                ],
            }

    def delete_speaker(self, speaker_id):
        self._ensure_open()
        with self._lock:
            self._speakers.pop(speaker_id, None)

    def resolve_claim(self, speaker_id, claim_key):
        self._ensure_open()
        with self._lock:
            claim = self._claims.get(claim_key)
            if claim is not None:
                if claim["speaker_id"] != speaker_id:
                    owner = self._speakers[claim["speaker_id"]]
                    raise SpeakerClaimConflictError(
                        "claim_key_conflict",
                        "claim_key 已用于其他说话人",
                        speaker_id=owner["id"],
                        speaker_name=owner["name"],
                        source=owner["source"],
                    )
                return claim["response"]
            speaker = self._speakers[speaker_id]
            if speaker["source"] != "auto":
                raise SpeakerClaimConflictError(
                    "speaker_not_claimable",
                    "仅自动登记的说话人可以被认领",
                    speaker_id=speaker_id,
                    speaker_name=speaker["name"],
                    source=speaker["source"],
                )
            return None

    def claim_auto_speaker(
        self,
        speaker_id,
        name,
        note,
        vectors,
        durations,
        claim_key,
        quality_hint=None,
    ):
        self._ensure_open()
        with self._lock:
            replay = self.resolve_claim(speaker_id, claim_key)
            if replay is not None:
                return replay, False
            templates = self._new_templates(vectors, durations)
            speaker = self._speakers[speaker_id]
            speaker.update({
                "name": name,
                "note": note,
                "source": "manual",
                "centroid": self._centroid(vectors),
                "templates": templates,
            })
            response = {
                "speaker_id": speaker_id,
                "name": name,
                "source": "manual",
                "templates": len(templates),
                "template_ids": [template["id"] for template in templates],
                "quality_hint": quality_hint,
            }
            self._claims[claim_key] = {
                "speaker_id": speaker_id,
                "response": response,
            }
            return response, True

    def _identify_locked(self, vector, threshold, margin):
        rows = sorted(
            (
                (float(speaker["centroid"] @ vector), speaker)
                for speaker in self._speakers.values()
                if speaker["templates"]
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:2]
        if not rows:
            return SpeakerIdentification(False, "empty_library")
        top1 = rows[0][0]
        top2 = rows[1][0] if len(rows) > 1 else None
        score_margin = top1 - top2 if top2 is not None else None
        if top1 < threshold:
            return SpeakerIdentification(
                False,
                "below_threshold",
                top1_score=top1,
                top2_score=top2,
                margin=score_margin,
            )
        if score_margin is not None and score_margin < margin:
            return SpeakerIdentification(
                False,
                "ambiguous_margin",
                top1_score=top1,
                top2_score=top2,
                margin=score_margin,
            )
        speaker = rows[0][1]
        return SpeakerIdentification(
            True,
            "matched",
            top1_score=top1,
            top2_score=top2,
            margin=score_margin,
            speaker_id=speaker["id"],
            name=speaker["name"],
            source=speaker["source"],
        )

    def identify_detailed(self, vector, threshold=0.45, margin=0.10):
        self._ensure_open()
        with self._lock:
            return self._identify_locked(vector, threshold, margin)

    def identify_many_detailed(
        self,
        vectors,
        threshold=0.45,
        margin=0.10,
    ):
        self._ensure_open()
        with self._lock:
            return [
                self._identify_locked(vector, threshold, margin)
                for vector in vectors
            ]

    def identify(
        self,
        vector,
        threshold=0.45,
        margin=0.10,
        *,
        include_source=False,
    ):
        return self.identify_detailed(vector, threshold, margin).as_match(
            include_source=include_source
        )

    def identify_or_enroll_auto(
        self,
        vector,
        duration,
        *,
        threshold,
        margin,
    ):
        self._ensure_open()
        with self._lock:
            identification = self._identify_locked(vector, threshold, margin)
            if identification.matched:
                return SpeakerAutoEnrollment(
                    identification,
                    identification.speaker_id,
                    identification.name,
                    False,
                )
            self._auto_name_seq += 1
            name = f"说话人_{self._auto_name_seq:02d}"
            speaker_id = self.enroll_speaker(
                name,
                None,
                [vector],
                [duration],
                True,
                source="auto",
            )
            return SpeakerAutoEnrollment(
                identification,
                speaker_id,
                name,
                True,
            )

    def audit(self, *args, **kwargs):
        self._ensure_open()

    @property
    def speaker_count(self):
        self._ensure_open()
        with self._lock:
            return len(self._speakers)

    def close(self):
        self._closed = True


def make_vad(segments):
    return types.SimpleNamespace(detect=lambda p: segments)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """隔离上传目录/服务根 + fake 音频 IO（convert 落空文件、sf.read 给 8s 假音频）。"""
    monkeypatch.setattr(cfg, "UPLOADS_DIR", str(tmp_path / "up"))
    monkeypatch.setattr(cfg, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr("app.runtime.speaker_service.convert_to_wav",
                        lambda src, dst: open(dst, "wb").write(b"x"))
    monkeypatch.setattr("app.runtime.speaker_service.sf.read",
                        lambda p, dtype=None: (np.zeros(16000 * 8, dtype=np.float32), 16000))
    return tmp_path


@pytest.fixture
def store():
    return FakeSpeakerStore()


def make_service(store, engine=None, vad_segments=((0, 5000),)):
    return SpeakerService(store, engine or FakeEngine(), make_vad(list(vad_segments)))


def _src(tmp_path, name="a.mp3"):
    p = tmp_path / name
    p.write_bytes(b"fake")
    return str(p)


# ─── enroll ───

def test_enroll_ok_with_quality_hint(env, store):
    svc = make_service(store)
    resp = svc.enroll("张三", "备注", [_src(env)], consent=True)
    assert len(resp["speaker_id"]) == 32
    assert resp["templates"] == 1
    assert "quality_hint" in resp                       # 模板 <3 提示
    assert store.get_speaker(resp["speaker_id"])["source"] == "manual"


def test_enroll_three_samples_no_hint(env, store):
    svc = make_service(store)
    resp = svc.enroll("张三", None, [_src(env, f"{i}.mp3") for i in range(3)], consent=True)
    assert resp["templates"] == 3
    assert "quality_hint" not in resp


def test_enroll_rejects_short_speech(env, store):
    svc = make_service(store, vad_segments=[(0, 2000)])   # 2s < 3s 门槛
    with pytest.raises(ValueError, match="有效语音不足"):
        svc.enroll("x", None, [_src(env)], consent=True)


def test_enroll_rejects_multi_speaker_sample(env, store):
    svc = make_service(store, engine=FakeEngine(split_at=2.5))   # 前后两人
    with pytest.raises(ValueError, match="多个说话人"):
        svc.enroll("x", None, [_src(env)], consent=True)


def test_enroll_requires_consent(env, store):
    svc = make_service(store)
    with pytest.raises(ValueError, match="consent"):
        svc.enroll("x", None, [_src(env)], consent=False)


def test_enroll_rejects_empty_files(env, store):
    with pytest.raises(ValueError, match="样本"):
        make_service(store).enroll("x", None, [], consent=True)


def test_temp_wavs_cleaned_on_success_and_failure(env, store):
    svc = make_service(store)
    svc.enroll("a", None, [_src(env)], consent=True)
    svc_fail = make_service(store, vad_segments=[(0, 1000)])
    with pytest.raises(ValueError):
        svc_fail.enroll("b", None, [_src(env)], consent=True)
    leftovers = [f for f in os.listdir(cfg.UPLOADS_DIR) if f.startswith("spk_")]
    assert leftovers == []


def test_store_audio_kept_and_removed_on_delete(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_STORE_AUDIO", True)
    svc = make_service(store)
    sid = svc.enroll("a", None, [_src(env)], consent=True)["speaker_id"]
    audio_dir = os.path.join(str(env), "data", "speaker_audio", sid)
    assert os.path.isfile(os.path.join(audio_dir, "00.wav"))
    svc.delete_speaker(sid)
    assert not os.path.isdir(audio_dir)                  # 被遗忘权：音频同步清理


# ─── claim ───

def test_claim_auto_speaker_replaces_templates(env, store):
    sid = store.enroll_speaker(
        "说话人_01",
        None,
        [unit(5)],
        [12.0],
        consent=True,
        source="auto",
    )
    svc = make_service(store)

    response = svc.claim(
        sid,
        " 张三 ",
        "产品部",
        [_src(env, "claim.mp3")],
        True,
        " profile-1 ",
    )

    assert response["speaker_id"] == sid
    assert response["source"] == "manual"
    assert response["templates"] == 1
    assert len(response["template_ids"]) == 1
    assert response["quality_hint"]
    info = store.get_speaker(sid)
    assert info["name"] == "张三" and info["note"] == "产品部"
    assert info["source"] == "manual"
    assert store.identify(unit(0))["speaker_id"] == sid


def test_claim_replay_skips_audio_processing(env, store, monkeypatch):
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(5)], [12.0], True, source="auto"
    )
    svc = make_service(store)
    first = svc.claim(
        sid, "张三", None, [_src(env)], True, "profile-1"
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("幂等重试不应重新处理音频")

    monkeypatch.setattr(svc, "_embed_file", fail_if_called)
    replay = svc.claim(
        sid, "其他名称", "其他备注", [_src(env, "retry.mp3")],
        True, "profile-1",
    )
    assert replay == first


def test_claim_manual_conflict_happens_before_audio_processing(
    env, store, monkeypatch
):
    sid = store.enroll_speaker(
        "张三", None, [unit(0)], [5.0], consent=True
    )
    svc = make_service(store)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("手动声纹冲突不应处理音频")

    monkeypatch.setattr(svc, "_embed_file", fail_if_called)
    with pytest.raises(SpeakerClaimConflictError) as exc:
        svc.claim(
            sid, "李四", None, [_src(env)], True, "profile-1"
        )
    assert exc.value.code == "speaker_not_claimable"


def test_claim_validates_consent_name_and_claim_key(env, store):
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(5)], [12.0], True, source="auto"
    )
    svc = make_service(store)
    path = _src(env)

    with pytest.raises(ValueError, match="consent"):
        svc.claim(sid, "张三", None, [path], False, "profile-1")
    with pytest.raises(ValueError, match="名称"):
        svc.claim(sid, " ", None, [path], True, "profile-1")
    with pytest.raises(ValueError, match="claim_key"):
        svc.claim(sid, "张三", None, [path], True, " ")


def test_claim_replaces_retained_audio(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_STORE_AUDIO", True)
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(5)], [12.0], True, source="auto"
    )
    audio_dir = os.path.join(str(env), "data", "speaker_audio", sid)
    os.makedirs(audio_dir, exist_ok=True)
    with open(os.path.join(audio_dir, "old.wav"), "wb") as file:
        file.write(b"old")

    response = make_service(store).claim(
        sid,
        "张三",
        None,
        [_src(env, "a.mp3"), _src(env, "b.mp3")],
        True,
        "profile-1",
    )

    expected = {f"{template_id}.wav" for template_id in response["template_ids"]}
    assert set(os.listdir(audio_dir)) == expected


def test_claim_and_delete_serialize_retained_audio(
    env, store, monkeypatch
):
    monkeypatch.setattr(cfg, "SPEAKER_STORE_AUDIO", True)
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(5)], [12.0], True, source="auto"
    )
    svc = make_service(store)
    audio_dir = os.path.join(str(env), "data", "speaker_audio", sid)
    replace_started = threading.Event()
    allow_replace = threading.Event()
    delete_started = threading.Event()
    delete_finished = threading.Event()
    errors = []
    original_replace = svc._replace_retained_audio

    def blocking_replace(*args):
        replace_started.set()
        if not allow_replace.wait(timeout=2):
            raise AssertionError("测试未放行留存音频替换")
        original_replace(*args)

    def run_claim():
        try:
            svc.claim(
                sid, "张三", None, [_src(env)], True, "profile-1"
            )
        except Exception as exc:  # pragma: no cover - 失败由主线程断言
            errors.append(exc)

    def run_delete():
        delete_started.set()
        try:
            svc.delete_speaker(sid)
        except Exception as exc:  # pragma: no cover - 失败由主线程断言
            errors.append(exc)
        finally:
            delete_finished.set()

    monkeypatch.setattr(svc, "_replace_retained_audio", blocking_replace)
    claim_thread = threading.Thread(target=run_claim)
    delete_thread = threading.Thread(target=run_delete)
    claim_thread.start()
    assert replace_started.wait(timeout=2)
    delete_thread.start()
    assert delete_started.wait(timeout=2)
    try:
        assert not delete_finished.wait(timeout=0.1)
    finally:
        allow_replace.set()
        claim_thread.join(timeout=2)
        delete_thread.join(timeout=2)

    assert not claim_thread.is_alive() and not delete_thread.is_alive()
    assert errors == []
    assert store.get_speaker(sid) is None
    assert not os.path.isdir(audio_dir)


# ─── identify_file ───

def test_identify_file_hit_and_miss(env, store):
    svc = make_service(store)
    sid = svc.enroll("张三", None, [_src(env)], consent=True)["speaker_id"]
    hit = svc.identify_file(_src(env, "q.mp3"))
    assert hit["matched"] is True and hit["speaker_id"] == sid and hit["name"] == "张三"
    assert hit["source"] == "manual"

    svc_other = make_service(store, engine=FakeEngine(vec_idx=5))
    miss = svc_other.identify_file(_src(env, "q2.mp3"))
    assert miss["matched"] is False
    assert miss["reason"] == "below_threshold"


def test_identify_file_returns_auto_source(env, store):
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(0)], [12.0], True, source="auto"
    )
    hit = make_service(store).identify_file(_src(env))
    assert hit["speaker_id"] == sid
    assert hit["name"] == "说话人_01"
    assert hit["source"] == "auto"


# ─── map_clusters（实时联动：仅识别）───

def test_map_clusters_hit_and_miss(env, store):
    svc = make_service(store)
    sid = svc.enroll("张三", None, [_src(env)], consent=True)["speaker_id"]
    out = svc.map_clusters([
        {"label": "A", "centroid": unit(0), "dur_sec": 6.0},
        {"label": "B", "centroid": unit(7), "dur_sec": 6.0},
    ])
    assert out[0]["speaker_id"] == sid and out[0]["name"] == "张三"
    assert out[1] == {"label": "B", "speaker_id": None, "name": None, "score": None}


def test_map_clusters_exception_falls_back_anonymous(env, store):
    svc = make_service(store)
    out = svc.map_clusters([{"label": "A"}])             # 缺 centroid → 内部异常
    assert out == [{"label": "A", "speaker_id": None, "name": None, "score": None}]


def test_map_clusters_never_auto_enrolls(env, store):
    svc = make_service(store)
    svc.map_clusters([{"label": "A", "centroid": unit(3), "dur_sec": 99.0}])
    assert store.speaker_count == 0                      # 实时路径绝不建档


# ─── map_and_enroll_clusters（离线联动：识别 + 自动登记）───

def _cluster(label, vec, dur):
    identity_index = int(np.argmax(vec))
    return {
        "label": label,
        "centroid": vec,
        "dur_sec": dur,
        "identity_audio": np.full(
            int(dur * 16000),
            identity_index + 1,
            dtype=np.float32,
        ),
        "identity_intervals": [(0.0, dur)],
        "identity_duration_sec": dur,
    }


def test_auto_enroll_above_threshold(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_AUTO_ENROLL", True)
    svc = make_service(store)
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 12.0)])
    assert out[0]["name"] == "说话人_01" and out[0]["auto_enrolled"] is True
    assert store.get_speaker(out[0]["speaker_id"])["source"] == "auto"


def test_auto_enroll_sequence_increments(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_AUTO_ENROLL", True)
    svc = make_service(store)
    out = svc.map_and_enroll_clusters([
        _cluster("A", unit(0), 12.0), _cluster("B", unit(1), 15.0),
    ])
    assert [m["name"] for m in out] == ["说话人_01", "说话人_02"]


def test_auto_enroll_below_duration_stays_anonymous(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_AUTO_ENROLL", True)
    svc = make_service(store)
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 5.0)])  # <10s
    assert out[0]["speaker_id"] is None
    assert store.speaker_count == 0


def test_auto_enroll_disabled_stays_anonymous(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_AUTO_ENROLL", False)
    svc = make_service(store)
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 12.0)])
    assert out[0]["speaker_id"] is None
    assert store.speaker_count == 0


def test_auto_enroll_hit_does_not_re_enroll(env, store):
    svc = make_service(store)
    sid = svc.enroll("张三", None, [_src(env)], consent=True)["speaker_id"]
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 12.0)])
    assert out[0]["speaker_id"] == sid and out[0]["name"] == "张三"
    assert store.speaker_count == 1                      # 命中不重复建档（防投毒）


def test_offline_short_cluster_stays_anonymous_even_when_voiceprint_matches(
    env, store
):
    svc = make_service(store)
    svc.enroll("张三", None, [_src(env)], consent=True)
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 4.0)])
    assert {
        key: out[0][key]
        for key in ("label", "speaker_id", "name", "score")
    } == {"label": "A", "speaker_id": None, "name": None, "score": None}
    assert out[0]["_identity_anchors"] == []


def test_offline_missing_evidence_does_not_fallback_to_centroid(env, store):
    store.enroll_speaker("张三", None, [unit(0)], [6.0], consent=True)
    svc = make_service(store)
    out = svc.map_and_enroll_clusters([{
        "label": "A",
        "centroid": unit(0),
        "dur_sec": 12.0,
    }])
    assert out == [
        {"label": "A", "speaker_id": None, "name": None, "score": None}
    ]


def test_offline_inconsistent_evidence_stays_anonymous(env, store):
    store.enroll_speaker("甲", None, [unit(0)], [6.0], consent=True)
    store.enroll_speaker("乙", None, [unit(1)], [6.0], consent=True)
    svc = make_service(store, engine=FakeEngine(split_at=2.5))
    centroid = (unit(0) + unit(1)) / np.sqrt(2)
    out = svc.map_and_enroll_clusters([_cluster("A", centroid, 10.0)])
    assert {
        key: out[0][key]
        for key in ("label", "speaker_id", "name", "score")
    } == {"label": "A", "speaker_id": None, "name": None, "score": None}
    assert out[0]["_identity_anchors"]


def test_auto_enroll_serializes_identify_and_enroll(
    env, store, monkeypatch
):
    monkeypatch.setattr(cfg, "SPEAKER_AUTO_ENROLL", True)
    svc = make_service(store)
    original_identify = store.identify_detailed
    first_identify_entered = threading.Event()
    second_identify_entered = threading.Event()
    second_worker_started = threading.Event()
    release_first_identify = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def blocking_identify(*args, **kwargs):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_identify_entered.set()
            if not release_first_identify.wait(timeout=2):
                raise AssertionError("测试未放行首次识别")
        else:
            second_identify_entered.set()
        return original_identify(*args, **kwargs)

    results = {}

    def run(label):
        if label == "B":
            second_worker_started.set()
        results[label] = svc.map_and_enroll_clusters(
            [_cluster(label, unit(0), 12.0)]
        )[0]

    monkeypatch.setattr(store, "identify_detailed", blocking_identify)
    first_thread = threading.Thread(target=run, args=("A",))
    second_thread = threading.Thread(target=run, args=("B",))
    first_thread.start()
    assert first_identify_entered.wait(timeout=2)
    second_thread.start()
    assert second_worker_started.wait(timeout=2)
    try:
        assert not second_identify_entered.wait(timeout=0.2)
    finally:
        release_first_identify.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert second_identify_entered.is_set()
    assert store.speaker_count == 1
    assert results["A"]["speaker_id"] == results["B"]["speaker_id"]
    assert sum("auto_enrolled" in result for result in results.values()) == 1


def test_auto_enroll_failure_falls_back_anonymous(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_AUTO_ENROLL", True)
    svc = make_service(store)
    store.close()                                        # alloc/enroll 将失败
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 12.0)])
    assert out[0]["speaker_id"] is None                  # 退回匿名，不抛错


def test_auto_enroll_stops_when_cancelled_after_identify(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_AUTO_ENROLL", True)
    svc = make_service(store)
    cancelled = False
    identify = store.identify_detailed

    def identify_then_cancel(*args, **kwargs):
        nonlocal cancelled
        result = identify(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(store, "identify_detailed", identify_then_cancel)
    out = svc.map_and_enroll_clusters(
        [_cluster("A", unit(0), 12.0)],
        cancelled=lambda: cancelled,
    )

    assert out == []
    assert store.speaker_count == 0
