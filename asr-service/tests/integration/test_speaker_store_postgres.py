"""PostgreSQL/pgvector 声纹库存储集成测试。

设置 TEST_SPEAKER_DATABASE_URL 后运行；测试库必须是可清空的专用数据库。
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psycopg
import pytest

from app.runtime.speaker_store import (
    SpeakerClaimConflictError,
    SpeakerNotFoundError,
    SpeakerStore,
    SpeakerStoreError,
)

DATABASE_URL = os.environ.get("TEST_SPEAKER_DATABASE_URL")
TAG = "redimnet2_b6_vb2_vox2_cnc2_v0_lm@v1"
DIM = 192

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="未配置 TEST_SPEAKER_DATABASE_URL",
)


def unit(index: int) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_concurrent_initialization_on_empty_database():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(
            """
            DROP TABLE IF EXISTS speaker_audit_log, speaker_claims,
              speaker_templates, speakers, speaker_meta CASCADE
            """
        )
        conn.execute("DROP EXTENSION IF EXISTS vector")

    barrier = threading.Barrier(8)

    def open_store(_index):
        barrier.wait()
        return SpeakerStore(
            DATABASE_URL,
            model_tag=TAG,
            pool_min_size=1,
            pool_max_size=1,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        stores = list(executor.map(open_store, range(8)))
    try:
        assert len(stores) == 8
    finally:
        for instance in stores:
            instance.close()


def test_schema_initialization_lock_has_timeout():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(
            "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
            (SpeakerStore._SCHEMA_INIT_LOCK,),
        )
        started = time.monotonic()
        try:
            with pytest.raises(SpeakerStoreError, match="初始化锁等待超时"):
                SpeakerStore(
                    DATABASE_URL,
                    model_tag=TAG,
                    pool_min_size=1,
                    pool_max_size=1,
                    pool_timeout=0.2,
                )
        finally:
            conn.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (SpeakerStore._SCHEMA_INIT_LOCK,),
            )
        assert time.monotonic() - started < 2.0


@pytest.fixture
def store():
    instance = SpeakerStore(DATABASE_URL, model_tag=TAG)
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(
            "TRUNCATE speaker_audit_log, speaker_claims, "
            "speaker_templates, speakers RESTART IDENTITY CASCADE"
        )
        conn.execute("DELETE FROM speaker_meta WHERE key='auto_name_seq'")
    yield instance
    instance.close()


def test_shared_store_write_is_immediately_visible(store):
    second = SpeakerStore(DATABASE_URL, model_tag=TAG)
    try:
        speaker_id = store.enroll_speaker(
            "张三", None, [unit(0)], [6.0], consent=True
        )
        match = second.identify(unit(0))
        assert match["speaker_id"] == speaker_id
        assert match["name"] == "张三"

        store.update_speaker(speaker_id, name="张三（新）")
        assert second.identify(unit(0))["name"] == "张三（新）"

        store.delete_speaker(speaker_id)
        assert second.identify(unit(0)) is None
    finally:
        second.close()


def test_identification_rejects_low_score_and_ambiguous_margin(store):
    store.enroll_speaker("甲", None, [unit(0)], [6.0], consent=True)
    store.enroll_speaker("乙", None, [unit(1)], [6.0], consent=True)

    low_score = store.identify_detailed(unit(2), threshold=0.45, margin=0.10)
    assert low_score.matched is False
    assert low_score.reason == "below_threshold"

    ambiguous_vector = (unit(0) + unit(1)) / np.sqrt(2)
    ambiguous = store.identify_detailed(
        ambiguous_vector.astype(np.float32), threshold=0.45, margin=0.10
    )
    assert ambiguous.matched is False
    assert ambiguous.reason == "ambiguous_margin"
    assert ambiguous.margin == pytest.approx(0.0, abs=1e-6)


def test_template_crud_recomputes_centroid_and_empty_speaker_cannot_match(store):
    speaker_id = store.enroll_speaker(
        "甲", None, [unit(0)], [6.0], consent=True
    )
    store.add_template(speaker_id, unit(1), 7.0)
    info = store.get_speaker(speaker_id)
    assert len(info["templates"]) == 2

    for template in info["templates"]:
        remaining = store.delete_template(speaker_id, template["id"])
    assert remaining == 0
    assert store.identify(unit(0)) is None

    with pytest.raises(SpeakerNotFoundError):
        store.delete_template(speaker_id, 999999)


def test_claim_is_idempotent_and_preserves_speaker_id(store):
    speaker_id = store.enroll_speaker(
        "说话人_01",
        None,
        [unit(0)],
        [12.0],
        consent=True,
        source="auto",
    )
    first, created = store.claim_auto_speaker(
        speaker_id,
        "张三",
        "本人确认",
        [unit(1)],
        [8.0],
        "claim-001",
    )
    replay, replay_created = store.claim_auto_speaker(
        speaker_id,
        "不应覆盖",
        None,
        [unit(2)],
        [8.0],
        "claim-001",
    )
    assert created is True
    assert replay_created is False
    assert replay == first
    assert replay["speaker_id"] == speaker_id
    assert store.get_speaker(speaker_id)["name"] == "张三"


def test_claim_key_cannot_be_reused_for_another_speaker(store):
    first_id = store.enroll_speaker(
        "说话人_01", None, [unit(0)], [12.0], True, source="auto"
    )
    second_id = store.enroll_speaker(
        "说话人_02", None, [unit(1)], [12.0], True, source="auto"
    )
    store.claim_auto_speaker(
        first_id, "甲", None, [unit(0)], [8.0], "same-key"
    )
    with pytest.raises(SpeakerClaimConflictError) as exc_info:
        store.resolve_claim(second_id, "same-key")
    assert exc_info.value.code == "claim_key_conflict"


def test_auto_name_allocation_is_atomic_across_instances(store):
    second = SpeakerStore(DATABASE_URL, model_tag=TAG)
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            names = list(
                executor.map(
                    lambda index: (store if index % 2 else second).alloc_auto_name(),
                    range(20),
                )
            )
        assert len(set(names)) == 20
        assert min(names) == "说话人_01"
        assert max(names) == "说话人_20"
    finally:
        second.close()


def test_auto_enroll_is_atomic_across_instances(store):
    second = SpeakerStore(DATABASE_URL, model_tag=TAG)
    barrier = threading.Barrier(2)

    def auto_enroll(instance):
        barrier.wait()
        return instance.identify_or_enroll_auto(
            unit(0),
            12.0,
            threshold=0.45,
            margin=0.10,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    auto_enroll,
                    (store, second),
                )
            )
        assert sum(result.created for result in results) == 1
        assert len({result.speaker_id for result in results}) == 1
        assert store.speaker_count == 1
    finally:
        second.close()


@pytest.mark.parametrize(
    "vector,error",
    [
        (np.zeros(DIM, dtype=np.float32), "L2 归一"),
        (np.ones(DIM - 1, dtype=np.float32), "向量维度"),
    ],
)
def test_invalid_vectors_are_rejected(store, vector, error):
    with pytest.raises(SpeakerStoreError, match=error):
        store.enroll_speaker("非法", None, [vector], [6.0], True)
