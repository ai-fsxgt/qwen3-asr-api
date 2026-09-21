#!/usr/bin/env python3
"""手工迁移旧 SQLite 声纹库到 PostgreSQL/pgvector。

默认只检查源库；只有显式传入 --apply 才会连接并写入目标库。目标业务表必须
全部为空，避免把两个事实源合并成不可追溯的数据。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.runtime.speaker_store import SpeakerStore  # noqa: E402


@dataclass(frozen=True)
class SourceData:
    model_tag: str
    meta: dict[str, str]
    speakers: list[sqlite3.Row]
    templates: list[sqlite3.Row]
    claims: list[sqlite3.Row]
    audits: list[sqlite3.Row]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查或迁移 SQLite 声纹库到 PostgreSQL/pgvector"
    )
    parser.add_argument("--sqlite-path", required=True, help="旧 speakers.db 路径")
    parser.add_argument(
        "--postgres-url",
        default=os.environ.get("SPEAKER_DATABASE_URL"),
        help="目标连接地址；缺省读取 SPEAKER_DATABASE_URL",
    )
    parser.add_argument(
        "--source-timezone",
        default="Asia/Shanghai",
        help="旧 SQLite 无时区时间的所属时区 (default: Asia/Shanghai)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际写入；不传时只检查源库",
    )
    return parser


def _read_source(path: Path, source_timezone: ZoneInfo) -> SourceData:
    if not path.is_file():
        raise ValueError(f"SQLite 文件不存在: {path}")
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        # 所有表必须来自同一读取快照，避免旧服务写入时读到跨时点数据。
        connection.execute("BEGIN")
        meta = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM meta")
        }
        model_tag = meta.get("model_tag")
        if not model_tag:
            raise ValueError("源库缺少 meta.model_tag")
        claims_exist = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='speaker_claims'
            """
        ).fetchone()
        claims = (
            list(connection.execute("SELECT * FROM speaker_claims ORDER BY claim_key"))
            if claims_exist
            else []
        )
        data = SourceData(
            model_tag=model_tag,
            meta=meta,
            speakers=list(connection.execute("SELECT * FROM speakers ORDER BY id")),
            templates=list(connection.execute("SELECT * FROM templates ORDER BY id")),
            claims=claims,
            audits=list(connection.execute("SELECT * FROM audit_log ORDER BY id")),
        )
    except sqlite3.Error as exc:
        raise ValueError(f"读取 SQLite 失败: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    _validate_source(data, source_timezone)
    return data


def _vector(blob: bytes, *, owner: str) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32).copy()
    if vector.shape != (SpeakerStore.DIM,):
        raise ValueError(f"{owner} 向量维度异常: {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or abs(norm - 1.0) > SpeakerStore._NORM_TOL:
        raise ValueError(f"{owner} 向量未归一化: norm={norm:.6f}")
    return vector


def _validate_source(data: SourceData, source_timezone: ZoneInfo) -> None:
    speaker_ids = {row["id"] for row in data.speakers}
    for row in data.speakers:
        _vector(row["centroid"], owner=f"speaker {row['id']} centroid")
        _timestamp(
            row["created_at"],
            source_timezone,
            owner=f"speaker {row['id']} created_at",
        )
        _timestamp(
            row["updated_at"],
            source_timezone,
            owner=f"speaker {row['id']} updated_at",
        )
    for row in data.templates:
        if row["speaker_id"] not in speaker_ids:
            raise ValueError(f"模板 {row['id']} 引用了不存在的说话人")
        _vector(row["vector"], owner=f"template {row['id']}")
        try:
            duration = float(row["dur_sec"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"模板 {row['id']} 时长无效") from exc
        if not np.isfinite(duration) or duration < 0:
            raise ValueError(f"模板 {row['id']} 时长无效: {duration!r}")
        _timestamp(
            row["created_at"],
            source_timezone,
            owner=f"template {row['id']} created_at",
        )
    for row in data.claims:
        if row["speaker_id"] not in speaker_ids:
            raise ValueError(f"认领记录 {row['claim_key']} 引用了不存在的说话人")
        response = json.loads(row["response"])
        if not isinstance(response, dict):
            raise ValueError(f"认领记录 {row['claim_key']} 的 response 不是对象")
        _timestamp(
            row["created_at"],
            source_timezone,
            owner=f"claim {row['claim_key']} created_at",
        )
    for row in data.audits:
        detail = json.loads(row["detail"]) if row["detail"] is not None else None
        if detail is not None and not isinstance(detail, dict):
            raise ValueError(f"审计记录 {row['id']} 的 detail 不是对象")
        _timestamp(
            row["ts"],
            source_timezone,
            owner=f"audit {row['id']} ts",
        )


def _timestamp(
    raw: str,
    source_timezone: ZoneInfo,
    *,
    owner: str = "timestamp",
) -> datetime:
    try:
        value = datetime.fromisoformat(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner} 无效: {raw!r}") from exc
    return value.replace(tzinfo=source_timezone) if value.tzinfo is None else value


def _target_counts(conn: psycopg.Connection) -> dict[str, int]:
    tables = {
        "speakers": "speakers",
        "templates": "speaker_templates",
        "claims": "speaker_claims",
        "audits": "speaker_audit_log",
    }
    return {
        name: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for name, table in tables.items()
    }


def _insert_source(
    data: SourceData,
    postgres_url: str,
    source_timezone: ZoneInfo,
) -> None:
    store = SpeakerStore(postgres_url, model_tag=data.model_tag)
    try:
        if not store.check_model_tag(data.model_tag):
            raise ValueError("目标库 model_tag 与源库不一致")
        speakers_with_templates = {
            row["speaker_id"] for row in data.templates
        }
        with psycopg.connect(postgres_url, autocommit=True) as conn:
            register_vector(conn)
            with conn.transaction():
                # 阻止运行中的 ASR 实例在空库检查与迁移提交之间写入。
                conn.execute(
                    """
                    LOCK TABLE speakers, speaker_templates, speaker_claims,
                      speaker_audit_log, speaker_meta
                    IN ACCESS EXCLUSIVE MODE
                    """
                )
                counts = _target_counts(conn)
                if any(counts.values()):
                    raise ValueError(f"目标库不是空库，拒绝迁移: {counts}")
                for row in data.speakers:
                    conn.execute(
                        """
                        INSERT INTO speakers(
                          id, name, note, consent, source, model_tag,
                          centroid, created_at, updated_at
                        ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            row["id"],
                            row["name"],
                            row["note"],
                            bool(row["consent"]),
                            row["source"],
                            row["model_tag"],
                            (
                                _vector(
                                    row["centroid"],
                                    owner=f"speaker {row['id']} centroid",
                                )
                                if row["id"] in speakers_with_templates
                                else None
                            ),
                            _timestamp(row["created_at"], source_timezone),
                            _timestamp(row["updated_at"], source_timezone),
                        ),
                    )
                for row in data.templates:
                    conn.execute(
                        """
                        INSERT INTO speaker_templates(
                          id, speaker_id, vector, dur_sec, created_at
                        ) VALUES(%s, %s, %s, %s, %s)
                        """,
                        (
                            row["id"],
                            row["speaker_id"],
                            _vector(row["vector"], owner=f"template {row['id']}"),
                            float(row["dur_sec"]),
                            _timestamp(row["created_at"], source_timezone),
                        ),
                    )
                for row in data.claims:
                    conn.execute(
                        """
                        INSERT INTO speaker_claims(
                          claim_key, speaker_id, response, created_at
                        ) VALUES(%s, %s, %s, %s)
                        """,
                        (
                            row["claim_key"],
                            row["speaker_id"],
                            Jsonb(json.loads(row["response"])),
                            _timestamp(row["created_at"], source_timezone),
                        ),
                    )
                for row in data.audits:
                    detail = (
                        json.loads(row["detail"])
                        if row["detail"] is not None
                        else None
                    )
                    conn.execute(
                        """
                        INSERT INTO speaker_audit_log(
                          id, ts, action, speaker_id, detail
                        ) VALUES(%s, %s, %s, %s, %s)
                        """,
                        (
                            row["id"],
                            _timestamp(row["ts"], source_timezone),
                            row["action"],
                            row["speaker_id"],
                            Jsonb(detail) if detail is not None else None,
                        ),
                    )
                for key in ("created_at", "auto_name_seq"):
                    if key in data.meta:
                        conn.execute(
                            """
                            INSERT INTO speaker_meta(key, value) VALUES(%s, %s)
                            ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
                            """,
                            (key, data.meta[key]),
                        )
                conn.execute(
                    """
                    SELECT setval(
                      pg_get_serial_sequence('speaker_templates', 'id'),
                      GREATEST(COALESCE(MAX(id), 0), 1),
                      COALESCE(MAX(id), 0) > 0
                    ) FROM speaker_templates
                    """
                )
                conn.execute(
                    """
                    SELECT setval(
                      pg_get_serial_sequence('speaker_audit_log', 'id'),
                      GREATEST(COALESCE(MAX(id), 0), 1),
                      COALESCE(MAX(id), 0) > 0
                    ) FROM speaker_audit_log
                    """
                )
                expected = {
                    "speakers": len(data.speakers),
                    "templates": len(data.templates),
                    "claims": len(data.claims),
                    "audits": len(data.audits),
                }
                actual = _target_counts(conn)
                if actual != expected:
                    raise RuntimeError(
                        "迁移后计数校验失败: "
                        f"expected={expected}, actual={actual}"
                    )
    finally:
        store.close()


def main() -> int:
    args = _parser().parse_args()
    try:
        source_timezone = ZoneInfo(args.source_timezone)
        data = _read_source(
            Path(args.sqlite_path).expanduser().resolve(),
            source_timezone,
        )
        summary = {
            "speakers": len(data.speakers),
            "templates": len(data.templates),
            "claims": len(data.claims),
            "audits": len(data.audits),
            "model_tag": data.model_tag,
        }
        print(f"源库检查通过: {summary}")
        if not args.apply:
            print("仅检查模式：未连接目标库、未写入任何数据")
            return 0
        if not args.postgres_url:
            raise ValueError("--apply 时必须提供 --postgres-url 或 SPEAKER_DATABASE_URL")
        _insert_source(data, args.postgres_url, source_timezone)
        print("迁移完成，源库与目标库数量校验一致")
        return 0
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"迁移失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
