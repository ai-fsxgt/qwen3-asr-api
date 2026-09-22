#!/usr/bin/env python3
"""从有效原始音频重建 ReDimNet2 声纹并迁移到全新 PostgreSQL。

脚本只读取旧声纹库、Poco 业务库和 S3。只有显式传入 ``--apply``
才会初始化目标库并插入数据；目标库已有任意声纹业务数据时拒绝执行。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import psycopg
import soundfile as sf
import torch
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import MODEL_LOCAL_MAP  # noqa: E402
from app.engines.speaker_embedding_engine import (  # noqa: E402
    SpeakerEmbeddingEngine,
    make_identity_windows,
)
from app.pipeline.audio_preprocessor import convert_to_wav  # noqa: E402
from app.runtime.energy_vad import EnergyVAD  # noqa: E402
from app.runtime.speaker_store import SpeakerStore  # noqa: E402


@dataclass(frozen=True)
class SourceSpeaker:
    id: str
    name: str
    note: str | None
    consent: bool
    source: str
    created_at: dt.datetime
    updated_at: dt.datetime


@dataclass(frozen=True)
class SourceTemplate:
    id: int
    speaker_id: str
    created_at: dt.datetime


@dataclass(frozen=True)
class MigrationSample:
    speaker_id: str
    speaker_name: str
    template_id: int
    object_key: str
    storage_index: int


@dataclass(frozen=True)
class SourceState:
    meta: dict[str, str]
    speakers: dict[str, SourceSpeaker]
    templates: dict[int, SourceTemplate]
    claims: list[dict]
    audits: list[dict]


@dataclass(frozen=True)
class PreparedTemplate:
    source: SourceTemplate
    vector: np.ndarray
    duration_sec: float


class S3Client:
    """满足本次迁移所需的最小 S3 SigV4 客户端。"""

    _XML_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        region: str,
        bucket: str,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.bucket = bucket
        parsed = urllib.parse.urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("S3 endpoint 必须是有效的 HTTP(S) 地址")
        self._host = parsed.netloc
        self._client = httpx.Client(timeout=60.0)

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _sign(key: bytes, message: str) -> bytes:
        return hmac.new(key, message.encode(), hashlib.sha256).digest()

    def _headers(self, method: str, path: str, query: str) -> dict[str, str]:
        now = dt.datetime.now(dt.timezone.utc)
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")
        date = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(b"").hexdigest()
        canonical_headers = (
            f"host:{self._host}\n"
            f"x-amz-content-sha256:{payload_hash}\n"
            f"x-amz-date:{timestamp}\n"
        )
        signed_headers = "host;x-amz-content-sha256;x-amz-date"
        canonical_request = "\n".join(
            [
                method,
                path,
                query,
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )
        scope = f"{date}/{self.region}/s3/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                timestamp,
                scope,
                hashlib.sha256(canonical_request.encode()).hexdigest(),
            ]
        )
        date_key = self._sign(
            ("AWS4" + self.secret_key).encode(),
            date,
        )
        region_key = self._sign(date_key, self.region)
        service_key = self._sign(region_key, "s3")
        signing_key = self._sign(service_key, "aws4_request")
        signature = hmac.new(
            signing_key,
            string_to_sign.encode(),
            hashlib.sha256,
        ).hexdigest()
        authorization = (
            "AWS4-HMAC-SHA256 "
            f"Credential={self.access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        return {
            "Authorization": authorization,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": timestamp,
        }

    @staticmethod
    def _query(params: dict[str, str]) -> str:
        return urllib.parse.urlencode(
            sorted(params.items()),
            quote_via=urllib.parse.quote,
            safe="~",
        )

    def list_prefix(self, prefix: str) -> dict[str, int]:
        objects: dict[str, int] = {}
        continuation_token: str | None = None
        while True:
            params = {
                "list-type": "2",
                "max-keys": "1000",
                "prefix": prefix,
            }
            if continuation_token:
                params["continuation-token"] = continuation_token
            query = self._query(params)
            path = f"/{self.bucket}"
            response = self._client.get(
                f"{self.endpoint}{path}?{query}",
                headers=self._headers("GET", path, query),
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall("s3:Contents", self._XML_NAMESPACE):
                key = item.findtext("s3:Key", namespaces=self._XML_NAMESPACE)
                size = item.findtext("s3:Size", namespaces=self._XML_NAMESPACE)
                if key is not None and size is not None:
                    objects[key] = int(size)
            truncated = root.findtext(
                "s3:IsTruncated",
                default="false",
                namespaces=self._XML_NAMESPACE,
            )
            if truncated != "true":
                return objects
            continuation_token = root.findtext(
                "s3:NextContinuationToken",
                namespaces=self._XML_NAMESPACE,
            )
            if not continuation_token:
                raise RuntimeError("S3 分页响应缺少 continuation token")

    def download(self, key: str, destination: Path, expected_size: int) -> None:
        if destination.exists():
            actual_size = destination.stat().st_size
            if actual_size != expected_size:
                raise ValueError(
                    f"缓存文件大小不一致，拒绝覆盖: {destination} "
                    f"({actual_size} != {expected_size})"
                )
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        path = urllib.parse.quote(
            f"/{self.bucket}/{key}",
            safe="/-_.~",
        )
        with self._client.stream(
            "GET",
            f"{self.endpoint}{path}",
            headers=self._headers("GET", path, ""),
        ) as response:
            response.raise_for_status()
            with destination.open("xb") as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
        actual_size = destination.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"S3 下载文件大小不一致: {key} "
                f"({actual_size} != {expected_size})"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 S3 有效原始音频重建 ReDimNet2 声纹"
    )
    parser.add_argument(
        "--source-postgres-url",
        default=os.environ.get("SOURCE_SPEAKER_DATABASE_URL"),
    )
    parser.add_argument(
        "--poco-postgres-url",
        action="append",
        default=[],
        help="Poco 业务库连接地址；可重复提供补充业务库",
    )
    parser.add_argument(
        "--target-postgres-url",
        default=os.environ.get("SPEAKER_DATABASE_URL"),
    )
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-access-key", required=True)
    parser.add_argument("--s3-secret-key", required=True)
    parser.add_argument("--s3-region", default="us-east-1")
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--supplemental-s3-endpoint")
    parser.add_argument("--supplemental-s3-access-key")
    parser.add_argument("--supplemental-s3-secret-key")
    parser.add_argument("--supplemental-s3-region", default="us-east-1")
    parser.add_argument("--supplemental-s3-bucket")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--min-speech-sec", type=float, default=3.0)
    parser.add_argument("--energy-floor-dbfs", type=float, default=-45.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "实际生成向量并写入目标库；"
            "不传时只执行完整只读检查"
        ),
    )
    return parser


def _connect_read_only(database_url: str) -> psycopg.Connection:
    connection = psycopg.connect(
        database_url,
        connect_timeout=10,
        row_factory=dict_row,
        autocommit=True,
    )
    connection.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
    return connection


def _read_source(database_url: str) -> SourceState:
    with _connect_read_only(database_url) as connection:
        meta = {
            row["key"]: row["value"]
            for row in connection.execute(
                "SELECT key, value FROM speaker_meta"
            )
        }
        speakers = {
            row["id"]: SourceSpeaker(
                id=row["id"],
                name=row["name"],
                note=row["note"],
                consent=bool(row["consent"]),
                source=row["source"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in connection.execute(
                """
                SELECT id, name, note, consent, source, created_at, updated_at
                FROM speakers ORDER BY id
                """
            )
        }
        templates = {
            int(row["id"]): SourceTemplate(
                id=int(row["id"]),
                speaker_id=row["speaker_id"],
                created_at=row["created_at"],
            )
            for row in connection.execute(
                """
                SELECT id, speaker_id, created_at
                FROM speaker_templates ORDER BY id
                """
            )
        }
        claims = list(
            connection.execute(
                """
                SELECT claim_key, speaker_id, response, created_at
                FROM speaker_claims ORDER BY claim_key
                """
            )
        )
        audits = list(
            connection.execute(
                """
                SELECT id, ts, action, speaker_id, detail
                FROM speaker_audit_log ORDER BY id
                """
            )
        )
    if not speakers or not templates:
        raise ValueError("旧声纹库没有可迁移的说话人或模板")
    for template in templates.values():
        if template.speaker_id not in speakers:
            raise ValueError(f"旧模板 {template.id} 引用了不存在的说话人")
    return SourceState(
        meta=meta,
        speakers=speakers,
        templates=templates,
        claims=claims,
        audits=audits,
    )


def _read_business_profiles(database_url: str) -> dict[str, dict]:
    with _connect_read_only(database_url) as connection:
        rows = connection.execute(
            """
            SELECT p.id::text AS profile_id, p.name, b.target_type,
                   b.remote_speaker_id, s.id::text AS sample_id,
                   s.object_key, s.remote_template_ids
            FROM ai_note_speaker_profiles p
            JOIN ai_note_speaker_bindings b ON b.profile_id = p.id
            JOIN ai_note_speaker_samples s ON s.profile_id = p.id
            WHERE b.remote_speaker_id IS NOT NULL
            ORDER BY p.id, s.created_at, b.target_type
            """
        ).fetchall()
    profiles: dict[str, dict] = {}
    for row in rows:
        profile = profiles.setdefault(
            row["profile_id"],
            {
                "name": row["name"],
                "bindings": {},
                "samples": {},
            },
        )
        if profile["name"] != row["name"]:
            raise ValueError(f"档案名称不一致: {row['profile_id']}")
        profile["bindings"][row["target_type"]] = row["remote_speaker_id"]
        profile["samples"][row["sample_id"]] = {
            "object_key": row["object_key"],
            "template_ids": row["remote_template_ids"] or {},
        }
    return profiles


def _build_manifest(
    source: SourceState,
    poco_urls: list[str],
) -> list[MigrationSample]:
    mapped_speakers: set[str] = set()
    used_templates: set[int] = set()
    used_objects: set[tuple[int, str]] = set()
    manifest: list[MigrationSample] = []
    template_owner = {
        template.id: template.speaker_id
        for template in source.templates.values()
    }

    for storage_index, database_url in enumerate(poco_urls):
        for profile_id, profile in _read_business_profiles(database_url).items():
            candidates = {
                speaker_id
                for speaker_id in profile["bindings"].values()
                if speaker_id in source.speakers
            }
            if not candidates:
                continue
            if len(candidates) != 1:
                raise ValueError(
                    f"档案 {profile_id} 同时映射到多个旧库说话人: "
                    f"{sorted(candidates)}"
                )
            speaker_id = candidates.pop()
            if speaker_id in mapped_speakers:
                continue
            speaker = source.speakers[speaker_id]
            if profile["name"] != speaker.name:
                raise ValueError(
                    "档案与旧库名称不一致: "
                    f"{profile['name']} != {speaker.name}"
                )
            current_samples: list[MigrationSample] = []
            for sample in profile["samples"].values():
                template_ids = {
                    int(template_id)
                    for template_id in sample["template_ids"].values()
                    if isinstance(template_id, int)
                    and not isinstance(template_id, bool)
                    and template_owner.get(int(template_id)) == speaker_id
                }
                if len(template_ids) != 1:
                    raise ValueError(
                        f"样本无法唯一映射旧模板: {sample['object_key']} "
                        f"-> {sorted(template_ids)}"
                    )
                template_id = template_ids.pop()
                object_key = sample["object_key"]
                object_identity = (storage_index, object_key)
                if template_id in used_templates:
                    raise ValueError(f"模板被多个样本引用: {template_id}")
                if object_identity in used_objects:
                    raise ValueError(f"S3 对象被多个模板引用: {object_key}")
                current_samples.append(
                    MigrationSample(
                        speaker_id=speaker_id,
                        speaker_name=speaker.name,
                        template_id=template_id,
                        object_key=object_key,
                        storage_index=storage_index,
                    )
                )
                used_templates.add(template_id)
                used_objects.add(object_identity)
            expected_count = sum(
                template.speaker_id == speaker_id
                for template in source.templates.values()
            )
            if len(current_samples) != expected_count:
                raise ValueError(
                    f"{speaker.name} 的有效样本数与旧模板数不一致: "
                    f"{len(current_samples)} != {expected_count}"
                )
            manifest.extend(current_samples)
            mapped_speakers.add(speaker_id)

    missing_speakers = set(source.speakers) - mapped_speakers
    missing_templates = set(source.templates) - used_templates
    if missing_speakers or missing_templates:
        names = [
            source.speakers[speaker_id].name
            for speaker_id in sorted(missing_speakers)
        ]
        raise ValueError(
            "迁移清单覆盖不完整: "
            f"missing_speakers={names}, "
            f"missing_templates={sorted(missing_templates)}"
        )
    return sorted(manifest, key=lambda item: item.template_id)


def _object_inventory(
    clients: list[S3Client],
    manifest: list[MigrationSample],
) -> dict[tuple[int, str], int]:
    inventory: dict[tuple[int, str], int] = {}
    for storage_index, client in enumerate(clients):
        prefixes = {
            item.object_key.split("/", 1)[0] + "/"
            for item in manifest
            if item.storage_index == storage_index
        }
        for prefix in sorted(prefixes):
            inventory.update({
                (storage_index, key): size
                for key, size in client.list_prefix(prefix).items()
            })
    missing = [
        item.object_key
        for item in manifest
        if (item.storage_index, item.object_key) not in inventory
    ]
    empty = [
        item.object_key
        for item in manifest
        if inventory.get((item.storage_index, item.object_key), 0) <= 0
    ]
    if missing or empty:
        raise ValueError(
            f"S3 原始音频不完整: missing={missing}, empty={empty}"
        )
    return inventory


def _target_counts(connection: psycopg.Connection) -> dict[str, int]:
    tables = {
        "speakers": "speakers",
        "templates": "speaker_templates",
        "claims": "speaker_claims",
        "audits": "speaker_audit_log",
    }
    return {
        name: int(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        )
        for name, table in tables.items()
    }


def _check_target_empty(database_url: str) -> None:
    with _connect_read_only(database_url) as connection:
        existing_tables = {
            row["table_name"]
            for row in connection.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN (
                    'speakers', 'speaker_templates', 'speaker_claims',
                    'speaker_audit_log', 'speaker_meta'
                  )
                """
            )
        }
        if not existing_tables:
            return
        counts = {}
        for table in sorted(existing_tables - {"speaker_meta"}):
            counts[table] = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            )
        if any(counts.values()):
            raise ValueError(f"目标库不是空库，拒绝迁移: {counts}")
        if "speaker_meta" in existing_tables:
            row = connection.execute(
                "SELECT value FROM speaker_meta WHERE key = 'model_tag'"
            ).fetchone()
            if row and row["value"] != SpeakerEmbeddingEngine.MODEL_TAG:
                raise ValueError(
                    "目标库 model_tag 不一致，拒绝迁移: "
                    f"{row['value']}"
                )


def _cache_paths(cache_dir: Path, sample: MigrationSample) -> tuple[Path, Path]:
    suffix = Path(sample.object_key).suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".audio"
    raw_base = cache_dir / "raw" / sample.speaker_id / str(sample.template_id)
    wav_base = cache_dir / "wav" / sample.speaker_id / str(sample.template_id)
    return raw_base.with_suffix(suffix), wav_base.with_suffix(".wav")


def _embed_audio(
    engine: SpeakerEmbeddingEngine,
    vad: EnergyVAD,
    source_path: Path,
    wav_path: Path,
    *,
    min_speech_sec: float,
) -> tuple[np.ndarray, float]:
    if not wav_path.exists():
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        convert_to_wav(str(source_path), str(wav_path))
    segments = vad.detect(str(wav_path))
    duration_sec = sum(end - start for start, end in segments) / 1000.0
    if duration_sec < min_speech_sec:
        raise ValueError(
            f"有效语音不足: {source_path} "
            f"({duration_sec:.2f}s < {min_speech_sec:.2f}s)"
        )
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if sample_rate != engine.SAMPLE_RATE:
        raise ValueError(f"WAV 采样率异常: {wav_path} ({sample_rate})")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    speech_chunks = [
        waveform[
            max(int(start * sample_rate / 1000), 0):
            min(int(end * sample_rate / 1000), len(waveform))
        ]
        for start, end in segments
    ]
    speech = np.ascontiguousarray(np.concatenate(speech_chunks))
    identity_windows = make_identity_windows(
        0.0,
        len(speech) / sample_rate,
    )
    if not identity_windows:
        identity_windows = [(0.0, len(speech) / sample_rate)]
    embeddings = engine.embed_windows(speech, identity_windows)
    vector = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"声纹向量无法归一化: {source_path}")
    return (vector / norm).astype(np.float32), duration_sec


def _prepare_templates(
    args: argparse.Namespace,
    source: SourceState,
    manifest: list[MigrationSample],
    inventory: dict[tuple[int, str], int],
    s3_clients: list[S3Client],
) -> dict[int, PreparedTemplate]:
    if args.model_file.name != SpeakerEmbeddingEngine._MODEL_FILE:
        raise ValueError(
            "模型文件名不正确: "
            f"{args.model_file.name} != {SpeakerEmbeddingEngine._MODEL_FILE}"
        )
    MODEL_LOCAL_MAP["redimnet2_b6"] = str(args.model_file.resolve().parent)
    torch.set_num_threads(args.torch_threads)
    engine = SpeakerEmbeddingEngine(device="cpu")
    engine.load()
    vad = EnergyVAD(energy_floor_dbfs=args.energy_floor_dbfs)
    prepared: dict[int, PreparedTemplate] = {}
    try:
        for index, sample in enumerate(manifest, start=1):
            raw_path, wav_path = _cache_paths(args.cache_dir, sample)
            s3_clients[sample.storage_index].download(
                sample.object_key,
                raw_path,
                inventory[(sample.storage_index, sample.object_key)],
            )
            vector, duration_sec = _embed_audio(
                engine,
                vad,
                raw_path,
                wav_path,
                min_speech_sec=args.min_speech_sec,
            )
            prepared[sample.template_id] = PreparedTemplate(
                source=source.templates[sample.template_id],
                vector=vector,
                duration_sec=duration_sec,
            )
            print(
                f"[{index}/{len(manifest)}] "
                f"{sample.speaker_name} / template {sample.template_id} "
                f"({duration_sec:.2f}s)"
            )
    finally:
        engine.unload()
    if set(prepared) != set(source.templates):
        raise RuntimeError("向量生成结果未覆盖全部旧模板")
    return prepared


def _normalize_mean(vectors: list[np.ndarray]) -> np.ndarray:
    centroid = np.stack(vectors).mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("说话人质心无法归一化")
    return (centroid / norm).astype(np.float32)


def _source_core_changed(
    before: SourceState,
    after: SourceState,
) -> bool:
    """忽略持续增长的识别审计，只比较会影响声纹识别的数据。"""
    return any(
        (
            before.meta != after.meta,
            before.speakers != after.speakers,
            before.templates != after.templates,
            before.claims != after.claims,
        )
    )


def _insert_target(
    database_url: str,
    source: SourceState,
    prepared: dict[int, PreparedTemplate],
) -> None:
    store = SpeakerStore(
        database_url,
        model_tag=SpeakerEmbeddingEngine.MODEL_TAG,
    )
    try:
        if not store.check_model_tag(SpeakerEmbeddingEngine.MODEL_TAG):
            raise ValueError("目标库 model_tag 与 ReDimNet2 不一致")
        by_speaker: dict[str, list[PreparedTemplate]] = defaultdict(list)
        for template in prepared.values():
            by_speaker[template.source.speaker_id].append(template)
        with psycopg.connect(database_url, autocommit=True) as connection:
            register_vector(connection)
            with connection.transaction():
                connection.execute(
                    """
                    LOCK TABLE speakers, speaker_templates, speaker_claims,
                      speaker_audit_log, speaker_meta
                    IN ACCESS EXCLUSIVE MODE
                    """
                )
                counts = _target_counts(connection)
                if any(counts.values()):
                    raise ValueError(f"目标库不是空库，拒绝迁移: {counts}")
                for speaker in source.speakers.values():
                    templates = by_speaker[speaker.id]
                    centroid = _normalize_mean(
                        [template.vector for template in templates]
                    )
                    connection.execute(
                        """
                        INSERT INTO speakers(
                          id, name, note, consent, source, model_tag,
                          centroid, created_at, updated_at
                        ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            speaker.id,
                            speaker.name,
                            speaker.note,
                            speaker.consent,
                            speaker.source,
                            SpeakerEmbeddingEngine.MODEL_TAG,
                            centroid,
                            speaker.created_at,
                            speaker.updated_at,
                        ),
                    )
                for template_id in sorted(prepared):
                    template = prepared[template_id]
                    connection.execute(
                        """
                        INSERT INTO speaker_templates(
                          id, speaker_id, vector, dur_sec, created_at
                        ) VALUES(%s, %s, %s, %s, %s)
                        """,
                        (
                            template.source.id,
                            template.source.speaker_id,
                            template.vector,
                            template.duration_sec,
                            template.source.created_at,
                        ),
                    )
                for claim in source.claims:
                    connection.execute(
                        """
                        INSERT INTO speaker_claims(
                          claim_key, speaker_id, response, created_at
                        ) VALUES(%s, %s, %s, %s)
                        """,
                        (
                            claim["claim_key"],
                            claim["speaker_id"],
                            Jsonb(claim["response"]),
                            claim["created_at"],
                        ),
                    )
                for audit in source.audits:
                    connection.execute(
                        """
                        INSERT INTO speaker_audit_log(
                          id, ts, action, speaker_id, detail
                        ) VALUES(%s, %s, %s, %s, %s)
                        """,
                        (
                            audit["id"],
                            audit["ts"],
                            audit["action"],
                            audit["speaker_id"],
                            (
                                Jsonb(audit["detail"])
                                if audit["detail"] is not None
                                else None
                            ),
                        ),
                    )
                auto_name_seq = source.meta.get("auto_name_seq")
                if auto_name_seq is not None:
                    connection.execute(
                        """
                        INSERT INTO speaker_meta(key, value) VALUES(%s, %s)
                        ON CONFLICT (key) DO NOTHING
                        """,
                        ("auto_name_seq", auto_name_seq),
                    )
                connection.execute(
                    """
                    SELECT setval(
                      pg_get_serial_sequence('speaker_templates', 'id'),
                      GREATEST(COALESCE(MAX(id), 0), 1),
                      COALESCE(MAX(id), 0) > 0
                    ) FROM speaker_templates
                    """
                )
                connection.execute(
                    """
                    SELECT setval(
                      pg_get_serial_sequence('speaker_audit_log', 'id'),
                      GREATEST(COALESCE(MAX(id), 0), 1),
                      COALESCE(MAX(id), 0) > 0
                    ) FROM speaker_audit_log
                    """
                )
                expected = {
                    "speakers": len(source.speakers),
                    "templates": len(source.templates),
                    "claims": len(source.claims),
                    "audits": len(source.audits),
                }
                actual = _target_counts(connection)
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
        if not args.source_postgres_url:
            raise ValueError("缺少 --source-postgres-url")
        if not args.poco_postgres_url:
            raise ValueError("至少需要一个 --poco-postgres-url")
        if not args.target_postgres_url:
            raise ValueError("缺少 --target-postgres-url")
        if args.min_speech_sec <= 0:
            raise ValueError("--min-speech-sec 必须大于 0")
        if args.torch_threads < 1:
            raise ValueError("--torch-threads 必须大于 0")

        source = _read_source(args.source_postgres_url)
        manifest = _build_manifest(source, args.poco_postgres_url)
        _check_target_empty(args.target_postgres_url)
        s3_clients = [S3Client(
            endpoint=args.s3_endpoint,
            access_key=args.s3_access_key,
            secret_key=args.s3_secret_key,
            region=args.s3_region,
            bucket=args.s3_bucket,
        )]
        supplemental_s3_values = (
            args.supplemental_s3_endpoint,
            args.supplemental_s3_access_key,
            args.supplemental_s3_secret_key,
            args.supplemental_s3_bucket,
        )
        if any(supplemental_s3_values):
            if not all(supplemental_s3_values):
                raise ValueError("补充 S3 配置必须完整提供")
            s3_clients.append(
                S3Client(
                    endpoint=args.supplemental_s3_endpoint,
                    access_key=args.supplemental_s3_access_key,
                    secret_key=args.supplemental_s3_secret_key,
                    region=args.supplemental_s3_region,
                    bucket=args.supplemental_s3_bucket,
                )
            )
        required_storage_count = max(
            item.storage_index for item in manifest
        ) + 1
        if len(s3_clients) < required_storage_count:
            raise ValueError("业务库数量超过已配置的 S3 数据源数量")
        try:
            inventory = _object_inventory(s3_clients, manifest)
            summary = {
                "speakers": len(source.speakers),
                "templates": len(source.templates),
                "claims": len(source.claims),
                "audits": len(source.audits),
                "s3_objects": len(manifest),
                "target_model_tag": SpeakerEmbeddingEngine.MODEL_TAG,
            }
            print(f"迁移前检查通过: {json.dumps(summary, ensure_ascii=False)}")
            if not args.apply:
                print(
                    "只读检查模式：未下载音频、未生成向量、"
                    "未写入目标库"
                )
                return 0
            prepared = _prepare_templates(
                args,
                source,
                manifest,
                inventory,
                s3_clients,
            )
        finally:
            for s3_client in s3_clients:
                s3_client.close()
        latest_source = _read_source(args.source_postgres_url)
        latest_manifest = _build_manifest(
            latest_source,
            args.poco_postgres_url,
        )
        if _source_core_changed(source, latest_source) or latest_manifest != manifest:
            raise RuntimeError(
                "向量生成期间源声纹或有效样本发生变化，"
                "拒绝写入目标库，请重试"
            )
        _insert_target(args.target_postgres_url, source, prepared)
        print("迁移完成：全部有效原始音频已重建为 ReDimNet2 向量")
        return 0
    except (
        httpx.HTTPError,
        psycopg.Error,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"迁移失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
