# Shared Voiceprint Database Deployment and Accuracy

[中文](声纹识别共享库部署与准确度方案.md) | **English**

## 1. Architecture

Two ASR services connect to the same PostgreSQL/pgvector instance. Enrollment,
template updates, renaming, deletion, and identification query the database
directly. There is no process-local full-library cache, so changes made through
one service are visible to the other service on its next query.

For a library of a few hundred speakers, identification uses exact pgvector
cosine-distance queries without an HNSW index. This avoids approximate-recall
loss; add an index only after measured latency justifies it.

## 2. Start PostgreSQL/pgvector

The database image is:

```text
registry.cn-hangzhou.aliyuncs.com/ripper/pgvector:pg16
```

Start it from the deployment directory:

```bash
cd docker
POSTGRES_PASSWORD='replace-with-a-strong-password' docker compose \
  -f docker-compose.pgvector.yml up -d
```

The compose file binds port 5432 to `127.0.0.1` by default. If another server
must connect directly, set `POSTGRES_BIND_ADDRESS` to the database server's
private IP and allow only the ASR servers through the firewall.

Production requirements:

- Never commit the database password.
- Do not expose port 5432 to the public Internet.
- Back up the `qwen3_asr` database and the `postgres_data` volume regularly.
- Configure both ASR services with the same `SPEAKER_DATABASE_URL`.
- Keep `speaker_store_audio: false` in multi-instance deployments. If audio
  retention is required, every instance must mount the same shared
  `data/speaker_audio` directory so deletion physically removes the files.

Example connection URL:

```bash
SPEAKER_DATABASE_URL='postgresql://qwen3_asr:strong-password@database-ip:5432/qwen3_asr'
```

For containers on the same host, use an address reachable from the container or
attach the services to the same Docker network. Do not use `127.0.0.1` inside an
ASR container to refer to a database running in another container.

## 3. ASR Configuration

Configure both services with:

```yaml
enable_speaker: true
enable_speaker_db: true
speaker_identify_min_sec: 5.0
speaker_realtime_confirmations: 2
speaker_id_threshold: 0.45
speaker_id_margin: 0.10
speaker_auto_enroll: false
speaker_store_audio: false
```

An `api_key` is still mandatory for the voiceprint management endpoints. Prefer
passing the database URL through `SPEAKER_DATABASE_URL` instead of storing a
password in `config.yaml`.

## 4. Accuracy Guards

- Manual enrollment accepts one template and recommends at least three without
  making them mandatory.
- Real-time identification starts after a cluster accumulates at least 5 seconds
  of effective speech.
- A real-time name is exposed after two consecutive matches to the same identity.
- Real-time cluster centroids are weighted by speech duration.
- Low-confidence segments remain anonymous after the maximum speaker count is reached.
- Offline clusters shorter than 5 seconds remain anonymous.
- Identification is rejected when top1 is below `speaker_id_threshold` or the
  top1-top2 gap is below `speaker_id_margin`.
- Unknown speakers are not auto-enrolled by default.
- Identification audits record duration, top1, top2, margin, and rejection reason
  for later threshold calibration.

## 5. Manual Migration from SQLite

The service never migrates or reads the old SQLite database automatically. The
migration script only validates the source unless `--apply` is explicitly given:

```bash
cd asr-service
.venv/bin/python scripts/migrate_speaker_sqlite_to_postgres.py \
  --sqlite-path data/speakers.db
```

After reviewing the validation result, backing up the source, and confirming the
target is empty, an administrator may run:

```bash
cd asr-service
SPEAKER_DATABASE_URL='postgresql://qwen3_asr:strong-password@database-ip:5432/qwen3_asr' \
  .venv/bin/python scripts/migrate_speaker_sqlite_to_postgres.py \
  --sqlite-path data/speakers.db \
  --apply
```

The script locks the target tables and performs the empty-target check, writes,
and final row-count verification in one transaction. Any failure rolls back the
entire migration. Speaker IDs, template IDs, claim idempotency records, and audit
records are preserved.

## 6. Rollout Order

1. Start pgvector and confirm its health check passes.
2. Back up the old `speakers.db` and run source validation.
3. If old data is needed, schedule a maintenance window and run `--apply` manually.
4. Switch one ASR service and validate enrollment, identification, and deletion.
5. Switch the second service and verify that it sees changes without synchronization.
6. Build same-speaker and different-speaker validation sets before tuning thresholds.
