# Speaker Management (v2)

[← API v2 overview](../v2_EN.md) ｜ [中文](speakers.md) | **English**

How speaker diarization (anonymous labels) and voiceprint identification (real names) work, plus the voiceprint database `/v2/speakers*` enrollment / management / identification endpoints.

## Table of Contents

- [Speaker Diarization & Voiceprint Identification](#speaker-diarization--voiceprint-identification)
- [Voiceprint Database Endpoints `/v2/speakers*`](#voiceprint-database-endpoints)
  - [Enroll a Speaker](#enroll-a-speaker)
  - [Claim an Auto-enrolled Speaker](#claim-an-auto-enrolled-speaker)
  - [List / Detail / Rename & Note / Delete](#list--detail--rename--note--delete)
  - [Template Management / Identify](#template-management--identify)
  - [Error Codes](#error-codes)

---

## Speaker Diarization & Voiceprint Identification

Two layers of capability, both disabled by default and enabled on demand (configuration in the [configuration reference](../../configuration_EN.md)):

| Layer | Switch | Output |
|-------|--------|--------|
| **Speaker diarization** (anonymous) | `enable_speaker` | Offline `segments[].speaker` and top-level `speakers` — labels `A`/`B`/`C`… ordered by first time speaking, **scoped to a single file** (same person is not guaranteed to keep the same label across tasks) |
| **Voiceprint identification** (real name) | `enable_speaker_db` (depends on the layer above + `api_key` must be configured) | With `identify_speakers=true`, speakers are matched against the shared PostgreSQL/pgvector database and matches emit `speaker_name`. `speaker_auto_enroll` is disabled by default; when explicitly enabled, only unmatched offline speakers with enough speech (default ≥10s) are enrolled as `说话人_NN` |

Key points:

- **Failures always degrade gracefully**: a failure in any diarization/identification step only drops the label/real name; the transcription result is unaffected.
- **The realtime path only transcribes text**: it does not run diarization or voiceprint identification. For existing clients, `final.speaker` and `final.speaker_name` are always `null`.
- `speakers[].speaker_id` is a **pure value snapshot**: it has no association with the task store; after a speaker is deleted, the id in historical tasks dangles (`GET /v2/speakers/{id}` returning 404 means "deleted"), and callers must tolerate this.
- Consent semantics of auto-enrollment: enabling `speaker_auto_enroll` means the deployer declares that consent for voiceprint enrollment has been obtained from the data subject (the same liability attribution as manual enrollment with `consent=true`).

See [Transcription · Submit ASR Task](transcription_EN.md#submit-asr-task) for the offline `identify_speakers` request field and [Task Management · Result Structure](tasks_EN.md#result-structure) for the offline speaker fields.

Offline result example with voiceprint identification (the incremental part of `result`):

```json
{
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "大家好。", "speaker": "A", "speaker_name": "张三"},
    {"start": 3.5, "end": 6.0, "text": "我先说两句。", "speaker": "B", "speaker_name": "说话人_07"}
  ],
  "speakers": [
    {"label": "A", "speaker_id": "9f86d081884c7d659a2feaa0c55ad015", "name": "张三", "score": 0.62},
    {"label": "B", "speaker_id": "3c2a91f0a1b24e83b6f1c2d3e4f5a6b7", "name": "说话人_07", "score": null, "auto_enrolled": true},
    {"label": "C", "speaker_id": null, "name": null, "score": null}
  ]
}
```

## Voiceprint Database Endpoints

```
/v2/speakers* (v2 only; all endpoints enforce Bearer authentication — when the server has no api_key configured, the voiceprint database is entirely unavailable)
```

> Browser management page: start with `--web` and open `/web-ui/speakers` (list / rename & note / delete).

Voiceprint data is **never auto-cleaned** (unlike the task store's 7-day TTL); the only way to delete is via the DELETE endpoint.
After the DELETE transaction commits, the speaker and associated voiceprint
templates are immediately absent from application queries. PostgreSQL handles
dead tuples and storage reuse through autovacuum and operational maintenance;
WAL and backups follow their respective retention policies.

### Enroll a Speaker

```
POST /v2/speakers        (multipart) → 201
```

```bash
curl -X POST http://127.0.0.1:8765/v2/speakers \
  -H "Authorization: Bearer sk-your-key" \
  -F "name=张三" -F "consent=true" -F "note=产品部" \
  -F "files=@sample1.wav" -F "files=@sample2.wav" -F "files=@sample3.wav"
```

| Parameter | Description |
|-----------|-------------|
| name | Display name (required) |
| consent | Must be `true`: confirms that consent has been obtained from the data subject (otherwise 400) |
| note | Note (optional) |
| files | ≥1 clear **single-speaker** audio sample; each sample must have ≥3s of effective speech after VAD (configurable via `speaker_enroll_min_sec`); samples in which multiple speakers are detected are rejected with 400; ≥3 samples from different scenes are recommended (insufficient samples only return a `quality_hint`, without blocking) |

Response: `{"speaker_id": "9f86…", "name": "张三", "templates": 3, "quality_hint": null}`

### Claim an Auto-enrolled Speaker

```
POST /v2/speakers/{id}/claim        (multipart) → 200
```

When an administrator's sample matches a `source="auto"` placeholder such as `说话人_NN`, the caller can claim that record. Claiming preserves the original `speaker_id` and, in one database transaction, changes the source to `manual`, updates the name and note, and replaces all auto-generated templates with the administrator-provided samples.

```bash
curl -X POST http://127.0.0.1:8765/v2/speakers/9f86.../claim \
  -H "Authorization: Bearer sk-your-key" \
  -F "name=Alice" -F "consent=true" -F "note=Product" \
  -F "claim_key=0f34c41a-1db8-4cc7-91e5-086bf4f73f86" \
  -F "files=@sample1.wav" -F "files=@sample2.wav"
```

| Parameter | Description |
|-----------|-------------|
| name | Display name after the claim (required) |
| consent | Must be `true` |
| note | Optional note |
| claim_key | Caller-provided idempotency key; using the caller's voiceprint profile UUID is recommended; max 128 characters |
| files | 1–16 single-speaker audio samples, with the same quality requirements as enrollment |

The `template_ids` in a successful response follow the upload order of `files`, so callers can map samples directly:

```json
{
  "speaker_id": "9f86d081884c7d659a2feaa0c55ad015",
  "name": "Alice",
  "source": "manual",
  "templates": 2,
  "template_ids": [12, 13],
  "quality_hint": "建议提供 ≥3 个不同场景的样本以提升识别稳健性"
}
```

- Retrying the same `speaker_id` and `claim_key` returns the original successful result without replacing templates again.
- A `claim_key` already used for another speaker, or a target whose source is not `auto`, returns a structured `409`; `detail.conflict` contains the conflicting `speaker_id`, `speaker_name`, and `source`.
- Claimability is determined only by `source`, which the ordinary rename endpoint cannot change, never by matching the placeholder name.

### List / Detail / Rename & Note / Delete

```
GET    /v2/speakers                 → {"total": N, "speakers": [{id,name,note,source,template_count,created_at,updated_at}]}
GET    /v2/speakers/{id}            → detail (includes template summary templates: [{id,dur_sec,created_at}], without feature vectors)
PATCH  /v2/speakers/{id}            → body {"name"?, "note"?}; renaming does not affect speaker_id or templates, and subsequent transcriptions show the new name immediately
DELETE /v2/speakers/{id}            → deletes the database row and cascaded templates, and cleans up retained audio; it cannot be restored through the API, and subsequent transcriptions fall back to anonymous
```

- The `source` field distinguishes `manual` (manually enrolled) / `auto` (auto-enrolled), which helps audit placeholder-name entries.

### Template Management / Identify

```
POST   /v2/speakers/{id}/templates          (multipart file) → 201, appends a sample and recomputes the centroid (max 16 per speaker)
DELETE /v2/speakers/{id}/templates/{tid}    → {"remaining": N, "hint"?} (reaching 0 templates does not auto-delete the speaker; hint suggests adding samples or deleting)
POST   /v2/speakers/identify                (multipart file) → {"matched": bool, "speaker_id"?, "name"?, "source"?, "score"?, "top2_score"?, "margin"?, "reason"?}
```

Identification is a 1:N open-set decision: it returns `matched: false` when the highest similarity is below the threshold (`speaker_id_threshold`, default 0.45) or when the margin to the second-highest is too small (`speaker_id_margin`, default 0.10). `reason` is `matched`, `empty_library`, `below_threshold`, or `ambiguous_margin`; `score`, `top2_score`, and `margin` support threshold calibration. On a match, `source` is `manual` or `auto`.

### Error Codes

| Status Code | Meaning |
|-------------|---------|
| 400 | Quality threshold not met (insufficient duration / multiple speakers / unsupported format) / missing consent |
| 401 | Authentication failed |
| 404 | Speaker / template not found |
| 409 | The target is not claimable (not `auto`) or the `claim_key` belongs to another speaker; the response includes a structured conflict object |
| 503 | `speaker_db_disabled` (module not enabled / degraded) / `model_tag_mismatch` (templates in the database are inconsistent with the current engine version: enrollment, claiming, and identification are disabled, **viewing and deletion remain available**) |
| 500 | Voiceprint database read/write failure |
