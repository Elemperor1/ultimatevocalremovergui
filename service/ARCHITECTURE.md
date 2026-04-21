# Async Deployment Pattern (Vercel + GPU Worker)

This service now follows an async job architecture compatible with Vercel + external GPU infra.

## Responsibilities

- **Vercel API layer (`SeparationJobAPI`)**
  - `post_jobs(...)` stores a durable job record in SQLite (can be replaced with Postgres/SQS/Upstash).
  - `get_job(...)` is poll-friendly and returns status/progress/logs.
  - `get_job_artifacts(...)` returns signed artifact URLs with opaque artifact IDs after completion.
  - Signing secret must be provided via constructor or `UVR_URL_SIGNING_SECRET`.

- **Durable queue/store (`DurableJobStore`)**
  - Persists jobs in a `jobs` table.
  - Supports leasing (`claim_next`) for at-least-once worker processing.
  - Tracks heartbeats, completion, and failures for retry-safe behavior.

- **GPU worker (`SeparationWorker`)**
  - Polls queue for leased jobs.
  - Streams input referenced in metadata from storage (`inputs/{job_id}/...`).
  - Preloads common models on startup from `models/...` with checksum-aware cache reuse.
  - Runs source separation on temp local scratch storage.
- Uploads outputs immediately to `outputs/{job_id}/...`, stores artifact metadata, and serves signed URLs from the API.
  - Applies retention cleanup for stale artifacts and cache invalidation by version/hash.

## Production swaps

`DurableJobStore` currently uses SQLite for local durability. For production deploys, swap this for:

- Upstash Redis queue + Postgres metadata table, or
- SQS + Postgres metadata table, or
- pure Postgres job table with `FOR UPDATE SKIP LOCKED`.

The API contract remains the same across those options.

## Worker Docker image

A dedicated worker image is available at `Dockerfile.worker` with:

- Python `3.10.9` (the explicit project Python 3.10 line from manual install docs),
- system binaries for `ffmpeg` and `rubberband`,
- CPU/GPU worker dependency sets that avoid GUI-only packages.

### Build variants

CPU worker:

```bash
docker build -f Dockerfile.worker --build-arg WORKER_VARIANT=cpu -t uvr-worker:cpu .
```

GPU worker:

```bash
docker build -f Dockerfile.worker --build-arg WORKER_VARIANT=gpu -t uvr-worker:gpu .
```

### Runtime checks

The image includes startup and container health checks through `service.worker_healthcheck`:

- verifies required model folders from `UVR_MODEL_PATHS`,
- verifies model availability by checking required model files (`UVR_REQUIRED_MODEL_FILES`) or model artifact patterns (`UVR_MODEL_ARTIFACT_PATTERNS`),
- verifies `ffmpeg` executable availability,
- verifies ONNX Runtime providers and enforces `CUDAExecutionProvider` when `UVR_EXPECT_CUDA=1` is set (otherwise CPU fallback is accepted).

Worker job-store path can be configured with `UVR_JOB_DB_PATH` (default: `data/separation_jobs.sqlite3`), which is suitable for volume mounting.

## Object storage layout

The API/worker contract now assumes portable object-key layout compatible with S3, Cloudflare R2, and GCS:

- `inputs/{job_id}/...` for uploaded source assets
- `models/...` for globally shared model artifacts
- `outputs/{job_id}/...` for per-job result stems

The default implementation (`LocalObjectStorage`) maps these keys onto local disk for development. In production, replace this class with cloud-provider implementations while preserving keys and metadata shape.
