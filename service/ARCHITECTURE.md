# Async Deployment Pattern (Vercel + GPU Worker)

This service now follows an async job architecture compatible with Vercel + external GPU infra.

## Responsibilities

- **Vercel API layer (`SeparationJobAPI`)**
  - `post_jobs(...)` stores a durable job record in SQLite (can be replaced with Postgres/SQS/Upstash).
  - `get_job(...)` is poll-friendly and returns status/progress/logs.
  - `get_job_artifacts(...)` returns signed artifact URLs after completion.

- **Durable queue/store (`DurableJobStore`)**
  - Persists jobs in a `jobs` table.
  - Supports leasing (`claim_next`) for at-least-once worker processing.
  - Tracks heartbeats, completion, and failures for retry-safe behavior.

- **GPU worker (`SeparationWorker`)**
  - Polls queue for leased jobs.
  - Downloads/reads input referenced in metadata.
  - Runs source separation.
  - Writes artifact paths back to durable store.

## Production swaps

`DurableJobStore` currently uses SQLite for local durability. For production deploys, swap this for:

- Upstash Redis queue + Postgres metadata table, or
- SQS + Postgres metadata table, or
- pure Postgres job table with `FOR UPDATE SKIP LOCKED`.

The API contract remains the same across those options.
