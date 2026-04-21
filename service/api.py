from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep, time
from typing import Any, Callable
from uuid import uuid4

from separate import run_separation


JobStatus = str


@dataclass
class SeparationJobRequest:
    input_audio: str
    output_dir: str
    audio_file_base: str
    model_data: Any
    settings: dict[str, Any] = field(default_factory=dict)
    cached_source_callback: Callable[..., Any] | None = None
    cached_model_source_holder: dict[str, Any] | None = None
    list_all_models: list[Any] = field(default_factory=list)


@dataclass
class SeparationJob:
    id: str
    request: SeparationJobRequest
    status: JobStatus = "queued"
    progress: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


class DurableJobStore:
    """SQLite-backed store that survives Vercel cold starts/retries."""

    def __init__(self, db_path: str | Path = "separation_jobs.sqlite3") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    request_json TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    logs_json TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_until REAL
                )
                """
            )

    def enqueue(self, payload: SeparationJobRequest) -> SeparationJob:
        now = time()
        job = SeparationJob(id=str(uuid4()), request=payload, created_at=now, updated_at=now)
        serialized = self._serialize_request(payload)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, status, progress, request_json, artifacts_json, logs_json, error, created_at, updated_at, lease_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (job.id, "queued", 0.0, serialized, "[]", "[]", None, now, now),
            )
        return job

    def get(self, job_id: str) -> SeparationJob:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

        if row is None:
            raise KeyError(f"unknown job: {job_id}")

        return self._deserialize_job(row)

    def claim_next(self, lease_seconds: int = 120) -> SeparationJob | None:
        now = time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued' OR (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?)
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None

            lease_until = now + lease_seconds
            conn.execute(
                "UPDATE jobs SET status = 'running', updated_at = ?, lease_until = ? WHERE id = ?",
                (now, lease_until, row["id"]),
            )

        claimed = dict(row)
        claimed["status"] = "running"
        claimed["updated_at"] = now
        claimed["lease_until"] = lease_until
        return self._deserialize_job(claimed)

    def heartbeat(self, job_id: str, progress: float, logs: list[str]) -> None:
        now = time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress = ?, logs_json = ?, updated_at = ?, lease_until = ? WHERE id = ?",
                (max(0.0, min(1.0, progress)), json.dumps(logs), now, now + 120, job_id),
            )

    def complete(self, job_id: str, artifacts: list[str]) -> None:
        now = time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'completed', progress = 1.0, artifacts_json = ?, updated_at = ?, lease_until = NULL
                WHERE id = ?
                """,
                (json.dumps(artifacts), now, job_id),
            )

    def fail(self, job_id: str, error: str, logs: list[str]) -> None:
        now = time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', error = ?, logs_json = ?, updated_at = ?, lease_until = NULL
                WHERE id = ?
                """,
                (error, json.dumps(logs), now, job_id),
            )

    @staticmethod
    def _serialize_request(payload: SeparationJobRequest) -> str:
        storable = {
            "input_audio": payload.input_audio,
            "output_dir": payload.output_dir,
            "audio_file_base": payload.audio_file_base,
            "model_data": payload.model_data,
            "settings": payload.settings,
            "list_all_models": payload.list_all_models,
        }
        return json.dumps(storable)

    @staticmethod
    def _deserialize_request(serialized: str) -> SeparationJobRequest:
        data = json.loads(serialized)
        return SeparationJobRequest(
            input_audio=data["input_audio"],
            output_dir=data["output_dir"],
            audio_file_base=data["audio_file_base"],
            model_data=data["model_data"],
            settings=data.get("settings", {}),
            list_all_models=data.get("list_all_models", []),
        )

    def _deserialize_job(self, row: sqlite3.Row | dict[str, Any]) -> SeparationJob:
        return SeparationJob(
            id=row["id"],
            request=self._deserialize_request(row["request_json"]),
            status=row["status"],
            progress=float(row["progress"]),
            artifacts=json.loads(row["artifacts_json"]),
            logs=json.loads(row["logs_json"]),
            error=row["error"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )


class SeparationWorker:
    """GPU worker process loop for asynchronous separation jobs."""

    def __init__(self, store: DurableJobStore, poll_interval_seconds: float = 0.5) -> None:
        self.store = store
        self.poll_interval_seconds = poll_interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.claim_next()
            if job is None:
                sleep(self.poll_interval_seconds)
                continue

            logs = list(job.logs)
            try:
                self._run_job(job, logs)
            except Exception as exc:
                logs.append(f"job failed: {exc}")
                self.store.fail(job.id, str(exc), logs)

    def _run_job(self, job: SeparationJob, logs: list[str]) -> None:
        request = job.request
        Path(request.output_dir).mkdir(parents=True, exist_ok=True)

        def set_progress_bar(step: float, inference_iterations: int = 0):
            progress = float(step + inference_iterations)
            self.store.heartbeat(job.id, progress, logs)

        def write_to_console(message: str):
            logs.append(message)
            self.store.heartbeat(job.id, job.progress, logs)

        process_data = {
            "model_data": request.model_data,
            "export_path": request.output_dir,
            "audio_file_base": request.audio_file_base,
            "audio_file": request.input_audio,
            "set_progress_bar": set_progress_bar,
            "write_to_console": write_to_console,
            "process_iteration": request.settings.get("process_iteration", lambda: 0.0),
            "cached_source_callback": request.cached_source_callback or (lambda *args, **kwargs: (None, None)),
            "cached_model_source_holder": request.cached_model_source_holder or {},
            "list_all_models": request.list_all_models,
            "is_ensemble_master": bool(request.settings.get("is_ensemble_master", False)),
            "is_4_stem_ensemble": bool(request.settings.get("is_4_stem_ensemble", False)),
        }

        run_separation(request.model_data, process_data)
        artifacts = sorted(str(path) for path in Path(request.output_dir).glob(f"{request.audio_file_base}_*.wav"))
        self.store.complete(job.id, artifacts)


class SeparationJobAPI:
    """HTTP-style contract for Vercel API routes with external GPU workers."""

    def __init__(self, store: DurableJobStore | None = None, url_signing_secret: str = "uvr-secret") -> None:
        self.store = store or DurableJobStore()
        self._url_signing_secret = url_signing_secret

    def post_jobs(self, payload: SeparationJobRequest) -> dict[str, Any]:
        """Vercel endpoint behavior: persist metadata and enqueue job."""
        job = self.store.enqueue(payload)
        return {"id": job.id, "status": job.status}

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Vercel polling endpoint for job state."""
        job = self.store.get(job_id)
        return {
            "id": job.id,
            "status": job.status,
            "progress": job.progress,
            "error": job.error,
            "logs": list(job.logs),
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    def get_job_artifacts(self, job_id: str, expires_in_seconds: int = 900) -> dict[str, Any]:
        """Return signed URLs (tokenized local paths here; swap for S3/GCS presigned URLs in prod)."""
        job = self.store.get(job_id)
        now = int(time())
        expires_at = now + expires_in_seconds
        signed_artifacts = [self._sign_artifact_path(path, expires_at) for path in job.artifacts]
        return {"id": job.id, "status": job.status, "artifacts": signed_artifacts, "expires_at": expires_at}

    def _sign_artifact_path(self, artifact_path: str, expires_at: int) -> dict[str, str | int]:
        payload = f"{artifact_path}:{expires_at}".encode("utf-8")
        signature = hmac.new(self._url_signing_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return {
            "path": artifact_path,
            "url": f"/artifacts?path={artifact_path}&expires={expires_at}&sig={signature}",
            "expires_at": expires_at,
        }
