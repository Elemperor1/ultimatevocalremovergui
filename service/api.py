from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep, time
from typing import Any, Callable
from urllib.parse import urlencode
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


class _DeserializedObject:
    def __init__(self, attrs: dict[str, Any]) -> None:
        self.__dict__.update(attrs)


class DurableJobStore:
    """SQLite-backed store that survives Vercel cold starts/retries."""
    SERIALIZED_TYPE_KEY = "__uvr_serialized_type__"

    def __init__(self, db_path: str | Path = "separation_jobs.sqlite3", lease_seconds: int = 120) -> None:
        self.db_path = str(db_path)
        self.lease_seconds = lease_seconds
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

    def claim_next(self, lease_seconds: int | None = None) -> SeparationJob | None:
        now = time()
        lease_duration = self.lease_seconds if lease_seconds is None else lease_seconds
        lease_until = now + lease_duration
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT id
                FROM jobs
                WHERE status = 'queued' OR (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?)
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None

            updated = conn.execute(
                """
                UPDATE jobs
                SET status = 'running', updated_at = ?, lease_until = ?
                WHERE id = ?
                  AND (status = 'queued' OR (status = 'running' AND lease_until IS NOT NULL AND lease_until < ?))
                """,
                (now, lease_until, row["id"], now),
            ).rowcount
            if updated == 0:
                conn.commit()
                return None

            claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
            if claimed is None:
                conn.commit()
                return None
            conn.commit()

        return self._deserialize_job(claimed)

    def heartbeat(self, job_id: str, progress: float, logs: list[str], lease_seconds: int | None = None) -> None:
        now = time()
        lease_duration = self.lease_seconds if lease_seconds is None else lease_seconds
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress = ?, logs_json = ?, updated_at = ?, lease_until = ? WHERE id = ?",
                (max(0.0, min(1.0, progress)), json.dumps(logs), now, now + lease_duration, job_id),
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
            "model_data": DurableJobStore._to_serializable(payload.model_data),
            "settings": DurableJobStore._to_serializable(payload.settings),
            "list_all_models": DurableJobStore._to_serializable(payload.list_all_models),
        }
        return json.dumps(storable)

    @staticmethod
    def _deserialize_request(serialized: str) -> SeparationJobRequest:
        data = json.loads(serialized)
        return SeparationJobRequest(
            input_audio=data["input_audio"],
            output_dir=data["output_dir"],
            audio_file_base=data["audio_file_base"],
            model_data=DurableJobStore._from_serialized(data["model_data"]),
            settings=DurableJobStore._from_serialized(data.get("settings", {})),
            list_all_models=DurableJobStore._from_serialized(data.get("list_all_models", [])),
        )

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): DurableJobStore._to_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [DurableJobStore._to_serializable(v) for v in value]
        if isinstance(value, tuple):
            return {DurableJobStore.SERIALIZED_TYPE_KEY: "tuple", "items": [DurableJobStore._to_serializable(v) for v in value]}
        if isinstance(value, set):
            return {DurableJobStore.SERIALIZED_TYPE_KEY: "set", "items": [DurableJobStore._to_serializable(v) for v in value]}
        if callable(value):
            return {
                DurableJobStore.SERIALIZED_TYPE_KEY: "callable",
                "repr": repr(value),  # callables are not rehydrated for safety; worker uses safe defaults for callbacks
            }
        if hasattr(value, "__dict__"):
            attrs = {k: DurableJobStore._to_serializable(v) for k, v in vars(value).items() if not callable(v)}
            return {DurableJobStore.SERIALIZED_TYPE_KEY: "object", "attrs": attrs}
        return str(value)

    @staticmethod
    def _from_serialized(value: Any) -> Any:
        if isinstance(value, list):
            return [DurableJobStore._from_serialized(v) for v in value]
        if isinstance(value, dict):
            kind = value.get(DurableJobStore.SERIALIZED_TYPE_KEY)
            if kind == "tuple":
                return tuple(DurableJobStore._from_serialized(v) for v in value.get("items", []))
            if kind == "set":
                return set(DurableJobStore._from_serialized(v) for v in value.get("items", []))
            if kind == "object":
                attrs = {k: DurableJobStore._from_serialized(v) for k, v in value.get("attrs", {}).items()}
                return _DeserializedObject(attrs)
            if kind == "callable":
                return None
            return {k: DurableJobStore._from_serialized(v) for k, v in value.items()}
        return value

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
        latest_progress = job.progress

        def set_progress_bar(step: float, inference_iterations: int = 0):
            nonlocal latest_progress
            latest_progress = float(step + inference_iterations)
            self.store.heartbeat(job.id, latest_progress, logs)

        def write_to_console(message: str):
            logs.append(message)
            self.store.heartbeat(job.id, latest_progress, logs)

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

    def __init__(self, store: DurableJobStore | None = None, url_signing_secret: str | None = None) -> None:
        self.store = store or DurableJobStore()
        secret = (url_signing_secret or os.getenv("UVR_URL_SIGNING_SECRET", "")).strip()
        if not secret:
            raise ValueError(
                "A non-empty url_signing_secret must be provided via constructor parameter "
                "or UVR_URL_SIGNING_SECRET environment variable for security."
            )
        self._url_signing_secret = secret

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
        signed_artifacts = [self._sign_artifact_path(job.id, path, expires_at) for path in job.artifacts]
        return {"id": job.id, "status": job.status, "artifacts": signed_artifacts, "expires_at": expires_at}

    def _sign_artifact_path(self, job_id: str, artifact_path: str, expires_at: int) -> dict[str, str | int]:
        artifact_id = hmac.new(
            self._url_signing_secret.encode("utf-8"),
            f"{job_id}:{artifact_path}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload = f"{job_id}:{artifact_id}:{artifact_path}:{expires_at}".encode("utf-8")
        signature = hmac.new(self._url_signing_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        query = urlencode({"job_id": job_id, "artifact_id": artifact_id, "expires": expires_at, "sig": signature})
        return {
            "id": artifact_id,
            "name": Path(artifact_path).name,
            "url": f"/artifacts?{query}",
            "expires_at": expires_at,
        }
