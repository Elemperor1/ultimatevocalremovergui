from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from time import time
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


class SeparationJobAPI:
    """Headless, in-process API contract for separation jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, SeparationJob] = {}
        self._lock = Lock()

    def post_jobs(self, payload: SeparationJobRequest) -> dict[str, Any]:
        """HTTP-style handler for POST /jobs."""
        job_id = str(uuid4())
        job = SeparationJob(id=job_id, request=payload)
        with self._lock:
            self._jobs[job_id] = job

        thread = Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return {"id": job_id, "status": job.status}

    def get_job(self, job_id: str) -> dict[str, Any]:
        """HTTP-style handler for GET /jobs/{id}."""
        job = self._jobs[job_id]
        return {
            "id": job.id,
            "status": job.status,
            "progress": job.progress,
            "error": job.error,
            "logs": list(job.logs),
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    def get_job_artifacts(self, job_id: str) -> dict[str, Any]:
        """HTTP-style handler for GET /jobs/{id}/artifacts."""
        job = self._jobs[job_id]
        return {"id": job.id, "artifacts": list(job.artifacts)}

    def _run_job(self, job_id: str) -> None:
        job = self._jobs[job_id]
        request = job.request

        def set_progress_bar(step: float, inference_iterations: int = 0):
            progress = float(step + inference_iterations)
            with self._lock:
                job.progress = max(0.0, min(1.0, progress))
                job.updated_at = time()

        def write_to_console(message: str):
            with self._lock:
                job.logs.append(message)
                job.updated_at = time()

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

        try:
            with self._lock:
                job.status = "running"
                job.updated_at = time()

            Path(request.output_dir).mkdir(parents=True, exist_ok=True)
            run_separation(request.model_data, process_data)

            artifact_pattern = f"{request.audio_file_base}_(*.wav"
            artifacts = sorted(str(p) for p in Path(request.output_dir).glob(artifact_pattern))

            with self._lock:
                job.status = "completed"
                job.progress = 1.0
                job.artifacts = artifacts
                job.updated_at = time()
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.updated_at = time()
