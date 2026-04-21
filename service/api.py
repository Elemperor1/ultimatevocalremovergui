from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import tempfile
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
    artifacts: list[Any] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


class _DeserializedObject:
    def __init__(self, attrs: dict[str, Any]) -> None:
        self.__dict__.update(attrs)


class StorageLayout:
    """Canonical key layout for object storage providers (S3/R2/GCS)."""

    INPUTS_PREFIX = "inputs"
    MODELS_PREFIX = "models"
    OUTPUTS_PREFIX = "outputs"

    @classmethod
    def input_prefix(cls, job_id: str) -> str:
        return f"{cls.INPUTS_PREFIX}/{job_id}"

    @classmethod
    def output_prefix(cls, job_id: str) -> str:
        return f"{cls.OUTPUTS_PREFIX}/{job_id}"

    @classmethod
    def model_key(cls, name: str) -> str:
        clean_name = name.lstrip("/")
        if clean_name.startswith(f"{cls.MODELS_PREFIX}/"):
            return clean_name
        return f"{cls.MODELS_PREFIX}/{clean_name}"


class LocalObjectStorage:
    """Filesystem-backed storage that mirrors S3/R2/GCS key prefixes."""

    def __init__(self, root_dir: str | Path, url_ttl_seconds: int = 3600) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.url_ttl_seconds = int(url_ttl_seconds)

    def path_for_key(self, key: str) -> Path:
        key_path = Path(str(key))
        if key_path.is_absolute() or ".." in key_path.parts:
            raise ValueError(f"unsafe storage key: {key}")
        resolved = (self.root_dir / key_path).resolve()
        try:
            resolved.relative_to(self.root_dir)
        except ValueError as exc:
            raise ValueError(f"unsafe storage key: {key}") from exc
        return resolved

    def stream_download(self, key: str, destination: str | Path, chunk_size: int = 1024 * 1024) -> Path:
        src = self.path_for_key(key)
        if not src.exists():
            raise FileNotFoundError(f"storage key not found: {key}")
        dst = Path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as src_fp, dst.open("wb") as dst_fp:
            for chunk in iter(lambda: src_fp.read(chunk_size), b""):
                dst_fp.write(chunk)
        return dst

    def upload_file(self, local_path: str | Path, key: str) -> dict[str, Any]:
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"cannot upload missing file: {local_path}")
        dst = self.path_for_key(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        checksum = _sha256_file(dst)
        return {
            "key": key,
            "name": src.name,
            "size_bytes": dst.stat().st_size,
            "sha256": checksum,
            "url": self.signed_url(key),
            "uploaded_at": int(time()),
        }

    def signed_url(self, key: str, expires_in_seconds: int | None = None) -> str:
        ttl = self.url_ttl_seconds if expires_in_seconds is None else max(1, int(expires_in_seconds))
        expires_at = int(time()) + ttl
        # Local worker path is intentionally opaque but stable for API return payloads.
        file_uri = self.path_for_key(key).resolve().as_uri()
        query = urlencode({"expires": expires_at})
        return f"{file_uri}?{query}"

    def remove_prefix_older_than(self, prefix: str, cutoff_epoch: float) -> int:
        prefix_path = self.path_for_key(prefix)
        if not prefix_path.exists():
            return 0
        deleted = 0
        for candidate in prefix_path.rglob("*"):
            if candidate.is_file() and candidate.stat().st_mtime < cutoff_epoch:
                candidate.unlink(missing_ok=True)
                deleted += 1
        return deleted


@dataclass
class PreloadModelSpec:
    key: str
    sha256: str | None = None
    version: str | None = None


class ModelCacheManager:
    def __init__(self, storage: LocalObjectStorage, cache_dir: str | Path, specs: list[PreloadModelSpec]) -> None:
        self.storage = storage
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.specs = specs
        self.manifest_path = self.cache_dir / "manifest.json"

    def preload(self) -> list[str]:
        logs: list[str] = []
        manifest = self._read_manifest()
        expected_keys = {spec.key for spec in self.specs}

        # cache invalidation by version/hash or removed specs.
        for key, entry in list(manifest.items()):
            cache_name = entry.get("cache_name")
            if key not in expected_keys:
                if cache_name:
                    cache_path = self.cache_dir / cache_name
                    if cache_path.is_file():
                        cache_path.unlink(missing_ok=True)
                manifest.pop(key, None)
                logs.append(f"model cache invalidated (removed from preload list): {key}")

        for spec in self.specs:
            cache_name = spec.key.replace("/", "__")
            target = self.cache_dir / cache_name
            entry = manifest.get(spec.key, {})
            expected_hash = spec.sha256.strip().lower() if spec.sha256 else None
            valid = target.exists()

            if valid and expected_hash:
                local_hash = _sha256_file(target)
                if local_hash != expected_hash:
                    valid = False
                    logs.append(f"model cache hash mismatch; reloading {spec.key}")

            if valid and spec.version and entry.get("version") != spec.version:
                valid = False
                logs.append(f"model cache version changed; reloading {spec.key}")

            if valid:
                logs.append(f"model cache hit: {spec.key}")
                manifest[spec.key] = {
                    "cache_name": cache_name,
                    "version": spec.version,
                    "sha256": expected_hash or entry.get("sha256"),
                }
                continue

            self.storage.stream_download(spec.key, target)
            local_hash = _sha256_file(target)
            if expected_hash and local_hash != expected_hash:
                target.unlink(missing_ok=True)
                raise ValueError(f"checksum verification failed for model {spec.key}")
            manifest[spec.key] = {
                "cache_name": cache_name,
                "version": spec.version,
                "sha256": expected_hash or local_hash,
            }
            logs.append(f"model preloaded: {spec.key}")

        self._write_manifest(manifest)
        return logs

    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_manifest(self, payload: dict[str, dict[str, Any]]) -> None:
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
                return None

            claimed = conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
            if claimed is None:
                return None

        return self._deserialize_job(claimed)

    def heartbeat(self, job_id: str, progress: float, logs: list[str], lease_seconds: int | None = None) -> None:
        now = time()
        lease_duration = self.lease_seconds if lease_seconds is None else lease_seconds
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET progress = ?, logs_json = ?, updated_at = ?, lease_until = ?
                WHERE id = ?
                  AND status = 'running'
                  AND lease_until IS NOT NULL
                  AND lease_until >= ?
                """,
                (max(0.0, min(1.0, progress)), json.dumps(logs), now, now + lease_duration, job_id, now),
            )

    def complete(self, job_id: str, artifacts: list[Any]) -> None:
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
            raise TypeError("callable values are not serializable in durable job payloads")
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
        self.storage = LocalObjectStorage(os.getenv("UVR_STORAGE_ROOT", "data/object_storage"))
        self.model_cache = ModelCacheManager(
            storage=self.storage,
            cache_dir=os.getenv("UVR_MODEL_CACHE_DIR", "data/model_cache"),
            specs=_parse_preload_model_specs(os.getenv("UVR_PRELOAD_MODELS", "[]")),
        )
        self.output_retention_seconds = _safe_int_env("UVR_OUTPUT_RETENTION_SECONDS", 7 * 24 * 3600)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._preload_models_and_prune()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _preload_models_and_prune(self) -> None:
        preload_logs = self.model_cache.preload()
        for line in preload_logs:
            print(f"[worker-startup] {line}")
        cutoff = time() - self.output_retention_seconds
        deleted = self.storage.remove_prefix_older_than(StorageLayout.OUTPUTS_PREFIX, cutoff)
        if deleted:
            print(f"[worker-startup] removed {deleted} expired output artifacts")

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
        latest_progress = job.progress

        def set_progress_bar(step: float, inference_iterations: int = 0):
            nonlocal latest_progress
            latest_progress = float(step + inference_iterations)
            self.store.heartbeat(job.id, latest_progress, logs)

        def write_to_console(message: str):
            logs.append(message)
            self.store.heartbeat(job.id, latest_progress, logs)

        with tempfile.TemporaryDirectory(prefix=f"uvr-job-{job.id}-") as temp_dir:
            temp_root = Path(temp_dir)
            local_input = self._materialize_input_audio(job.id, request, temp_root, logs)
            local_outputs = temp_root / "outputs"
            local_outputs.mkdir(parents=True, exist_ok=True)

            process_data = {
                "model_data": request.model_data,
                "export_path": str(local_outputs),
                "audio_file_base": request.audio_file_base,
                "audio_file": str(local_input),
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

            artifacts: list[dict[str, Any]] = []
            for path in sorted(local_outputs.glob(f"{request.audio_file_base}_*.wav")):
                key = f"{StorageLayout.output_prefix(job.id)}/{path.name}"
                metadata = self.storage.upload_file(path, key)
                artifacts.append(metadata)
                logs.append(f"uploaded output artifact: {key}")
                self.store.heartbeat(job.id, latest_progress, logs)

        self.store.complete(job.id, artifacts)

    def _materialize_input_audio(
        self,
        job_id: str,
        request: SeparationJobRequest,
        temp_root: Path,
        logs: list[str],
    ) -> Path:
        source = request.settings.get("input_storage_key") or request.input_audio
        source_str = str(source)
        source_suffix = Path(source_str).suffix or ".wav"
        local_input = temp_root / Path(request.audio_file_base).with_suffix(source_suffix).name

        if source_str.startswith((f"{StorageLayout.INPUTS_PREFIX}/", f"{StorageLayout.MODELS_PREFIX}/")):
            self.storage.stream_download(source_str, local_input)
            logs.append(f"streamed input from storage key: {source_str}")
            return local_input

        src_path = Path(source_str)
        if src_path.exists():
            with src_path.open("rb") as src_fp, local_input.open("wb") as dst_fp:
                for chunk in iter(lambda: src_fp.read(1024 * 1024), b""):
                    dst_fp.write(chunk)
            logs.append(f"streamed input from local path: {source_str}")
            return local_input

        # Fallback to canonical storage location for the job.
        ext = Path(source_str).suffix or ".wav"
        guessed_key = f"{StorageLayout.input_prefix(job_id)}/input{ext}"
        self.storage.stream_download(guessed_key, local_input)
        logs.append(f"streamed input from inferred storage key: {guessed_key}")
        return local_input


class SeparationJobAPI:
    """HTTP-style contract for Vercel API routes with external GPU workers."""

    def __init__(self, store: DurableJobStore | None = None, url_signing_secret: str | None = None) -> None:
        self.store = store or DurableJobStore()
        secret = (url_signing_secret or os.getenv("UVR_URL_SIGNING_SECRET", "")).strip()
        if not secret:
            raise ValueError(
                "URL signing secret must be provided via constructor parameter "
                "or UVR_URL_SIGNING_SECRET environment variable."
            )
        self._url_signing_secret = secret

    def post_jobs(self, payload: SeparationJobRequest) -> dict[str, Any]:
        """Vercel endpoint behavior: persist metadata and enqueue job."""
        job = self.store.enqueue(payload)
        return {
            "id": job.id,
            "status": job.status,
            "storage_layout": {
                "inputs": f"{StorageLayout.input_prefix(job.id)}/",
                "models": f"{StorageLayout.MODELS_PREFIX}/",
                "outputs": f"{StorageLayout.output_prefix(job.id)}/",
            },
        }

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
        """Return signed URLs and metadata for artifacts after completion."""
        job = self.store.get(job_id)
        now = int(time())
        expires_at = now + expires_in_seconds
        signed_artifacts = [self._to_artifact_response(job.id, item, expires_at) for item in job.artifacts]
        return {"id": job.id, "status": job.status, "artifacts": signed_artifacts, "expires_at": expires_at}

    def _to_artifact_response(self, job_id: str, artifact: Any, expires_at: int) -> dict[str, Any]:
        if isinstance(artifact, dict):
            artifact_path = artifact.get("key") or artifact.get("path") or artifact.get("name") or "unknown"
            signed = self._sign_artifact_path(job_id, artifact_path, expires_at)
            merged = dict(artifact)
            merged.update(
                {
                    "id": signed["id"],
                    "name": artifact.get("name") or signed["name"],
                    "url": signed["url"],
                    "expires_at": expires_at,
                }
            )
            return merged

        artifact_path = str(artifact)
        signed = self._sign_artifact_path(job_id, artifact_path, expires_at)
        return {
            "id": signed["id"],
            "name": Path(artifact_path).name,
            "url": signed["url"],
            "expires_at": expires_at,
        }

    def _artifact_id(self, job_id: str, artifact_path: str) -> str:
        return hmac.new(
            self._url_signing_secret.encode("utf-8"),
            f"{job_id}:{artifact_path}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _sign_artifact_path(self, job_id: str, artifact_path: str, expires_at: int) -> dict[str, str | int]:
        artifact_id = self._artifact_id(job_id, artifact_path)
        payload = f"{job_id}:{artifact_id}:{artifact_path}:{expires_at}".encode("utf-8")
        signature = hmac.new(self._url_signing_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        query = urlencode({"job_id": job_id, "artifact_id": artifact_id, "expires": expires_at, "sig": signature})
        return {
            "id": artifact_id,
            "name": Path(artifact_path).name,
            "url": f"/artifacts?{query}",
            "expires_at": expires_at,
        }


def _parse_preload_model_specs(raw: str) -> list[PreloadModelSpec]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    specs: list[PreloadModelSpec] = []
    for item in payload:
        if isinstance(item, str):
            specs.append(PreloadModelSpec(key=StorageLayout.model_key(item)))
            continue
        if not isinstance(item, dict) or "key" not in item:
            continue
        specs.append(
            PreloadModelSpec(
                key=StorageLayout.model_key(str(item["key"])),
                sha256=str(item["sha256"]).strip() if item.get("sha256") else None,
                version=str(item["version"]).strip() if item.get("version") else None,
            )
        )
    return specs


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default
