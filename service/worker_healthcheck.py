from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "cuda", "gpu"}


def _split_csv_env(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _directory_contains_model_artifact(path: Path, patterns: list[str]) -> bool:
    for pattern in patterns:
        if any(candidate.is_file() for candidate in path.rglob(pattern)):
            return True
    return False


def _resolve_required_model_file(path_value: str, model_paths: list[Path]) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate

    for model_path in model_paths:
        resolved = model_path / candidate
        if resolved.exists():
            return resolved

    return candidate


def verify_model_paths() -> list[str]:
    raw_paths = os.getenv(
        "UVR_MODEL_PATHS",
        "models/VR_Models,models/MDX_Net_Models,models/Demucs_Models",
    )
    model_paths = _split_csv_env(raw_paths)

    missing: list[str] = []
    invalid: list[str] = []
    model_dirs: list[Path] = []
    for item in model_paths:
        path = Path(item)
        if not path.exists():
            missing.append(item)
            continue
        if not path.is_dir():
            invalid.append(item)
            continue
        model_dirs.append(path)

    if missing:
        raise RuntimeError(f"missing model path(s): {', '.join(missing)}")
    if invalid:
        raise RuntimeError(f"model path(s) are not directories: {', '.join(invalid)}")

    raw_required_files = os.getenv("UVR_REQUIRED_MODEL_FILES")
    if raw_required_files:
        required_files = _split_csv_env(raw_required_files)
        missing_files: list[str] = []
        for required_file in required_files:
            resolved = _resolve_required_model_file(required_file, model_dirs)
            if not resolved.is_file():
                missing_files.append(required_file)
        if missing_files:
            raise RuntimeError(f"missing required model file(s): {', '.join(missing_files)}")
        return model_paths

    artifact_patterns = _split_csv_env(
        os.getenv(
            "UVR_MODEL_ARTIFACT_PATTERNS",
            "*.onnx,*.pth,*.pt,*.ckpt,*.bin,*.pb",
        )
    )
    empty_or_unusable: list[str] = []
    for model_dir in model_dirs:
        if not _directory_contains_model_artifact(model_dir, artifact_patterns):
            empty_or_unusable.append(str(model_dir))

    if empty_or_unusable:
        raise RuntimeError(
            "model path(s) contain no matching model artifacts "
            f"for patterns {artifact_patterns}: {', '.join(empty_or_unusable)}"
        )

    return model_paths


def verify_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg executable not found on PATH")
    subprocess.run([ffmpeg_path, "-version"], check=True, capture_output=True, text=True)
    return ffmpeg_path


def verify_backend_provider() -> tuple[str, list[str]]:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    expect_cuda = _as_bool(os.getenv("UVR_EXPECT_CUDA"), default=False)
    if expect_cuda:
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "UVR_EXPECT_CUDA is enabled but CUDAExecutionProvider is unavailable. "
                f"available providers: {providers}"
            )
        return "CUDAExecutionProvider", providers

    if "CPUExecutionProvider" not in providers:
        raise RuntimeError(f"CPUExecutionProvider unavailable. available providers: {providers}")

    if "CUDAExecutionProvider" in providers:
        return "CUDAExecutionProvider", providers

    return "CPUExecutionProvider", providers


def main() -> int:
    model_paths = verify_model_paths()
    ffmpeg_path = verify_ffmpeg()
    selected_provider, providers = verify_backend_provider()

    print(f"[healthcheck] model paths verified: {model_paths}")
    print(f"[healthcheck] ffmpeg: {ffmpeg_path}")
    print(f"[healthcheck] available providers: {providers}")
    print(f"[healthcheck] selected provider: {selected_provider}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[healthcheck] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
