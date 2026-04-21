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


def verify_model_paths() -> list[str]:
    raw_paths = os.getenv(
        "UVR_MODEL_PATHS",
        "models/VR_Models,models/MDX_Net_Models,models/Demucs_Models",
    )
    missing: list[str] = []
    for item in [p.strip() for p in raw_paths.split(",") if p.strip()]:
        if not Path(item).exists():
            missing.append(item)
    if missing:
        raise RuntimeError(f"missing model path(s): {', '.join(missing)}")
    return [p.strip() for p in raw_paths.split(",") if p.strip()]


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
