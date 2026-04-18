"""Device detection and configuration (CPU / CUDA / Apple MPS)."""

from __future__ import annotations

import os

import torch


def has_mps() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def best_available_device() -> str:
    if has_mps():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(requested: str | None = None) -> str:
    if requested is None:
        requested = "auto"

    requested = requested.lower()
    valid_devices = {"auto", "cpu", "cuda", "mps"}
    if requested not in valid_devices:
        raise ValueError(f"Unsupported device '{requested}'. Choose from {sorted(valid_devices)}.")

    if requested == "auto":
        return best_available_device()

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")

    if requested == "mps" and not has_mps():
        raise RuntimeError("MPS was requested but is not available on this machine.")

    return requested


def configure_device_runtime(device: str) -> None:
    # Allow unsupported ops to fall back to CPU when running on Apple Silicon.
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
