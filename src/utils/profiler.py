"""Lightweight training profiler: peak VRAM, wall-clock, throughput.

Used by experiments/run_lora.py and experiments/run_distilbert.py (optional
monkey-patch) to record resource consumption for the efficiency frontier.

The profiler is a context manager that takes a CUDA device index (or None for
CPU / MPS), resets peak-memory counters on entry, and on exit emits a single
JSON-serializable dict with:

    {
        "seconds": float,        # wall-clock duration
        "peak_vram_mb": float,   # peak allocated VRAM in MiB (CUDA only)
        "reserved_vram_mb": float,
        "device": "cuda:0" | "cpu" | "mps",
    }

Design notes
------------
- Uses torch.cuda.max_memory_allocated (not nvidia-smi) so readings are
  process-scoped and reproducible across runs.
- Calls torch.cuda.synchronize() on enter/exit so the timing window is not
  cut short by async kernel launches.
- Safe to use when CUDA is unavailable — falls back to wall-clock only.
"""

from __future__ import annotations

import json
import time
from contextlib import ContextDecorator
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep in practice
    torch = None  # type: ignore


class TrainingProfiler(ContextDecorator):
    """Context manager that records peak VRAM and wall-clock time.

    Example
    -------
    >>> profiler = TrainingProfiler(device="cuda:0")
    >>> with profiler:
    ...     trainer.train()
    >>> print(profiler.stats)
    """

    def __init__(self, device: str = "cuda:0", label: str = "train"):
        self.device = device
        self.label = label
        self._t0: float = 0.0
        self.stats: dict[str, Any] = {}

    def __enter__(self) -> "TrainingProfiler":
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            reserved_bytes = torch.cuda.max_memory_reserved()
        else:
            peak_bytes = 0
            reserved_bytes = 0

        self.stats = {
            "label": self.label,
            "device": self.device,
            "seconds": round(time.perf_counter() - self._t0, 3),
            "peak_vram_mb": round(peak_bytes / (1024 * 1024), 2),
            "reserved_vram_mb": round(reserved_bytes / (1024 * 1024), 2),
        }

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)


def count_trainable_params(model: Any) -> dict[str, int]:
    """Return {'total': X, 'trainable': Y, 'ratio': Y/X} for a PyTorch module."""
    if torch is None:
        return {"total": 0, "trainable": 0, "ratio": 0.0}
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "ratio": float(trainable / total) if total else 0.0,
    }
