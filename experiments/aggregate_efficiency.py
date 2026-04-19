"""Aggregate all `*_bundle.json` files in results/tables into a single
efficiency-frontier CSV + generate the comparison figures.

This is intentionally self-contained so you can re-run it any time new bundles
appear — it will not re-train anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir
from src.visualization.efficiency_frontier import (
    save_efficiency_pareto,
    save_training_time_bar,
    save_vram_bar,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_bundles() -> list[dict]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    bundles: list[dict] = []
    for path in sorted(tables_dir.glob("*_bundle.json")):
        with path.open("r", encoding="utf-8") as fh:
            bundles.append(json.load(fh))
    return bundles


def _flatten(bundle: dict) -> dict:
    test = bundle.get("test_metrics") or {}
    train_prof = bundle.get("train_profile") or {}
    infer_prof = bundle.get("infer_profile") or {}
    params = bundle.get("param_stats") or {}
    flat = {
        "run_name": bundle.get("run_name"),
        "model_name": bundle.get("model_name"),
        "dataset": bundle.get("dataset"),
        "method": bundle.get("method", "unknown"),
        "lora_r": bundle.get("lora_r"),
        "batch_size": bundle.get("batch_size"),
        "epochs": bundle.get("epochs"),
        "bf16": bundle.get("bf16", False),
        "total_params": params.get("total", 0),
        "trainable_params": params.get("trainable", 0),
        "trainable_ratio": params.get("ratio", 0.0),
        "train_seconds": train_prof.get("seconds", 0.0),
        "train_peak_vram_mb": train_prof.get("peak_vram_mb", 0.0),
        "infer_seconds": infer_prof.get("seconds", 0.0),
        "infer_peak_vram_mb": infer_prof.get("peak_vram_mb", 0.0),
        # metric keys are prefixed "test_" by HF Trainer
        "test_accuracy": test.get("test_accuracy", test.get("accuracy")),
        "test_f1": test.get("test_f1", test.get("f1")),
        "test_precision": test.get("test_precision", test.get("precision")),
        "test_recall": test.get("test_recall", test.get("recall")),
    }
    return flat


def main() -> None:
    bundles = _load_bundles()
    if not bundles:
        print("No *_bundle.json files found under results/tables. Run experiments first.")
        return

    rows = [_flatten(b) for b in bundles]
    df = pd.DataFrame(rows)

    tables_dir = ensure_dir(PROJECT_ROOT / "results" / "tables")
    figures_dir = ensure_dir(PROJECT_ROOT / "results" / "figures")

    out_csv = tables_dir / "efficiency_frontier.csv"
    df.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv} with {len(df)} rows")

    # One Pareto plot per dataset so the axes stay readable.
    for ds, sub in df.groupby("dataset"):
        if sub["test_f1"].isnull().all():
            continue
        save_efficiency_pareto(
            sub,
            figures_dir / f"efficiency_frontier_{ds}.png",
            dataset=ds,
        )
        save_training_time_bar(
            sub,
            figures_dir / f"training_time_{ds}.png",
            dataset=ds,
        )
        save_vram_bar(
            sub,
            figures_dir / f"vram_{ds}.png",
            dataset=ds,
        )
    print("[ok] figures written to results/figures/")


if __name__ == "__main__":
    main()
