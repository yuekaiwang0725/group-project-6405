"""Efficiency-frontier plots: Pareto (params vs F1 vs VRAM), bar charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _short_label(row: pd.Series) -> str:
    """Compact legend name for a model row."""
    model = str(row.get("model_name", "")).split("/")[-1]
    method = "LoRA" if str(row.get("method", "")).lower() == "lora" else "full-FT"
    r = row.get("lora_r")
    if method == "LoRA" and pd.notna(r):
        return f"{model} LoRA(r={int(r)})"
    return f"{model} ({method})"


def save_efficiency_pareto(
    df: pd.DataFrame,
    output_path: str | Path,
    dataset: str,
) -> None:
    """Scatter: x=trainable params (log), y=test F1, bubble size=train VRAM.

    Expects columns: trainable_params, test_f1, train_peak_vram_mb, model_name.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Clean + guard against NaN rows
    d = df.dropna(subset=["trainable_params", "test_f1"]).copy()
    if d.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    d["trainable_params"] = d["trainable_params"].clip(lower=1)
    vmax = max(d["train_peak_vram_mb"].max(), 1.0)
    sizes = 80 + (d["train_peak_vram_mb"] / vmax) * 600

    for _, row in d.iterrows():
        ax.scatter(
            row["trainable_params"],
            row["test_f1"],
            s=float(80 + (row["train_peak_vram_mb"] / vmax) * 600),
            alpha=0.75,
            edgecolors="black",
            linewidths=0.8,
            label=_short_label(row),
        )
        ax.annotate(
            _short_label(row),
            (row["trainable_params"], row["test_f1"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test F1")
    ax.set_title(f"Efficiency frontier — {dataset.upper()}\n(bubble size = peak train VRAM)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_training_time_bar(
    df: pd.DataFrame,
    output_path: str | Path,
    dataset: str,
) -> None:
    d = df.dropna(subset=["train_seconds"]).copy()
    if d.empty:
        return
    d["label"] = d.apply(_short_label, axis=1)
    d = d.sort_values("train_seconds")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(d))))
    ax.barh(d["label"], d["train_seconds"] / 60.0)
    ax.set_xlabel("Training wall-clock (minutes)")
    ax.set_title(f"Training time — {dataset.upper()}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_vram_bar(
    df: pd.DataFrame,
    output_path: str | Path,
    dataset: str,
) -> None:
    d = df.dropna(subset=["train_peak_vram_mb"]).copy()
    if d.empty:
        return
    d["label"] = d.apply(_short_label, axis=1)
    d = d.sort_values("train_peak_vram_mb")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(d))))
    ax.barh(d["label"], d["train_peak_vram_mb"] / 1024.0)
    ax.set_xlabel("Peak training VRAM (GiB)")
    ax.set_title(f"VRAM footprint — {dataset.upper()}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
