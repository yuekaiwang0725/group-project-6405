"""Plot training loss and eval F1 curves from HuggingFace Trainer logs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def save_training_curves(
    history_df: pd.DataFrame, output_path: str | Path, title: str = "Training Curves"
) -> None:
    """Save a 2-panel figure: loss curve (left) and F1 curve (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "loss" in history_df.columns:
        loss_df = history_df.dropna(subset=["loss"])
        axes[0].plot(loss_df["step"], loss_df["loss"], label="train_loss")
    if "eval_loss" in history_df.columns:
        eval_loss_df = history_df.dropna(subset=["eval_loss"])
        axes[0].plot(eval_loss_df["step"], eval_loss_df["eval_loss"], label="eval_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Step")
    axes[0].legend()

    if "eval_f1" in history_df.columns:
        f1_df = history_df.dropna(subset=["eval_f1"])
        axes[1].plot(f1_df["step"], f1_df["eval_f1"], label="eval_f1")
    axes[1].set_title("F1")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    fig.suptitle(title)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
