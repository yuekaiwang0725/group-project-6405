"""Grouped bar chart comparing in-domain F1 across models and datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


def save_model_comparison_barplot(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric: str = "f1",
    title: str = "Model Comparison",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=metrics_df, x="model", y=metric, hue="dataset", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
