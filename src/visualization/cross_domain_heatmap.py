from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


def save_cross_domain_heatmap(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    value_column: str = "f1",
    title: str = "Cross-domain F1 Heatmap",
) -> None:
    pivot_df = metrics_df.pivot(index="train_domain", columns="test_domain", values=value_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Test Domain")
    ax.set_ylabel("Train Domain")
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
