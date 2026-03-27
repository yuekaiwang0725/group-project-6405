from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


def save_robustness_plot(
    robustness_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Robustness F1 Drop by Perturbation",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=robustness_df, x="perturbation", y="f1_drop", hue="model", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel("F1 Drop")
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
