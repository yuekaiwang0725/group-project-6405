"""EDA plots: label distribution bar chart and text-length histogram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


def plot_label_distribution(df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_text_length_distribution(
    df: pd.DataFrame, output_path: str | Path, title: str
) -> None:
    lengths = df["text"].str.split().map(len)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(lengths, bins=40, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Frequency")
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
