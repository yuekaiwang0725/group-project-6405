"""Save a confusion matrix heatmap as a PNG image."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.io import ensure_dir


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_path: str | Path,
    title: str,
    labels: list[int] | None = None,
    class_names: list[str] | None = None,
) -> None:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    if class_names is not None and len(class_names) != len(labels):
        raise ValueError("class_names length must match labels length.")

    if class_names is None:
        if labels == [0, 1]:
            tick_labels = ["neg", "pos"]
        else:
            tick_labels = [str(label) for label in labels]
    else:
        tick_labels = class_names

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
