"""Classification metrics: accuracy, precision, recall, F1 (binary or macro)."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def infer_average_mode(labels: list[int]) -> str:
    """Return 'binary' for 2-class, 'macro' for multi-class."""
    return "binary" if len(set(labels)) <= 2 else "macro"


def classification_metrics(
    y_true: list[int], y_pred: list[int], average: str | None = None
) -> dict[str, float]:
    effective_average = average or infer_average_mode(y_true + y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=effective_average, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}
