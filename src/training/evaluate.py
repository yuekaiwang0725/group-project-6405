"""Thin evaluation wrapper — delegates to classification_metrics."""

from __future__ import annotations

from src.utils.metrics import classification_metrics


def evaluate_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Return accuracy, precision, recall, F1 for a set of predictions."""
    return classification_metrics(y_true=y_true, y_pred=y_pred)
