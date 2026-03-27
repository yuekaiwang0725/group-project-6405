from __future__ import annotations

from collections.abc import Callable

from src.training.evaluate import evaluate_predictions


def evaluate_robustness(
    texts: list[str],
    labels: list[int],
    predict_fn: Callable[[list[str]], list[int]],
    perturb_fn: Callable[[str], str],
) -> dict[str, float]:
    baseline_preds = predict_fn(texts)
    baseline_metrics = evaluate_predictions(labels, baseline_preds)

    perturbed = [perturb_fn(text) for text in texts]
    perturbed_preds = predict_fn(perturbed)
    perturbed_metrics = evaluate_predictions(labels, perturbed_preds)

    return {
        "baseline_f1": baseline_metrics["f1"],
        "perturbed_f1": perturbed_metrics["f1"],
        "f1_drop": baseline_metrics["f1"] - perturbed_metrics["f1"],
        "baseline_acc": baseline_metrics["accuracy"],
        "perturbed_acc": perturbed_metrics["accuracy"],
    }
