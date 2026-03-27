from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from src.models.inference import predict_with_baseline, predict_with_distilbert
from src.robustness.perturbation import available_perturbations
from src.robustness.robustness_eval import evaluate_robustness
from src.utils.io import ensure_dir
from src.visualization.robustness_curve import save_robustness_plot

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_test(domain: str = "imdb") -> tuple[list[str], list[int]]:
    path = PROJECT_ROOT / "data" / "processed" / f"{domain}_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    return df["text"].astype(str).tolist(), df["label"].astype(int).tolist()


def main() -> None:
    texts, labels = _load_test("imdb")
    perturbations = available_perturbations()

    rows: list[dict[str, object]] = []
    models: list[tuple[str, Callable[[list[str]], list[int]]]] = []

    vec_path = PROJECT_ROOT / "checkpoints" / "baseline" / "imdb" / "tfidf_vectorizer.joblib"
    svm_path = PROJECT_ROOT / "checkpoints" / "baseline" / "imdb" / "svm_model.joblib"
    if vec_path.exists() and svm_path.exists():
        models.append(
            (
                "baseline_svm",
                lambda batch: predict_with_baseline(batch, vec_path, svm_path),
            )
        )

    distil_checkpoint = PROJECT_ROOT / "checkpoints" / "distilbert" / "imdb" / "best"
    if distil_checkpoint.exists():
        models.append(
            (
                "distilbert",
                lambda batch: predict_with_distilbert(batch, distil_checkpoint),
            )
        )

    if not models:
        raise RuntimeError("No baseline/distilbert checkpoints found for robustness evaluation.")

    for model_name, predict_fn in models:
        for perturb_name, perturb_fn in perturbations.items():
            metrics = evaluate_robustness(
                texts=texts,
                labels=labels,
                predict_fn=predict_fn,
                perturb_fn=perturb_fn,
            )
            rows.append(
                {
                    "model": model_name,
                    "dataset": "imdb",
                    "perturbation": perturb_name,
                    **metrics,
                }
            )

    output_df = pd.DataFrame(rows)
    tables_dir = ensure_dir(PROJECT_ROOT / "results" / "tables")
    figures_dir = ensure_dir(PROJECT_ROOT / "results" / "figures")
    output_df.to_csv(tables_dir / "robustness_scores.csv", index=False)
    save_robustness_plot(
        output_df,
        output_path=figures_dir / "robustness_drop_curve.png",
        title="Robustness F1 Drop on IMDb Test",
    )
    print("Robustness evaluation completed.")


if __name__ == "__main__":
    main()
