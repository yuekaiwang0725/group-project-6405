from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.inference import (
    load_texts_labels,
    predict_with_baseline,
    predict_with_distilbert,
)
from src.training.evaluate import evaluate_predictions
from src.utils.io import ensure_dir
from src.visualization.comparison_plots import save_model_comparison_barplot
from src.visualization.cross_domain_heatmap import save_cross_domain_heatmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ["imdb", "sst2"]


def _load_test_df(domain: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / f"{domain}_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing test split: {path}")
    return pd.read_csv(path)


def _evaluate_baseline() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for train_domain in DOMAINS:
        vec_path = (
            PROJECT_ROOT
            / "checkpoints"
            / "baseline"
            / train_domain
            / "tfidf_vectorizer.joblib"
        )
        model_path = PROJECT_ROOT / "checkpoints" / "baseline" / train_domain / "svm_model.joblib"
        if not vec_path.exists() or not model_path.exists():
            continue

        for test_domain in DOMAINS:
            test_df = _load_test_df(test_domain)
            texts, labels = load_texts_labels(test_df)
            preds = predict_with_baseline(texts, vec_path, model_path)
            metrics = evaluate_predictions(labels, preds)
            rows.append(
                {
                    "model": "baseline_svm",
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _evaluate_distilbert() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for train_domain in DOMAINS:
        checkpoint = PROJECT_ROOT / "checkpoints" / "distilbert" / train_domain / "best"
        if not checkpoint.exists():
            continue

        for test_domain in DOMAINS:
            test_df = _load_test_df(test_domain)
            texts, labels = load_texts_labels(test_df)
            preds = predict_with_distilbert(texts, checkpoint)
            metrics = evaluate_predictions(labels, preds)
            rows.append(
                {
                    "model": "distilbert",
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    tables_dir = ensure_dir(PROJECT_ROOT / "results" / "tables")
    figures_dir = ensure_dir(PROJECT_ROOT / "results" / "figures")

    baseline_df = _evaluate_baseline()
    distilbert_df = _evaluate_distilbert()
    all_df = pd.concat([baseline_df, distilbert_df], ignore_index=True)

    if all_df.empty:
        raise RuntimeError(
            "No available checkpoints found. Run baseline/distilbert experiments first."
        )

    all_df.to_csv(tables_dir / "metrics_cross_domain.csv", index=False)

    for model_name, model_df in all_df.groupby("model"):
        save_cross_domain_heatmap(
            metrics_df=model_df,
            output_path=figures_dir / f"{model_name}_cross_domain_heatmap.png",
            value_column="f1",
            title=f"{model_name} Cross-domain F1",
        )

    in_domain_df = all_df[all_df["train_domain"] == all_df["test_domain"]].copy()
    in_domain_df["dataset"] = in_domain_df["test_domain"]
    save_model_comparison_barplot(
        in_domain_df,
        output_path=figures_dir / "model_comparison_in_domain_f1.png",
        metric="f1",
        title="In-domain Model Comparison (F1)",
    )
    print("Cross-domain evaluation completed.")


if __name__ == "__main__":
    main()
