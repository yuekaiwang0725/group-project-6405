from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features.tfidf import build_vectorizer
from src.models.baseline_svm import BaselineSVM
from src.training.evaluate import evaluate_predictions
from src.utils.io import ensure_dir, write_json
from src.visualization.confusion_matrix import save_confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_split(dataset_name: str, split: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / f"{dataset_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}. Run `python -m src.main prepare-data` first."
        )
    return pd.read_csv(path)


def run_baseline(dataset_name: str = "imdb") -> None:
    train_df = _load_split(dataset_name, "train")
    test_df = _load_split(dataset_name, "test")

    vectorizer = build_vectorizer(max_features=30000, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_df["text"].tolist())
    x_test = vectorizer.transform(test_df["text"].tolist())
    y_train = train_df["label"].astype(int).tolist()
    y_test = test_df["label"].astype(int).tolist()

    model = BaselineSVM(c=1.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test).tolist()
    metrics = evaluate_predictions(y_true=y_test, y_pred=y_pred)

    checkpoint_dir = ensure_dir(PROJECT_ROOT / "checkpoints" / "baseline" / dataset_name)
    results_tables = ensure_dir(PROJECT_ROOT / "results" / "tables")
    results_figures = ensure_dir(PROJECT_ROOT / "results" / "figures")

    joblib.dump(vectorizer, checkpoint_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, checkpoint_dir / "svm_model.joblib")
    write_json(metrics, results_tables / f"baseline_{dataset_name}_metrics.json")
    pd.DataFrame([metrics]).to_csv(
        results_tables / f"baseline_{dataset_name}_metrics.csv", index=False
    )
    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        output_path=results_figures / f"baseline_{dataset_name}_confusion_matrix.png",
        title=f"Baseline SVM Confusion Matrix ({dataset_name.upper()})",
    )

    print(f"Baseline completed for {dataset_name}. Metrics: {metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF + SVM baseline experiment")
    parser.add_argument("--dataset", default="imdb", choices=["imdb", "sst2"])
    args = parser.parse_args()
    run_baseline(args.dataset)


if __name__ == "__main__":
    main()
