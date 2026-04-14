from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils.metrics import classification_metrics, infer_average_mode
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_NAME = "logistic_regression"



# DATA LOADING
# ========================
def load_split(dataset: str, split: str):
    path = PROJECT_ROOT / "data" / "processed" / f"{dataset}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)



# TRAINING
# ========================
def train(dataset_name: str, max_features: int, run_name: str):

    set_seed(42)

    train_df = load_split(dataset_name, "train")
    val_df = load_split(dataset_name, "val")
    test_df = load_split(dataset_name, "test")

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val = val_df["text"], val_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]


    # TF-IDF
    # ========================
    vectorizer = TfidfVectorizer(max_features=max_features)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)


    # MODEL
    # ========================
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)


    # METRICS
    # ========================
    avg = infer_average_mode(y_train.astype(int).tolist())

    val_metrics = classification_metrics(
        y_val.tolist(),
        model.predict(X_val_vec).tolist(),
        avg
    )

    test_metrics = classification_metrics(
        y_test.tolist(),
        model.predict(X_test_vec).tolist(),
        avg
    )


    # SHAP
    # ========================
    print("Running SHAP...")

    feature_names = vectorizer.get_feature_names_out()

    explainer = shap.LinearExplainer(
        model,
        X_train_vec
    )

    shap_values = explainer(X_test_vec[:100])


    # PATHS
    # ========================
    model_dir = ensure_dir(PROJECT_ROOT / "checkpoints" / run_name / dataset_name)
    table_dir = ensure_dir(PROJECT_ROOT / "results" / "tables")
    fig_dir = ensure_dir(PROJECT_ROOT / "results" / "figures")


    # SAVE MODEL
    # ========================
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")


    # SAVE SHAP BACKGROUND (for Streamlit)
    # ========================
    joblib.dump(X_train_vec[:200], model_dir / "shap_background.joblib")


    # SAVE METRICS
    # ========================
    write_json(test_metrics, table_dir / f"{run_name}_{dataset_name}_test.json")

    pd.DataFrame([test_metrics]).to_csv(
        table_dir / f"{run_name}_{dataset_name}_test.csv",
        index=False
    )

    pd.DataFrame([val_metrics]).to_csv(
        table_dir / f"{run_name}_{dataset_name}_val.csv",
        index=False
    )


    # SHAP PLOT (SHOW WORDS)
    # ========================
    shap.summary_plot(
        shap_values,
        features=X_test_vec[:100].toarray(),
        feature_names=feature_names,
        show=False
    )

    plt.savefig(
        fig_dir / f"{run_name}_{dataset_name}_shap.png",
        bbox_inches="tight"
    )
    plt.close()

    print(f"Training complete. Test metrics: {test_metrics}")



# ENTRY POINT
# ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imdb", choices=["imdb", "sst2", "emotion"])
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--run-name", default=RUN_NAME)

    args = parser.parse_args()

    train(args.dataset, args.max_features, args.run_name)


if __name__ == "__main__":
    main()