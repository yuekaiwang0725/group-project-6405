from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from transformers import pipeline

from src.models.baseline_svm import BaselineSVM


def predict_with_baseline(
    texts: list[str], vectorizer_path: str | Path, model_path: str | Path
) -> list[int]:
    vectorizer = joblib.load(vectorizer_path)
    model: BaselineSVM = joblib.load(model_path)
    features = vectorizer.transform(texts)
    return model.predict(features).tolist()


def predict_with_distilbert(texts: list[str], checkpoint_dir: str | Path) -> list[int]:
    clf = pipeline(
        "text-classification",
        model=str(checkpoint_dir),
        tokenizer=str(checkpoint_dir),
        truncation=True,
        max_length=128,
    )
    outputs = clf(texts)
    predictions: list[int] = []
    for result in outputs:
        label = str(result["label"]).lower()
        predictions.append(1 if label.endswith("1") else 0)
    return predictions


def load_texts_labels(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels
