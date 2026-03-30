"""Unified batch prediction API for the Sentiment Dashboard.

Wraps existing SVM and DistilBERT models into a single interface that
the dashboard can call to analyse arbitrary lists of texts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.baseline_svm import BaselineSVM

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENT_LABELS = {0: "negative", 1: "positive"}
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# ---------------------------------------------------------------------------
# Model loading helpers (cached at module level for Streamlit)
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def _get_baseline(task: str = "imdb") -> tuple[Any, BaselineSVM] | None:
    key = f"baseline_{task}"
    if key not in _cache:
        vec_path = PROJECT_ROOT / "checkpoints" / "baseline" / task / "tfidf_vectorizer.joblib"
        mdl_path = PROJECT_ROOT / "checkpoints" / "baseline" / task / "svm_model.joblib"
        if not vec_path.exists() or not mdl_path.exists():
            _cache[key] = None
        else:
            _cache[key] = (joblib.load(vec_path), joblib.load(mdl_path))
    return _cache[key]


def _get_transformer(run_name: str, task: str) -> tuple[AutoTokenizer, AutoModelForSequenceClassification] | None:
    key = f"{run_name}_{task}"
    if key not in _cache:
        model_dir = PROJECT_ROOT / "checkpoints" / run_name / task / "best"
        if not model_dir.exists():
            model_dir = PROJECT_ROOT / "checkpoints" / run_name / task
        if not model_dir.exists():
            _cache[key] = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            model.eval()
            _cache[key] = (tokenizer, model)
    return _cache[key]


# ---------------------------------------------------------------------------
# Batch prediction functions
# ---------------------------------------------------------------------------


def _predict_baseline_batch(
    texts: list[str], task: str = "imdb"
) -> list[dict[str, Any]] | None:
    """Run SVM baseline on a batch of texts."""
    pair = _get_baseline(task)
    if pair is None:
        return None
    vectorizer, model = pair
    features = vectorizer.transform(texts)
    preds = model.predict(features).tolist()
    conf_raw = model.predict_confidence(features)

    results: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        pred = int(preds[i])
        if hasattr(conf_raw, "ndim") and conf_raw.ndim == 2:
            confidence = float(conf_raw[i][pred])
        else:
            confidence = float(conf_raw[i])
        results.append({"label": pred, "confidence": confidence})
    return results


def _predict_transformer_batch(
    texts: list[str],
    run_name: str = "distilbert",
    task: str = "imdb",
    batch_size: int = 32,
) -> list[dict[str, Any]] | None:
    """Run a transformer (DistilBERT / BERT) on a batch of texts."""
    pair = _get_transformer(run_name, task)
    if pair is None:
        return None
    tokenizer, model = pair

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    model = model.to(device)

    all_results: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        inputs = tokenizer(
            chunk,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for prob_row in probs:
            label = int(np.argmax(prob_row))
            confidence = float(prob_row[label])
            all_results.append({"label": label, "confidence": confidence})
    return all_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def batch_predict_sentiment(texts: list[str]) -> list[dict[str, Any]]:
    """Predict sentiment (pos/neg) for a list of texts using all available models.

    Returns a list of dicts, one per text:
    {
        "text": ...,
        "svm_label": "positive" / "negative",
        "svm_confidence": float,
        "distilbert_label": ...,
        "distilbert_confidence": float,
        "consensus": bool,
    }
    """
    svm_results = _predict_baseline_batch(texts, task="imdb")
    distilbert_results = _predict_transformer_batch(texts, run_name="distilbert", task="imdb")

    combined: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        row: dict[str, Any] = {"text": text}

        if svm_results is not None:
            row["svm_label"] = SENTIMENT_LABELS.get(svm_results[i]["label"], "unknown")
            row["svm_confidence"] = svm_results[i]["confidence"]
        else:
            row["svm_label"] = None
            row["svm_confidence"] = None

        if distilbert_results is not None:
            row["distilbert_label"] = SENTIMENT_LABELS.get(distilbert_results[i]["label"], "unknown")
            row["distilbert_confidence"] = distilbert_results[i]["confidence"]
        else:
            row["distilbert_label"] = None
            row["distilbert_confidence"] = None

        # Consensus
        if row["svm_label"] is not None and row["distilbert_label"] is not None:
            row["consensus"] = row["svm_label"] == row["distilbert_label"]
        else:
            row["consensus"] = None

        combined.append(row)
    return combined


def batch_predict_emotion(texts: list[str]) -> list[dict[str, Any]]:
    """Predict 6-class emotion for a list of texts.

    Returns a list of dicts:
    {
        "text": ...,
        "svm_emotion": "joy" / "sadness" / ...,
        "svm_confidence": float,
        "distilbert_emotion": ...,
        "distilbert_confidence": float,
    }
    """
    svm_results = _predict_baseline_batch(texts, task="emotion")
    distilbert_results = _predict_transformer_batch(texts, run_name="distilbert", task="emotion")

    combined: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        row: dict[str, Any] = {"text": text}

        if svm_results is not None:
            row["svm_emotion"] = EMOTION_LABELS.get(svm_results[i]["label"], "unknown")
            row["svm_confidence"] = svm_results[i]["confidence"]
        else:
            row["svm_emotion"] = None
            row["svm_confidence"] = None

        if distilbert_results is not None:
            row["distilbert_emotion"] = EMOTION_LABELS.get(distilbert_results[i]["label"], "unknown")
            row["distilbert_confidence"] = distilbert_results[i]["confidence"]
        else:
            row["distilbert_emotion"] = None
            row["distilbert_confidence"] = None

        combined.append(row)
    return combined
