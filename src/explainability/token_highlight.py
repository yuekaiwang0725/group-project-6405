"""Explainability: compute per-token contribution scores from SVM coefficients."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.baseline_svm import BaselineSVM


def top_token_contributions(
    text: str,
    vectorizer: TfidfVectorizer,
    model: BaselineSVM,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-k tokens ranked by |tfidf_weight × svm_coefficient|."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    row = vectorizer.transform([text])
    if row.nnz == 0:
        return []

    active_indices = row.indices
    active_values = row.data
    weights = model.model.coef_.ravel()[active_indices]
    contributions = active_values * weights

    top_idx = np.argsort(np.abs(contributions))[::-1][:top_k]
    return [
        (feature_names[active_indices[i]], float(contributions[i]))
        for i in top_idx
    ]


def simple_word_salience(text: str, top_k: int = 10) -> list[tuple[str, int]]:
    tokens = [token.lower() for token in text.split() if token.isalpha()]
    return Counter(tokens).most_common(top_k)
