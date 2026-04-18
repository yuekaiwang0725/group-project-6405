"""Build a TF-IDF vectorizer for the baseline SVM pipeline."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 30000, ngram_range: tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    """Create a TfidfVectorizer with unigrams + bigrams."""
    return TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
    )
