from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 30000, ngram_range: tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
    )
