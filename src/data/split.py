"""Compute basic statistics (row count, label ratio, text length) for a data split."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitStats:
    rows: int
    positive_ratio: float
    avg_length_chars: float
    avg_length_words: float


def compute_split_stats(df: pd.DataFrame) -> SplitStats:
    length_chars = df["text"].str.len()
    length_words = df["text"].str.split().map(len)
    return SplitStats(
        rows=int(len(df)),
        positive_ratio=float(df["label"].mean()),
        avg_length_chars=float(length_chars.mean()),
        avg_length_words=float(length_words.mean()),
    )
