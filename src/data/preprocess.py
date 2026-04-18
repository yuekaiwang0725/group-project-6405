"""Text preprocessing utilities: HTML removal, whitespace normalization."""

from __future__ import annotations

import re

import pandas as pd

WHITESPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["text"] = normalized["text"].astype(str).map(clean_text)
    normalized["label"] = normalized["label"].astype(int)
    normalized = normalized[normalized["text"].str.len() > 0].reset_index(drop=True)
    return normalized
