"""Load and split the IMDb movie review dataset from HuggingFace."""

from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import load_dataset

from src.data.preprocess import normalize_dataframe


def load_imdb_splits() -> dict[str, pd.DataFrame]:
    """Return train/val/test DataFrames with 'text' and 'label' columns."""
    dataset: dict[str, Any] = load_dataset("imdb")
    train_df = pd.DataFrame(dataset["train"])[["text", "label"]]
    test_df = pd.DataFrame(dataset["test"])[["text", "label"]]

    # Create a validation split from train for consistent pipeline.
    val_df = train_df.sample(frac=0.1, random_state=42)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return {
        "train": normalize_dataframe(train_df),
        "val": normalize_dataframe(val_df),
        "test": normalize_dataframe(test_df),
    }
