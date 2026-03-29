from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import load_dataset

from src.data.preprocess import normalize_dataframe


def load_emotion_splits() -> dict[str, pd.DataFrame]:
    dataset: dict[str, Any] = load_dataset("dair-ai/emotion")
    train_df = pd.DataFrame(dataset["train"])[["text", "label"]]
    val_df = pd.DataFrame(dataset["validation"])[["text", "label"]]
    test_df = pd.DataFrame(dataset["test"])[["text", "label"]]

    return {
        "train": normalize_dataframe(train_df),
        "val": normalize_dataframe(val_df),
        "test": normalize_dataframe(test_df),
    }
