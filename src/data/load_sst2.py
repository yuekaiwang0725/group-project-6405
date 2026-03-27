from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import load_dataset

from src.data.preprocess import normalize_dataframe


def load_sst2_splits() -> dict[str, pd.DataFrame]:
    dataset: dict[str, Any] = load_dataset("glue", "sst2")
    train_df = pd.DataFrame(dataset["train"])[["sentence", "label"]]
    val_df = pd.DataFrame(dataset["validation"])[["sentence", "label"]]
    test_df = pd.DataFrame(dataset["validation"])[["sentence", "label"]]

    rename_map = {"sentence": "text"}
    train_df = train_df.rename(columns=rename_map)
    val_df = val_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    return {
        "train": normalize_dataframe(train_df),
        "val": normalize_dataframe(val_df),
        "test": normalize_dataframe(test_df),
    }
