from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import load_dataset

from src.data.preprocess import normalize_dataframe


def load_sst2_splits() -> dict[str, pd.DataFrame]:
    dataset: dict[str, Any] = load_dataset("glue", "sst2")
    train_df = pd.DataFrame(dataset["train"])[["sentence", "label"]]
    # SST-2 official test set has no public labels, so we split
    # the validation set 50/50 into val and test to avoid data leakage.
    full_val = pd.DataFrame(dataset["validation"])[["sentence", "label"]]
    val_df = full_val.sample(frac=0.5, random_state=42)
    test_df = full_val.drop(val_df.index)

    rename_map = {"sentence": "text"}
    train_df = train_df.rename(columns=rename_map)
    val_df = val_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    return {
        "train": normalize_dataframe(train_df),
        "val": normalize_dataframe(val_df),
        "test": normalize_dataframe(test_df),
    }
