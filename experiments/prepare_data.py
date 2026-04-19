"""Standalone data-preparation script for the Option-B LoRA package.

Downloads the three datasets used by the entire sweep from HuggingFace Hub and
writes them as CSVs to ``data/processed/``. Everything downstream
(``run_lora.py``, ``aggregate_efficiency.py``, the Streamlit tab) reads from
that directory, so this is the single prerequisite before the sweep.

Datasets written (same splits, same seed=42 everyone else uses):

    data/processed/imdb_{train,val,test}.csv       — binary sentiment
    data/processed/sst2_{train,val,test}.csv       — binary sentiment (GLUE)
    data/processed/emotion_{train,val,test}.csv    — 6-class emotion

Usage:

    python -m experiments.prepare_data
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from src.data.load_emotion import load_emotion_splits
from src.data.load_imdb import load_imdb_splits
from src.data.load_sst2 import load_sst2_splits
from src.data.split import compute_split_stats
from src.utils.io import ensure_dir, write_dataframe, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_MAPPINGS: dict[str, dict[str, str]] = {
    "imdb": {"0": "negative", "1": "positive"},
    "sst2": {"0": "negative", "1": "positive"},
    "emotion": {
        "0": "sadness",
        "1": "joy",
        "2": "love",
        "3": "anger",
        "4": "fear",
        "5": "surprise",
    },
}


def prepare_data() -> None:
    processed_dir = ensure_dir(PROJECT_ROOT / "data" / "processed")
    artifacts_dir = ensure_dir(PROJECT_ROOT / "data" / "artifacts")

    loaders = {
        "imdb": load_imdb_splits,
        "sst2": load_sst2_splits,
        "emotion": load_emotion_splits,
    }

    stats_payload: dict[str, dict[str, dict[str, float | int]]] = {}
    for dataset_name, loader in loaders.items():
        print(f"[prepare-data] loading {dataset_name} …")
        splits = loader()
        stats_payload[dataset_name] = {}
        for split_name, df in splits.items():
            out_csv = processed_dir / f"{dataset_name}_{split_name}.csv"
            write_dataframe(df, out_csv)
            stats_payload[dataset_name][split_name] = asdict(compute_split_stats(df))
            print(f"  wrote {out_csv.relative_to(PROJECT_ROOT)}  ({len(df):,} rows)")

    write_json(stats_payload, artifacts_dir / "stats.json")
    write_json(LABEL_MAPPINGS, artifacts_dir / "label_mapping.json")
    print("[prepare-data] done. Artifacts → data/artifacts/, splits → data/processed/")


if __name__ == "__main__":
    prepare_data()
