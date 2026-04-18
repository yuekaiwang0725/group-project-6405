"""Project entry point: data preparation (download, preprocess, save splits)."""

import argparse
from dataclasses import asdict
from pathlib import Path

from src.data.load_emotion import load_emotion_splits
from src.data.load_imdb import load_imdb_splits
from src.data.load_sst2 import load_sst2_splits
from src.data.split import compute_split_stats
from src.utils.io import ensure_dir, write_dataframe, write_json
from src.visualization.eda_plots import (
    plot_label_distribution,
    plot_text_length_distribution,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_LABEL_MAPPINGS: dict[str, dict[str, str]] = {
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
    processed_dir = PROJECT_ROOT / "data" / "processed"
    artifacts_dir = PROJECT_ROOT / "data" / "artifacts"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    ensure_dir(processed_dir)
    ensure_dir(artifacts_dir)
    ensure_dir(figures_dir)

    datasets = {
        "imdb": load_imdb_splits(),
        "sst2": load_sst2_splits(),
        "emotion": load_emotion_splits(),
    }

    stats_payload: dict[str, dict[str, dict[str, float | int]]] = {}
    for dataset_name, splits in datasets.items():
        stats_payload[dataset_name] = {}
        for split_name, df in splits.items():
            write_dataframe(df, processed_dir / f"{dataset_name}_{split_name}.csv")
            stats_payload[dataset_name][split_name] = asdict(compute_split_stats(df))

            if split_name == "train":
                plot_label_distribution(
                    df,
                    figures_dir / f"{dataset_name}_class_distribution.png",
                    title=f"{dataset_name.upper()} Label Distribution",
                )
                plot_text_length_distribution(
                    df,
                    figures_dir / f"{dataset_name}_text_length_distribution.png",
                    title=f"{dataset_name.upper()} Text Length Distribution",
                )

    write_json(stats_payload, artifacts_dir / "stats.json")
    write_json(DATASET_LABEL_MAPPINGS, artifacts_dir / "label_mapping.json")
    print("Data preparation completed. Files written to data/processed and data/artifacts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="EE6405 sentiment project entry point")
    parser.add_argument(
        "command",
        choices=["prepare-data"],
        help="Project command to execute",
    )
    args = parser.parse_args()

    if args.command == "prepare-data":
        prepare_data()


if __name__ == "__main__":
    main()
