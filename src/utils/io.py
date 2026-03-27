from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=True)


def read_json(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    df.to_csv(path_obj, index=False)
