from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR


def load_parquet_or_empty(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
    df = pd.read_parquet(path, columns=columns)
    return df


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    required_columns = []
    if path.name == "games.parquet":
        required_columns = [
            "game_id",
            "game_date_utc",
            "season",
            "home_team_id",
            "away_team_id",
        ]
    if path.name == "quarter_scores.parquet":
        required_columns = ["game_id", "team_id"]
    if path.name == "team_games.parquet":
        required_columns = ["game_id", "team_id", "points"]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {path.name}: {missing}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet", dir=DATA_DIR) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        df.to_parquet(temp_path, compression="zstd", index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def upsert_dataframe(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    keys: list[str],
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    if existing.empty:
        combined = new.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    if sort_columns is not None:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
    else:
        combined = combined.reset_index(drop=True)
    return combined


def read_update_metadata() -> dict[str, Any]:
    metadata_path = DATA_DIR / "last_update.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_update_metadata(metadata: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = DATA_DIR / "last_update.json"
    with tempfile.NamedTemporaryFile("w", delete=False, dir=DATA_DIR, encoding="utf-8", suffix=".json") as handle:
        json.dump(metadata, handle, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(metadata_path)
