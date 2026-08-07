from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REQUIRED_FILES = ["games.parquet", "quarter_scores.parquet", "team_games.parquet", "last_update.json"]
CACHE_DIR = Path("/content/wnba-data-cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _download_raw_file(github_user: str, repository: str, branch: str, filename: str, timeout: int = 60) -> bytes:
    url = (
        "https://raw.githubusercontent.com/"
        f"{github_user}/{repository}/{branch}/{filename}"
    )
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        raise FileNotFoundError(f"Remote file not found: {filename}")
    response.raise_for_status()
    return response.content


def _load_parquet_bytes(content: bytes, expected_columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(io.BytesIO(content))
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def _ensure_utc(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def read_remote_parquet(filename: str, github_user: str, repository: str = "wnba-data", branch: str = "main", force_refresh: bool = False) -> bytes:
    cache_path = CACHE_DIR / filename
    if cache_path.exists() and not force_refresh:
        return cache_path.read_bytes()
    content = _download_raw_file(github_user, repository, branch, filename)
    cache_path.write_bytes(content)
    return content


def load_wnba_repository(github_user: str, repository: str = "wnba-data", branch: str = "main", force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    files = {}
    for filename in REQUIRED_FILES:
        content = read_remote_parquet(filename, github_user, repository, branch, force_refresh=force_refresh)
        if filename.endswith(".json"):
            files[filename] = json.loads(content.decode("utf-8"))
        else:
            files[filename] = content

    games = _load_parquet_bytes(files["games.parquet"], ["game_id", "game_date_utc"])
    quarter_scores = _load_parquet_bytes(files["quarter_scores.parquet"], ["game_id", "team_id"])
    team_games = _load_parquet_bytes(files["team_games.parquet"], ["game_id", "team_id"])

    games = _ensure_utc(games, ["game_date_utc", "updated_at_utc"])
    quarter_scores = _ensure_utc(quarter_scores, ["updated_at_utc"])
    team_games = _ensure_utc(team_games, ["game_date_utc", "updated_at_utc"])

    metadata = files["last_update.json"]
    print(
        f"Loaded repository {repository}@{branch}: "
        f"games={len(games)}, quarter_scores={len(quarter_scores)}, team_games={len(team_games)}"
    )
    return games, quarter_scores, team_games, metadata
