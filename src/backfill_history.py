from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    BACKFILL_PROGRESS_PATH,
    GAMES_PATH,
    HISTORICAL_SEASON_END,
    HISTORICAL_SEASON_START,
    QUARTER_SCORES_PATH,
    SEASONS_RETAINED,
    TEAM_GAMES_PATH,
)
from .dataset_builder import build_team_games
from .espn_api import ESPNApiClient
from .parsing import parse_quarter_scores, parse_scoreboard_event
from .storage import (
    load_parquet_or_empty,
    write_parquet_atomic,
    write_update_metadata,
)

logger = logging.getLogger(__name__)


def _date_range(start_date: date, end_date: date) -> list[date]:
    result: list[date] = []
    current = start_date
    while current <= end_date:
        result.append(current)
        current += timedelta(days=1)
    return result


def _season_dates(season: int) -> list[date]:
    start = date(season, HISTORICAL_SEASON_START[0], HISTORICAL_SEASON_START[1])
    end = date(season, HISTORICAL_SEASON_END[0], HISTORICAL_SEASON_END[1])
    return _date_range(start, end)


def _load_progress() -> dict[str, Any]:
    if not BACKFILL_PROGRESS_PATH.exists():
        return {}
    with BACKFILL_PROGRESS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_progress(progress: dict[str, Any]) -> None:
    with BACKFILL_PROGRESS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2)


def _trim_to_selected_seasons(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    if df.empty:
        return df
    if "season" not in df.columns:
        return df
    return df[df["season"].isin(seasons)].copy()


def run_backfill(seasons: list[int], force: bool = False) -> dict[str, object]:
    if not seasons:
        raise ValueError("At least one season must be provided for backfill")

    client = ESPNApiClient()
    games_df = pd.DataFrame()
    quarter_df = pd.DataFrame()
    progress = _load_progress() if not force else {}
    progress.setdefault("processed_dates", {})

    for season in sorted(seasons):
        for scoreboard_date in _season_dates(season):
            key = f"{season}-{scoreboard_date.isoformat()}"
            if not force and progress["processed_dates"].get(key):
                continue
            try:
                payload = client.fetch_scoreboard(scoreboard_date)
                events = payload.get("events")
                if not isinstance(events, list):
                    raise ValueError("Scoreboard response is missing events list")
                parsed_games = [parsed for event in events if (parsed := parse_scoreboard_event(event))]
                if parsed_games:
                    games_df = pd.concat([games_df, pd.DataFrame(parsed_games)], ignore_index=True)

                completed_ids = [row["game_id"] for row in parsed_games if row["completed"]]
                for game_id in completed_ids:
                    summary_payload = client.fetch_summary(game_id)
                    quarter_df = pd.concat(
                        [quarter_df, pd.DataFrame(parse_quarter_scores(summary_payload, game_id))],
                        ignore_index=True,
                    )
                progress["processed_dates"][key] = {
                    "date": scoreboard_date.isoformat(),
                    "season": season,
                    "games": len(parsed_games),
                }
                _write_progress(progress)
            except Exception as exc:
                logger.warning("Backfill date failed %s: %s", scoreboard_date.isoformat(), exc)
                progress["processed_dates"][key] = {
                    "date": scoreboard_date.isoformat(),
                    "season": season,
                    "error": str(exc),
                }
                _write_progress(progress)

    if games_df.empty:
        raise RuntimeError("Backfill did not retrieve any games")

    games_df = games_df.drop_duplicates(subset=["game_id"], keep="last")
    quarter_df = quarter_df.drop_duplicates(subset=["game_id", "team_id"], keep="last")
    games_df = _trim_to_selected_seasons(games_df, seasons)
    retained_game_ids = set(games_df["game_id"]) if not games_df.empty else set()
    quarter_df = quarter_df[quarter_df["game_id"].isin(retained_game_ids)].copy()
    team_df = build_team_games(games_df, quarter_df)
    team_df = _trim_to_selected_seasons(team_df, seasons)

    write_parquet_atomic(games_df, GAMES_PATH)
    write_parquet_atomic(quarter_df, QUARTER_SCORES_PATH)
    write_parquet_atomic(team_df, TEAM_GAMES_PATH)

    metadata = {
        "updated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "min_season": int(min(seasons)),
        "max_season": int(max(seasons)),
        "games_rows": len(games_df),
        "completed_games": int(games_df[games_df["completed"]].shape[0]),
        "quarter_rows": len(quarter_df),
        "team_game_rows": len(team_df),
        "source": "ESPN public endpoints",
        "schema_version": "1.0.0",
    }
    write_update_metadata(metadata)
    return {
        "status": "ok",
        "season_count": len(seasons),
        "games_rows": len(games_df),
        "quarter_rows": len(quarter_df),
        "team_game_rows": len(team_df),
        "metadata": metadata,
    }
