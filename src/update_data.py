from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .config import (
    DAILY_QUERY_OFFSETS,
    GAMES_PATH,
    QUARTER_SCORES_PATH,
    SEASONS_RETAINED,
    TEAM_GAMES_PATH,
)
from .dataset_builder import build_team_games
from .espn_api import ESPNApiClient
from .parsing import parse_quarter_scores, parse_scoreboard_event
from .storage import (
    load_parquet_or_empty,
    upsert_dataframe,
    write_parquet_atomic,
    write_update_metadata,
)

logger = logging.getLogger(__name__)


def _collect_scoreboard_dates() -> list[date]:
    today = date.today()
    return [today + timedelta(days=offset) for offset in DAILY_QUERY_OFFSETS]


def _trim_to_recent_seasons(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "season" not in df.columns:
        return df
    seasons = sorted(df["season"].dropna().unique().astype(int))
    if not seasons:
        return df
    retained = seasons[-SEASONS_RETAINED :]
    return df[df["season"].isin(retained)].copy()


def _trim_quarters_to_game_ids(quarters: pd.DataFrame, game_ids: set[str]) -> pd.DataFrame:
    if quarters.empty:
        return quarters
    return quarters[quarters["game_id"].isin(game_ids)].copy()


def _games_missing_quarters(games: pd.DataFrame, quarters: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["game_id"])
    if quarters.empty:
        return games[games["completed"]].copy()
    completed = games[games["completed"]].copy()
    present = quarters[quarters["team_id"].notna()]["game_id"].unique()
    return completed[~completed["game_id"].isin(present)].copy()


def run_daily_update() -> dict[str, object]:
    client = ESPNApiClient()
    games_df = load_parquet_or_empty(GAMES_PATH)
    quarter_df = load_parquet_or_empty(QUARTER_SCORES_PATH)

    scores = []
    errors: dict[date, str] = {}
    for scoreboard_date in _collect_scoreboard_dates():
        try:
            payload = client.fetch_scoreboard(scoreboard_date)
            events = payload.get("events")
            if not isinstance(events, list):
                raise ValueError("Scoreboard response is missing events list")
            for event in events:
                parsed = parse_scoreboard_event(event)
                if parsed:
                    scores.append(parsed)
        except Exception as exc:
            errors[scoreboard_date] = str(exc)

    if not scores and errors:
        raise RuntimeError("Daily update could not retrieve any scoreboard dates")

    games_update = pd.DataFrame(scores)
    if not games_update.empty:
        games_df = upsert_dataframe(
            existing=games_df,
            new=games_update,
            keys=["game_id"],
            sort_columns=["game_date_utc", "game_id"],
        )
    missing_quarters = _games_missing_quarters(games_df, quarter_df)
    missing_game_ids = missing_quarters["game_id"].unique().tolist()
    summary_errors: list[str] = []
    summary_rows = []
    for game_id in missing_game_ids:
        try:
            summary_payload = client.fetch_summary(game_id)
            summary_rows.extend(parse_quarter_scores(summary_payload, game_id))
        except Exception as exc:
            summary_errors.append(f"{game_id}: {exc}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        quarter_df = upsert_dataframe(
            existing=quarter_df,
            new=summary_df,
            keys=["game_id", "team_id"],
            sort_columns=["game_id", "team_id"],
        )

    games_df = _trim_to_recent_seasons(games_df)
    retained_game_ids = set(games_df["game_id"]) if not games_df.empty else set()
    quarter_df = _trim_quarters_to_game_ids(quarter_df, retained_game_ids)
    team_df = build_team_games(games_df, quarter_df)
    team_df = _trim_to_recent_seasons(team_df)

    write_parquet_atomic(games_df, GAMES_PATH)
    write_parquet_atomic(quarter_df, QUARTER_SCORES_PATH)
    write_parquet_atomic(team_df, TEAM_GAMES_PATH)

    latest_game_date = None
    if not games_df.empty:
        latest_game_date = games_df["game_date_utc"].max()
        if isinstance(latest_game_date, pd.Timestamp):
            latest_game_date = latest_game_date.to_pydatetime().isoformat()

    metadata = {
        "updated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "min_season": int(games_df["season"].min()) if not games_df.empty else None,
        "max_season": int(games_df["season"].max()) if not games_df.empty else None,
        "games_rows": len(games_df),
        "completed_games": int(games_df[games_df["completed"]].shape[0]) if not games_df.empty else 0,
        "quarter_rows": len(quarter_df),
        "team_game_rows": len(team_df),
        "missing_completed_quarter_games": len(missing_game_ids),
        "latest_game_date": latest_game_date,
        "source": "ESPN public endpoints",
        "schema_version": "1.0.0",
    }
    write_update_metadata(metadata)

    if summary_errors:
        logger.warning("Some summaries failed: %s", summary_errors)

    return {
        "status": "ok",
        "updated_at_utc": metadata["updated_at_utc"],
        "games_rows": metadata["games_rows"],
        "quarter_rows": metadata["quarter_rows"],
        "team_game_rows": metadata["team_game_rows"],
        "errors": errors,
        "summary_errors": summary_errors,
    }
