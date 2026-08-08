from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import (
    CHICAGO,
    ESPN_SCOREBOARD_URL,
    LIVE_STATE_PATH,
    LIVE_SNAPSHOTS_PATH,
    QUARTER_EVENTS_PATH,
)
from .parsing import parse_scoreboard_event
from .quarter_detector import detect_quarter_event

logger = logging.getLogger(__name__)


def _today_chicago_date() -> str:
    return pd.Timestamp.now(tz=CHICAGO).strftime("%Y%m%d")


def _load_live_state() -> dict[str, Any]:
    if not LIVE_STATE_PATH.exists():
        return {}
    with LIVE_STATE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_live_state(state: dict[str, Any]) -> None:
    LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _fetch_scoreboard(date_str: str) -> dict[str, Any]:
    response = requests.get(ESPN_SCOREBOARD_URL, params={"dates": date_str}, timeout=20)
    response.raise_for_status()
    return response.json()


def _parse_live_games(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events")
    if not isinstance(events, list):
        return []

    results: list[dict[str, Any]] = []
    for event in events:
        game = parse_scoreboard_event(event)
        if not game:
            continue
        game_state: dict[str, Any] = {
            "game_id": game["game_id"],
            "home_team_id": game["home_team_id"],
            "away_team_id": game["away_team_id"],
            "status": game["status"],
            "status_detail": game.get("status_detail"),
            "period": game.get("period"),
            "clock": game.get("clock"),
            "home_score": game.get("home_score"),
            "away_score": game.get("away_score"),
        }
        results.append(game_state)
    return results


def _snapshot_game(game: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **game,
    }


def monitor_live_games() -> dict[str, Any]:
    today = _today_chicago_date()
    payload = _fetch_scoreboard(today)
    games = _parse_live_games(payload)
    state = _load_live_state()
    snapshots: list[dict[str, Any]] = []
    state_updates: dict[str, Any] = {}
    quarter_events: list[dict[str, Any]] = []

    for game in games:
        game_id = game["game_id"]
        previous = state.get(game_id)
        event = detect_quarter_event(previous, game)
        if event:
            quarter_events.append(
                {
                    "game_id": event.game_id,
                    "event_type": event.event_type,
                    "period": event.period,
                    "clock": event.clock,
                    "status": event.status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        last_event_type = event.event_type if event else (previous.get("last_event_type") if isinstance(previous, dict) else None)
        state_updates[game_id] = {
            "last_seen_period": game.get("period"),
            "last_seen_clock": game.get("clock"),
            "last_home_score": game.get("home_score"),
            "last_away_score": game.get("away_score"),
            "last_event_type": last_event_type,
            "last_prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        snapshots.append(_snapshot_game(game))

    _write_live_state(state_updates)
    LIVE_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LIVE_SNAPSHOTS_PATH.exists():
        existing = pd.read_parquet(LIVE_SNAPSHOTS_PATH)
        snapshots = pd.concat([existing, pd.DataFrame(snapshots)], ignore_index=True)
    else:
        snapshots = pd.DataFrame(snapshots)
    snapshots.to_parquet(LIVE_SNAPSHOTS_PATH, index=False)

    if quarter_events:
        QUARTER_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_events = pd.DataFrame(quarter_events)
        if QUARTER_EVENTS_PATH.exists():
            existing_events = pd.read_parquet(QUARTER_EVENTS_PATH)
            combined_events = pd.concat([existing_events, new_events], ignore_index=True)
        else:
            combined_events = new_events
        combined_events = combined_events.drop_duplicates(subset=["game_id", "event_type"], keep="last")
        combined_events.to_parquet(QUARTER_EVENTS_PATH, index=False)

    return {"games": games, "snapshot_rows": len(snapshots)}


if __name__ == "__main__":
    print(json.dumps(monitor_live_games(), indent=2))
