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
    PREDICTION_LATEST_CSV,
    PREDICTION_LATEST_JSON,
    QUARTER_EVENTS_PATH,
)
from .live_predict import project_live_game
from .parsing import parse_scoreboard_event
from .quarter_detector import detect_quarter_event

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_chicago_date() -> str:
    return pd.Timestamp.now(tz=CHICAGO).strftime("%Y%m%d")


def _load_live_state() -> dict[str, Any]:
    if not LIVE_STATE_PATH.exists():
        return {}
    try:
        with LIVE_STATE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.warning("Could not read live state: %s", exc)
        return {}


def _write_live_state(state: dict[str, Any]) -> None:
    LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _fetch_scoreboard(date_str: str) -> dict[str, Any]:
    response = requests.get(
        ESPN_SCOREBOARD_URL,
        params={"dates": date_str},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected ESPN scoreboard response")
    return payload


def _parse_live_games(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events")
    if not isinstance(events, list):
        return []

    results: list[dict[str, Any]] = []
    for event in events:
        game = parse_scoreboard_event(event)
        if not game:
            continue
        results.append(
            {
                "game_id": str(game["game_id"]),
                "home_team_id": str(game["home_team_id"]),
                "away_team_id": str(game["away_team_id"]),
                "status": game.get("status"),
                "status_detail": game.get("status_detail"),
                "period": game.get("period"),
                "clock": game.get("clock"),
                "home_score": game.get("home_score"),
                "away_score": game.get("away_score"),
            }
        )
    return results


def _snapshot_game(game: dict[str, Any]) -> dict[str, Any]:
    return {"timestamp": _utc_now_iso(), **game}


def _load_latest_predictions() -> dict[str, Any]:
    if not Path(PREDICTION_LATEST_JSON).exists():
        return {}
    try:
        with open(PREDICTION_LATEST_JSON, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.warning("Could not read latest predictions: %s", exc)
        return {}


def _is_live_or_final(status: Any) -> bool:
    text = str(status or "").upper()
    return any(
        marker in text
        for marker in (
            "IN_PROGRESS",
            "HALFTIME",
            "END_PERIOD",
            "FINAL",
        )
    )


def _update_latest_predictions_with_live_state(
    live_games: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Merge ESPN state and a continuously refreshed live model projection into
    predictions/latest.json while preserving every original pregame field.
    """
    latest = _load_latest_predictions()
    if not latest:
        return {
            "updated": 0,
            "reason": "prediction latest file does not exist",
        }

    prediction_games = latest.get("games")
    if not isinstance(prediction_games, list):
        return {
            "updated": 0,
            "reason": "latest prediction payload contains no games list",
        }

    live_lookup = {
        str(game["game_id"]): game
        for game in live_games
        if game.get("game_id") is not None
    }

    updated_count = 0
    projected_count = 0
    live_timestamp = _utc_now_iso()

    for prediction in prediction_games:
        game_id = str(prediction.get("game_id", ""))
        live = live_lookup.get(game_id)
        if not live:
            continue

        prediction["live_status"] = live.get("status")
        prediction["live_status_detail"] = live.get("status_detail")
        prediction["live_period"] = live.get("period")
        prediction["live_clock"] = live.get("clock")
        prediction["live_home_score"] = live.get("home_score")
        prediction["live_away_score"] = live.get("away_score")
        prediction["live_updated_at"] = live_timestamp

        if live.get("status") is not None:
            prediction["status"] = live["status"]
        if live.get("status_detail") is not None:
            prediction["status_detail"] = live["status_detail"]

        if _is_live_or_final(live.get("status")):
            projection = project_live_game(live, prediction)
            prediction["live_projected_home_score"] = projection["home_final"]
            prediction["live_projected_away_score"] = projection["away_final"]
            prediction["live_predicted_margin"] = projection["final_margin"]
            prediction["live_predicted_total"] = projection["final_total"]
            prediction["live_home_win_probability"] = projection[
                "home_win_probability"
            ]
            prediction["live_away_win_probability"] = projection[
                "away_win_probability"
            ]
            prediction["live_elapsed_fraction"] = projection[
                "elapsed_fraction"
            ]
            prediction["live_projection_updated_at"] = live_timestamp

            margin = projection["final_margin"]
            prediction["live_projected_winner_side"] = (
                "home" if margin > 0 else "away" if margin < 0 else "pickem"
            )
            prediction["live_projected_winner_abbr"] = (
                prediction.get("home_abbr")
                if margin > 0
                else prediction.get("away_abbr")
                if margin < 0
                else "PK"
            )
            projected_count += 1

        updated_count += 1

    if updated_count == 0:
        return {
            "updated": 0,
            "reason": "no live ESPN games matched prediction game ids",
        }

    latest["live_generated_at_utc"] = live_timestamp
    latest["last_live_update_utc"] = live_timestamp

    PREDICTION_LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_LATEST_JSON, "w", encoding="utf-8") as handle:
        json.dump(latest, handle, indent=2)

    PREDICTION_LATEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_games).to_csv(
        PREDICTION_LATEST_CSV,
        index=False,
    )

    return {
        "updated": updated_count,
        "live_projections_updated": projected_count,
        "live_updated_at": live_timestamp,
    }


def _append_live_snapshots(new_snapshots: list[dict[str, Any]]) -> int:
    LIVE_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_snapshots)

    if LIVE_SNAPSHOTS_PATH.exists():
        existing = pd.read_parquet(LIVE_SNAPSHOTS_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["timestamp", "game_id"],
            keep="last",
        )

    combined.to_parquet(LIVE_SNAPSHOTS_PATH, index=False)
    return len(combined)


def _append_quarter_events(quarter_events: list[dict[str, Any]]) -> int:
    if not quarter_events:
        if QUARTER_EVENTS_PATH.exists():
            try:
                return len(pd.read_parquet(QUARTER_EVENTS_PATH))
            except Exception:
                return 0
        return 0

    QUARTER_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_events = pd.DataFrame(quarter_events)

    if QUARTER_EVENTS_PATH.exists():
        existing_events = pd.read_parquet(QUARTER_EVENTS_PATH)
        combined_events = pd.concat(
            [existing_events, new_events],
            ignore_index=True,
        )
    else:
        combined_events = new_events

    combined_events = combined_events.drop_duplicates(
        subset=["game_id", "event_type"],
        keep="last",
    )
    combined_events.to_parquet(QUARTER_EVENTS_PATH, index=False)
    return len(combined_events)


def monitor_live_games() -> dict[str, Any]:
    today = _today_chicago_date()
    payload = _fetch_scoreboard(today)
    games = _parse_live_games(payload)
    previous_state = _load_live_state()

    snapshots: list[dict[str, Any]] = []
    state_updates: dict[str, Any] = {}
    quarter_events: list[dict[str, Any]] = []
    now_iso = _utc_now_iso()

    for game in games:
        game_id = str(game["game_id"])
        previous = previous_state.get(game_id)
        event = detect_quarter_event(previous, game)

        if event:
            quarter_events.append(
                {
                    "game_id": str(event.game_id),
                    "event_type": event.event_type,
                    "period": event.period,
                    "clock": event.clock,
                    "status": event.status,
                    "timestamp": now_iso,
                }
            )

        previous_event_type = (
            previous.get("last_event_type")
            if isinstance(previous, dict)
            else None
        )
        last_event_type = event.event_type if event else previous_event_type

        state_updates[game_id] = {
            "last_seen_period": game.get("period"),
            "last_seen_clock": game.get("clock"),
            "last_home_score": game.get("home_score"),
            "last_away_score": game.get("away_score"),
            "last_status": game.get("status"),
            "last_status_detail": game.get("status_detail"),
            "last_event_type": last_event_type,
            "last_prediction_timestamp": now_iso,
        }
        snapshots.append(_snapshot_game(game))

    merged_state = dict(previous_state)
    merged_state.update(state_updates)
    _write_live_state(merged_state)

    snapshot_rows = _append_live_snapshots(snapshots)
    quarter_event_rows = _append_quarter_events(quarter_events)
    prediction_update = _update_latest_predictions_with_live_state(games)

    return {
        "date": today,
        "games": games,
        "games_seen": len(games),
        "snapshot_rows": snapshot_rows,
        "quarter_events_detected": len(quarter_events),
        "quarter_event_rows": quarter_event_rows,
        "prediction_latest_update": prediction_update,
    }


if __name__ == "__main__":
    print(json.dumps(monitor_live_games(), indent=2))
