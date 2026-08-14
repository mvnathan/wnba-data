from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import (
    ALERT_STATE_PATH,
    DOCS_OPPORTUNITIES_JSON,
    OPPORTUNITY_ALERTS_PATH,
    PREDICTION_LATEST_JSON,
)

# Initial policy thresholds. These are deliberately explicit and centralized so
# they can later be calibrated from performance_history.parquet rather than
# remaining hand-tuned constants.
SPREAD_THRESHOLDS = {"moderate": 2.0, "strong": 4.0, "very_strong": 7.0}
TOTAL_THRESHOLDS = {"moderate": 3.0, "strong": 6.0, "very_strong": 10.0}
ALERT_LEVELS = {"strong", "very_strong"}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _level(edge: float, thresholds: dict[str, float]) -> str:
    magnitude = abs(edge)
    if magnitude >= thresholds["very_strong"]:
        return "very_strong"
    if magnitude >= thresholds["strong"]:
        return "strong"
    if magnitude >= thresholds["moderate"]:
        return "moderate"
    return "neutral"


def _is_live(game: dict[str, Any]) -> bool:
    status = str(game.get("live_status") or game.get("status") or "").upper()
    return "IN_PROGRESS" in status or "HALFTIME" in status


def _spread_opportunity(game: dict[str, Any], live: bool) -> dict[str, Any] | None:
    model_key = "live_predicted_margin" if live else "predicted_margin"
    model_margin = _number(game.get(model_key))
    market_home_spread = _number(game.get("market_home_spread"))
    if model_margin is None or market_home_spread is None:
        return None

    # Model margin is home minus away; a market home spread of -5 implies a
    # market-implied home margin of +5. The difference is therefore additive.
    edge = model_margin + market_home_spread
    level = _level(edge, SPREAD_THRESHOLDS)
    home = str(game.get("home_abbr") or "HOME")
    away = str(game.get("away_abbr") or "AWAY")
    side = home if edge > 0 else away

    return {
        "market": "spread",
        "level": level,
        "side": side,
        "edge": abs(edge),
        "signed_edge_home": edge,
        "model_margin": model_margin,
        "market_home_spread": market_home_spread,
    }


def _total_opportunity(game: dict[str, Any], live: bool) -> dict[str, Any] | None:
    model_key = "live_predicted_total" if live else "predicted_total"
    model_total = _number(game.get(model_key))
    market_total = _number(game.get("market_total"))
    if model_total is None or market_total is None:
        return None

    edge = model_total - market_total
    level = _level(edge, TOTAL_THRESHOLDS)
    return {
        "market": "total",
        "level": level,
        "side": "OVER" if edge > 0 else "UNDER",
        "edge": abs(edge),
        "signed_edge": edge,
        "model_total": model_total,
        "market_total": market_total,
    }


def _alert_id(game_id: str, phase: str, market: str, side: str, level: str) -> str:
    raw = f"{game_id}|{phase}|{market}|{side}|{level}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def evaluate_game(game: dict[str, Any]) -> list[dict[str, Any]]:
    live = _is_live(game)
    phase = "live" if live else "pregame"
    game_id = str(game.get("game_id") or "")
    if not game_id:
        return []

    candidates = [
        _spread_opportunity(game, live),
        _total_opportunity(game, live),
    ]
    alerts: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate or candidate["level"] not in ALERT_LEVELS:
            continue
        candidate.update(
            {
                "alert_id": _alert_id(
                    game_id,
                    phase,
                    candidate["market"],
                    candidate["side"],
                    candidate["level"],
                ),
                "game_id": game_id,
                "phase": phase,
                "away_abbr": game.get("away_abbr"),
                "home_abbr": game.get("home_abbr"),
                "game_date_utc": game.get("game_date_utc"),
                "status": game.get("live_status") or game.get("status"),
                "live_period": game.get("live_period"),
                "live_clock": game.get("live_clock"),
                "live_away_score": game.get("live_away_score"),
                "live_home_score": game.get("live_home_score"),
                "market_bookmaker": game.get("market_bookmaker"),
                "market_updated_at": game.get("market_updated_at"),
                "home_win_probability": game.get(
                    "live_home_win_probability" if live else "home_win_probability"
                ),
                "away_win_probability": game.get(
                    "live_away_win_probability" if live else "away_win_probability"
                ),
            }
        )
        alerts.append(candidate)
    return alerts


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def build_opportunity_alerts(
    latest_path: Path = PREDICTION_LATEST_JSON,
    state_path: Path = ALERT_STATE_PATH,
    output_path: Path = OPPORTUNITY_ALERTS_PATH,
    docs_path: Path = DOCS_OPPORTUNITIES_JSON,
) -> dict[str, Any]:
    latest = _load_json(latest_path, {})
    games = latest.get("games", []) if isinstance(latest, dict) else []
    now = datetime.now(timezone.utc).isoformat()

    state = _load_json(state_path, {"seen": {}})
    seen = state.get("seen", {}) if isinstance(state, dict) else {}
    if not isinstance(seen, dict):
        seen = {}

    active: list[dict[str, Any]] = []
    new_alerts: list[dict[str, Any]] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        for alert in evaluate_game(game):
            alert["evaluated_at_utc"] = now
            alert["is_new"] = alert["alert_id"] not in seen
            active.append(alert)
            if alert["is_new"]:
                new_alerts.append(alert)
                seen[alert["alert_id"]] = {
                    "first_seen_at_utc": now,
                    "game_id": alert["game_id"],
                    "phase": alert["phase"],
                    "market": alert["market"],
                    "side": alert["side"],
                    "level": alert["level"],
                }

    # Keep the state bounded. Game IDs are stable and old dedupe entries do not
    # need to live forever; retaining the newest 1000 is ample for this app.
    if len(seen) > 1000:
        ordered = sorted(
            seen.items(),
            key=lambda item: item[1].get("first_seen_at_utc", ""),
            reverse=True,
        )[:1000]
        seen = dict(ordered)

    payload = {
        "generated_at_utc": now,
        "source_generated_at_utc": latest.get("generated_at_utc") if isinstance(latest, dict) else None,
        "active_opportunities": active,
        "new_alerts": new_alerts,
        "summary": {
            "active_count": len(active),
            "new_count": len(new_alerts),
            "very_strong_count": sum(a["level"] == "very_strong" for a in active),
            "strong_count": sum(a["level"] == "strong" for a in active),
        },
        "policy": {
            "spread_thresholds": SPREAD_THRESHOLDS,
            "total_thresholds": TOTAL_THRESHOLDS,
            "alert_levels": sorted(ALERT_LEVELS),
            "calibration": "static_initial_policy",
        },
    }
    _write_json(output_path, payload)
    _write_json(docs_path, payload)
    _write_json(state_path, {"updated_at_utc": now, "seen": seen})
    return payload
