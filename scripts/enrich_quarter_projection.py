#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from src.config import PREDICTION_LATEST_JSON


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clock_seconds(clock) -> float:
    if clock is None:
        return 600.0
    text = str(clock).strip()
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return max(0.0, min(600.0, int(minutes) * 60 + float(seconds)))
        return max(0.0, min(600.0, float(text)))
    except (TypeError, ValueError):
        return 600.0


def _target_quarter(game: dict) -> int | None:
    status = str(game.get("live_status") or game.get("status") or "").upper()
    if "FINAL" in status:
        return None

    period = game.get("live_period")
    try:
        period = int(period) if period is not None else 0
    except (TypeError, ValueError):
        period = 0

    if "SCHEDULED" in status or period <= 0:
        return 1
    if "HALFTIME" in status:
        return 3
    if 1 <= period < 4:
        return period + 1
    return None


def _observed_quarter_pace(game: dict) -> tuple[float | None, float | None, float]:
    period = _num(game.get("live_period"), 0.0) or 0.0
    home = _num(game.get("live_home_score"))
    away = _num(game.get("live_away_score"))
    if period <= 0 or home is None or away is None:
        return None, None, 0.0

    clock = _clock_seconds(game.get("live_clock"))
    elapsed_current = (600.0 - clock) / 600.0
    elapsed_quarters = max(0.05, (period - 1.0) + elapsed_current)
    home_pace = home / elapsed_quarters
    away_pace = away / elapsed_quarters

    # Keep early possessions from overpowering the pregame quarter model.
    evidence = min(1.0, elapsed_quarters / 2.0)
    weight = 0.55 * evidence
    return home_pace, away_pace, weight


def enrich_game(game: dict, timestamp: str) -> dict:
    row = dict(game)
    target = _target_quarter(row)
    if target is None:
        for key in (
            "next_quarter_period",
            "next_quarter_home_score",
            "next_quarter_away_score",
            "next_quarter_margin",
            "next_quarter_total",
            "next_quarter_updated_at",
            "next_quarter_basis",
        ):
            row.pop(key, None)
        return row

    base_home = _num(row.get(f"home_q{target}"))
    base_away = _num(row.get(f"away_q{target}"))
    if base_home is None or base_away is None:
        return row

    status = str(row.get("live_status") or row.get("status") or "").upper()
    if "SCHEDULED" in status:
        home_projection = base_home
        away_projection = base_away
        basis = "pregame_quarter_model"
    else:
        home_pace, away_pace, weight = _observed_quarter_pace(row)
        if home_pace is None or away_pace is None:
            home_projection = base_home
            away_projection = base_away
            basis = "pregame_quarter_model"
        else:
            home_projection = (1.0 - weight) * base_home + weight * home_pace
            away_projection = (1.0 - weight) * base_away + weight * away_pace
            basis = "pregame_plus_live_scoring_pace"

    row["next_quarter_period"] = target
    row["next_quarter_home_score"] = home_projection
    row["next_quarter_away_score"] = away_projection
    row["next_quarter_margin"] = home_projection - away_projection
    row["next_quarter_total"] = home_projection + away_projection
    row["next_quarter_updated_at"] = timestamp
    row["next_quarter_basis"] = basis
    return row


def main() -> None:
    path = Path(PREDICTION_LATEST_JSON)
    if not path.exists():
        print("No predictions/latest.json; skipping quarter enrichment")
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamp = _utc_now()
    games = payload.get("games") or []
    payload["games"] = [enrich_game(game, timestamp) for game in games]
    payload["quarter_projection_updated_at"] = timestamp
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"games": len(games), "updated_at": timestamp}, indent=2))


if __name__ == "__main__":
    main()
