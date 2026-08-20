#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import PREDICTION_LATEST_JSON
from src.market_odds import attach_market_odds, fetch_draftkings_wnba_odds

MARKET_FIELDS = (
    "market_bookmaker",
    "market_home_spread",
    "market_away_spread",
    "market_home_spread_price",
    "market_away_spread_price",
    "market_total",
    "market_over_price",
    "market_under_price",
    "market_home_moneyline",
    "market_away_moneyline",
    "market_updated_at",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _restore_model_baseline(game: dict) -> dict:
    row = dict(game)
    if row.get("model_predicted_margin") is not None:
        row["predicted_margin"] = row["model_predicted_margin"]
    if row.get("model_predicted_total") is not None:
        row["predicted_total"] = row["model_predicted_total"]
    if row.get("model_home_win_probability") is not None:
        p = float(row["model_home_win_probability"])
        row["home_win_probability"] = p
        row["away_win_probability"] = 1.0 - p
    return row


def main() -> None:
    path = Path(PREDICTION_LATEST_JSON)
    if not path.exists():
        print("No predictions/latest.json; skipping market refresh")
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    games = payload.get("games") or []
    if not games:
        print("No games in predictions/latest.json; skipping market refresh")
        return

    prior_updates = {
        str(g.get("game_id")): g.get("market_updated_at")
        for g in games
    }

    odds = fetch_draftkings_wnba_odds()
    refreshed = attach_market_odds(
        [_restore_model_baseline(g) for g in games],
        odds,
    )

    attempted_at = _utc_now()
    for game in refreshed:
        game_id = str(game.get("game_id"))
        previous = prior_updates.get(game_id)
        current = game.get("market_updated_at")
        game["market_refresh_attempted_at"] = attempted_at
        game["market_refresh_status"] = (
            "updated" if current and current != previous
            else "current_quote_returned" if current
            else "no_current_quote"
        )

    payload["games"] = refreshed
    payload["market_refresh_attempted_at"] = attempted_at
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"games": len(refreshed), "attempted_at": attempted_at}, indent=2))


if __name__ == "__main__":
    main()
