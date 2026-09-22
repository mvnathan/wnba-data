#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from src.draftkings_direct import fetch_draftkings_direct

MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
CHICAGO = ZoneInfo("America/Chicago")
OUT = Path("predictions/mlb-latest.json")
DOCS_OUT = Path("docs/mlb-latest.json")

# Conservative priors keep early-season/small-sample forecasts stable.
PRIOR_GAMES = 12.0
RECENT_HALF_LIFE_DAYS = 24.0
HOME_ADV_RUNS = 0.18
POISSON_MAX = 18


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    r = requests.get(url, params=params, timeout=30, headers={"user-agent": "SportsModelHub/1.0"})
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected MLB API response")
    return data


def _schedule(start: date, end: date) -> list[dict[str, Any]]:
    data = _get_json(
        MLB_SCHEDULE,
        {
            "sportId": 1,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "hydrate": "team,linescore",
        },
    )
    games: list[dict[str, Any]] = []
    for day in data.get("dates", []):
        games.extend(day.get("games", []))
    return games


def _team_name(team: dict[str, Any]) -> str:
    return str((team or {}).get("name") or "")


def _team_id(team: dict[str, Any]) -> int | None:
    try:
        return int((team or {}).get("id"))
    except (TypeError, ValueError):
        return None


def _score(game: dict[str, Any], side: str) -> int | None:
    try:
        value = game["teams"][side].get("score")
        return int(value) if value is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def _completed(game: dict[str, Any]) -> bool:
    status = game.get("status", {})
    return status.get("abstractGameState") == "Final" or status.get("codedGameState") in {"F", "O"}


def _weight(game_date: date, today: date) -> float:
    age = max(0, (today - game_date).days)
    return 0.5 ** (age / RECENT_HALF_LIFE_DAYS)


def _date_of_game(game: dict[str, Any]) -> date | None:
    try:
        return datetime.fromisoformat(str(game["gameDate"]).replace("Z", "+00:00")).astimezone(CHICAGO).date()
    except Exception:
        return None


def _league_and_team_rates(history: list[dict[str, Any]], today: date):
    # Weighted observed runs, split into offense and defense context.
    offense = defaultdict(lambda: [0.0, 0.0])
    defense = defaultdict(lambda: [0.0, 0.0])
    total_runs = 0.0
    total_team_games = 0.0
    home_margin_num = 0.0
    home_margin_den = 0.0

    for game in history:
        if not _completed(game):
            continue
        gd = _date_of_game(game)
        if gd is None:
            continue
        hs, aws = _score(game, "home"), _score(game, "away")
        hid = _team_id(game.get("teams", {}).get("home", {}).get("team", {}))
        aid = _team_id(game.get("teams", {}).get("away", {}).get("team", {}))
        if None in {hs, aws, hid, aid}:
            continue
        w = _weight(gd, today)
        offense[hid][0] += hs * w; offense[hid][1] += w
        defense[hid][0] += aws * w; defense[hid][1] += w
        offense[aid][0] += aws * w; offense[aid][1] += w
        defense[aid][0] += hs * w; defense[aid][1] += w
        total_runs += (hs + aws) * w
        total_team_games += 2 * w
        home_margin_num += (hs - aws) * w
        home_margin_den += w

    league = total_runs / total_team_games if total_team_games else 4.4
    observed_home_adv = home_margin_num / home_margin_den if home_margin_den else HOME_ADV_RUNS
    # Shrink home advantage toward a conservative baseball prior.
    home_adv = 0.5 * observed_home_adv + 0.5 * HOME_ADV_RUNS

    def shrunk(table, team_id: int) -> float:
        total, weight = table[team_id]
        return (total + PRIOR_GAMES * league) / (weight + PRIOR_GAMES)

    return league, home_adv, shrunk, offense, defense


def _poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _win_probability(home_lam: float, away_lam: float) -> float:
    hp = [_poisson_pmf(home_lam, i) for i in range(POISSON_MAX + 1)]
    ap = [_poisson_pmf(away_lam, i) for i in range(POISSON_MAX + 1)]
    home_win = 0.0
    tie = 0.0
    for h, ph in enumerate(hp):
        for a, pa in enumerate(ap):
            if h > a:
                home_win += ph * pa
            elif h == a:
                tie += ph * pa
    # MLB games cannot end tied; split modeled regulation ties evenly.
    return max(0.0, min(1.0, home_win + 0.5 * tie))


def _dk_market_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    try:
        rows = fetch_draftkings_direct("mlb")
    except Exception as exc:
        print(f"DraftKings MLB feed unavailable: {exc}")
        return {}
    return {
        (str(row.get("home_team", "")).lower(), str(row.get("away_team", "")).lower()): row
        for row in rows
    }


def _extract_dk(row: dict[str, Any] | None) -> dict[str, Any]:
    out = {
        "market_bookmaker": None,
        "market_home_moneyline": None,
        "market_away_moneyline": None,
        "market_run_line_home": None,
        "market_total": None,
        "market_updated_at": None,
    }
    if not row:
        return out
    books = row.get("bookmakers") or []
    if not books:
        return out
    book = books[0]
    out["market_bookmaker"] = book.get("title") or "DraftKings"
    out["market_updated_at"] = book.get("last_update")
    for market in book.get("markets", []):
        outcomes = market.get("outcomes") or []
        if market.get("key") == "h2h":
            for x in outcomes:
                if x.get("name") == row.get("home_team"):
                    out["market_home_moneyline"] = x.get("price")
                elif x.get("name") == row.get("away_team"):
                    out["market_away_moneyline"] = x.get("price")
        elif market.get("key") == "spreads":
            for x in outcomes:
                if x.get("name") == row.get("home_team"):
                    out["market_run_line_home"] = x.get("point")
        elif market.get("key") == "totals":
            over = next((x for x in outcomes if x.get("name") == "Over"), None)
            if over:
                out["market_total"] = over.get("point")
    return out


def build_predictions(target_date: date | None = None) -> dict[str, Any]:
    target_date = target_date or datetime.now(CHICAGO).date()
    history_start = target_date - timedelta(days=120)
    history = _schedule(history_start, target_date - timedelta(days=1))
    todays = _schedule(target_date, target_date)

    league, home_adv, shrunk, offense, defense = _league_and_team_rates(history, target_date)
    dk = _dk_market_lookup()

    games = []
    for game in todays:
        home_team = game.get("teams", {}).get("home", {}).get("team", {})
        away_team = game.get("teams", {}).get("away", {}).get("team", {})
        hid, aid = _team_id(home_team), _team_id(away_team)
        if hid is None or aid is None:
            continue
        home_name, away_name = _team_name(home_team), _team_name(away_team)

        home_off = shrunk(offense, hid)
        away_off = shrunk(offense, aid)
        home_def = shrunk(defense, hid)
        away_def = shrunk(defense, aid)

        # Blend team scoring and opponent prevention around league mean.
        home_runs = max(1.6, min(8.5, 0.52 * home_off + 0.48 * away_def + home_adv / 2))
        away_runs = max(1.6, min(8.5, 0.52 * away_off + 0.48 * home_def - home_adv / 2))
        home_prob = _win_probability(home_runs, away_runs)
        predicted_winner = home_name if home_prob >= 0.5 else away_name

        market_row = dk.get((home_name.lower(), away_name.lower()))
        market = _extract_dk(market_row)

        start = str(game.get("gameDate") or "")
        status = game.get("status", {})
        games.append({
            "game_id": str(game.get("gamePk") or ""),
            "game_date_utc": start,
            "home_team": home_name,
            "away_team": away_name,
            "status": status.get("detailedState"),
            "status_state": status.get("abstractGameState"),
            "predicted_home_runs": round(home_runs, 2),
            "predicted_away_runs": round(away_runs, 2),
            "predicted_total_runs": round(home_runs + away_runs, 2),
            "predicted_winner": predicted_winner,
            "home_win_probability": round(home_prob, 4),
            "away_win_probability": round(1 - home_prob, 4),
            "model_version": "mlb-runs-v1",
            "model_inputs": {
                "home_offense_runs_pg": round(home_off, 3),
                "away_offense_runs_pg": round(away_off, 3),
                "home_runs_allowed_pg": round(home_def, 3),
                "away_runs_allowed_pg": round(away_def, 3),
                "league_runs_per_team_game": round(league, 3),
            },
            "market_used_in_prediction": False,
            **market,
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date.isoformat(),
        "sport": "MLB",
        "model_version": "mlb-runs-v1",
        "model_description": (
            "Independent run-scoring model using exponentially weighted recent team "
            "offense and run prevention, shrinkage to league scoring, home advantage, "
            "and independent Poisson score distributions. Market data is display-only."
        ),
        "games": games,
    }


def main() -> None:
    payload = build_predictions()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False)
    OUT.write_text(text, encoding="utf-8")
    DOCS_OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
