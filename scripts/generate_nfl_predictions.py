#!/usr/bin/env python3
"""Generate SportsModelHub NFL v1 predictions.

Independent pregame model:
- current-season team scoring / points allowed
- recency-weighted form with shrinkage toward league average
- modest home-field adjustment
- logistic win probability from projected scoring margin
- DraftKings lines are display-only and never model inputs
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from src.sbr_market_odds import fetch_sbr_draftkings

CHICAGO = ZoneInfo("America/Chicago")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
OUT = Path("predictions/nfl-latest.json")
DOCS_OUT = Path("docs/nfl-latest.json")
PRIOR_GAMES = 3.5
HOME_ADV_PRIOR = 1.7
PROB_SCALE = 7.25


def _get(params: dict[str, Any]) -> dict[str, Any]:
    r = requests.get(ESPN, params=params, timeout=30, headers={"user-agent": "SportsModelHub/1.0"})
    r.raise_for_status()
    return r.json()


def _week_payload(week: int, season: int, season_type: int = 2) -> dict[str, Any]:
    return _get({"week": week, "seasontype": season_type, "dates": season})


def _competition(event: dict[str, Any]) -> dict[str, Any]:
    comps = event.get("competitions") or []
    return comps[0] if comps else {}


def _teams(event: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    cs = _competition(event).get("competitors") or []
    home = next((x for x in cs if x.get("homeAway") == "home"), None)
    away = next((x for x in cs if x.get("homeAway") == "away"), None)
    return home, away


def _team_name(c: dict[str, Any] | None) -> str:
    return str(((c or {}).get("team") or {}).get("displayName") or "")


def _team_abbr(c: dict[str, Any] | None) -> str:
    return str(((c or {}).get("team") or {}).get("abbreviation") or "")


def _score(c: dict[str, Any] | None) -> float | None:
    try:
        return float((c or {}).get("score"))
    except (TypeError, ValueError):
        return None


def _complete(event: dict[str, Any]) -> bool:
    st = (event.get("status") or {}).get("type") or {}
    return bool(st.get("completed")) or str(st.get("state") or "").lower() == "post"


def _event_date(event: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(event.get("date")).replace("Z", "+00:00"))


def _current_week(season: int) -> int:
    # ESPN's undated scoreboard tracks the league's active week.
    data = _get({"dates": season, "seasontype": 2})
    try:
        return int((data.get("week") or {}).get("number") or 1)
    except (TypeError, ValueError):
        return 1


def _history_and_slate(season: int, week: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    for w in range(1, week + 1):
        try:
            events = _week_payload(w, season).get("events") or []
        except Exception:
            continue
        for e in events:
            if _complete(e):
                history.append(e)
    slate = [e for e in (_week_payload(week, season).get("events") or []) if not _complete(e)]
    return history, slate


def _rates(history: list[dict[str, Any]]) -> tuple[float, float, dict[str, float], dict[str, float], dict[str, int]]:
    now = datetime.now(timezone.utc)
    pf_sum = defaultdict(float)
    pa_sum = defaultdict(float)
    weight = defaultdict(float)
    games = defaultdict(int)
    league_pts = []
    home_edges = []

    for e in history:
        home, away = _teams(e)
        hs, aws = _score(home), _score(away)
        ha, aa = _team_abbr(home), _team_abbr(away)
        if hs is None or aws is None or not ha or not aa:
            continue
        league_pts.extend([hs, aws])
        home_edges.append(hs - aws)
        age_days = max(0.0, (now - _event_date(e)).total_seconds() / 86400)
        wt = math.exp(-math.log(2) * age_days / 28.0)
        pf_sum[ha] += hs * wt
        pa_sum[ha] += aws * wt
        weight[ha] += wt
        games[ha] += 1
        pf_sum[aa] += aws * wt
        pa_sum[aa] += hs * wt
        weight[aa] += wt
        games[aa] += 1

    league = sum(league_pts) / len(league_pts) if league_pts else 22.0
    raw_home = sum(home_edges) / len(home_edges) if home_edges else HOME_ADV_PRIOR
    home_adv = (raw_home * len(home_edges) + HOME_ADV_PRIOR * 32) / (len(home_edges) + 32)

    offense, defense = {}, {}
    for team in set(pf_sum) | set(pa_sum):
        w = weight[team]
        obs_pf = pf_sum[team] / w if w else league
        obs_pa = pa_sum[team] / w if w else league
        n = max(0.0, w)
        offense[team] = (obs_pf * n + league * PRIOR_GAMES) / (n + PRIOR_GAMES)
        defense[team] = (obs_pa * n + league * PRIOR_GAMES) / (n + PRIOR_GAMES)
    return league, home_adv, offense, defense, dict(games)


def _american_prob(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not x:
        return None
    return (-x) / ((-x) + 100.0) if x < 0 else 100.0 / (x + 100.0)


def _market_rows(date_strings: set[str]) -> list[dict[str, Any]]:
    rows = []
    for ds in sorted(date_strings):
        try:
            rows.extend(fetch_sbr_draftkings("nfl", ds))
        except Exception as exc:
            print(f"NFL market fetch failed for {ds}: {exc}")
    return rows


def _market_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("home_team") or "").lower(), str(r.get("away_team") or "").lower())
        out.setdefault(key, []).append(r)
    return out


def _closest(rows: list[dict[str, Any]] | None, start: str) -> dict[str, Any] | None:
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    try:
        target = datetime.fromisoformat(start.replace("Z", "+00:00"))
    except Exception:
        return rows[0]
    best, delta = rows[0], float("inf")
    for r in rows:
        try:
            stamp = datetime.fromisoformat(str(r.get("commence_time") or "").replace("Z", "+00:00"))
            d = abs((stamp - target).total_seconds())
            if d < delta:
                best, delta = r, d
        except Exception:
            pass
    return best


def _extract_market(row: dict[str, Any] | None) -> dict[str, Any]:
    out = {
        "market_bookmaker": None,
        "market_home_moneyline": None,
        "market_away_moneyline": None,
        "market_home_spread": None,
        "market_home_spread_price": None,
        "market_away_spread_price": None,
        "market_total": None,
        "market_over_price": None,
        "market_under_price": None,
        "market_updated_at": None,
    }
    if not row:
        return out
    books = row.get("bookmakers") or []
    if not books:
        return out
    b = books[0]
    out["market_bookmaker"] = b.get("title") or "DraftKings"
    out["market_updated_at"] = b.get("last_update")
    for market in b.get("markets") or []:
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
                    out["market_home_spread"] = x.get("point")
                    out["market_home_spread_price"] = x.get("price")
                elif x.get("name") == row.get("away_team"):
                    out["market_away_spread_price"] = x.get("price")
        elif market.get("key") == "totals":
            for x in outcomes:
                if x.get("name") == "Over":
                    out["market_total"] = x.get("point")
                    out["market_over_price"] = x.get("price")
                elif x.get("name") == "Under":
                    if out["market_total"] is None:
                        out["market_total"] = x.get("point")
                    out["market_under_price"] = x.get("price")
    return out


def build() -> dict[str, Any]:
    now = datetime.now(CHICAGO)
    season = now.year
    week = _current_week(season)
    history, slate = _history_and_slate(season, week)

    # If the active week has no remaining games, advance one week.
    if not slate and week < 18:
        week += 1
        _, slate = _history_and_slate(season, week)

    league, home_adv, offense, defense, counts = _rates(history)
    market_dates = {_event_date(e).astimezone(CHICAGO).date().isoformat() for e in slate}
    markets = _market_lookup(_market_rows(market_dates))

    games = []
    for e in slate:
        home, away = _teams(e)
        hn, an = _team_name(home), _team_name(away)
        ha, aa = _team_abbr(home), _team_abbr(away)
        if not hn or not an or not ha or not aa:
            continue

        home_pf = offense.get(ha, league)
        away_pf = offense.get(aa, league)
        home_pa = defense.get(ha, league)
        away_pa = defense.get(aa, league)

        ph = 0.53 * home_pf + 0.47 * away_pa + home_adv / 2
        pa = 0.53 * away_pf + 0.47 * home_pa - home_adv / 2
        ph = max(10.0, min(39.5, ph))
        pa = max(10.0, min(39.5, pa))
        margin = ph - pa
        total = ph + pa
        hp = 1 / (1 + math.exp(-margin / PROB_SCALE))

        market = _extract_market(_closest(markets.get((hn.lower(), an.lower())), str(e.get("date") or "")))
        ml_prob = _american_prob(market.get("market_home_moneyline"))
        spread_edge = None
        total_edge = None
        ml_edge = None
        if market.get("market_home_spread") is not None:
            spread_edge = margin + float(market["market_home_spread"])
        if market.get("market_total") is not None:
            total_edge = total - float(market["market_total"])
        if ml_prob is not None:
            ml_edge = hp - ml_prob

        comp = _competition(e)
        status = (e.get("status") or {}).get("type") or {}
        games.append({
            "game_id": str(e.get("id") or comp.get("id") or ""),
            "game_date_utc": e.get("date"),
            "season": season,
            "week": week,
            "home_team": hn,
            "away_team": an,
            "home_abbr": ha,
            "away_abbr": aa,
            "venue": (comp.get("venue") or {}).get("fullName"),
            "status": status.get("description") or status.get("detail"),
            "status_state": status.get("state"),
            "predicted_home_points": round(ph, 1),
            "predicted_away_points": round(pa, 1),
            "predicted_margin": round(margin, 1),
            "predicted_total": round(total, 1),
            "predicted_winner": hn if hp >= .5 else an,
            "home_win_probability": round(hp, 4),
            "away_win_probability": round(1-hp, 4),
            "team_games_home": counts.get(ha, 0),
            "team_games_away": counts.get(aa, 0),
            "model_version": "nfl-v1-beta",
            "market_used_in_prediction": False,
            **market,
            "model_market_spread_edge": round(spread_edge, 2) if spread_edge is not None else None,
            "model_market_total_edge": round(total_edge, 2) if total_edge is not None else None,
            "model_market_home_win_edge": round(ml_edge, 4) if ml_edge is not None else None,
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_date": now.date().isoformat(),
        "season": season,
        "week": week,
        "model_version": "nfl-v1-beta",
        "model_status": "beta",
        "market_used_in_prediction": False,
        "market_event_count": sum(1 for g in games if g.get("market_bookmaker")),
        "games": games,
        "methodology": {
            "history": "Current-season completed games only",
            "recency": "28-day exponential half-life",
            "shrinkage_games": PRIOR_GAMES,
            "home_field_points": round(home_adv, 3),
            "league_points_per_team": round(league, 3),
            "probability_scale": PROB_SCALE,
        },
    }


def main() -> None:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False)
    OUT.write_text(text)
    DOCS_OUT.write_text(text)
    print(json.dumps({
        "model": payload["model_version"],
        "status": payload["model_status"],
        "week": payload["week"],
        "games": len(payload["games"]),
        "markets": payload["market_event_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
