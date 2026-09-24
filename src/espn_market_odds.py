from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

ESPN_SCOREBOARD = {
    "wnba": "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard",
    "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("o", "").replace("u", "").replace("+", "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _american(value: Any) -> int | None:
    n = _num(value)
    return int(n) if n is not None else None


def fetch_espn_draftkings(sport: str) -> list[dict[str, Any]]:
    """Return DraftKings markets from ESPN's public scoreboard feed.

    ESPN has integrated DraftKings as its sportsbook provider. This parser
    normalizes ESPN's game odds into the same event/bookmaker/market shape used
    elsewhere in SportsModelHub.
    """
    sport = sport.lower()
    url = ESPN_SCOREBOARD.get(sport)
    if not url:
        raise ValueError(f"Unsupported ESPN odds sport: {sport}")

    r = requests.get(url, timeout=25, headers={"user-agent": "SportsModelHub/1.0"})
    r.raise_for_status()
    payload = r.json()

    rows: list[dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for event in payload.get("events", []) or []:
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home = next((x for x in competitors if x.get("homeAway") == "home"), None)
        away = next((x for x in competitors if x.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_team = (home.get("team") or {}).get("displayName") or (home.get("team") or {}).get("name")
        away_team = (away.get("team") or {}).get("displayName") or (away.get("team") or {}).get("name")
        if not home_team or not away_team:
            continue

        odds_list = comp.get("odds") or []
        odds = next(
            (
                o for o in odds_list
                if "draftkings" in str((o.get("provider") or {}).get("name") or "").lower()
                or "draftkings" in str((o.get("provider") or {}).get("displayName") or "").lower()
            ),
            odds_list[0] if odds_list else None,
        )
        if not odds:
            continue

        markets: list[dict[str, Any]] = []

        ml = odds.get("moneyline") or {}
        ml_home = (((ml.get("home") or {}).get("close") or {}).get("odds"))
        ml_away = (((ml.get("away") or {}).get("close") or {}).get("odds"))
        if ml_home is not None or ml_away is not None:
            markets.append({
                "key": "h2h",
                "outcomes": [
                    {"name": home_team, "price": _american(ml_home)},
                    {"name": away_team, "price": _american(ml_away)},
                ],
            })

        ps = odds.get("pointSpread") or {}
        ps_home = (ps.get("home") or {}).get("close") or {}
        ps_away = (ps.get("away") or {}).get("close") or {}
        if ps_home or ps_away or odds.get("spread") is not None:
            home_line = _num(ps_home.get("line"))
            away_line = _num(ps_away.get("line"))
            if home_line is None and odds.get("spread") is not None:
                home_line = _num(odds.get("spread"))
            if away_line is None and home_line is not None:
                away_line = -home_line
            markets.append({
                "key": "spreads",
                "outcomes": [
                    {"name": home_team, "point": home_line, "price": _american(ps_home.get("odds"))},
                    {"name": away_team, "point": away_line, "price": _american(ps_away.get("odds"))},
                ],
            })

        total = odds.get("total") or {}
        over = (total.get("over") or {}).get("close") or {}
        under = (total.get("under") or {}).get("close") or {}
        line = _num(over.get("line"))
        if line is None:
            line = _num(under.get("line"))
        if line is None:
            line = _num(odds.get("overUnder"))
        if line is not None:
            markets.append({
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": line, "price": _american(over.get("odds"))},
                    {"name": "Under", "point": line, "price": _american(under.get("odds"))},
                ],
            })

        if not markets:
            continue

        rows.append({
            "id": str(event.get("id") or comp.get("id") or ""),
            "sport_key": {"wnba": "basketball_wnba", "mlb": "baseball_mlb", "nfl": "americanfootball_nfl"}[sport],
            "commence_time": event.get("date") or comp.get("date"),
            "home_team": home_team,
            "away_team": away_team,
            "bookmakers": [{
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": fetched_at,
                "markets": markets,
            }],
            "market_provider": "espn_draftkings",
        })

    return rows
