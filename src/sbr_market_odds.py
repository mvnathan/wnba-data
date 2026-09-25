from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import requests

SLUGS = {
    "wnba": "wnba-basketball",
    "mlb": "mlb-baseball",
    "nfl": "nfl-football",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sportsbookreview.com/",
}


def _next_data(url: str) -> dict[str, Any]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    m = re.search(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', r.text, re.S)
    if not m:
        raise RuntimeError("SportsBookReview __NEXT_DATA__ not found")
    return json.loads(m.group(1))


def _rows(url: str) -> list[dict[str, Any]]:
    data = _next_data(url)
    props = data.get("props", {}).get("pageProps", {})
    tables = props.get("oddsTables") or []
    if not tables:
        return []
    model = tables[0].get("oddsTableModel") or {}
    sportsbooks = model.get("sportsbooks") or []
    rows = model.get("gameRows") or []
    for row in rows:
        if isinstance(row, dict):
            row["_sportsbooks"] = sportsbooks
    return rows


def _book_line(row: dict[str, Any], book: str = "draftkings") -> dict[str, Any] | None:
    views = row.get("oddsViews") or []
    # Some SBR pages put the book name on each odds view.
    for view in views:
        if not isinstance(view, dict):
            continue
        if str(view.get("sportsbook") or "").lower() != book:
            continue
        line = view.get("currentLine") or view.get("openingLine") or {}
        if isinstance(line, dict):
            return line

    # Other pages use a table-level sportsbook array whose position matches
    # the oddsViews array. NFL currently uses this representation.
    sportsbooks = row.get("_sportsbooks") or []
    for idx, sb in enumerate(sportsbooks):
        machine = str((sb or {}).get("machineName") or (sb or {}).get("name") or "").lower()
        if machine != book:
            continue
        if idx >= len(views) or not isinstance(views[idx], dict):
            return None
        line = views[idx].get("currentLine") or views[idx].get("openingLine") or {}
        return line if isinstance(line, dict) else None
    return None


def _key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    gv = row.get("gameView") or {}
    home = ((gv.get("homeTeam") or {}).get("fullName") or "").strip()
    away = ((gv.get("awayTeam") or {}).get("fullName") or "").strip()
    start = str(gv.get("startDate") or "")
    return (home, away, start) if home and away else None


def fetch_sbr_draftkings(sport: str, date_str: str | None = None) -> list[dict[str, Any]]:
    """Fetch current DraftKings game lines from SportsBookReview's public odds pages."""
    sport = sport.lower()
    slug = SLUGS.get(sport)
    if not slug:
        raise ValueError(f"Unsupported SBR sport: {sport}")
    if not date_str:
        date_str = datetime.now(timezone.utc).date().isoformat()

    base = f"https://www.sportsbookreview.com/betting-odds/{slug}"
    urls = {
        "spreads": f"{base}/pointspread/full-game/?date={date_str}",
        "h2h": f"{base}/money-line/full-game/?date={date_str}",
        "totals": f"{base}/totals/full-game/?date={date_str}",
    }

    by_market: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for market, url in urls.items():
        try:
            items = {}
            for row in _rows(url):
                k = _key(row)
                line = _book_line(row)
                if k and line:
                    items[k] = line
            by_market[market] = items
        except Exception as exc:
            print(f"SBR {sport} {market} unavailable: {exc}")
            by_market[market] = {}

    keys = set().union(*(set(v) for v in by_market.values()))
    fetched = datetime.now(timezone.utc).isoformat()
    out: list[dict[str, Any]] = []
    sport_key = {"wnba": "basketball_wnba", "mlb": "baseball_mlb", "nfl": "americanfootball_nfl"}[sport]

    for home, away, start in sorted(keys):
        markets = []
        ml = by_market["h2h"].get((home, away, start))
        if ml:
            markets.append({"key": "h2h", "outcomes": [
                {"name": home, "price": ml.get("homeOdds")},
                {"name": away, "price": ml.get("awayOdds")},
            ]})
        sp = by_market["spreads"].get((home, away, start))
        if sp:
            markets.append({"key": "spreads", "outcomes": [
                {"name": home, "point": sp.get("homeSpread"), "price": sp.get("homeOdds")},
                {"name": away, "point": sp.get("awaySpread"), "price": sp.get("awayOdds")},
            ]})
        tot = by_market["totals"].get((home, away, start))
        if tot:
            point = tot.get("total")
            if point is None:
                point = tot.get("totalPoints")
            markets.append({"key": "totals", "outcomes": [
                {"name": "Over", "point": point, "price": tot.get("overOdds")},
                {"name": "Under", "point": point, "price": tot.get("underOdds")},
            ]})
        if markets:
            out.append({
                "id": f"sbr-{sport}-{home}-{away}-{date_str}",
                "sport_key": sport_key,
                "commence_time": start or None,
                "home_team": home,
                "away_team": away,
                "bookmakers": [{
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": fetched,
                    "markets": markets,
                }],
                "market_provider": "sportsbookreview_draftkings",
            })

    return out
# Validation touch: 2026-09-25

# Final DK mapping validation: 2026-09-25
