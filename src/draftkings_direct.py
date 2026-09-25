from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from curl_cffi import requests as http_requests
except ImportError:  # pragma: no cover
    import requests as http_requests


LEAGUE_IDS = {
    "wnba": 94682,
    "mlb": 84240,
    "nfl": 88808,
}

CATEGORY_ID = 493  # DraftKings full-game lines
BASE_URL = (
    "https://sportsbook-nash.draftkings.com/api/sportscontent/"
    "dkusnj/v1/leagues/{league_id}/categories/{category_id}"
)
V5_BASE_URL = (
    "https://sportsbook-nash.draftkings.com/sites/US-SB/api/v5/"
    "eventgroups/{league_id}?format=json"
)

WNBA_CANONICAL = {
    "dream": "Atlanta Dream",
    "sky": "Chicago Sky",
    "sun": "Connecticut Sun",
    "wings": "Dallas Wings",
    "valkyries": "Golden State Valkyries",
    "fever": "Indiana Fever",
    "sparks": "Los Angeles Sparks",
    "aces": "Las Vegas Aces",
    "lynx": "Minnesota Lynx",
    "liberty": "New York Liberty",
    "mercury": "Phoenix Mercury",
    "fire": "Portland Fire",
    "storm": "Seattle Storm",
    "tempo": "Toronto Tempo",
    "mystics": "Washington Mystics",
}


def _american(value: Any) -> int | None:
    if isinstance(value, dict):
        value = value.get("american") or value.get("americanOdds")
    if value is None:
        return None
    text = str(value).strip().replace("+", "").replace("−", "-")
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _canonical_team(name: str, sport: str) -> str:
    cleaned = " ".join(str(name or "").split())
    if sport != "wnba":
        return cleaned
    lower = cleaned.lower()
    for nickname, canonical in WNBA_CANONICAL.items():
        if lower == nickname or lower.endswith(" " + nickname):
            return canonical
    return cleaned


def _event_teams(event: dict[str, Any], sport: str) -> tuple[str, str] | None:
    name = str(event.get("name") or event.get("eventName") or "")
    if " @ " in name:
        away, home = name.split(" @ ", 1)
        return _canonical_team(away, sport), _canonical_team(home, sport)

    participants = event.get("participants") or []
    names = [
        p.get("name") or p.get("fullName")
        for p in participants
        if p.get("name") or p.get("fullName")
    ]
    if len(names) >= 2:
        return _canonical_team(names[0], sport), _canonical_team(names[1], sport)
    return None


def _fetch_json(url: str) -> dict[str, Any]:
    kwargs = {
        "timeout": 20,
        "headers": {
            "accept": "application/json, text/plain, */*",
            "referer": "https://sportsbook.draftkings.com/",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/140.0.0.0 Safari/537.36"
            ),
        },
    }
    try:
        response = http_requests.get(url, impersonate="chrome120", **kwargs)
    except TypeError:
        response = http_requests.get(url, **kwargs)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected DraftKings response format")
    return data


def _fetch_v5_full_game(sport: str, league_id: int) -> list[dict[str, Any]]:
    """Fallback parser for DraftKings' v5 event-group feed."""
    data = _fetch_json(V5_BASE_URL.format(league_id=league_id))
    group = data.get("eventGroup") or data
    events = {
        str(e.get("eventId") or e.get("id")): e
        for e in (group.get("events") or data.get("events") or [])
    }
    fetched_at = datetime.now(timezone.utc).isoformat()
    by_event: dict[str, list[dict[str, Any]]] = {}

    categories = group.get("offerCategories") or []
    for cat in categories:
        cat_name = str(cat.get("name") or "").lower()
        for desc in cat.get("offerSubcategoryDescriptors", []) or []:
            sub_name = str(desc.get("name") or "").lower()
            sub = desc.get("offerSubcategory") or {}
            offers = sub.get("offers") or []
            for offer_group in offers:
                group_offers = offer_group if isinstance(offer_group, list) else [offer_group]
                for offer in group_offers:
                    eid = str(offer.get("eventId") or "")
                    if not eid:
                        continue
                    label = str(offer.get("label") or offer.get("name") or sub_name or cat_name).lower()
                    key = None
                    if "moneyline" in label or "money line" in label:
                        key = "h2h"
                    elif "spread" in label or "run line" in label:
                        key = "spreads"
                    elif label in {"total", "totals", "game total"} or "total runs" in label or "total points" in label:
                        key = "totals"
                    if key is None:
                        continue
                    ev = events.get(eid) or {}
                    teams = _event_teams(ev, sport)
                    if not teams:
                        continue
                    away_team, home_team = teams
                    outcomes = []
                    for out in offer.get("outcomes", []) or []:
                        raw_label = str(out.get("label") or out.get("name") or out.get("participant") or "")
                        price = _american(out.get("oddsAmerican") or out.get("americanOdds") or out.get("displayOdds") or out.get("odds"))
                        point = out.get("line")
                        if point is None:
                            point = out.get("points")
                        if key == "totals":
                            low = raw_label.lower()
                            name = "Over" if "over" in low else "Under" if "under" in low else None
                        else:
                            low = raw_label.lower()
                            hlast = home_team.lower().split()[-1]
                            alast = away_team.lower().split()[-1]
                            name = home_team if hlast in low else away_team if alast in low else None
                        if name:
                            rec = {"name": name, "price": price}
                            if point is not None:
                                rec["point"] = point
                            outcomes.append(rec)
                    if outcomes:
                        by_event.setdefault(eid, []).append({"key": key, "outcomes": outcomes})

    rows = []
    for eid, markets in by_event.items():
        ev = events.get(eid) or {}
        teams = _event_teams(ev, sport)
        if not teams:
            continue
        away_team, home_team = teams
        rows.append({
            "id": eid,
            "sport_key": {"wnba":"basketball_wnba","mlb":"baseball_mlb","nfl":"americanfootball_nfl"}[sport],
            "commence_time": ev.get("startEventDate") or ev.get("startDate"),
            "home_team": home_team,
            "away_team": away_team,
            "bookmakers": [{
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": fetched_at,
                "markets": markets,
            }],
            "market_provider": "draftkings_v5_direct",
        })
    if rows:
        return rows
    try:
        return _fetch_v5_full_game(sport, league_id)
    except Exception:
        return []


def fetch_draftkings_direct(sport: str) -> list[dict[str, Any]]:
    """Fetch free, unauthenticated DraftKings full-game markets.

    This uses DraftKings' public sportsbook frontend feed. It is unofficial and
    may change without notice, so callers should retain a second provider as a
    fallback.
    """
    sport = sport.lower()
    league_id = LEAGUE_IDS.get(sport)
    if league_id is None:
        raise ValueError(f"Unsupported DraftKings sport: {sport}")

    data = _fetch_json(BASE_URL.format(league_id=league_id, category_id=CATEGORY_ID))
    events = {str(e.get("id") or e.get("eventId")): e for e in data.get("events", [])}
    markets = data.get("markets", []) or []
    selections = data.get("selections", []) or []
    selections_by_market: dict[str, list[dict[str, Any]]] = {}
    for selection in selections:
        selections_by_market.setdefault(str(selection.get("marketId")), []).append(selection)

    market_by_event: dict[str, list[dict[str, Any]]] = {}
    for market in markets:
        event_id = str(market.get("eventId") or "")
        if event_id:
            market_by_event.setdefault(event_id, []).append(market)

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for event_id, event in events.items():
        teams = _event_teams(event, sport)
        if not teams:
            continue
        away_team, home_team = teams
        dk_markets: list[dict[str, Any]] = []

        for market in market_by_event.get(event_id, []):
            name = str(market.get("name") or "").strip()
            key = None
            if name == "Moneyline":
                key = "h2h"
            elif name in {"Spread", "Run Line"}:
                key = "spreads"
            elif name == "Total":
                key = "totals"
            if key is None:
                continue

            outcomes: list[dict[str, Any]] = []
            for selection in selections_by_market.get(str(market.get("id")), []):
                label = str(selection.get("label") or "")
                price = _american(selection.get("displayOdds") or selection.get("oddsAmerican"))
                point = selection.get("points")

                if key == "totals":
                    lower = label.lower()
                    if "over" in lower:
                        outcome_name = "Over"
                    elif "under" in lower:
                        outcome_name = "Under"
                    else:
                        continue
                else:
                    # Use canonical event teams rather than abbreviated DK labels.
                    lower = label.lower()
                    home_tokens = home_team.lower().split()
                    away_tokens = away_team.lower().split()
                    if home_team.lower() in lower or (home_tokens and home_tokens[-1] in lower):
                        outcome_name = home_team
                    elif away_team.lower() in lower or (away_tokens and away_tokens[-1] in lower):
                        outcome_name = away_team
                    else:
                        continue

                outcome: dict[str, Any] = {"name": outcome_name, "price": price}
                if point is not None:
                    outcome["point"] = point
                outcomes.append(outcome)

            if outcomes:
                dk_markets.append({"key": key, "outcomes": outcomes})

        if not dk_markets:
            continue

        rows.append(
            {
                "id": event_id,
                "sport_key": {
                    "wnba": "basketball_wnba",
                    "mlb": "baseball_mlb",
                    "nfl": "americanfootball_nfl",
                }[sport],
                "commence_time": event.get("startEventDate") or event.get("startDate"),
                "home_team": home_team,
                "away_team": away_team,
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "last_update": fetched_at,
                        "markets": dk_markets,
                    }
                ],
                "market_provider": "draftkings_direct",
            }
        )

    return rows
