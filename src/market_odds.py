from __future__ import annotations

import math
import os
import statistics
from datetime import datetime, timezone
from typing import Any

import requests


ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_wnba/odds"
MARKET_BLEND_VERSION = "independent_model_v3_consensus"
BOOKMAKERS = "draftkings,fanduel,betmgm,caesars,pointsbetus"

TEAM_NAME_TO_ABBR = {
    "atlanta dream": "ATL", "chicago sky": "CHI", "connecticut sun": "CON",
    "dallas wings": "DAL", "golden state valkyries": "GS", "indiana fever": "IND",
    "los angeles sparks": "LA", "las vegas aces": "LV", "minnesota lynx": "MIN",
    "new york liberty": "NY", "phoenix mercury": "PHX", "portland fire": "POR",
    "seattle storm": "SEA", "toronto tempo": "TOR", "washington mystics": "WSH",
}


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().lower().replace(".", "").replace("-", " ")


def _team_abbr_from_name(name: str | None) -> str:
    return TEAM_NAME_TO_ABBR.get(_normalize_text(name), "")


def fetch_draftkings_wnba_odds() -> list[dict[str, Any]]:
    """Fetch current WNBA markets from DK plus major comparison books.

    Kept under the legacy function name to avoid breaking callers.
    """
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY environment variable is not set")
    response = requests.get(
        ODDS_API_URL,
        params={
            "apiKey": api_key,
            "bookmakers": BOOKMAKERS,
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected odds API response format")
    return payload


def _build_lookup(odds_data: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for game in odds_data:
        home_abbr = _team_abbr_from_name(game.get("home_team"))
        away_abbr = _team_abbr_from_name(game.get("away_team"))
        if home_abbr and away_abbr:
            lookup[(home_abbr, away_abbr)] = game
    return lookup


def _book_markets(odds_game: dict[str, Any], book: dict[str, Any]) -> dict[str, Any]:
    home_name = odds_game.get("home_team")
    away_name = odds_game.get("away_team")
    out = {
        "book": book.get("title") or book.get("key"),
        "key": book.get("key"),
        "updated_at": book.get("last_update"),
        "home_spread": None,
        "away_spread": None,
        "home_spread_price": None,
        "away_spread_price": None,
        "total": None,
        "over_price": None,
        "under_price": None,
        "home_moneyline": None,
        "away_moneyline": None,
    }
    for market in book.get("markets", []):
        key = market.get("key")
        for outcome in market.get("outcomes", []):
            name = outcome.get("name")
            if key == "h2h":
                if name == home_name: out["home_moneyline"] = outcome.get("price")
                elif name == away_name: out["away_moneyline"] = outcome.get("price")
            elif key == "spreads":
                if name == home_name:
                    out["home_spread"] = outcome.get("point")
                    out["home_spread_price"] = outcome.get("price")
                elif name == away_name:
                    out["away_spread"] = outcome.get("point")
                    out["away_spread_price"] = outcome.get("price")
            elif key == "totals":
                if name == "Over":
                    out["total"] = outcome.get("point")
                    out["over_price"] = outcome.get("price")
                elif name == "Under":
                    if out["total"] is None: out["total"] = outcome.get("point")
                    out["under_price"] = outcome.get("price")
    return out


def _median(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return statistics.median(nums) if nums else None


def _extract_markets(odds_game: dict[str, Any]) -> dict[str, Any]:
    books = [_book_markets(odds_game, b) for b in odds_game.get("bookmakers", [])]
    dk = next((b for b in books if b.get("key") == "draftkings"), books[0] if books else None)
    result: dict[str, Any] = {
        "market_bookmaker": dk.get("book") if dk else "DraftKings",
        "market_home_spread": dk.get("home_spread") if dk else None,
        "market_away_spread": dk.get("away_spread") if dk else None,
        "market_home_spread_price": dk.get("home_spread_price") if dk else None,
        "market_away_spread_price": dk.get("away_spread_price") if dk else None,
        "market_total": dk.get("total") if dk else None,
        "market_over_price": dk.get("over_price") if dk else None,
        "market_under_price": dk.get("under_price") if dk else None,
        "market_home_moneyline": dk.get("home_moneyline") if dk else None,
        "market_away_moneyline": dk.get("away_moneyline") if dk else None,
        "market_updated_at": (dk.get("updated_at") if dk else None) or datetime.now(timezone.utc).isoformat(),
        "market_books": books,
        "market_book_count": len(books),
        "consensus_home_spread": _median([b.get("home_spread") for b in books]),
        "consensus_total": _median([b.get("total") for b in books]),
        "consensus_home_moneyline": _median([b.get("home_moneyline") for b in books]),
        "consensus_away_moneyline": _median([b.get("away_moneyline") for b in books]),
    }
    spreads = [float(b["home_spread"]) for b in books if b.get("home_spread") is not None]
    totals = [float(b["total"]) for b in books if b.get("total") is not None]
    result["market_home_spread_min"] = min(spreads) if spreads else None
    result["market_home_spread_max"] = max(spreads) if spreads else None
    result["market_total_min"] = min(totals) if totals else None
    result["market_total_max"] = max(totals) if totals else None
    return result


def _american_implied_probability(odds: Any) -> float | None:
    try: value = float(odds)
    except (TypeError, ValueError): return None
    if value == 0: return None
    return (-value) / ((-value) + 100.0) if value < 0 else 100.0 / (value + 100.0)


def _as_float(value: Any) -> float | None:
    try: number = float(value)
    except (TypeError, ValueError): return None
    return number if number == number else None


def _margin_probability(margin: float) -> float:
    return 1.0 / (1.0 + math.exp(-margin / 6.5))


def _reconcile_probability(row: dict[str, Any], model_margin: float | None) -> None:
    raw_home = _as_float(row.get("home_win_probability"))
    row["raw_model_home_win_probability"] = raw_home
    row["raw_model_away_win_probability"] = 1.0 - raw_home if raw_home is not None else None
    row["probability_reconciled"] = False
    if model_margin is None: return
    margin_home = _margin_probability(model_margin)
    row["margin_implied_home_win_probability"] = margin_home
    if raw_home is None or (model_margin > 0 and raw_home < 0.5) or (model_margin < 0 and raw_home > 0.5):
        coherent = margin_home
        row["probability_reconciled"] = True
    else:
        coherent = raw_home
    row["home_win_probability"] = coherent
    row["away_win_probability"] = 1.0 - coherent
    row["home_win"] = coherent


def _edge_confidence(edge: float | None, *, scale: float, coherent_probability: float | None = None) -> float | None:
    if edge is None: return None
    edge_component = 1.0 - math.exp(-abs(edge) / scale)
    if coherent_probability is None: return max(0.0, min(1.0, edge_component))
    directional = abs(coherent_probability - 0.5) * 2.0
    return max(0.0, min(1.0, 0.7 * edge_component + 0.3 * directional))


def _attach_market_benchmark(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["market_blend_version"] = MARKET_BLEND_VERSION
    row["market_used_in_prediction"] = False
    model_margin = _as_float(row.get("predicted_margin"))
    model_total = _as_float(row.get("predicted_total"))
    _reconcile_probability(row, model_margin)
    model_home_probability = _as_float(row.get("home_win_probability"))
    dk_home_spread = _as_float(row.get("market_home_spread"))
    dk_total = _as_float(row.get("market_total"))
    consensus_home_spread = _as_float(row.get("consensus_home_spread"))
    consensus_total = _as_float(row.get("consensus_total"))
    row["model_predicted_margin"] = model_margin
    row["model_predicted_total"] = model_total
    row["model_home_win_probability"] = model_home_probability
    dk_margin = -dk_home_spread if dk_home_spread is not None else None
    consensus_margin = -consensus_home_spread if consensus_home_spread is not None else None
    row["market_implied_margin"] = dk_margin
    row["consensus_implied_margin"] = consensus_margin
    row["model_market_margin_edge"] = model_margin - dk_margin if model_margin is not None and dk_margin is not None else None
    row["model_consensus_margin_edge"] = model_margin - consensus_margin if model_margin is not None and consensus_margin is not None else None
    row["model_market_total_edge"] = model_total - dk_total if model_total is not None and dk_total is not None else None
    row["model_consensus_total_edge"] = model_total - consensus_total if model_total is not None and consensus_total is not None else None
    home_implied = _american_implied_probability(row.get("market_home_moneyline"))
    away_implied = _american_implied_probability(row.get("market_away_moneyline"))
    if home_implied is not None and away_implied is not None and home_implied + away_implied > 0:
        no_vig_home = home_implied / (home_implied + away_implied)
        row["market_no_vig_home_win_probability"] = no_vig_home
        row["model_market_home_win_edge"] = model_home_probability - no_vig_home if model_home_probability is not None else None
    else:
        row["market_no_vig_home_win_probability"] = None
        row["model_market_home_win_edge"] = None
    row["spread_edge_confidence"] = _edge_confidence(row["model_consensus_margin_edge"], scale=6.0, coherent_probability=model_home_probability)
    row["total_edge_confidence"] = _edge_confidence(row["model_consensus_total_edge"], scale=9.0)
    row["market_spread_weight"] = row["market_total_weight"] = row["market_moneyline_weight"] = 0.0
    row["market_display_mode"] = "pure_model_vs_dk_and_consensus"
    return row


def attach_market_odds(games: list[dict[str, Any]], odds_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _build_lookup(odds_data)
    enriched = []
    matched = 0
    for game in games:
        row = dict(game)
        home_abbr = str(row.get("home_abbr", "")).strip().upper()
        away_abbr = str(row.get("away_abbr", "")).strip().upper()
        odds_game = lookup.get((home_abbr, away_abbr))
        if odds_game is None:
            row["market_blend_version"] = MARKET_BLEND_VERSION
            row["market_used_in_prediction"] = False
            _reconcile_probability(row, _as_float(row.get("predicted_margin")))
            enriched.append(row)
            continue
        row.update(_extract_markets(odds_game))
        row = _attach_market_benchmark(row)
        matched += 1
        enriched.append(row)
    print("Market matches:", matched, "/", len(games))
    return enriched
