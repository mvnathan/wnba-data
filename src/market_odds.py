from __future__ import annotations

import math
import os
import statistics
from datetime import datetime, timezone
from typing import Any

import requests


ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_wnba/odds"
MARKET_BLEND_VERSION = "independent_model_v3_consensus"
BOOKMAKERS = "draftkings,fanduel,betmgm,caesars"

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
    """Fetch current WNBA markets from DK plus major comparison books."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY environment variable is not set")
    response = requests.get(
        ODDS_API_URL,
        params={"apiKey": api_key, "bookmakers": BOOKMAKERS, "markets": "h2h,spreads,totals", "oddsFormat": "american"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected odds API response format")
    return payload


def _build_lookup(odds_data: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup = {}
    for game in odds_data:
        home_abbr = _team_abbr_from_name(game.get("home_team"))
        away_abbr = _team_abbr_from_name(game.get("away_team"))
        if home_abbr and away_abbr:
            lookup[(home_abbr, away_abbr)] = game
    return lookup


def _book_markets(odds_game: dict[str, Any], book: dict[str, Any]) -> dict[str, Any]:
    home_name, away_name = odds_game.get("home_team"), odds_game.get("away_team")
    out = {"book": book.get("title") or book.get("key"), "key": book.get("key"), "updated_at": book.get("last_update"), "home_spread": None, "away_spread": None, "home_spread_price": None, "away_spread_price": None, "total": None, "over_price": None, "under_price": None, "home_moneyline": None, "away_moneyline": None}
    for market in book.get("markets", []):
        key = market.get("key")
        for outcome in market.get("outcomes", []):
            name = outcome.get("name")
            if key == "h2h":
                if name == home_name: out["home_moneyline"] = outcome.get("price")
                elif name == away_name: out["away_moneyline"] = outcome.get("price")
            elif key == "spreads":
                if name == home_name: out["home_spread"], out["home_spread_price"] = outcome.get("point"), outcome.get("price")
                elif name == away_name: out["away_spread"], out["away_spread_price"] = outcome.get("point"), outcome.get("price")
            elif key == "totals":
                if name == "Over": out["total"], out["over_price"] = outcome.get("point"), outcome.get("price")
                elif name == "Under":
                    if out["total"] is None: out["total"] = outcome.get("point")
                    out["under_price"] = outcome.get("price")
    return out


def _median(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return statistics.median(nums) if nums else None


def _american_implied_probability(odds: Any) -> float | None:
    try: value = float(odds)
    except (TypeError, ValueError): return None
    if value == 0: return None
    return (-value) / ((-value) + 100.0) if value < 0 else 100.0 / (value + 100.0)


def _book_no_vig_home(book: dict[str, Any]) -> float | None:
    hp = _american_implied_probability(book.get("home_moneyline"))
    ap = _american_implied_probability(book.get("away_moneyline"))
    if hp is None or ap is None or hp + ap <= 0: return None
    return hp / (hp + ap)


def _extract_markets(odds_game: dict[str, Any]) -> dict[str, Any]:
    books = [_book_markets(odds_game, b) for b in odds_game.get("bookmakers", [])]
    dk = next((b for b in books if b.get("key") == "draftkings"), None)
    fallback = dk or (books[0] if books else None)
    spreads = [float(b["home_spread"]) for b in books if b.get("home_spread") is not None]
    totals = [float(b["total"]) for b in books if b.get("total") is not None]
    no_vig = [_book_no_vig_home(b) for b in books]
    return {
        "market_bookmaker": fallback.get("book") if fallback else None,
        "market_bookmaker_key": fallback.get("key") if fallback else None,
        "market_home_spread": fallback.get("home_spread") if fallback else None,
        "market_away_spread": fallback.get("away_spread") if fallback else None,
        "market_home_spread_price": fallback.get("home_spread_price") if fallback else None,
        "market_away_spread_price": fallback.get("away_spread_price") if fallback else None,
        "market_total": fallback.get("total") if fallback else None,
        "market_over_price": fallback.get("over_price") if fallback else None,
        "market_under_price": fallback.get("under_price") if fallback else None,
        "market_home_moneyline": fallback.get("home_moneyline") if fallback else None,
        "market_away_moneyline": fallback.get("away_moneyline") if fallback else None,
        "market_updated_at": (fallback.get("updated_at") if fallback else None) or datetime.now(timezone.utc).isoformat(),
        "market_books": books, "market_book_count": len(books),
        "consensus_home_spread": _median([b.get("home_spread") for b in books]),
        "consensus_total": _median([b.get("total") for b in books]),
        "consensus_no_vig_home_win_probability": _median(no_vig),
        "market_home_spread_min": min(spreads) if spreads else None, "market_home_spread_max": max(spreads) if spreads else None,
        "market_total_min": min(totals) if totals else None, "market_total_max": max(totals) if totals else None,
    }


def _as_float(value: Any) -> float | None:
    try: number = float(value)
    except (TypeError, ValueError): return None
    return number if number == number else None


def _margin_probability(margin: float) -> float:
    return 1.0 / (1.0 + math.exp(-margin / 6.5))


def _reconcile_probability(row: dict[str, Any], model_margin: float | None) -> None:
    raw_home = _as_float(row.get("home_win_probability")); row["raw_model_home_win_probability"] = raw_home; row["raw_model_away_win_probability"] = 1.0 - raw_home if raw_home is not None else None; row["probability_reconciled"] = False
    if model_margin is None: return
    margin_home = _margin_probability(model_margin); row["margin_implied_home_win_probability"] = margin_home
    coherent = margin_home if raw_home is None or (model_margin > 0 and raw_home < .5) or (model_margin < 0 and raw_home > .5) else raw_home
    row["probability_reconciled"] = coherent != raw_home; row["home_win_probability"] = coherent; row["away_win_probability"] = 1.0 - coherent; row["home_win"] = coherent


def _edge_confidence(edge: float | None, *, scale: float, coherent_probability: float | None = None) -> float | None:
    if edge is None: return None
    component = 1.0 - math.exp(-abs(edge) / scale)
    if coherent_probability is None: return max(0.0, min(1.0, component))
    return max(0.0, min(1.0, .7 * component + .3 * abs(coherent_probability - .5) * 2.0))


def _attach_market_benchmark(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row); row["market_blend_version"] = MARKET_BLEND_VERSION; row["market_used_in_prediction"] = False
    model_margin, model_total = _as_float(row.get("predicted_margin")), _as_float(row.get("predicted_total")); _reconcile_probability(row, model_margin)
    model_prob = _as_float(row.get("home_win_probability")); hs = _as_float(row.get("market_home_spread")); total = _as_float(row.get("market_total")); chs = _as_float(row.get("consensus_home_spread")); ct = _as_float(row.get("consensus_total"))
    row["model_predicted_margin"] = model_margin; row["model_predicted_total"] = model_total; row["model_home_win_probability"] = model_prob
    dm, cm = (-hs if hs is not None else None), (-chs if chs is not None else None)
    row["market_implied_margin"] = dm; row["consensus_implied_margin"] = cm
    row["model_market_margin_edge"] = model_margin - dm if model_margin is not None and dm is not None else None
    row["model_consensus_margin_edge"] = model_margin - cm if model_margin is not None and cm is not None else None
    row["model_market_total_edge"] = model_total - total if model_total is not None and total is not None else None
    row["model_consensus_total_edge"] = model_total - ct if model_total is not None and ct is not None else None
    nv = _as_float(row.get("consensus_no_vig_home_win_probability")); row["model_consensus_home_win_edge"] = model_prob - nv if model_prob is not None and nv is not None else None
    row["spread_edge_confidence"] = _edge_confidence(row["model_consensus_margin_edge"], scale=6.0, coherent_probability=model_prob); row["total_edge_confidence"] = _edge_confidence(row["model_consensus_total_edge"], scale=9.0)
    row["market_spread_weight"] = row["market_total_weight"] = row["market_moneyline_weight"] = 0.0; row["market_display_mode"] = "pure_model_vs_dk_and_consensus"
    return row


def attach_market_odds(games: list[dict[str, Any]], odds_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _build_lookup(odds_data); enriched = []; matched = 0
    for game in games:
        row = dict(game); odds_game = lookup.get((str(row.get("home_abbr", "")).strip().upper(), str(row.get("away_abbr", "")).strip().upper()))
        if odds_game is None:
            row["market_blend_version"] = MARKET_BLEND_VERSION; row["market_used_in_prediction"] = False; _reconcile_probability(row, _as_float(row.get("predicted_margin"))); enriched.append(row); continue
        row.update(_extract_markets(odds_game)); enriched.append(_attach_market_benchmark(row)); matched += 1
    print("Market matches:", matched, "/", len(games)); return enriched
