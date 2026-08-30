from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

import requests


ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/"
    "sports/basketball_wnba/odds"
)

# The market is deliberately NOT blended into the model forecast.
# It is retained as an external benchmark so model-vs-market disagreement
# remains visible and can be evaluated for repeatable edge.
MARKET_BLEND_VERSION = "independent_model_v2"

TEAM_NAME_TO_ABBR = {
    "atlanta dream": "ATL",
    "chicago sky": "CHI",
    "connecticut sun": "CON",
    "dallas wings": "DAL",
    "golden state valkyries": "GS",
    "indiana fever": "IND",
    "los angeles sparks": "LA",
    "las vegas aces": "LV",
    "minnesota lynx": "MIN",
    "new york liberty": "NY",
    "phoenix mercury": "PHX",
    "portland fire": "POR",
    "seattle storm": "SEA",
    "toronto tempo": "TOR",
    "washington mystics": "WSH",
}


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().lower().replace(".", "").replace("-", " ")


def _team_abbr_from_name(name: str | None) -> str:
    return TEAM_NAME_TO_ABBR.get(_normalize_text(name), "")


def fetch_draftkings_wnba_odds() -> list[dict[str, Any]]:
    """Fetch current DraftKings WNBA moneyline, spread and total markets."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY environment variable is not set")

    response = requests.get(
        ODDS_API_URL,
        params={
            "apiKey": api_key,
            "bookmakers": "draftkings",
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


def _build_lookup(
    odds_data: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for game in odds_data:
        home_name = game.get("home_team")
        away_name = game.get("away_team")
        home_abbr = _team_abbr_from_name(home_name)
        away_abbr = _team_abbr_from_name(away_name)
        if not home_abbr or not away_abbr:
            print("Warning: could not normalize odds teams:", away_name, "@", home_name)
            continue
        lookup[(home_abbr, away_abbr)] = game
    return lookup


def _extract_markets(odds_game: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "market_bookmaker": "DraftKings",
        "market_home_spread": None,
        "market_away_spread": None,
        "market_home_spread_price": None,
        "market_away_spread_price": None,
        "market_total": None,
        "market_over_price": None,
        "market_under_price": None,
        "market_home_moneyline": None,
        "market_away_moneyline": None,
        "market_updated_at": None,
    }

    home_name = odds_game.get("home_team")
    away_name = odds_game.get("away_team")
    bookmakers = odds_game.get("bookmakers", [])
    if not bookmakers:
        return result

    book = bookmakers[0]
    result["market_bookmaker"] = book.get("title") or "DraftKings"
    result["market_updated_at"] = (
        book.get("last_update") or datetime.now(timezone.utc).isoformat()
    )

    for market in book.get("markets", []):
        market_key = market.get("key")
        outcomes = market.get("outcomes", [])

        if market_key == "h2h":
            for outcome in outcomes:
                name = outcome.get("name")
                price = outcome.get("price")
                if name == home_name:
                    result["market_home_moneyline"] = price
                elif name == away_name:
                    result["market_away_moneyline"] = price

        elif market_key == "spreads":
            for outcome in outcomes:
                name = outcome.get("name")
                point = outcome.get("point")
                price = outcome.get("price")
                if name == home_name:
                    result["market_home_spread"] = point
                    result["market_home_spread_price"] = price
                elif name == away_name:
                    result["market_away_spread"] = point
                    result["market_away_spread_price"] = price

        elif market_key == "totals":
            for outcome in outcomes:
                name = outcome.get("name")
                point = outcome.get("point")
                price = outcome.get("price")
                if name == "Over":
                    result["market_total"] = point
                    result["market_over_price"] = price
                elif name == "Under":
                    if result["market_total"] is None:
                        result["market_total"] = point
                    result["market_under_price"] = price

    return result


def _american_implied_probability(odds: Any) -> float | None:
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return None
    if value < 0:
        return (-value) / ((-value) + 100.0)
    return 100.0 / (value + 100.0)


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _margin_probability(margin: float) -> float:
    """Map an independent projected margin to a smooth win probability."""
    return 1.0 / (1.0 + math.exp(-margin / 6.5))


def _reconcile_probability(row: dict[str, Any], model_margin: float | None) -> None:
    """Make displayed winner probability coherent with the score/margin model.

    The raw classifier output is preserved. No sportsbook information is used.
    If the classifier and score model disagree on the winning side, use the
    margin-derived probability for the public decision view rather than showing
    contradictory recommendations.
    """
    raw_home = _as_float(row.get("home_win_probability"))
    row["raw_model_home_win_probability"] = raw_home
    row["raw_model_away_win_probability"] = (
        1.0 - raw_home if raw_home is not None else None
    )
    row["probability_reconciled"] = False

    if model_margin is None:
        return

    margin_home = _margin_probability(model_margin)
    row["margin_implied_home_win_probability"] = margin_home

    if raw_home is None:
        coherent = margin_home
        row["probability_reconciled"] = True
    else:
        side_disagrees = (model_margin > 0 and raw_home < 0.5) or (
            model_margin < 0 and raw_home > 0.5
        )
        if side_disagrees:
            coherent = margin_home
            row["probability_reconciled"] = True
        else:
            coherent = raw_home

    row["home_win_probability"] = coherent
    row["away_win_probability"] = 1.0 - coherent
    row["home_win"] = coherent


def _edge_confidence(
    edge: float | None,
    *,
    scale: float,
    coherent_probability: float | None = None,
) -> float | None:
    """Preliminary transparent confidence score pending learned edge model.

    This is not a claim of historical profitability. It is a ranking score that
    rewards larger independent disagreement and, for spreads, stronger model
    directional conviction. The research workflow will later replace/calibrate
    this with walk-forward empirical probabilities.
    """
    if edge is None:
        return None
    edge_component = 1.0 - math.exp(-abs(edge) / scale)
    if coherent_probability is None:
        return max(0.0, min(1.0, edge_component))
    directional = abs(coherent_probability - 0.5) * 2.0
    score = 0.7 * edge_component + 0.3 * directional
    return max(0.0, min(1.0, score))


def _attach_market_benchmark(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve pure model outputs and add market-comparison fields only."""
    row = dict(row)
    row["market_blend_version"] = MARKET_BLEND_VERSION
    row["market_used_in_prediction"] = False

    model_margin = _as_float(row.get("predicted_margin"))
    model_total = _as_float(row.get("predicted_total"))

    _reconcile_probability(row, model_margin)
    model_home_probability = _as_float(row.get("home_win_probability"))
    market_home_spread = _as_float(row.get("market_home_spread"))
    market_total = _as_float(row.get("market_total"))

    row["model_predicted_margin"] = model_margin
    row["model_predicted_total"] = model_total
    row["model_home_win_probability"] = model_home_probability

    # Home spread -5 corresponds to market-implied home margin +5.
    market_margin = -market_home_spread if market_home_spread is not None else None
    row["market_implied_margin"] = market_margin

    margin_edge = (
        model_margin - market_margin
        if model_margin is not None and market_margin is not None
        else None
    )
    total_edge = (
        model_total - market_total
        if model_total is not None and market_total is not None
        else None
    )
    row["model_market_margin_edge"] = margin_edge
    row["model_market_total_edge"] = total_edge

    home_implied = _american_implied_probability(row.get("market_home_moneyline"))
    away_implied = _american_implied_probability(row.get("market_away_moneyline"))
    if (
        home_implied is not None
        and away_implied is not None
        and (home_implied + away_implied) > 0
    ):
        no_vig_home = home_implied / (home_implied + away_implied)
        row["market_no_vig_home_win_probability"] = no_vig_home
        row["model_market_home_win_edge"] = (
            model_home_probability - no_vig_home
            if model_home_probability is not None
            else None
        )
    else:
        row["market_no_vig_home_win_probability"] = None
        row["model_market_home_win_edge"] = None

    row["spread_edge_confidence"] = _edge_confidence(
        margin_edge,
        scale=6.0,
        coherent_probability=model_home_probability,
    )
    row["total_edge_confidence"] = _edge_confidence(total_edge, scale=9.0)

    # Explicitly clear old blend weights if a row is refreshed from an older
    # market-anchored prediction payload.
    row["market_spread_weight"] = 0.0
    row["market_total_weight"] = 0.0
    row["market_moneyline_weight"] = 0.0
    row["market_display_mode"] = "pure_model_vs_market"
    return row


def attach_market_odds(
    games: list[dict[str, Any]],
    odds_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach DraftKings markets without changing the independent model forecast."""
    lookup = _build_lookup(odds_data)
    enriched: list[dict[str, Any]] = []
    matched = 0

    for game in games:
        row = dict(game)
        home_abbr = str(row.get("home_abbr", "")).strip().upper()
        away_abbr = str(row.get("away_abbr", "")).strip().upper()
        odds_game = lookup.get((home_abbr, away_abbr))

        if odds_game is None:
            print("Warning: no DraftKings match for", away_abbr, "@", home_abbr)
            row["market_blend_version"] = MARKET_BLEND_VERSION
            row["market_used_in_prediction"] = False
            _reconcile_probability(row, _as_float(row.get("predicted_margin")))
            enriched.append(row)
            continue

        row.update(_extract_markets(odds_game))
        row = _attach_market_benchmark(row)
        matched += 1
        enriched.append(row)

    print("DraftKings matches:", matched, "/", len(games))
    return enriched
