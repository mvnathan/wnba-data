from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests


ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/"
    "sports/basketball_wnba/odds"
)

# Conservative initial shrinkage toward the market. These weights are explicit
# and versioned so they can later be replaced by walk-forward learned weights
# once prediction-history contains enough market snapshots.
MARKET_SPREAD_WEIGHT = 0.35
MARKET_TOTAL_WEIGHT = 0.35
MARKET_MONEYLINE_WEIGHT = 0.25
MARKET_BLEND_VERSION = "market_anchor_v1"

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
    result["market_updated_at"] = book.get("last_update") or datetime.now(timezone.utc).isoformat()

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


def _blend_number(model_value: Any, market_value: Any, market_weight: float) -> float | None:
    try:
        model = float(model_value)
        market = float(market_value)
    except (TypeError, ValueError):
        return None
    return (1.0 - market_weight) * model + market_weight * market


def _refresh_directional_fields(row: dict[str, Any]) -> None:
    """Keep winner/spread/coherence labels aligned with the final forecast."""
    try:
        margin = float(row.get("predicted_margin"))
    except (TypeError, ValueError):
        return

    if abs(margin) < 0.5:
        row["projected_winner_side"] = "pick"
        row["projected_winner_abbr"] = None
        row["projected_spread"] = 0.0
        row["prediction_coherent"] = None
        return

    margin_home_pick = margin > 0
    row["projected_winner_side"] = "home" if margin_home_pick else "away"
    row["projected_winner_abbr"] = (
        row.get("home_abbr") if margin_home_pick else row.get("away_abbr")
    )
    row["projected_spread"] = -abs(margin)

    try:
        home_probability = float(row.get("home_win_probability"))
    except (TypeError, ValueError):
        row["prediction_coherent"] = None
        return

    row["prediction_coherent"] = (home_probability >= 0.5) == margin_home_pick


def _apply_market_anchor(row: dict[str, Any]) -> dict[str, Any]:
    """Blend model-only forecasts with market priors while preserving both."""
    row = dict(row)
    row["market_blend_version"] = MARKET_BLEND_VERSION

    model_margin = row.get("predicted_margin")
    home_spread = row.get("market_home_spread")
    if model_margin is not None:
        row["model_predicted_margin"] = model_margin
    try:
        market_margin = -float(home_spread)
    except (TypeError, ValueError):
        market_margin = None
    row["market_implied_margin"] = market_margin
    blended_margin = _blend_number(model_margin, market_margin, MARKET_SPREAD_WEIGHT)
    if blended_margin is not None:
        row["predicted_margin"] = blended_margin
        row["market_spread_weight"] = MARKET_SPREAD_WEIGHT

    model_total = row.get("predicted_total")
    market_total = row.get("market_total")
    if model_total is not None:
        row["model_predicted_total"] = model_total
    blended_total = _blend_number(model_total, market_total, MARKET_TOTAL_WEIGHT)
    if blended_total is not None:
        row["predicted_total"] = blended_total
        row["market_total_weight"] = MARKET_TOTAL_WEIGHT

    model_home_probability = row.get("home_win_probability")
    if model_home_probability is not None:
        row["model_home_win_probability"] = model_home_probability

    home_implied = _american_implied_probability(row.get("market_home_moneyline"))
    away_implied = _american_implied_probability(row.get("market_away_moneyline"))
    if home_implied is not None and away_implied is not None and (home_implied + away_implied) > 0:
        no_vig_home = home_implied / (home_implied + away_implied)
        row["market_no_vig_home_win_probability"] = no_vig_home
        blended_probability = _blend_number(
            model_home_probability,
            no_vig_home,
            MARKET_MONEYLINE_WEIGHT,
        )
        if blended_probability is not None:
            blended_probability = min(1.0, max(0.0, blended_probability))
            row["home_win_probability"] = blended_probability
            row["away_win_probability"] = 1.0 - blended_probability
            row["market_moneyline_weight"] = MARKET_MONEYLINE_WEIGHT

    _refresh_directional_fields(row)
    return row


def attach_market_odds(
    games: list[dict[str, Any]],
    odds_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach DraftKings markets and apply the conservative market anchor."""
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
            enriched.append(row)
            continue

        row.update(_extract_markets(odds_game))
        row = _apply_market_anchor(row)
        matched += 1
        enriched.append(row)

    print("DraftKings matches:", matched, "/", len(games))
    return enriched
