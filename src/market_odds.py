from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests


ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/"
    "sports/basketball_wnba/odds"
)


# ---------------------------------------------------------------------
# Team normalization
#
# Our prediction data uses abbreviations such as ATL / CON / NY.
# The Odds API uses full names such as Atlanta Dream.
# ---------------------------------------------------------------------

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


def _normalize_text(
    value: str | None,
) -> str:
    if not value:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(".", "")
        .replace("-", " ")
    )


def _team_abbr_from_name(
    name: str | None,
) -> str:
    normalized = _normalize_text(
        name
    )

    return TEAM_NAME_TO_ABBR.get(
        normalized,
        "",
    )


def fetch_draftkings_wnba_odds(
) -> list[dict[str, Any]]:
    """
    Fetch current DraftKings WNBA moneyline, spread and total markets.
    """
    api_key = os.environ.get(
        "ODDS_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY environment variable is not set"
        )

    params = {
        "apiKey": api_key,
        "bookmakers": "draftkings",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }

    response = requests.get(
        ODDS_API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(
        payload,
        list,
    ):
        raise RuntimeError(
            "Unexpected odds API response format"
        )

    return payload


def _build_lookup(
    odds_data: list[
        dict[str, Any]
    ],
) -> dict[
    tuple[str, str],
    dict[str, Any],
]:
    """
    Build lookup as:

        (home_abbr, away_abbr) -> odds game
    """
    lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for game in odds_data:
        home_name = game.get(
            "home_team"
        )

        away_name = game.get(
            "away_team"
        )

        home_abbr = (
            _team_abbr_from_name(
                home_name
            )
        )

        away_abbr = (
            _team_abbr_from_name(
                away_name
            )
        )

        if (
            not home_abbr
            or not away_abbr
        ):
            print(
                "Warning: could not normalize odds teams:",
                away_name,
                "@",
                home_name,
            )

            continue

        lookup[
            (
                home_abbr,
                away_abbr,
            )
        ] = game

    return lookup


def _extract_markets(
    odds_game: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract DraftKings markets from one Odds API event.
    """
    result: dict[
        str,
        Any,
    ] = {
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

    home_name = odds_game.get(
        "home_team"
    )

    away_name = odds_game.get(
        "away_team"
    )

    bookmakers = odds_game.get(
        "bookmakers",
        [],
    )

    if not bookmakers:
        return result

    # We request only DraftKings, so the first bookmaker should be DK.
    book = bookmakers[0]

    result[
        "market_bookmaker"
    ] = (
        book.get("title")
        or "DraftKings"
    )

    result[
        "market_updated_at"
    ] = (
        book.get(
            "last_update"
        )
        or datetime.now(
            timezone.utc
        ).isoformat()
    )

    for market in book.get(
        "markets",
        [],
    ):
        market_key = (
            market.get("key")
        )

        outcomes = (
            market.get(
                "outcomes",
                [],
            )
        )

        # -------------------------------------------------------------
        # Moneyline
        # -------------------------------------------------------------
        if market_key == "h2h":

            for outcome in outcomes:
                name = outcome.get(
                    "name"
                )

                price = outcome.get(
                    "price"
                )

                if name == home_name:
                    result[
                        "market_home_moneyline"
                    ] = price

                elif name == away_name:
                    result[
                        "market_away_moneyline"
                    ] = price

        # -------------------------------------------------------------
        # Spread
        # -------------------------------------------------------------
        elif market_key == "spreads":

            for outcome in outcomes:
                name = outcome.get(
                    "name"
                )

                point = outcome.get(
                    "point"
                )

                price = outcome.get(
                    "price"
                )

                if name == home_name:
                    result[
                        "market_home_spread"
                    ] = point

                    result[
                        "market_home_spread_price"
                    ] = price

                elif name == away_name:
                    result[
                        "market_away_spread"
                    ] = point

                    result[
                        "market_away_spread_price"
                    ] = price

        # -------------------------------------------------------------
        # Game total
        # -------------------------------------------------------------
        elif market_key == "totals":

            for outcome in outcomes:
                name = outcome.get(
                    "name"
                )

                point = outcome.get(
                    "point"
                )

                price = outcome.get(
                    "price"
                )

                if name == "Over":
                    result[
                        "market_total"
                    ] = point

                    result[
                        "market_over_price"
                    ] = price

                elif name == "Under":
                    if (
                        result[
                            "market_total"
                        ]
                        is None
                    ):
                        result[
                            "market_total"
                        ] = point

                    result[
                        "market_under_price"
                    ] = price

    return result


def attach_market_odds(
    games: list[
        dict[str, Any]
    ],
    odds_data: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Attach DraftKings markets to our model prediction rows.

    Prediction games are matched by WNBA team abbreviations.
    """
    lookup = _build_lookup(
        odds_data
    )

    enriched: list[
        dict[str, Any]
    ] = []

    matched = 0

    for game in games:
        row = dict(
            game
        )

        home_abbr = (
            str(
                row.get(
                    "home_abbr",
                    "",
                )
            )
            .strip()
            .upper()
        )

        away_abbr = (
            str(
                row.get(
                    "away_abbr",
                    "",
                )
            )
            .strip()
            .upper()
        )

        key = (
            home_abbr,
            away_abbr,
        )

        odds_game = (
            lookup.get(
                key
            )
        )

        if odds_game is None:
            print(
                "Warning: no DraftKings match for",
                away_abbr,
                "@",
                home_abbr,
            )

            enriched.append(
                row
            )

            continue

        row.update(
            _extract_markets(
                odds_game
            )
        )

        matched += 1

        enriched.append(
            row
        )

    print(
        "DraftKings matches:",
        matched,
        "/",
        len(games),
    )

    return enriched