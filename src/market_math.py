from __future__ import annotations

from typing import Optional


def american_implied_probability(odds: float | int | None) -> Optional[float]:
    if odds is None:
        return None
    o = float(odds)
    if o == 0:
        return None
    return (-o / (-o + 100.0)) if o < 0 else (100.0 / (o + 100.0))


def no_vig_two_way_probabilities(
    home_odds: float | int | None,
    away_odds: float | int | None,
) -> tuple[Optional[float], Optional[float]]:
    """Return normalized two-way fair probabilities from American moneylines.

    If either side is missing/invalid, returns (None, None) rather than treating
    a single raw implied probability as a fair market probability.
    """
    home = american_implied_probability(home_odds)
    away = american_implied_probability(away_odds)
    if home is None or away is None:
        return None, None
    total = home + away
    if total <= 0:
        return None, None
    return home / total, away / total
