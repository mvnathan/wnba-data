"""Provider-neutral, simulation-only market evaluation primitives.

No bookmaker connection, no automatic wagering. All market types can be cataloged;
only selections with validated model probabilities can generate research alerts.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class MarketQuote:
    event_id: str
    market_id: str
    selection_id: str
    market_name: str
    period: str
    american_odds: int
    quote_timestamp: datetime
    game_timestamp: datetime
    suspended: bool = False
    provider: str = "DraftKings-covered licensed feed"


@dataclass(frozen=True)
class Evaluation:
    status: str
    implied_probability: Optional[float] = None
    expected_value_per_unit: Optional[float] = None
    probability_gap: Optional[float] = None


def implied_probability(american_odds: int) -> float:
    if american_odds == 0:
        raise ValueError("American odds cannot be zero")
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return -american_odds / (-american_odds + 100.0)


def decimal_odds(american_odds: int) -> float:
    if american_odds == 0:
        raise ValueError("American odds cannot be zero")
    return 1.0 + (american_odds / 100.0 if american_odds > 0 else 100.0 / -american_odds)


def evaluate(
    quote: MarketQuote,
    model_probability: Optional[float],
    *,
    now: Optional[datetime] = None,
    max_quote_age_seconds: float = 5,
    max_game_age_seconds: float = 5,
    min_expected_value: float = 0.04,
) -> Evaluation:
    now = now or datetime.now(timezone.utc)
    if quote.suspended:
        return Evaluation("SUSPENDED")
    for timestamp in (quote.quote_timestamp, quote.game_timestamp):
        if timestamp.tzinfo is None or timestamp > now:
            return Evaluation("INVALID_TIMESTAMP")
    if (now - quote.quote_timestamp).total_seconds() > max_quote_age_seconds:
        return Evaluation("STALE_QUOTE")
    if (now - quote.game_timestamp).total_seconds() > max_game_age_seconds:
        return Evaluation("STALE_GAME")
    if model_probability is None:
        return Evaluation("UNMODELED")
    if not 0 <= model_probability <= 1:
        return Evaluation("INVALID_PROBABILITY")
    implied = implied_probability(quote.american_odds)
    ev = model_probability * decimal_odds(quote.american_odds) - 1.0
    return Evaluation(
        "RESEARCH_ALERT" if ev >= min_expected_value else "NO_EDGE",
        implied_probability=implied,
        expected_value_per_unit=ev,
        probability_gap=model_probability - implied,
    )
