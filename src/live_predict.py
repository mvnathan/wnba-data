from __future__ import annotations

import math
from typing import Any


def _game_elapsed_fraction(
    period: int | None,
    clock: str | None,
) -> float:
    """Return the fraction of regulation time that has elapsed."""
    if period is None:
        return 0.0

    try:
        period_number = int(period)
    except (TypeError, ValueError):
        return 0.0

    # Any overtime period means regulation is complete.  The current
    # scoreboard score is therefore already at least the regulation final.
    if period_number > 4:
        return 1.0

    if not clock:
        return 0.0

    text = str(clock).strip()

    # ESPN occasionally supplies 0.0 before tip rather than MM:SS.
    if ":" not in text:
        return 0.0

    try:
        minutes_text, seconds_text = text.split(":", 1)
        minutes = int(minutes_text)
        seconds = int(float(seconds_text))
    except (TypeError, ValueError):
        return 0.0

    period_seconds = 10 * 60
    clock_seconds = max(
        0,
        min(
            period_seconds,
            minutes * 60 + seconds,
        ),
    )

    elapsed_in_period = period_seconds - clock_seconds
    elapsed_seconds = (
        (period_number - 1) * period_seconds
        + elapsed_in_period
    )

    return min(
        max(
            elapsed_seconds / (4 * period_seconds),
            0.0,
        ),
        1.0,
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def _blend_final_score(
    current_score: float,
    pregame_final: float,
    fraction: float,
) -> float:
    """
    Blend the pregame scoring rate with the scoring pace observed so far.

    Early in a game the pregame forecast dominates.  As more game time is
    observed, the in-game scoring pace receives progressively more weight.
    The current score is always a hard lower bound on the projected final.
    """
    if fraction <= 0.0:
        return max(current_score, pregame_final)

    if fraction >= 1.0:
        return current_score

    observed_full_game_pace = current_score / fraction

    # A sub-linear curve prevents the first few possessions from moving the
    # forecast too aggressively while still allowing the live game to become
    # the dominant signal late.
    observed_weight = min(
        0.90,
        fraction ** 0.75,
    )

    blended_rate = (
        (1.0 - observed_weight) * pregame_final
        + observed_weight * observed_full_game_pace
    )

    remaining_fraction = 1.0 - fraction

    return max(
        current_score,
        current_score + remaining_fraction * blended_rate,
    )


def _live_home_win_probability(
    projected_margin: float,
    pregame_probability: float | None,
    fraction: float,
    current_margin: float,
) -> float:
    if fraction >= 1.0:
        if current_margin > 0:
            return 1.0
        if current_margin < 0:
            return 0.0
        return 0.5

    # Margin-to-probability scale narrows as the game approaches completion.
    remaining = max(0.05, 1.0 - fraction)
    scale = max(2.5, 10.0 * math.sqrt(remaining))
    margin_probability = 1.0 / (
        1.0 + math.exp(-projected_margin / scale)
    )

    if pregame_probability is None:
        return margin_probability

    pregame_probability = min(
        max(pregame_probability, 0.0),
        1.0,
    )

    # The pregame classifier is still informative early; the live margin
    # forecast becomes the dominant probability signal later in the game.
    live_weight = min(0.95, fraction ** 0.75)

    probability = (
        (1.0 - live_weight) * pregame_probability
        + live_weight * margin_probability
    )

    return min(max(probability, 0.0), 1.0)


def project_live_game(
    game_state: dict[str, Any],
    pregame: dict[str, Any],
) -> dict[str, float]:
    """Create a live final-score and win-probability projection."""
    fraction = _game_elapsed_fraction(
        game_state.get("period"),
        game_state.get("clock"),
    )

    current_home = _safe_float(
        game_state.get("home_score"),
    )
    current_away = _safe_float(
        game_state.get("away_score"),
    )

    pregame_home = _safe_float(
        pregame.get("home_score"),
        current_home,
    )
    pregame_away = _safe_float(
        pregame.get("away_score"),
        current_away,
    )

    projected_home = _blend_final_score(
        current_home,
        pregame_home,
        fraction,
    )
    projected_away = _blend_final_score(
        current_away,
        pregame_away,
        fraction,
    )

    projected_margin = projected_home - projected_away
    projected_total = projected_home + projected_away
    current_margin = current_home - current_away

    pregame_home_win = None
    if pregame.get("home_win_probability") is not None:
        try:
            pregame_home_win = float(
                pregame["home_win_probability"]
            )
        except (TypeError, ValueError):
            pregame_home_win = None

    home_win_probability = _live_home_win_probability(
        projected_margin,
        pregame_home_win,
        fraction,
        current_margin,
    )

    return {
        "elapsed_fraction": fraction,
        "home_final": projected_home,
        "away_final": projected_away,
        "final_margin": projected_margin,
        "final_total": projected_total,
        "home_win_probability": home_win_probability,
        "away_win_probability": 1.0 - home_win_probability,
    }
