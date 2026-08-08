from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuarterEvent:
    game_id: str
    event_type: str
    period: int | None
    clock: str | None
    status: str | None


def detect_quarter_event(previous: dict[str, Any] | None, current: dict[str, Any]) -> QuarterEvent | None:
    if not current or "game_id" not in current:
        return None

    game_id = str(current["game_id"])
    current_period = current.get("period")
    current_status = str(current.get("status", "")).lower()
    current_clock = current.get("clock")

    if previous is None:
        return QuarterEvent(game_id=game_id, event_type="PREGAME", period=current_period, clock=current_clock, status=current_status)

    prior_period = previous.get("period")
    prior_status = str(previous.get("status", "")).lower()

    if prior_period is None and current_period == 1:
        return QuarterEvent(game_id=game_id, event_type="START", period=current_period, clock=current_clock, status=current_status)
    if prior_period == 1 and current_period == 2:
        return QuarterEvent(game_id=game_id, event_type="END_Q1", period=current_period, clock=current_clock, status=current_status)
    if prior_period == 2 and current_period == 3:
        return QuarterEvent(game_id=game_id, event_type="HALFTIME", period=current_period, clock=current_clock, status=current_status)
    if prior_period == 3 and current_period == 4:
        return QuarterEvent(game_id=game_id, event_type="END_Q3", period=current_period, clock=current_clock, status=current_status)
    if current_status in {"final", "status_final"} or current_period is not None and current_period > 4 and prior_period == 4:
        return QuarterEvent(game_id=game_id, event_type="FINAL", period=current_period, clock=current_clock, status=current_status)
    return None
