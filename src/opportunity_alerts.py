from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config import (
    ALERT_STATE_PATH,
    DOCS_OPPORTUNITIES_JSON,
    OPPORTUNITY_ALERTS_PATH,
    PREDICTION_LATEST_JSON,
)

# Initial policy thresholds. These remain explicit so they can later be
# calibrated from performance_history.parquet.
SPREAD_THRESHOLDS = {"moderate": 2.0, "strong": 4.0, "very_strong": 7.0}
TOTAL_THRESHOLDS = {"moderate": 3.0, "strong": 6.0, "very_strong": 10.0}
ALERT_LEVELS = {"strong", "very_strong"}
LEVEL_RANK = {"neutral": 0, "moderate": 1, "strong": 2, "very_strong": 3}

# Re-notify only when a still-active opportunity has moved materially and the
# cooldown has expired. Escalations, side flips, resolutions, and reactivations
# bypass this cooldown because they represent a meaningful state transition.
COOLDOWN_MINUTES = 30
MATERIAL_EDGE_CHANGE = {"spread": 2.0, "total": 3.0}
RECENT_EVENT_LIMIT = 200


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _level(edge: float, thresholds: dict[str, float]) -> str:
    magnitude = abs(edge)
    if magnitude >= thresholds["very_strong"]:
        return "very_strong"
    if magnitude >= thresholds["strong"]:
        return "strong"
    if magnitude >= thresholds["moderate"]:
        return "moderate"
    return "neutral"


def _is_live(game: dict[str, Any]) -> bool:
    status = str(game.get("live_status") or game.get("status") or "").upper()
    return "IN_PROGRESS" in status or "HALFTIME" in status


def _spread_opportunity(game: dict[str, Any], live: bool) -> dict[str, Any] | None:
    model_key = "live_predicted_margin" if live else "predicted_margin"
    model_margin = _number(game.get(model_key))
    market_home_spread = _number(game.get("market_home_spread"))
    if model_margin is None or market_home_spread is None:
        return None

    # Model margin is home minus away; a market home spread of -5 implies a
    # market-implied home margin of +5. The difference is therefore additive.
    edge = model_margin + market_home_spread
    level = _level(edge, SPREAD_THRESHOLDS)
    home = str(game.get("home_abbr") or "HOME")
    away = str(game.get("away_abbr") or "AWAY")
    side = home if edge > 0 else away

    return {
        "market": "spread",
        "level": level,
        "side": side,
        "edge": abs(edge),
        "signed_edge_home": edge,
        "model_margin": model_margin,
        "market_home_spread": market_home_spread,
    }


def _total_opportunity(game: dict[str, Any], live: bool) -> dict[str, Any] | None:
    model_key = "live_predicted_total" if live else "predicted_total"
    model_total = _number(game.get(model_key))
    market_total = _number(game.get("market_total"))
    if model_total is None or market_total is None:
        return None

    edge = model_total - market_total
    level = _level(edge, TOTAL_THRESHOLDS)
    return {
        "market": "total",
        "level": level,
        "side": "OVER" if edge > 0 else "UNDER",
        "edge": abs(edge),
        "signed_edge": edge,
        "model_total": model_total,
        "market_total": market_total,
    }


def _legacy_alert_id(
    game_id: str,
    phase: str,
    market: str,
    side: str,
    level: str,
) -> str:
    raw = f"{game_id}|{phase}|{market}|{side}|{level}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _lifecycle_key(game_id: str, phase: str, market: str) -> str:
    return f"{game_id}|{phase}|{market}"


def _event_id(key: str, event_type: str, when: str) -> str:
    raw = f"{key}|{event_type}|{when}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _candidate_records(game: dict[str, Any]) -> list[dict[str, Any]]:
    live = _is_live(game)
    phase = "live" if live else "pregame"
    game_id = str(game.get("game_id") or "")
    if not game_id:
        return []

    candidates = [
        _spread_opportunity(game, live),
        _total_opportunity(game, live),
    ]
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate.update(
            {
                "game_id": game_id,
                "phase": phase,
                "lifecycle_key": _lifecycle_key(
                    game_id,
                    phase,
                    candidate["market"],
                ),
                "away_abbr": game.get("away_abbr"),
                "home_abbr": game.get("home_abbr"),
                "game_date_utc": game.get("game_date_utc"),
                "status": game.get("live_status") or game.get("status"),
                "live_period": game.get("live_period"),
                "live_clock": game.get("live_clock"),
                "live_away_score": game.get("live_away_score"),
                "live_home_score": game.get("live_home_score"),
                "market_bookmaker": game.get("market_bookmaker"),
                "market_updated_at": game.get("market_updated_at"),
                "home_win_probability": game.get(
                    "live_home_win_probability" if live else "home_win_probability"
                ),
                "away_win_probability": game.get(
                    "live_away_win_probability" if live else "away_win_probability"
                ),
            }
        )
        records.append(candidate)
    return records


def evaluate_game(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Return currently alertable (strong or very strong) opportunities."""
    return [
        record
        for record in _candidate_records(game)
        if record["level"] in ALERT_LEVELS
    ]


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _event(
    candidate: dict[str, Any],
    event_type: str,
    now_iso: str,
    previous: dict[str, Any] | None,
    *,
    notify: bool = True,
    reason: str | None = None,
) -> dict[str, Any]:
    event = {
        "event_id": _event_id(candidate["lifecycle_key"], event_type, now_iso),
        "event_type": event_type,
        "notify": notify,
        "occurred_at_utc": now_iso,
        "game_id": candidate["game_id"],
        "phase": candidate["phase"],
        "market": candidate["market"],
        "side": candidate["side"],
        "level": candidate["level"],
        "edge": candidate["edge"],
        "away_abbr": candidate.get("away_abbr"),
        "home_abbr": candidate.get("home_abbr"),
        "live_period": candidate.get("live_period"),
        "live_clock": candidate.get("live_clock"),
        "live_away_score": candidate.get("live_away_score"),
        "live_home_score": candidate.get("live_home_score"),
        "market_bookmaker": candidate.get("market_bookmaker"),
        "market_updated_at": candidate.get("market_updated_at"),
    }
    if reason:
        event["reason"] = reason
    if previous:
        event["previous_side"] = previous.get("side")
        event["previous_level"] = previous.get("level")
        event["previous_edge"] = previous.get("edge")
    return event


def _resolution_candidate(previous: dict[str, Any]) -> dict[str, Any]:
    return {
        "lifecycle_key": previous["lifecycle_key"],
        "game_id": previous.get("game_id"),
        "phase": previous.get("phase"),
        "market": previous.get("market"),
        "side": previous.get("side"),
        "level": previous.get("level", "neutral"),
        "edge": previous.get("edge", 0.0),
        "away_abbr": previous.get("away_abbr"),
        "home_abbr": previous.get("home_abbr"),
        "live_period": previous.get("live_period"),
        "live_clock": previous.get("live_clock"),
        "live_away_score": previous.get("live_away_score"),
        "live_home_score": previous.get("live_home_score"),
        "market_bookmaker": previous.get("market_bookmaker"),
        "market_updated_at": previous.get("market_updated_at"),
    }


def _migrate_legacy_seen(
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    now_iso: str,
) -> dict[str, dict[str, Any]]:
    opportunities = state.get("opportunities", {})
    if isinstance(opportunities, dict) and opportunities:
        return opportunities

    seen = state.get("seen", {})
    if not isinstance(seen, dict) or not seen:
        return {}

    migrated: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate["level"] not in ALERT_LEVELS:
            continue
        legacy_id = _legacy_alert_id(
            candidate["game_id"],
            candidate["phase"],
            candidate["market"],
            candidate["side"],
            candidate["level"],
        )
        if legacy_id not in seen:
            continue
        key = candidate["lifecycle_key"]
        first_seen = seen[legacy_id].get("first_seen_at_utc", now_iso)
        migrated[key] = {
            **candidate,
            "active": True,
            "first_seen_at_utc": first_seen,
            "last_seen_at_utc": now_iso,
            "last_notified_at_utc": first_seen,
            "last_notified_edge": candidate["edge"],
        }
    return migrated


def build_opportunity_alerts(
    latest_path: Path = PREDICTION_LATEST_JSON,
    state_path: Path = ALERT_STATE_PATH,
    output_path: Path = OPPORTUNITY_ALERTS_PATH,
    docs_path: Path = DOCS_OPPORTUNITIES_JSON,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    latest = _load_json(latest_path, {})
    games = latest.get("games", []) if isinstance(latest, dict) else []
    now_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    now_iso = now_dt.isoformat()

    candidates: list[dict[str, Any]] = []
    current_game_ids: set[str] = set()
    current_phase_by_game: dict[str, str] = {}
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = str(game.get("game_id") or "")
        if game_id:
            current_game_ids.add(game_id)
            current_phase_by_game[game_id] = "live" if _is_live(game) else "pregame"
        candidates.extend(_candidate_records(game))

    state = _load_json(state_path, {})
    if not isinstance(state, dict):
        state = {}
    opportunities = _migrate_legacy_seen(state, candidates, now_iso)
    recent_events = state.get("recent_events", [])
    if not isinstance(recent_events, list):
        recent_events = []

    events: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    observed_keys: set[str] = set()

    for candidate in candidates:
        key = candidate["lifecycle_key"]
        observed_keys.add(key)
        previous = opportunities.get(key)
        alertable = candidate["level"] in ALERT_LEVELS
        was_active = bool(previous and previous.get("active"))
        event: dict[str, Any] | None = None

        if alertable:
            if previous is None:
                event = _event(candidate, "new", now_iso, None)
            elif not was_active:
                event = _event(candidate, "reactivated", now_iso, previous)
            elif candidate["side"] != previous.get("side"):
                event = _event(candidate, "side_flip", now_iso, previous)
            elif LEVEL_RANK[candidate["level"]] > LEVEL_RANK.get(
                str(previous.get("level", "neutral")), 0
            ):
                event = _event(candidate, "escalated", now_iso, previous)
            elif LEVEL_RANK[candidate["level"]] < LEVEL_RANK.get(
                str(previous.get("level", "neutral")), 0
            ):
                event = _event(candidate, "downgraded", now_iso, previous)
            else:
                baseline_edge = _number(previous.get("last_notified_edge"))
                if baseline_edge is None:
                    baseline_edge = _number(previous.get("edge")) or 0.0
                edge_change = abs(candidate["edge"] - baseline_edge)
                material = edge_change >= MATERIAL_EDGE_CHANGE[candidate["market"]]
                last_notified = _parse_time(previous.get("last_notified_at_utc"))
                cooldown_ok = (
                    last_notified is None
                    or now_dt - last_notified >= timedelta(minutes=COOLDOWN_MINUTES)
                )
                if material and cooldown_ok:
                    event = _event(
                        candidate,
                        "material_change",
                        now_iso,
                        previous,
                        reason=f"edge_changed_by_{edge_change:.2f}",
                    )

            first_seen = previous.get("first_seen_at_utc") if previous else now_iso
            event_notifies = bool(event and event.get("notify"))
            last_notified_at = (
                now_iso
                if event_notifies
                else (previous or {}).get("last_notified_at_utc")
            )
            last_notified_edge = (
                candidate["edge"]
                if event_notifies
                else (previous or {}).get("last_notified_edge")
            )
            if last_notified_edge is None:
                last_notified_edge = candidate["edge"]
            opportunities[key] = {
                **candidate,
                "active": True,
                "first_seen_at_utc": first_seen,
                "last_seen_at_utc": now_iso,
                "last_notified_at_utc": last_notified_at,
                "last_notified_edge": last_notified_edge,
            }
            active_record = {**candidate, "evaluated_at_utc": now_iso}
            active_record["is_new"] = bool(event and event["event_type"] == "new")
            active_record["lifecycle_event"] = event["event_type"] if event else None
            active.append(active_record)
        else:
            if was_active:
                resolved_candidate = {**candidate, "level": candidate["level"]}
                event = _event(
                    resolved_candidate,
                    "resolved",
                    now_iso,
                    previous,
                    reason="edge_below_alert_threshold",
                )
            opportunities[key] = {
                **candidate,
                "active": False,
                "first_seen_at_utc": (previous or {}).get("first_seen_at_utc"),
                "last_seen_at_utc": now_iso,
                "last_notified_at_utc": (
                    now_iso if event else (previous or {}).get("last_notified_at_utc")
                ),
                "last_notified_edge": (
                    candidate["edge"]
                    if event
                    else (previous or {}).get("last_notified_edge")
                ),
                "resolved_at_utc": now_iso if event else (previous or {}).get("resolved_at_utc"),
            }

        if event:
            events.append(event)

    # Resolve a prior pregame/live opportunity when the same game has moved to
    # a different phase. Do not resolve old games merely because latest.json no
    # longer contains them.
    for key, previous in list(opportunities.items()):
        if key in observed_keys or not previous.get("active"):
            continue
        game_id = str(previous.get("game_id") or "")
        if game_id not in current_game_ids:
            continue
        current_phase = current_phase_by_game.get(game_id)
        if current_phase and current_phase != previous.get("phase"):
            candidate = _resolution_candidate(previous)
            event = _event(
                candidate,
                "resolved",
                now_iso,
                previous,
                reason="phase_changed",
            )
            events.append(event)
            opportunities[key] = {
                **previous,
                "active": False,
                "last_seen_at_utc": now_iso,
                "last_notified_at_utc": now_iso,
                "last_notified_edge": previous.get("edge"),
                "resolved_at_utc": now_iso,
            }

    recent_events = (events + recent_events)[:RECENT_EVENT_LIMIT]
    notification_events = [event for event in events if event.get("notify")]

    # Bound retained opportunity state without discarding active records.
    if len(opportunities) > 1000:
        active_items = [
            (key, value) for key, value in opportunities.items() if value.get("active")
        ]
        inactive_items = sorted(
            [
                (key, value)
                for key, value in opportunities.items()
                if not value.get("active")
            ],
            key=lambda item: item[1].get("last_seen_at_utc", ""),
            reverse=True,
        )
        opportunities = dict((active_items + inactive_items)[:1000])

    payload = {
        "generated_at_utc": now_iso,
        "source_generated_at_utc": (
            latest.get("generated_at_utc") if isinstance(latest, dict) else None
        ),
        "active_opportunities": active,
        # Backwards-compatible name retained for downstream consumers. It now
        # means newly created lifecycle alerts, not every active opportunity.
        "new_alerts": [
            event for event in notification_events if event["event_type"] == "new"
        ],
        "notification_events": notification_events,
        "recent_events": recent_events,
        "summary": {
            "active_count": len(active),
            "notification_event_count": len(notification_events),
            "new_count": sum(e["event_type"] == "new" for e in notification_events),
            "very_strong_count": sum(a["level"] == "very_strong" for a in active),
            "strong_count": sum(a["level"] == "strong" for a in active),
        },
        "policy": {
            "spread_thresholds": SPREAD_THRESHOLDS,
            "total_thresholds": TOTAL_THRESHOLDS,
            "alert_levels": sorted(ALERT_LEVELS),
            "cooldown_minutes": COOLDOWN_MINUTES,
            "material_edge_change": MATERIAL_EDGE_CHANGE,
            "lifecycle_events": [
                "new",
                "escalated",
                "downgraded",
                "side_flip",
                "reactivated",
                "material_change",
                "resolved",
            ],
            "calibration": "static_initial_policy",
        },
    }
    _write_json(output_path, payload)
    _write_json(docs_path, payload)
    _write_json(
        state_path,
        {
            "updated_at_utc": now_iso,
            "opportunities": opportunities,
            "recent_events": recent_events,
        },
    )
    return payload
