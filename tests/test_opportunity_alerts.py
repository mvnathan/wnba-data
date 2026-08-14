from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.opportunity_alerts import build_opportunity_alerts


def _game(*, margin: float = 0.0, spread: float = -8.0) -> dict:
    return {
        "game_id": "game-1",
        "game_date_utc": "2026-08-14T23:30:00+00:00",
        "away_abbr": "DAL",
        "home_abbr": "IND",
        "status": "STATUS_SCHEDULED",
        "predicted_margin": margin,
        "predicted_total": 172.0,
        "market_home_spread": spread,
        "market_total": 186.0,
        "market_bookmaker": "DraftKings",
    }


def _run(tmp_path, game, now):
    latest = tmp_path / "latest.json"
    state = tmp_path / "state.json"
    output = tmp_path / "alerts.json"
    docs = tmp_path / "docs.json"
    latest.write_text(json.dumps({"generated_at_utc": now.isoformat(), "games": [game]}))
    return build_opportunity_alerts(latest, state, output, docs, now=now)


def test_alert_lifecycle_dedupes_escalates_resolves_and_reactivates(tmp_path):
    now = datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc)

    first = _run(tmp_path, _game(margin=3.0, spread=-8.0), now)
    spread = [e for e in first["notification_events"] if e["market"] == "spread"]
    assert spread[0]["event_type"] == "new"
    assert spread[0]["level"] == "strong"

    unchanged = _run(
        tmp_path,
        _game(margin=3.0, spread=-8.0),
        now + timedelta(minutes=5),
    )
    assert not [e for e in unchanged["notification_events"] if e["market"] == "spread"]

    escalated = _run(
        tmp_path,
        _game(margin=0.0, spread=-8.0),
        now + timedelta(minutes=10),
    )
    spread = [e for e in escalated["notification_events"] if e["market"] == "spread"]
    assert spread[0]["event_type"] == "escalated"
    assert spread[0]["level"] == "very_strong"

    resolved = _run(
        tmp_path,
        _game(margin=7.0, spread=-8.0),
        now + timedelta(minutes=15),
    )
    spread = [e for e in resolved["notification_events"] if e["market"] == "spread"]
    assert spread[0]["event_type"] == "resolved"

    reactivated = _run(
        tmp_path,
        _game(margin=2.0, spread=-8.0),
        now + timedelta(minutes=20),
    )
    spread = [e for e in reactivated["notification_events"] if e["market"] == "spread"]
    assert spread[0]["event_type"] == "reactivated"


def test_material_change_respects_cooldown(tmp_path):
    now = datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc)

    # Start with a strong DAL spread edge of 4.2 points.
    _run(tmp_path, _game(margin=3.8, spread=-8.0), now)

    # Move to 5.7 inside the cooldown. The move is 1.5 points, so no event.
    during_cooldown = _run(
        tmp_path,
        _game(margin=2.3, spread=-8.0),
        now + timedelta(minutes=10),
    )
    assert not [
        e for e in during_cooldown["notification_events"]
        if e["market"] == "spread" and e["event_type"] == "material_change"
    ]

    # After the cooldown, 6.1 is still in the strong bucket and is only 1.9
    # points from the last-notified 4.2 baseline, so there is still no event.
    below_threshold = _run(
        tmp_path,
        _game(margin=1.9, spread=-8.0),
        now + timedelta(minutes=31),
    )
    assert not [
        e for e in below_threshold["notification_events"]
        if e["market"] == "spread" and e["event_type"] == "material_change"
    ]

    # A 2.1-point move from the 4.2 last-notified baseline yields a 6.3 edge,
    # still below the 7.0 very-strong threshold, so this is a material change
    # rather than an escalation.
    after_cooldown = _run(
        tmp_path,
        _game(margin=1.7, spread=-8.0),
        now + timedelta(minutes=32),
    )
    spread = [
        e for e in after_cooldown["notification_events"]
        if e["market"] == "spread"
    ]
    assert spread[0]["event_type"] == "material_change"
