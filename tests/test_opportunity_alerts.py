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
    _run(tmp_path, _game(margin=3.0, spread=-8.0), now)

    during_cooldown = _run(
        tmp_path,
        _game(margin=0.5, spread=-8.0),
        now + timedelta(minutes=10),
    )
    assert not [
        e for e in during_cooldown["notification_events"]
        if e["market"] == "spread" and e["event_type"] == "material_change"
    ]

    after_cooldown = _run(
        tmp_path,
        _game(margin=0.5, spread=-8.0),
        now + timedelta(minutes=31),
    )
    spread = [
        e for e in after_cooldown["notification_events"]
        if e["market"] == "spread"
    ]
    assert spread[0]["event_type"] == "material_change"
