from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.live_monitor import monitor_live_games


def test_monitor_live_games_updates_state_and_snapshots(tmp_path, monkeypatch):
    live_state = tmp_path / "live_state.json"
    live_snapshots = tmp_path / "live_snapshots.parquet"
    sample_payload = {
        "events": [
            {
                "id": "123",
                "date": "2026-07-01T19:00:00Z",
                "season": 2026,
                "status": {"type": {"name": "STATUS_IN_PROGRESS", "detail": "Q3", "completed": False, "period": 3, "displayClock": "05:00"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "1", "name": "Home", "abbreviation": "HME"}, "score": 65},
                            {"homeAway": "away", "team": {"id": "2", "name": "Away", "abbreviation": "AWY"}, "score": 61},
                        ],
                        "venue": {"fullName": "Arena"},
                        "neutralSite": False,
                    }
                ],
            }
        ]
    }

    monkeypatch.setattr("src.live_monitor.LIVE_STATE_PATH", live_state)
    monkeypatch.setattr("src.live_monitor.LIVE_SNAPSHOTS_PATH", live_snapshots)
    monkeypatch.setattr("src.live_monitor._today_chicago_date", lambda: "20260701")
    monkeypatch.setattr("src.live_monitor._fetch_scoreboard", lambda date_str: sample_payload)

    result = monitor_live_games()
    assert result["games"][0]["game_id"] == "123"
    assert live_state.exists()
    assert live_snapshots.exists()
    loaded = json.loads(live_state.read_text(encoding="utf-8"))
    assert loaded["123"]["last_seen_period"] == 3
    df = pd.read_parquet(live_snapshots)
    assert not df.empty
