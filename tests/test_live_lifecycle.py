from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.live_monitor import monitor_live_games
from src.predict_pregame import predict_today
from src.dashboard import build_dashboard


def test_end_to_end_game_lifecycle(tmp_path, monkeypatch):
    live_state = tmp_path / "live_state.json"
    live_snapshots = tmp_path / "live_snapshots.parquet"
    quarter_events = tmp_path / "quarter_events.parquet"
    prediction_history = tmp_path / "prediction_history.parquet"
    latest_json = tmp_path / "latest.json"
    latest_csv = tmp_path / "latest.csv"
    docs_dir = tmp_path / "docs"

    model_payload = {
        "feature_columns": ["feature_a"],
        "models": {"home_win": type("M", (), {"predict": lambda self, X: [0.7]})()},
        "holdout_residuals": {"home_win": [0.1]},
    }

    schedule_df = pd.DataFrame(
        [{"game_id": "123", "home_team_id": "1", "away_team_id": "2", "game_date_utc": pd.Timestamp("2026-08-08T19:00:00Z"), "feature_a": 1.0}]
    )

    monkeypatch.setattr("src.predict_pregame.PRODUCTION_MODEL_PATH", tmp_path / "production_model.joblib")
    monkeypatch.setattr("src.predict_pregame.PREDICTION_LATEST_JSON", latest_json)
    monkeypatch.setattr("src.predict_pregame.PREDICTION_LATEST_CSV", latest_csv)
    monkeypatch.setattr("src.predict_pregame.PREDICTION_HISTORY_PATH", prediction_history)
    monkeypatch.setattr("src.predict_pregame.build_model_features", lambda *_: schedule_df)
    monkeypatch.setattr("src.predict_pregame.joblib.load", lambda _: model_payload)
    (tmp_path / "production_model.joblib").write_bytes(b"dummy")

    monkeypatch.setattr("src.live_monitor.LIVE_STATE_PATH", live_state)
    monkeypatch.setattr("src.live_monitor.LIVE_SNAPSHOTS_PATH", live_snapshots)
    monkeypatch.setattr("src.live_monitor.QUARTER_EVENTS_PATH", quarter_events)
    monkeypatch.setattr("src.live_monitor._today_chicago_date", lambda: "20260808")

    # PREGAME: produce predictions from pregame model
    result = predict_today(date(2026, 8, 8))
    assert result["games"][0]["home_win"] == 0.7
    assert latest_json.exists()
    assert latest_csv.exists()
    assert prediction_history.exists()

    # START
    payload_start = {
        "events": [
            {
                "id": "123",
                "date": "2026-08-08T19:00:00Z",
                "season": 2026,
                "status": {"type": {"name": "STATUS_IN_PROGRESS", "detail": "Q1", "completed": False, "period": 1, "displayClock": "10:00"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "1", "name": "Home", "abbreviation": "HME"}, "score": 0},
                            {"homeAway": "away", "team": {"id": "2", "name": "Away", "abbreviation": "AWY"}, "score": 0},
                        ],
                        "venue": {"fullName": "Arena"},
                        "neutralSite": False,
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr("src.live_monitor._fetch_scoreboard", lambda date_str: payload_start)
    result_start = monitor_live_games()
    assert result_start["games"][0]["period"] == 1

    # END_Q1
    payload_end_q1 = {
        "events": [
            {
                "id": "123",
                "date": "2026-08-08T19:10:00Z",
                "season": 2026,
                "status": {"type": {"name": "STATUS_IN_PROGRESS", "detail": "Q2", "completed": False, "period": 2, "displayClock": "10:00"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "1", "name": "Home", "abbreviation": "HME"}, "score": 20},
                            {"homeAway": "away", "team": {"id": "2", "name": "Away", "abbreviation": "AWY"}, "score": 18},
                        ],
                        "venue": {"fullName": "Arena"},
                        "neutralSite": False,
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr("src.live_monitor._fetch_scoreboard", lambda date_str: payload_end_q1)
    result_end_q1 = monitor_live_games()
    assert result_end_q1["games"][0]["period"] == 2

    # HALFTIME
    payload_halftime = {
        "events": [
            {
                "id": "123",
                "date": "2026-08-08T19:20:00Z",
                "season": 2026,
                "status": {"type": {"name": "STATUS_IN_PROGRESS", "detail": "Q3", "completed": False, "period": 3, "displayClock": "10:00"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "1", "name": "Home", "abbreviation": "HME"}, "score": 40},
                            {"homeAway": "away", "team": {"id": "2", "name": "Away", "abbreviation": "AWY"}, "score": 35},
                        ],
                        "venue": {"fullName": "Arena"},
                        "neutralSite": False,
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr("src.live_monitor._fetch_scoreboard", lambda date_str: payload_halftime)
    result_halftime = monitor_live_games()
    assert result_halftime["games"][0]["period"] == 3

    # END_Q3
    payload_end_q3 = {
        "events": [
            {
                "id": "123",
                "date": "2026-08-08T19:30:00Z",
                "season": 2026,
                "status": {"type": {"name": "STATUS_IN_PROGRESS", "detail": "Q4", "completed": False, "period": 4, "displayClock": "10:00"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "1", "name": "Home", "abbreviation": "HME"}, "score": 60},
                            {"homeAway": "away", "team": {"id": "2", "name": "Away", "abbreviation": "AWY"}, "score": 55},
                        ],
                        "venue": {"fullName": "Arena"},
                        "neutralSite": False,
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr("src.live_monitor._fetch_scoreboard", lambda date_str: payload_end_q3)
    result_end_q3 = monitor_live_games()
    assert result_end_q3["games"][0]["period"] == 4

    # FINAL
    payload_final = {
        "events": [
            {
                "id": "123",
                "date": "2026-08-08T19:40:00Z",
                "season": 2026,
                "status": {"type": {"name": "STATUS_FINAL", "detail": "Final", "completed": True, "period": 4, "displayClock": "0:00"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "1", "name": "Home", "abbreviation": "HME"}, "score": 80},
                            {"homeAway": "away", "team": {"id": "2", "name": "Away", "abbreviation": "AWY"}, "score": 72},
                        ],
                        "venue": {"fullName": "Arena"},
                        "neutralSite": False,
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr("src.live_monitor._fetch_scoreboard", lambda date_str: payload_final)
    result_final = monitor_live_games()
    assert result_final["games"][0]["status"] == "STATUS_FINAL"

    # Verify live state transitions and no duplicate quarter events
    loaded_state = json.loads(live_state.read_text(encoding="utf-8"))
    assert loaded_state["123"]["last_seen_period"] == 4
    assert loaded_state["123"]["last_event_type"] == "FINAL"

    loaded_events = pd.read_parquet(quarter_events)
    assert len(loaded_events) == len({(row["game_id"], row["event_type"]) for _, row in loaded_events.iterrows()})

    loaded_history = pd.read_parquet(prediction_history)
    assert len(loaded_history) == 1

    monkeypatch.setattr("src.dashboard.PREDICTION_LATEST_JSON", latest_json)
    monkeypatch.setattr("src.dashboard.PREDICTION_HISTORY_PATH", prediction_history)
    monkeypatch.setattr("src.dashboard.DOCS_DIR", docs_dir)
    dashboard_result = build_dashboard()
    assert dashboard_result["latest"]["target_date"] == "2026-08-08"
    assert dashboard_result["history"]
