from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.predict_pregame import predict_today


def test_predict_today_writes_outputs(tmp_path, monkeypatch):
    production_model_path = tmp_path / "production_model.joblib"
    latest_json = tmp_path / "latest.json"
    latest_csv = tmp_path / "latest.csv"
    history_path = tmp_path / "prediction_history.parquet"

    model_payload = {
        "feature_columns": ["feature_a"],
        "models": {"home_win": type("M", (), {"predict": lambda self, X: [0.7]})()},
        "holdout_residuals": {"home_win": [0.1]},
    }

    schedule_df = pd.DataFrame(
        [{"game_id": "123", "home_team_id": "1", "away_team_id": "2", "game_date_utc": pd.Timestamp("2026-07-01T19:00:00Z"), "feature_a": 1.0}]
    )

    monkeypatch.setattr("src.predict_pregame.PRODUCTION_MODEL_PATH", production_model_path)
    monkeypatch.setattr("src.predict_pregame.PREDICTION_LATEST_JSON", latest_json)
    monkeypatch.setattr("src.predict_pregame.PREDICTION_LATEST_CSV", latest_csv)
    monkeypatch.setattr("src.predict_pregame.PREDICTION_HISTORY_PATH", history_path)
    monkeypatch.setattr("src.predict_pregame.build_model_features", lambda *_: schedule_df)
    monkeypatch.setattr("src.predict_pregame.joblib.load", lambda _: model_payload)
    production_model_path.write_bytes(b"dummy")

    result = predict_today(date(2026, 7, 1))
    assert result["games"][0]["home_win"] == 0.7
    assert latest_json.exists()
    assert latest_csv.exists()
    assert history_path.exists()
