from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dashboard import build_dashboard
from src.config import DOCS_HISTORY_JSON, DOCS_LATEST_JSON, PREDICTION_HISTORY_PATH, PREDICTION_LATEST_JSON


def test_build_dashboard_generates_json(tmp_path, monkeypatch):
    latest_path = tmp_path / "latest.json"
    history_path = tmp_path / "prediction_history.parquet"
    docs_latest = tmp_path / "docs" / "latest.json"
    docs_history = tmp_path / "docs" / "history.json"

    monkeypatch.setattr("src.dashboard.PREDICTION_LATEST_JSON", latest_path)
    monkeypatch.setattr("src.dashboard.PREDICTION_HISTORY_PATH", history_path)
    monkeypatch.setattr("src.dashboard.DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr("src.dashboard.DOCS_LATEST_JSON", docs_latest)
    monkeypatch.setattr("src.dashboard.DOCS_HISTORY_JSON", docs_history)

    latest_data = {"generated_at_utc": "2026-07-01T19:00:00Z", "games": []}
    latest_path.write_text(json.dumps(latest_data), encoding="utf-8")
    history_df = pd.DataFrame([
        {"game_id": "123", "prediction_date": "2026-07-01T19:00:00Z"}
    ])
    history_df.to_parquet(history_path, index=False)

    result = build_dashboard()
    assert result["latest"] == latest_data
    assert isinstance(result["history"], list)
    assert docs_latest.exists()
    assert docs_history.exists()
