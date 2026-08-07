import pandas as pd

from src.update_data import run_daily_update


class DummyClient:
    def fetch_scoreboard(self, day):
        return {"events": []}

    def fetch_summary(self, game_id):
        return {"boxscore": {"teams": []}, "date": None}


def test_run_daily_update_returns_summary(monkeypatch):
    monkeypatch.setattr("src.update_data.ESPNApiClient", lambda: DummyClient())
    monkeypatch.setattr("src.update_data.load_parquet_or_empty", lambda path: pd.DataFrame())
    monkeypatch.setattr("src.update_data.write_parquet_atomic", lambda df, path: None)
    monkeypatch.setattr("src.update_data.write_update_metadata", lambda metadata: None)
    monkeypatch.setattr("src.update_data.build_team_games", lambda games, quarters: pd.DataFrame())

    summary = run_daily_update()
    assert summary["status"] == "ok"
    assert summary["errors"] == {}
