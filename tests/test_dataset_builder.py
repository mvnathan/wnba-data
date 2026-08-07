import pandas as pd

from src.dataset_builder import build_team_games


def test_build_team_games_creates_two_rows_per_game():
    games = pd.DataFrame(
        [
            {
                "game_id": "123",
                "game_date_utc": pd.Timestamp("2026-07-01T19:00:00Z"),
                "season": 2026,
                "completed": True,
                "home_team_id": "1",
                "home_team": "Home",
                "home_abbr": "HME",
                "away_team_id": "2",
                "away_team": "Away",
                "away_abbr": "AWY",
                "home_score": 75,
                "away_score": 68,
            }
        ]
    )
    quarters = pd.DataFrame(
        [
            {
                "game_id": "123",
                "team_id": "1",
                "team": "Home",
                "team_abbr": "HME",
                "q1": 20,
                "q2": 18,
                "q3": 22,
                "q4": 15,
                "ot1": None,
                "ot2": None,
                "total_periods": 4,
                "updated_at_utc": pd.Timestamp("2026-07-01T19:00:00Z"),
            },
            {
                "game_id": "123",
                "team_id": "2",
                "team": "Away",
                "team_abbr": "AWY",
                "q1": 18,
                "q2": 16,
                "q3": 20,
                "q4": 12,
                "ot1": None,
                "ot2": None,
                "total_periods": 4,
                "updated_at_utc": pd.Timestamp("2026-07-01T19:00:00Z"),
            },
        ]
    )
    team_games = build_team_games(games, quarters)
    assert len(team_games) == 2
    assert set(team_games["team_id"]) == {"1", "2"}
    assert "opp_q1" in team_games.columns
    assert team_games.loc[team_games["team_id"] == "1", "margin"].iloc[0] == 7
