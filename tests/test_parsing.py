from datetime import datetime

import pandas as pd

from src.parsing import parse_quarter_scores, parse_scoreboard_event


def test_parse_scoreboard_event_valid():
    event = {
        "id": "123",
        "date": "2026-07-01T19:00:00Z",
        "season": 2026,
        "status": {"type": {"name": "STATUS_FINAL", "detail": "Final", "completed": True}},
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"id": "1", "name": "Home", "abbreviation": "HME"}, "score": "75"},
                    {"homeAway": "away", "team": {"id": "2", "name": "Away", "abbreviation": "AWY"}, "score": 68},
                ],
                "venue": {"fullName": "Arena"},
                "neutralSite": False,
            }
        ],
    }
    result = parse_scoreboard_event(event)
    assert result["game_id"] == "123"
    assert result["season"] == 2026
    assert result["home_score"] == 75
    assert result["away_score"] == 68
    assert result["completed"] is True


def test_parse_quarter_scores_regular_game():
    summary = {
        "date": "2026-07-01T19:00:00Z",
        "boxscore": {
            "teams": [
                {
                    "team": {"id": "1", "name": "Home", "abbreviation": "HME"},
                    "linescores": [
                        {"period": 1, "score": 20},
                        {"period": 2, "score": 18},
                        {"period": 3, "score": 22},
                        {"period": 4, "score": 15},
                    ],
                },
                {
                    "team": {"id": "2", "name": "Away", "abbreviation": "AWY"},
                    "linescores": [
                        {"period": 1, "score": 18},
                        {"period": 2, "score": 16},
                        {"period": 3, "score": 20},
                        {"period": 4, "score": 12},
                    ],
                },
            ]
        },
    }
    rows = parse_quarter_scores(summary, "123")
    assert len(rows) == 2
    assert rows[0]["q1"] == 20
    assert rows[1]["q4"] == 12
    assert rows[0]["total_periods"] == 4
    assert rows[1]["updated_at_utc"] == pd.Timestamp("2026-07-01T19:00:00Z")
