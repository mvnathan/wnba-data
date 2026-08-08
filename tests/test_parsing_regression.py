from __future__ import annotations

import pandas as pd

from src.parsing import parse_quarter_scores, parse_scoreboard_event


def test_parse_scoreboard_event_live_status_and_clock():
    event = {
        "id": "123",
        "date": "2026-08-08T19:00:00Z",
        "season": 2026,
        "status": {
            "type": {
                "name": "STATUS_IN_PROGRESS",
                "detail": "Q3",
                "completed": False,
                "period": 3,
                "displayClock": "05:00",
            }
        },
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

    result = parse_scoreboard_event(event)
    assert result is not None
    assert result["game_id"] == "123"
    assert result["period"] == 3
    assert result["clock"] == "05:00"
    assert result["status"] == "STATUS_IN_PROGRESS"
    assert result["status_detail"] == "Q3"
    assert result["home_score"] == 65
    assert result["away_score"] == 61


def test_parse_quarter_scores_from_play_by_play_cumulative_totals():
    summary = {
        "date": "2026-08-08T19:00:00Z",
        "boxscore": {
            "teams": [
                {"team": {"id": "1", "name": "Home", "abbreviation": "HME"}, "homeAway": "home"},
                {"team": {"id": "2", "name": "Away", "abbreviation": "AWY"}, "homeAway": "away"},
            ]
        },
        "plays": [
            {"period": 1, "homeScore": 20, "awayScore": 18},
            {"period": 2, "homeScore": 38, "awayScore": 34},
            {"period": 3, "homeScore": 60, "awayScore": 54},
            {"period": 4, "homeScore": 75, "awayScore": 66},
        ],
    }

    rows = parse_quarter_scores(summary, "123")
    assert len(rows) == 2
    home = next(row for row in rows if row["team_id"] == "1")
    away = next(row for row in rows if row["team_id"] == "2")
    assert home["q1"] == 20
    assert home["q2"] == 18
    assert home["q3"] == 22
    assert home["q4"] == 15
    assert away["q1"] == 18
    assert away["q2"] == 16
    assert away["q3"] == 20
    assert away["q4"] == 12
    assert home["total_periods"] == 4
    assert away["updated_at_utc"] == pd.Timestamp("2026-08-08T19:00:00Z")
