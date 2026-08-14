from __future__ import annotations

import math

from src.live_predict import project_live_game


def test_live_projection_starts_from_pregame_forecast() -> None:
    game_state = {
        "period": None,
        "clock": None,
        "home_score": 0,
        "away_score": 0,
    }
    pregame = {
        "home_score": 86.0,
        "away_score": 82.0,
        "home_win_probability": 0.65,
    }

    result = project_live_game(game_state, pregame)

    assert result["home_final"] == 86.0
    assert result["away_final"] == 82.0
    assert result["final_margin"] == 4.0
   