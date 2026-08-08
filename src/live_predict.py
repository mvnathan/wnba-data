from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .config import PRODUCTION_MODEL_PATH


def _load_model() -> dict[str, Any]:
    model_path = Path(PRODUCTION_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Production model not found: {model_path}")
    return joblib.load(model_path)


def _game_elapsed_fraction(period: int | None, clock: str | None) -> float:
    if period is None or not clock:
        return 0.0
    try:
        minutes, seconds = [int(x) for x in clock.split(":")]
    except ValueError:
        return 0.0
    clock_seconds = minutes * 60 + seconds
    period_length = 10 * 60
    elapsed_in_period = period_length - clock_seconds
    total_seconds = 4 * period_length
    passed = (period - 1) * period_length + max(0, min(period_length, elapsed_in_period))
    return min(max(passed / total_seconds, 0.0), 1.0)


def _live_projection(row: dict[str, Any], pregame: dict[str, Any]) -> dict[str, float]:
    fraction = _game_elapsed_fraction(row.get("period"), row.get("clock"))
    home_score = float(row.get("home_score", 0) or 0)
    away_score = float(row.get("away_score", 0) or 0)
    pre_home = float(pregame.get("home_score", 0) or 0)
    pre_away = float(pregame.get("away_score", 0) or 0)
    remaining = max(0.0, 1.0 - fraction)
    home_proj = max(home_score, home_score + remaining * (pre_home - home_score))
    away_proj = max(away_score, away_score + remaining * (pre_away - away_score))
    return {"home_final": home_proj, "away_final": away_proj, "final_margin": home_proj - away_proj, "final_total": home_proj + away_proj}


def project_live_game(game_state: dict[str, Any], pregame: dict[str, Any]) -> dict[str, float]:
    return _live_projection(game_state, pregame)
