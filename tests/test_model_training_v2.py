from __future__ import annotations

import numpy as np

from src.market_odds import _apply_market_anchor
from src.model_training_v2 import ProbabilityCalibratedClassifier


class _BaseClassifier:
    def predict_proba(self, X):
        values = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack([1.0 - values, values])


class _IdentityCalibrator:
    def predict(self, values):
        return np.asarray(values, dtype=float)


def test_probability_calibrated_classifier_preserves_probability_shape():
    model = ProbabilityCalibratedClassifier(
        base_estimator=_BaseClassifier(),
        calibrator=_IdentityCalibrator(),
    )
    probabilities = model.predict_proba(np.asarray([[0.2], [0.8]]))

    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities[:, 1], [0.2, 0.8])
    assert model.predict(np.asarray([[0.2], [0.8]])).tolist() == [0, 1]


def test_market_anchor_preserves_model_values_and_blends_forecasts():
    row = _apply_market_anchor(
        {
            "predicted_margin": 8.0,
            "predicted_total": 170.0,
            "home_win_probability": 0.70,
            "away_win_probability": 0.30,
            "market_home_spread": -4.0,
            "market_total": 166.0,
            "market_home_moneyline": -150,
            "market_away_moneyline": 130,
        }
    )

    assert row["model_predicted_margin"] == 8.0
    assert row["market_implied_margin"] == 4.0
    assert row["predicted_margin"] < 8.0
    assert row["predicted_margin"] > 4.0

    assert row["model_predicted_total"] == 170.0
    assert row["predicted_total"] < 170.0
    assert row["predicted_total"] > 166.0

    assert row["model_home_win_probability"] == 0.70
    assert 0.0 < row["home_win_probability"] < 1.0
    assert row["home_win_probability"] + row["away_win_probability"] == 1.0


def test_market_spread_sign_matches_home_minus_away_margin_convention():
    home_favorite = _apply_market_anchor(
        {
            "home_abbr": "CON",
            "away_abbr": "IND",
            "predicted_margin": 5.25,
            "predicted_total": 165.0,
            "home_win_probability": 0.65,
            "away_win_probability": 0.35,
            "market_home_spread": -10.5,
        }
    )
    assert home_favorite["market_implied_margin"] == 10.5
    assert home_favorite["predicted_margin"] > 5.25
    assert home_favorite["projected_winner_abbr"] == "CON"

    home_underdog = _apply_market_anchor(
        {
            "home_abbr": "SEA",
            "away_abbr": "CHI",
            "predicted_margin": 0.2,
            "predicted_total": 173.0,
            "home_win_probability": 0.42,
            "away_win_probability": 0.58,
            "market_home_spread": 2.5,
        }
    )
    assert home_underdog["market_implied_margin"] == -2.5
    assert home_underdog["predicted_margin"] < 0
    assert home_underdog["projected_winner_abbr"] == "CHI"
