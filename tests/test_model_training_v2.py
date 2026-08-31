from __future__ import annotations

import numpy as np

from src.market_odds import _attach_market_benchmark
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


def test_market_benchmark_preserves_independent_model_forecasts():
    row = _attach_market_benchmark(
        {
            "predicted_margin": 8.0,
            "predicted_total": 170.0,
            "home_win_probability": 0.70,
            "away_win_probability": 0.30,
            "market_home_spread": -4.0,
            "market_total": 166.0,
            "consensus_home_spread": -5.0,
            "consensus_total": 167.0,
            "consensus_no_vig_home_win_probability": 0.60,
        }
    )

    assert row["market_used_in_prediction"] is False
    assert row["market_blend_version"] == "independent_model_v3_consensus"

    assert row["model_predicted_margin"] == 8.0
    assert row["market_implied_margin"] == 4.0
    assert row["consensus_implied_margin"] == 5.0
    assert row["predicted_margin"] == 8.0
    assert row["model_market_margin_edge"] == 4.0
    assert row["model_consensus_margin_edge"] == 3.0

    assert row["model_predicted_total"] == 170.0
    assert row["predicted_total"] == 170.0
    assert row["model_market_total_edge"] == 4.0
    assert row["model_consensus_total_edge"] == 3.0

    assert np.isclose(row["model_home_win_probability"], 0.70)
    assert np.isclose(row["home_win_probability"], 0.70)
    assert np.isclose(row["away_win_probability"], 0.30)
    assert np.isclose(row["model_consensus_home_win_edge"], 0.10)

    assert row["market_spread_weight"] == 0.0
    assert row["market_total_weight"] == 0.0
    assert row["market_moneyline_weight"] == 0.0


def test_market_spread_sign_matches_home_minus_away_margin_convention_without_anchor():
    home_favorite = _attach_market_benchmark(
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
    assert home_favorite["predicted_margin"] == 5.25
    assert home_favorite["model_market_margin_edge"] == -5.25

    home_underdog = _attach_market_benchmark(
        {
            "home_abbr": "SEA",
            "away_abbr": "CHI",
            "predicted_margin": 0.2,
            "predicted_total": 173.0,
            "home_win_probability": 0.51,
            "away_win_probability": 0.49,
            "market_home_spread": 4.5,
        }
    )
    assert home_underdog["market_implied_margin"] == -4.5
    assert home_underdog["predicted_margin"] == 0.2
    assert row["market_display_mode"] == "pure_model_vs_dk_and_consensus"
    assert home_underdog["model_market_margin_edge"] == 4.7
