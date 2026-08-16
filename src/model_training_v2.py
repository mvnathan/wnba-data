from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

from .config import (
    MODEL_LEADERBOARD_PATH,
    MODEL_METADATA_PATH,
    PRODUCTION_MODEL_PATH,
)
from .features import build_model_features
from .train_models import (
    CLASSIFICATION_SPECS,
    REGRESSION_SPECS,
    TARGETS,
    _build_pipeline,
    _evaluate_classification,
    _evaluate_regression,
    _prepare_feature_matrix,
)


MAX_WALK_FORWARD_SPLITS = 5
MIN_TARGET_ROWS = 60
MIN_CALIBRATION_ROWS = 40


@dataclass
class ProbabilityCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a fitted classifier with an out-of-fold isotonic calibrator."""

    base_estimator: Any
    calibrator: Any

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = np.asarray(self.base_estimator.predict_proba(X)[:, 1], dtype=float)
        calibrated = np.asarray(self.calibrator.predict(raw), dtype=float)
        calibrated = np.clip(calibrated, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)



def _time_series_splitter(row_count: int) -> TimeSeriesSplit:
    # TimeSeriesSplit needs enough observations to populate every fold.
    n_splits = min(MAX_WALK_FORWARD_SPLITS, max(2, row_count // 100))
    return TimeSeriesSplit(n_splits=n_splits)



def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    aggregated: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            aggregated[key] = float(values.mean())
    return aggregated



def _selection_score(target_type: str, metrics: dict[str, float]) -> float:
    if target_type == "classification":
        return -metrics.get("log_loss", float("inf"))
    return -metrics.get("rmse", float("inf"))



def _model_specs(target_type: str) -> list[tuple[str, Any]]:
    return CLASSIFICATION_SPECS if target_type == "classification" else REGRESSION_SPECS



def _walk_forward_evaluate(
    dataset: pd.DataFrame,
    target: str,
    target_type: str,
    model_name: str,
    estimator: Any,
) -> tuple[dict[str, float], pd.Series]:
    X, _ = _prepare_feature_matrix(dataset)
    y = dataset[target].reset_index(drop=True)
    X = X.reset_index(drop=True)
    splitter = _time_series_splitter(len(dataset))

    fold_metrics: list[dict[str, float]] = []
    oof = pd.Series(np.nan, index=dataset.index, dtype=float)

    for train_idx, valid_idx in splitter.split(X):
        pipeline = _build_pipeline(estimator)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

        if target_type == "classification":
            probability = pipeline.predict_proba(X.iloc[valid_idx])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            metrics = _evaluate_classification(y.iloc[valid_idx], prediction, probability)
            values = probability
        else:
            values = pipeline.predict(X.iloc[valid_idx])
            metrics = _evaluate_regression(y.iloc[valid_idx], values)

        fold_metrics.append(metrics)
        oof.iloc[valid_idx] = np.asarray(values, dtype=float)

    metrics = _aggregate_metrics(fold_metrics)
    metrics["folds"] = float(splitter.n_splits)
    return metrics, oof



def _fit_calibrated_classifier(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    oof_probability: pd.Series,
) -> Any:
    final_model = _build_pipeline(estimator)
    final_model.fit(X, y)

    mask = oof_probability.notna()
    if int(mask.sum()) < MIN_CALIBRATION_ROWS or y.loc[mask].nunique() < 2:
        return final_model

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(oof_probability.loc[mask].to_numpy(), y.loc[mask].to_numpy())
    return ProbabilityCalibratedClassifier(final_model, calibrator)



def train_models() -> dict[str, Any]:
    features = build_model_features()
    if features.empty:
        raise RuntimeError("No feature data available")

    features = features.copy()
    features["game_date_utc"] = pd.to_datetime(features["game_date_utc"], utc=True)
    features = features.sort_values(["game_date_utc", "game_id"]).reset_index(drop=True)

    leaderboard_rows: list[dict[str, Any]] = []
    production_models: dict[str, Any] = {}
    selected_models: dict[str, str] = {}
    out_of_fold_residuals: dict[str, list[float]] = {}
    walk_forward_metrics: dict[str, dict[str, float]] = {}

    for target, target_type in TARGETS.items():
        mask = features[target].notna()
        if int(mask.sum()) < MIN_TARGET_ROWS:
            continue

        dataset = features.loc[mask].copy().reset_index(drop=True)
        X, feature_columns = _prepare_feature_matrix(dataset)
        y = dataset[target].reset_index(drop=True)

        best_score = float("-inf")
        best_name = ""
        best_estimator: Any = None
        best_oof = pd.Series(dtype=float)
        best_metrics: dict[str, float] = {}

        for model_name, estimator in _model_specs(target_type):
            if estimator is None:
                continue
            try:
                metrics, oof = _walk_forward_evaluate(
                    dataset, target, target_type, model_name, estimator
                )
            except Exception as exc:
                leaderboard_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "target_type": target_type,
                        "rows": len(dataset),
                        "status": "error",
                        "error": str(exc),
                    }
                )
                continue

            leaderboard_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "target_type": target_type,
                    "rows": len(dataset),
                    "status": "ok",
                    **metrics,
                }
            )

            score = _selection_score(target_type, metrics)
            if score > best_score:
                best_score = score
                best_name = model_name
                best_estimator = clone(estimator)
                best_oof = oof
                best_metrics = metrics

        if best_estimator is None:
            continue

        if target_type == "classification":
            final_model = _fit_calibrated_classifier(best_estimator, X, y, best_oof)
            residual_mask = best_oof.notna()
            residual_values = y.loc[residual_mask] - best_oof.loc[residual_mask]
        else:
            final_model = _build_pipeline(best_estimator)
            final_model.fit(X, y)
            residual_mask = best_oof.notna()
            residual_values = y.loc[residual_mask] - best_oof.loc[residual_mask]

        production_models[target] = final_model
        selected_models[target] = best_name
        out_of_fold_residuals[target] = residual_values.astype(float).tolist()
        walk_forward_metrics[target] = best_metrics

    leaderboard = pd.DataFrame(leaderboard_rows)
    MODEL_LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(MODEL_LEADERBOARD_PATH, index=False)

    feature_columns = _prepare_feature_matrix(features)[1]
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "trainer_version": 2,
        "validation": "walk_forward_time_series",
        "max_walk_forward_splits": MAX_WALK_FORWARD_SPLITS,
        "model_count": len(production_models),
        "targets": list(production_models.keys()),
        "selected_models": selected_models,
        "walk_forward_metrics": walk_forward_metrics,
        "probability_calibration": "isotonic_on_oof_predictions",
        "uncertainty_residuals": "out_of_fold_only",
        "feature_columns": feature_columns,
    }
    MODEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    PRODUCTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metadata": metadata,
            "models": production_models,
            "feature_columns": feature_columns,
            # Keep the existing key for prediction compatibility, but these are now
            # genuinely out-of-fold residuals rather than in-sample residuals.
            "holdout_residuals": out_of_fold_residuals,
            "out_of_fold_residuals": out_of_fold_residuals,
        },
        PRODUCTION_MODEL_PATH,
        compress=3,
    )

    return {
        "status": "ok",
        "models_trained": len(production_models),
        "leaderboard": leaderboard,
        "metadata": metadata,
    }



def main() -> None:
    result = train_models()
    print(json.dumps({key: value for key, value in result.items() if key != "leaderboard"}, indent=2))
    print("Saved leaderboard to", MODEL_LEADERBOARD_PATH)
    print("Saved production model to", PRODUCTION_MODEL_PATH)


if __name__ == "__main__":
    main()
