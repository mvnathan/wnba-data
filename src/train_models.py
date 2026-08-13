from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from .config import (
    FEATURES_PATH,
    MODEL_LEADERBOARD_PATH,
    MODEL_METADATA_PATH,
    PRODUCTION_MODEL_PATH,
)
from .features import build_model_features

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore
    LGBMRegressor = None  # type: ignore

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore
    CatBoostRegressor = None  # type: ignore


CLASSIFICATION_SPECS = []
REGRESSION_SPECS = []

if XGBClassifier is not None:
    CLASSIFICATION_SPECS.append(("xgboost", XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, random_state=42, n_jobs=-1)))
if LGBMClassifier is not None:
    CLASSIFICATION_SPECS.append(("lightgbm", LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
if CatBoostClassifier is not None:
    CLASSIFICATION_SPECS.append(("catboost", CatBoostClassifier(verbose=0, random_state=42, iterations=200)))
CLASSIFICATION_SPECS.extend(
    [
        ("logistic_regression", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")),
        ("extra_trees", ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]
)

if XGBRegressor is not None:
    REGRESSION_SPECS.append(("xgboost", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
if LGBMRegressor is not None:
    REGRESSION_SPECS.append(("lightgbm", LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
if CatBoostRegressor is not None:
    REGRESSION_SPECS.append(("catboost", CatBoostRegressor(verbose=0, random_state=42, iterations=200)))
REGRESSION_SPECS.extend(
    [
        ("ridge", Ridge(alpha=1.0, random_state=42)),
        ("extra_trees", ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("hist_gb", HistGradientBoostingRegressor(max_iter=200, random_state=42)),
    ]
)

TARGETS = {
    "home_win": "classification",
    "home_score": "regression",
    "away_score": "regression",
    "full_margin": "regression",
    "full_total": "regression",
    "home_first_half": "regression",
    "away_first_half": "regression",
    "first_half_margin": "regression",
    "first_half_total": "regression",
    "home_second_half": "regression",
    "away_second_half": "regression",
    "second_half_margin": "regression",
    "second_half_total": "regression",
    "home_q1": "regression",
    "away_q1": "regression",
    "q1_margin": "regression",
    "q1_total": "regression",
    "home_q2": "regression",
    "away_q2": "regression",
    "q2_margin": "regression",
    "q2_total": "regression",
    "home_q3": "regression",
    "away_q3": "regression",
    "q3_margin": "regression",
    "q3_total": "regression",
    "home_q4": "regression",
    "away_q4": "regression",
    "q4_margin": "regression",
    "q4_total": "regression",
}

NUMERIC_FEATURE_EXCLUDE = {
    "game_id",
    "game_date_utc",
    "season",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "home_abbr",
    "away_abbr",
    "status",
    "status_detail",
    "venue",
    "completed",
    "home_score",
    "away_score",
    "home_q1",
    "away_q1",
    "home_q2",
    "away_q2",
    "home_q3",
    "away_q3",
    "home_q4",
    "away_q4",
    "first_half_margin",
    "first_half_total",
    "second_half_margin",
    "second_half_total",
    "q1_margin",
    "q1_total",
    "q2_margin",
    "q2_total",
    "q3_margin",
    "q3_total",
    "q4_margin",
    "q4_total",
    "period",
    "clock",
    "home_points",
    "home_opp_points",
    "home_margin",
    "home_win",
    "home_q1",
    "home_q2",
    "home_q3",
    "home_q4",
    "home_ot1",
    "home_ot2",
    "home_first_half_points",
    "home_second_half_points",
    "home_regulation_points",
    "home_opp_q1",
    "home_opp_q2",
    "home_opp_q3",
    "home_opp_q4",
    "home_q1_margin",
    "home_q2_margin",
    "home_q3_margin",
    "home_q4_margin",
    "home_first_half_margin",
    "away_points",
    "away_opp_points",
    "away_margin",
    "away_win",
    "away_q1",
    "away_q2",
    "away_q3",
    "away_q4",
    "away_ot1",
    "away_ot2",
    "away_first_half_points",
    "away_second_half_points",
    "away_regulation_points",
    "away_opp_q1",
    "away_opp_q2",
    "away_opp_q3",
    "away_opp_q4",
    "away_q1_margin",
    "away_q2_margin",
    "away_q3_margin",
    "away_q4_margin",
    "away_first_half_margin",
    "diff_score",
    "diff_points",
    "diff_opp_points",
    "diff_margin",
    "diff_win",
    "diff_q1",
    "diff_q2",
    "diff_q3",
    "diff_q4",
    "diff_ot1",
    "diff_ot2",
    "diff_first_half_points",
    "diff_second_half_points",
    "diff_regulation_points",
    "diff_opp_q1",
    "diff_opp_q2",
    "diff_opp_q3",
    "diff_opp_q4",
    "diff_q1_margin",
    "diff_q2_margin",
    "diff_q3_margin",
    "diff_q4_margin",
    "diff_first_half_margin",
}


def _build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clone(estimator)),
        ]
    )


def _prepare_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:

    excluded = NUMERIC_FEATURE_EXCLUDE | set(TARGETS.keys())

    feature_columns = [
        col
        for col in df.columns
        if col not in excluded
        and col not in {"game_id", "game_date_utc"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    return df[feature_columns].astype(float), feature_columns


def _chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_dates = sorted(df["game_date_utc"].dropna().unique())
    if not sorted_dates:
        return df, df
    split_index = int(len(sorted_dates) * 0.8)
    split_date = sorted_dates[max(0, split_index)]
    train = df[df["game_date_utc"] <= split_date].copy()
    holdout = df[df["game_date_utc"] > split_date].copy()
    if holdout.empty:
        holdout = df.iloc[-max(1, int(len(df) * 0.2)) :].copy()
        train = df.drop(holdout.index)
    return train, holdout


def _evaluate_classification(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _evaluate_regression(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _score_for_selection(target_type: str, metrics: dict[str, float]) -> float:
    if target_type == "classification":
        return -metrics.get("log_loss", float("inf"))
    return -metrics.get("rmse", float("inf"))


def _target_model_specs(target_type: str) -> list[tuple[str, Any]]:
    return CLASSIFICATION_SPECS if target_type == "classification" else REGRESSION_SPECS


def train_models() -> dict[str, Any]:
    features = build_model_features()
    if features.empty:
        raise RuntimeError("No feature data available")
    features["game_date_utc"] = pd.to_datetime(features["game_date_utc"], utc=True)

    leaderboard_rows: list[dict[str, Any]] = []
    production_models: dict[str, Any] = {}
    selected_models: dict[str, str] = {}
    holdout_residuals: dict[str, list[float]] = {}

    for target, target_type in TARGETS.items():
        mask = features[target].notna()
        if mask.sum() < 30:
            continue
        dataset = features[mask].copy()
        train, holdout = _chronological_split(dataset)
        if train.empty or holdout.empty:
            continue
        X_train, feature_columns = _prepare_feature_matrix(train)
        X_holdout, _ = _prepare_feature_matrix(holdout)
        y_train = train[target]
        y_holdout = holdout[target]

        best_score = float("-inf")
        best_model = None
        best_name = ""
        best_metrics: dict[str, float] = {}

        for model_name, estimator in _target_model_specs(target_type):
            if estimator is None:
                continue
            pipeline = _build_pipeline(estimator)
            try:
                pipeline.fit(X_train, y_train)
                if target_type == "classification":
                    y_pred = pipeline.predict(X_holdout)
                    y_prob = pipeline.predict_proba(X_holdout)[:, 1]
                    metrics = _evaluate_classification(y_holdout, y_pred, y_prob)
                else:
                    y_pred = pipeline.predict(X_holdout)
                    metrics = _evaluate_regression(y_holdout, y_pred)

                leaderboard_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "target_type": target_type,
                        "train_rows": len(X_train),
                        "holdout_rows": len(X_holdout),
                        **metrics,
                    }
                )
                score = _score_for_selection(target_type, metrics)
                if score > best_score:
                    best_score = score
                    best_model = pipeline
                    best_name = model_name
                    best_metrics = metrics
            except Exception:
                continue

        if best_model is not None:
            production_models[target] = best_model
            selected_models[target] = best_name
            y_pred_full = best_model.predict(_prepare_feature_matrix(dataset)[0])
            holdout_residuals[target] = (dataset[target] - y_pred_full).tolist()

    leaderboard = pd.DataFrame(leaderboard_rows)
    MODEL_LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(MODEL_LEADERBOARD_PATH, index=False)

    metadata = {
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "model_count": len(production_models),
        "targets": list(production_models.keys()),
        "selected_models": selected_models,
        "feature_columns": _prepare_feature_matrix(features)[1],
    }
    metadata_path = MODEL_METADATA_PATH
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    PRODUCTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metadata": metadata,
            "models": production_models,
            "feature_columns": metadata["feature_columns"],
            "holdout_residuals": holdout_residuals,
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
    print(json.dumps({k: v for k, v in result.items() if k != "leaderboard"}, indent=2))
    print("Saved leaderboard to", MODEL_LEADERBOARD_PATH)
    print("Saved production model to", PRODUCTION_MODEL_PATH)


if __name__ == "__main__":
    main()
