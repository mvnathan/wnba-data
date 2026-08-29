from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone

from .features import build_model_features
from .model_training_v2 import _time_series_splitter
from .train_models import (
    CLASSIFICATION_SPECS,
    REGRESSION_SPECS,
    NUMERIC_FEATURE_EXCLUDE,
    TARGETS,
    _build_pipeline,
    _evaluate_classification,
    _evaluate_regression,
)


OUTPUT_JSON = Path("models/feature_ablation.json")
OUTPUT_CSV = Path("models/feature_ablation.csv")
CORE_TARGETS = ("home_win", "full_margin", "full_total")


def _all_numeric_columns(df: pd.DataFrame) -> list[str]:
    excluded = NUMERIC_FEATURE_EXCLUDE | set(TARGETS.keys()) | {"game_id", "game_date_utc"}
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def _is_rate_or_context(col: str) -> bool:
    keep_tokens = (
        "rolling_5_",
        "win_pct",
        "points_per_game",
        "margin_per_game",
        "days_since_last_game",
        "back_to_back",
        "games_last_7",
        "games_last_14",
        "elo",
        "head_last_3_",
        "head_q1_last_3_",
        "head_fh_last_3_",
    )
    return col == "neutral_site" or any(token in col for token in keep_tokens)


def _is_compact(col: str) -> bool:
    keep_tokens = (
        "rolling_5_points",
        "rolling_5_allowed",
        "rolling_5_margin",
        "rolling_5_total",
        "season_win_pct",
        "home_points_per_game",
        "home_margin_per_game",
        "away_points_per_game",
        "away_margin_per_game",
        "days_since_last_game",
        "back_to_back",
        "games_last_7",
        "games_last_14",
        "elo",
        "head_last_3_margin",
        "head_last_3_total",
    )
    return col == "neutral_site" or any(token in col for token in keep_tokens)


def _drop_redundant_cumulative(col: str) -> bool:
    redundant_tokens = (
        "season_points",
        "season_allowed",
        "season_margin",
        "season_total",
        "season_wins",
        "games_before",
        "last_q",
        "last_first_half",
        "last_second_half",
    )
    return not any(token in col for token in redundant_tokens)


def _feature_sets(columns: list[str]) -> dict[str, list[str]]:
    sets = {
        "current_full": list(columns),
        "pruned_cumulative": [c for c in columns if _drop_redundant_cumulative(c)],
        "rates_form_rest": [c for c in columns if _is_rate_or_context(c)],
        "compact_core": [c for c in columns if _is_compact(c)],
    }
    return {name: cols for name, cols in sets.items() if cols}


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    out: dict[str, float] = {}
    for key in keys:
        vals = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[key] = float(vals.mean())
    return out


def _evaluate_model(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    target_type: str,
    estimator: Any,
) -> dict[str, float]:
    X = dataset[feature_columns].astype(float).reset_index(drop=True)
    y = dataset[target].reset_index(drop=True)
    splitter = _time_series_splitter(len(dataset))
    fold_metrics: list[dict[str, float]] = []

    for train_idx, valid_idx in splitter.split(X):
        model = _build_pipeline(clone(estimator))
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        if target_type == "classification":
            prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            fold_metrics.append(_evaluate_classification(y.iloc[valid_idx], pred, prob))
        else:
            pred = model.predict(X.iloc[valid_idx])
            fold_metrics.append(_evaluate_regression(y.iloc[valid_idx], pred))

    metrics = _aggregate(fold_metrics)
    metrics["folds"] = float(splitter.n_splits)
    return metrics


def _selection_metric(target_type: str, metrics: dict[str, float]) -> float:
    if target_type == "classification":
        return metrics.get("log_loss", float("inf"))
    return metrics.get("rmse", float("inf"))


def run_feature_ablation() -> dict[str, Any]:
    features = build_model_features().copy()
    if features.empty:
        raise RuntimeError("No feature data available")

    features["game_date_utc"] = pd.to_datetime(features["game_date_utc"], utc=True)
    features = features.sort_values(["game_date_utc", "game_id"]).reset_index(drop=True)
    all_columns = _all_numeric_columns(features)
    feature_sets = _feature_sets(all_columns)

    rows: list[dict[str, Any]] = []
    winners: dict[str, Any] = {}

    for target in CORE_TARGETS:
        target_type = TARGETS[target]
        dataset = features.loc[features[target].notna()].copy().reset_index(drop=True)
        specs = CLASSIFICATION_SPECS if target_type == "classification" else REGRESSION_SPECS
        best_overall: dict[str, Any] | None = None

        for set_name, columns in feature_sets.items():
            usable = [c for c in columns if c in dataset.columns]
            if not usable:
                continue
            for model_name, estimator in specs:
                if estimator is None:
                    continue
                try:
                    metrics = _evaluate_model(dataset, usable, target, target_type, estimator)
                    row = {
                        "target": target,
                        "target_type": target_type,
                        "feature_set": set_name,
                        "feature_count": len(usable),
                        "rows": len(dataset),
                        "model": model_name,
                        "status": "ok",
                        **metrics,
                    }
                except Exception as exc:
                    row = {
                        "target": target,
                        "target_type": target_type,
                        "feature_set": set_name,
                        "feature_count": len(usable),
                        "rows": len(dataset),
                        "model": model_name,
                        "status": "error",
                        "error": str(exc),
                    }
                rows.append(row)

                if row["status"] != "ok":
                    continue
                score = _selection_metric(target_type, row)
                if best_overall is None or score < best_overall["selection_score"]:
                    best_overall = {**row, "selection_score": score}

        if best_overall:
            winners[target] = best_overall

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Compare independent-model feature families using walk-forward validation. Market prices are excluded from all feature sets.",
        "core_targets": list(CORE_TARGETS),
        "feature_sets": {name: len(cols) for name, cols in feature_sets.items()},
        "best_by_target": winners,
        "results": rows,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    return result


def main() -> None:
    result = run_feature_ablation()
    print(json.dumps({
        "generated_at_utc": result["generated_at_utc"],
        "feature_sets": result["feature_sets"],
        "best_by_target": result["best_by_target"],
    }, indent=2))


if __name__ == "__main__":
    main()
