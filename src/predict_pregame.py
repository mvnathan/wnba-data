from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas import Timestamp

from .config import (
    CHICAGO,
    FEATURES_PATH,
    GAMES_PATH,
    PREDICTION_HISTORY_PATH,
    PREDICTION_LATEST_CSV,
    PREDICTION_LATEST_JSON,
    PRODUCTION_MODEL_PATH,
)
from .features import build_model_features


def _today_chicago() -> date:
    return pd.Timestamp.now(tz=CHICAGO).date()


def _load_production_model() -> dict[str, Any]:
    if not Path(PRODUCTION_MODEL_PATH).exists():
        raise FileNotFoundError(f"Production model not found: {PRODUCTION_MODEL_PATH}")
    return joblib.load(PRODUCTION_MODEL_PATH)


def _build_pregame_data(target_date: date) -> pd.DataFrame:
    features = build_model_features("data")
    if features.empty:
        return pd.DataFrame()
    features["game_date_utc"] = pd.to_datetime(features["game_date_utc"], utc=True)
    return features[features["game_date_utc"].dt.date == target_date]


def _simulate_residuals(model_payload: dict[str, Any], predictions: pd.DataFrame) -> pd.DataFrame:
    samples = []
    residuals = model_payload.get("holdout_residuals", {})
    for _, row in predictions.iterrows():
        sample = {}
        for target in residuals:
            target_resids = np.array(residuals[target])
            if target_resids.size == 0:
                sample[target] = row[target]
                continue
            sample[target] = float(row[target] + np.random.choice(target_resids, size=1)[0])
        samples.append(sample)
    return pd.DataFrame(samples)


def _percentiles(df: pd.Series) -> dict[str, Any]:
    if df.empty:
        return {"p10": None, "median": None, "p90": None}
    quantiles = df.quantile([0.1, 0.5, 0.9])
    return {
        "p10": float(quantiles.loc[0.1]) if 0.1 in quantiles.index else None,
        "median": float(quantiles.loc[0.5]) if 0.5 in quantiles.index else None,
        "p90": float(quantiles.loc[0.9]) if 0.9 in quantiles.index else None,
    }


def predict_today(target_date: date | None = None) -> dict[str, Any]:
    target_date = target_date or _today_chicago()
    schedule = _build_pregame_data(target_date)
    if schedule.empty:
        return {"games": []}

    model_payload = _load_production_model()
    feature_columns = model_payload["feature_columns"]
    models = model_payload["models"]
    X = schedule[feature_columns].astype(float).fillna(0)

    predictions: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for idx, row in schedule.iterrows():
        game_id = row["game_id"]
        game_predictions: dict[str, Any] = {
            "game_id": game_id,
            "home_team_id": row["home_team_id"],
            "away_team_id": row["away_team_id"],
            "game_date_utc": str(row["game_date_utc"]),
        }
        for target, model in models.items():
            try:
                value = model.predict(X.loc[[idx]])[0]
            except Exception:
                value = None
            game_predictions[target] = float(value) if value is not None else None
        combined_rows.append(game_predictions)

    df = pd.DataFrame(combined_rows)
    if "home_win" in df.columns:
        df["home_win_probability"] = df["home_win"].astype(float)
        df["away_win_probability"] = 1.0 - df["home_win_probability"]
    else:
        df["home_win_probability"] = None
        df["away_win_probability"] = None

    df["predicted_margin"] = df["full_margin"] if "full_margin" in df.columns else None
    df["predicted_total"] = df["full_total"] if "full_total" in df.columns else None

    samples = _simulate_residuals(model_payload, df)
    totals = samples["full_total"] if "full_total" in samples else pd.Series([], dtype=float)
    scores = samples["home_score"] if "home_score" in samples else pd.Series([], dtype=float)

    output = {
        "generated_at_utc": Timestamp.now(tz="UTC").isoformat(),
        "target_date": str(target_date),
        "games": df.to_dict(orient="records"),
        "uncertainty": {
            "full_total": _percentiles(totals),
            "home_score": _percentiles(scores),
        },
    }

    PREDICTION_LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_LATEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_LATEST_JSON, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    pd.DataFrame(output["games"]).to_csv(PREDICTION_LATEST_CSV, index=False)

    history = pd.DataFrame(output["games"]).copy()
    history["prediction_date"] = pd.Timestamp(output["generated_at_utc"])
    if PREDICTION_HISTORY_PATH.exists():
        existing = pd.read_parquet(PREDICTION_HISTORY_PATH)
        history = pd.concat([existing, history], ignore_index=True)
    history = history.drop_duplicates(subset=["game_id", "prediction_date"], keep="last")
    history.to_parquet(PREDICTION_HISTORY_PATH, index=False)

    return output


def main() -> None:
    result = predict_today()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
