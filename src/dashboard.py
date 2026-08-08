from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DOCS_DIR, DOCS_HISTORY_JSON, DOCS_LATEST_JSON, PREDICTION_LATEST_JSON, PREDICTION_HISTORY_PATH


def build_dashboard() -> dict[str, Any]:
    latest = {}
    if Path(PREDICTION_LATEST_JSON).exists():
        with open(PREDICTION_LATEST_JSON, "r", encoding="utf-8") as handle:
            latest = json.load(handle)

    history = []
    if Path(PREDICTION_HISTORY_PATH).exists():
        history_df = pd.read_parquet(PREDICTION_HISTORY_PATH)
        history_df = history_df.sort_values("prediction_date", ascending=False).head(50)
        history_df = history_df.copy()
        if "prediction_date" in history_df.columns:
            history_df["prediction_date"] = history_df["prediction_date"].apply(
                lambda value: value.isoformat() if hasattr(value, "isoformat") else value
            )
        history = history_df.to_dict(orient="records")

    docs_latest_path = Path(DOCS_LATEST_JSON)
    docs_history_path = Path(DOCS_HISTORY_JSON)
    docs_latest_path.parent.mkdir(parents=True, exist_ok=True)
    docs_history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_latest_path, "w", encoding="utf-8") as handle:
        json.dump(latest, handle, indent=2)
    with open(docs_history_path, "w", encoding="utf-8") as handle:
        json.dump({"events": history}, handle, indent=2)
    return {"latest": latest, "history": history}
