from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DOCS_DIR,
    DOCS_HISTORY_JSON,
    DOCS_LATEST_JSON,
    PREDICTION_HISTORY_PATH,
    PREDICTION_LATEST_JSON,
)


def _json_safe_value(value: Any) -> Any:
    """
    Convert pandas / numpy values into strict JSON-safe values.

    In particular:
    - NaN -> None
    - NaT -> None
    - numpy scalar -> Python scalar
    - Timestamp -> ISO string
    """
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.generic):
        value = value.item()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def _records_json_safe(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert a DataFrame to strict JSON-safe records.
    """
    records: list[dict[str, Any]] = []

    for raw_row in df.to_dict(
        orient="records"
    ):
        row: dict[str, Any] = {}

        for key, value in raw_row.items():
            row[key] = _json_safe_value(
                value
            )

        records.append(row)

    return records


def _load_latest() -> dict[str, Any]:
    if not Path(
        PREDICTION_LATEST_JSON
    ).exists():
        return {}

    with open(
        PREDICTION_LATEST_JSON,
        "r",
        encoding="utf-8",
    ) as handle:
        latest = json.load(
            handle
        )

    return (
        latest
        if isinstance(latest, dict)
        else {}
    )


def _load_history() -> list[dict[str, Any]]:
    if not Path(
        PREDICTION_HISTORY_PATH
    ).exists():
        return []

    history_df = pd.read_parquet(
        PREDICTION_HISTORY_PATH
    )

    if history_df.empty:
        return []

    # Keep legitimate lifecycle/history rows even if some older
    # records do not yet contain every modern prediction field.
    #
    # We only require a game id and prediction timestamp when
    # those columns are present. Missing optional prediction
    # values are converted to null later for valid JSON output.
    required_columns = [
        column
        for column in (
            "game_id",
            "prediction_date",
        )
        if column in history_df.columns
    ]

    if required_columns:
        history_df = (
            history_df.dropna(
                subset=required_columns
            )
        )

    if (
        "prediction_date"
        in history_df.columns
    ):
        history_df[
            "prediction_date"
        ] = pd.to_datetime(
            history_df[
                "prediction_date"
            ],
            utc=True,
            errors="coerce",
        )

        history_df = (
            history_df.dropna(
                subset=[
                    "prediction_date"
                ]
            )
            .sort_values(
                "prediction_date",
                ascending=False,
            )
        )

    # ---------------------------------------------------------
    # Keep a reasonable amount of history for the static page.
    # ---------------------------------------------------------
    history_df = (
        history_df.head(
            100
        )
        .copy()
    )

    return _records_json_safe(
        history_df
    )


def build_dashboard() -> dict[str, Any]:
    latest = _load_latest()
    history = _load_history()

    docs_latest_path = Path(
        DOCS_LATEST_JSON
    )

    docs_history_path = Path(
        DOCS_HISTORY_JSON
    )

    docs_latest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    docs_history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # allow_nan=False is intentional.
    #
    # If NaN somehow survives our normalization, fail here
    # instead of publishing JSON that browsers cannot parse.
    # ---------------------------------------------------------
    with open(
        docs_latest_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            latest,
            handle,
            indent=2,
            allow_nan=False,
        )

    with open(
        docs_history_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "events": history,
            },
            handle,
            indent=2,
            allow_nan=False,
        )

    return {
        "latest": latest,
        "history": history,
    }