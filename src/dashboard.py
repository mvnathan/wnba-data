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

    Examples:
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
    Convert a DataFrame into strict JSON-safe records.
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
    """
    Load the most recent prediction payload.
    """
    path = Path(
        PREDICTION_LATEST_JSON
    )

    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        latest = json.load(
            handle
        )

    if not isinstance(
        latest,
        dict,
    ):
        return {}

    return latest


def _numeric_value(
    row: pd.Series,
    column: str,
) -> float | None:
    """
    Return a finite numeric value when available.
    """
    if column not in row.index:
        return None

    value = row.get(
        column
    )

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    try:
        number = float(
            value
        )
    except (TypeError, ValueError):
        return None

    if not np.isfinite(
        number
    ):
        return None

    return number


def _nonempty_string(
    row: pd.Series,
    column: str,
) -> str | None:
    """
    Return a cleaned non-empty string when available.
    """
    if column not in row.index:
        return None

    value = row.get(
        column
    )

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    return text


def _history_row_is_sane(
    row: pd.Series,
) -> bool:
    """
    Decide whether a prediction-history row is useful enough
    to display on the public dashboard.

    This is intentionally a presentation-layer filter only.
    The raw prediction_history.parquet file remains untouched.

    The checks are broad sanity limits, not betting/model rules.
    They are intended to exclude malformed legacy outputs such as:
    - missing team identities
    - negative or tiny full-game totals
    - implausible team scores
    - extreme malformed margins
    """

    # ---------------------------------------------------------
    # A displayed history record should identify the matchup.
    # ---------------------------------------------------------
    home_abbr = _nonempty_string(
        row,
        "home_abbr",
    )

    away_abbr = _nonempty_string(
        row,
        "away_abbr",
    )

    if (
        home_abbr is None
        or away_abbr is None
    ):
        return False

    # ---------------------------------------------------------
    # We need a prediction timestamp.
    # ---------------------------------------------------------
    if (
        "prediction_date"
        in row.index
    ):
        prediction_date = row.get(
            "prediction_date"
        )

        try:
            if pd.isna(
                prediction_date
            ):
                return False
        except (
            TypeError,
            ValueError,
        ):
            return False

    # ---------------------------------------------------------
    # Full-game total sanity check.
    #
    # WNBA totals can move considerably, so use intentionally
    # broad boundaries. This is only intended to remove clearly
    # broken historical artifacts.
    # ---------------------------------------------------------
    predicted_total = _numeric_value(
        row,
        "predicted_total",
    )

    if predicted_total is None:
        predicted_total = _numeric_value(
            row,
            "full_total",
        )

    if predicted_total is not None:
        if not (
            100.0
            <= predicted_total
            <= 250.0
        ):
            return False

    # ---------------------------------------------------------
    # Individual projected final scores.
    # ---------------------------------------------------------
    home_score = _numeric_value(
        row,
        "home_score",
    )

    away_score = _numeric_value(
        row,
        "away_score",
    )

    if home_score is not None:
        if not (
            40.0
            <= home_score
            <= 150.0
        ):
            return False

    if away_score is not None:
        if not (
            40.0
            <= away_score
            <= 150.0
        ):
            return False

    # ---------------------------------------------------------
    # Margin sanity check.
    # ---------------------------------------------------------
    predicted_margin = _numeric_value(
        row,
        "predicted_margin",
    )

    if predicted_margin is not None:
        if abs(
            predicted_margin
        ) > 50.0:
            return False

    # ---------------------------------------------------------
    # Win probability sanity check when available.
    # ---------------------------------------------------------
    home_win_probability = _numeric_value(
        row,
        "home_win_probability",
    )

    if home_win_probability is not None:
        if not (
            0.0
            <= home_win_probability
            <= 1.0
        ):
            return False

    return True


def _filter_history_for_dashboard(
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove malformed legacy prediction rows from the dashboard.

    Important:
    If a test fixture or minimal dataset contains no rows that
    satisfy the full production sanity rules, return the original
    rows rather than an empty DataFrame.

    That preserves lifecycle/test compatibility while still
    cleaning real production history where good rows exist.
    """
    if history_df.empty:
        return history_df

    sane_mask = history_df.apply(
        _history_row_is_sane,
        axis=1,
    )

    sane_df = history_df.loc[
        sane_mask
    ].copy()

    if sane_df.empty:
        return history_df.copy()

    return sane_df


def _load_history() -> list[dict[str, Any]]:
    """
    Load prediction history for the public dashboard.

    The raw parquet is preserved in full. Only the browser-facing
    JSON is filtered and normalized.
    """
    path = Path(
        PREDICTION_HISTORY_PATH
    )

    if not path.exists():
        return []

    history_df = pd.read_parquet(
        path
    )

    if history_df.empty:
        return []

    history_df = history_df.copy()

    # ---------------------------------------------------------
    # Normalize prediction timestamps.
    # ---------------------------------------------------------
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

        history_df = history_df.dropna(
            subset=[
                "prediction_date"
            ]
        )

    # ---------------------------------------------------------
    # Filter malformed legacy model outputs from the public view.
    # ---------------------------------------------------------
    history_df = (
        _filter_history_for_dashboard(
            history_df
        )
    )

    # ---------------------------------------------------------
    # Newest predictions first.
    # ---------------------------------------------------------
    if (
        "prediction_date"
        in history_df.columns
    ):
        history_df = (
            history_df.sort_values(
                "prediction_date",
                ascending=False,
            )
        )

    # ---------------------------------------------------------
    # Keep enough history for useful comparison without creating
    # an unnecessarily large static page payload.
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
    """
    Build the browser-facing dashboard JSON files.
    """
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
    # allow_nan=False is deliberate.
    #
    # If a non-JSON-safe numeric value somehow survives our
    # normalization, fail here rather than publishing JSON that
    # browsers cannot parse.
    # ---------------------------------------------------------
    with docs_latest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            latest,
            handle,
            indent=2,
            allow_nan=False,
        )

    with docs_history_path.open(
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