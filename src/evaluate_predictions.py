from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DOCS_PERFORMANCE_JSON,
    GAMES_PATH,
    PERFORMANCE_HISTORY_PATH,
    PREDICTION_HISTORY_PATH,
    QUARTER_SCORES_PATH,
)


PICK_THRESHOLD = 0.5


def _safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def _safe_bool(
    value: Any,
) -> bool | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    return bool(value)


def _safe_string(
    value: Any,
) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    text = str(value).strip()

    return text or None


def _json_safe(
    value: Any,
) -> Any:
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


def _json_records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for raw in df.to_dict(
        orient="records"
    ):
        record = {
            key: _json_safe(value)
            for key, value in raw.items()
        }

        records.append(record)

    return records


def _prediction_row_is_usable(
    row: pd.Series,
) -> bool:
    game_id = _safe_string(
        row.get("game_id")
    )

    if game_id is None:
        return False

    home_abbr = _safe_string(
        row.get("home_abbr")
    )

    away_abbr = _safe_string(
        row.get("away_abbr")
    )

    if (
        home_abbr is None
        or away_abbr is None
    ):
        return False

    predicted_total = _safe_float(
        row.get("predicted_total")
    )

    predicted_margin = _safe_float(
        row.get("predicted_margin")
    )

    predicted_home = _safe_float(
        row.get("home_score")
    )

    predicted_away = _safe_float(
        row.get("away_score")
    )

    if (
        predicted_total is None
        or predicted_margin is None
        or predicted_home is None
        or predicted_away is None
    ):
        return False

    if not (
        100.0
        <= predicted_total
        <= 250.0
    ):
        return False

    if not (
        40.0
        <= predicted_home
        <= 150.0
    ):
        return False

    if not (
        40.0
        <= predicted_away
        <= 150.0
    ):
        return False

    if abs(predicted_margin) > 50.0:
        return False

    return True


def _load_pregame_predictions() -> pd.DataFrame:
    path = Path(
        PREDICTION_HISTORY_PATH
    )

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(
        path
    )

    if df.empty:
        return df

    df = df.copy()

    df["game_id"] = (
        df["game_id"]
        .astype(str)
    )

    df["prediction_date"] = pd.to_datetime(
        df["prediction_date"],
        utc=True,
        errors="coerce",
    )

    df["game_date_utc"] = pd.to_datetime(
        df["game_date_utc"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "game_id",
            "prediction_date",
            "game_date_utc",
        ]
    )

    # Only evaluate predictions made before tip-off.
    df = df[
        df["prediction_date"]
        <= df["game_date_utc"]
    ].copy()

    if df.empty:
        return df

    usable_mask = df.apply(
        _prediction_row_is_usable,
        axis=1,
    )

    df = df.loc[
        usable_mask
    ].copy()

    if df.empty:
        return df

    # One prediction per game:
    # last usable pregame forecast before tip-off.
    df = (
        df.sort_values(
            [
                "game_id",
                "prediction_date",
            ]
        )
        .drop_duplicates(
            subset=[
                "game_id",
            ],
            keep="last",
        )
    )

    return df


def _load_completed_games() -> pd.DataFrame:
    path = Path(
        GAMES_PATH
    )

    if not path.exists():
        return pd.DataFrame()

    games = pd.read_parquet(
        path
    )

    if games.empty:
        return games

    games = games.copy()

    games["game_id"] = (
        games["game_id"]
        .astype(str)
    )

    if "completed" in games.columns:
        games = games[
            games["completed"]
            .fillna(False)
            .astype(bool)
        ].copy()

    games["game_date_utc"] = pd.to_datetime(
        games["game_date_utc"],
        utc=True,
        errors="coerce",
    )

    games["actual_home_score"] = pd.to_numeric(
        games["home_score"],
        errors="coerce",
    )

    games["actual_away_score"] = pd.to_numeric(
        games["away_score"],
        errors="coerce",
    )

    games = games.dropna(
        subset=[
            "game_id",
            "actual_home_score",
            "actual_away_score",
        ]
    )

    keep_columns = [
        "game_id",
        "game_date_utc",
        "season",
        "home_team_id",
        "home_team",
        "home_abbr",
        "away_team_id",
        "away_team",
        "away_abbr",
        "actual_home_score",
        "actual_away_score",
        "venue",
    ]

    keep_columns = [
        column
        for column in keep_columns
        if column in games.columns
    ]

    return games[
        keep_columns
    ].copy()


def _load_actual_quarters() -> pd.DataFrame:
    path = Path(
        QUARTER_SCORES_PATH
    )

    if not path.exists():
        return pd.DataFrame()

    quarters = pd.read_parquet(
        path
    )

    if quarters.empty:
        return quarters

    quarters = quarters.copy()

    quarters["game_id"] = (
        quarters["game_id"]
        .astype(str)
    )

    quarters["team_id"] = (
        quarters["team_id"]
        .astype(str)
    )

    quarter_columns = [
        "q1",
        "q2",
        "q3",
        "q4",
        "ot1",
        "ot2",
    ]

    for column in quarter_columns:
        if column in quarters.columns:
            quarters[column] = pd.to_numeric(
                quarters[column],
                errors="coerce",
            )

    return quarters


def _quarter_row(
    quarters: pd.DataFrame,
    game_id: str,
    team_id: str,
) -> pd.Series | None:
    matches = quarters[
        (
            quarters["game_id"]
            == str(game_id)
        )
        & (
            quarters["team_id"]
            == str(team_id)
        )
    ]

    if matches.empty:
        return None

    return matches.iloc[-1]


def _winner_from_margin(
    margin: float,
    home_abbr: str | None,
    away_abbr: str | None,
) -> str | None:
    if abs(margin) < PICK_THRESHOLD:
        return None

    if margin > 0:
        return home_abbr

    return away_abbr


def _market_side_result(
    actual_margin: float,
    market_home_spread: float | None,
) -> str | None:
    if market_home_spread is None:
        return None

    ats_margin = (
        actual_margin
        + market_home_spread
    )

    if abs(ats_margin) < 1e-9:
        return "push"

    if ats_margin > 0:
        return "home"

    return "away"


def _model_market_side(
    predicted_margin: float,
    market_home_spread: float | None,
) -> str | None:
    if market_home_spread is None:
        return None

    market_implied_home_margin = (
        -market_home_spread
    )

    edge = (
        predicted_margin
        - market_implied_home_margin
    )

    if abs(edge) < PICK_THRESHOLD:
        return None

    if edge > 0:
        return "home"

    return "away"


def _model_total_side(
    predicted_total: float,
    market_total: float | None,
) -> str | None:
    if market_total is None:
        return None

    edge = (
        predicted_total
        - market_total
    )

    if abs(edge) < PICK_THRESHOLD:
        return None

    if edge > 0:
        return "over"

    return "under"


def _actual_total_side(
    actual_total: float,
    market_total: float | None,
) -> str | None:
    if market_total is None:
        return None

    difference = (
        actual_total
        - market_total
    )

    if abs(difference) < 1e-9:
        return "push"

    if difference > 0:
        return "over"

    return "under"


def _evaluate_row(
    prediction: pd.Series,
    actual: pd.Series,
    quarters: pd.DataFrame,
) -> dict[str, Any]:
    game_id = str(
        prediction["game_id"]
    )

    home_team_id = str(
        actual.get(
            "home_team_id"
        )
    )

    away_team_id = str(
        actual.get(
            "away_team_id"
        )
    )

    home_quarters = _quarter_row(
        quarters,
        game_id,
        home_team_id,
    )

    away_quarters = _quarter_row(
        quarters,
        game_id,
        away_team_id,
    )

    predicted_home_score = float(
        prediction["home_score"]
    )

    predicted_away_score = float(
        prediction["away_score"]
    )

    predicted_margin = float(
        prediction["predicted_margin"]
    )

    predicted_total = float(
        prediction["predicted_total"]
    )

    actual_home_score = float(
        actual["actual_home_score"]
    )

    actual_away_score = float(
        actual["actual_away_score"]
    )

    actual_margin = (
        actual_home_score
        - actual_away_score
    )

    actual_total = (
        actual_home_score
        + actual_away_score
    )

    home_abbr = _safe_string(
        actual.get("home_abbr")
    )

    away_abbr = _safe_string(
        actual.get("away_abbr")
    )

    predicted_winner = _winner_from_margin(
        predicted_margin,
        home_abbr,
        away_abbr,
    )

    actual_winner = _winner_from_margin(
        actual_margin,
        home_abbr,
        away_abbr,
    )

    winner_correct: bool | None

    if (
        predicted_winner is None
        or actual_winner is None
    ):
        winner_correct = None
    else:
        winner_correct = (
            predicted_winner
            == actual_winner
        )

    home_win_probability = _safe_float(
        prediction.get(
            "home_win_probability"
        )
    )

    actual_home_win = (
        1.0
        if actual_margin > 0
        else 0.0
    )

    brier_score = (
        (
            home_win_probability
            - actual_home_win
        )
        ** 2
        if home_win_probability
        is not None
        else None
    )

    row: dict[str, Any] = {
        "game_id": game_id,
        "game_date_utc": actual.get(
            "game_date_utc"
        ),
        "season": actual.get(
            "season"
        ),
        "prediction_date": prediction.get(
            "prediction_date"
        ),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team": actual.get(
            "home_team"
        ),
        "away_team": actual.get(
            "away_team"
        ),
        "home_abbr": home_abbr,
        "away_abbr": away_abbr,
        "venue": actual.get(
            "venue"
        ),
        "predicted_home_score": predicted_home_score,
        "predicted_away_score": predicted_away_score,
        "actual_home_score": actual_home_score,
        "actual_away_score": actual_away_score,
        "home_score_error": (
            predicted_home_score
            - actual_home_score
        ),
        "away_score_error": (
            predicted_away_score
            - actual_away_score
        ),
        "absolute_home_score_error": abs(
            predicted_home_score
            - actual_home_score
        ),
        "absolute_away_score_error": abs(
            predicted_away_score
            - actual_away_score
        ),
        "predicted_margin": predicted_margin,
        "actual_margin": actual_margin,
        "margin_error": (
            predicted_margin
            - actual_margin
        ),
        "absolute_margin_error": abs(
            predicted_margin
            - actual_margin
        ),
        "predicted_total": predicted_total,
        "actual_total": actual_total,
        "total_error": (
            predicted_total
            - actual_total
        ),
        "absolute_total_error": abs(
            predicted_total
            - actual_total
        ),
        "predicted_winner_abbr": predicted_winner,
        "actual_winner_abbr": actual_winner,
        "winner_correct": winner_correct,
        "home_win_probability": home_win_probability,
        "actual_home_win": bool(
            actual_margin > 0
        ),
        "brier_score": brier_score,
        "market_bookmaker": prediction.get(
            "market_bookmaker"
        ),
        "market_home_spread": _safe_float(
            prediction.get(
                "market_home_spread"
            )
        ),
        "market_away_spread": _safe_float(
            prediction.get(
                "market_away_spread"
            )
        ),
        "market_total": _safe_float(
            prediction.get(
                "market_total"
            )
        ),
        "market_home_moneyline": _safe_float(
            prediction.get(
                "market_home_moneyline"
            )
        ),
        "market_away_moneyline": _safe_float(
            prediction.get(
                "market_away_moneyline"
            )
        ),
        "market_updated_at": prediction.get(
            "market_updated_at"
        ),
    }

    # ---------------------------------------------------------
    # Regulation quarters
    # ---------------------------------------------------------
    for quarter_number in range(
        1,
        5,
    ):
        key = f"q{quarter_number}"

        predicted_home = _safe_float(
            prediction.get(
                f"home_{key}"
            )
        )

        predicted_away = _safe_float(
            prediction.get(
                f"away_{key}"
            )
        )

        actual_home = (
            _safe_float(
                home_quarters.get(key)
            )
            if home_quarters
            is not None
            else None
        )

        actual_away = (
            _safe_float(
                away_quarters.get(key)
            )
            if away_quarters
            is not None
            else None
        )

        row[
            f"predicted_home_{key}"
        ] = predicted_home

        row[
            f"predicted_away_{key}"
        ] = predicted_away

        row[
            f"actual_home_{key}"
        ] = actual_home

        row[
            f"actual_away_{key}"
        ] = actual_away

        if (
            predicted_home is not None
            and predicted_away is not None
            and actual_home is not None
            and actual_away is not None
        ):
            predicted_q_total = (
                predicted_home
                + predicted_away
            )

            actual_q_total = (
                actual_home
                + actual_away
            )

            predicted_q_margin = (
                predicted_home
                - predicted_away
            )

            actual_q_margin = (
                actual_home
                - actual_away
            )

            row[
                f"predicted_{key}_total"
            ] = predicted_q_total

            row[
                f"actual_{key}_total"
            ] = actual_q_total

            row[
                f"{key}_total_error"
            ] = (
                predicted_q_total
                - actual_q_total
            )

            row[
                f"absolute_{key}_total_error"
            ] = abs(
                predicted_q_total
                - actual_q_total
            )

            row[
                f"predicted_{key}_margin"
            ] = predicted_q_margin

            row[
                f"actual_{key}_margin"
            ] = actual_q_margin

            row[
                f"{key}_margin_error"
            ] = (
                predicted_q_margin
                - actual_q_margin
            )

            row[
                f"absolute_{key}_margin_error"
            ] = abs(
                predicted_q_margin
                - actual_q_margin
            )

            row[
                f"absolute_home_{key}_error"
            ] = abs(
                predicted_home
                - actual_home
            )

            row[
                f"absolute_away_{key}_error"
            ] = abs(
                predicted_away
                - actual_away
            )

        else:
            row[
                f"predicted_{key}_total"
            ] = None

            row[
                f"actual_{key}_total"
            ] = None

            row[
                f"{key}_total_error"
            ] = None

            row[
                f"absolute_{key}_total_error"
            ] = None

            row[
                f"predicted_{key}_margin"
            ] = None

            row[
                f"actual_{key}_margin"
            ] = None

            row[
                f"{key}_margin_error"
            ] = None

            row[
                f"absolute_{key}_margin_error"
            ] = None

            row[
                f"absolute_home_{key}_error"
            ] = None

            row[
                f"absolute_away_{key}_error"
            ] = None

    # ---------------------------------------------------------
    # Halves from regulation quarters
    # ---------------------------------------------------------
    if all(
        row.get(
            f"actual_home_q{q}"
        )
        is not None
        and row.get(
            f"actual_away_q{q}"
        )
        is not None
        for q in (
            1,
            2,
            3,
            4,
        )
    ):
        actual_home_first_half = (
            row["actual_home_q1"]
            + row["actual_home_q2"]
        )

        actual_away_first_half = (
            row["actual_away_q1"]
            + row["actual_away_q2"]
        )

        actual_home_second_half = (
            row["actual_home_q3"]
            + row["actual_home_q4"]
        )

        actual_away_second_half = (
            row["actual_away_q3"]
            + row["actual_away_q4"]
        )

        row.update(
            {
                "actual_home_first_half": (
                    actual_home_first_half
                ),
                "actual_away_first_half": (
                    actual_away_first_half
                ),
                "actual_first_half_total": (
                    actual_home_first_half
                    + actual_away_first_half
                ),
                "actual_first_half_margin": (
                    actual_home_first_half
                    - actual_away_first_half
                ),
                "actual_home_second_half": (
                    actual_home_second_half
                ),
                "actual_away_second_half": (
                    actual_away_second_half
                ),
                "actual_second_half_total": (
                    actual_home_second_half
                    + actual_away_second_half
                ),
                "actual_second_half_margin": (
                    actual_home_second_half
                    - actual_away_second_half
                ),
            }
        )

        predicted_first_half_total = _safe_float(
            prediction.get(
                "first_half_total"
            )
        )

        predicted_first_half_margin = _safe_float(
            prediction.get(
                "first_half_margin"
            )
        )

        predicted_second_half_total = _safe_float(
            prediction.get(
                "second_half_total"
            )
        )

        predicted_second_half_margin = _safe_float(
            prediction.get(
                "second_half_margin"
            )
        )

        row[
            "predicted_first_half_total"
        ] = predicted_first_half_total

        row[
            "predicted_first_half_margin"
        ] = predicted_first_half_margin

        row[
            "predicted_second_half_total"
        ] = predicted_second_half_total

        row[
            "predicted_second_half_margin"
        ] = predicted_second_half_margin

        row[
            "absolute_first_half_total_error"
        ] = (
            abs(
                predicted_first_half_total
                - row[
                    "actual_first_half_total"
                ]
            )
            if predicted_first_half_total
            is not None
            else None
        )

        row[
            "absolute_first_half_margin_error"
        ] = (
            abs(
                predicted_first_half_margin
                - row[
                    "actual_first_half_margin"
                ]
            )
            if predicted_first_half_margin
            is not None
            else None
        )

        row[
            "absolute_second_half_total_error"
        ] = (
            abs(
                predicted_second_half_total
                - row[
                    "actual_second_half_total"
                ]
            )
            if predicted_second_half_total
            is not None
            else None
        )

        row[
            "absolute_second_half_margin_error"
        ] = (
            abs(
                predicted_second_half_margin
                - row[
                    "actual_second_half_margin"
                ]
            )
            if predicted_second_half_margin
            is not None
            else None
        )

    # ---------------------------------------------------------
    # Market evaluation
    # ---------------------------------------------------------
    market_home_spread = row[
        "market_home_spread"
    ]

    market_total = row[
        "market_total"
    ]

    model_market_side = _model_market_side(
        predicted_margin,
        market_home_spread,
    )

    actual_market_side = _market_side_result(
        actual_margin,
        market_home_spread,
    )

    row[
        "model_market_side"
    ] = model_market_side

    row[
        "actual_market_side"
    ] = actual_market_side

    if (
        model_market_side is not None
        and actual_market_side is not None
        and actual_market_side != "push"
    ):
        row[
            "ats_model_correct"
        ] = (
            model_market_side
            == actual_market_side
        )
    else:
        row[
            "ats_model_correct"
        ] = None

    model_total_side = _model_total_side(
        predicted_total,
        market_total,
    )

    actual_total_side = _actual_total_side(
        actual_total,
        market_total,
    )

    row[
        "model_total_side"
    ] = model_total_side

    row[
        "actual_total_side"
    ] = actual_total_side

    if (
        model_total_side is not None
        and actual_total_side is not None
        and actual_total_side != "push"
    ):
        row[
            "total_model_correct"
        ] = (
            model_total_side
            == actual_total_side
        )
    else:
        row[
            "total_model_correct"
        ] = None

    return row


def _mean(
    series: pd.Series,
) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(
        values.mean()
    )


def _rate(
    series: pd.Series,
) -> float | None:
    values = series.dropna()

    if values.empty:
        return None

    return float(
        values.astype(bool).mean()
    )


def _build_summary(
    performance: pd.DataFrame,
) -> dict[str, Any]:
    if performance.empty:
        return {
            "games_evaluated": 0,
        }

    summary: dict[str, Any] = {
        "games_evaluated": int(
            len(performance)
        ),
        "winner_accuracy": (
            _rate(
                performance[
                    "winner_correct"
                ]
            )
            if "winner_correct"
            in performance.columns
            else None
        ),
        "margin_mae": (
            _mean(
                performance[
                    "absolute_margin_error"
                ]
            )
        ),
        "total_mae": (
            _mean(
                performance[
                    "absolute_total_error"
                ]
            )
        ),
        "home_score_mae": (
            _mean(
                performance[
                    "absolute_home_score_error"
                ]
            )
        ),
        "away_score_mae": (
            _mean(
                performance[
                    "absolute_away_score_error"
                ]
            )
        ),
        "brier_score": (
            _mean(
                performance[
                    "brier_score"
                ]
            )
        ),
        "ats_direction_accuracy": (
            _rate(
                performance[
                    "ats_model_correct"
                ]
            )
        ),
        "total_direction_accuracy": (
            _rate(
                performance[
                    "total_model_correct"
                ]
            )
        ),
    }

    for quarter_number in range(
        1,
        5,
    ):
        key = f"q{quarter_number}"

        column = (
            f"absolute_{key}_total_error"
        )

        summary[
            f"{key}_total_mae"
        ] = (
            _mean(
                performance[column]
            )
            if column
            in performance.columns
            else None
        )

        margin_column = (
            f"absolute_{key}_margin_error"
        )

        summary[
            f"{key}_margin_mae"
        ] = (
            _mean(
                performance[
                    margin_column
                ]
            )
            if margin_column
            in performance.columns
            else None
        )

    for half in (
        "first_half",
        "second_half",
    ):
        total_column = (
            f"absolute_{half}_total_error"
        )

        margin_column = (
            f"absolute_{half}_margin_error"
        )

        summary[
            f"{half}_total_mae"
        ] = (
            _mean(
                performance[
                    total_column
                ]
            )
            if total_column
            in performance.columns
            else None
        )

        summary[
            f"{half}_margin_mae"
        ] = (
            _mean(
                performance[
                    margin_column
                ]
            )
            if margin_column
            in performance.columns
            else None
        )

    if (
        "game_date_utc"
        in performance.columns
    ):
        latest = pd.to_datetime(
            performance[
                "game_date_utc"
            ],
            utc=True,
            errors="coerce",
        ).max()

        summary[
            "latest_game_date"
        ] = (
            latest.isoformat()
            if pd.notna(latest)
            else None
        )

    return summary


def evaluate_predictions() -> dict[str, Any]:
    predictions = (
        _load_pregame_predictions()
    )

    completed_games = (
        _load_completed_games()
    )

    quarters = (
        _load_actual_quarters()
    )

    if (
        predictions.empty
        or completed_games.empty
    ):
        performance = pd.DataFrame()
    else:
        merged = predictions.merge(
            completed_games,
            on="game_id",
            how="inner",
            suffixes=(
                "_prediction",
                "_actual",
            ),
        )

        records: list[
            dict[str, Any]
        ] = []

        for _, row in merged.iterrows():
            prediction_data = {}

            actual_data = {}

            for column in predictions.columns:
                if column in row.index:
                    prediction_data[
                        column
                    ] = row[column]

                prediction_name = (
                    f"{column}_prediction"
                )

                if (
                    prediction_name
                    in row.index
                ):
                    prediction_data[
                        column
                    ] = row[
                        prediction_name
                    ]

            for column in completed_games.columns:
                if column in row.index:
                    actual_data[
                        column
                    ] = row[column]

                actual_name = (
                    f"{column}_actual"
                )

                if actual_name in row.index:
                    actual_data[
                        column
                    ] = row[
                        actual_name
                    ]

            records.append(
                _evaluate_row(
                    pd.Series(
                        prediction_data
                    ),
                    pd.Series(
                        actual_data
                    ),
                    quarters,
                )
            )

        performance = pd.DataFrame(
            records
        )

    if not performance.empty:
        performance[
            "game_date_utc"
        ] = pd.to_datetime(
            performance[
                "game_date_utc"
            ],
            utc=True,
            errors="coerce",
        )

        performance[
            "prediction_date"
        ] = pd.to_datetime(
            performance[
                "prediction_date"
            ],
            utc=True,
            errors="coerce",
        )

        performance = (
            performance.sort_values(
                "game_date_utc",
                ascending=False,
            )
        )

    PERFORMANCE_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance.to_parquet(
        PERFORMANCE_HISTORY_PATH,
        index=False,
    )

    summary = _build_summary(
        performance
    )

    payload = {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "summary": {
            key: _json_safe(value)
            for key, value
            in summary.items()
        },
        "games": (
            _json_records(
                performance
            )
        ),
    }

    DOCS_PERFORMANCE_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DOCS_PERFORMANCE_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=False,
        )

    return payload