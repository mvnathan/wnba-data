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
    PREDICTION_HISTORY_PATH,
    PREDICTION_LATEST_CSV,
    PREDICTION_LATEST_JSON,
    PRODUCTION_MODEL_PATH,
)
from .features import build_model_features
from .market_odds import (
    attach_market_odds,
    fetch_draftkings_wnba_odds,
)


def _today_chicago() -> date:
    """
    Return today's calendar date in the configured Chicago timezone.
    """
    return pd.Timestamp.now(
        tz=CHICAGO
    ).date()


def _load_production_model() -> dict[str, Any]:
    """
    Load the trained production model artifact.
    """
    if not Path(
        PRODUCTION_MODEL_PATH
    ).exists():
        raise FileNotFoundError(
            "Production model not found: "
            f"{PRODUCTION_MODEL_PATH}"
        )

    return joblib.load(
        PRODUCTION_MODEL_PATH
    )


def _build_pregame_data(
    target_date: date,
) -> pd.DataFrame:
    """
    Build feature rows eligible for pregame prediction.

    Rules:
    - Match the requested calendar date in Chicago time.
    - Exclude completed games.
    - Exclude games whose status indicates play has begun.
    - For today's predictions, exclude games whose scheduled
      start time has already passed.

    The final time-based safeguard protects against stale status data.
    """
    features = build_model_features(
        "data"
    )

    if features.empty:
        return pd.DataFrame()

    features = features.copy()

    features[
        "game_date_utc"
    ] = pd.to_datetime(
        features[
            "game_date_utc"
        ],
        utc=True,
        errors="coerce",
    )

    local_dates = (
        features[
            "game_date_utc"
        ]
        .dt.tz_convert(
            CHICAGO
        )
        .dt.date
    )

    schedule = features[
        local_dates == target_date
    ].copy()

    if schedule.empty:
        return schedule

    # ---------------------------------------------------------
    # Remove completed games
    # ---------------------------------------------------------
    if (
        "completed"
        in schedule.columns
    ):
        completed = (
            schedule[
                "completed"
            ]
            .fillna(False)
            .astype(bool)
        )

        schedule = schedule[
            ~completed
        ].copy()

    # ---------------------------------------------------------
    # Remove games whose status indicates they have started
    # ---------------------------------------------------------
    if (
        "status"
        in schedule.columns
    ):
        status = (
            schedule[
                "status"
            ]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        excluded_statuses = {
            "STATUS_IN_PROGRESS",
            "STATUS_HALFTIME",
            "STATUS_FINAL",
            "STATUS_FINAL_OT",
            "STATUS_POSTPONED",
            "STATUS_CANCELED",
            "STATUS_CANCELLED",
        }

        schedule = schedule[
            ~status.isin(
                excluded_statuses
            )
        ].copy()

    # ---------------------------------------------------------
    # If predicting today, also remove games whose scheduled
    # start has already passed.
    # ---------------------------------------------------------
    if (
        target_date
        == _today_chicago()
    ):
        now_utc = pd.Timestamp.now(
            tz="UTC"
        )

        schedule = schedule[
            schedule[
                "game_date_utc"
            ]
            > now_utc
        ].copy()

    return (
        schedule.sort_values(
            [
                "game_date_utc",
                "game_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _simulate_residuals(
    model_payload: dict[str, Any],
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply randomly sampled historical residuals to point predictions.

    This preserves the existing uncertainty-output mechanism.
    """
    samples: list[
        dict[str, float]
    ] = []

    residuals = (
        model_payload.get(
            "holdout_residuals",
            {},
        )
    )

    for _, row in predictions.iterrows():
        sample: dict[
            str,
            float
        ] = {}

        for (
            target,
            target_residuals,
        ) in residuals.items():

            if (
                target
                not in row.index
            ):
                continue

            base_value = row[
                target
            ]

            if pd.isna(
                base_value
            ):
                continue

            residual_array = np.asarray(
                target_residuals,
                dtype=float,
            )

            residual_array = (
                residual_array[
                    np.isfinite(
                        residual_array
                    )
                ]
            )

            if (
                residual_array.size
                == 0
            ):
                sample[
                    target
                ] = float(
                    base_value
                )

                continue

            sampled_residual = (
                np.random.choice(
                    residual_array,
                    size=1,
                )[0]
            )

            sample[
                target
            ] = float(
                base_value
                + sampled_residual
            )

        samples.append(
            sample
        )

    return pd.DataFrame(
        samples
    )


def _percentiles(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Return p10, median, and p90 for a numeric series.
    """
    if series.empty:
        return {
            "p10": None,
            "median": None,
            "p90": None,
        }

    numeric = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
    )

    if numeric.empty:
        return {
            "p10": None,
            "median": None,
            "p90": None,
        }

    quantiles = numeric.quantile(
        [
            0.1,
            0.5,
            0.9,
        ]
    )

    return {
        "p10": float(
            quantiles.loc[
                0.1
            ]
        ),
        "median": float(
            quantiles.loc[
                0.5
            ]
        ),
        "p90": float(
            quantiles.loc[
                0.9
            ]
        ),
    }


def _enforce_prediction_coherence(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Make score-derived predictions mathematically consistent.

    Canonical structure:

        quarter team scores
            -> quarter margins/totals
            -> first/second-half scores
            -> half margins/totals
            -> full-game scores
            -> full-game margin/total

    Quarter team-score predictions therefore become the canonical
    scoring path for the dashboard.
    """
    df = df.copy()

    quarters = (
        "q1",
        "q2",
        "q3",
        "q4",
    )

    # ---------------------------------------------------------
    # Quarter coherence
    # ---------------------------------------------------------
    for quarter in quarters:
        home_col = (
            f"home_{quarter}"
        )

        away_col = (
            f"away_{quarter}"
        )

        margin_col = (
            f"{quarter}_margin"
        )

        total_col = (
            f"{quarter}_total"
        )

        if {
            home_col,
            away_col,
        }.issubset(
            df.columns
        ):
            df[
                margin_col
            ] = (
                df[
                    home_col
                ]
                - df[
                    away_col
                ]
            )

            df[
                total_col
            ] = (
                df[
                    home_col
                ]
                + df[
                    away_col
                ]
            )

    # ---------------------------------------------------------
    # First half = Q1 + Q2
    # ---------------------------------------------------------
    first_half_cols = {
        "home_q1",
        "away_q1",
        "home_q2",
        "away_q2",
    }

    if first_half_cols.issubset(
        df.columns
    ):
        df[
            "home_first_half"
        ] = (
            df[
                "home_q1"
            ]
            + df[
                "home_q2"
            ]
        )

        df[
            "away_first_half"
        ] = (
            df[
                "away_q1"
            ]
            + df[
                "away_q2"
            ]
        )

        df[
            "first_half_margin"
        ] = (
            df[
                "home_first_half"
            ]
            - df[
                "away_first_half"
            ]
        )

        df[
            "first_half_total"
        ] = (
            df[
                "home_first_half"
            ]
            + df[
                "away_first_half"
            ]
        )

    # ---------------------------------------------------------
    # Second half = Q3 + Q4
    # ---------------------------------------------------------
    second_half_cols = {
        "home_q3",
        "away_q3",
        "home_q4",
        "away_q4",
    }

    if (
        second_half_cols.issubset(
            df.columns
        )
    ):
        df[
            "home_second_half"
        ] = (
            df[
                "home_q3"
            ]
            + df[
                "home_q4"
            ]
        )

        df[
            "away_second_half"
        ] = (
            df[
                "away_q3"
            ]
            + df[
                "away_q4"
            ]
        )

        df[
            "second_half_margin"
        ] = (
            df[
                "home_second_half"
            ]
            - df[
                "away_second_half"
            ]
        )

        df[
            "second_half_total"
        ] = (
            df[
                "home_second_half"
            ]
            + df[
                "away_second_half"
            ]
        )

    # ---------------------------------------------------------
    # Full game = Q1 + Q2 + Q3 + Q4
    # ---------------------------------------------------------
    all_quarter_cols = {
        "home_q1",
        "home_q2",
        "home_q3",
        "home_q4",
        "away_q1",
        "away_q2",
        "away_q3",
        "away_q4",
    }

    if (
        all_quarter_cols.issubset(
            df.columns
        )
    ):
        df[
            "home_score"
        ] = (
            df[
                "home_q1"
            ]
            + df[
                "home_q2"
            ]
            + df[
                "home_q3"
            ]
            + df[
                "home_q4"
            ]
        )

        df[
            "away_score"
        ] = (
            df[
                "away_q1"
            ]
            + df[
                "away_q2"
            ]
            + df[
                "away_q3"
            ]
            + df[
                "away_q4"
            ]
        )

    # ---------------------------------------------------------
    # Full-game margin and total
    # ---------------------------------------------------------
    if {
        "home_score",
        "away_score",
    }.issubset(
        df.columns
    ):
        df[
            "full_margin"
        ] = (
            df[
                "home_score"
            ]
            - df[
                "away_score"
            ]
        )

        df[
            "full_total"
        ] = (
            df[
                "home_score"
            ]
            + df[
                "away_score"
            ]
        )

    return df


def _attach_current_market_odds(
    games: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Fetch the latest DraftKings WNBA odds and attach them to
    prediction rows.

    Market-data failure must never cause prediction generation
    itself to fail.
    """
    try:
        odds_data = (
            fetch_draftkings_wnba_odds()
        )

        enriched_games = (
            attach_market_odds(
                games,
                odds_data,
            )
        )

        print(
            "Attached DraftKings market odds "
            f"to {len(enriched_games)} "
            "prediction row(s)."
        )

        return enriched_games

    except Exception as exc:
        print(
            "Warning: DraftKings market odds "
            f"unavailable: {exc}"
        )

        return games


def predict_today(
    target_date: date | None = None,
) -> dict[str, Any]:
    """
    Generate coherent pregame predictions for the requested
    Chicago calendar date and attach the current DraftKings
    market snapshot when available.
    """
    target_date = (
        target_date
        or _today_chicago()
    )

    schedule = (
        _build_pregame_data(
            target_date
        )
    )

    generated_at_utc = (
        Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    # ---------------------------------------------------------
    # No eligible games
    # ---------------------------------------------------------
    if schedule.empty:
        return {
            "generated_at_utc": (
                generated_at_utc
            ),
            "target_date": str(
                target_date
            ),
            "games": [],
            "uncertainty": {
                "full_total": {
                    "p10": None,
                    "median": None,
                    "p90": None,
                },
                "home_score": {
                    "p10": None,
                    "median": None,
                    "p90": None,
                },
            },
        }

    # ---------------------------------------------------------
    # Load production models
    # ---------------------------------------------------------
    model_payload = (
        _load_production_model()
    )

    feature_columns = (
        model_payload[
            "feature_columns"
        ]
    )

    models = (
        model_payload[
            "models"
        ]
    )

    # ---------------------------------------------------------
    # Validate prediction feature schema
    # ---------------------------------------------------------
    missing_features = [
        col
        for col in feature_columns
        if col not in schedule.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Prediction feature mismatch. "
            "Missing columns: "
            f"{missing_features}"
        )

    X = (
        schedule[
            feature_columns
        ]
        .astype(float)
        .fillna(0)
    )

    combined_rows: list[
        dict[str, Any]
    ] = []

    # ---------------------------------------------------------
    # Generate raw model outputs
    # ---------------------------------------------------------
    for idx, row in (
        schedule.iterrows()
    ):
        game_id = row[
            "game_id"
        ]

        game_predictions: dict[
            str,
            Any,
        ] = {
            "game_id": game_id,
            "home_team_id": row[
                "home_team_id"
            ],
            "away_team_id": row[
                "away_team_id"
            ],
            "game_date_utc": str(
                row[
                    "game_date_utc"
                ]
            ),
        }

        # -----------------------------------------------------
        # Descriptive fields useful to the dashboard and
        # market-matching logic
        # -----------------------------------------------------
        for optional_col in (
            "home_team",
            "away_team",
            "home_abbr",
            "away_abbr",
            "status",
            "status_detail",
            "venue",
        ):
            if (
                optional_col
                in row.index
            ):
                value = row[
                    optional_col
                ]

                if pd.isna(
                    value
                ):
                    value = None

                game_predictions[
                    optional_col
                ] = value

        X_row = X.loc[
            [idx]
        ]

        # -----------------------------------------------------
        # Predict each target
        # -----------------------------------------------------
        for (
            target,
            model,
        ) in models.items():
            try:
                if (
                    target
                    == "home_win"
                    and hasattr(
                        model,
                        "predict_proba",
                    )
                ):
                    value = (
                        model.predict_proba(
                            X_row
                        )[0, 1]
                    )

                else:
                    value = (
                        model.predict(
                            X_row
                        )[0]
                    )

            except Exception as exc:
                raise RuntimeError(
                    "Prediction failed for "
                    f"game {game_id}, "
                    f"target {target}: "
                    f"{exc}"
                ) from exc

            game_predictions[
                target
            ] = float(
                value
            )

        combined_rows.append(
            game_predictions
        )

    df = pd.DataFrame(
        combined_rows
    )

    # ---------------------------------------------------------
    # Win probabilities
    # ---------------------------------------------------------
    if (
        "home_win"
        in df.columns
    ):
        df[
            "home_win_probability"
        ] = (
            pd.to_numeric(
                df[
                    "home_win"
                ],
                errors="coerce",
            )
            .clip(
                0.0,
                1.0,
            )
        )

        df[
            "away_win_probability"
        ] = (
            1.0
            - df[
                "home_win_probability"
            ]
        )

    else:
        df[
            "home_win_probability"
        ] = None

        df[
            "away_win_probability"
        ] = None

    # ---------------------------------------------------------
    # Enforce score / quarter / half coherence
    # ---------------------------------------------------------
    df = (
        _enforce_prediction_coherence(
            df
        )
    )

    # ---------------------------------------------------------
    # Dashboard-friendly aliases
    # ---------------------------------------------------------
    df[
        "predicted_margin"
    ] = (
        df[
            "full_margin"
        ]
        if (
            "full_margin"
            in df.columns
        )
        else None
    )

    df[
        "predicted_total"
    ] = (
        df[
            "full_total"
        ]
        if (
            "full_total"
            in df.columns
        )
        else None
    )

    # ---------------------------------------------------------
    # Convert to records before attaching market data
    # ---------------------------------------------------------
    games_records = (
        df.to_dict(
            orient="records"
        )
    )

    # ---------------------------------------------------------
    # Attach latest DraftKings spread / total / moneyline
    #
    # If ODDS_API_KEY is missing or the provider is unavailable,
    # predictions still succeed without market fields.
    # ---------------------------------------------------------
    games_records = (
        _attach_current_market_odds(
            games_records
        )
    )

    # ---------------------------------------------------------
    # Uncertainty
    #
    # Use model-only DataFrame here because the market fields are
    # unrelated to model residual simulation.
    # ---------------------------------------------------------
    samples = (
        _simulate_residuals(
            model_payload,
            df,
        )
    )

    totals = (
        samples[
            "full_total"
        ]
        if (
            "full_total"
            in samples.columns
        )
        else pd.Series(
            [],
            dtype=float,
        )
    )

    scores = (
        samples[
            "home_score"
        ]
        if (
            "home_score"
            in samples.columns
        )
        else pd.Series(
            [],
            dtype=float,
        )
    )

    output = {
        "generated_at_utc": (
            generated_at_utc
        ),
        "target_date": str(
            target_date
        ),
        "games": games_records,
        "uncertainty": {
            "full_total": (
                _percentiles(
                    totals
                )
            ),
            "home_score": (
                _percentiles(
                    scores
                )
            ),
        },
    }

    # ---------------------------------------------------------
    # Latest prediction JSON / CSV
    # ---------------------------------------------------------
    PREDICTION_LATEST_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_LATEST_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PREDICTION_LATEST_JSON,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
        )

    pd.DataFrame(
        output[
            "games"
        ]
    ).to_csv(
        PREDICTION_LATEST_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # Prediction history
    #
    # This intentionally stores the market snapshot alongside
    # the model forecast so later we can evaluate:
    #
    # - model vs market at prediction time
    # - line movement
    # - closing-line value
    # ---------------------------------------------------------
    history = pd.DataFrame(
        output[
            "games"
        ]
    ).copy()

    if not history.empty:
        history[
            "prediction_date"
        ] = pd.Timestamp(
            generated_at_utc
        )

        if (
            PREDICTION_HISTORY_PATH.exists()
        ):
            existing = (
                pd.read_parquet(
                    PREDICTION_HISTORY_PATH
                )
            )

            history = pd.concat(
                [
                    existing,
                    history,
                ],
                ignore_index=True,
            )

        history = (
            history.drop_duplicates(
                subset=[
                    "game_id",
                    "prediction_date",
                ],
                keep="last",
            )
        )

        PREDICTION_HISTORY_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        history.to_parquet(
            PREDICTION_HISTORY_PATH,
            index=False,
        )

    return output


def main() -> None:
    result = (
        predict_today()
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()