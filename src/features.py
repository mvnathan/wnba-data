from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURES_PATH


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert mixed boolean-like values into a clean boolean Series.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "1",
            "true",
            "t",
            "yes",
            "y",
        }
    )


def _count_recent_games(
    dates: pd.Series,
    window_days: int,
) -> pd.Series:
    """
    Count games occurring strictly before each row within the
    requested number of days.
    """
    dates = pd.to_datetime(
        dates,
        utc=True,
        errors="coerce",
    )

    result: list[int] = []

    for idx, current in enumerate(dates):
        if pd.isna(current):
            result.append(0)
            continue

        prior = dates.iloc[:idx]

        start = (
            current
            - pd.Timedelta(days=window_days)
        )

        count = (
            (prior >= start)
            & (prior < current)
        ).sum()

        result.append(int(count))

    return pd.Series(
        result,
        index=dates.index,
        dtype="int64",
    )


# ---------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------


def _build_elo(
    games: pd.DataFrame,
    k: float = 18.0,
    home_adv: float = 60.0,
) -> pd.DataFrame:
    """
    Build pregame Elo ratings for EVERY game.

    Ratings are recorded before the game.

    Completed games update Elo afterward.

    Scheduled/unplayed games receive the current ratings but do not
    alter them. This is critical for real pregame predictions.
    """
    if games.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "home_team_id",
                "away_team_id",
                "home_elo",
                "away_elo",
                "home_opp_elo",
                "away_opp_elo",
            ]
        )

    games = games.copy()

    games["game_date_utc"] = pd.to_datetime(
        games["game_date_utc"],
        utc=True,
        errors="coerce",
    )

    games["game_id"] = (
        games["game_id"]
        .astype("string")
    )

    games["home_team_id"] = (
        games["home_team_id"]
        .astype("string")
    )

    games["away_team_id"] = (
        games["away_team_id"]
        .astype("string")
    )

    completed = _coerce_bool_series(
        games["completed"]
    )

    rating: dict[str, float] = defaultdict(
        lambda: 1500.0
    )

    rows: list[dict[str, Any]] = []

    ordered = games.sort_values(
        [
            "game_date_utc",
            "game_id",
        ]
    )

    for idx, row in ordered.iterrows():
        home_id = str(
            row["home_team_id"]
        )

        away_id = str(
            row["away_team_id"]
        )

        home_rating = rating[home_id]
        away_rating = rating[away_id]

        rows.append(
            {
                "game_id": str(
                    row["game_id"]
                ),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_elo": home_rating,
                "away_elo": away_rating,
                "home_opp_elo": away_rating,
                "away_opp_elo": home_rating,
            }
        )

        # Scheduled games receive current Elo,
        # but MUST NOT update the ratings.
        if not bool(completed.loc[idx]):
            continue

        home_score = _safe_float(
            row.get("home_score")
        )

        away_score = _safe_float(
            row.get("away_score")
        )

        if (
            home_score is None
            or away_score is None
        ):
            continue

        elo_diff = (
            home_rating
            + home_adv
            - away_rating
        )

        expected_home = (
            1.0
            / (
                1.0
                + 10
                ** (
                    -elo_diff
                    / 400.0
                )
            )
        )

        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        rating[home_id] = (
            home_rating
            + k
            * (
                actual_home
                - expected_home
            )
        )

        rating[away_id] = (
            away_rating
            + k
            * (
                (
                    1.0
                    - actual_home
                )
                - (
                    1.0
                    - expected_home
                )
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Scheduled-team row expansion
# ---------------------------------------------------------------------


def _expand_team_games_with_schedule(
    team_games: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add synthetic team-game rows for scheduled games that do not
    exist in team_games.parquet.

    The synthetic rows contain team/opponent/date/home-away identity,
    but outcome fields remain NaN.

    Running the normal historical feature logic over these synthetic
    rows gives scheduled games the state accumulated from completed
    games strictly before tipoff.
    """
    team_games = team_games.copy()
    games = games.copy()

    team_games["game_id"] = (
        team_games["game_id"]
        .astype("string")
    )

    games["game_id"] = (
        games["game_id"]
        .astype("string")
    )

    team_games["team_id"] = (
        team_games["team_id"]
        .astype("string")
    )

    if "opp_team_id" in team_games.columns:
        team_games["opp_team_id"] = (
            team_games["opp_team_id"]
            .astype("string")
        )

    games["home_team_id"] = (
        games["home_team_id"]
        .astype("string")
    )

    games["away_team_id"] = (
        games["away_team_id"]
        .astype("string")
    )

    team_games["game_date_utc"] = (
        pd.to_datetime(
            team_games["game_date_utc"],
            utc=True,
            errors="coerce",
        )
    )

    games["game_date_utc"] = (
        pd.to_datetime(
            games["game_date_utc"],
            utc=True,
            errors="coerce",
        )
    )

    existing_game_ids = set(
        team_games[
            "game_id"
        ].dropna()
    )

    missing_games = games[
        ~games["game_id"].isin(
            existing_game_ids
        )
    ].copy()

    if missing_games.empty:
        return team_games

    synthetic_rows: list[
        dict[str, Any]
    ] = []

    for _, game in missing_games.iterrows():

        home_row = {
            "game_id": str(
                game["game_id"]
            ),
            "game_date_utc": game[
                "game_date_utc"
            ],
            "season": game.get(
                "season"
            ),
            "team_id": str(
                game["home_team_id"]
            ),
            "team": game.get(
                "home_team"
            ),
            "team_abbr": game.get(
                "home_abbr"
            ),
            "opp_team_id": str(
                game["away_team_id"]
            ),
            "opp_team": game.get(
                "away_team"
            ),
            "opp_abbr": game.get(
                "away_abbr"
            ),
            "is_home": True,
            "updated_at_utc": game.get(
                "updated_at_utc"
            ),
        }

        away_row = {
            "game_id": str(
                game["game_id"]
            ),
            "game_date_utc": game[
                "game_date_utc"
            ],
            "season": game.get(
                "season"
            ),
            "team_id": str(
                game["away_team_id"]
            ),
            "team": game.get(
                "away_team"
            ),
            "team_abbr": game.get(
                "away_abbr"
            ),
            "opp_team_id": str(
                game["home_team_id"]
            ),
            "opp_team": game.get(
                "home_team"
            ),
            "opp_abbr": game.get(
                "home_abbr"
            ),
            "is_home": False,
            "updated_at_utc": game.get(
                "updated_at_utc"
            ),
        }

        synthetic_rows.extend(
            [
                home_row,
                away_row,
            ]
        )

    synthetic = pd.DataFrame(
        synthetic_rows
    )

    # Guarantee synthetic rows have every
    # column present in team_games.
    for col in team_games.columns:
        if col not in synthetic.columns:
            synthetic[col] = np.nan

    # Ignore any extra synthetic fields
    # not present in the real schema.
    synthetic = synthetic[
        team_games.columns
    ]

    expanded = pd.concat(
        [
            team_games,
            synthetic,
        ],
        ignore_index=True,
        sort=False,
    )

    return expanded


# ---------------------------------------------------------------------
# Historical team features
# ---------------------------------------------------------------------


def _build_team_game_features(
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build team-level pregame features.

    Every statistic represents information available BEFORE the
    corresponding game.

    Scheduled synthetic rows therefore inherit the historical state
    from the team's completed games.
    """
    if team_games.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team_id",
                "game_date_utc",
            ]
        )

    team_games = team_games.copy()

    team_games["game_id"] = (
        team_games["game_id"]
        .astype("string")
    )

    team_games["team_id"] = (
        team_games["team_id"]
        .astype("string")
    )

    team_games["game_date_utc"] = (
        pd.to_datetime(
            team_games[
                "game_date_utc"
            ],
            utc=True,
            errors="coerce",
        )
    )

    team_games["is_home"] = (
        _coerce_bool_series(
            team_games["is_home"]
        )
    )

    numeric_columns = [
        "points",
        "opp_points",
        "margin",
        "win",
        "q1",
        "q2",
        "q3",
        "q4",
        "ot1",
        "ot2",
        "first_half_points",
        "second_half_points",
        "regulation_points",
        "opp_q1",
        "opp_q2",
        "opp_q3",
        "opp_q4",
        "q1_margin",
        "q2_margin",
        "q3_margin",
        "q4_margin",
        "first_half_margin",
    ]

    for col in numeric_columns:
        if col in team_games.columns:
            team_games[col] = (
                pd.to_numeric(
                    team_games[col],
                    errors="coerce",
                )
            )

    # Guarantee fields used below exist.
    for col in (
        "points",
        "opp_points",
        "margin",
        "win",
        "q1",
        "q2",
        "q3",
        "q4",
        "opp_q1",
        "opp_q2",
        "opp_q3",
        "opp_q4",
    ):
        if col not in team_games.columns:
            team_games[col] = np.nan

    team_games = (
        team_games.sort_values(
            [
                "team_id",
                "game_date_utc",
                "game_id",
            ]
        )
        .reset_index(drop=True)
    )

    def build_group(
        group: pd.DataFrame,
    ) -> pd.DataFrame:
        group = group.copy()

        group = group.sort_values(
            [
                "game_date_utc",
                "game_id",
            ]
        ).copy()

        # -------------------------------------------------------------
        # Season counts
        # -------------------------------------------------------------
        group["season_game_number"] = (
            group.groupby(
                "season",
                dropna=False,
            )
            .cumcount()
            + 1
        )

        group["games_before"] = (
            group[
                "season_game_number"
            ]
            - 1
        )

        # -------------------------------------------------------------
        # Season cumulative values BEFORE current game
        # -------------------------------------------------------------
        for source, target in (
            (
                "points",
                "season_points",
            ),
            (
                "opp_points",
                "season_allowed",
            ),
            (
                "margin",
                "season_margin",
            ),
            (
                "win",
                "season_wins",
            ),
        ):
            group[target] = (
                group.groupby(
                    "season",
                    dropna=False,
                )[source]
                .transform(
                    lambda s:
                    s.cumsum()
                    .shift(1)
                )
            )

        total_points = (
            group["points"]
            + group["opp_points"]
        )

        group["season_total"] = (
            total_points.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.cumsum()
                .shift(1)
            )
        )

        group["season_win_pct"] = (
            group["season_wins"]
            / group[
                "games_before"
            ].replace(
                0,
                np.nan,
            )
        )

        # -------------------------------------------------------------
        # Rolling recent form
        # -------------------------------------------------------------
        for window in (
            3,
            5,
            10,
        ):
            for source, suffix in (
                (
                    "points",
                    "points",
                ),
                (
                    "opp_points",
                    "allowed",
                ),
                (
                    "margin",
                    "margin",
                ),
            ):
                group[
                    f"rolling_{window}_{suffix}"
                ] = (
                    group.groupby(
                        "season",
                        dropna=False,
                    )[source]
                    .transform(
                        lambda s:
                        s.shift(1)
                        .rolling(
                            window,
                            min_periods=1,
                        )
                        .sum()
                    )
                )

            group[
                f"rolling_{window}_total"
            ] = (
                total_points.groupby(
                    group["season"],
                    dropna=False,
                )
                .transform(
                    lambda s:
                    s.shift(1)
                    .rolling(
                        window,
                        min_periods=1,
                    )
                    .sum()
                )
            )

        # -------------------------------------------------------------
        # Quarter history
        # -------------------------------------------------------------
        for quarter in (
            "q1",
            "q2",
            "q3",
            "q4",
        ):
            opp_quarter = (
                f"opp_{quarter}"
            )

            group[
                f"last_{quarter}_points"
            ] = (
                group.groupby(
                    "season",
                    dropna=False,
                )[quarter]
                .shift(1)
            )

            group[
                f"last_{quarter}_allowed"
            ] = (
                group.groupby(
                    "season",
                    dropna=False,
                )[opp_quarter]
                .shift(1)
            )

            group[
                f"last_{quarter}_margin"
            ] = (
                group[
                    f"last_{quarter}_points"
                ]
                - group[
                    f"last_{quarter}_allowed"
                ]
            )

            group[
                f"last_{quarter}_total"
            ] = (
                group[
                    f"last_{quarter}_points"
                ]
                + group[
                    f"last_{quarter}_allowed"
                ]
            )

            group[
                f"avg_{quarter}_points"
            ] = (
                group.groupby(
                    "season",
                    dropna=False,
                )[quarter]
                .transform(
                    lambda s:
                    s.shift(1)
                    .expanding()
                    .mean()
                )
            )

            group[
                f"avg_{quarter}_allowed"
            ] = (
                group.groupby(
                    "season",
                    dropna=False,
                )[opp_quarter]
                .transform(
                    lambda s:
                    s.shift(1)
                    .expanding()
                    .mean()
                )
            )

            group[
                f"avg_{quarter}_margin"
            ] = (
                group[
                    f"avg_{quarter}_points"
                ]
                - group[
                    f"avg_{quarter}_allowed"
                ]
            )

            group[
                f"avg_{quarter}_total"
            ] = (
                group[
                    f"avg_{quarter}_points"
                ]
                + group[
                    f"avg_{quarter}_allowed"
                ]
            )

        # -------------------------------------------------------------
        # Half history
        # -------------------------------------------------------------
        first_half_points = (
            group["q1"]
            + group["q2"]
        )

        first_half_allowed = (
            group["opp_q1"]
            + group["opp_q2"]
        )

        second_half_points = (
            group["q3"]
            + group["q4"]
        )

        second_half_allowed = (
            group["opp_q3"]
            + group["opp_q4"]
        )

        group[
            "last_first_half_points"
        ] = (
            first_half_points.groupby(
                group["season"],
                dropna=False,
            )
            .shift(1)
        )

        group[
            "last_first_half_allowed"
        ] = (
            first_half_allowed.groupby(
                group["season"],
                dropna=False,
            )
            .shift(1)
        )

        group[
            "last_first_half_margin"
        ] = (
            group[
                "last_first_half_points"
            ]
            - group[
                "last_first_half_allowed"
            ]
        )

        group[
            "last_first_half_total"
        ] = (
            group[
                "last_first_half_points"
            ]
            + group[
                "last_first_half_allowed"
            ]
        )

        group[
            "avg_first_half_points"
        ] = (
            first_half_points.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.shift(1)
                .expanding()
                .mean()
            )
        )

        group[
            "avg_first_half_allowed"
        ] = (
            first_half_allowed.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.shift(1)
                .expanding()
                .mean()
            )
        )

        group[
            "avg_first_half_margin"
        ] = (
            group[
                "avg_first_half_points"
            ]
            - group[
                "avg_first_half_allowed"
            ]
        )

        group[
            "avg_first_half_total"
        ] = (
            group[
                "avg_first_half_points"
            ]
            + group[
                "avg_first_half_allowed"
            ]
        )

        group[
            "last_second_half_points"
        ] = (
            second_half_points.groupby(
                group["season"],
                dropna=False,
            )
            .shift(1)
        )

        group[
            "last_second_half_allowed"
        ] = (
            second_half_allowed.groupby(
                group["season"],
                dropna=False,
            )
            .shift(1)
        )

        group[
            "last_second_half_margin"
        ] = (
            group[
                "last_second_half_points"
            ]
            - group[
                "last_second_half_allowed"
            ]
        )

        group[
            "last_second_half_total"
        ] = (
            group[
                "last_second_half_points"
            ]
            + group[
                "last_second_half_allowed"
            ]
        )

        group[
            "avg_second_half_points"
        ] = (
            second_half_points.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.shift(1)
                .expanding()
                .mean()
            )
        )

        group[
            "avg_second_half_allowed"
        ] = (
            second_half_allowed.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.shift(1)
                .expanding()
                .mean()
            )
        )

        group[
            "avg_second_half_margin"
        ] = (
            group[
                "avg_second_half_points"
            ]
            - group[
                "avg_second_half_allowed"
            ]
        )

        group[
            "avg_second_half_total"
        ] = (
            group[
                "avg_second_half_points"
            ]
            + group[
                "avg_second_half_allowed"
            ]
        )

        # -------------------------------------------------------------
        # Home / away splits
        # -------------------------------------------------------------
        home_points = (
            group["points"]
            .where(
                group["is_home"],
                0.0,
            )
        )

        home_margin = (
            group["margin"]
            .where(
                group["is_home"],
                0.0,
            )
        )

        home_games = (
            group[
                "is_home"
            ].astype(int)
        )

        away_points = (
            group["points"]
            .where(
                ~group["is_home"],
                0.0,
            )
        )

        away_margin = (
            group["margin"]
            .where(
                ~group["is_home"],
                0.0,
            )
        )

        away_games = (
            (~group["is_home"])
            .astype(int)
        )

        group[
            "home_points_season"
        ] = (
            home_points.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.cumsum()
                .shift(1)
            )
        )

        group[
            "home_games_before"
        ] = (
            home_games.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.cumsum()
                .shift(1)
            )
        )

        group[
            "home_margin_season"
        ] = (
            home_margin.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.cumsum()
                .shift(1)
            )
        )

        group[
            "home_points_per_game"
        ] = (
            group[
                "home_points_season"
            ]
            / group[
                "home_games_before"
            ].replace(
                0,
                np.nan,
            )
        )

        group[
            "home_margin_per_game"
        ] = (
            group[
                "home_margin_season"
            ]
            / group[
                "home_games_before"
            ].replace(
                0,
                np.nan,
            )
        )

        group[
            "away_points_season"
        ] = (
            away_points.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.cumsum()
                .shift(1)
            )
        )

        group[
            "away_games_before"
        ] = (
            away_games.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.cumsum()
                .shift(1)
            )
        )

        group[
            "away_margin_season"
        ] = (
            away_margin.groupby(
                group["season"],
                dropna=False,
            )
            .transform(
                lambda s:
                s.cumsum()
                .shift(1)
            )
        )

        group[
            "away_points_per_game"
        ] = (
            group[
                "away_points_season"
            ]
            / group[
                "away_games_before"
            ].replace(
                0,
                np.nan,
            )
        )

        group[
            "away_margin_per_game"
        ] = (
            group[
                "away_margin_season"
            ]
            / group[
                "away_games_before"
            ].replace(
                0,
                np.nan,
            )
        )

        # -------------------------------------------------------------
        # Rest / schedule
        # -------------------------------------------------------------
        group[
            "days_since_last_game"
        ] = (
            group[
                "game_date_utc"
            ]
            .diff()
            .dt.days
        )

        group[
            "days_since_last_game"
        ] = (
            group[
                "days_since_last_game"
            ]
            .fillna(-1)
        )

        group[
            "back_to_back"
        ] = (
            group[
                "days_since_last_game"
            ]
            .between(
                0,
                1,
            )
        )

        group[
            "games_last_7"
        ] = _count_recent_games(
            group[
                "game_date_utc"
            ],
            7,
        )

        group[
            "games_last_14"
        ] = _count_recent_games(
            group[
                "game_date_utc"
            ],
            14,
        )

        return group

    groups = []

    for _, group in team_games.groupby(
        "team_id",
        sort=False,
    ):
        groups.append(
            build_group(group)
        )

    if not groups:
        return pd.DataFrame()

    return (
        pd.concat(
            groups,
            ignore_index=True,
        )
        .sort_values(
            [
                "game_date_utc",
                "game_id",
                "team_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Head-to-head features
# ---------------------------------------------------------------------


def _build_matchup_features(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build head-to-head features for every game.

    The feature row is created before the current result is added to
    pair history, so it is always pregame-safe.

    Scheduled games therefore receive prior matchup history without
    altering it.
    """
    if games.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
            ]
        )

    games = games.copy()

    games["game_date_utc"] = (
        pd.to_datetime(
            games[
                "game_date_utc"
            ],
            utc=True,
            errors="coerce",
        )
    )

    games["game_id"] = (
        games["game_id"]
        .astype("string")
    )

    completed = _coerce_bool_series(
        games["completed"]
    )

    result: list[
        dict[str, Any]
    ] = []

    pair_history: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    ordered = games.sort_values(
        [
            "game_date_utc",
            "game_id",
        ]
    )

    for idx, row in ordered.iterrows():
        home_id = str(
            row[
                "home_team_id"
            ]
        )

        away_id = str(
            row[
                "away_team_id"
            ]
        )

        key = (
            home_id,
            away_id,
        )

        prior = pair_history[key]

        last_3 = prior[-3:]

        q1_history = [
            item
            for item in prior
            if item[
                "has_q1"
            ]
        ]

        fh_history = [
            item
            for item in prior
            if item[
                "has_first_half"
            ]
        ]

        result.append(
            {
                "game_id": str(
                    row["game_id"]
                ),
                "head_last_3_margin": (
                    np.nan
                    if not last_3
                    else float(
                        np.mean(
                            [
                                item[
                                    "margin"
                                ]
                                for item
                                in last_3
                            ]
                        )
                    )
                ),
                "head_last_3_total": (
                    np.nan
                    if not last_3
                    else float(
                        np.mean(
                            [
                                item[
                                    "total"
                                ]
                                for item
                                in last_3
                            ]
                        )
                    )
                ),
                "head_q1_last_3_margin": (
                    np.nan
                    if not q1_history
                    else float(
                        np.mean(
                            [
                                item[
                                    "q1_margin"
                                ]
                                for item
                                in q1_history[
                                    -3:
                                ]
                            ]
                        )
                    )
                ),
                "head_q1_last_3_total": (
                    np.nan
                    if not q1_history
                    else float(
                        np.mean(
                            [
                                item[
                                    "q1_total"
                                ]
                                for item
                                in q1_history[
                                    -3:
                                ]
                            ]
                        )
                    )
                ),
                "head_fh_last_3_margin": (
                    np.nan
                    if not fh_history
                    else float(
                        np.mean(
                            [
                                item[
                                    "first_half_margin"
                                ]
                                for item
                                in fh_history[
                                    -3:
                                ]
                            ]
                        )
                    )
                ),
                "head_fh_last_3_total": (
                    np.nan
                    if not fh_history
                    else float(
                        np.mean(
                            [
                                item[
                                    "first_half_total"
                                ]
                                for item
                                in fh_history[
                                    -3:
                                ]
                            ]
                        )
                    )
                ),
            }
        )

        # Future games receive prior history
        # but MUST NOT modify it.
        if not bool(
            completed.loc[idx]
        ):
            continue

        home_score = _safe_float(
            row.get(
                "home_score"
            )
        )

        away_score = _safe_float(
            row.get(
                "away_score"
            )
        )

        if (
            home_score is None
            or away_score is None
        ):
            continue

        home_q1 = _safe_float(
            row.get(
                "home_q1"
            )
        )

        away_q1 = _safe_float(
            row.get(
                "away_q1"
            )
        )

        home_q2 = _safe_float(
            row.get(
                "home_q2"
            )
        )

        away_q2 = _safe_float(
            row.get(
                "away_q2"
            )
        )

        has_q1 = (
            home_q1 is not None
            and away_q1 is not None
        )

        has_first_half = (
            has_q1
            and home_q2 is not None
            and away_q2 is not None
        )

        pair_history[key].append(
            {
                "margin": (
                    home_score
                    - away_score
                ),
                "total": (
                    home_score
                    + away_score
                ),
                "q1_margin": (
                    (
                        home_q1
                        - away_q1
                    )
                    if has_q1
                    else np.nan
                ),
                "q1_total": (
                    (
                        home_q1
                        + away_q1
                    )
                    if has_q1
                    else np.nan
                ),
                "first_half_margin": (
                    (
                        home_q1
                        + home_q2
                        - away_q1
                        - away_q2
                    )
                    if has_first_half
                    else np.nan
                ),
                "first_half_total": (
                    (
                        home_q1
                        + home_q2
                        + away_q1
                        + away_q2
                    )
                    if has_first_half
                    else np.nan
                ),
                "has_q1": has_q1,
                "has_first_half": (
                    has_first_half
                ),
            }
        )

    return pd.DataFrame(
        result
    )


# ---------------------------------------------------------------------
# Difference features
# ---------------------------------------------------------------------


def _build_difference_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build numeric home-minus-away difference features.

    Boolean values are converted to 1.0 / 0.0 before subtraction.
    Columns are accumulated and concatenated once to avoid DataFrame
    fragmentation warnings.
    """
    df = df.copy()

    diff_columns: dict[str, pd.Series] = {}

    for col in list(df.columns):
        if not col.startswith("home_"):
            continue

        suffix = col[len("home_"):]
        away_col = f"away_{suffix}"

        if away_col not in df.columns:
            continue

        home_values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        away_values = pd.to_numeric(
            df[away_col],
            errors="coerce",
        )

        # Skip pairs that contain no usable numeric values.
        if (
            home_values.notna().sum() == 0
            or away_values.notna().sum() == 0
        ):
            continue

        # NumPy does not support bool - bool.
        # Converting to float makes True=1.0 and False=0.0.
        home_values = home_values.astype(float)
        away_values = away_values.astype(float)

        diff_columns[f"diff_{suffix}"] = (
            home_values - away_values
        )

    if diff_columns:
        diff_df = pd.DataFrame(
            diff_columns,
            index=df.index,
        )

        # Replace any difference columns that already exist.
        duplicate_cols = [
            col
            for col in diff_df.columns
            if col in df.columns
        ]

        if duplicate_cols:
            df = df.drop(
                columns=duplicate_cols
            )

        df = pd.concat(
            [df, diff_df],
            axis=1,
        )

    return df


# ---------------------------------------------------------------------
# Full model feature builder
# ---------------------------------------------------------------------


def build_model_features(
    data_root: str = "data",
) -> pd.DataFrame:
    games_path = (
        Path(data_root)
        / "games.parquet"
    )

    team_games_path = (
        Path(data_root)
        / "team_games.parquet"
    )

    games = pd.read_parquet(
        games_path
    )

    team_games = pd.read_parquet(
        team_games_path
    )

    if games.empty:
        return pd.DataFrame()

    games = games.copy()

    games["game_id"] = (
        games["game_id"]
        .astype("string")
    )

    games["home_team_id"] = (
        games["home_team_id"]
        .astype("string")
    )

    games["away_team_id"] = (
        games["away_team_id"]
        .astype("string")
    )

    games["game_date_utc"] = (
        pd.to_datetime(
            games[
                "game_date_utc"
            ],
            utc=True,
            errors="coerce",
        )
    )

    games["completed"] = (
        _coerce_bool_series(
            games["completed"]
        )
    )

    if team_games.empty:
        return pd.DataFrame()

    team_games = team_games.copy()

    team_games["game_id"] = (
        team_games["game_id"]
        .astype("string")
    )

    team_games["team_id"] = (
        team_games["team_id"]
        .astype("string")
    )

    if "opp_team_id" in team_games.columns:
        team_games[
            "opp_team_id"
        ] = (
            team_games[
                "opp_team_id"
            ]
            .astype("string")
        )

    team_games[
        "game_date_utc"
    ] = (
        pd.to_datetime(
            team_games[
                "game_date_utc"
            ],
            utc=True,
            errors="coerce",
        )
    )

    # ---------------------------------------------------------
    # Add scheduled-game team rows before feature construction.
    # ---------------------------------------------------------
    expanded_team_games = (
        _expand_team_games_with_schedule(
            team_games,
            games,
        )
    )

    original_team_columns = set(
        expanded_team_games.columns
    )

    team_features = (
        _build_team_game_features(
            expanded_team_games
        )
    )

    # Derived feature columns only.
    derived_team_features = [
        c
        for c in team_features.columns
        if c not in original_team_columns
    ]

    # Raw outcome fields required as training targets.
    raw_target_fields = [
        c
        for c in (
            "q1",
            "q2",
            "q3",
            "q4",
            "first_half_points",
            "second_half_points",
        )
        if c in team_features.columns
    ]

    side_fields = (
        derived_team_features
        + raw_target_fields
    )

    # ---------------------------------------------------------
    # Home team features
    # ---------------------------------------------------------
    home_features = (
        team_features[
            team_features[
                "is_home"
            ]
        ]
        .copy()
    )

    home_features = (
        home_features[
            [
                "game_id",
                *side_fields,
            ]
        ]
        .rename(
            columns={
                col: f"home_{col}"
                for col
                in side_fields
            }
        )
    )

    # ---------------------------------------------------------
    # Away team features
    # ---------------------------------------------------------
    away_features = (
        team_features[
            ~team_features[
                "is_home"
            ]
        ]
        .copy()
    )

    away_features = (
        away_features[
            [
                "game_id",
                *side_fields,
            ]
        ]
        .rename(
            columns={
                col: f"away_{col}"
                for col
                in side_fields
            }
        )
    )

    games = games.merge(
        home_features,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    games = games.merge(
        away_features,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Elo for historical AND scheduled games
    # ---------------------------------------------------------
    elo_features = (
        _build_elo(
            games.copy()
        )
    )

    games = games.merge(
        elo_features,
        on=[
            "game_id",
            "home_team_id",
            "away_team_id",
        ],
        how="left",
        validate="one_to_one",
    )

    games["home_elo"] = (
        pd.to_numeric(
            games["home_elo"],
            errors="coerce",
        )
        .fillna(1500.0)
    )

    games["away_elo"] = (
        pd.to_numeric(
            games["away_elo"],
            errors="coerce",
        )
        .fillna(1500.0)
    )

    games["home_opp_elo"] = (
        pd.to_numeric(
            games[
                "home_opp_elo"
            ],
            errors="coerce",
        )
        .fillna(1500.0)
    )

    games["away_opp_elo"] = (
        pd.to_numeric(
            games[
                "away_opp_elo"
            ],
            errors="coerce",
        )
        .fillna(1500.0)
    )

    games["elo_diff"] = (
        games["home_elo"]
        - games["away_elo"]
    )

    # ---------------------------------------------------------
    # Pregame head-to-head history
    # ---------------------------------------------------------
    matchup = (
        _build_matchup_features(
            games
        )
    )

    games = games.merge(
        matchup,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Home-away difference features
    # ---------------------------------------------------------
    games = (
        _build_difference_features(
            games
        )
    )

    # ---------------------------------------------------------
    # Training targets
    #
    # IMPORTANT:
    # Only completed games receive outcome labels.
    # Future games retain NaN targets.
    # ---------------------------------------------------------
    completed_mask = (
        games["completed"]
        .fillna(False)
        .astype(bool)
    )

    home_score_raw = (
        pd.to_numeric(
            games["home_score"],
            errors="coerce",
        )
    )

    away_score_raw = (
        pd.to_numeric(
            games["away_score"],
            errors="coerce",
        )
    )

    final_valid = (
        completed_mask
        & home_score_raw.notna()
        & away_score_raw.notna()
    )

    games["home_score"] = (
        home_score_raw.where(
            final_valid
        )
    )

    games["away_score"] = (
        away_score_raw.where(
            final_valid
        )
    )

    games["home_win"] = (
        (
            home_score_raw
            > away_score_raw
        )
        .astype(float)
        .where(
            final_valid
        )
    )

    games["full_margin"] = (
        (
            home_score_raw
            - away_score_raw
        )
        .where(
            final_valid
        )
    )

    games["full_total"] = (
        (
            home_score_raw
            + away_score_raw
        )
        .where(
            final_valid
        )
    )

    # ---------------------------------------------------------
    # Quarter targets
    # ---------------------------------------------------------
    quarter_raw: dict[
        str,
        tuple[
            pd.Series,
            pd.Series,
        ],
    ] = {}

    for quarter in (
        "q1",
        "q2",
        "q3",
        "q4",
    ):
        home_col = (
            f"home_{quarter}"
        )

        away_col = (
            f"away_{quarter}"
        )

        if home_col in games.columns:
            home_raw = (
                pd.to_numeric(
                    games[
                        home_col
                    ],
                    errors="coerce",
                )
            )
        else:
            home_raw = pd.Series(
                np.nan,
                index=games.index,
                dtype=float,
            )

        if away_col in games.columns:
            away_raw = (
                pd.to_numeric(
                    games[
                        away_col
                    ],
                    errors="coerce",
                )
            )
        else:
            away_raw = pd.Series(
                np.nan,
                index=games.index,
                dtype=float,
            )

        quarter_raw[
            quarter
        ] = (
            home_raw,
            away_raw,
        )

        valid = (
            completed_mask
            & home_raw.notna()
            & away_raw.notna()
        )

        games[
            home_col
        ] = home_raw.where(
            valid
        )

        games[
            away_col
        ] = away_raw.where(
            valid
        )

        games[
            f"{quarter}_margin"
        ] = (
            home_raw
            - away_raw
        ).where(
            valid
        )

        games[
            f"{quarter}_total"
        ] = (
            home_raw
            + away_raw
        ).where(
            valid
        )

    # ---------------------------------------------------------
    # Half targets
    # ---------------------------------------------------------
    if (
        "home_first_half_points"
        in games.columns
    ):
        home_first_half_raw = (
            pd.to_numeric(
                games[
                    "home_first_half_points"
                ],
                errors="coerce",
            )
        )
    else:
        home_first_half_raw = (
            quarter_raw["q1"][0]
            + quarter_raw["q2"][0]
        )

    if (
        "away_first_half_points"
        in games.columns
    ):
        away_first_half_raw = (
            pd.to_numeric(
                games[
                    "away_first_half_points"
                ],
                errors="coerce",
            )
        )
    else:
        away_first_half_raw = (
            quarter_raw["q1"][1]
            + quarter_raw["q2"][1]
        )

    first_half_valid = (
        completed_mask
        & home_first_half_raw.notna()
        & away_first_half_raw.notna()
    )

    games[
        "home_first_half"
    ] = (
        home_first_half_raw.where(
            first_half_valid
        )
    )

    games[
        "away_first_half"
    ] = (
        away_first_half_raw.where(
            first_half_valid
        )
    )

    games[
        "first_half_margin"
    ] = (
        (
            home_first_half_raw
            - away_first_half_raw
        )
        .where(
            first_half_valid
        )
    )

    games[
        "first_half_total"
    ] = (
        (
            home_first_half_raw
            + away_first_half_raw
        )
        .where(
            first_half_valid
        )
    )

    if (
        "home_second_half_points"
        in games.columns
    ):
        home_second_half_raw = (
            pd.to_numeric(
                games[
                    "home_second_half_points"
                ],
                errors="coerce",
            )
        )
    else:
        home_second_half_raw = (
            quarter_raw["q3"][0]
            + quarter_raw["q4"][0]
        )

    if (
        "away_second_half_points"
        in games.columns
    ):
        away_second_half_raw = (
            pd.to_numeric(
                games[
                    "away_second_half_points"
                ],
                errors="coerce",
            )
        )
    else:
        away_second_half_raw = (
            quarter_raw["q3"][1]
            + quarter_raw["q4"][1]
        )

    second_half_valid = (
        completed_mask
        & home_second_half_raw.notna()
        & away_second_half_raw.notna()
    )

    games[
        "home_second_half"
    ] = (
        home_second_half_raw.where(
            second_half_valid
        )
    )

    games[
        "away_second_half"
    ] = (
        away_second_half_raw.where(
            second_half_valid
        )
    )

    games[
        "second_half_margin"
    ] = (
        (
            home_second_half_raw
            - away_second_half_raw
        )
        .where(
            second_half_valid
        )
    )

    games[
        "second_half_total"
    ] = (
        (
            home_second_half_raw
            + away_second_half_raw
        )
        .where(
            second_half_valid
        )
    )

    return (
        games.sort_values(
            [
                "game_date_utc",
                "game_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# Save feature table
# ---------------------------------------------------------------------


def build_and_save(
    data_root: str = "data",
    out_path: str | Path = FEATURES_PATH,
) -> None:
    df = build_model_features(
        data_root
    )

    out_path = Path(
        out_path
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Normalize datatypes for Arrow/Parquet.
    #
    # Numeric and boolean columns remain numeric/bool.
    # Mixed object columns become pandas strings so PyArrow does not
    # encounter mixed int/string or bool/int object columns.
    # ---------------------------------------------------------
    object_columns = (
        df.select_dtypes(
            include=["object"]
        )
        .columns
    )

    for col in object_columns:
        df[col] = (
            df[col]
            .astype("string")
        )

    # Explicitly retain completed as bool.
    if "completed" in df.columns:
        df["completed"] = (
            _coerce_bool_series(
                df["completed"]
            )
        )

    df.to_parquet(
        out_path,
        index=False,
    )