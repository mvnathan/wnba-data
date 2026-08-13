from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURES_PATH, GAMES_PATH, TEAM_GAMES_PATH


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_recent_games(dates: pd.Series, window_days: int) -> pd.Series:
    dates = pd.to_datetime(dates, utc=True)
    result: list[int] = []
    for idx, current in enumerate(dates):
        prior = dates.iloc[:idx]
        start = current - np.timedelta64(window_days, "D")
        result.append(int((prior >= start).sum()))
    return pd.Series(result, index=dates.index)


def _build_elo(games: pd.DataFrame, k: float = 18.0, home_adv: float = 60.0) -> pd.DataFrame:
    rating: dict[str, float] = defaultdict(lambda: 1500.0)
    rows: list[dict[str, Any]] = []
    for _, row in games.sort_values(["game_date_utc", "game_id"]).iterrows():
        home_id = str(row["home_team_id"])
        away_id = str(row["away_team_id"])
        home_rating = rating[home_id]
        away_rating = rating[away_id]
        diff = home_rating + home_adv - away_rating
        expected_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        home_score = int(row.get("home_score") or 0)
        away_score = int(row.get("away_score") or 0)
        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        rows.append(
            {
                "game_id": row["game_id"],
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_elo": home_rating,
                "away_elo": away_rating,
                "home_opp_elo": away_rating,
                "away_opp_elo": home_rating,
            }
        )

        rating[home_id] = home_rating + k * (actual_home - expected_home)
        rating[away_id] = away_rating + k * ((1 - actual_home) - (1 - expected_home))
    return pd.DataFrame(rows)


def _build_team_game_features(team_games: pd.DataFrame) -> pd.DataFrame:
    if team_games.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "game_date_utc"])

    team_games = team_games.copy()
    team_games["game_date_utc"] = pd.to_datetime(team_games["game_date_utc"], utc=True)
    team_games = team_games.sort_values(["team_id", "game_date_utc", "game_id"]).reset_index(drop=True)

    def build_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group["season_game_number"] = group.groupby("season").cumcount() + 1
        group["games_before"] = group["season_game_number"] - 1
        group["season_points"] = group.groupby("season")["points"].cumsum().shift(1)
        group["season_allowed"] = group.groupby("season")["opp_points"].cumsum().shift(1)
        group["season_margin"] = group.groupby("season")["margin"].cumsum().shift(1)
        group["season_total"] = (group["points"] + group["opp_points"]).groupby(group["season"]).cumsum().shift(1)
        group["season_wins"] = group.groupby("season")["win"].cumsum().shift(1)
        group["season_win_pct"] = group["season_wins"] / group["games_before"].replace(0, np.nan)

        for window in (3, 5, 10):
            group[f"rolling_{window}_points"] = group["points"].rolling(window, min_periods=1).sum().shift(1)
            group[f"rolling_{window}_allowed"] = group["opp_points"].rolling(window, min_periods=1).sum().shift(1)
            group[f"rolling_{window}_margin"] = group["margin"].rolling(window, min_periods=1).sum().shift(1)
            group[f"rolling_{window}_total"] = (
                (group["points"] + group["opp_points"]).rolling(window, min_periods=1).sum().shift(1)
            )

        for quarter in ("q1", "q2", "q3", "q4"):
            group[f"last_{quarter}_points"] = group[quarter].shift(1)
            group[f"last_{quarter}_allowed"] = group[f"opp_{quarter}"].shift(1)
            group[f"last_{quarter}_margin"] = group[f"last_{quarter}_points"] - group[f"last_{quarter}_allowed"]
            group[f"last_{quarter}_total"] = group[f"last_{quarter}_points"] + group[f"last_{quarter}_allowed"]
            group[f"avg_{quarter}_points"] = (
                group.groupby("season")[quarter].cumsum().shift(1) / group["games_before"].replace(0, np.nan)
            )
            group[f"avg_{quarter}_allowed"] = (
                group.groupby("season")[f"opp_{quarter}"].cumsum().shift(1) / group["games_before"].replace(0, np.nan)
            )
            group[f"avg_{quarter}_margin"] = group[f"avg_{quarter}_points"] - group[f"avg_{quarter}_allowed"]
            group[f"avg_{quarter}_total"] = group[f"avg_{quarter}_points"] + group[f"avg_{quarter}_allowed"]

        group["last_first_half_points"] = group[["q1", "q2"]].sum(axis=1).shift(1)
        group["last_first_half_allowed"] = group[["opp_q1", "opp_q2"]].sum(axis=1).shift(1)
        group["last_first_half_margin"] = group["last_first_half_points"] - group["last_first_half_allowed"]
        group["last_first_half_total"] = group["last_first_half_points"] + group["last_first_half_allowed"]
        group["avg_first_half_points"] = (
            group[["q1", "q2"]].sum(axis=1).groupby(group["season"]).cumsum().shift(1) / group["games_before"].replace(0, np.nan)
        )
        group["avg_first_half_allowed"] = (
            group[["opp_q1", "opp_q2"]].sum(axis=1).groupby(group["season"]).cumsum().shift(1) / group["games_before"].replace(0, np.nan)
        )
        group["avg_first_half_margin"] = group["avg_first_half_points"] - group["avg_first_half_allowed"]
        group["avg_first_half_total"] = group["avg_first_half_points"] + group["avg_first_half_allowed"]

        group["last_second_half_points"] = group[["q3", "q4"]].sum(axis=1).shift(1)
        group["last_second_half_allowed"] = group[["opp_q3", "opp_q4"]].sum(axis=1).shift(1)
        group["last_second_half_margin"] = group["last_second_half_points"] - group["last_second_half_allowed"]
        group["last_second_half_total"] = group["last_second_half_points"] + group["last_second_half_allowed"]
        group["avg_second_half_points"] = (
            group[["q3", "q4"]].sum(axis=1).groupby(group["season"]).cumsum().shift(1) / group["games_before"].replace(0, np.nan)
        )
        group["avg_second_half_allowed"] = (
            group[["opp_q3", "opp_q4"]].sum(axis=1).groupby(group["season"]).cumsum().shift(1) / group["games_before"].replace(0, np.nan)
        )
        group["avg_second_half_margin"] = group["avg_second_half_points"] - group["avg_second_half_allowed"]
        group["avg_second_half_total"] = group["avg_second_half_points"] + group["avg_second_half_allowed"]

        group["home_points_season"] = group["points"].where(group["is_home"], 0).cumsum().shift(1)
        group["home_games_before"] = group["is_home"].cumsum().shift(1)
        group["home_points_per_game"] = group["home_points_season"] / group["home_games_before"].replace(0, np.nan)
        group["home_margin_season"] = group["margin"].where(group["is_home"], 0).cumsum().shift(1)
        group["home_margin_per_game"] = group["home_margin_season"] / group["home_games_before"].replace(0, np.nan)

        group["away_points_season"] = group["points"].where(~group["is_home"], 0).cumsum().shift(1)
        group["away_games_before"] = (~group["is_home"]).cumsum().shift(1)
        group["away_points_per_game"] = group["away_points_season"] / group["away_games_before"].replace(0, np.nan)
        group["away_margin_season"] = group["margin"].where(~group["is_home"], 0).cumsum().shift(1)
        group["away_margin_per_game"] = group["away_margin_season"] / group["away_games_before"].replace(0, np.nan)

        group["days_since_last_game"] = group["game_date_utc"].diff().dt.days.shift(1)
        group["days_since_last_game"] = group["days_since_last_game"].fillna(-1)
        group["back_to_back"] = group["days_since_last_game"].le(1)
        group["games_last_7"] = _count_recent_games(group["game_date_utc"], 7)
        group["games_last_14"] = _count_recent_games(group["game_date_utc"], 14)
        return group

    return team_games.groupby("team_id", group_keys=False).apply(build_group).reset_index(drop=True)


def _build_matchup_features(games: pd.DataFrame) -> pd.DataFrame:
    result: list[dict[str, Any]] = []
    pair_history: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    completed_games = games[games["completed"]].sort_values(["game_date_utc", "game_id"])
    for _, row in completed_games.iterrows():
        key = (str(row["home_team_id"]), str(row["away_team_id"]))
        prior = pair_history[key]
        last_3 = prior[-3:]
        last_q1 = [item for item in prior if item["has_q1"]]
        last_fh = [item for item in prior if item["has_first_half"]]

        result.append(
            {
                "game_id": row["game_id"],
                "head_last_3_margin": np.nan if not last_3 else np.mean([item["margin"] for item in last_3]),
                "head_last_3_total": np.nan if not last_3 else np.mean([item["total"] for item in last_3]),
                "head_q1_last_3_margin": np.nan if not last_q1 else np.mean([item["q1_margin"] for item in last_q1[-3:]]),
                "head_q1_last_3_total": np.nan if not last_q1 else np.mean([item["q1_total"] for item in last_q1[-3:]]),
                "head_fh_last_3_margin": np.nan if not last_fh else np.mean([item["first_half_margin"] for item in last_fh[-3:]]),
                "head_fh_last_3_total": np.nan if not last_fh else np.mean([item["first_half_total"] for item in last_fh[-3:]]),
            }
        )

        home_q1 = _safe_float(row.get("home_q1"))
        away_q1 = _safe_float(row.get("away_q1"))
        home_q2 = _safe_float(row.get("home_q2"))
        away_q2 = _safe_float(row.get("away_q2"))
        has_q1 = home_q1 is not None and away_q1 is not None
        has_first_half = has_q1 and home_q2 is not None and away_q2 is not None

        pair_history[key].append(
            {
                "margin": float(row.get("home_score", 0) - row.get("away_score", 0)),
                "total": float((row.get("home_score", 0) + row.get("away_score", 0))),
                "q1_margin": float((home_q1 - away_q1)) if has_q1 else np.nan,
                "q1_total": float((home_q1 + away_q1)) if has_q1 else np.nan,
                "first_half_margin": float((home_q1 + home_q2) - (away_q1 + away_q2)) if has_first_half else np.nan,
                "first_half_total": float((home_q1 + home_q2) + (away_q1 + away_q2)) if has_first_half else np.nan,
                "has_q1": has_q1,
                "has_first_half": has_first_half,
            }
        )
    return pd.DataFrame(result)


def _build_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in list(df.columns):
        if not col.startswith("home_"):
            continue

        suffix = col[len("home_"):]
        away_col = f"away_{suffix}"

        if away_col not in df.columns:
            continue

        # Difference features only make sense for numeric values.
        home_values = pd.to_numeric(df[col], errors="coerce")
        away_values = pd.to_numeric(df[away_col], errors="coerce")

        # Skip pairs that are entirely non-numeric, such as team names,
        # abbreviations, status fields, dates, IDs stored as strings, etc.
        if home_values.notna().sum() == 0 or away_values.notna().sum() == 0:
            continue

        diff_col = f"diff_{suffix}"
        df[diff_col] = home_values - away_values

    return df


def build_model_features(data_root: str = "data") -> pd.DataFrame:
    games = pd.read_parquet(Path(data_root) / "games.parquet")
    team_games = pd.read_parquet(Path(data_root) / "team_games.parquet")

    if games.empty or team_games.empty:
        return pd.DataFrame()

    games = games.copy()
    games["game_date_utc"] = pd.to_datetime(games["game_date_utc"], utc=True)
    team_games = team_games.copy()
    team_games["game_date_utc"] = pd.to_datetime(team_games["game_date_utc"], utc=True)

    team_features = _build_team_game_features(team_games)

    home_features = team_features[team_features["is_home"]].copy()
    away_features = team_features[~team_features["is_home"]].copy()

    home_features = home_features.rename(columns={col: f"home_{col}" for col in home_features.columns if col not in ["game_id", "game_date_utc", "team_id"]})
    away_features = away_features.rename(columns={col: f"away_{col}" for col in away_features.columns if col not in ["game_id", "game_date_utc", "team_id"]})

    games = games.merge(home_features[["game_id"] + [c for c in home_features.columns if c.startswith("home_")]], on="game_id", how="left")
    games = games.merge(away_features[["game_id"] + [c for c in away_features.columns if c.startswith("away_")]], on="game_id", how="left")

    elo_features = _build_elo(games[games["completed"]].copy())
    games = games.merge(elo_features, on=["game_id", "home_team_id", "away_team_id"], how="left")
    games["home_elo"] = games["home_elo"].fillna(1500)
    games["away_elo"] = games["away_elo"].fillna(1500)
    games["home_opp_elo"] = games["home_opp_elo"].fillna(1500)
    games["away_opp_elo"] = games["away_opp_elo"].fillna(1500)
    games["elo_diff"] = games["home_elo"] - games["away_elo"]

    matchup = _build_matchup_features(games)
    games = games.merge(matchup, on="game_id", how="left")
    games = _build_difference_features(games)
    games = games.fillna(0)
    return games.sort_values(["game_date_utc", "game_id"]).reset_index(drop=True)


def build_and_save(
    data_root: str = "data",
    out_path: str = "features/model_features.parquet"
) -> None:
    df = build_model_features(data_root)

    object_cols = df.select_dtypes(include=["object"]).columns

    for col in object_cols:
        col_lower = col.lower()
        s = df[col]

        # Normalize boolean-like object columns.
        normalized = (
            s.astype("string")
             .str.strip()
             .str.lower()
        )

        boolean_tokens = {"0", "1", "true", "false", "<na>"}

        if set(normalized.dropna().unique()).issubset(boolean_tokens):
            df[col] = normalized.map({
                "1": True,
                "0": False,
                "true": True,
                "false": False,
            }).astype("boolean")

        elif col_lower == "clock":
            df[col] = df[col].astype("string")
            
        # Normalize identifier / descriptive columns to strings.
        elif (
            "team" in col_lower
            or "abbr" in col_lower
            or "status" in col_lower
            or col_lower.endswith("_id")
            or col_lower == "game_id"
        ):
            df[col] = df[col].astype("string")

    Path(out_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        out_path,
        index=False,
    )


if __name__ == "__main__":
    build_and_save()
