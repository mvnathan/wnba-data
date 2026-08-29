from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.advanced_features import (
    PLAYER_BOX_STATS_PATH,
    TEAM_BOX_STATS_PATH,
    build_possession_features,
    build_travel_features,
)
from src.config import GAMES_PATH
from src.storage import write_parquet_atomic


def _rotation_features(games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    if games.empty or players.empty:
        return pd.DataFrame(columns=["game_id"])

    schedule = games[["game_id", "game_date_utc", "home_team_id", "away_team_id"]].copy()
    schedule["game_id"] = schedule["game_id"].astype("string")
    schedule["game_date_utc"] = pd.to_datetime(schedule["game_date_utc"], utc=True, errors="coerce")

    players = players.copy()
    players["game_id"] = players["game_id"].astype("string")
    players["team_id"] = players["team_id"].astype("string")
    players["player_id"] = players["player_id"].astype("string")
    players["minutes"] = pd.to_numeric(players.get("minutes"), errors="coerce")
    players["starter"] = players.get("starter", False).fillna(False).astype(bool)
    players = players.merge(schedule[["game_id", "game_date_utc"]], on="game_id", how="left")

    rows: list[dict] = []
    for _, game in schedule.sort_values(["game_date_utc", "game_id"]).iterrows():
        game_time = game["game_date_utc"]
        out: dict = {"game_id": str(game["game_id"])}

        for side in ("home", "away"):
            team_id = str(game[f"{side}_team_id"])
            prior = players[(players["team_id"] == team_id) & (players["game_date_utc"] < game_time)].copy()
            if prior.empty:
                continue

            prior_games = (
                prior[["game_id", "game_date_utc"]]
                .drop_duplicates()
                .sort_values(["game_date_utc", "game_id"])
            )
            last_ids = prior_games.tail(5)["game_id"].astype(str).tolist()
            recent = prior[prior["game_id"].astype(str).isin(last_ids)].copy()
            if recent.empty:
                continue

            minute_totals = recent.groupby("player_id")["minutes"].sum(min_count=1).sort_values(ascending=False)
            total_minutes = float(minute_totals.sum()) if len(minute_totals) else 0.0
            top8 = minute_totals.head(8)
            top8_share = float(top8.sum() / total_minutes) if total_minutes > 0 else np.nan

            latest_game_id = prior_games.iloc[-1]["game_id"]
            latest = prior[prior["game_id"] == latest_game_id]
            latest_players = set(latest.loc[latest["minutes"].fillna(0) > 0, "player_id"].astype(str))
            prior_before_latest = prior[prior["game_id"] != latest_game_id]
            prior_rotation = (
                prior_before_latest.groupby("player_id")["minutes"].sum(min_count=1).sort_values(ascending=False).head(8)
            )
            expected_core = set(prior_rotation.index.astype(str))
            overlap = len(latest_players & expected_core) / len(expected_core) if expected_core else np.nan

            recent_starters = recent[recent["starter"]].groupby("player_id").size().sort_values(ascending=False)
            established_starters = set(recent_starters.head(5).index.astype(str))
            latest_starters = set(latest.loc[latest["starter"], "player_id"].astype(str))
            starter_continuity = (
                len(latest_starters & established_starters) / len(established_starters)
                if established_starters
                else np.nan
            )

            out[f"{side}_rotation_top8_minutes_share_last5"] = top8_share
            out[f"{side}_rotation_core_overlap_last_game"] = overlap
            out[f"{side}_starter_continuity_last5"] = starter_continuity
            out[f"{side}_players_used_last_game"] = float(len(latest_players))

        rows.append(out)

    result = pd.DataFrame(rows)
    for metric in (
        "rotation_top8_minutes_share_last5",
        "rotation_core_overlap_last_game",
        "starter_continuity_last5",
        "players_used_last_game",
    ):
        h = f"home_{metric}"
        a = f"away_{metric}"
        if h in result.columns and a in result.columns:
            result[f"diff_{metric}"] = result[h] - result[a]
    return result


def main() -> None:
    games = pd.read_parquet(GAMES_PATH).copy()
    if games.empty:
        raise SystemExit("No games available")

    derived_prefixes = (
        "home_travel_",
        "away_travel_",
        "diff_travel_",
        "home_road_trip_",
        "away_road_trip_",
        "diff_road_trip_",
        "home_home_after_road_trip",
        "away_home_after_road_trip",
        "home_possessions_",
        "away_possessions_",
        "diff_possessions_",
        "home_offensive_rating_",
        "away_offensive_rating_",
        "diff_offensive_rating_",
        "home_defensive_rating_",
        "away_defensive_rating_",
        "diff_defensive_rating_",
        "home_effective_fg_pct_",
        "away_effective_fg_pct_",
        "diff_effective_fg_pct_",
        "home_turnover_rate_",
        "away_turnover_rate_",
        "diff_turnover_rate_",
        "home_offensive_rebound_rate_",
        "away_offensive_rebound_rate_",
        "diff_offensive_rebound_rate_",
        "home_free_throw_rate_",
        "away_free_throw_rate_",
        "diff_free_throw_rate_",
        "home_three_point_attempt_rate_",
        "away_three_point_attempt_rate_",
        "diff_three_point_attempt_rate_",
        "home_rotation_",
        "away_rotation_",
        "diff_rotation_",
        "home_starter_continuity_",
        "away_starter_continuity_",
        "diff_starter_continuity_",
        "home_players_used_",
        "away_players_used_",
        "diff_players_used_",
    )
    stale = [c for c in games.columns if c.startswith(derived_prefixes)]
    if stale:
        games = games.drop(columns=stale)

    travel = build_travel_features(games)
    if not travel.empty:
        games = games.merge(travel, on="game_id", how="left", validate="one_to_one")

    if TEAM_BOX_STATS_PATH.exists():
        team_stats = pd.read_parquet(TEAM_BOX_STATS_PATH)
        possession = build_possession_features(games, team_stats)
        if not possession.empty:
            games = games.merge(possession, on="game_id", how="left", validate="one_to_one")

    if PLAYER_BOX_STATS_PATH.exists():
        player_stats = pd.read_parquet(PLAYER_BOX_STATS_PATH)
        rotation = _rotation_features(games, player_stats)
        if not rotation.empty:
            games = games.merge(rotation, on="game_id", how="left", validate="one_to_one")

    write_parquet_atomic(games, GAMES_PATH)

    added = [c for c in games.columns if c.startswith(derived_prefixes)]
    report = {
        "games": int(len(games)),
        "advanced_feature_columns": len(added),
        "feature_columns": added,
        "team_box_stats_available": TEAM_BOX_STATS_PATH.exists(),
        "player_box_stats_available": PLAYER_BOX_STATS_PATH.exists(),
    }
    Path("models/advanced_feature_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
