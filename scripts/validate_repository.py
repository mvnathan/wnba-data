import json
import sys

import pandas as pd

from src.config import (
    GAMES_PATH,
    QUARTER_SCORES_PATH,
    TEAM_GAMES_PATH,
    LAST_UPDATE_PATH,
)


def validate() -> dict[str, object]:
    report: dict[str, object] = {}
    report["games_exists"] = GAMES_PATH.exists()
    report["quarter_scores_exists"] = QUARTER_SCORES_PATH.exists()
    report["team_games_exists"] = TEAM_GAMES_PATH.exists()
    report["metadata_exists"] = LAST_UPDATE_PATH.exists()

    if not all([report["games_exists"], report["quarter_scores_exists"], report["team_games_exists"], report["metadata_exists"]]):
        raise FileNotFoundError("One or more required repository files are missing")

    games = pd.read_parquet(GAMES_PATH)
    quarters = pd.read_parquet(QUARTER_SCORES_PATH)
    team_games = pd.read_parquet(TEAM_GAMES_PATH)

    if not all([report["games_exists"], report["quarter_scores_exists"], report["team_games_exists"], report["metadata_exists"]]):
        raise FileNotFoundError("One or more required repository files are missing")

    required_games = [
        "game_id",
        "game_date_utc",
        "season",
        "home_team_id",
        "away_team_id",
        "completed",
    ]
    if not set(required_games).issubset(games.columns):
        raise ValueError("games.parquet missing required columns")

    if games["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id values found in games.parquet")

    if quarters[["game_id", "team_id"]].duplicated().any():
        raise ValueError("Duplicate game_id / team_id combinations found in quarter_scores.parquet")

    completed_games = games[games["completed"]]
    if not completed_games.empty:
        completed_ids = set(completed_games["game_id"])
        team_rows = team_games[team_games["game_id"].isin(completed_ids)]
        duplicates = team_rows[team_rows.duplicated(["game_id", "team_id"], keep=False)]
        if duplicates.empty is False and len(duplicates) != 2 * len(completed_ids):
            raise ValueError("Completed games do not have two team-game rows each")

    if (games["home_score"].fillna(0) < 0).any() or (games["away_score"].fillna(0) < 0).any():
        raise ValueError("Negative scores found in games.parquet")

    if (quarters[["q1", "q2", "q3", "q4"]].fillna(0) < 0).any().any():
        raise ValueError("Negative period scores found in quarter_scores.parquet")

    seasons = sorted(games["season"].dropna().unique().astype(int))
    if len(seasons) > 3:
        raise ValueError("More than three seasons are present in games.parquet")

    completed_ids = set(completed_games["game_id"])
    if completed_ids:
        team_rows = team_games[team_games["game_id"].isin(completed_ids)]
        game_counts = team_rows.groupby("game_id").size()
        if not (game_counts == 2).all():
            raise ValueError("Each completed game must have exactly two team-game rows")

    metadata = json.loads(LAST_UPDATE_PATH.read_text(encoding="utf-8"))
    if metadata.get("games_rows") != len(games):
        raise ValueError("Metadata games_rows does not match actual games row count")
    if metadata.get("quarter_rows") != len(quarters):
        raise ValueError("Metadata quarter_rows does not match actual quarter row count")
    if metadata.get("team_game_rows") != len(team_games):
        raise ValueError("Metadata team_game_rows does not match actual team_games row count")

    return {
        "status": "ok",
        "games_rows": len(games),
        "quarter_rows": len(quarters),
        "team_game_rows": len(team_games),
        "seasons": seasons,
    }


if __name__ == "__main__":
    try:
        result = validate()
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as exc:
        print("Validation failed:", exc)
        sys.exit(1)
