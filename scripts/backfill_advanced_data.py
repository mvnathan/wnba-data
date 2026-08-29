from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.advanced_features import (
    PLAYER_BOX_STATS_PATH,
    TEAM_BOX_STATS_PATH,
    parse_player_box_stats,
    parse_team_box_stats,
)
from src.config import GAMES_PATH
from src.espn_api import ESPNApiClient
from src.storage import load_parquet_or_empty, upsert_dataframe, write_parquet_atomic


def main() -> None:
    games = pd.read_parquet(GAMES_PATH)
    completed = games[games["completed"].fillna(False).astype(bool)].copy()
    team_existing = load_parquet_or_empty(TEAM_BOX_STATS_PATH)
    player_existing = load_parquet_or_empty(PLAYER_BOX_STATS_PATH)

    team_complete_ids = set(team_existing.get("game_id", pd.Series(dtype="string")).astype(str)) if not team_existing.empty else set()
    player_complete_ids = set(player_existing.get("game_id", pd.Series(dtype="string")).astype(str)) if not player_existing.empty else set()
    needed_ids = [
        str(game_id)
        for game_id in completed["game_id"].astype(str).tolist()
        if str(game_id) not in team_complete_ids or str(game_id) not in player_complete_ids
    ]

    client = ESPNApiClient()
    team_rows: list[dict] = []
    player_rows: list[dict] = []
    errors: list[str] = []

    for game_id in needed_ids:
        try:
            summary = client.fetch_summary(game_id)
            team_rows.extend(parse_team_box_stats(summary, game_id))
            player_rows.extend(parse_player_box_stats(summary, game_id))
        except Exception as exc:
            errors.append(f"{game_id}: {exc}")

    if team_rows:
        team_existing = upsert_dataframe(
            team_existing,
            pd.DataFrame(team_rows),
            keys=["game_id", "team_id"],
            sort_columns=["game_id", "team_id"],
        )
        write_parquet_atomic(team_existing, TEAM_BOX_STATS_PATH)

    if player_rows:
        player_df = pd.DataFrame(player_rows)
        player_df = player_df[player_df["player_id"].astype(str) != ""]
        player_existing = upsert_dataframe(
            player_existing,
            player_df,
            keys=["game_id", "team_id", "player_id"],
            sort_columns=["game_id", "team_id", "player_id"],
        )
        write_parquet_atomic(player_existing, PLAYER_BOX_STATS_PATH)

    report = {
        "completed_games": int(len(completed)),
        "games_requested": len(needed_ids),
        "team_stat_rows": int(len(team_existing)) if not team_existing.empty else 0,
        "player_stat_rows": int(len(player_existing)) if not player_existing.empty else 0,
        "errors": errors,
    }
    Path("models/advanced_data_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
