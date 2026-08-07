from __future__ import annotations

import pandas as pd


def build_team_games(games: pd.DataFrame, quarters: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date_utc",
                "season",
                "team_id",
                "team",
                "team_abbr",
                "opp_team_id",
                "opp_team",
                "opp_abbr",
                "is_home",
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
                "updated_at_utc",
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
        )

    completed_games = games[games["completed"]].copy()
    if completed_games.empty:
        return pd.DataFrame(columns=[])

    home_rows = completed_games.assign(
        team_id=completed_games["home_team_id"],
        team=completed_games["home_team"],
        team_abbr=completed_games["home_abbr"],
        opp_team_id=completed_games["away_team_id"],
        opp_team=completed_games["away_team"],
        opp_abbr=completed_games["away_abbr"],
        is_home=True,
        points=completed_games["home_score"],
        opp_points=completed_games["away_score"],
    )
    away_rows = completed_games.assign(
        team_id=completed_games["away_team_id"],
        team=completed_games["away_team"],
        team_abbr=completed_games["away_abbr"],
        opp_team_id=completed_games["home_team_id"],
        opp_team=completed_games["home_team"],
        opp_abbr=completed_games["home_abbr"],
        is_home=False,
        points=completed_games["away_score"],
        opp_points=completed_games["home_score"],
    )
    team_games = pd.concat([home_rows, away_rows], ignore_index=True)

    quarter_columns = ["q1", "q2", "q3", "q4", "ot1", "ot2", "updated_at_utc"]
    if not quarters.empty:
        quarters = quarters.copy()
        opp_quarters = quarters.rename(
            columns={
                "team_id": "opp_team_id",
                "team": "opp_team",
                "team_abbr": "opp_abbr",
                "q1": "opp_q1",
                "q2": "opp_q2",
                "q3": "opp_q3",
                "q4": "opp_q4",
                "ot1": "opp_ot1",
                "ot2": "opp_ot2",
                "total_periods": "opp_total_periods",
                "updated_at_utc": "opp_updated_at_utc",
            }
        )
        team_games = team_games.merge(
            quarters,
            on=["game_id", "team_id"],
            how="left",
            suffixes=("", "_src"),
        )
        team_games = team_games.merge(
            opp_quarters[
                ["game_id", "opp_team_id", "opp_q1", "opp_q2", "opp_q3", "opp_q4"]
            ],
            on=["game_id", "opp_team_id"],
            how="left",
        )
        for quarter in ["q1", "q2", "q3", "q4"]:
            team_games[f"{quarter}_margin"] = (
                team_games[quarter].fillna(0) - team_games[f"opp_{quarter}"].fillna(0)
            )
        team_games["first_half_points"] = team_games[["q1", "q2"]].sum(axis=1, min_count=1)
        team_games["second_half_points"] = team_games[["q3", "q4"]].sum(axis=1, min_count=1)
        team_games["regulation_points"] = team_games[["q1", "q2", "q3", "q4"]].sum(axis=1, min_count=1)
        team_games["first_half_margin"] = team_games["first_half_points"] - team_games["opp_q1"].fillna(0) - team_games["opp_q2"].fillna(0)
    else:
        team_games["q1"] = None
        team_games["q2"] = None
        team_games["q3"] = None
        team_games["q4"] = None
        team_games["ot1"] = None
        team_games["ot2"] = None
        team_games["opp_q1"] = None
        team_games["opp_q2"] = None
        team_games["opp_q3"] = None
        team_games["opp_q4"] = None
        team_games["q1_margin"] = None
        team_games["q2_margin"] = None
        team_games["q3_margin"] = None
        team_games["q4_margin"] = None
        team_games["first_half_points"] = None
        team_games["second_half_points"] = None
        team_games["regulation_points"] = None
        team_games["first_half_margin"] = None

    team_games["margin"] = team_games["points"] - team_games["opp_points"]
    team_games["win"] = team_games["margin"] > 0
    team_games["updated_at_utc"] = team_games["updated_at_utc"].fillna(team_games["game_date_utc"])

    team_games = team_games[
        [
            "game_id",
            "game_date_utc",
            "season",
            "team_id",
            "team",
            "team_abbr",
            "opp_team_id",
            "opp_team",
            "opp_abbr",
            "is_home",
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
            "updated_at_utc",
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
    ]

    team_games = team_games.sort_values(["game_date_utc", "game_id", "is_home"], ignore_index=True)
    return team_games
