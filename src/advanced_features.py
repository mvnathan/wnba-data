from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TEAM_BOX_STATS_PATH = Path("data/team_box_stats.parquet")
PLAYER_BOX_STATS_PATH = Path("data/player_box_stats.parquet")
PLAYER_AVAILABILITY_PATH = Path("data/player_availability.parquet")

# Approximate WNBA home-arena coordinates. These are used only to derive
# schedule/travel context; they are not outcomes and therefore introduce no
# target leakage. Neutral-site games are handled conservatively.
TEAM_COORDS: dict[str, tuple[float, float]] = {
    "ATL": (33.7573, -84.3963),
    "CHI": (41.8807, -87.6742),
    "CON": (41.4912, -72.0908),
    "DAL": (32.7905, -96.8103),
    "GS": (37.7680, -122.3877),
    "IND": (39.7639, -86.1555),
    "LA": (34.0430, -118.2673),
    "LV": (36.1028, -115.1782),
    "MIN": (44.9795, -93.2760),
    "NY": (40.6826, -73.9754),
    "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668),
    "SEA": (47.6221, -122.3540),
    "TOR": (43.6435, -79.3791),
    "WSH": (38.8469, -76.9910),
}


def _number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text or text in {"--", "-", "None", "nan"}:
        return None
    try:
        value_float = float(text)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def _split_made_attempted(value: Any) -> tuple[float | None, float | None]:
    text = str(value or "").strip()
    for separator in ("-", "/"):
        if separator in text:
            left, right = text.split(separator, 1)
            return _number(left), _number(right)
    return None, None


def _normalized_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _stat_map(statistics: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(statistics, list):
        return result
    for item in statistics:
        if not isinstance(item, dict):
            continue
        value = item.get("displayValue", item.get("value"))
        for key in (item.get("name"), item.get("label"), item.get("abbreviation")):
            normalized = _normalized_key(key)
            if normalized:
                result[normalized] = value
    return result


def _first(stats: dict[str, Any], *aliases: str) -> Any:
    for alias in aliases:
        key = _normalized_key(alias)
        if key in stats:
            return stats[key]
    return None


def parse_team_box_stats(summary: dict[str, Any], game_id: str) -> list[dict[str, Any]]:
    """Extract team box-score inputs and possession-based efficiency metrics."""
    boxscore = summary.get("boxscore", {}) if isinstance(summary, dict) else {}
    teams = boxscore.get("teams", []) if isinstance(boxscore, dict) else []
    if not isinstance(teams, list):
        return []

    rows: list[dict[str, Any]] = []
    for team_block in teams:
        if not isinstance(team_block, dict):
            continue
        team = team_block.get("team", {}) if isinstance(team_block.get("team"), dict) else {}
        team_id = str(team.get("id") or "")
        if not team_id:
            continue
        stats = _stat_map(team_block.get("statistics"))

        fgm, fga = _split_made_attempted(
            _first(stats, "fieldGoalsMade-fieldGoalsAttempted", "fieldGoals")
        )
        if fgm is None:
            fgm = _number(_first(stats, "fieldGoalsMade"))
        if fga is None:
            fga = _number(_first(stats, "fieldGoalsAttempted"))

        tpm, tpa = _split_made_attempted(
            _first(stats, "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "threePointFieldGoals")
        )
        if tpm is None:
            tpm = _number(_first(stats, "threePointFieldGoalsMade", "threePointersMade"))
        if tpa is None:
            tpa = _number(_first(stats, "threePointFieldGoalsAttempted", "threePointersAttempted"))

        ftm, fta = _split_made_attempted(
            _first(stats, "freeThrowsMade-freeThrowsAttempted", "freeThrows")
        )
        if ftm is None:
            ftm = _number(_first(stats, "freeThrowsMade"))
        if fta is None:
            fta = _number(_first(stats, "freeThrowsAttempted"))

        orb = _number(_first(stats, "offensiveRebounds", "offRebounds"))
        drb = _number(_first(stats, "defensiveRebounds", "defRebounds"))
        trb = _number(_first(stats, "totalRebounds", "rebounds"))
        tov = _number(_first(stats, "turnovers", "totalTurnovers"))
        ast = _number(_first(stats, "assists"))
        stl = _number(_first(stats, "steals"))
        blk = _number(_first(stats, "blocks"))
        fouls = _number(_first(stats, "fouls", "totalFouls"))
        points = _number(_first(stats, "points"))

        possessions = None
        if fga is not None and tov is not None and fta is not None:
            possessions = fga - (orb or 0.0) + tov + 0.44 * fta

        efg = None
        if fga and fgm is not None and tpm is not None:
            efg = (fgm + 0.5 * tpm) / fga

        tov_rate = None
        if possessions and tov is not None:
            tov_rate = tov / possessions

        ft_rate = None
        if fga and fta is not None:
            ft_rate = fta / fga

        three_rate = None
        if fga and tpa is not None:
            three_rate = tpa / fga

        offensive_rating = None
        if possessions and points is not None:
            offensive_rating = 100.0 * points / possessions

        rows.append(
            {
                "game_id": str(game_id),
                "team_id": team_id,
                "team": team.get("displayName") or team.get("name"),
                "team_abbr": team.get("abbreviation"),
                "field_goals_made": fgm,
                "field_goals_attempted": fga,
                "three_points_made": tpm,
                "three_points_attempted": tpa,
                "free_throws_made": ftm,
                "free_throws_attempted": fta,
                "offensive_rebounds": orb,
                "defensive_rebounds": drb,
                "total_rebounds": trb,
                "turnovers": tov,
                "assists": ast,
                "steals": stl,
                "blocks": blk,
                "fouls": fouls,
                "points_box": points,
                "possessions": possessions,
                "offensive_rating": offensive_rating,
                "effective_fg_pct": efg,
                "turnover_rate": tov_rate,
                "free_throw_rate": ft_rate,
                "three_point_attempt_rate": three_rate,
            }
        )

    # Offensive rebound rate requires the opponent's defensive rebounds.
    by_team = {row["team_id"]: row for row in rows}
    if len(rows) == 2:
        a, b = rows
        for team_row, opp_row in ((a, b), (b, a)):
            orb = _number(team_row.get("offensive_rebounds"))
            opp_drb = _number(opp_row.get("defensive_rebounds"))
            denominator = (orb or 0.0) + (opp_drb or 0.0)
            team_row["offensive_rebound_rate"] = (
                orb / denominator if orb is not None and denominator > 0 else None
            )
            team_row["defensive_rating"] = opp_row.get("offensive_rating")
    else:
        for row in rows:
            row["offensive_rebound_rate"] = None
            row["defensive_rating"] = None
    return list(by_team.values())


def parse_player_box_stats(summary: dict[str, Any], game_id: str) -> list[dict[str, Any]]:
    """Extract player minutes/role data for future rotation-continuity features."""
    boxscore = summary.get("boxscore", {}) if isinstance(summary, dict) else {}
    player_groups = boxscore.get("players", []) if isinstance(boxscore, dict) else []
    if not isinstance(player_groups, list):
        return []

    rows: list[dict[str, Any]] = []
    for team_group in player_groups:
        if not isinstance(team_group, dict):
            continue
        team = team_group.get("team", {}) if isinstance(team_group.get("team"), dict) else {}
        team_id = str(team.get("id") or "")
        for category in team_group.get("statistics", []) or []:
            if not isinstance(category, dict):
                continue
            names = category.get("names") or category.get("labels") or []
            names = [_normalized_key(name) for name in names]
            for athlete_row in category.get("athletes", []) or []:
                if not isinstance(athlete_row, dict):
                    continue
                athlete = athlete_row.get("athlete", {}) if isinstance(athlete_row.get("athlete"), dict) else {}
                stats = athlete_row.get("stats") or []
                mapping = {names[i]: stats[i] for i in range(min(len(names), len(stats)))}
                minutes = _number(mapping.get("min") or mapping.get("minutes"))
                points = _number(mapping.get("pts") or mapping.get("points"))
                rows.append(
                    {
                        "game_id": str(game_id),
                        "team_id": team_id,
                        "team_abbr": team.get("abbreviation"),
                        "player_id": str(athlete.get("id") or ""),
                        "player": athlete.get("displayName") or athlete.get("fullName"),
                        "starter": bool(athlete_row.get("starter")),
                        "did_not_play": bool(athlete_row.get("didNotPlay")),
                        "minutes": minutes,
                        "points": points,
                    }
                )
    return rows


def _haversine_miles(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3958.7613 * 2 * math.asin(math.sqrt(h))


def build_travel_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build pregame travel and road-trip context from the schedule only."""
    if games.empty:
        return pd.DataFrame(columns=["game_id"])
    games = games.copy()
    games["game_date_utc"] = pd.to_datetime(games["game_date_utc"], utc=True, errors="coerce")
    games["game_id"] = games["game_id"].astype("string")

    team_rows: list[dict[str, Any]] = []
    for _, game in games.sort_values(["game_date_utc", "game_id"]).iterrows():
        for side in ("home", "away"):
            is_home = side == "home"
            team_abbr = str(game.get(f"{side}_abbr") or "")
            opp_side = "away" if is_home else "home"
            opp_abbr = str(game.get(f"{opp_side}_abbr") or "")
            location_abbr = team_abbr if is_home else opp_abbr
            team_rows.append(
                {
                    "game_id": str(game.get("game_id")),
                    "game_date_utc": game.get("game_date_utc"),
                    "team_abbr": team_abbr,
                    "is_home": is_home,
                    "location_abbr": location_abbr,
                }
            )

    frame = pd.DataFrame(team_rows).sort_values(["team_abbr", "game_date_utc", "game_id"])
    enriched: list[pd.DataFrame] = []
    for _, group in frame.groupby("team_abbr", sort=False):
        group = group.copy().reset_index(drop=True)
        coords = [TEAM_COORDS.get(str(abbr)) for abbr in group["location_abbr"]]
        travel: list[float] = []
        prior_coord: tuple[float, float] | None = None
        road_streak = 0
        prior_was_away = False
        road_numbers: list[int] = []
        home_after_trip: list[int] = []
        for idx, row in group.iterrows():
            current = coords[idx]
            miles = _haversine_miles(prior_coord, current) if prior_coord is not None else 0.0
            travel.append(float(miles or 0.0))
            is_home = bool(row["is_home"])
            home_after_trip.append(int(is_home and prior_was_away))
            if is_home:
                road_streak = 0
                road_numbers.append(0)
            else:
                road_streak += 1
                road_numbers.append(road_streak)
            prior_was_away = not is_home
            prior_coord = current or prior_coord
        group["travel_miles_since_last_game"] = travel
        group["road_trip_game_number"] = road_numbers
        group["home_after_road_trip"] = home_after_trip
        enriched.append(group)

    team = pd.concat(enriched, ignore_index=True)
    home = team[team["is_home"]].copy().rename(
        columns={
            "travel_miles_since_last_game": "home_travel_miles_since_last_game",
            "road_trip_game_number": "home_road_trip_game_number",
            "home_after_road_trip": "home_home_after_road_trip",
        }
    )
    away = team[~team["is_home"]].copy().rename(
        columns={
            "travel_miles_since_last_game": "away_travel_miles_since_last_game",
            "road_trip_game_number": "away_road_trip_game_number",
            "home_after_road_trip": "away_home_after_road_trip",
        }
    )
    result = home[["game_id", "home_travel_miles_since_last_game", "home_road_trip_game_number", "home_home_after_road_trip"]].merge(
        away[["game_id", "away_travel_miles_since_last_game", "away_road_trip_game_number", "away_home_after_road_trip"]],
        on="game_id",
        how="outer",
    )
    result["diff_travel_miles_since_last_game"] = (
        result["home_travel_miles_since_last_game"] - result["away_travel_miles_since_last_game"]
    )
    result["diff_road_trip_game_number"] = (
        result["home_road_trip_game_number"] - result["away_road_trip_game_number"]
    )
    return result


def build_possession_features(games: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe pregame rolling possession/efficiency features."""
    if games.empty or team_stats.empty:
        return pd.DataFrame(columns=["game_id"])

    stats = team_stats.copy()
    stats["game_id"] = stats["game_id"].astype("string")
    stats["team_id"] = stats["team_id"].astype("string")
    schedule = games[["game_id", "game_date_utc", "home_team_id", "away_team_id"]].copy()
    schedule["game_id"] = schedule["game_id"].astype("string")
    schedule["game_date_utc"] = pd.to_datetime(schedule["game_date_utc"], utc=True, errors="coerce")
    stats = stats.merge(schedule[["game_id", "game_date_utc"]], on="game_id", how="left")

    metrics = [
        "possessions",
        "offensive_rating",
        "defensive_rating",
        "effective_fg_pct",
        "turnover_rate",
        "offensive_rebound_rate",
        "free_throw_rate",
        "three_point_attempt_rate",
    ]
    for metric in metrics:
        if metric not in stats.columns:
            stats[metric] = np.nan
        stats[metric] = pd.to_numeric(stats[metric], errors="coerce")

    rows: list[pd.DataFrame] = []
    for _, group in stats.sort_values(["team_id", "game_date_utc", "game_id"]).groupby("team_id", sort=False):
        group = group.copy()
        for metric in metrics:
            prior = group[metric].shift(1)
            group[f"{metric}_season_avg"] = prior.expanding(min_periods=1).mean()
            group[f"{metric}_rolling_5"] = prior.rolling(5, min_periods=1).mean()
            group[f"{metric}_rolling_10"] = prior.rolling(10, min_periods=1).mean()
        rows.append(group)
    history = pd.concat(rows, ignore_index=True)

    # Build scheduled rows by taking each team's latest pregame-safe state before tip.
    game_rows: list[dict[str, Any]] = []
    for _, game in games.sort_values(["game_date_utc", "game_id"]).iterrows():
        row: dict[str, Any] = {"game_id": str(game.get("game_id"))}
        game_time = pd.to_datetime(game.get("game_date_utc"), utc=True, errors="coerce")
        for side in ("home", "away"):
            team_id = str(game.get(f"{side}_team_id") or "")
            prior = history[(history["team_id"] == team_id) & (history["game_date_utc"] < game_time)]
            if prior.empty:
                continue
            latest = prior.sort_values(["game_date_utc", "game_id"]).iloc[-1]
            # The latest row's post-shift rolling state excludes that row itself. Add
            # its actual game to produce the state available before the target game.
            team_games = stats[(stats["team_id"] == team_id) & (stats["game_date_utc"] < game_time)].sort_values(["game_date_utc", "game_id"])
            for metric in metrics:
                values = pd.to_numeric(team_games[metric], errors="coerce").dropna()
                row[f"{side}_{metric}_season_avg"] = float(values.mean()) if len(values) else np.nan
                row[f"{side}_{metric}_rolling_5"] = float(values.tail(5).mean()) if len(values) else np.nan
                row[f"{side}_{metric}_rolling_10"] = float(values.tail(10).mean()) if len(values) else np.nan
        game_rows.append(row)

    result = pd.DataFrame(game_rows)
    for metric in metrics:
        for window in ("season_avg", "rolling_5", "rolling_10"):
            home_col = f"home_{metric}_{window}"
            away_col = f"away_{metric}_{window}"
            if home_col in result.columns and away_col in result.columns:
                result[f"diff_{metric}_{window}"] = result[home_col] - result[away_col]
    return result


def augment_model_features(features: pd.DataFrame, games: pd.DataFrame, team_stats_path: Path = TEAM_BOX_STATS_PATH) -> pd.DataFrame:
    """Merge travel and, when available, possession features into model rows."""
    if features.empty:
        return features
    result = features.copy()
    travel = build_travel_features(games)
    if not travel.empty:
        result = result.merge(travel, on="game_id", how="left", validate="one_to_one")
    if team_stats_path.exists():
        team_stats = pd.read_parquet(team_stats_path)
        possession = build_possession_features(games, team_stats)
        if not possession.empty:
            result = result.merge(possession, on="game_id", how="left", validate="one_to_one")
    return result
