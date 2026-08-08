from __future__ import annotations

import zoneinfo
from datetime import datetime, timezone
from typing import Any

UTC = zoneinfo.ZoneInfo("UTC")


def _parse_datetime_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(UTC)
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def parse_scoreboard_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None

    game_id = event.get("id")
    if game_id is None:
        return None
    game_id = str(game_id)

    game_date_utc = _parse_datetime_utc(event.get("date") or event.get("scheduled"))
    if not game_date_utc:
        return None

    season = event.get("season")
    if isinstance(season, dict):
        season = season.get("year")
    season = _safe_int(season)
    if season is None:
        return None

    status_info = event.get("status", {})
    if not isinstance(status_info, dict):
        status_info = {}
    type_info = status_info.get("type", {}) if isinstance(status_info.get("type"), dict) else {}
    status = type_info.get("name") or status_info.get("type") or ""
    status_detail = type_info.get("detail") or type_info.get("shortDetail") or ""
    completed = bool(type_info.get("completed") or status.lower() == "final")
    clock = (
        status_info.get("displayClock")
        or status_info.get("clock")
        or type_info.get("displayClock")
        or type_info.get("clock")
        or ""
    )
    period = (
        _safe_int(status_info.get("period"))
        or _safe_int(type_info.get("period"))
        or _safe_int(status_info.get("periodNumber"))
        or _safe_int(type_info.get("periodNumber"))
    )

    competitions = event.get("competitions")
    if not isinstance(competitions, list) or not competitions:
        return None
    competition = competitions[0]
    if not isinstance(competition, dict):
        return None

    competitors = competition.get("competitors")
    if not isinstance(competitors, list) or len(competitors) < 2:
        return None

    home_competitor = None
    away_competitor = None
    for competitor in competitors:
        if not isinstance(competitor, dict):
            continue
        home_away = competitor.get("homeAway")
        if home_away == "home":
            home_competitor = competitor
        elif home_away == "away":
            away_competitor = competitor

    if home_competitor is None or away_competitor is None:
        return None

    def _team_fields(entry: dict[str, Any]) -> tuple[str, str, str, int | None]:
        team = entry.get("team", {})
        team_id = str(team.get("id")) if team.get("id") is not None else ""
        return (
            team_id,
            team.get("name") or team.get("displayName") or "",
            team.get("abbreviation") or "",
            _safe_int(entry.get("score")),
        )

    home_team_id, home_team, home_abbr, home_score = _team_fields(home_competitor)
    away_team_id, away_team, away_abbr, away_score = _team_fields(away_competitor)

    venue = ""
    neutral_site = False
    if isinstance(competition.get("venue"), dict):
        venue = competition.get("venue", {}).get("fullName") or ""
    neutral_site = bool(competition.get("neutralSite"))

    updated_at_utc = _parse_datetime_utc(event.get("date")) or _parse_datetime_utc(type_info.get("dateModified"))

    return {
        "game_id": game_id,
        "game_date_utc": game_date_utc,
        "season": season,
        "completed": completed,
        "status": str(status),
        "status_detail": str(status_detail),
        "clock": clock,
        "period": period,
        "home_team_id": home_team_id,
        "home_team": home_team,
        "home_abbr": home_abbr,
        "away_team_id": away_team_id,
        "away_team": away_team,
        "away_abbr": away_abbr,
        "home_score": home_score,
        "away_score": away_score,
        "venue": venue,
        "neutral_site": neutral_site,
        "updated_at_utc": updated_at_utc,
    }


def parse_quarter_scores(
    summary: dict[str, Any],
    game_id: str,
) -> list[dict[str, Any]]:
    """
    Extract quarter scores from ESPN WNBA play-by-play.

    ESPN summary responses do not consistently expose quarter
    line scores in the team objects. The play-by-play does expose
    the cumulative homeScore and awayScore after each play, along
    with the period number.

    We therefore:
      1. Find the final cumulative score recorded in each period.
      2. Convert cumulative scores into individual quarter scores.
      3. Map home/away scores back to ESPN team IDs.
    """

    if not isinstance(summary, dict):
        return []

    game_id = str(game_id)
    
    boxscore = summary.get("boxscore", {})
    if not isinstance(boxscore, dict):
        return []

    teams = boxscore.get("teams")
    if not isinstance(teams, list) or len(teams) < 2:
        return []

    # Try to identify home/away teams
    team_info = {}
    for team_data in teams:
        if not isinstance(team_data, dict):
            continue

        team = team_data.get("team", {})
        if not isinstance(team, dict):
            continue

        home_away = team_data.get("homeAway")
        if home_away in ("home", "away"):
            team_info[home_away] = {
                "team_id": str(team.get("id", "")),
                "team": team.get("displayName") or team.get("name") or "",
                "team_abbr": team.get("abbreviation") or "",
            }

    # ---------------------------------------------------------
    # Read ESPN play-by-play for cumulative scores
    # ---------------------------------------------------------

    plays = summary.get("plays", [])
    if not isinstance(plays, list):
        plays = []

    # Stores the FINAL cumulative score observed in each period.
    period_end_scores = {}

    for play in plays:
        if not isinstance(play, dict):
            continue

        period = play.get("period", {})
        if isinstance(period, dict):
            period_number = period.get("number")
        else:
            period_number = period

        try:
            period_number = int(period_number)
        except (TypeError, ValueError):
            continue

        try:
            home_score = int(float(play.get("homeScore")))
            away_score = int(float(play.get("awayScore")))
        except (TypeError, ValueError):
            continue

        # Because ESPN plays are chronological, repeatedly assigning here
        # leaves us with the final cumulative score observed in the period.
        period_end_scores[period_number] = {"home": home_score, "away": away_score}

    # ---------------------------------------------------------
    # Convert cumulative scores to period scores
    # ---------------------------------------------------------

    home_period_scores = {}
    away_period_scores = {}

    # If we have play-by-play data, convert cumulative to individual period scores
    if period_end_scores:
        previous_home = 0
        previous_away = 0

        for period_number in sorted(period_end_scores):
            cumulative = period_end_scores[period_number]
            cumulative_home = cumulative["home"]
            cumulative_away = cumulative["away"]

            home_period_scores[period_number] = cumulative_home - previous_home
            away_period_scores[period_number] = cumulative_away - previous_away

            previous_home = cumulative_home
            previous_away = cumulative_away

    # ---------------------------------------------------------
    # Build output rows
    # ---------------------------------------------------------

    rows = []
    updated_at_utc = _parse_datetime_utc(summary.get("date"))

    def score_for(scores: dict, period_number: int):
        value = scores.get(period_number)
        if value is None:
            return None
        return int(value)

    # If we have identified home/away teams and have period scores, use that approach
    if team_info.get("home") and team_info.get("away") and (home_period_scores or away_period_scores):
        for home_away in ("home", "away"):
            info = team_info[home_away]
            scores = home_period_scores if home_away == "home" else away_period_scores

            rows.append(
                {
                    "game_id": game_id,
                    "team_id": info["team_id"],
                    "team": info["team"],
                    "team_abbr": info["team_abbr"],
                    "q1": score_for(scores, 1),
                    "q2": score_for(scores, 2),
                    "q3": score_for(scores, 3),
                    "q4": score_for(scores, 4),
                    "ot1": score_for(scores, 5),
                    "ot2": score_for(scores, 6),
                    "total_periods": len(scores),
                    "updated_at_utc": updated_at_utc,
                }
            )
    else:
        # Fallback: iterate through teams directly and extract linescores
        for team_data in teams:
            if not isinstance(team_data, dict):
                continue

            team = team_data.get("team", {})
            if not isinstance(team, dict):
                continue

            team_id = str(team.get("id")) if team.get("id") is not None else ""
            team_name = team.get("name") or team.get("displayName") or ""
            team_abbr = team.get("abbreviation") or ""

            scores: dict[str, int | None] = {f"q{i}": None for i in range(1, 5)}
            scores.update({f"ot{i}": None for i in range(1, 3)})

            linescores = team_data.get("linescores")
            if isinstance(linescores, list) and linescores:
                for line in linescores:
                    if not isinstance(line, dict):
                        continue
                    period_number = _safe_int(line.get("period")) or _safe_int(line.get("number"))
                    score_value = _safe_int(line.get("score"))
                    if period_number is None or score_value is None:
                        continue
                    if period_number <= 4:
                        scores[f"q{period_number}"] = score_value
                    else:
                        overtime_index = period_number - 4
                        if 1 <= overtime_index <= 2:
                            scores[f"ot{overtime_index}"] = score_value

            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "team": team_name,
                    "team_abbr": team_abbr,
                    "q1": scores["q1"],
                    "q2": scores["q2"],
                    "q3": scores["q3"],
                    "q4": scores["q4"],
                    "ot1": scores["ot1"],
                    "ot2": scores["ot2"],
                    "total_periods": sum(1 for v in scores.values() if v is not None),
                    "updated_at_utc": updated_at_utc,
                }
            )

    return rows
