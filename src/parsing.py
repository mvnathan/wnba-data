from __future__ import annotations

import zoneinfo
from datetime import datetime
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


def parse_quarter_scores(summary: dict[str, Any], game_id: str) -> list[dict[str, Any]]:
    if not isinstance(summary, dict) or not isinstance(game_id, str):
        raise TypeError("Invalid summary or game_id")

    boxscore = summary.get("boxscore", {})
    if not isinstance(boxscore, dict):
        return []

    teams = boxscore.get("teams")
    if not isinstance(teams, list) or len(teams) < 2:
        return []

    period_map: dict[int, dict[str, Any]] = {}
    if isinstance(boxscore.get("periods"), list):
        for period in boxscore.get("periods", []):
            if not isinstance(period, dict):
                continue
            period_number = _safe_int(period.get("number")) or _safe_int(period.get("period"))
            if period_number is None:
                continue
            period_map[period_number] = period

    rows: list[dict[str, Any]] = []
    updated_at_utc = _parse_datetime_utc(summary.get("date"))
    for team_data in teams:
        if not isinstance(team_data, dict):
            continue
        team = team_data.get("team", {})
        team_id = str(team.get("id")) if team.get("id") is not None else ""
        team_name = team.get("name") or team.get("displayName") or ""
        team_abbr = team.get("abbreviation") or ""

        scores: dict[str, int | None] = {f"q{i}": None for i in range(1, 5)}
        scores.update({f"ot{i}": None for i in range(1, 3)})
        total_periods = 0

        linescores = team_data.get("linescores")
        if isinstance(linescores, list) and linescores:
            for line in linescores:
                period_number = _safe_int(line.get("period")) or _safe_int(line.get("number"))
                if period_number is None:
                    continue
                score_value = _safe_int(line.get("score"))
                if score_value is None:
                    continue
                if period_number <= 4:
                    scores[f"q{period_number}"] = score_value
                else:
                    overtime_index = period_number - 4
                    if 1 <= overtime_index <= 2:
                        scores[f"ot{overtime_index}"] = score_value
                total_periods += 1
        elif period_map:
            home_away = team_data.get("homeAway")
            for period_number, period in period_map.items():
                if home_away == "home":
                    score_value = _safe_int(period.get("home", {}).get("score")) if isinstance(period.get("home"), dict) else _safe_int(period.get("home"))
                else:
                    score_value = _safe_int(period.get("away", {}).get("score")) if isinstance(period.get("away"), dict) else _safe_int(period.get("away"))
                if score_value is None:
                    continue
                if period_number <= 4:
                    scores[f"q{period_number}"] = score_value
                else:
                    overtime_index = period_number - 4
                    if 1 <= overtime_index <= 2:
                        scores[f"ot{overtime_index}"] = score_value
                total_periods += 1

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
                "total_periods": total_periods,
                "updated_at_utc": updated_at_utc,
            }
        )
    return rows
