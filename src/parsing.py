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


def parse_quarter_scores(
    summary: dict[str, Any],
    game_id: str,
) -> list[dict[str, Any]]:
    """Extract WNBA quarter scores from several ESPN summary layouts."""

    if not isinstance(summary, dict):
        return []

    game_id = str(game_id)

    competitors = []

    # -----------------------------------------------------
    # Layout 1:
    # header -> competitions -> competitors
    # -----------------------------------------------------
    try:
        competitors = (
            summary
            .get("header", {})
            .get("competitions", [])[0]
            .get("competitors", [])
        )
    except (IndexError, AttributeError, TypeError):
        competitors = []

    # -----------------------------------------------------
    # Layout 2:
    # boxscore -> teams
    # -----------------------------------------------------
    if not competitors:
        teams = (
            summary
            .get("boxscore", {})
            .get("teams", [])
        )

        if isinstance(teams, list):
            competitors = teams

    rows = []

    for competitor in competitors:

        if not isinstance(competitor, dict):
            continue

        # Team can be directly on competitor or nested under team
        team_obj = competitor.get("team", {})

        if not isinstance(team_obj, dict):
            team_obj = {}

        team_id = str(
            team_obj.get("id")
            or competitor.get("teamId")
            or competitor.get("id")
            or ""
        )

        team_name = (
            team_obj.get("displayName")
            or team_obj.get("name")
            or competitor.get("displayName")
            or ""
        )

        team_abbr = (
            team_obj.get("abbreviation")
            or competitor.get("abbreviation")
            or ""
        )

        # -------------------------------------------------
        # Try several line-score fields
        # -------------------------------------------------
        linescores = (
            competitor.get("linescores")
            or competitor.get("lineScores")
            or competitor.get("linescore")
            or competitor.get("periods")
            or []
        )

        # Some ESPN responses nest these under statistics
        if not linescores:
            stats = competitor.get("statistics", {})
            if isinstance(stats, dict):
                linescores = (
                    stats.get("linescores")
                    or stats.get("periods")
                    or []
                )

        q_values = {
            "q1": None,
            "q2": None,
            "q3": None,
            "q4": None,
            "ot1": None,
            "ot2": None,
        }

        valid_periods = 0

        if isinstance(linescores, list):

            for idx, item in enumerate(linescores, start=1):

                if not isinstance(item, dict):
                    continue

                raw_value = (
                    item.get("value")
                    if item.get("value") is not None
                    else item.get("score")
                )

                if raw_value is None:
                    raw_value = item.get("points")

                try:
                    value = int(float(raw_value))
                except (TypeError, ValueError):
                    continue

                period = (
                    item.get("period")
                    or item.get("number")
                    or item.get("sequence")
                    or idx
                )

                try:
                    period = int(period)
                except (TypeError, ValueError):
                    period = idx

                if period == 1:
                    q_values["q1"] = value
                elif period == 2:
                    q_values["q2"] = value
                elif period == 3:
                    q_values["q3"] = value
                elif period == 4:
                    q_values["q4"] = value
                elif period == 5:
                    q_values["ot1"] = value
                elif period == 6:
                    q_values["ot2"] = value

                valid_periods += 1

        rows.append(
            {
                "game_id": game_id,
                "team_id": team_id,
                "team": team_name,
                "team_abbr": team_abbr,
                "q1": q_values["q1"],
                "q2": q_values["q2"],
                "q3": q_values["q3"],
                "q4": q_values["q4"],
                "ot1": q_values["ot1"],
                "ot2": q_values["ot2"],
                "total_periods": valid_periods,
                "updated_at_utc": datetime.now(tz=UTC),
            }
        )

    return rows
