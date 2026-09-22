#!/usr/bin/env python3
"""SportsModelHub MLB v2 candidate model.

Adds baseball-specific context to the v1 team run baseline:
- probable starting-pitcher recent ERA/WHIP/IP
- bullpen workload over the previous three days
- venue run environment estimated from recent completed games

The market remains display-only and is never an input.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from scripts.generate_mlb_predictions import (
    CHICAGO, MLB_SCHEDULE, _completed, _date_of_game, _get_json, _league_and_team_rates,
    _schedule, _score, _team_id, _team_name, _win_probability, _dk_market_lookup, _extract_dk,
)

PEOPLE_STATS = "https://statsapi.mlb.com/api/v1/people/{person_id}/stats"
OUT = Path("predictions/mlb-v2-latest.json")
DOCS_OUT = Path("docs/mlb-v2-latest.json")


def _pitcher_id(game: dict[str, Any], side: str) -> int | None:
    p = game.get("teams", {}).get(side, {}).get("probablePitcher") or {}
    try:
        return int(p.get("id"))
    except (TypeError, ValueError):
        return None


def _pitcher_name(game: dict[str, Any], side: str) -> str | None:
    p = game.get("teams", {}).get(side, {}).get("probablePitcher") or {}
    return p.get("fullName") or p.get("name")


def _pitcher_recent(person_id: int | None) -> dict[str, float | None]:
    if not person_id:
        return {"era": None, "whip": None, "ip": None}
    try:
        data = _get_json(
            PEOPLE_STATS.format(person_id=person_id),
            {"stats": "season", "group": "pitching"},
        )
        splits = ((data.get("stats") or [{}])[0].get("splits") or [])
        stat = (splits[0].get("stat") if splits else {}) or {}
        def num(k):
            try: return float(stat.get(k))
            except (TypeError, ValueError): return None
        return {"era": num("era"), "whip": num("whip"), "ip": num("inningsPitched")}
    except Exception:
        return {"era": None, "whip": None, "ip": None}


def _bullpen_workload(history: list[dict[str, Any]], target: date) -> dict[int, float]:
    # Proxy available from schedule data: team games played in the prior 3 days,
    # weighted more heavily for extra-inning games. This is intentionally modest
    # until pitch-level bullpen usage is incorporated.
    work = defaultdict(float)
    cutoff = target - timedelta(days=3)
    for g in history:
        gd = _date_of_game(g)
        if not gd or gd < cutoff or gd >= target or not _completed(g):
            continue
        innings = 9
        try:
            innings = int((g.get("linescore") or {}).get("currentInning") or 9)
        except (TypeError, ValueError):
            pass
        burden = 1.0 + max(0, innings - 9) * 0.18
        for side in ("home", "away"):
            tid = _team_id(g.get("teams", {}).get(side, {}).get("team", {}))
            if tid:
                work[tid] += burden
    return work


def _venue_factors(history: list[dict[str, Any]], league_total: float) -> dict[int, float]:
    vals = defaultdict(lambda: [0.0, 0.0])
    for g in history:
        if not _completed(g):
            continue
        vid = (g.get("venue") or {}).get("id")
        hs, aws = _score(g, "home"), _score(g, "away")
        if vid and hs is not None and aws is not None:
            vals[int(vid)][0] += hs + aws
            vals[int(vid)][1] += 1
    out = {}
    for vid, (runs, n) in vals.items():
        # 20-game shrinkage keeps this a contextual adjustment, not a park overfit.
        out[vid] = (runs + 20 * league_total) / (n + 20) / league_total
    return out


def _starter_adjustment(stats: dict[str, float | None], league_ra9: float) -> float:
    era, whip, ip = stats.get("era"), stats.get("whip"), stats.get("ip")
    if era is None or ip is None or ip < 15:
        return 0.0
    reliability = min(1.0, ip / 100.0)
    era_delta = (league_ra9 - era) * 0.34 * reliability
    whip_delta = 0.0 if whip is None else (1.30 - whip) * 0.65 * reliability
    return max(-1.1, min(1.1, era_delta + whip_delta))


def build_v2(target_date: date | None = None) -> dict[str, Any]:
    target_date = target_date or datetime.now(CHICAGO).date()
    history = _schedule(target_date - timedelta(days=120), target_date - timedelta(days=1))
    todays = _schedule(target_date, target_date)
    league, home_adv, shrunk, offense, defense = _league_and_team_rates(history, target_date)
    bullpen = _bullpen_workload(history, target_date)
    parks = _venue_factors(history, league * 2)
    dk = _dk_market_lookup()

    games = []
    for game in todays:
        status_detail = str(game.get("status", {}).get("detailedState") or "")
        if status_detail.lower() in {"postponed", "cancelled", "canceled"}:
            continue
        ht = game.get("teams", {}).get("home", {}).get("team", {})
        at = game.get("teams", {}).get("away", {}).get("team", {})
        hid, aid = _team_id(ht), _team_id(at)
        if hid is None or aid is None:
            continue
        hn, an = _team_name(ht), _team_name(at)
        home_off, away_off = shrunk(offense, hid), shrunk(offense, aid)
        home_def, away_def = shrunk(defense, hid), shrunk(defense, aid)

        base_home = 0.52 * home_off + 0.48 * away_def + home_adv / 2
        base_away = 0.52 * away_off + 0.48 * home_def - home_adv / 2

        hpid, apid = _pitcher_id(game, "home"), _pitcher_id(game, "away")
        hp, ap = _pitcher_recent(hpid), _pitcher_recent(apid)
        # A strong home starter suppresses away runs and vice versa.
        home_starter = _starter_adjustment(hp, league)
        away_starter = _starter_adjustment(ap, league)

        # Fatigue proxy: excess workload above two games in the previous 3 days.
        home_pen = max(0.0, bullpen.get(hid, 0.0) - 2.0) * 0.10
        away_pen = max(0.0, bullpen.get(aid, 0.0) - 2.0) * 0.10

        vid = (game.get("venue") or {}).get("id")
        park = parks.get(int(vid), 1.0) if vid else 1.0
        home_runs = (base_home - away_starter + away_pen) * park
        away_runs = (base_away - home_starter + home_pen) * park
        home_runs = max(1.5, min(8.8, home_runs))
        away_runs = max(1.5, min(8.8, away_runs))
        hpct = _win_probability(home_runs, away_runs)

        market = _extract_dk(dk.get((hn.lower(), an.lower())))
        games.append({
            "game_id": str(game.get("gamePk") or ""),
            "game_date_utc": game.get("gameDate"),
            "home_team": hn, "away_team": an,
            "venue": (game.get("venue") or {}).get("name"),
            "status": status_detail,
            "predicted_home_runs": round(home_runs, 2),
            "predicted_away_runs": round(away_runs, 2),
            "predicted_total_runs": round(home_runs + away_runs, 2),
            "predicted_winner": hn if hpct >= .5 else an,
            "home_win_probability": round(hpct, 4),
            "away_win_probability": round(1-hpct, 4),
            "home_probable_pitcher": _pitcher_name(game, "home"),
            "away_probable_pitcher": _pitcher_name(game, "away"),
            "home_starter_era": hp.get("era"), "away_starter_era": ap.get("era"),
            "home_starter_whip": hp.get("whip"), "away_starter_whip": ap.get("whip"),
            "home_bullpen_workload_3d": round(bullpen.get(hid, 0.0), 2),
            "away_bullpen_workload_3d": round(bullpen.get(aid, 0.0), 2),
            "park_run_factor": round(park, 3),
            "market_used_in_prediction": False,
            **market,
        })
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date.isoformat(),
        "sport": "MLB",
        "model_version": "mlb-runs-v2-candidate",
        "model_status": "candidate_not_promoted",
        "features": ["recent offense", "recent run prevention", "probable starter ERA/WHIP", "3-day bullpen workload proxy", "venue run factor", "home advantage"],
        "games": games,
    }


def main():
    payload = build_v2()
    OUT.parent.mkdir(parents=True, exist_ok=True); DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False)
    OUT.write_text(text, encoding="utf-8"); DOCS_OUT.write_text(text, encoding="utf-8")
    print(json.dumps({"model": payload["model_version"], "games": len(payload["games"]), "target_date": payload["target_date"]}, indent=2))


if __name__ == "__main__":
    main()
