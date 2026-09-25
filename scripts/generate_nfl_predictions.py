#!/usr/bin/env python3
"""Generate SportsModelHub NFL v1 beta predictions.

Schedule/results come from nflverse's public games.csv. DraftKings is a
comparison layer only and is never used as a model input.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.sbr_market_odds import fetch_sbr_draftkings

CHICAGO = ZoneInfo("America/Chicago")
EASTERN = ZoneInfo("America/New_York")
NFLVERSE_GAMES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT = Path("predictions/nfl-latest.json")
DOCS_OUT = Path("docs/nfl-latest.json")

PRIOR_GAMES = 3.5
HOME_ADV_PRIOR = 1.7
PROB_SCALE = 7.25

TEAM_NAMES = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
    "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
    "LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","LA":"Los Angeles Rams",
    "MIA":"Miami Dolphins","MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints",
    "NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers",
    "SEA":"Seattle Seahawks","SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans",
    "WAS":"Washington Commanders",
}


def _load_season(season: int) -> pd.DataFrame:
    df = pd.read_csv(NFLVERSE_GAMES, low_memory=False)
    df = df[(df["season"] == season) & (df["game_type"] == "REG")].copy()
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    return df.sort_values(["week", "gameday", "gametime"], na_position="last")


def _game_datetime(row: pd.Series) -> datetime:
    day = str(row.get("gameday") or "")
    time = str(row.get("gametime") or "13:00")
    if not time or time == "nan":
        time = "13:00"
    try:
        local = datetime.fromisoformat(f"{day}T{time}").replace(tzinfo=EASTERN)
    except Exception:
        local = datetime.fromisoformat(f"{day}T13:00").replace(tzinfo=EASTERN)
    return local.astimezone(timezone.utc)


def _active_week(df: pd.DataFrame, now: datetime) -> int:
    remaining = df[df["home_score"].isna() | df["away_score"].isna()].copy()
    if remaining.empty:
        return 18
    today = now.astimezone(CHICAGO).date()
    upcoming = remaining[pd.to_datetime(remaining["gameday"], errors="coerce").dt.date >= today - timedelta(days=1)]
    if upcoming.empty:
        return int(remaining["week"].min())
    return int(upcoming["week"].min())


def _rates(history: pd.DataFrame, now: datetime) -> tuple[float, float, dict[str,float], dict[str,float], dict[str,int]]:
    pf_sum, pa_sum, weight, games = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(int)
    league_pts, home_edges = [], []

    for _, row in history.iterrows():
        try:
            hs, aws = float(row.home_score), float(row.away_score)
        except Exception:
            continue
        home, away = str(row.home_team), str(row.away_team)
        league_pts.extend([hs, aws])
        home_edges.append(hs - aws)
        age_days = max(0.0, (now.astimezone(timezone.utc) - _game_datetime(row)).total_seconds() / 86400)
        wt = math.exp(-math.log(2) * age_days / 28.0)
        for team, scored, allowed in ((home, hs, aws), (away, aws, hs)):
            pf_sum[team] += scored * wt
            pa_sum[team] += allowed * wt
            weight[team] += wt
            games[team] += 1

    league = sum(league_pts) / len(league_pts) if league_pts else 22.0
    raw_home = sum(home_edges) / len(home_edges) if home_edges else HOME_ADV_PRIOR
    home_adv = (raw_home * len(home_edges) + HOME_ADV_PRIOR * 32) / (len(home_edges) + 32)

    offense, defense = {}, {}
    for team in set(pf_sum) | set(pa_sum):
        w = weight[team]
        obs_pf = pf_sum[team] / w if w else league
        obs_pa = pa_sum[team] / w if w else league
        offense[team] = (obs_pf * w + league * PRIOR_GAMES) / (w + PRIOR_GAMES)
        defense[team] = (obs_pa * w + league * PRIOR_GAMES) / (w + PRIOR_GAMES)
    return league, home_adv, offense, defense, dict(games)


def _american_prob(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not x:
        return None
    return (-x) / ((-x) + 100.0) if x < 0 else 100.0 / (x + 100.0)


def _market_rows(date_strings: set[str]) -> list[dict[str, Any]]:
    rows = []
    for ds in sorted(date_strings):
        try:
            rows.extend(fetch_sbr_draftkings("nfl", ds))
        except Exception as exc:
            print(f"NFL DraftKings fetch failed for {ds}: {exc}")
    return rows


def _market_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str,str], list[dict[str, Any]]]:
    out: dict[tuple[str,str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("home_team") or "").lower(), str(r.get("away_team") or "").lower())
        out.setdefault(key, []).append(r)
    return out


def _closest(rows: list[dict[str, Any]] | None, start: str) -> dict[str, Any] | None:
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    try:
        target = datetime.fromisoformat(start.replace("Z", "+00:00"))
    except Exception:
        return rows[0]
    best, delta = rows[0], float("inf")
    for r in rows:
        try:
            stamp = datetime.fromisoformat(str(r.get("commence_time") or "").replace("Z", "+00:00"))
            d = abs((stamp - target).total_seconds())
            if d < delta:
                best, delta = r, d
        except Exception:
            pass
    return best


def _extract_market(row: dict[str, Any] | None) -> dict[str, Any]:
    out = {
        "market_bookmaker": None, "market_home_moneyline": None, "market_away_moneyline": None,
        "market_home_spread": None, "market_home_spread_price": None, "market_away_spread_price": None,
        "market_total": None, "market_over_price": None, "market_under_price": None, "market_updated_at": None,
    }
    if not row or not (row.get("bookmakers") or []):
        return out
    b = row["bookmakers"][0]
    out["market_bookmaker"] = b.get("title") or "DraftKings"
    out["market_updated_at"] = b.get("last_update")
    for market in b.get("markets") or []:
        outcomes = market.get("outcomes") or []
        if market.get("key") == "h2h":
            for x in outcomes:
                if x.get("name") == row.get("home_team"): out["market_home_moneyline"] = x.get("price")
                elif x.get("name") == row.get("away_team"): out["market_away_moneyline"] = x.get("price")
        elif market.get("key") == "spreads":
            for x in outcomes:
                if x.get("name") == row.get("home_team"):
                    out["market_home_spread"], out["market_home_spread_price"] = x.get("point"), x.get("price")
                elif x.get("name") == row.get("away_team"):
                    out["market_away_spread_price"] = x.get("price")
        elif market.get("key") == "totals":
            for x in outcomes:
                if x.get("name") == "Over":
                    out["market_total"], out["market_over_price"] = x.get("point"), x.get("price")
                elif x.get("name") == "Under":
                    if out["market_total"] is None: out["market_total"] = x.get("point")
                    out["market_under_price"] = x.get("price")
    return out


def build() -> dict[str, Any]:
    now = datetime.now(CHICAGO)
    season = now.year
    df = _load_season(season)
    week = _active_week(df, now)

    history = df[(df["week"] < week) & df["home_score"].notna() & df["away_score"].notna()].copy()
    slate = df[(df["week"] == week) & (df["home_score"].isna() | df["away_score"].isna())].copy()
    league, home_adv, offense, defense, counts = _rates(history, now)

    dates = {str(x) for x in slate["gameday"].dropna().tolist()}
    markets = _market_lookup(_market_rows(dates))

    games = []
    for _, row in slate.iterrows():
        ha, aa = str(row.home_team), str(row.away_team)
        hn, an = TEAM_NAMES.get(ha, ha), TEAM_NAMES.get(aa, aa)
        start = _game_datetime(row)

        home_pf, away_pf = offense.get(ha, league), offense.get(aa, league)
        home_pa, away_pa = defense.get(ha, league), defense.get(aa, league)
        ph = max(10.0, min(39.5, 0.53 * home_pf + 0.47 * away_pa + home_adv / 2))
        pa = max(10.0, min(39.5, 0.53 * away_pf + 0.47 * home_pa - home_adv / 2))
        margin, total = ph - pa, ph + pa
        hp = 1 / (1 + math.exp(-margin / PROB_SCALE))

        market = _extract_market(_closest(markets.get((hn.lower(), an.lower())), start.isoformat()))
        ml_prob = _american_prob(market.get("market_home_moneyline"))
        spread_edge = margin + float(market["market_home_spread"]) if market.get("market_home_spread") is not None else None
        total_edge = total - float(market["market_total"]) if market.get("market_total") is not None else None
        ml_edge = hp - ml_prob if ml_prob is not None else None

        games.append({
            "game_id": str(row.game_id),
            "game_date_utc": start.isoformat().replace("+00:00", "Z"),
            "season": season, "week": week,
            "home_team": hn, "away_team": an, "home_abbr": ha, "away_abbr": aa,
            "venue": None if pd.isna(row.get("stadium")) else str(row.get("stadium")),
            "status": "Scheduled", "status_state": "pre",
            "predicted_home_points": round(ph,1), "predicted_away_points": round(pa,1),
            "predicted_margin": round(margin,1), "predicted_total": round(total,1),
            "predicted_winner": hn if hp >= .5 else an,
            "home_win_probability": round(hp,4), "away_win_probability": round(1-hp,4),
            "team_games_home": counts.get(ha,0), "team_games_away": counts.get(aa,0),
            "model_version": "nfl-v1-beta", "market_used_in_prediction": False,
            **market,
            "model_market_spread_edge": round(spread_edge,2) if spread_edge is not None else None,
            "model_market_total_edge": round(total_edge,2) if total_edge is not None else None,
            "model_market_home_win_edge": round(ml_edge,4) if ml_edge is not None else None,
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_date": now.date().isoformat(),
        "season": season, "week": week,
        "model_version": "nfl-v1-beta", "model_status": "beta",
        "market_used_in_prediction": False,
        "market_event_count": sum(1 for g in games if g.get("market_bookmaker")),
        "games": games,
        "methodology": {
            "schedule_source": "nflverse games.csv",
            "history": "Current-season completed regular-season games before target week",
            "recency": "28-day exponential half-life",
            "shrinkage_games": PRIOR_GAMES,
            "home_field_points": round(home_adv,3),
            "league_points_per_team": round(league,3),
            "probability_scale": PROB_SCALE,
        },
    }


def main() -> None:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False)
    OUT.write_text(text)
    DOCS_OUT.write_text(text)
    print(json.dumps({"model":payload["model_version"],"week":payload["week"],"games":len(payload["games"]),"markets":payload["market_event_count"]},indent=2))


if __name__ == "__main__":
    main()

# nflverse schedule source replaces blocked ESPN datacenter access.

# Validate indexed SBR books: 2026-09-25
