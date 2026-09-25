from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

PRIOR_GAMES = 4.0
HOME_ADV_PRIOR = 1.7
PROB_SCALE = 7.6
MAX_TEAM_GAMES = 16
GAME_DECAY = 0.88


def _num(v: Any) -> float | None:
    try:
        x=float(v)
        return None if math.isnan(x) else x
    except Exception:
        return None


def _history_rows(df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    season=int(target["season"]); week=int(target["week"])
    mask=(df["game_type"]=="REG") & df["home_score"].notna() & df["away_score"].notna()
    prior=df[mask & ((df["season"]<season) | ((df["season"]==season) & (df["week"]<week)))].copy()
    return prior


def _league_mean(history: pd.DataFrame) -> float:
    vals=pd.concat([history["home_score"],history["away_score"]],ignore_index=True)
    vals=pd.to_numeric(vals,errors="coerce").dropna()
    return float(vals.tail(512).mean()) if len(vals) else 22.0


def _home_adv(history: pd.DataFrame) -> float:
    h=history.tail(512)
    dif=(pd.to_numeric(h["home_score"],errors="coerce")-pd.to_numeric(h["away_score"],errors="coerce")).dropna()
    if not len(dif): return HOME_ADV_PRIOR
    return float((dif.sum()+HOME_ADV_PRIOR*64)/(len(dif)+64))


def _team_games(history: pd.DataFrame, team: str) -> list[tuple[float,float,str | None]]:
    rows=[]
    h=history[(history["home_team"]==team)|(history["away_team"]==team)].tail(MAX_TEAM_GAMES)
    for _,r in h.iterrows():
        if r["home_team"]==team:
            scored,allowed=_num(r["home_score"]),_num(r["away_score"])
            qb=r.get("home_qb_name")
        else:
            scored,allowed=_num(r["away_score"]),_num(r["home_score"])
            qb=r.get("away_qb_name")
        if scored is not None and allowed is not None:
            rows.append((scored,allowed,None if pd.isna(qb) else str(qb)))
    return rows


def _weighted_rate(values: list[float], league: float) -> float:
    if not values: return league
    # newest gets weight 1.0
    rev=list(reversed(values))
    weights=[GAME_DECAY**i for i in range(len(rev))]
    sw=sum(weights)
    obs=sum(v*w for v,w in zip(rev,weights))/sw
    return (obs*sw+league*PRIOR_GAMES)/(sw+PRIOR_GAMES)


def _qb_adjust(games: list[tuple[float,float,str | None]], qb: str | None, league: float) -> float:
    if not qb or not games: return 0.0
    qb_scores=[s for s,_,q in games if q==qb]
    if not qb_scores: return -0.55
    all_scores=[s for s,_,_ in games]
    team_base=sum(all_scores)/len(all_scores)
    qb_avg=sum(qb_scores)/len(qb_scores)
    reliability=min(1.0,len(qb_scores)/6.0)
    return max(-1.5,min(1.5,(qb_avg-team_base)*0.22*reliability))


def predict_row(df: pd.DataFrame, row: pd.Series, variant: str="v2") -> dict[str,float]:
    history=_history_rows(df,row)
    league=_league_mean(history)
    home_adv=_home_adv(history)
    home=str(row["home_team"]); away=str(row["away_team"])
    hg=_team_games(history,home); ag=_team_games(history,away)

    home_off=_weighted_rate([x[0] for x in hg],league)
    home_def=_weighted_rate([x[1] for x in hg],league)
    away_off=_weighted_rate([x[0] for x in ag],league)
    away_def=_weighted_rate([x[1] for x in ag],league)

    location=str(row.get("location") or "Home").lower()
    hfa=0.0 if "neutral" in location else home_adv
    ph=0.53*home_off+0.47*away_def+hfa/2
    pa=0.53*away_off+0.47*home_def-hfa/2

    if variant in ("rest","full"):
        hr=_num(row.get("home_rest")); ar=_num(row.get("away_rest"))
        if hr is not None and ar is not None:
            rest=max(-7.0,min(7.0,hr-ar))*0.09
            ph+=rest/2; pa-=rest/2

    if variant in ("qb","full"):
        hqb=None if pd.isna(row.get("home_qb_name")) else str(row.get("home_qb_name"))
        aqb=None if pd.isna(row.get("away_qb_name")) else str(row.get("away_qb_name"))
        ph+=_qb_adjust(hg,hqb,league)
        pa+=_qb_adjust(ag,aqb,league)

    margin=ph-pa
    if variant in ("division","full") and int(_num(row.get("div_game")) or 0)==1:
        margin*=0.94
        mid=(ph+pa)/2
        ph=mid+margin/2; pa=mid-margin/2

    ph=max(9.0,min(41.0,ph)); pa=max(9.0,min(41.0,pa))
    margin=ph-pa
    total=ph+pa
    hp=1/(1+math.exp(-margin/PROB_SCALE))
    return {
        "home_points":ph,"away_points":pa,"margin":margin,"total":total,
        "home_win_probability":hp,"league_mean":league,"home_advantage":hfa,
        "home_offense":home_off,"away_offense":away_off,"home_defense":home_def,"away_defense":away_def,
    }
