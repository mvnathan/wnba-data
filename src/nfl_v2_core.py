from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.competitive_context import build_context, bounded_effort_adjustment, nfl_groups

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


def predict_v1_row(df: pd.DataFrame, row: pd.Series) -> dict[str,float]:
    season=int(row["season"]); week=int(row["week"])
    hist=df[(df["season"]==season)&(df["game_type"]=="REG")&(df["week"]<week)&df["home_score"].notna()&df["away_score"].notna()].copy()
    league=_league_mean(hist) if len(hist) else 22.0
    home_adv=_home_adv(hist) if len(hist) else HOME_ADV_PRIOR
    pf=defaultdict(list); pa=defaultdict(list)
    for _,g in hist.iterrows():
        pf[str(g.home_team)].append(float(g.home_score)); pa[str(g.home_team)].append(float(g.away_score))
        pf[str(g.away_team)].append(float(g.away_score)); pa[str(g.away_team)].append(float(g.home_score))
    def shr(values):
        if not values:return league
        obs=sum(values)/len(values); n=len(values)
        return (obs*n+league*3.5)/(n+3.5)
    home=str(row["home_team"]); away=str(row["away_team"])
    ho,hd=shr(pf[home]),shr(pa[home]); ao,ad=shr(pf[away]),shr(pa[away])
    location=str(row.get("location") or "Home").lower()
    hfa=0.0 if "neutral" in location else home_adv
    ph=max(10.0,min(39.5,0.53*ho+0.47*ad+hfa/2))
    pa_=max(10.0,min(39.5,0.53*ao+0.47*hd-hfa/2))
    margin=ph-pa_; total=ph+pa_; hp=1/(1+math.exp(-margin/7.25))
    return {"home_points":ph,"away_points":pa_,"margin":margin,"total":total,"home_win_probability":hp,"league_mean":league,"home_advantage":hfa,"home_offense":ho,"away_offense":ao,"home_defense":hd,"away_defense":ad}


def predict_hybrid_row(df: pd.DataFrame, row: pd.Series) -> tuple[dict[str,float], str]:
    week=int(row["week"])
    if week<=4:
        return predict_row(df,row,"full"), "early_season_enhanced"
    return predict_v1_row(df,row), "current_season_baseline"


def _nfl_context(df: pd.DataFrame, row: pd.Series):
    season=int(row["season"]); week=int(row["week"])
    hist=df[(df["season"]==season)&(df["game_type"]=="REG")&(df["week"]<week)&df["home_score"].notna()&df["away_score"].notna()].copy()
    rows=[]
    for _,g in hist.iterrows():
        rows.append({"home":str(g.home_team),"away":str(g.away_team),"home_score":float(g.home_score),"away_score":float(g.away_score)})
    return build_context(rows,total_games=17,playoff_slots=7,groups=nfl_groups(),late_season_threshold=.64)


def predict_hybrid_context_row(df: pd.DataFrame, row: pd.Series) -> tuple[dict[str,float], str, dict[str,Any]]:
    pred,strategy=predict_hybrid_row(df,row)
    week=int(row["week"])
    context_map=_nfl_context(df,row)
    home=str(row["home_team"]); away=str(row["away_team"])
    hc=context_map.get(home); ac=context_map.get(away)
    adjustment=0.0
    # Keep early/mid-season unchanged. Late-season competitive context is bounded
    # to less than one point until prospectively validated further.
    if week>=12:
        adjustment=bounded_effort_adjustment(hc,ac,.75)
        pred=dict(pred)
        pred["home_points"]+=adjustment/2
        pred["away_points"]-=adjustment/2
        pred["margin"]=pred["home_points"]-pred["away_points"]
        pred["total"]=pred["home_points"]+pred["away_points"]
        pred["home_win_probability"]=1/(1+math.exp(-pred["margin"]/PROB_SCALE))
        strategy=strategy+"+postseason_context"
    meta={
        "home":hc.to_dict() if hc else None,
        "away":ac.to_dict() if ac else None,
        "score_adjustment_home_minus_away":round(adjustment,3),
        "actual_player_availability_required_for_star_rest_confirmation":True,
    }
    return pred,strategy,meta
