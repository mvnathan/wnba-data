#!/usr/bin/env python3
"""Leakage-safe chronological comparison of NFL v1 vs NFL v2 variants."""

from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd

from src.nfl_v2_core import predict_row

URL="https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT=Path("data/nfl_model_comparison.json")
VARIANTS=("v1","base","rest","qb","division","full")


def load():
    df=pd.read_csv(URL,low_memory=False)
    for c in ("season","week","home_score","away_score","home_rest","away_rest","div_game"):
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.sort_values(["season","week","gameday","gametime"],na_position="last")


def v1_predict(df,row):
    season=int(row.season);week=int(row.week)
    hist=df[(df["season"]==season)&(df["game_type"]=="REG")&(df["week"]<week)&df["home_score"].notna()&df["away_score"].notna()].copy()
    league=pd.concat([hist["home_score"],hist["away_score"]],ignore_index=True).mean()
    if pd.isna(league): league=22.0
    home_edges=(hist["home_score"]-hist["away_score"]).dropna()
    hfa=float((home_edges.sum()+1.7*32)/(len(home_edges)+32)) if len(home_edges) else 1.7
    pf=defaultdict(list);pa=defaultdict(list)
    for _,g in hist.iterrows():
        pf[str(g.home_team)].append(float(g.home_score));pa[str(g.home_team)].append(float(g.away_score))
        pf[str(g.away_team)].append(float(g.away_score));pa[str(g.away_team)].append(float(g.home_score))
    def shr(vals):
        if not vals:return float(league)
        obs=sum(vals)/len(vals);n=len(vals)
        return (obs*n+float(league)*3.5)/(n+3.5)
    ho,hd=shr(pf[str(row.home_team)]),shr(pa[str(row.home_team)])
    ao,ad=shr(pf[str(row.away_team)]),shr(pa[str(row.away_team)])
    ph=.53*ho+.47*ad+hfa/2;aw=.53*ao+.47*hd-hfa/2
    ph=max(10,min(39.5,ph));aw=max(10,min(39.5,aw));m=ph-aw;t=ph+aw;p=1/(1+math.exp(-m/7.25))
    return {"home_points":ph,"away_points":aw,"margin":m,"total":t,"home_win_probability":p}


def metrics(rows,key):
    if not rows:return {}
    acc=[];brier=[];margin=[];total=[];team=[]
    for r in rows:
        p=r[key];y=1.0 if r["actual_home"]>r["actual_away"] else 0.0
        acc.append((p["home_win_probability"]>=.5)==bool(y))
        brier.append((p["home_win_probability"]-y)**2)
        actual_margin=r["actual_home"]-r["actual_away"]
        margin.append(abs(p["margin"]-actual_margin))
        total.append(abs(p["total"]-(r["actual_home"]+r["actual_away"])))
        team.extend([abs(p["home_points"]-r["actual_home"]),abs(p["away_points"]-r["actual_away"])])
    return {"n_games":len(rows),"winner_accuracy":round(mean(acc),4),"brier_score":round(mean(brier),4),"margin_mae":round(mean(margin),3),"total_mae":round(mean(total),3),"team_score_mae":round(mean(team),3)}


def main():
    df=load();rows=[]
    eval_df=df[(df["game_type"]=="REG")&(df["season"]>=2024)&df["home_score"].notna()&df["away_score"].notna()].copy()
    for _,row in eval_df.iterrows():
        rec={"game_id":str(row.game_id),"season":int(row.season),"week":int(row.week),"actual_home":float(row.home_score),"actual_away":float(row.away_score)}
        rec["v1"]=v1_predict(df,row)
        for v in VARIANTS[1:]:rec[v]=predict_row(df,row,v)
        rows.append(rec)
    out={"evaluation":"leakage-safe chronological regular-season backtest; each prediction uses only games from earlier weeks","seasons":sorted({r["season"] for r in rows}),"metrics":{v:metrics(rows,v) for v in VARIANTS},"delta_full_minus_v1":{}}
    a=out["metrics"]["v1"];b=out["metrics"]["full"]
    for k in ("winner_accuracy","brier_score","margin_mae","total_mae","team_score_mae"):out["delta_full_minus_v1"][k]=round(b[k]-a[k],4)
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=="__main__":main()
