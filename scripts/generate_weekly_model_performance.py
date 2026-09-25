#!/usr/bin/env python3
"""Build unified current-season weekly model performance for SportsModelHub."""
from __future__ import annotations
import json, math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs"/"model-performance-weekly.json"
OUT_DATA=ROOT/"data"/"model-performance-weekly.json"

def load_json(path, default):
    try:return json.loads((ROOT/path).read_text())
    except Exception:return default

def week_label(dt):
    monday=dt-timedelta(days=dt.weekday())
    sunday=monday+timedelta(days=6)
    return monday.strftime("%b %-d")+"–"+sunday.strftime("%b %-d")

def metrics(rows):
    if not rows:return {}
    def avg(k):
        v=[float(r[k]) for r in rows if r.get(k) is not None]
        return mean(v) if v else None
    wins=[bool(r["winner_correct"]) for r in rows if r.get("winner_correct") is not None]
    return {
        "n":len(rows),
        "winner_accuracy":mean(wins) if wins else None,
        "brier_score":avg("brier_score"),
        "margin_mae":avg("margin_mae"),
        "total_mae":avg("total_mae"),
    }

def aggregate(name,sport,rows,source,season):
    buckets=defaultdict(list)
    for r in rows:
        dt=r["_date"]
        buckets[dt.date()-timedelta(days=dt.weekday())].append(r)
    weekly=[]
    for monday in sorted(buckets):
        m=metrics(buckets[monday])
        weekly.append({
            "week_start":monday.isoformat(),
            "week_label":week_label(datetime.combine(monday,datetime.min.time())),
            **m
        })
    return {
        "name":name,"sport":sport,"season":season,"source_type":source,
        "overall":metrics(rows),"weekly":weekly
    }

def nfl_current_season(season):
    from scripts.backtest_nfl_models import load
    from src.nfl_v2_core import predict_hybrid_context_row
    df=load()
    eval_df=df[(df["game_type"]=="REG")&(df["season"]==season)&df["home_score"].notna()&df["away_score"].notna()].copy()
    rows=[]
    for _,row in eval_df.iterrows():
        p,_,_=predict_hybrid_context_row(df,row)
        y=1.0 if float(row.home_score)>float(row.away_score) else 0.0
        dt=pd.to_datetime(row.gameday,utc=True).to_pydatetime()
        rows.append({
            "_date":dt,
            "winner_correct":(float(p["home_win_probability"])>=.5)==bool(y),
            "brier_score":(float(p["home_win_probability"])-y)**2,
            "margin_mae":abs(float(p["margin"])-(float(row.home_score)-float(row.away_score))),
            "total_mae":abs(float(p["total"])-(float(row.home_score)+float(row.away_score))),
        })
    return rows

def mlb_rows(season):
    d=load_json("data/mlb_model_comparison.json",{})
    rows=[]
    for r in d.get("rows",[]):
        dt=datetime.fromisoformat(str(r["date"])).replace(tzinfo=timezone.utc)
        if dt.year!=season:continue
        hp=float(r["v2c_home_prob"]); y=1.0 if float(r["home_score"])>float(r["away_score"]) else 0.0
        pm=float(r["v2c_home_runs"])-float(r["v2c_away_runs"])
        am=float(r["home_score"])-float(r["away_score"])
        pt=float(r["v2c_home_runs"])+float(r["v2c_away_runs"])
        at=float(r["home_score"])+float(r["away_score"])
        rows.append({"_date":dt,"winner_correct":(hp>=.5)==bool(y),"brier_score":(hp-y)**2,"margin_mae":abs(pm-am),"total_mae":abs(pt-at)})
    return rows

def wnba_rows(season):
    d=load_json("docs/performance.json",{})
    rows=[]
    for r in d.get("games",[]):
        dt=datetime.fromisoformat(str(r.get("game_date_utc")).replace("Z","+00:00"))
        if int(r.get("season") or dt.year)!=season:continue
        rows.append({"_date":dt,"winner_correct":r.get("winner_correct"),"brier_score":r.get("brier_score"),"margin_mae":r.get("absolute_margin_error"),"total_mae":r.get("absolute_total_error")})
    return rows

def tennis_rows(season,tour):
    d=load_json("docs/tennis-performance.json",{})
    rows=[]
    for r in d.get("matches",[]):
        if r.get("tour")!=tour:continue
        dt=datetime.fromisoformat(str(r.get("start_time_utc")).replace("Z","+00:00"))
        if dt.year!=season:continue
        rows.append({"_date":dt,"winner_correct":r.get("winner_correct"),"brier_score":r.get("brier_score"),"margin_mae":r.get("absolute_margin_error"),"total_mae":r.get("absolute_total_error")})
    return rows

def main():
    season=datetime.now(timezone.utc).year
    models=[]
    for fn,args,name,sport,source in [
        (nfl_current_season,(season,),"NFL v2","NFL","leakage_safe_current_season_reconstruction"),
        (mlb_rows,(season,),"MLB v2","MLB","leakage_safe_current_season_backtest"),
        (wnba_rows,(season,),"WNBA production","WNBA","issued_predictions"),
        (tennis_rows,(season,"ATP"),"ATP production","ATP","issued_predictions"),
        (tennis_rows,(season,"WTA"),"WTA production","WTA","issued_predictions"),
    ]:
        try:
            rows=fn(*args)
            models.append(aggregate(name,sport,rows,source,season))
        except Exception as e:
            models.append({"name":name,"sport":sport,"season":season,"source_type":source,"overall":{},"weekly":[],"error":str(e)})

    payload={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "season":season,
        "status":"ok",
        "models":models,
        "method_note":"WNBA and tennis use forecasts actually issued before events. NFL and MLB use leakage-safe current-season historical reconstruction because complete issued-prediction archives were not retained for the full season. Weekly buckets run Monday-Sunday.",
    }
    for p in (OUT,OUT_DATA):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(payload,indent=2,allow_nan=False))
    print(json.dumps({"season":season,"models":{m["sport"]:len(m.get("weekly",[])) for m in models}},indent=2))

if __name__=="__main__":main()
