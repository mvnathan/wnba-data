#!/usr/bin/env python3
"""Build simple weekly market-performance ratios for SportsModelHub.

Primary metrics:
- ML: model winner hit rate
- ATS: model side vs spread hit rate
- O/U: model over/under hit rate

WNBA uses the sportsbook line saved with the issued pregame forecast.
NFL ATS/O-U use nflverse historical closing lines because a complete issued-line archive
was not retained for the full season.
MLB and Tennis expose ML history now; ATS/O-U remain unavailable until issued-line
history is retained prospectively.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs"/"model-performance-weekly.json"
OUT_DATA=ROOT/"data"/"model-performance-weekly.json"
PICK_EDGE=0.5

def load_json(path, default):
    try:return json.loads((ROOT/path).read_text())
    except Exception:return default

def week_label(day):
    monday=day-timedelta(days=day.weekday())
    sunday=monday+timedelta(days=6)
    try:return monday.strftime("%b %-d")+"–"+sunday.strftime("%b %-d")
    except ValueError:return monday.strftime("%b %d").replace(" 0"," ")+"–"+sunday.strftime("%b %d").replace(" 0"," ")

def result_record(value):
    if value is True:return "win"
    if value is False:return "loss"
    return None

def market_summary(rows,key):
    vals=[r.get(key) for r in rows if r.get(key) in {"win","loss","push"}]
    wins=sum(v=="win" for v in vals); losses=sum(v=="loss" for v in vals); pushes=sum(v=="push" for v in vals)
    graded=wins+losses
    return {
        "wins":wins,"losses":losses,"pushes":pushes,
        "graded":graded,
        "rate":(wins/graded) if graded else None,
    }

def summarize(rows):
    return {
        "n":len(rows),
        "ml":market_summary(rows,"ml_result"),
        "ats":market_summary(rows,"ats_result"),
        "ou":market_summary(rows,"ou_result"),
    }

def aggregate(name,sport,rows,source,season,market_notes):
    buckets=defaultdict(list)
    for r in rows:
        d=r["_date"].date()
        buckets[d-timedelta(days=d.weekday())].append(r)
    weekly=[]
    for monday in sorted(buckets):
        weekly.append({
            "week_start":monday.isoformat(),
            "week_label":week_label(monday),
            **summarize(buckets[monday]),
        })
    return {
        "name":name,"sport":sport,"season":season,"source_type":source,
        "market_notes":market_notes,
        "overall":summarize(rows),
        "weekly":weekly,
    }

def nfl_current_season(season):
    from scripts.backtest_nfl_models import load
    from src.nfl_v2_core import predict_hybrid_context_row
    df=load()
    eval_df=df[(df["game_type"]=="REG")&(df["season"]==season)&df["home_score"].notna()&df["away_score"].notna()].copy()
    rows=[]
    for _,row in eval_df.iterrows():
        p,_,_=predict_hybrid_context_row(df,row)
        actual_margin=float(row.home_score)-float(row.away_score)
        actual_total=float(row.home_score)+float(row.away_score)
        pred_margin=float(p["margin"]); pred_total=float(p["total"])
        pred_home_win=float(p["home_win_probability"])>=.5
        ml="win" if (pred_home_win==(actual_margin>0)) else "loss"

        ats=None
        spread=pd.to_numeric(pd.Series([row.get("spread_line")]),errors="coerce").iloc[0]
        if pd.notna(spread):
            spread=float(spread) # nflverse: points the home team was favored by
            edge=pred_margin-spread
            actual_edge=actual_margin-spread
            if abs(actual_edge)<1e-9: ats="push"
            elif abs(edge)>=PICK_EDGE: ats="win" if ((edge>0)==(actual_edge>0)) else "loss"

        ou=None
        total_line=pd.to_numeric(pd.Series([row.get("total_line")]),errors="coerce").iloc[0]
        if pd.notna(total_line):
            total_line=float(total_line)
            model_edge=pred_total-total_line
            actual_edge=actual_total-total_line
            if abs(actual_edge)<1e-9: ou="push"
            elif abs(model_edge)>=PICK_EDGE: ou="win" if ((model_edge>0)==(actual_edge>0)) else "loss"

        dt=pd.to_datetime(row.gameday,utc=True).to_pydatetime()
        rows.append({"_date":dt,"ml_result":ml,"ats_result":ats,"ou_result":ou})
    return rows

def nba_rows(season):
    issued=load_json("docs/nba-performance.json",{})
    if issued.get("games"):
        rows=[]
        for r in issued.get("games",[]):
            try:dt=datetime.fromisoformat(str(r.get("game_date_utc")).replace("Z","+00:00"))
            except Exception:continue
            if dt.year not in {season-1,season}:continue
            rows.append({
                "_date":dt,
                "ml_result":"win" if r.get("winner_correct") else "loss",
                "ats_result":r.get("ats_result"),
                "ou_result":r.get("ou_result"),
            })
        if rows:return rows
    d=load_json("data/nba_model_comparison.json",{})
    rows=[]
    for r in d.get("test_predictions",[]):
        try:dt=datetime.fromisoformat(str(r.get("date")).replace("Z","+00:00"))
        except Exception:continue
        if dt.year not in {season-1,season}:continue
        rows.append({"_date":dt,"ml_result":"win" if r.get("winner_correct") else "loss","ats_result":None,"ou_result":None})
    return rows

def mlb_rows(season):
    d=load_json("data/mlb_model_comparison.json",{})
    rows=[]
    for r in d.get("rows",[]):
        dt=datetime.fromisoformat(str(r["date"])).replace(tzinfo=timezone.utc)
        if dt.year!=season:continue
        pred_home=float(r["v2c_home_prob"])>=.5
        actual_home=float(r["home_score"])>float(r["away_score"])
        rows.append({"_date":dt,"ml_result":"win" if pred_home==actual_home else "loss","ats_result":None,"ou_result":None})
    return rows

def wnba_rows(season):
    d=load_json("docs/performance.json",{})
    rows=[]
    for r in d.get("games",[]):
        try:dt=datetime.fromisoformat(str(r.get("game_date_utc")).replace("Z","+00:00"))
        except Exception:continue
        if int(r.get("season") or dt.year)!=season:continue
        ml=result_record(r.get("winner_correct"))
        actual_side=r.get("actual_market_side")
        ats="push" if actual_side=="push" else result_record(r.get("ats_model_correct"))
        actual_total=r.get("actual_total_side")
        ou="push" if actual_total=="push" else result_record(r.get("total_model_correct"))
        rows.append({"_date":dt,"ml_result":ml,"ats_result":ats,"ou_result":ou})
    return rows

def tennis_rows(season,tour):
    d=load_json("docs/tennis-performance.json",{})
    rows=[]
    for r in d.get("matches",[]):
        if r.get("tour")!=tour:continue
        try:dt=datetime.fromisoformat(str(r.get("start_time_utc")).replace("Z","+00:00"))
        except Exception:continue
        if dt.year!=season:continue
        rows.append({"_date":dt,"ml_result":result_record(r.get("winner_correct")),"ats_result":None,"ou_result":None})
    return rows

def main():
    season=datetime.now(timezone.utc).year
    specs=[
        (nfl_current_season,(season,),"NFL v2","NFL","leakage_safe_current_season_reconstruction",{
            "ml":"Leakage-safe model winner vs final result.",
            "ats":"Model side vs nflverse historical closing spread.",
            "ou":"Model total side vs nflverse historical closing total."
        }),
        (nba_rows,(season,),"NBA roster model","NBA","chronological_two_season_holdout",{
            "ml":"Untouched chronological holdout winner prediction vs final result.",
            "ats":"Prospective issued-line tracking begins with NBA launch; historical line data is not imputed.",
            "ou":"Prospective issued-line tracking begins with NBA launch; historical line data is not imputed."
        }),
        (mlb_rows,(season,),"MLB v2","MLB","leakage_safe_current_season_backtest",{
            "ml":"Leakage-safe model winner vs final result.",
            "ats":"Historical issued spread lines were not retained; prospective tracking is required.",
            "ou":"Historical issued totals were not retained; prospective tracking is required."
        }),
        (wnba_rows,(season,),"WNBA production","WNBA","issued_predictions",{
            "ml":"Issued pregame model winner vs final result.",
            "ats":"Issued model side vs sportsbook spread saved with the forecast.",
            "ou":"Issued model total side vs sportsbook total saved with the forecast."
        }),
        (tennis_rows,(season,"ATP"),"ATP production","ATP","issued_predictions",{
            "ml":"Issued pre-match model winner vs final result.",
            "ats":"Historical sportsbook game spreads were not retained with issued forecasts.",
            "ou":"Historical sportsbook totals were not retained with issued forecasts."
        }),
        (tennis_rows,(season,"WTA"),"WTA production","WTA","issued_predictions",{
            "ml":"Issued pre-match model winner vs final result.",
            "ats":"Historical sportsbook game spreads were not retained with issued forecasts.",
            "ou":"Historical sportsbook totals were not retained with issued forecasts."
        }),
    ]
    models=[]
    for fn,args,name,sport,source,notes in specs:
        try:models.append(aggregate(name,sport,fn(*args),source,season,notes))
        except Exception as e:models.append({"name":name,"sport":sport,"season":season,"source_type":source,"market_notes":notes,"overall":{},"weekly":[],"error":str(e)})
    payload={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "season":season,"status":"ok","models":models,
        "metric_definition":"Hit rate = wins / (wins + losses). Pushes are displayed separately and excluded from the denominator.",
        "method_note":"Market performance is only calculated where a valid benchmark line was retained. No missing historical lines are imputed."
    }
    for p in (OUT,OUT_DATA):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(payload,indent=2,allow_nan=False))
    print(json.dumps({m["sport"]:m.get("overall") for m in models},indent=2))

if __name__=="__main__":main()

# NBA v2 performance refresh trigger: 2026-10-05
