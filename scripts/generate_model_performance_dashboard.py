#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs"/"model-performance-weekly.json"
def load(path,default):
    try:return json.loads((ROOT/path).read_text(encoding="utf-8"))
    except Exception:return default
def week_label(dtstr):
    dt=datetime.fromisoformat(str(dtstr).replace("Z","+00:00"))
    y,w,_=dt.isocalendar();m=datetime.fromisocalendar(y,w,1);return m.strftime("%b %d")
def agg(rows):
    if not rows:return None
    def avg(k):
        vals=[float(r[k]) for r in rows if r.get(k) is not None];return mean(vals) if vals else None
    acc=[1.0 if r.get("winner_correct") else 0.0 for r in rows if r.get("winner_correct") is not None]
    return {"n":len(rows),"winner_accuracy":mean(acc) if acc else None,"brier_score":avg("brier_score"),"margin_mae":avg("margin_mae"),"total_mae":avg("total_mae")}
def build_wnba():
    d=load("docs/performance.json",{});rows=[]
    for g in d.get("games") or []:
        if int(g.get("season") or 0)!=2026:continue
        rows.append({"date":g.get("game_date_utc"),"winner_correct":g.get("winner_correct"),"brier_score":g.get("brier_score"),"margin_mae":g.get("absolute_margin_error"),"total_mae":g.get("absolute_total_error")})
    return rows,"issued pregame forecasts"
def build_tennis():
    d=load("docs/tennis-performance.json",{});rows=[]
    for m in d.get("matches") or []:
        dt=str(m.get("start_time_utc") or m.get("target_date") or "")
        if not dt.startswith("2026"):continue
        rows.append({"date":dt,"winner_correct":m.get("winner_correct"),"brier_score":m.get("brier_score"),"margin_mae":m.get("absolute_margin_error"),"total_mae":m.get("absolute_total_error")})
    return rows,"issued pre-match forecasts"
def build_nfl():
    d=load("data/nfl_model_comparison.json",{});rows=[]
    for r in d.get("rows") or []:
        if int(r.get("season") or 0)!=2026:continue
        p=r.get("hybrid_context") or r.get("hybrid") or r.get("full")
        if not p:continue
        y=float(r["actual_home"])>float(r["actual_away"]);hp=float(p["home_win_probability"])
        rows.append({"date":"2026-01-01","winner_correct":(hp>=.5)==y,"brier_score":(hp-(1.0 if y else 0.0))**2,"margin_mae":abs(float(p["margin"])-(float(r["actual_home"])-float(r["actual_away"]))),"total_mae":abs(float(p["total"])-(float(r["actual_home"])+float(r["actual_away"]))),"week":int(r.get("week") or 0)})
    return rows,"leakage-safe current-model replay; issued-history tracking is accumulating prospectively"
def build_mlb():
    d=load("data/mlb_model_comparison.json",{});rows=[]
    for r in d.get("rows") or []:
        dt=str(r.get("date") or "")
        if not dt.startswith("2026"):continue
        hp=float(r["v2c_home_prob"]);y=float(r["home_score"])>float(r["away_score"]);ph=float(r["v2c_home_runs"]);pa=float(r["v2c_away_runs"])
        rows.append({"date":dt,"winner_correct":(hp>=.5)==y,"brier_score":(hp-(1.0 if y else 0.0))**2,"margin_mae":abs((ph-pa)-(float(r["home_score"])-float(r["away_score"]))),"total_mae":abs((ph+pa)-(float(r["home_score"])+float(r["away_score"])))})
    return rows,"leakage-safe current-model replay for available 2026 backtest window"
def weekly(rows,sport):
    groups=defaultdict(list)
    if sport=="NFL":
        for r in rows:groups[f"Week {r['week']}"].append(r)
        order=sorted(groups,key=lambda x:int(x.split()[-1]))
    else:
        for r in rows:groups[week_label(r["date"])].append(r)
        order=list(dict.fromkeys(week_label(r["date"]) for r in sorted(rows,key=lambda x:x["date"])))
    return [{"week":label,**agg(groups[label])} for label in order if agg(groups[label])]
def main():
    sports={}
    for sport,fn in {"WNBA":build_wnba,"NFL":build_nfl,"MLB":build_mlb,"Tennis":build_tennis}.items():
        rows,method=fn();dates=[r.get("date") for r in rows if r.get("date")]
        sports[sport]={"method":method,"tracking_start":min(dates) if dates else None,"tracking_end":max(dates) if dates else None,"overall":agg(rows),"weekly":weekly(rows,sport)}
    payload={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"season":2026,"sports":sports,"notes":["Higher winner accuracy is better; lower Brier, margin MAE, and total MAE are better.","WNBA and Tennis use issued forecasts. NFL and MLB use leakage-safe current-model replay where complete issued-history archives are not yet available."]}
    OUT.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v["overall"] for k,v in sports.items()},indent=2))
if __name__=="__main__":main()
