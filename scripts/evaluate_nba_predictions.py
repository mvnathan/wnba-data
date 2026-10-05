#!/usr/bin/env python3
from __future__ import annotations
import json, math, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

ROOT=Path(__file__).resolve().parents[1]
SOURCE="docs/nba-latest.json"
OUT=ROOT/"docs"/"nba-performance.json"
ESPN="https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

def dt(v):
    try:
        x=datetime.fromisoformat(str(v).replace("Z","+00:00"))
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    except Exception:return None

def finite(v):
    try:
        x=float(v);return x if math.isfinite(x) else None
    except Exception:return None

def snapshots():
    try:commits=subprocess.check_output(["git","log","--format=%H","--",SOURCE],cwd=ROOT,text=True).split()
    except Exception:return []
    out=[]
    for c in commits:
        try:
            raw=subprocess.check_output(["git","show",f"{c}:{SOURCE}"],cwd=ROOT,text=True)
            d=json.loads(raw);gen=dt(d.get("generated_at_utc"))
            if gen:out.append((c,gen,d))
        except Exception:continue
    return out

def issued():
    earliest={}
    for commit,gen,d in snapshots():
        for g in d.get("games",[]):
            start=dt(g.get("date") or g.get("game_date_utc"))
            if not start or gen>=start:continue
            key=str(g.get("game_id"))
            row={**g,"prediction_issued_at":gen.isoformat(),"prediction_commit":commit[:8],"target_date":d.get("target_date")}
            if key not in earliest or gen<dt(earliest[key]["prediction_issued_at"]):earliest[key]=row
    return list(earliest.values())

def actuals(dates):
    out={}
    for day in sorted(dates):
        try:
            r=requests.get(ESPN,params={"dates":day.replace("-","")},timeout=30,headers={"user-agent":"SportsModelHub/1.0"});r.raise_for_status()
        except Exception:continue
        for e in r.json().get("events",[]):
            typ=(e.get("status") or {}).get("type") or {}
            if not typ.get("completed"):continue
            comp=(e.get("competitions") or [{}])[0];cs=comp.get("competitors") or []
            h=next((x for x in cs if x.get("homeAway")=="home"),None);a=next((x for x in cs if x.get("homeAway")=="away"),None)
            if not h or not a:continue
            hs=finite(h.get("score"));as_=finite(a.get("score"))
            if hs is None or as_ is None:continue
            out[str(e.get("id"))]={"actual_home_score":hs,"actual_away_score":as_,"actual_margin":hs-as_,"actual_total":hs+as_}
    return out

def grade(pred,act):
    hp=finite(pred.get("home_win_probability"));pm=finite(pred.get("predicted_margin"));pt=finite(pred.get("predicted_total"))
    hs=finite(pred.get("market_home_spread"));mt=finite(pred.get("market_total"))
    if hp is None or pm is None or pt is None:return None
    actual_home=act["actual_margin"]>0
    row={
        "game_id":str(pred.get("game_id")),"game_date_utc":pred.get("date") or pred.get("game_date_utc"),
        "home_team":pred.get("home_team"),"away_team":pred.get("away_team"),
        "home_abbr":pred.get("home_abbr"),"away_abbr":pred.get("away_abbr"),
        "prediction_issued_at":pred.get("prediction_issued_at"),"prediction_commit":pred.get("prediction_commit"),
        "home_win_probability":hp,"predicted_margin":pm,"predicted_total":pt,
        **act,"winner_correct":bool((hp>=.5)==actual_home),
        "market_home_spread":hs,"market_total":mt,"market_bookmaker":pred.get("market_bookmaker"),
        "ats_result":None,"ou_result":None,
    }
    if hs is not None:
        medge=pm+hs;aedge=act["actual_margin"]+hs
        if abs(aedge)<1e-9:row["ats_result"]="push"
        elif abs(medge)>=.5:row["ats_result"]="win" if ((medge>0)==(aedge>0)) else "loss"
    if mt is not None:
        medge=pt-mt;aedge=act["actual_total"]-mt
        if abs(aedge)<1e-9:row["ou_result"]="push"
        elif abs(medge)>=.5:row["ou_result"]="win" if ((medge>0)==(aedge>0)) else "loss"
    return row

def summary(rows,key):
    if key=="ml":vals=["win" if r["winner_correct"] else "loss" for r in rows]
    else:vals=[r.get(key+"_result") for r in rows if r.get(key+"_result") in {"win","loss","push"}]
    w=vals.count("win");l=vals.count("loss");p=vals.count("push")
    return {"wins":w,"losses":l,"pushes":p,"graded":w+l,"rate":w/(w+l) if w+l else None}

def main():
    preds=issued();acts=actuals({str(p.get("target_date") or "") for p in preds if p.get("target_date")})
    rows=[]
    for p in preds:
        a=acts.get(str(p.get("game_id")))
        if a:
            r=grade(p,a)
            if r:rows.append(r)
    payload={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"method":"Earliest committed pregame NBA forecast and market line joined to ESPN final score by game ID","summary":{"n":len(rows),"ml":summary(rows,"ml"),"ats":summary(rows,"ats"),"ou":summary(rows,"ou")},"games":rows}
    OUT.write_text(json.dumps(payload,indent=2,allow_nan=False))
    print(json.dumps(payload["summary"],indent=2))

if __name__=="__main__":main()
