#!/usr/bin/env python3
"""Chronological, leakage-safe comparison of MLB v1 and v2."""
from __future__ import annotations
import argparse, json, math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from scripts.generate_mlb_predictions import build_predictions, _schedule, _completed, _score
from scripts.generate_mlb_v2 import build_v2

OUT=Path("data/mlb_model_comparison.json")

def metrics(rows, prefix):
    if not rows: return {}
    acc=[]; brier=[]; team_mae=[]; total_mae=[]
    for r in rows:
        hp=r[prefix+"_home_prob"]; y=1.0 if r["home_score"]>r["away_score"] else 0.0
        acc.append((hp>=.5)==bool(y))
        brier.append((hp-y)**2)
        team_mae.extend([abs(r[prefix+"_home_runs"]-r["home_score"]),abs(r[prefix+"_away_runs"]-r["away_score"])])
        total_mae.append(abs((r[prefix+"_home_runs"]+r[prefix+"_away_runs"])-(r["home_score"]+r["away_score"])))
    return {"n_games":len(rows),"winner_accuracy":round(mean(acc),4),"brier_score":round(mean(brier),4),"team_score_mae":round(mean(team_mae),3),"total_score_mae":round(mean(total_mae),3)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default=(date.today()-timedelta(days=21)).isoformat())
    ap.add_argument("--end",default=(date.today()-timedelta(days=1)).isoformat())
    args=ap.parse_args(); start=date.fromisoformat(args.start); end=date.fromisoformat(args.end)
    rows=[]; day=start
    while day<=end:
        actual={str(g.get("gamePk")):g for g in _schedule(day,day) if _completed(g)}
        if actual:
            v1={g["game_id"]:g for g in build_predictions(day)["games"]}
            v2={g["game_id"]:g for g in build_v2(day)["games"]}
            for gid,a in actual.items():
                if gid not in v1 or gid not in v2: continue
                hs,aws=_score(a,"home"),_score(a,"away")
                if hs is None or aws is None or hs==aws: continue
                x,y=v1[gid],v2[gid]
                rows.append({"date":day.isoformat(),"game_id":gid,"home_score":hs,"away_score":aws,
                    "v1_home_prob":x["home_win_probability"],"v1_home_runs":x["predicted_home_runs"],"v1_away_runs":x["predicted_away_runs"],
                    "v2_home_prob":y["home_win_probability"],"v2_home_runs":y["predicted_home_runs"],"v2_away_runs":y["predicted_away_runs"]})
        print(day.isoformat(),len(rows)); day+=timedelta(days=1)
    a,b=metrics(rows,"v1"),metrics(rows,"v2")
    payload={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"start":start.isoformat(),"end":end.isoformat(),"v1":a,"v2":b,
      "delta_v2_minus_v1":{"winner_accuracy":round(b.get("winner_accuracy",0)-a.get("winner_accuracy",0),4),"brier_score":round(b.get("brier_score",0)-a.get("brier_score",0),4),"team_score_mae":round(b.get("team_score_mae",0)-a.get("team_score_mae",0),3),"total_score_mae":round(b.get("total_score_mae",0)-a.get("total_score_mae",0),3)},
      "interpretation":"Higher winner accuracy is better; lower Brier and MAE are better.","rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in payload.items() if k!="rows"},indent=2))
if __name__=="__main__": main()
