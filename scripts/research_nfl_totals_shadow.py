"""Leakage-safe NFL totals shadow candidates. Research only; no production writes.
Input: historical nflverse games.csv with game-day results. Models use completed prior weeks only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.nfl_v2_core import predict_hybrid_context_row, predict_row

def number(x):
    try:
        z=float(x)
        return z if np.isfinite(z) else None
    except (TypeError,ValueError): return None

def candidate(df,row):
    history=df[(df.game_type=="REG") & df.home_score.notna() & df.away_score.notna() & ((df.season<row.season)|((df.season==row.season)&(df.week<row.week)))]
    base=predict_row(df,row,"full")
    # Context candidate: blend stable prior-season enhanced forecast and current hybrid.
    hybrid,_,_=predict_hybrid_context_row(df,row)
    context=.65*base["total"]+.35*hybrid["total"]
    # Possession proxy: when actual drive counts are unavailable, do not invent them.
    # Use independent scored/allowed rates and regression to league scoring mean.
    league=2*base["league_mean"]
    possession_proxy=.75*base["total"]+.25*league
    return {"production":hybrid["total"],"context_shadow":context,"scoring_proxy_shadow":possession_proxy}

def run(df,season,minimum_edge=0.5):
    df=df.sort_values(["season","week","gameday","gametime"],na_position="last")
    evals=df[(df.game_type=="REG")&(df.season==season)&df.home_score.notna()&df.away_score.notna()]
    rows=[]
    for _,g in evals.iterrows():
        preds=candidate(df,g)
        actual=float(g.home_score+g.away_score)
        line=number(g.get("total_line"))
        rec={"game_id":str(g.game_id),"season":int(g.season),"week":int(g.week),"actual_total":actual,"historical_closing_total":line}
        for key,pred in preds.items():
            item={"predicted_total":round(pred,3),"error":round(pred-actual,3)}
            if line is not None:
                edge=pred-line
                item["edge"]=round(edge,3)
                item["side"]="OVER" if edge>minimum_edge else "UNDER" if edge< -minimum_edge else "PASS"
                item["grade"]=("PUSH" if actual==line else "WIN" if (edge>0)==(actual>line) else "LOSS") if item["side"]!="PASS" else "PASS"
            rec[key]=item
        rows.append(rec)
    result={}
    for key in ("production","context_shadow","scoring_proxy_shadow"):
        errors=np.array([r[key]["error"] for r in rows],dtype=float)
        grades=[r[key].get("grade") for r in rows]
        w=grades.count("WIN");l=grades.count("LOSS")
        result[key]={"n":len(errors),"mae":round(float(np.abs(errors).mean()),3) if len(errors) else None,"bias":round(float(errors.mean()),3) if len(errors) else None,"rmse":round(float(np.sqrt((errors**2).mean())),3) if len(errors) else None,"historical_closing_line_wins":w,"losses":l,"pushes":grades.count("PUSH"),"hit_rate":round(w/(w+l),4) if w+l else None}
    return {"season":season,"status":"research_only","warning":"Closing-line research grades are not issued strategy positions. Scoring proxy is NOT a possession model; drive-level data required.","models":result,"rows":rows}

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--season",type=int,default=2026);p.add_argument("--output",default="data/nfl_totals_shadow_research.json");a=p.parse_args()
    df=pd.read_csv(a.input,low_memory=False)
    for col in ("season","week","home_score","away_score","total_line"):
        df[col]=pd.to_numeric(df[col],errors="coerce")
    out=run(df,a.season);dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps(out["models"],indent=2))
if __name__=="__main__":main()
