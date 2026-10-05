#!/usr/bin/env python3
from __future__ import annotations
import json, pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

from src.nba_model import (
    FEATURES,TEAM_ABBR_TO_NAME,season_label,fetch_logs,team_games,build_dataset,train_evaluate,
    prior_player_profiles,current_roster,roster_features_from_current,_team_pre,espn_schedule,
    parse_market,model_points,apply_x_adjustment
)

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/"docs"/"nba-latest.json";PRED=ROOT/"predictions"/"nba-latest.json"
PERF=ROOT/"data"/"nba_model_comparison.json";MODEL=ROOT/"models"/"nba-production.pkl"

def current_start_year(now):
    return now.year if now.month>=7 else now.year-1

def load_availability():
    try:return json.loads((ROOT/"docs"/"availability-latest.json").read_text()).get("signals") or []
    except Exception:return []

def market_events():
    for source in ("draftkings","espn"):
        try:
            if source=="draftkings":
                from src.draftkings_direct import fetch_draftkings_direct
                rows=fetch_draftkings_direct("nba")
            else:
                from src.espn_market_odds import fetch_espn_draftkings
                rows=fetch_espn_draftkings("nba")
            if rows:return rows
        except Exception as exc:print(source,"NBA market unavailable",exc)
    return []

def main():
    now=datetime.now(timezone.utc);current_year=current_start_year(now)
    target=[season_label(current_year-2),season_label(current_year-1)]
    context=[season_label(current_year-3),*target]
    logs={}
    for s in context:
        print("fetch",s)
        tl=fetch_logs(s,False);pl=fetch_logs(s,True)
        logs[s]={"games":team_games(tl,s),"players":pl}
    dataset=build_dataset(target,logs)
    models,report=train_evaluate(dataset)
    PERF.parent.mkdir(parents=True,exist_ok=True)
    PERF.write_text(json.dumps({"generated_at_utc":now.isoformat(),"training_seasons":target,"feature_count":len(FEATURES),"features":FEATURES,**report},indent=2))
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    with MODEL.open("wb") as f:pickle.dump({"models":models,"report":report,"features":FEATURES},f)

    current=season_label(current_year);prior=season_label(current_year-1)
    current_team=fetch_logs(current,False);current_player=fetch_logs(current,True)
    prior_player=logs.get(prior,{}).get("players")
    if prior_player is None or prior_player.empty:prior_player=fetch_logs(prior,True)
    prior_profiles=prior_player_profiles(prior_player)
    current_games=team_games(current_team,current)
    history=pd.concat([logs[prior]["games"],current_games],ignore_index=True).sort_values(["date","game_id"])
    date=now.astimezone().date().isoformat()
    slate=espn_schedule(date)
    if not slate:
        for i in (1,2):
            d=(now+timedelta(days=i)).date().isoformat();slate=espn_schedule(d)
            if slate:date=d;break
    markets=market_events();signals=load_availability()
    games=[]
    for g in slate:
        dt=pd.to_datetime(g["date"],utc=True)
        hbase=_team_pre(g["home_id"],dt,history);abase=_team_pre(g["away_id"],dt,history)
        hr=roster_features_from_current(g["home_id"],current_roster(g["home_id"],current,prior_profiles))
        ar=roster_features_from_current(g["away_id"],current_roster(g["away_id"],current,prior_profiles))
        x={
            "home_off":hbase["off"],"home_def":hbase["def"],"home_margin":hbase["margin"],"home_pace":hbase["pace"],"home_win_pct":hbase["win_pct"],"home_rest":hbase["rest"],
            "away_off":abase["off"],"away_def":abase["def"],"away_margin":abase["margin"],"away_pace":abase["pace"],"away_win_pct":abase["win_pct"],"away_rest":abase["rest"],
            "home_continuity":hr["continuity"],"away_continuity":ar["continuity"],"home_roster_value":hr["roster_value"],"away_roster_value":ar["roster_value"],
            "home_incoming_share":hr["incoming_share"],"away_incoming_share":ar["incoming_share"],"home_top3_share":hr["top3_share"],"away_top3_share":ar["top3_share"],
            "home_games_played":hbase["games"],"away_games_played":abase["games"],"neutral_site":0.0,
        }
        X=pd.DataFrame([x])[FEATURES]
        hp=float(models["winner"].predict_proba(X)[0,1]);margin=float(models["margin"].predict(X)[0]);total=float(models["total"].predict(X)[0])
        ph,pa=model_points(margin,total)
        row={**g,"model_version":"nba-v1-roster-context","home_win_probability":hp,"away_win_probability":1-hp,
             "predicted_margin":margin,"predicted_total":total,"predicted_home_points":ph,"predicted_away_points":pa,
             "predicted_winner":g["home_team"] if hp>=.5 else g["away_team"],
             "roster_context":{"home":{"continuity":hr["continuity"],"incoming_share":hr["incoming_share"],"top3_share":hr["top3_share"]},
                               "away":{"continuity":ar["continuity"],"incoming_share":ar["incoming_share"],"top3_share":ar["top3_share"]}}}
        market=parse_market(markets,g["home_team"],g["away_team"]);row.update(market)
        hp2,ap2,xap=apply_x_adjustment(row,hr,ar,signals)
        if xap:
            row["base_predicted_home_points"]=row["predicted_home_points"];row["base_predicted_away_points"]=row["predicted_away_points"]
            row["predicted_home_points"]=round(hp2,2);row["predicted_away_points"]=round(ap2,2)
            row["predicted_margin"]=round(hp2-ap2,2);row["predicted_total"]=round(hp2+ap2,2)
            scale=7.8;prob=1/(1+__import__("math").exp(-row["predicted_margin"]/scale))
            row["home_win_probability"]=round(prob,4);row["away_win_probability"]=round(1-prob,4);row["predicted_winner"]=g["home_team"] if prob>=.5 else g["away_team"]
        row["x_availability_adjustments"]=xap
        if row.get("market_home_spread") is not None:row["model_market_spread_edge"]=round(row["predicted_margin"]+float(row["market_home_spread"]),2)
        if row.get("market_total") is not None:row["model_market_total_edge"]=round(row["predicted_total"]-float(row["market_total"]),2)
        if row.get("market_home_moneyline") is not None:
            o=float(row["market_home_moneyline"]);mp=(-o/(-o+100)) if o<0 else 100/(o+100);row["model_market_home_win_edge"]=round(row["home_win_probability"]-mp,4)
        games.append(row)
    payload={"sport":"NBA","generated_at_utc":now.isoformat(),"target_date":date,"season":current,"model_version":"nba-v1-roster-context",
             "model_status":"production","market_used_in_prediction":False,"training_seasons":target,"validation":report,"games":games,
             "x_integration":"trusted NBA availability posts matched to current roster players and weighted by prior rotation share"}
    for p in (DOCS,PRED):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(payload,indent=2,allow_nan=False))
    print(json.dumps({"games":len(games),"target_date":date,"test":report["test"],"selected":report["selected"]},indent=2))

if __name__=="__main__":main()
