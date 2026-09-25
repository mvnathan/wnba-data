#!/usr/bin/env python3
"""Generate NFL v2 candidate predictions without replacing production v1."""

from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.nfl_v2_core import predict_hybrid_row
from src.sbr_market_odds import fetch_sbr_draftkings

CHICAGO=ZoneInfo("America/Chicago")
EASTERN=ZoneInfo("America/New_York")
NFLVERSE_GAMES="https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT=Path("predictions/nfl-v2-latest.json")
DOCS_OUT=Path("docs/nfl-v2-latest.json")
TEAM_NAMES={
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers","HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs","LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","LA":"Los Angeles Rams","MIA":"Miami Dolphins","MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks","SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders",
}


def _load() -> pd.DataFrame:
    df=pd.read_csv(NFLVERSE_GAMES,low_memory=False)
    for col in ("season","week","home_score","away_score","home_rest","away_rest","div_game"):
        if col in df.columns: df[col]=pd.to_numeric(df[col],errors="coerce")
    return df.sort_values(["season","week","gameday","gametime"],na_position="last")


def _dt(row: pd.Series) -> datetime:
    day=str(row.get("gameday") or "")
    time=str(row.get("gametime") or "13:00")
    if not time or time=="nan": time="13:00"
    try: local=datetime.fromisoformat(f"{day}T{time}").replace(tzinfo=EASTERN)
    except Exception: local=datetime.fromisoformat(f"{day}T13:00").replace(tzinfo=EASTERN)
    return local.astimezone(timezone.utc)


def _active_week(df: pd.DataFrame,season:int) -> int:
    reg=df[(df["season"]==season)&(df["game_type"]=="REG")]
    remaining=reg[reg["home_score"].isna()|reg["away_score"].isna()]
    return int(remaining["week"].min()) if len(remaining) else 18


def _market_lookup(rows):
    out={}
    for r in rows:
        out.setdefault((str(r.get("home_team") or "").lower(),str(r.get("away_team") or "").lower()),[]).append(r)
    return out


def _closest(rows,start):
    if not rows:return None
    if len(rows)==1:return rows[0]
    target=datetime.fromisoformat(start.replace("Z","+00:00"))
    best=rows[0];delta=float("inf")
    for r in rows:
        try:
            d=abs((datetime.fromisoformat(str(r.get("commence_time") or "").replace("Z","+00:00"))-target).total_seconds())
            if d<delta:best,delta=r,d
        except Exception:pass
    return best


def _extract(row):
    out={"market_bookmaker":None,"market_home_moneyline":None,"market_away_moneyline":None,"market_home_spread":None,"market_total":None,"market_updated_at":None}
    if not row or not row.get("bookmakers"):return out
    b=row["bookmakers"][0];out["market_bookmaker"]=b.get("title") or "DraftKings";out["market_updated_at"]=b.get("last_update")
    for m in b.get("markets") or []:
        for x in m.get("outcomes") or []:
            if m.get("key")=="h2h":
                if x.get("name")==row.get("home_team"):out["market_home_moneyline"]=x.get("price")
                elif x.get("name")==row.get("away_team"):out["market_away_moneyline"]=x.get("price")
            elif m.get("key")=="spreads" and x.get("name")==row.get("home_team"):out["market_home_spread"]=x.get("point")
            elif m.get("key")=="totals" and x.get("name")=="Over":out["market_total"]=x.get("point")
    return out


def build():
    now=datetime.now(CHICAGO);season=now.year;df=_load();week=_active_week(df,season)
    slate=df[(df["season"]==season)&(df["game_type"]=="REG")&(df["week"]==week)&(df["home_score"].isna()|df["away_score"].isna())].copy()
    dates={str(x) for x in slate["gameday"].dropna().tolist()}
    rows=[]
    for ds in sorted(dates):
        try: rows.extend(fetch_sbr_draftkings("nfl",ds,week=week))
        except Exception as exc: print("market",exc)
    markets=_market_lookup(rows)
    games=[]
    for _,row in slate.iterrows():
        pred,strategy=predict_hybrid_row(df,row)
        ha,aa=str(row.home_team),str(row.away_team);hn,an=TEAM_NAMES.get(ha,ha),TEAM_NAMES.get(aa,aa)
        start=_dt(row).isoformat().replace("+00:00","Z")
        market=_extract(_closest(markets.get((hn.lower(),an.lower())),start))
        spread_edge=pred["margin"]+float(market["market_home_spread"]) if market.get("market_home_spread") is not None else None
        total_edge=pred["total"]-float(market["market_total"]) if market.get("market_total") is not None else None
        games.append({
            "game_id":str(row.game_id),"game_date_utc":start,"season":season,"week":week,
            "home_team":hn,"away_team":an,"home_abbr":ha,"away_abbr":aa,
            "venue":None if pd.isna(row.get("stadium")) else str(row.get("stadium")),
            "status":"Scheduled","status_state":"pre",
            "predicted_home_points":round(pred["home_points"],1),"predicted_away_points":round(pred["away_points"],1),
            "predicted_margin":round(pred["margin"],1),"predicted_total":round(pred["total"],1),
            "predicted_winner":hn if pred["home_win_probability"]>=.5 else an,
            "home_win_probability":round(pred["home_win_probability"],4),"away_win_probability":round(1-pred["home_win_probability"],4),
            "model_version":"nfl-v2-candidate","model_status":"candidate_not_promoted","market_used_in_prediction":False,
            "v2_strategy":strategy,
            "v2_features":{"early_season_multi_season_form":True,"rest":strategy=="early_season_enhanced","qb_continuity":strategy=="early_season_enhanced","divisional_margin_shrink":strategy=="early_season_enhanced","neutral_site_hfa":True},
            **market,
            "model_market_spread_edge":round(spread_edge,2) if spread_edge is not None else None,
            "model_market_total_edge":round(total_edge,2) if total_edge is not None else None,
        })
    return {"generated_at_utc":datetime.now(timezone.utc).isoformat(),"target_date":now.date().isoformat(),"season":season,"week":week,"model_version":"nfl-v2-candidate","model_status":"candidate_not_promoted","market_used_in_prediction":False,"market_event_count":sum(1 for g in games if g["market_bookmaker"]),"games":games}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--production",action="store_true")
    args=ap.parse_args()
    p=build()
    if args.production:
        p["model_version"]="nfl-v2"
        p["model_status"]="production"
        p["validation_basis"]="577-game leakage-safe backtest across 2024-2026; hybrid v2 improved winner accuracy, Brier score, margin MAE, total MAE, and team-score MAE vs v1."
        for g in p["games"]:
            g["model_version"]="nfl-v2"
            g["model_status"]="production"
    text=json.dumps(p,indent=2,allow_nan=False)
    OUT.parent.mkdir(parents=True,exist_ok=True);DOCS_OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(text);DOCS_OUT.write_text(text)
    if args.production:
        Path("predictions/nfl-latest.json").write_text(text)
        Path("docs/nfl-latest.json").write_text(text)
    print(json.dumps({"games":len(p["games"]),"markets":p["market_event_count"],"model":p["model_version"],"status":p["model_status"]},indent=2))


if __name__=="__main__":main()
