from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import commonteamroster, leaguegamelog
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ESPN_SCOREBOARD="https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SEASON_TYPE="Regular Season"
TEAM_ABBR_TO_NAME={
"ATL":"Atlanta Hawks","BOS":"Boston Celtics","BKN":"Brooklyn Nets","CHA":"Charlotte Hornets","CHI":"Chicago Bulls",
"CLE":"Cleveland Cavaliers","DAL":"Dallas Mavericks","DEN":"Denver Nuggets","DET":"Detroit Pistons","GSW":"Golden State Warriors",
"HOU":"Houston Rockets","IND":"Indiana Pacers","LAC":"LA Clippers","LAL":"Los Angeles Lakers","MEM":"Memphis Grizzlies",
"MIA":"Miami Heat","MIL":"Milwaukee Bucks","MIN":"Minnesota Timberwolves","NOP":"New Orleans Pelicans","NYK":"New York Knicks",
"OKC":"Oklahoma City Thunder","ORL":"Orlando Magic","PHI":"Philadelphia 76ers","PHX":"Phoenix Suns","POR":"Portland Trail Blazers",
"SAC":"Sacramento Kings","SAS":"San Antonio Spurs","TOR":"Toronto Raptors","UTA":"Utah Jazz","WAS":"Washington Wizards"}
FEATURES=[
"home_off","home_def","home_margin","home_pace","home_win_pct","home_rest",
"away_off","away_def","away_margin","away_pace","away_win_pct","away_rest",
"home_continuity","away_continuity","home_roster_value","away_roster_value",
"home_incoming_share","away_incoming_share","home_top3_share","away_top3_share",
"home_games_played","away_games_played","neutral_site"
]

def season_label(start_year:int)->str:
    return f"{start_year}-{str(start_year+1)[-2:]}"

def _retry(fn, tries=4):
    last=None
    for i in range(tries):
        try:return fn()
        except Exception as exc:
            last=exc
            time.sleep(1.5*(i+1))
    raise last

def fetch_logs(season:str, player=False)->pd.DataFrame:
    def call():
        ep=leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=SEASON_TYPE,
            player_or_team_abbreviation="P" if player else "T",
            timeout=75,
        )
        frames=ep.get_data_frames()
        return frames[0] if frames else pd.DataFrame()
    df=_retry(call)
    if df.empty:return df
    df=df.copy()
    df["GAME_DATE"]=pd.to_datetime(df["GAME_DATE"],utc=True,errors="coerce")
    return df

def _poss(row)->float:
    fga=float(row.get("FGA") or 0); fta=float(row.get("FTA") or 0); oreb=float(row.get("OREB") or 0); tov=float(row.get("TOV") or 0)
    return max(1.0,fga+0.44*fta-oreb+tov)

def team_games(team_logs:pd.DataFrame, season:str)->pd.DataFrame:
    if team_logs.empty:return pd.DataFrame()
    rows=[]
    for gid,g in team_logs.groupby("GAME_ID"):
        if len(g)<2:continue
        home=g[g["MATCHUP"].astype(str).str.contains("vs\.",regex=True,na=False)]
        away=g[g["MATCHUP"].astype(str).str.contains("@",regex=False,na=False)]
        if home.empty or away.empty:continue
        h=home.iloc[0]; a=away.iloc[0]
        rows.append({
            "game_id":str(gid),"date":h["GAME_DATE"],"season":season,
            "home_id":int(h["TEAM_ID"]),"away_id":int(a["TEAM_ID"]),
            "home_abbr":str(h["TEAM_ABBREVIATION"]),"away_abbr":str(a["TEAM_ABBREVIATION"]),
            "home_score":float(h["PTS"]),"away_score":float(a["PTS"]),
            "home_poss":_poss(h),"away_poss":_poss(a),
        })
    return pd.DataFrame(rows).sort_values(["date","game_id"]).reset_index(drop=True)

def player_value(row)->float:
    return (
        .35*float(row.get("PTS") or 0)+.72*float(row.get("REB") or 0)+.88*float(row.get("AST") or 0)
        +1.25*float(row.get("STL") or 0)+1.25*float(row.get("BLK") or 0)-.70*float(row.get("TOV") or 0)
        +.12*float(row.get("PLUS_MINUS") or 0)
    )

def prior_player_profiles(player_logs:pd.DataFrame)->dict[int,dict[str,Any]]:
    out={}
    if player_logs.empty:return out
    for pid,g in player_logs.groupby("PLAYER_ID"):
        mins=pd.to_numeric(g.get("MIN"),errors="coerce").fillna(0)
        team_minutes=defaultdict(float)
        for _,r in g.iterrows(): team_minutes[int(r["TEAM_ID"])]+=float(r.get("MIN") or 0)
        out[int(pid)]={
            "name":str(g.iloc[-1].get("PLAYER_NAME") or ""),
            "minutes":float(mins.sum()),
            "value":float(np.mean([player_value(r) for _,r in g.iterrows()])),
            "primary_team":max(team_minutes,key=team_minutes.get) if team_minutes else None,
        }
    return out

def _rotation_before(player_logs:pd.DataFrame, team_id:int, date:pd.Timestamp, lookback_days=60)->pd.DataFrame:
    if player_logs.empty:return pd.DataFrame()
    start=date-pd.Timedelta(days=lookback_days)
    g=player_logs[(player_logs["TEAM_ID"]==team_id)&(player_logs["GAME_DATE"]<date)&(player_logs["GAME_DATE"]>=start)].copy()
    return g

def roster_features(team_id:int,date:pd.Timestamp,current_players:pd.DataFrame,prior_profiles:dict[int,dict[str,Any]])->dict[str,Any]:
    g=_rotation_before(current_players,team_id,date)
    if g.empty:
        return {"continuity":0.5,"roster_value":0.0,"incoming_share":0.5,"top3_share":0.5,"players":[]}
    agg=[]
    for pid,p in g.groupby("PLAYER_ID"):
        mins=float(pd.to_numeric(p["MIN"],errors="coerce").fillna(0).sum())
        if mins<=0:continue
        recent_value=float(np.average([player_value(r) for _,r in p.iterrows()],weights=np.maximum(1,pd.to_numeric(p["MIN"],errors="coerce").fillna(1))))
        prior=prior_profiles.get(int(pid),{})
        prior_val=float(prior.get("value") or recent_value)
        agg.append({
            "player_id":int(pid),"name":str(p.iloc[-1].get("PLAYER_NAME") or prior.get("name") or ""),
            "minutes":mins,"prior_value":prior_val,"recent_value":recent_value,
            "retained":prior.get("primary_team")==team_id
        })
    if not agg:return {"continuity":0.5,"roster_value":0.0,"incoming_share":0.5,"top3_share":0.5,"players":[]}
    total_m=sum(x["minutes"] for x in agg)
    weighted_prior=sum(x["minutes"]*x["prior_value"] for x in agg)/max(1,total_m)
    retained=sum(x["minutes"] for x in agg if x["retained"])/max(1,total_m)
    incoming=1-retained
    top3=sum(sorted((x["minutes"] for x in agg),reverse=True)[:3])/max(1,total_m)
    for x in agg:x["rotation_share"]=x["minutes"]/max(1,total_m)
    return {"continuity":retained,"roster_value":weighted_prior,"incoming_share":incoming,"top3_share":top3,"players":agg}

def _team_pre(team_id:int,date:pd.Timestamp,all_games:pd.DataFrame)->dict[str,float]:
    hist=all_games[(all_games["date"]<date)&((all_games["home_id"]==team_id)|(all_games["away_id"]==team_id))].tail(10)
    if hist.empty:return {"off":112.0,"def":112.0,"margin":0.0,"pace":100.0,"win_pct":.5,"rest":4.0,"games":0}
    pts=[];pa=[];pace=[];wins=[]
    for _,r in hist.iterrows():
        home=int(r["home_id"])==team_id
        pf=float(r["home_score"] if home else r["away_score"]); ag=float(r["away_score"] if home else r["home_score"])
        poss=(float(r["home_poss"])+float(r["away_poss"]))/2
        pts.append(pf);pa.append(ag);pace.append(poss);wins.append(pf>ag)
    last=pd.to_datetime(hist.iloc[-1]["date"],utc=True)
    rest=max(0,min(7,(date-last).total_seconds()/86400))
    return {"off":float(np.mean(pts)),"def":float(np.mean(pa)),"margin":float(np.mean(np.array(pts)-np.array(pa))),"pace":float(np.mean(pace)),"win_pct":float(np.mean(wins)),"rest":rest,"games":len(hist)}

def feature_row(game:dict[str,Any],games_before:pd.DataFrame,current_players:pd.DataFrame,prior_profiles:dict[int,dict[str,Any]])->tuple[dict[str,float],dict[str,Any]]:
    date=pd.to_datetime(game["date"],utc=True)
    h=_team_pre(int(game["home_id"]),date,games_before); a=_team_pre(int(game["away_id"]),date,games_before)
    hr=roster_features(int(game["home_id"]),date,current_players,prior_profiles)
    ar=roster_features(int(game["away_id"]),date,current_players,prior_profiles)
    x={
        "home_off":h["off"],"home_def":h["def"],"home_margin":h["margin"],"home_pace":h["pace"],"home_win_pct":h["win_pct"],"home_rest":h["rest"],
        "away_off":a["off"],"away_def":a["def"],"away_margin":a["margin"],"away_pace":a["pace"],"away_win_pct":a["win_pct"],"away_rest":a["rest"],
        "home_continuity":hr["continuity"],"away_continuity":ar["continuity"],"home_roster_value":hr["roster_value"],"away_roster_value":ar["roster_value"],
        "home_incoming_share":hr["incoming_share"],"away_incoming_share":ar["incoming_share"],"home_top3_share":hr["top3_share"],"away_top3_share":ar["top3_share"],
        "home_games_played":h["games"],"away_games_played":a["games"],"neutral_site":0.0,
    }
    return x,{"home_roster":hr,"away_roster":ar}

def build_dataset(target_seasons:list[str],context_logs:dict[str,dict[str,pd.DataFrame]])->pd.DataFrame:
    all_games=pd.concat([context_logs[s]["games"] for s in context_logs],ignore_index=True).sort_values(["date","game_id"])
    rows=[]
    labels=sorted(context_logs.keys())
    for season in target_seasons:
        idx=labels.index(season); prior=labels[idx-1]
        current_games=context_logs[season]["games"]; current_players=context_logs[season]["players"]
        prior_profiles=prior_player_profiles(context_logs[prior]["players"])
        for _,g in current_games.iterrows():
            base=g.to_dict()
            x,_=feature_row(base,all_games,current_players,prior_profiles)
            rows.append({
                **x,"date":g["date"],"season":season,"game_id":g["game_id"],
                "home_id":g["home_id"],"away_id":g["away_id"],"home_abbr":g["home_abbr"],"away_abbr":g["away_abbr"],
                "home_score":g["home_score"],"away_score":g["away_score"],
                "home_win":float(g["home_score"]>g["away_score"]),
                "margin":float(g["home_score"]-g["away_score"]),"total":float(g["home_score"]+g["away_score"])
            })
    return pd.DataFrame(rows).sort_values(["date","game_id"]).reset_index(drop=True)

def _clf_candidates():
    return {
      "logistic":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",LogisticRegression(max_iter=1500,C=.7))]),
      "hist_gb":Pipeline([("impute",SimpleImputer(strategy="median")),("m",HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,l2_regularization=1.0,random_state=42))]),
      "rf":Pipeline([("impute",SimpleImputer(strategy="median")),("m",RandomForestClassifier(n_estimators=450,min_samples_leaf=8,max_features=.75,random_state=42,n_jobs=-1))])
    }

def _reg_candidates():
    return {
      "ridge":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",Ridge(alpha=8.0))]),
      "hist_gb":Pipeline([("impute",SimpleImputer(strategy="median")),("m",HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=15,l2_regularization=2.0,random_state=42))]),
      "rf":Pipeline([("impute",SimpleImputer(strategy="median")),("m",RandomForestRegressor(n_estimators=450,min_samples_leaf=8,max_features=.75,random_state=42,n_jobs=-1))])
    }

def train_evaluate(dataset:pd.DataFrame)->tuple[dict[str,Any],dict[str,Any]]:
    n=len(dataset); a=max(1,int(n*.65)); b=max(a+1,int(n*.80))
    train=dataset.iloc[:a]; val=dataset.iloc[a:b]; test=dataset.iloc[b:]
    Xtr=train[FEATURES]; Xv=val[FEATURES]; Xt=test[FEATURES]
    selected={}; validation={}
    best=None;score=1e9
    for name,m in _clf_candidates().items():
        m.fit(Xtr,train["home_win"]); p=m.predict_proba(Xv)[:,1]
        s=float(brier_score_loss(val["home_win"],p)); validation["winner_"+name]={"brier":s,"accuracy":float(accuracy_score(val["home_win"],p>=.5))}
        if s<score:score=s;best=(name,m)
    selected["winner"]=best[0]
    best_reg={}
    for target in ("margin","total"):
        best=None;score=1e9
        for name,m in _reg_candidates().items():
            m.fit(Xtr,train[target]); pred=m.predict(Xv); s=float(mean_absolute_error(val[target],pred))
            validation[target+"_"+name]={"mae":s}
            if s<score:score=s;best=(name,m)
        selected[target]=best[0];best_reg[target]=best[1]
    fit=dataset.iloc[:b]; Xfit=fit[FEATURES]
    clf=_clf_candidates()[selected["winner"]];clf.fit(Xfit,fit["home_win"])
    regs={}
    for target in ("margin","total"):
        m=_reg_candidates()[selected[target]];m.fit(Xfit,fit[target]);regs[target]=m
    p=clf.predict_proba(Xt)[:,1]; pm=regs["margin"].predict(Xt); pt=regs["total"].predict(Xt)
    test_metrics={
      "n":len(test),"winner_accuracy":float(accuracy_score(test["home_win"],p>=.5)),
      "brier_score":float(brier_score_loss(test["home_win"],p)),
      "margin_mae":float(mean_absolute_error(test["margin"],pm)),
      "total_mae":float(mean_absolute_error(test["total"],pt)),
      "test_start":str(test.iloc[0]["date"]) if len(test) else None,"test_end":str(test.iloc[-1]["date"]) if len(test) else None
    }
    test_predictions=[]
    for i,(_,r) in enumerate(test.iterrows()):
        test_predictions.append({
            "date":pd.to_datetime(r["date"],utc=True).isoformat(),
            "game_id":str(r["game_id"]),
            "home_abbr":str(r["home_abbr"]),"away_abbr":str(r["away_abbr"]),
            "home_win_probability":float(p[i]),
            "predicted_margin":float(pm[i]),"actual_margin":float(r["margin"]),
            "predicted_total":float(pt[i]),"actual_total":float(r["total"]),
            "winner_correct":bool((p[i]>=.5)==bool(r["home_win"])),
        })
    # Production models refit on all two-season observations.
    prod_clf=_clf_candidates()[selected["winner"]];prod_clf.fit(dataset[FEATURES],dataset["home_win"])
    prod_regs={}
    for target in ("margin","total"):
        m=_reg_candidates()[selected[target]];m.fit(dataset[FEATURES],dataset[target]);prod_regs[target]=m
    return {"winner":prod_clf,"margin":prod_regs["margin"],"total":prod_regs["total"]},{"selected":selected,"validation":validation,"test":test_metrics,"test_predictions":test_predictions,"rows":n,"split":{"train":a,"validation":b-a,"test":n-b}}

def current_roster(team_id:int,season:str,prior_profiles:dict[int,dict[str,Any]])->list[dict[str,Any]]:
    def call():
        return commonteamroster.CommonTeamRoster(team_id=team_id,season=season,timeout=45).get_data_frames()[0]
    try:df=_retry(call,tries=3)
    except Exception:return []
    out=[]
    for _,r in df.iterrows():
        pid=int(r["PLAYER_ID"]); pr=prior_profiles.get(pid,{})
        out.append({"player_id":pid,"name":str(r.get("PLAYER") or pr.get("name") or ""),"prior_value":float(pr.get("value") or 0),"prior_minutes":float(pr.get("minutes") or 0),"prior_team":pr.get("primary_team")})
    total=sum(max(0,x["prior_minutes"]) for x in out)
    for x in out:x["rotation_share"]=max(0,x["prior_minutes"])/max(1,total)
    return out

def roster_features_from_current(team_id:int,roster:list[dict[str,Any]])->dict[str,Any]:
    if not roster:return {"continuity":.5,"roster_value":0,"incoming_share":.5,"top3_share":.5,"players":[]}
    total=sum(max(1,x["prior_minutes"]) for x in roster)
    retained=sum(max(1,x["prior_minutes"]) for x in roster if x.get("prior_team")==team_id)/total
    val=sum(max(1,x["prior_minutes"])*float(x.get("prior_value") or 0) for x in roster)/total
    shares=[max(1,x["prior_minutes"])/total for x in roster]
    return {"continuity":retained,"roster_value":val,"incoming_share":1-retained,"top3_share":sum(sorted(shares,reverse=True)[:3]),"players":roster}

NBA_TEAM_LOOKUP={str(x["abbreviation"]).upper():int(x["id"]) for x in nba_teams.get_teams()}
ESPN_ABBR_TO_NBA={"GS":"GSW","NY":"NYK","NO":"NOP","SA":"SAS"}

def _nba_team_id(team:dict[str,Any])->int|None:
    ab=str(team.get("abbreviation") or "").upper()
    ab=ESPN_ABBR_TO_NBA.get(ab,ab)
    if ab in NBA_TEAM_LOOKUP:return NBA_TEAM_LOOKUP[ab]
    name=str(team.get("displayName") or "").lower()
    for row in nba_teams.get_teams():
        if str(row.get("full_name") or "").lower()==name:return int(row["id"])
    return None

def espn_schedule(date_str:str)->list[dict[str,Any]]:
    stamp=date_str.replace("-","")
    r=requests.get(ESPN_SCOREBOARD,params={"dates":stamp},timeout=30,headers={"user-agent":"SportsModelHub/1.0"});r.raise_for_status()
    out=[]
    for e in r.json().get("events",[]):
        comp=(e.get("competitions") or [{}])[0]; cs=comp.get("competitors") or []
        h=next((x for x in cs if x.get("homeAway")=="home"),None);a=next((x for x in cs if x.get("homeAway")=="away"),None)
        if not h or not a:continue
        ht=h.get("team") or {};at=a.get("team") or {}
        hid=_nba_team_id(ht);aid=_nba_team_id(at)
        if hid is None or aid is None:continue
        hab=ESPN_ABBR_TO_NBA.get(str(ht.get("abbreviation") or "").upper(),str(ht.get("abbreviation") or "").upper())
        aab=ESPN_ABBR_TO_NBA.get(str(at.get("abbreviation") or "").upper(),str(at.get("abbreviation") or "").upper())
        out.append({"game_id":str(e.get("id")),"date":e.get("date"),"status":((e.get("status") or {}).get("type") or {}).get("description"),
                    "home_id":hid,"away_id":aid,"home_abbr":hab,"away_abbr":aab,
                    "home_team":ht.get("displayName"),"away_team":at.get("displayName")})
    return out

def model_points(margin:float,total:float)->tuple[float,float]:
    return (total+margin)/2,(total-margin)/2

def american_prob(odds):
    try:o=float(odds)
    except:return None
    return (-o/(-o+100)) if o<0 else (100/(o+100))

def parse_market(events:list[dict[str,Any]],home:str,away:str)->dict[str,Any]:
    def norm(x):return "".join(c.lower() for c in str(x or "") if c.isalnum())
    for ev in events:
        if norm(ev.get("home_team"))!=norm(home) or norm(ev.get("away_team"))!=norm(away):continue
        b=(ev.get("bookmakers") or [{}])[0];out={"market_bookmaker":b.get("title"),"market_updated_at":b.get("last_update")}
        for m in b.get("markets") or []:
            if m.get("key")=="h2h":
                for x in m.get("outcomes") or []:
                    if norm(x.get("name"))==norm(home):out["market_home_moneyline"]=x.get("price")
                    elif norm(x.get("name"))==norm(away):out["market_away_moneyline"]=x.get("price")
            elif m.get("key")=="spreads":
                for x in m.get("outcomes") or []:
                    if norm(x.get("name"))==norm(home):out["market_home_spread"]=x.get("point")
                    elif norm(x.get("name"))==norm(away):out["market_away_spread"]=x.get("point")
            elif m.get("key")=="totals":
                for x in m.get("outcomes") or []:
                    if str(x.get("name")).lower()=="over":out["market_total"]=x.get("point")
        return out
    return {}

def apply_x_adjustment(game:dict[str,Any],home_roster:dict[str,Any],away_roster:dict[str,Any],signals:list[dict[str,Any]])->tuple[float,float,list[dict[str,Any]]]:
    hp=float(game["predicted_home_points"]);ap=float(game["predicted_away_points"]);applied=[]
    multipliers={"out":1.0,"inactive":1.0,"doubtful":.75,"questionable":.35,"probable":.1}
    for side,roster in (("home",home_roster),("away",away_roster)):
        for player in roster.get("players") or []:
            name=str(player.get("name") or "").lower()
            if len(name)<5:continue
            for sig in signals:
                if sig.get("sport")!="NBA" or name not in str(sig.get("text") or "").lower():continue
                mult=multipliers.get(str(sig.get("status_signal") or "").lower(),0)
                if mult<=0:continue
                importance=min(.9,max(.05,float(player.get("rotation_share") or 0)))
                penalty=min(4.5,.5+6.0*importance)*mult*float(sig.get("confidence") or .6)
                if side=="home":hp-=penalty
                else:ap-=penalty
                applied.append({"team_side":side,"player":player.get("name"),"status":sig.get("status_signal"),"points":round(penalty,2),"source":sig.get("source"),"url":sig.get("url"),"text":sig.get("text")})
    return hp,ap,applied
