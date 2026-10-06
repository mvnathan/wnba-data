from __future__ import annotations

import io
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ESPN_SCOREBOARD="https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_TEAMS="https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
SD_TEAM="https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_team_boxscores/team_box_{year}.csv"
SD_PLAYER="https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{year}.csv"
BASE_FEATURES=[
"home_off","home_def","home_margin","home_pace","home_win_pct","home_rest",
"away_off","away_def","away_margin","away_pace","away_win_pct","away_rest",
"home_games_played","away_games_played","neutral_site",
"home_ortg","home_drtg","home_efg","home_tov_rate","home_orb_rate","home_ft_rate","home_form_volatility",
"away_ortg","away_drtg","away_efg","away_tov_rate","away_orb_rate","away_ft_rate","away_form_volatility"
]
ROSTER_FEATURES=[
"home_continuity","away_continuity","home_roster_value","away_roster_value",
"home_incoming_share","away_incoming_share","home_top3_share","away_top3_share"
]
FEATURES=BASE_FEATURES+ROSTER_FEATURES

def season_label_from_end(end_year:int)->str:
    return f"{end_year-1}-{str(end_year)[-2:]}"

def _download_csv(url:str, timeout=90)->pd.DataFrame:
    r=requests.get(url,timeout=timeout,headers={"User-Agent":"SportsModelHub/1.0"})
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content),low_memory=False)

def fetch_bulk_season(end_year:int, allow_missing=False)->dict[str,pd.DataFrame]:
    try:
        team=_download_csv(SD_TEAM.format(year=end_year))
        player=_download_csv(SD_PLAYER.format(year=end_year))
    except Exception:
        if allow_missing:return {"team":pd.DataFrame(),"player":pd.DataFrame(),"games":pd.DataFrame()}
        raise
    for df in (team,player):
        if not df.empty:
            if "game_date_time" in df.columns:
                df["game_date_time"]=pd.to_datetime(df["game_date_time"],utc=True,errors="coerce")
            if "game_date" in df.columns:
                fallback=pd.to_datetime(df["game_date"],utc=True,errors="coerce")
                if "game_date_time" not in df.columns:df["game_date_time"]=fallback
                else:df["game_date_time"]=df["game_date_time"].fillna(fallback)
            if "season_type" in df.columns:
                st=pd.to_numeric(df["season_type"],errors="coerce")
                df.drop(df.index[st.notna() & (st!=2)],inplace=True)
    return {"team":team,"player":player,"games":team_games(team,end_year)}

def _num(v,default=0.0)->float:
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except Exception:return default

def _minutes(v)->float:
    if v is None:return 0.0
    s=str(v)
    if ":" in s:
        try:
            a,b=s.split(":",1);return float(a)+float(b)/60
        except Exception:return 0.0
    return _num(v)

def _poss(r)->float:
    return max(1.0,_num(r.get("field_goals_attempted"))+0.44*_num(r.get("free_throws_attempted"))-_num(r.get("offensive_rebounds"))+_num(r.get("turnovers")))

def _ratio(a,b,default=0.0)->float:
    d=_num(b)
    return _num(a)/d if d>0 else default

def _team_box_metrics(r,opp)->dict[str,float]:
    fga=max(1.0,_num(r.get("field_goals_attempted")))
    fgm=_num(r.get("field_goals_made"))
    tpm=_num(r.get("three_point_field_goals_made") or r.get("three_point_field_goals_made"))
    poss=_poss(r)
    orb=_num(r.get("offensive_rebounds"))
    opp_drb=_num(opp.get("defensive_rebounds"))
    return {
        "poss":poss,
        "efg":(fgm+0.5*tpm)/fga,
        "tov_rate":_num(r.get("turnovers"))/poss,
        "orb_rate":orb/max(1.0,orb+opp_drb),
        "ft_rate":_num(r.get("free_throws_attempted"))/fga,
    }

def team_games(team_box:pd.DataFrame,end_year:int)->pd.DataFrame:
    if team_box.empty:return pd.DataFrame(columns=["game_id","date","season","home_id","away_id","home_abbr","away_abbr","home_score","away_score","home_poss","away_poss"])
    rows=[]
    for gid,g in team_box.groupby("game_id"):
        if len(g)<2:continue
        h=g[g["team_home_away"].astype(str).str.lower()=="home"] if "team_home_away" in g.columns else pd.DataFrame()
        a=g[g["team_home_away"].astype(str).str.lower()=="away"] if "team_home_away" in g.columns else pd.DataFrame()
        if h.empty or a.empty:continue
        h=h.iloc[0];a=a.iloc[0]
        hm=_team_box_metrics(h,a);am=_team_box_metrics(a,h)
        rows.append({
            "game_id":str(gid),"date":pd.to_datetime(h.get("game_date_time"),utc=True),
            "season":season_label_from_end(end_year),
            "home_id":int(h["team_id"]),"away_id":int(a["team_id"]),
            "home_abbr":str(h.get("team_abbreviation") or ""),"away_abbr":str(a.get("team_abbreviation") or ""),
            "home_score":_num(h.get("team_score")),"away_score":_num(a.get("team_score")),
            "home_poss":hm["poss"],"away_poss":am["poss"],
            "home_efg":hm["efg"],"away_efg":am["efg"],
            "home_tov_rate":hm["tov_rate"],"away_tov_rate":am["tov_rate"],
            "home_orb_rate":hm["orb_rate"],"away_orb_rate":am["orb_rate"],
            "home_ft_rate":hm["ft_rate"],"away_ft_rate":am["ft_rate"],
        })
    return pd.DataFrame(rows).dropna(subset=["date"]).sort_values(["date","game_id"]).reset_index(drop=True)

def player_value(row)->float:
    return (
        .35*_num(row.get("points"))+.72*_num(row.get("rebounds"))+.88*_num(row.get("assists"))
        +1.25*_num(row.get("steals"))+1.25*_num(row.get("blocks"))-.70*_num(row.get("turnovers"))
        +.12*_num(row.get("plus_minus"))
    )

def prior_player_profiles(player_box:pd.DataFrame)->dict[int,dict[str,Any]]:
    out={}
    if player_box.empty:return out
    for pid,g in player_box.groupby("athlete_id"):
        team_minutes=defaultdict(float);total_minutes=0.0;weighted_value=0.0
        for _,r in g.iterrows():
            mins=_minutes(r.get("minutes"))
            total_minutes+=mins
            team_minutes[int(r["team_id"])]+=mins
            weighted_value+=max(1.0,mins)*player_value(r)
        denom=sum(max(1.0,_minutes(r.get("minutes"))) for _,r in g.iterrows())
        out[int(pid)]={
            "name":str(g.iloc[-1].get("athlete_display_name") or ""),
            "minutes":total_minutes,
            "value":weighted_value/max(1.0,denom),
            "primary_team":max(team_minutes,key=team_minutes.get) if team_minutes else None,
        }
    return out

def _rotation_before(player_box:pd.DataFrame,team_id:int,date:pd.Timestamp,lookback_days=60)->pd.DataFrame:
    if player_box.empty:return pd.DataFrame()
    start=date-pd.Timedelta(days=lookback_days)
    return player_box[(pd.to_numeric(player_box["team_id"],errors="coerce")==team_id)&(player_box["game_date_time"]<date)&(player_box["game_date_time"]>=start)].copy()

def roster_features(team_id:int,date:pd.Timestamp,current_players:pd.DataFrame,prior_profiles:dict[int,dict[str,Any]])->dict[str,Any]:
    g=_rotation_before(current_players,team_id,date)
    if g.empty:return {"continuity":.5,"roster_value":0.0,"incoming_share":.5,"top3_share":.5,"players":[]}
    agg=[]
    for pid,p in g.groupby("athlete_id"):
        mins=sum(_minutes(x) for x in p.get("minutes",[]))
        if mins<=0:continue
        vals=[];weights=[]
        for _,r in p.iterrows():
            vals.append(player_value(r));weights.append(max(1.0,_minutes(r.get("minutes"))))
        recent=float(np.average(vals,weights=weights)) if vals else 0.0
        prior=prior_profiles.get(int(pid),{})
        agg.append({"player_id":int(pid),"name":str(p.iloc[-1].get("athlete_display_name") or prior.get("name") or ""),
                    "minutes":mins,"prior_value":float(prior.get("value") or recent),"recent_value":recent,
                    "retained":prior.get("primary_team")==team_id})
    return _rollup_roster(team_id,agg)

def _rollup_roster(team_id:int,players:list[dict[str,Any]])->dict[str,Any]:
    if not players:return {"continuity":.5,"roster_value":0.0,"incoming_share":.5,"top3_share":.5,"players":[]}
    total=sum(max(0.0,float(x.get("minutes") or x.get("prior_minutes") or 0)) for x in players)
    if total<=0:return {"continuity":.5,"roster_value":0.0,"incoming_share":.5,"top3_share":.5,"players":players}
    def mins(x):return max(0.0,float(x.get("minutes") or x.get("prior_minutes") or 0))
    retained=sum(mins(x) for x in players if x.get("retained",x.get("prior_team")==team_id))/total
    value=sum(mins(x)*float(x.get("prior_value") or 0) for x in players)/total
    shares=[mins(x)/total for x in players]
    for x,share in zip(players,shares):x["rotation_share"]=share
    return {"continuity":retained,"roster_value":value,"incoming_share":1-retained,"top3_share":sum(sorted(shares,reverse=True)[:3]),"players":players}

def _team_pre(team_id:int,date:pd.Timestamp,all_games:pd.DataFrame)->dict[str,float]:
    hist=all_games[(all_games["date"]<date)&((all_games["home_id"]==team_id)|(all_games["away_id"]==team_id))].tail(10)
    empty={"off":112.0,"def":112.0,"margin":0.0,"pace":100.0,"win_pct":.5,"rest":4.0,"games":0,
           "ortg":112.0,"drtg":112.0,"efg":.52,"tov_rate":.13,"orb_rate":.25,"ft_rate":.25,"form_volatility":11.0}
    if hist.empty:return empty
    pts=[];pa=[];pace=[];wins=[];ortg=[];drtg=[];efg=[];tov=[];orb=[];ftr=[];margins=[]
    for _,r in hist.iterrows():
        home=int(r["home_id"])==team_id
        pf=float(r["home_score"] if home else r["away_score"]);ag=float(r["away_score"] if home else r["home_score"])
        tposs=float(r["home_poss"] if home else r["away_poss"]);oposs=float(r["away_poss"] if home else r["home_poss"])
        poss=max(1.0,(tposs+oposs)/2)
        pts.append(pf);pa.append(ag);pace.append(poss);wins.append(pf>ag);margins.append(pf-ag)
        ortg.append(100*pf/poss);drtg.append(100*ag/poss)
        prefix="home" if home else "away"
        efg.append(float(r.get(prefix+"_efg",.52)))
        tov.append(float(r.get(prefix+"_tov_rate",.13)))
        orb.append(float(r.get(prefix+"_orb_rate",.25)))
        ftr.append(float(r.get(prefix+"_ft_rate",.25)))
    last=pd.to_datetime(hist.iloc[-1]["date"],utc=True)
    rest=max(0,min(7,(date-last).total_seconds()/86400))
    return {"off":float(np.mean(pts)),"def":float(np.mean(pa)),"margin":float(np.mean(margins)),
            "pace":float(np.mean(pace)),"win_pct":float(np.mean(wins)),"rest":rest,"games":len(hist),
            "ortg":float(np.mean(ortg)),"drtg":float(np.mean(drtg)),"efg":float(np.mean(efg)),
            "tov_rate":float(np.mean(tov)),"orb_rate":float(np.mean(orb)),"ft_rate":float(np.mean(ftr)),
            "form_volatility":float(np.std(margins)) if len(margins)>1 else 11.0}

def feature_row(game:dict[str,Any],games_before:pd.DataFrame,current_players:pd.DataFrame,prior_profiles:dict[int,dict[str,Any]])->tuple[dict[str,float],dict[str,Any]]:
    date=pd.to_datetime(game["date"],utc=True)
    h=_team_pre(int(game["home_id"]),date,games_before);a=_team_pre(int(game["away_id"]),date,games_before)
    hr=roster_features(int(game["home_id"]),date,current_players,prior_profiles);ar=roster_features(int(game["away_id"]),date,current_players,prior_profiles)
    x={"home_off":h["off"],"home_def":h["def"],"home_margin":h["margin"],"home_pace":h["pace"],"home_win_pct":h["win_pct"],"home_rest":h["rest"],
       "away_off":a["off"],"away_def":a["def"],"away_margin":a["margin"],"away_pace":a["pace"],"away_win_pct":a["win_pct"],"away_rest":a["rest"],
       "home_continuity":hr["continuity"],"away_continuity":ar["continuity"],"home_roster_value":hr["roster_value"],"away_roster_value":ar["roster_value"],
       "home_incoming_share":hr["incoming_share"],"away_incoming_share":ar["incoming_share"],"home_top3_share":hr["top3_share"],"away_top3_share":ar["top3_share"],
       "home_games_played":h["games"],"away_games_played":a["games"],"neutral_site":0.0,
       "home_ortg":h["ortg"],"home_drtg":h["drtg"],"home_efg":h["efg"],"home_tov_rate":h["tov_rate"],"home_orb_rate":h["orb_rate"],"home_ft_rate":h["ft_rate"],"home_form_volatility":h["form_volatility"],
       "away_ortg":a["ortg"],"away_drtg":a["drtg"],"away_efg":a["efg"],"away_tov_rate":a["tov_rate"],"away_orb_rate":a["orb_rate"],"away_ft_rate":a["ft_rate"],"away_form_volatility":a["form_volatility"]}
    return x,{"home_roster":hr,"away_roster":ar}

def build_dataset(target_end_years:list[int],context:dict[int,dict[str,pd.DataFrame]])->pd.DataFrame:
    all_games=pd.concat([context[y]["games"] for y in sorted(context)],ignore_index=True).sort_values(["date","game_id"])
    rows=[]
    for end_year in target_end_years:
        prior=end_year-1;current=context[end_year]
        prior_profiles=prior_player_profiles(context[prior]["player"])
        for _,g in current["games"].iterrows():
            x,_=feature_row(g.to_dict(),all_games,current["player"],prior_profiles)
            rows.append({**x,"date":g["date"],"season":g["season"],"game_id":g["game_id"],"home_id":g["home_id"],"away_id":g["away_id"],
                         "home_abbr":g["home_abbr"],"away_abbr":g["away_abbr"],"home_score":g["home_score"],"away_score":g["away_score"],
                         "home_win":float(g["home_score"]>g["away_score"]),"margin":float(g["home_score"]-g["away_score"]),"total":float(g["home_score"]+g["away_score"])})
    return pd.DataFrame(rows).sort_values(["date","game_id"]).reset_index(drop=True)

def _clf_candidates():
    return {
      "logistic":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",LogisticRegression(max_iter=1500,C=.7))]),
      "hist_gb":Pipeline([("impute",SimpleImputer(strategy="median")),("m",HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,l2_regularization=1.0,random_state=42))]),
      "rf":Pipeline([("impute",SimpleImputer(strategy="median")),("m",RandomForestClassifier(n_estimators=500,min_samples_leaf=8,max_features=.75,random_state=42,n_jobs=-1))]),
      "extra_trees":Pipeline([("impute",SimpleImputer(strategy="median")),("m",ExtraTreesClassifier(n_estimators=600,min_samples_leaf=7,max_features=.85,random_state=42,n_jobs=-1))]),
    }

def _reg_candidates():
    return {
      "ridge":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",Ridge(alpha=8.0))]),
      "hist_gb":Pipeline([("impute",SimpleImputer(strategy="median")),("m",HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=15,l2_regularization=2.0,random_state=42))]),
      "rf":Pipeline([("impute",SimpleImputer(strategy="median")),("m",RandomForestRegressor(n_estimators=500,min_samples_leaf=8,max_features=.75,random_state=42,n_jobs=-1))]),
      "extra_trees":Pipeline([("impute",SimpleImputer(strategy="median")),("m",ExtraTreesRegressor(n_estimators=600,min_samples_leaf=7,max_features=.85,random_state=42,n_jobs=-1))]),
    }

def train_evaluate(dataset:pd.DataFrame,feature_columns:list[str]|None=None)->tuple[dict[str,Any],dict[str,Any]]:
    feature_columns=feature_columns or FEATURES
    n=len(dataset);a=max(1,int(n*.65));b=max(a+1,int(n*.80))
    train=dataset.iloc[:a];val=dataset.iloc[a:b];test=dataset.iloc[b:]
    Xtr=train[feature_columns];Xv=val[feature_columns];Xt=test[feature_columns]
    selected={};validation={}
    best=None;score=1e9
    for name,m in _clf_candidates().items():
        m.fit(Xtr,train["home_win"]);p=m.predict_proba(Xv)[:,1];s=float(brier_score_loss(val["home_win"],p))
        validation["winner_"+name]={"brier":s,"accuracy":float(accuracy_score(val["home_win"],p>=.5))}
        if s<score:score=s;best=(name,m)
    selected["winner"]=best[0]
    for target in ("margin","total"):
        best=None;score=1e9
        for name,m in _reg_candidates().items():
            m.fit(Xtr,train[target]);pred=m.predict(Xv);s=float(mean_absolute_error(val[target],pred));validation[target+"_"+name]={"mae":s}
            if s<score:score=s;best=(name,m)
        selected[target]=best[0]
    fit=dataset.iloc[:b];Xfit=fit[feature_columns]
    clf=_clf_candidates()[selected["winner"]];clf.fit(Xfit,fit["home_win"])
    regs={}
    for target in ("margin","total"):
        m=_reg_candidates()[selected[target]];m.fit(Xfit,fit[target]);regs[target]=m
    p=clf.predict_proba(Xt)[:,1];pm=regs["margin"].predict(Xt);pt=regs["total"].predict(Xt)
    test_metrics={"n":len(test),"winner_accuracy":float(accuracy_score(test["home_win"],p>=.5)),"brier_score":float(brier_score_loss(test["home_win"],p)),
                  "margin_mae":float(mean_absolute_error(test["margin"],pm)),"total_mae":float(mean_absolute_error(test["total"],pt)),
                  "test_start":str(test.iloc[0]["date"]) if len(test) else None,"test_end":str(test.iloc[-1]["date"]) if len(test) else None}
    test_predictions=[{"date":pd.to_datetime(r["date"],utc=True).isoformat(),"game_id":str(r["game_id"]),"home_abbr":str(r["home_abbr"]),"away_abbr":str(r["away_abbr"]),
                       "home_win_probability":float(p[i]),"predicted_margin":float(pm[i]),"actual_margin":float(r["margin"]),
                       "predicted_total":float(pt[i]),"actual_total":float(r["total"]),"winner_correct":bool((p[i]>=.5)==bool(r["home_win"]))}
                      for i,(_,r) in enumerate(test.iterrows())]
    prod_clf=_clf_candidates()[selected["winner"]];prod_clf.fit(dataset[feature_columns],dataset["home_win"])
    prod_regs={}
    for target in ("margin","total"):
        m=_reg_candidates()[selected[target]];m.fit(dataset[feature_columns],dataset[target]);prod_regs[target]=m
    return {"winner":prod_clf,"margin":prod_regs["margin"],"total":prod_regs["total"]},{"selected":selected,"validation":validation,"test":test_metrics,"test_predictions":test_predictions,"rows":n,"split":{"train":a,"validation":b-a,"test":n-b},"features":feature_columns}

def current_roster(team_id:int,prior_profiles:dict[int,dict[str,Any]])->list[dict[str,Any]]:
    url=f"{ESPN_TEAMS}/{team_id}/roster"
    try:
        r=requests.get(url,timeout=30);r.raise_for_status();data=r.json()
    except Exception:return []
    athletes=[]
    raw=data.get("athletes") or []
    if isinstance(raw,list):
        for group in raw:
            if isinstance(group,dict) and isinstance(group.get("items"),list):athletes.extend(group["items"])
            elif isinstance(group,dict):athletes.append(group)
    out=[]
    for a in athletes:
        pid=a.get("id") or (a.get("athlete") or {}).get("id")
        obj=a.get("athlete") if isinstance(a.get("athlete"),dict) else a
        try:pid=int(pid or obj.get("id"))
        except Exception:continue
        pr=prior_profiles.get(pid,{})
        out.append({"player_id":pid,"name":str(obj.get("displayName") or obj.get("fullName") or pr.get("name") or ""),
                    "prior_value":float(pr.get("value") or 0),"prior_minutes":float(pr.get("minutes") or 0),"prior_team":pr.get("primary_team")})
    return out

def roster_features_from_current(team_id:int,roster:list[dict[str,Any]])->dict[str,Any]:
    players=[{**x,"retained":x.get("prior_team")==team_id} for x in roster]
    return _rollup_roster(team_id,players)

def espn_schedule(date_str:str)->list[dict[str,Any]]:
    stamp=date_str.replace("-","")
    r=requests.get(ESPN_SCOREBOARD,params={"dates":stamp},timeout=30);r.raise_for_status()
    out=[]
    for e in r.json().get("events",[]):
        comp=(e.get("competitions") or [{}])[0];cs=comp.get("competitors") or []
        h=next((x for x in cs if x.get("homeAway")=="home"),None);a=next((x for x in cs if x.get("homeAway")=="away"),None)
        if not h or not a:continue
        ht=h.get("team") or {};at=a.get("team") or {}
        try:hid=int(ht.get("id"));aid=int(at.get("id"))
        except Exception:continue
        season_type=(e.get("season") or {}).get("type")
        out.append({"game_id":str(e.get("id")),"date":e.get("date"),"status":((e.get("status") or {}).get("type") or {}).get("description"),
                    "season_type":season_type,"decision_eligible":int(season_type or 0)==2,
                    "home_id":hid,"away_id":aid,"home_abbr":str(ht.get("abbreviation") or ""),"away_abbr":str(at.get("abbreviation") or ""),
                    "home_team":ht.get("displayName"),"away_team":at.get("displayName")})
    return out

def model_points(margin:float,total:float)->tuple[float,float]:
    return (total+margin)/2,(total-margin)/2

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
            best=None
            for sig in signals:
                if sig.get("sport")!="NBA" or name not in str(sig.get("text") or "").lower():continue
                mult=multipliers.get(str(sig.get("status_signal") or "").lower(),0)
                if mult<=0:continue
                importance=min(.9,max(.05,float(player.get("rotation_share") or 0)))
                penalty=min(4.5,.5+6.0*importance)*mult*float(sig.get("confidence") or .6)
                if best is None or penalty>best[0]:best=(penalty,sig)
            if best is None:continue
            penalty,sig=best
            if side=="home":hp-=penalty
            else:ap-=penalty
            applied.append({"team_side":side,"player":player.get("name"),"status":sig.get("status_signal"),"points":round(penalty,2),"source":sig.get("source"),"url":sig.get("url"),"text":sig.get("text")})
    return hp,ap,applied
