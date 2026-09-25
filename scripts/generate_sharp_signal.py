#!/usr/bin/env python3
"""Collect timestamped public betting picks from vetted X accounts.

Shadow research only: this feed never alters production predictions.
"""
from __future__ import annotations
import json, os, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

CONFIG=Path("config/sharp_signal_accounts.json")
STATE=Path("data/sharp-signal-state.json")
LEDGER=Path("data/sharp-picks-ledger.json")
DOCS=Path("docs/sharp-signal-latest.json")
X_URL="https://api.x.com/2/tweets/search/recent"

ODDS_RE=re.compile(r"(?<!\\d)(?P<odds>[+-](?:1\\d\\d|[2-9]\\d\\d|\\d{4}))(?!\\d)")
STAKE_RE=re.compile(r"\\b(?P<stake>\\d+(?:\\.\\d+)?)\\s*(?:u|unit(?:s)?)\\b",re.I)
SPREAD_RE=re.compile(r"(?P<team>[A-Za-z0-9 .#&'’-]{2,28})\\s*(?P<line>[+-]\\d+(?:\\.5)?)\\b",re.I)
TOTAL_RE=re.compile(r"\\b(?P<side>over|under|o|u)\\s*(?P<line>\\d+(?:\\.5)?)\\b",re.I)
ML_RE=re.compile(r"(?P<team>[A-Za-z0-9 .#&'’-]{2,28})\\s+(?:ml|moneyline)\\b",re.I)

def load(path:Path,default:Any)->Any:
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return default

def save(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")

def query(accounts:list[dict[str,Any]])->str:
    handles=" OR ".join(f"from:{a['username']}" for a in accounts)
    terms=" OR ".join(['"best bet"','"official play"','"my bet"','"I bet"','"play:"','"pick:"','"units"','"moneyline"','" ML"','"over"','"under"'])
    return f"({handles}) ({terms}) -is:retweet lang:en"

def parse_pick(text:str)->dict[str,Any]|None:
    clean=re.sub(r"https?://\\S+","",text).strip()
    low=f" {clean.lower()} "
    retrospective=any(x in low for x in ("✅","❌"," cash "," cashed "," went "," go 1-"," win-win"," result:"))\n    if retrospective:return None\n    explicit=any(x in low for x in ("best bet","official play","my bet","i bet","play:","pick:"," units"," unit","moneyline"," ml "," over "," under "," loves "," betting "))\n    if not explicit:return None\n    om=ODDS_RE.search(clean); odds=int(om.group("odds")) if om else None
    sm=STAKE_RE.search(clean); stake=float(sm.group("stake")) if sm else None
    m=TOTAL_RE.search(clean)\n    if not m:\n        m=ALT_TOTAL_RE.search(clean)\n    if m:\n        side=m.group("side").lower()\n        if side in {"o","ov"}:side="over"\n        if side in {"u","un"}:side="under"\n        event=m.groupdict().get("event") if hasattr(m,"groupdict") else None\n        selection=(str(event).strip(" #:-")+" "+side).strip() if event else side\n        return {"market":"total","selection":selection,"line":float(m.group("line")),"odds":odds,"stake_units":stake,"parse_confidence":0.75 if odds is not None else 0.65}\n    m=ML_RE.search(clean)
    if m:return {"market":"moneyline","selection":m.group("team").strip(" #:-"),"line":None,"odds":odds,"stake_units":stake,"parse_confidence":0.75 if odds is not None else 0.6}
    m=SPREAD_RE.search(clean)
    if m:return {"market":"spread","selection":m.group("team").strip(" #:-"),"line":float(m.group("line")),"odds":odds,"stake_units":stake,"parse_confidence":0.75 if odds is not None else 0.6}
    return None

def main()->None:
    cfg=load(CONFIG,{})
    accounts=cfg.get("accounts") or []
    token=os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")
    if not token:raise SystemExit("X_BEARER_TOKEN not configured")
    state=load(STATE,{})
    ledger=load(LEDGER,{"created_at_utc":datetime.now(timezone.utc).isoformat(),"picks":[]})
    params={"query":query(accounts),"max_results":100,"tweet.fields":"created_at,author_id,text","expansions":"author_id","user.fields":"username,name"}
    if state.get("newest_id"):params["since_id"]=str(state["newest_id"])
    r=requests.get(X_URL,params=params,headers={"Authorization":f"Bearer {token}"},timeout=30)
    if r.status_code in {401,402,403,429}:
        save(DOCS,{"status":"unavailable","http_status":r.status_code,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"production_adjustment_enabled":False})
        return
    r.raise_for_status(); payload=r.json()
    users={str(u["id"]):u for u in ((payload.get("includes") or {}).get("users") or [])}
    cfg_by={a["username"].lower():a for a in accounts}
    seen={str(p.get("post_id")) for p in ledger.get("picks",[])}
    added=[]; candidate_posts=[]; ids=[]
    for post in payload.get("data") or []:
        pid=str(post.get("id"))
        try:ids.append(int(pid))
        except Exception:pass
        if pid in seen:continue
        user=users.get(str(post.get("author_id")),{})
        username=str(user.get("username") or "").lower(); ac=cfg_by.get(username)
        if not ac:continue
        raw_text=str(post.get("text") or "")
        parsed=parse_pick(raw_text)
        if not parsed:
            candidate_posts.append({"post_id":pid,"source":"@"+str(user.get("username") or ac["username"]),"published_at_utc":post.get("created_at"),"text":raw_text,"url":f"https://x.com/{user.get('username')}/status/{pid}"})
            continue
        row={"post_id":pid,"source":"@"+str(user.get("username") or ac["username"]),"account_tier":ac.get("tier","candidate"),"sports":ac.get("sports",[]),"published_at_utc":post.get("created_at"),"captured_at_utc":datetime.now(timezone.utc).isoformat(),"text":post.get("text"),"url":f"https://x.com/{user.get('username')}/status/{pid}",**parsed,"status":"ungraded","closing_line":None,"closing_odds":None,"clv":None,"result":None,"profit_units":None}
        ledger.setdefault("picks",[]).append(row);added.append(row)
    newest=str(max(ids)) if ids else state.get("newest_id")
    save(STATE,{"newest_id":newest,"updated_at_utc":datetime.now(timezone.utc).isoformat()})
    ledger["updated_at_utc"]=datetime.now(timezone.utc).isoformat(); save(LEDGER,ledger)
    stats={}
    for ac in accounts:
        name="@"+ac["username"]; rows=[p for p in ledger.get("picks",[]) if str(p.get("source","")).lower()==name.lower()]
        graded=[p for p in rows if p.get("status")=="graded"]
        profit=sum(float(p.get("profit_units") or 0) for p in graded)
        clv=[float(p["clv"]) for p in graded if p.get("clv") is not None]
        stats[name]={"tier":ac.get("tier","candidate"),"captured_picks":len(rows),"graded_picks":len(graded),"profit_units":round(profit,3),"roi":round(profit/len(graded),4) if graded else None,"mean_clv":round(sum(clv)/len(clv),4) if clv else None,"model_weight":0.0,"promotion_eligible":False}
    out={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"ok","production_adjustment_enabled":False,"new_posts_seen":len(payload.get("data") or []),"new_picks_parsed":len(added),"ledger_size":len(ledger.get("picks",[])),"since_id_used":params.get("since_id"),"newest_id":newest,"accounts":stats,"new_picks":added[:50],"candidate_posts":candidate_posts[:100],"promotion_thresholds":cfg.get("promotion_thresholds"),"promotion_rule":"No account receives nonzero model weight until our prospectively graded ledger meets the configured sample, ROI and CLV thresholds and a leakage-safe shadow comparison improves predictive metrics."}
    save(DOCS,out); print(json.dumps(out,indent=2))

if __name__=="__main__":main()

# Initial candidate recapture trigger: 2026-09-25

# Bootstrap candidate review rerun

# Parser refinement for prospective totals
