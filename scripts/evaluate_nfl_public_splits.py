#!/usr/bin/env python3
"""Leakage-safe research summary for timestamped NFL betting split snapshots.

Input: data/nfl-public-splits-history.json, a list of records with:
game_id, kickoff_utc, captured_at_utc, market (spread|total|moneyline),
selection, ticket_pct, handle_pct, line, odds, outcome (win|loss|push),
and optional source. Only snapshots strictly before kickoff are eligible.
This file must come from a licensed/verified historical source; never
reconstruct historical splits from closing lines.
"""
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SRC=Path("data/nfl-public-splits-history.json")
OUT=Path("data/nfl-public-splits-evaluation.json")

def parse(s):
    d=datetime.fromisoformat(str(s).replace("Z","+00:00"))
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d

def evaluate(rows):
    latest={}
    for r in rows:
        try:
            if parse(r["captured_at_utc"]) >= parse(r["kickoff_utc"]):
                continue
            pct=float(r["ticket_pct"]); hp=float(r["handle_pct"])
            if not 0 <= pct <= 100 or not 0 <= hp <= 100:
                continue
            if r.get("outcome") not in {"win","loss","push"}:continue
            k=(str(r["game_id"]),str(r["market"]),str(r["selection"]))
            if k not in latest or parse(r["captured_at_utc"])>parse(latest[k]["captured_at_utc"]):
                latest[k]=r
        except (KeyError,TypeError,ValueError):
            continue
    buckets=defaultdict(list)
    for r in latest.values():
        ticket=float(r["ticket_pct"]);handle=float(r["handle_pct"])
        buckets["all"].append(r)
        if ticket>=70:buckets["public_70_plus"].append(r)
        if ticket<=30:buckets["fade_public_70_plus"].append(r)
        if ticket<=35 and handle>=50:buckets["ticket_handle_divergence"].append(r)
    stats={}
    for name,group in buckets.items():
        decided=[r for r in group if r["outcome"]!="push"]
        wins=sum(r["outcome"]=="win" for r in decided)
        stats[name]={"n_decided":len(decided),"wins":wins,"win_rate":round(wins/len(decided),4) if decided else None,
                     "sample_sufficient":len(decided)>=100}
    return {"status":"research_only","production_weight":0.0,
            "note":"Descriptive buckets only; estimate incremental value with chronological out-of-sample tests, odds-adjusted returns and CLV before promotion.",
            "unique_market_selections":len(latest),"groups":stats}

def main():
    rows=json.loads(SRC.read_text()) if SRC.exists() else []
    result=evaluate(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
