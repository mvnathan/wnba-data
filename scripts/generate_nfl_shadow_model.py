#!/usr/bin/env python3
"""Build research-only NFL shadow models from production + external signals.

Outputs three side-by-side variants:
1) production baseline (unchanged)
2) availability shadow (X availability overlay)
3) combined shadow (availability + prospective sharp-pick overlay)

This script NEVER writes predictions/nfl-latest.json or docs/nfl-latest.json.
"""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROD = Path("docs/nfl-latest.json")
AVAIL = Path("docs/availability-latest.json")
SHARP = Path("data/sharp-picks-ledger.json")
SHARP_STATUS = Path("docs/sharp-signal-latest.json")
OUT = Path("docs/nfl-shadow-latest.json")
DATA_OUT = Path("data/nfl-shadow-latest.json")

STATUS_WEIGHT = {
    "out": 1.00,
    "inactive": 1.00,
    "doubtful": 0.80,
    "questionable": 0.45,
    "probable": 0.15,
    "available": 0.00,
    "active": 0.00,
}

POSITION_BASE_POINTS = {
    "QB": 4.5,
    "WR": 1.3,
    "RB": 1.0,
    "TE": 0.9,
    "LT": 1.0,
    "RT": 0.9,
    "T": 0.8,
    "G": 0.6,
    "C": 0.7,
    "OL": 0.7,
    "DE": 1.0,
    "EDGE": 1.0,
    "DT": 0.65,
    "LB": 0.7,
    "CB": 0.8,
    "S": 0.6,
}

OFFENSE = {"QB","WR","RB","TE","LT","RT","T","G","C","OL"}
DEFENSE = {"DE","EDGE","DT","LB","CB","S"}

TEAM_ALIASES = {
    "ARI":["cardinals","arizona"],
    "ATL":["falcons","atlanta"],
    "BAL":["ravens","baltimore"],
    "BUF":["bills","buffalo"],
    "CAR":["panthers","carolina"],
    "CHI":["bears","chicago"],
    "CIN":["bengals","cincinnati"],
    "CLE":["browns","cleveland"],
    "DAL":["cowboys","dallas"],
    "DEN":["broncos","denver"],
    "DET":["lions","detroit"],
    "GB":["packers","green bay"],
    "HOU":["texans","houston"],
    "IND":["colts","indianapolis"],
    "JAX":["jaguars","jags","jacksonville"],
    "KC":["chiefs","kansas city"],
    "LAC":["chargers","los angeles chargers"],
    "LAR":["rams","los angeles rams"],
    "LA":["rams","los angeles rams"],
    "LV":["raiders","las vegas"],
    "MIA":["dolphins","miami"],
    "MIN":["vikings","minnesota"],
    "NE":["patriots","new england"],
    "NO":["saints","new orleans"],
    "NYG":["giants","new york giants"],
    "NYJ":["jets","new york jets"],
    "PHI":["eagles","philadelphia"],
    "PIT":["steelers","pittsburgh"],
    "SEA":["seahawks","seattle"],
    "SF":["49ers","niners","san francisco"],
    "TB":["buccaneers","bucs","tampa bay"],
    "TEN":["titans","tennessee"],
    "WAS":["commanders","washington"],
}

POS_PLAYER_RE = re.compile(
    r"\b(QB|WR|RB|TE|LT|RT|T|G|C|OL|DE|EDGE|DT|LB|CB|S)\s+"
    r"([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,2})\b"
)

def load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def logistic_prob(margin: float) -> float:
    return max(0.01, min(0.99, 1.0/(1.0+math.exp(-margin/7.25))))

def _alias_positions(text: str, abbr: str) -> list[int]:
    lower = text.lower()
    out = []
    for alias in TEAM_ALIASES.get(abbr, []):
        for m in re.finditer(r"(?<![a-z])#?"+re.escape(alias)+r"(?![a-z])", lower):
            out.append(m.start())
    return sorted(out)

def infer_affected_team(text: str, matchup_abbrs: set[str]) -> str | None:
    """Conservative heuristic: use the first explicit team mention unless it is only an opponent phrase."""
    hits = []
    for abbr in matchup_abbrs:
        for pos in _alias_positions(text, abbr):
            hits.append((pos,abbr))
    hits.sort()
    if not hits:
        return None

    first_pos, first_abbr = hits[0]
    prefix = text[max(0,first_pos-18):first_pos].lower()
    # "vs Broncos" / "against Seahawks" are opponent references, not affected team.
    if re.search(r"\b(vs\.?|versus|against)\s+(?:the\s+)?#?$", prefix):
        return None
    return first_abbr

def extract_player_position(text: str) -> tuple[str | None, str | None]:
    m = POS_PLAYER_RE.search(text)
    if m:
        return m.group(2).strip(), m.group(1).upper()

    # A small fallback for common sentence shapes without a position tag.
    m = re.search(r"\b([A-Z][a-z]+\s+[A-Z][A-Za-z'.-]+)\s+(?:has|is|was|will|has only been|now is)\b", text)
    if m:
        return m.group(1).strip(), None
    return None, None

def availability_adjustments(prod: dict[str,Any], avail: dict[str,Any]) -> dict[str,list[dict[str,Any]]]:
    by_game: dict[str,list[dict[str,Any]]] = {}
    games = prod.get("games") or []
    signals = [
        s for s in (avail.get("signals") or [])
        if s.get("sport")=="NFL" and str(s.get("source") or "").startswith("@")
    ]

    for g in games:
        gid=str(g.get("game_id"))
        teams={str(g.get("home_abbr")),str(g.get("away_abbr"))}
        candidates=[]
        for s in signals:
            text=str(s.get("text") or "")
            team=infer_affected_team(text,teams)
            if not team:
                continue
            status=str(s.get("status_signal") or "")
            sw=STATUS_WEIGHT.get(status,0.0)
            if sw<=0:
                continue
            player,pos=extract_player_position(text)
            if not pos:
                # Without a position we cannot assign a defensible point value.
                continue
            base=POSITION_BASE_POINTS.get(pos,0.0)
            if base<=0:
                continue
            confidence=float(s.get("confidence") or 0)
            impact=round(base*sw*confidence,3)
            candidates.append({
                "team":team,
                "player":player or "unknown",
                "position":pos,
                "status":status,
                "confidence":confidence,
                "base_points":base,
                "point_impact":impact,
                "source":s.get("source"),
                "published_at_utc":s.get("published_at_utc"),
                "url":s.get("url"),
                "text":text,
            })

        # Deduplicate repeated reports for the same player/team using strongest impact.
        dedup={}
        for c in candidates:
            key=(c["team"],str(c["player"]).lower())
            if key not in dedup or c["point_impact"]>dedup[key]["point_impact"]:
                dedup[key]=c
        by_game[gid]=sorted(dedup.values(),key=lambda x:x["point_impact"],reverse=True)
    return by_game

def apply_availability(g: dict[str,Any], items: list[dict[str,Any]]) -> dict[str,Any]:
    if not items:
        return {
            "predicted_home_points":g.get("predicted_home_points"),
            "predicted_away_points":g.get("predicted_away_points"),
            "predicted_margin":g.get("predicted_margin"),
            "predicted_total":g.get("predicted_total"),
            "home_win_probability":g.get("home_win_probability"),
            "away_win_probability":g.get("away_win_probability"),
            "predicted_winner":g.get("predicted_winner"),
            "availability_margin_delta":0.0,
            "availability_total_delta":0.0,
            "signals_applied":[],
        }
    hp=float(g["predicted_home_points"])
    ap=float(g["predicted_away_points"])
    h=str(g.get("home_abbr")); a=str(g.get("away_abbr"))
    applied=[]
    for x in items:
        imp=float(x["point_impact"])
        team=x["team"]; pos=x["position"]
        before=(hp,ap)
        if team==h:
            if pos in OFFENSE:
                hp-=imp
            elif pos in DEFENSE:
                ap+=imp*0.75
        elif team==a:
            if pos in OFFENSE:
                ap-=imp
            elif pos in DEFENSE:
                hp+=imp*0.75
        if before!=(hp,ap):
            applied.append(x)
    hp=max(6.0,hp); ap=max(6.0,ap)
    if not applied:
        return {
            "predicted_home_points":g.get("predicted_home_points"),
            "predicted_away_points":g.get("predicted_away_points"),
            "predicted_margin":g.get("predicted_margin"),
            "predicted_total":g.get("predicted_total"),
            "home_win_probability":g.get("home_win_probability"),
            "away_win_probability":g.get("away_win_probability"),
            "predicted_winner":g.get("predicted_winner"),
            "availability_margin_delta":0.0,
            "availability_total_delta":0.0,
            "signals_applied":[],
        }
    margin=hp-ap; total=hp+ap
    p=logistic_prob(margin)
    return {
        "predicted_home_points":round(hp,2),
        "predicted_away_points":round(ap,2),
        "predicted_margin":round(margin,2),
        "predicted_total":round(total,2),
        "home_win_probability":round(p,4),
        "away_win_probability":round(1-p,4),
        "predicted_winner":g.get("home_team") if p>=0.5 else g.get("away_team"),
        "availability_margin_delta":round(margin-float(g["predicted_margin"]),2),
        "availability_total_delta":round(total-float(g["predicted_total"]),2),
        "signals_applied":applied,
    }

def pick_matches_game(pick: dict[str,Any], g: dict[str,Any]) -> tuple[str | None,float]:
    """Return side ('home'/'away'/'over'/'under') and confidence if a normalized pick matches."""
    sel=str(pick.get("selection") or "").lower()
    h=str(g.get("home_abbr") or ""); a=str(g.get("away_abbr") or "")
    hname=str(g.get("home_team") or "").lower(); aname=str(g.get("away_team") or "").lower()
    market=str(pick.get("market") or "")
    if market=="total":
        # A total must identify this matchup/team; never apply a generic O/U to every game.
        team_terms = [hname, aname]
        team_terms += TEAM_ALIASES.get(h,[]) + TEAM_ALIASES.get(a,[])
        if not any(term and term in sel for term in team_terms):
            return None,0.0
        if "over" in sel:return "over",float(pick.get("parse_confidence") or 0.5)
        if "under" in sel:return "under",float(pick.get("parse_confidence") or 0.5)
    if market in {"spread","moneyline"}:
        if h.lower() in sel or any(x in sel for x in TEAM_ALIASES.get(h,[])) or hname in sel:
            return "home",float(pick.get("parse_confidence") or 0.5)
        if a.lower() in sel or any(x in sel for x in TEAM_ALIASES.get(a,[])) or aname in sel:
            return "away",float(pick.get("parse_confidence") or 0.5)
    return None,0.0

def sharp_adjustment(g: dict[str,Any], ledger: dict[str,Any], sharp_status: dict[str,Any]) -> dict[str,Any]:
    # Prospective shadow-only weights. Production account weights remain exactly zero.
    acct_meta=sharp_status.get("accounts") or {}
    applied=[]
    margin_delta=0.0
    total_delta=0.0
    game_dt=str(g.get("game_date_utc") or "")
    for p in ledger.get("picks") or []:
        if p.get("status") not in {"ungraded","graded"}:
            continue
        published=str(p.get("published_at_utc") or "")
        if game_dt and published and published>=game_dt:
            continue
        side,conf=pick_matches_game(p,g)
        if not side:
            continue
        src=str(p.get("source") or "")
        tier=str((acct_meta.get(src) or {}).get("tier") or p.get("account_tier") or "")
        shadow_weight=0.15 if tier=="established_professional" else 0.10
        amount=min(0.75,shadow_weight*conf*4.0)
        if side=="home":margin_delta+=amount
        elif side=="away":margin_delta-=amount
        elif side=="over":total_delta+=amount
        elif side=="under":total_delta-=amount
        applied.append({
            "source":src,
            "market":p.get("market"),
            "selection":p.get("selection"),
            "line":p.get("line"),
            "odds":p.get("odds"),
            "published_at_utc":published,
            "url":p.get("url"),
            "shadow_weight":shadow_weight,
            "applied_points":round(amount,3),
            "direction":side,
        })
    return {"margin_delta":round(margin_delta,3),"total_delta":round(total_delta,3),"signals_applied":applied}

def apply_combined(g:dict[str,Any], availability:dict[str,Any], sharp:dict[str,Any])->dict[str,Any]:
    if not sharp.get("signals_applied"):
        return {
            "predicted_home_points":availability.get("predicted_home_points"),
            "predicted_away_points":availability.get("predicted_away_points"),
            "predicted_margin":availability.get("predicted_margin"),
            "predicted_total":availability.get("predicted_total"),
            "home_win_probability":availability.get("home_win_probability"),
            "away_win_probability":availability.get("away_win_probability"),
            "predicted_winner":availability.get("predicted_winner"),
            "combined_margin_delta_vs_production":availability.get("availability_margin_delta",0.0),
            "combined_total_delta_vs_production":availability.get("availability_total_delta",0.0),
            "sharp_margin_delta":0.0,
            "sharp_total_delta":0.0,
            "sharp_signals_applied":[],
        }
    hp=float(availability["predicted_home_points"])
    ap=float(availability["predicted_away_points"])
    # Margin-only sharp adjustment is split symmetrically across teams.
    md=float(sharp["margin_delta"]); td=float(sharp["total_delta"])
    hp += md/2 + td/2
    ap -= md/2 + td/2
    margin=hp-ap; total=hp+ap; prob=logistic_prob(margin)
    return {
        "predicted_home_points":round(hp,2),
        "predicted_away_points":round(ap,2),
        "predicted_margin":round(margin,2),
        "predicted_total":round(total,2),
        "home_win_probability":round(prob,4),
        "away_win_probability":round(1-prob,4),
        "predicted_winner":g.get("home_team") if prob>=0.5 else g.get("away_team"),
        "combined_margin_delta_vs_production":round(margin-float(g["predicted_margin"]),2),
        "combined_total_delta_vs_production":round(total-float(g["predicted_total"]),2),
        "sharp_margin_delta":sharp["margin_delta"],
        "sharp_total_delta":sharp["total_delta"],
        "sharp_signals_applied":sharp["signals_applied"],
    }

def main()->None:
    prod=load(PROD,{})
    avail=load(AVAIL,{})
    ledger=load(SHARP,{"picks":[]})
    sharp_status=load(SHARP_STATUS,{})
    if prod.get("model_status")!="production":
        raise SystemExit("Production NFL baseline missing or invalid")

    avail_map=availability_adjustments(prod,avail)
    rows=[]
    changed_avail=0; changed_combined=0
    for base in prod.get("games") or []:
        gid=str(base.get("game_id"))
        av=apply_availability(base,avail_map.get(gid,[]))
        sp=sharp_adjustment(base,ledger,sharp_status)
        comb=apply_combined(base,av,sp)
        if abs(float(av["availability_margin_delta"]))>1e-9 or abs(float(av["availability_total_delta"]))>1e-9:
            changed_avail+=1
        if abs(float(comb["combined_margin_delta_vs_production"]))>1e-9 or abs(float(comb["combined_total_delta_vs_production"]))>1e-9:
            changed_combined+=1

        row={
            "game_id":gid,
            "game_date_utc":base.get("game_date_utc"),
            "home_team":base.get("home_team"),
            "away_team":base.get("away_team"),
            "home_abbr":base.get("home_abbr"),
            "away_abbr":base.get("away_abbr"),
            "market_home_spread":base.get("market_home_spread"),
            "market_total":base.get("market_total"),
            "production":{
                "predicted_home_points":base.get("predicted_home_points"),
                "predicted_away_points":base.get("predicted_away_points"),
                "predicted_margin":base.get("predicted_margin"),
                "predicted_total":base.get("predicted_total"),
                "home_win_probability":base.get("home_win_probability"),
                "predicted_winner":base.get("predicted_winner"),
            },
            "availability_shadow":av,
            "combined_shadow":comb,
        }
        rows.append(row)

    payload={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":"ok",
        "production_modified":False,
        "baseline_model_version":prod.get("model_version"),
        "baseline_validation":{
            "n_games":577,
            "winner_accuracy":0.636,
            "brier_score":0.2265,
            "margin_mae":10.419,
            "total_mae":10.421,
            "team_score_mae":7.545,
            "basis":"Existing leakage-safe 2024-2026 hybrid-context backtest in data/nfl_model_comparison.json."
        },
        "evaluation_status":"prospective_shadow_only",
        "note":"External signals were not available historically across the full 577-game backtest, so current deltas are prospective. Accuracy/Brier/MAE comparison begins as these shadow forecasts settle.",
        "summary":{
            "games":len(rows),
            "games_changed_by_availability":changed_avail,
            "games_changed_by_combined":changed_combined,
            "availability_x_signals_available":len([s for s in (avail.get("signals") or []) if s.get("sport")=="NFL" and str(s.get("source") or "").startswith("@")]),
            "sharp_ledger_picks":len(ledger.get("picks") or []),
            "sharp_production_weight":0.0,
        },
        "games":rows,
        "policy":{
            "production_untouched":True,
            "availability_requires_clear_team_and_position":True,
            "duplicate_player_reports_collapsed":True,
            "sharp_weights_are_shadow_only":True,
            "sharp_production_weight_remains_zero":True,
        },
    }
    for p in (OUT,DATA_OUT):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(payload["summary"],indent=2))

if __name__=="__main__":
    main()
