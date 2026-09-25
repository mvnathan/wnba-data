from __future__ import annotations

"""Objective competitive-context features for team-sport models.

This module intentionally avoids guessing at motivation. It converts observable
record, games remaining, postseason proximity, schedule compression and late-
season status into bounded context scores that models can validate historically.
"""

from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any

NFL_AFC={"BUF","MIA","NE","NYJ","BAL","CIN","CLE","PIT","HOU","IND","JAX","TEN","DEN","KC","LV","LAC"}
NFL_NFC={"DAL","NYG","PHI","WAS","CHI","DET","GB","MIN","ATL","CAR","NO","TB","ARI","LAR","LA","SF","SEA"}

MLB_AL={"BAL","BOS","NYY","TB","TOR","CWS","CLE","DET","KC","MIN","HOU","LAA","ATH","OAK","SEA","TEX"}
MLB_NL={"ATL","MIA","NYM","PHI","WSH","CHC","CIN","MIL","PIT","STL","ARI","COL","LAD","SD","SF"}

@dataclass
class TeamContext:
    wins: float
    losses: float
    games_played: float
    win_pct: float
    games_remaining: float
    rank_group: int
    playoff_cutoff_rank: int
    gap_to_cutoff_wins: float
    urgency_score: float
    rest_rotation_risk: float
    postseason_status: str
    context_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record(rows:list[dict[str,Any]]) -> dict[str,dict[str,float]]:
    rec=defaultdict(lambda:{"wins":0.0,"losses":0.0,"games":0.0})
    for g in rows:
        h,a=str(g.get("home") or ""),str(g.get("away") or "")
        hs,as_=g.get("home_score"),g.get("away_score")
        if not h or not a or hs is None or as_ is None:
            continue
        try: hs=float(hs);as_=float(as_)
        except Exception: continue
        rec[h]["games"]+=1;rec[a]["games"]+=1
        if hs>as_: rec[h]["wins"]+=1;rec[a]["losses"]+=1
        elif as_>hs: rec[a]["wins"]+=1;rec[h]["losses"]+=1
    return dict(rec)


def build_context(
    rows:list[dict[str,Any]],
    *,
    total_games:int,
    playoff_slots:int,
    groups:dict[str,str]|None=None,
    late_season_threshold:float=.72,
) -> dict[str,TeamContext]:
    rec=_record(rows)
    if not rec:
        return {}
    grouped=defaultdict(list)
    for team,r in rec.items():
        grp=(groups or {}).get(team,"league")
        grouped[grp].append((team,r))
    out={}
    for grp,items in grouped.items():
        items.sort(key=lambda x:(x[1]["wins"]/max(1,x[1]["games"]),x[1]["wins"]),reverse=True)
        cutoff_idx=min(playoff_slots,len(items))-1
        cutoff_wins=items[cutoff_idx][1]["wins"] if cutoff_idx>=0 else 0.0
        for idx,(team,r) in enumerate(items,1):
            games=r["games"];wins=r["wins"];losses=r["losses"]
            remain=max(0.0,total_games-games)
            pct=wins/max(1.0,games)
            gap=wins-cutoff_wins
            season_progress=games/max(1.0,total_games)
            late=max(0.0,min(1.0,(season_progress-late_season_threshold)/max(.01,1-late_season_threshold)))

            max_wins=wins+remain
            mathematically_out=max_wins<cutoff_wins
            effectively_secure=(idx<=playoff_slots and gap>max(2.0,remain*.45))
            if mathematically_out:
                status="eliminated"
            elif effectively_secure:
                status="secure"
            elif idx<=playoff_slots:
                status="in_position"
            else:
                status="chasing"

            # Urgency peaks around the cutoff late in the season.
            proximity=max(0.0,1.0-min(1.0,abs(gap)/max(2.0,remain*.35+1)))
            urgency=max(0.0,min(1.0,late*(.35+.65*proximity)))
            if status in {"eliminated","secure"}:
                urgency*=.25

            # Rotation/rest risk is an observable-risk proxy, not a claim of intent.
            rest_risk=late*(.70 if status=="secure" else .55 if status=="eliminated" else .10)
            confidence=max(0.25,min(1.0,games/max(8.0,total_games*.25)))
            out[team]=TeamContext(
                wins=wins,losses=losses,games_played=games,win_pct=pct,games_remaining=remain,
                rank_group=idx,playoff_cutoff_rank=playoff_slots,gap_to_cutoff_wins=gap,
                urgency_score=urgency,rest_rotation_risk=rest_risk,
                postseason_status=status,context_confidence=confidence,
            )
    return out


def nfl_groups() -> dict[str,str]:
    out={}
    for t in NFL_AFC: out[t]="AFC"
    for t in NFL_NFC: out[t]="NFC"
    return out


def mlb_groups() -> dict[str,str]:
    out={}
    for t in MLB_AL: out[t]="AL"
    for t in MLB_NL: out[t]="NL"
    return out


def bounded_effort_adjustment(home:TeamContext|None, away:TeamContext|None, scale:float) -> float:
    """Return a small home-minus-away context adjustment.

    Positive favors home. The cap is deliberately conservative because this is
    contextual signal, not direct evidence that a player will sit.
    """
    if not home or not away:
        return 0.0
    hs=home.urgency_score-.65*home.rest_rotation_risk
    aas=away.urgency_score-.65*away.rest_rotation_risk
    raw=(hs-aas)*scale
    return max(-scale,min(scale,raw))
