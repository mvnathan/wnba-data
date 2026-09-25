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
NFL_DIVISIONS={
    "BUF":"AFC_E","MIA":"AFC_E","NE":"AFC_E","NYJ":"AFC_E",
    "BAL":"AFC_N","CIN":"AFC_N","CLE":"AFC_N","PIT":"AFC_N",
    "HOU":"AFC_S","IND":"AFC_S","JAX":"AFC_S","TEN":"AFC_S",
    "DEN":"AFC_W","KC":"AFC_W","LV":"AFC_W","LAC":"AFC_W",
    "DAL":"NFC_E","NYG":"NFC_E","PHI":"NFC_E","WAS":"NFC_E",
    "CHI":"NFC_N","DET":"NFC_N","GB":"NFC_N","MIN":"NFC_N",
    "ATL":"NFC_S","CAR":"NFC_S","NO":"NFC_S","TB":"NFC_S",
    "ARI":"NFC_W","LAR":"NFC_W","LA":"NFC_W","SF":"NFC_W","SEA":"NFC_W",
}

MLB_AL={"BAL","BOS","NYY","TB","TOR","CWS","CLE","DET","KC","MIN","HOU","LAA","ATH","OAK","SEA","TEX"}
MLB_NL={"ATL","MIA","NYM","PHI","WSH","CHC","CIN","MIL","PIT","STL","ARI","COL","LAD","SD","SF"}
MLB_DIVISIONS={
    "BAL":"AL_E","BOS":"AL_E","NYY":"AL_E","TB":"AL_E","TOR":"AL_E",
    "CWS":"AL_C","CLE":"AL_C","DET":"AL_C","KC":"AL_C","MIN":"AL_C",
    "HOU":"AL_W","LAA":"AL_W","ATH":"AL_W","OAK":"AL_W","SEA":"AL_W","TEX":"AL_W",
    "ATL":"NL_E","MIA":"NL_E","NYM":"NL_E","PHI":"NL_E","WSH":"NL_E",
    "CHC":"NL_C","CIN":"NL_C","MIL":"NL_C","PIT":"NL_C","STL":"NL_C",
    "ARI":"NL_W","COL":"NL_W","LAD":"NL_W","SD":"NL_W","SF":"NL_W",
}

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
    wins_needed_to_current_cutoff: float
    wins_of_cushion_over_current_cutoff: float
    urgency_score: float
    rest_rotation_risk: float
    seed_pressure_score: float
    seed_locked_proxy: float
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

            # Urgency peaks around the cutoff late in the season. A team can
            # have a postseason berth effectively secure while still having a
            # meaningful seeding incentive, so "secure" does not automatically
            # imply low effort or high rest risk.
            proximity=max(0.0,1.0-min(1.0,abs(gap)/max(2.0,remain*.35+1)))
            neighbors=[]
            if idx>1: neighbors.append(abs(wins-items[idx-2][1]["wins"]))
            if idx<len(items): neighbors.append(abs(wins-items[idx][1]["wins"]))
            seed_gap=min(neighbors) if neighbors else remain+1.0
            seed_pressure=late*max(0.0,1.0-min(1.0,seed_gap/max(2.0,remain*.30+1)))
            seed_locked=float(effectively_secure and seed_gap>max(2.0,remain*.45))
            urgency=max(0.0,min(1.0,late*(.35+.65*proximity)))
            if status=="eliminated":
                urgency*=.25
            elif status=="secure":
                urgency=max(urgency*.25,seed_pressure*.65)

            # High rotation/rest risk requires more than a likely berth: the
            # model also looks for relatively stable seeding. This avoids
            # treating a clinched-but-still-seeding team like a no-stakes team.
            rest_risk=late*(.70 if seed_locked else .25 if status=="secure" else .55 if status=="eliminated" else .10)
            confidence=max(0.25,min(1.0,games/max(8.0,total_games*.25)))
            out[team]=TeamContext(
                wins=wins,losses=losses,games_played=games,win_pct=pct,games_remaining=remain,
                rank_group=idx,playoff_cutoff_rank=playoff_slots,gap_to_cutoff_wins=gap,
                wins_needed_to_current_cutoff=max(0.0,-gap+(1.0 if idx>playoff_slots else 0.0)),
                wins_of_cushion_over_current_cutoff=max(0.0,gap),
                urgency_score=urgency,rest_rotation_risk=rest_risk,
                seed_pressure_score=seed_pressure,seed_locked_proxy=seed_locked,
                postseason_status=status,context_confidence=confidence,
            )
    return out


def build_division_context(
    rows:list[dict[str,Any]],
    *,
    total_games:int,
    conferences:dict[str,str],
    divisions:dict[str,str],
    wildcard_slots:int=3,
    late_season_threshold:float=.72,
) -> dict[str,TeamContext]:
    """Context for leagues that qualify division winners plus wild cards.

    Tie-breakers are intentionally not guessed. Gap calculations use current
    win totals and should be interpreted as a model context proxy, not an
    official magic-number service.
    """
    rec=_record(rows)
    out={}
    by_conf=defaultdict(list)
    for team,r in rec.items():
        conf=conferences.get(team)
        if conf:
            by_conf[conf].append((team,r))
    for conf,items in by_conf.items():
        div_groups=defaultdict(list)
        for team,r in items:
            div_groups[divisions.get(team,"unknown")].append((team,r))
        div_leaders={}
        div_second={}
        for div,ditems in div_groups.items():
            ordered=sorted(ditems,key=lambda x:(x[1]["wins"]/max(1,x[1]["games"]),x[1]["wins"]),reverse=True)
            if ordered:
                div_leaders[div]=ordered[0][0]
                div_second[div]=ordered[1][1]["wins"] if len(ordered)>1 else ordered[0][1]["wins"]

        leaders=set(div_leaders.values())
        nonleaders=[x for x in items if x[0] not in leaders]
        nonleaders.sort(key=lambda x:(x[1]["wins"]/max(1,x[1]["games"]),x[1]["wins"]),reverse=True)
        wildcard_cutoff_wins=nonleaders[min(wildcard_slots,len(nonleaders))-1][1]["wins"] if nonleaders else 0.0
        wildcards={team for team,_ in nonleaders[:wildcard_slots]}
        overall=sorted(items,key=lambda x:(x[1]["wins"]/max(1,x[1]["games"]),x[1]["wins"]),reverse=True)
        ranks={team:i+1 for i,(team,_) in enumerate(overall)}

        for team,r in items:
            games=r["games"];wins=r["wins"];losses=r["losses"]
            remain=max(0.0,total_games-games);pct=wins/max(1.0,games)
            div=divisions.get(team,"unknown"); leader=div_leaders.get(div)
            leader_wins=rec.get(leader,{}).get("wins",wins)
            division_gap=wins-leader_wins
            wc_gap=wins-wildcard_cutoff_wins
            is_div_leader=team in leaders
            is_wildcard=team in wildcards
            pathway_gap=division_gap if is_div_leader else max(division_gap,wc_gap)
            max_wins=wins+remain
            eliminated=(max_wins<leader_wins and max_wins<wildcard_cutoff_wins)
            secure_div=is_div_leader and (wins-div_second.get(div,wins))>max(2.0,remain*.45)
            secure_wc=is_wildcard and wc_gap>max(2.0,remain*.45)
            secure=secure_div or secure_wc

            progress=games/max(1.0,total_games)
            late=max(0.0,min(1.0,(progress-late_season_threshold)/max(.01,1-late_season_threshold)))
            distance=min(abs(division_gap),abs(wc_gap))
            proximity=max(0.0,1.0-min(1.0,distance/max(2.0,remain*.35+1)))

            rank=ranks.get(team,len(items))
            overall_wins=[x[1]["wins"] for x in overall]
            seed_neighbors=[]
            if rank>1: seed_neighbors.append(abs(wins-overall_wins[rank-2]))
            if rank<len(overall_wins): seed_neighbors.append(abs(wins-overall_wins[rank]))
            seed_gap=min(seed_neighbors) if seed_neighbors else remain+1.0
            seed_pressure=late*max(0.0,1.0-min(1.0,seed_gap/max(2.0,remain*.30+1)))
            seed_locked=float(secure and seed_gap>max(2.0,remain*.45))

            urgency=late*(.35+.65*proximity)
            if eliminated:
                urgency*=.25
            elif secure:
                urgency=max(urgency*.25,seed_pressure*.65)
            rest_risk=late*(.70 if seed_locked else .25 if secure else .55 if eliminated else .10)

            if eliminated: status="eliminated"
            elif secure: status="secure"
            elif is_div_leader: status="division_leader"
            elif is_wildcard: status="wildcard_position"
            else: status="chasing"

            out[team]=TeamContext(
                wins=wins,losses=losses,games_played=games,win_pct=pct,games_remaining=remain,
                rank_group=rank,playoff_cutoff_rank=len(leaders)+wildcard_slots,
                gap_to_cutoff_wins=pathway_gap,
                wins_needed_to_current_cutoff=max(0.0,-pathway_gap+(0.0 if (is_div_leader or is_wildcard) else 1.0)),
                wins_of_cushion_over_current_cutoff=max(0.0,pathway_gap),
                urgency_score=urgency,rest_rotation_risk=rest_risk,
                seed_pressure_score=seed_pressure,seed_locked_proxy=seed_locked,
                postseason_status=status,context_confidence=max(.25,min(1.0,games/max(8.0,total_games*.25))),
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


def nfl_divisions() -> dict[str,str]:
    return dict(NFL_DIVISIONS)


def mlb_divisions() -> dict[str,str]:
    return dict(MLB_DIVISIONS)


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
