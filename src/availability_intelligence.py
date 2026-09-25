from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

STATUS_WEIGHTS = {
    "out": 1.00,
    "inactive": 1.00,
    "doubtful": 0.80,
    "questionable": 0.45,
    "probable": 0.15,
    "available": 0.00,
    "active": 0.00,
}

TACTICAL_PATTERNS = {
    "minutes_restriction": r"\b(minutes restriction|limited minutes|minutes limit|restriction)\b",
    "expected_to_start": r"\b(expected to start|will start|starting (?:tonight|today)|in the starting lineup)\b",
    "rest_candidate": r"\b(rest(?:ing|ed)?|maintenance day|scheduled rest|veteran rest)\b",
    "pitch_count_limit": r"\b(pitch count|pitch limit|limited to \d+ pitches)\b",
    "bullpen_unavailable": r"\b(bullpen unavailable|unavailable out of the bullpen|won't be available in relief)\b",
    "snap_limit": r"\b(snap count|snap limit|limited snaps)\b",
    "role_change": r"\b(benched|demoted|promoted|moving to the bench|new starter|named the starter)\b",
}

STATUS_PATTERNS = [
    ("out", r"\b(out|will not play|won't play|ruled out|inactive)\b"),
    ("doubtful", r"\b(doubtful|unlikely to play|not expected to play)\b"),
    ("questionable", r"\b(questionable|game[- ]time decision|status uncertain)\b"),
    ("probable", r"\b(probable|expected to play|likely to play)\b"),
    ("available", r"\b(available|active|cleared to play|will play)\b"),
]


@dataclass
class Evidence:
    sport: str
    team: str | None
    player: str | None
    source: str
    source_tier: str
    published_at_utc: str | None
    fetched_at_utc: str
    text: str
    url: str | None
    status: str | None
    tactical_flags: list[str]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _freshness(published_at_utc: str | None, now: datetime | None = None) -> float:
    if not published_at_utc:
        return 0.65
    now = now or datetime.now(timezone.utc)
    try:
        stamp = datetime.fromisoformat(str(published_at_utc).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        hours = max(0.0, (now - stamp.astimezone(timezone.utc)).total_seconds() / 3600.0)
    except Exception:
        return 0.65
    return math.exp(-math.log(2) * hours / 8.0)


def extract_status(text: str) -> str | None:
    lowered = text.lower()
    for status, pattern in STATUS_PATTERNS:
        if re.search(pattern, lowered):
            return status
    return None


def extract_tactical_flags(text: str) -> list[str]:
    lowered = text.lower()
    return [name for name, pattern in TACTICAL_PATTERNS.items() if re.search(pattern, lowered)]


def evidence_confidence(source_tier: str, published_at_utc: str | None, status: str | None) -> float:
    tier = {"official": 1.0, "team_official": 0.95, "trusted_media": 0.78, "secondary": 0.55}.get(source_tier, 0.45)
    status_factor = 1.0 if status in {"out", "inactive", "available", "active"} else 0.9 if status else 0.75
    return round(max(0.0, min(1.0, tier * _freshness(published_at_utc) * status_factor)), 4)


def make_evidence(
    *,
    sport: str,
    team: str | None,
    player: str | None,
    source: str,
    source_tier: str,
    text: str,
    published_at_utc: str | None = None,
    url: str | None = None,
) -> Evidence:
    status = extract_status(text)
    tactical = extract_tactical_flags(text)
    return Evidence(
        sport=sport,
        team=team,
        player=player,
        source=source,
        source_tier=source_tier,
        published_at_utc=published_at_utc,
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        text=text.strip(),
        url=url,
        status=status,
        tactical_flags=tactical,
        confidence=evidence_confidence(source_tier, published_at_utc, status),
    )


def aggregate_player(evidence: list[Evidence]) -> dict[str, Any]:
    if not evidence:
        return {}
    evidence = sorted(evidence, key=lambda e: e.confidence, reverse=True)
    official = [e for e in evidence if e.source_tier in {"official", "team_official"} and e.status]
    trusted = [e for e in evidence if e.source_tier == "trusted_media" and e.status]
    chosen = official[0] if official else trusted[0] if len(trusted) >= 2 and trusted[0].status == trusted[1].status else evidence[0]
    status = chosen.status
    weight = STATUS_WEIGHTS.get(status or "", 0.0)
    tactical = sorted({flag for e in evidence for flag in e.tactical_flags})
    hard_override = bool(official and status in {"out", "inactive", "available", "active"})
    corroborated = bool(official) or len({e.source for e in trusted if e.status == status}) >= 2
    return {
        "player": chosen.player,
        "team": chosen.team,
        "sport": chosen.sport,
        "status": status,
        "availability_impact": round(weight * chosen.confidence, 4),
        "hard_override": hard_override,
        "corroborated": corroborated,
        "tactical_flags": tactical,
        "best_confidence": chosen.confidence,
        "evidence_count": len(evidence),
        "sources": [e.to_dict() for e in evidence[:8]],
    }


def aggregate(evidence: list[Evidence]) -> dict[str, Any]:
    by_key: dict[tuple[str, str | None, str | None], list[Evidence]] = {}
    for item in evidence:
        key = (item.sport, item.team, item.player)
        by_key.setdefault(key, []).append(item)
    players = [aggregate_player(items) for items in by_key.values()]
    players = [p for p in players if p]
    team_summary: dict[str, dict[str, Any]] = {}
    for p in players:
        team = p.get("team") or "UNKNOWN"
        s = team_summary.setdefault(team, {"players_flagged": 0, "hard_outs": 0, "availability_risk": 0.0, "tactical_flags": []})
        if p.get("status") not in {None, "available", "active"}:
            s["players_flagged"] += 1
        if p.get("hard_override") and p.get("status") in {"out", "inactive"}:
            s["hard_outs"] += 1
        s["availability_risk"] = max(float(s["availability_risk"]), float(p.get("availability_impact") or 0.0))
        s["tactical_flags"] = sorted(set(s["tactical_flags"]) | set(p.get("tactical_flags") or []))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "players": players,
        "teams": team_summary,
        "policy": {
            "official_sources_can_hard_override": True,
            "trusted_media_requires_corroboration_for_hard_use": True,
            "uncorroborated_media_is_context_only": True,
            "market_data_is_not_a_source_of_availability_truth": True,
        },
    }
