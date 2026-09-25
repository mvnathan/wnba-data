#!/usr/bin/env python3
"""Build a source-scored availability/tactical intelligence snapshot.

Sources:
- official NFL/WNBA/MLB availability pages
- X recent-search API when X_BEARER_TOKEN is configured

This layer does NOT directly change model outputs. It creates a structured,
auditable context feed that can be validated historically before promotion.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.availability_intelligence import make_evidence, aggregate

CONFIG = Path("config/availability_sources.json")
DOCS_OUT = Path("docs/availability-latest.json")
DATA_OUT = Path("data/availability-latest.json")
X_URL = "https://api.x.com/2/tweets/search/recent"

UA = {
    "User-Agent": "Mozilla/5.0 (compatible; SportsModelHub/1.0; +https://github.com/mvnathan/wnba-data)"
}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def relevant(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def official_web_evidence(cfg: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    evidence = []
    diagnostics = []
    keywords = cfg.get("keywords") or []
    for src in cfg.get("official_web_sources") or []:
        row = {"source": src["name"], "sport": src["sport"], "url": src["url"], "status": "ok", "snippets": 0}
        try:
            r = requests.get(src["url"], headers=UA, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # Capture table rows, list items and article-like text blocks. Keep
            # the extraction deliberately conservative to reduce noisy context.
            blocks = []
            for tag in soup.find_all(["tr", "li", "p", "h2", "h3", "article"]):
                text = compact(tag.get_text(" ", strip=True))
                if 18 <= len(text) <= 500 and relevant(text, keywords):
                    blocks.append(text)
            seen = set()
            for text in blocks:
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    make_evidence(
                        sport=src["sport"],
                        team=None,
                        player=None,
                        source=src["name"],
                        source_tier=src.get("tier", "official"),
                        text=text,
                        url=src["url"],
                    )
                )
                if len(seen) >= 120:
                    break
            row["snippets"] = len(seen)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        diagnostics.append(row)
    return evidence, diagnostics


def x_query(accounts: list[dict[str, Any]], keywords: list[str]) -> str:
    handles = " OR ".join(f"from:{x['username']}" for x in accounts)
    terms = [
        "out", "inactive", "doubtful", "questionable", "injury", "injured",
        "rest", "resting", "\"expected to play\"", "\"will start\"",
        "\"minutes restriction\"", "\"snap count\"", "\"pitch count\"",
    ]
    return f"({handles}) ({' OR '.join(terms)}) -is:retweet lang:en"


def x_evidence(cfg: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    token = os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        return [], {"status": "disabled", "reason": "X_BEARER_TOKEN not configured"}

    accounts = cfg.get("trusted_x_accounts") or []
    if not accounts:
        return [], {"status": "disabled", "reason": "no trusted X accounts configured"}

    query = x_query(accounts, cfg.get("keywords") or [])
    params = {
        "query": query,
        "max_results": 100,
        "tweet.fields": "created_at,author_id,text",
        "expansions": "author_id",
        "user.fields": "username,name",
    }
    r = requests.get(X_URL, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    users = {str(x["id"]): x for x in ((payload.get("includes") or {}).get("users") or [])}
    by_username = {x["username"].lower(): x for x in accounts}

    evidence = []
    raw_posts = []
    for post in payload.get("data") or []:
        user = users.get(str(post.get("author_id")), {})
        username = str(user.get("username") or "").lower()
        src_cfg = by_username.get(username)
        if not src_cfg:
            continue
        text = compact(post.get("text") or "")
        ev = make_evidence(
            sport=src_cfg["sport"],
            team=None,
            player=None,
            source="@" + str(user.get("username") or src_cfg["username"]),
            source_tier=src_cfg.get("tier", "trusted_media"),
            text=text,
            published_at_utc=post.get("created_at"),
            url=f"https://x.com/{user.get('username')}/status/{post.get('id')}" if user.get("username") else None,
        )
        evidence.append(ev)
        raw_posts.append({
            "sport": src_cfg["sport"],
            "source": ev.source,
            "published_at_utc": ev.published_at_utc,
            "text": text,
            "status_signal": ev.status,
            "tactical_flags": ev.tactical_flags,
            "confidence": ev.confidence,
            "url": ev.url,
        })

    return evidence, {
        "status": "ok",
        "posts": len(raw_posts),
        "query": query,
        "raw_posts": raw_posts[:100],
    }


def raw_signals(evidence: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for ev in evidence:
        if not ev.status and not ev.tactical_flags:
            continue
        rows.append({
            "sport": ev.sport,
            "source": ev.source,
            "source_tier": ev.source_tier,
            "published_at_utc": ev.published_at_utc,
            "fetched_at_utc": ev.fetched_at_utc,
            "status_signal": ev.status,
            "tactical_flags": ev.tactical_flags,
            "confidence": ev.confidence,
            "text": ev.text,
            "url": ev.url,
        })
    return sorted(rows, key=lambda x: (x["confidence"], x["published_at_utc"] or ""), reverse=True)


def main() -> None:
    cfg = load_config()
    official, web_diag = official_web_evidence(cfg)
    x_items, x_diag = x_evidence(cfg)
    all_items = official + x_items

    structured = aggregate([e for e in all_items if e.player or e.team])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "direct_model_adjustment_enabled": False,
        "integration_stage": "context_and_edge_reliability_only",
        "source_policy": {
            "official": "highest trust; eligible for future hard availability override after player mapping validation",
            "team_official": "high trust",
            "trusted_media": "requires corroboration before hard use",
            "secondary": "context only",
        },
        "official_source_diagnostics": web_diag,
        "x_source_diagnostics": {k: v for k, v in x_diag.items() if k != "raw_posts"},
        "signals": raw_signals(all_items)[:250],
        "structured": structured,
        "next_validation_step": "Map signals to players/teams, backfill historical availability events, then test incremental Brier/accuracy/MAE before allowing direct model adjustments.",
    }

    DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False)
    DOCS_OUT.write_text(text, encoding="utf-8")
    DATA_OUT.write_text(text, encoding="utf-8")
    print(json.dumps({
        "official_signals": len(official),
        "x_signals": len(x_items),
        "output_signals": len(payload["signals"]),
        "x_status": x_diag.get("status"),
    }, indent=2))


if __name__ == "__main__":
    main()
