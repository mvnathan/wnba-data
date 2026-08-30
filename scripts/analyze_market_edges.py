#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("predictions/performance_history.parquet")
OUTPUT = Path("docs/edge_performance.json")
THRESHOLDS = [0.5, 2.0, 4.0, 6.0, 8.0, 10.0]


def num(series):
    return pd.to_numeric(series, errors="coerce")


def summarize(df: pd.DataFrame, market: str, threshold: float) -> dict:
    if market == "spread":
        edge = num(df["predicted_margin"]) + num(df["market_home_spread"])
        actual_edge = num(df["actual_margin"]) + num(df["market_home_spread"])
    else:
        edge = num(df["predicted_total"]) - num(df["market_total"])
        actual_edge = num(df["actual_total"]) - num(df["market_total"])

    valid = edge.notna() & actual_edge.notna() & (edge.abs() >= threshold)
    e = edge[valid]
    a = actual_edge[valid]
    non_push = a.abs() > 1e-9
    e = e[non_push]
    a = a[non_push]

    if len(e) == 0:
        return {"threshold": threshold, "bets": 0, "wins": 0, "losses": 0, "win_rate": None, "roi_at_minus_110": None, "mean_edge": None}

    wins = int(((e > 0) == (a > 0)).sum())
    losses = int(len(e) - wins)
    # Flat one-unit risk at -110: win returns 100/110 units profit, loss -1.
    profit = wins * (100.0 / 110.0) - losses
    return {
        "threshold": threshold,
        "bets": int(len(e)),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(e),
        "roi_at_minus_110": profit / len(e),
        "mean_edge": float(e.abs().mean()),
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        payload = {"available": False, "reason": "performance history not found", "spread": [], "total": []}
    else:
        df = pd.read_parquet(SOURCE)
        required = {"predicted_margin", "actual_margin", "market_home_spread", "predicted_total", "actual_total", "market_total"}
        missing = sorted(required - set(df.columns))
        if missing:
            payload = {"available": False, "reason": f"missing columns: {missing}", "spread": [], "total": []}
        else:
            payload = {
                "available": True,
                "method": "historical pregame model-vs-market disagreement; flat -110 reference ROI; descriptive until sample sizes are sufficient",
                "games": int(len(df)),
                "spread": [summarize(df, "spread", t) for t in THRESHOLDS],
                "total": [summarize(df, "total", t) for t in THRESHOLDS],
            }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
