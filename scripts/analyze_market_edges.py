#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

SOURCE = Path("predictions/performance_history.parquet")
OUTPUT = Path("docs/edge_performance.json")
THRESHOLDS = [0.5, 2.0, 4.0, 6.0, 8.0, 10.0]
BREAK_EVEN_MINUS_110 = 110.0 / 210.0
MIN_VALIDATION_BETS = 30
Z_95 = 1.959963984540054


def num(series):
    return pd.to_numeric(series, errors="coerce")


def wilson_interval(wins: int, bets: int) -> tuple[float | None, float | None]:
    if bets <= 0:
        return None, None
    p = wins / bets
    z2 = Z_95 * Z_95
    denom = 1.0 + z2 / bets
    center = (p + z2 / (2.0 * bets)) / denom
    radius = (
        Z_95
        * math.sqrt((p * (1.0 - p) / bets) + (z2 / (4.0 * bets * bets)))
        / denom
    )
    return max(0.0, center - radius), min(1.0, center + radius)


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
        return {
            "threshold": threshold,
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "win_rate_ci95_low": None,
            "win_rate_ci95_high": None,
            "break_even_win_rate_minus_110": BREAK_EVEN_MINUS_110,
            "roi_at_minus_110": None,
            "mean_edge": None,
            "validated": False,
            "validation_reason": "no qualifying historical bets",
        }

    wins = int(((e > 0) == (a > 0)).sum())
    bets = int(len(e))
    losses = int(bets - wins)
    win_rate = wins / bets
    ci_low, ci_high = wilson_interval(wins, bets)

    profit = wins * (100.0 / 110.0) - losses
    validated = bool(
        bets >= MIN_VALIDATION_BETS
        and ci_low is not None
        and ci_low > BREAK_EVEN_MINUS_110
    )
    if bets < MIN_VALIDATION_BETS:
        validation_reason = f"sample below {MIN_VALIDATION_BETS} bets"
    elif ci_low is None or ci_low <= BREAK_EVEN_MINUS_110:
        validation_reason = "95% confidence interval does not clear -110 break-even"
    else:
        validation_reason = "minimum sample met and 95% CI clears -110 break-even"

    return {
        "threshold": threshold,
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "win_rate_ci95_low": ci_low,
        "win_rate_ci95_high": ci_high,
        "break_even_win_rate_minus_110": BREAK_EVEN_MINUS_110,
        "excess_win_rate_vs_minus_110": win_rate - BREAK_EVEN_MINUS_110,
        "roi_at_minus_110": profit / bets,
        "mean_edge": float(e.abs().mean()),
        "validated": validated,
        "validation_reason": validation_reason,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        payload = {
            "available": False,
            "reason": "performance history not found",
            "spread": [],
            "total": [],
        }
    else:
        df = pd.read_parquet(SOURCE)
        required = {
            "predicted_margin",
            "actual_margin",
            "market_home_spread",
            "predicted_total",
            "actual_total",
            "market_total",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            payload = {
                "available": False,
                "reason": f"missing columns: {missing}",
                "spread": [],
                "total": [],
            }
        else:
            spread = [summarize(df, "spread", t) for t in THRESHOLDS]
            total = [summarize(df, "total", t) for t in THRESHOLDS]
            payload = {
                "available": True,
                "method": (
                    "historical pregame model-vs-market disagreement; flat -110 "
                    "reference ROI; Wilson 95% confidence intervals; thresholds are "
                    "research-only until minimum sample and confidence criteria pass"
                ),
                "games": int(len(df)),
                "break_even_win_rate_minus_110": BREAK_EVEN_MINUS_110,
                "minimum_validation_bets": MIN_VALIDATION_BETS,
                "validation_rule": (
                    "validated only when bets >= minimum_validation_bets and the "
                    "Wilson 95% lower confidence bound exceeds -110 break-even"
                ),
                "any_validated_spread_threshold": any(x["validated"] for x in spread),
                "any_validated_total_threshold": any(x["validated"] for x in total),
                "spread": spread,
                "total": total,
            }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
