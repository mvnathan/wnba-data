#!/usr/bin/env python3
"""Audit issued WNBA forecasts without changing the production model."""

from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import mean

SRC = Path("docs/performance.json")
OUT = Path("docs/wnba-model-audit.json")


def avg(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return mean(vals) if vals else None


def rate(rows, key):
    vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    return mean(vals) if vals else None


def metrics(rows):
    return {
        "n": len(rows),
        "winner_accuracy": rate(rows, "winner_correct"),
        "brier_score": avg(rows, "brier_score"),
        "margin_mae": avg(rows, "absolute_margin_error"),
        "total_mae": avg(rows, "absolute_total_error"),
        "home_score_mae": avg(rows, "absolute_home_score_error"),
        "away_score_mae": avg(rows, "absolute_away_score_error"),
        "ats_direction_accuracy": rate(rows, "ats_model_correct"),
        "total_direction_accuracy": rate(rows, "total_model_correct"),
    }


def main():
    data = json.loads(SRC.read_text())
    games = data.get("games") or []
    dated = sorted(games, key=lambda r: str(r.get("game_date_utc") or ""))

    confidence = {"50-55%": [], "55-60%": [], "60-70%": [], "70-80%": [], "80%+": []}
    calibration = {k: {"n": 0, "predicted_home_win_probability": [], "actual_home_win": []} for k in confidence}
    for g in dated:
        p = g.get("home_win_probability")
        if p is None:
            continue
        p = float(p)
        conf = max(p, 1-p)
        if conf < .55: band = "50-55%"
        elif conf < .60: band = "55-60%"
        elif conf < .70: band = "60-70%"
        elif conf < .80: band = "70-80%"
        else: band = "80%+"
        confidence[band].append(g)
        calibration[band]["n"] += 1
        calibration[band]["predicted_home_win_probability"].append(p)
        calibration[band]["actual_home_win"].append(1.0 if g.get("actual_home_win") else 0.0)

    by_month = {}
    by_team = {}
    for g in dated:
        month = str(g.get("game_date_utc") or "")[:7]
        by_month.setdefault(month, []).append(g)
        for team in (g.get("home_abbr"), g.get("away_abbr")):
            if team: by_team.setdefault(team, []).append(g)

    market_games = [g for g in dated if g.get("market_home_spread") is not None or g.get("market_total") is not None]
    recent_n = min(20, len(dated))

    cal_out = {}
    for band, x in calibration.items():
        cal_out[band] = {
            "n": x["n"],
            "mean_predicted_home_win_probability": mean(x["predicted_home_win_probability"]) if x["n"] else None,
            "observed_home_win_rate": mean(x["actual_home_win"]) if x["n"] else None,
        }

    audit = {
        "status": "production_baseline_audit",
        "source": "issued pregame forecasts only",
        "overall": metrics(dated),
        "recent_20": metrics(dated[-recent_n:]) if recent_n else metrics([]),
        "market_coverage": {
            "games_with_market": len(market_games),
            "games_total": len(dated),
            "coverage": len(market_games) / len(dated) if dated else None,
        },
        "by_confidence": {k: metrics(v) for k, v in confidence.items()},
        "calibration_by_confidence": cal_out,
        "by_month": {k: metrics(v) for k, v in sorted(by_month.items())},
        "by_team": {k: metrics(v) for k, v in sorted(by_team.items())},
        "diagnostics": {
            "mean_margin_error": avg(dated, "margin_error"),
            "mean_total_error": avg(dated, "total_error"),
            "q1_total_mae": avg(dated, "absolute_q1_total_error"),
            "q2_total_mae": avg(dated, "absolute_q2_total_error"),
            "q3_total_mae": avg(dated, "absolute_q3_total_error"),
            "q4_total_mae": avg(dated, "absolute_q4_total_error"),
            "first_half_margin_mae": avg(dated, "absolute_first_half_margin_error"),
            "second_half_margin_mae": avg(dated, "absolute_second_half_margin_error"),
        },
        "promotion_rule": "Do not replace production from this audit alone; compare candidate models prospectively and with leakage-safe chronological validation.",
    }
    OUT.write_text(json.dumps(audit, indent=2, allow_nan=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
