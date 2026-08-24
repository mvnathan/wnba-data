#!/usr/bin/env python3
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import PREDICTION_HISTORY_PATH, PREDICTION_LATEST_JSON

DATA_REPO = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
OFFICIALS_TAG = "espn_wnba_officials"
BOX_TAG = "espn_wnba_team_boxscores"
TRAIN_SEASONS = (2024, 2025)
VALID_SEASON = 2026
SHRINK_K = 20.0
TOTAL_WEIGHT = 0.35
MARGIN_WEIGHT = 0.25

# Official assignments published by NBA Official for 2026-08-24.
TODAY_CREWS = {
    "401857171": ["Isaac Barnett", "Amy Bonner", "Sarah Williams"],
    "401857172": ["Clare Simmons", "Ashley Gloss", "Josh Reed"],
}


def _csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))


def _load_season(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    off = _csv(f"{DATA_REPO}/{OFFICIALS_TAG}/officials_{season}.csv")
    box = _csv(f"{DATA_REPO}/{BOX_TAG}/team_box_{season}.csv")
    off["game_id"] = off["game_id"].astype(str)
    box["game_id"] = box["game_id"].astype(str)
    return off, box


def _game_outcomes(box: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game_id, g in box.groupby("game_id"):
        h = g[g["team_home_away"].astype(str).str.lower().eq("home")]
        a = g[g["team_home_away"].astype(str).str.lower().eq("away")]
        if h.empty or a.empty:
            continue
        hs = pd.to_numeric(h.iloc[0].get("team_score"), errors="coerce")
        aws = pd.to_numeric(a.iloc[0].get("team_score"), errors="coerce")
        if pd.isna(hs) or pd.isna(aws):
            continue
        rows.append({"game_id": str(game_id), "actual_total": float(hs + aws), "actual_margin": float(hs - aws)})
    return pd.DataFrame(rows)


def _ref_effects(off: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined = off.merge(outcomes, on="game_id", how="inner")
    baseline_total = outcomes["actual_total"].mean()
    baseline_margin = outcomes["actual_margin"].mean()
    stats = joined.groupby("official_full_name").agg(
        games=("game_id", "nunique"),
        mean_total=("actual_total", "mean"),
        mean_margin=("actual_margin", "mean"),
    ).reset_index()
    stats["shrink"] = stats["games"] / (stats["games"] + SHRINK_K)
    stats["total_effect"] = (stats["mean_total"] - baseline_total) * stats["shrink"]
    stats["margin_effect"] = (stats["mean_margin"] - baseline_margin) * stats["shrink"]
    return stats, {"baseline_total": float(baseline_total), "baseline_margin": float(baseline_margin)}


def _crew_effect(names: list[str], stats: pd.DataFrame) -> dict:
    s = stats.set_index("official_full_name")
    found = s.loc[[n for n in names if n in s.index]] if any(n in s.index for n in names) else pd.DataFrame()
    if found.empty:
        return {"total": 0.0, "margin": 0.0, "matched": 0}
    return {
        "total": float(np.clip(found["total_effect"].mean(), -8.0, 8.0)),
        "margin": float(np.clip(found["margin_effect"].mean(), -6.0, 6.0)),
        "matched": int(len(found)),
    }


def _validation(stats: pd.DataFrame, off26: pd.DataFrame, out26: pd.DataFrame) -> dict:
    path = Path(PREDICTION_HISTORY_PATH)
    if not path.exists():
        return {"available": False, "reason": "prediction_history missing"}
    hist = pd.read_parquet(path)
    if "game_id" not in hist.columns:
        return {"available": False, "reason": "game_id missing from prediction history"}
    hist["game_id"] = hist["game_id"].astype(str)
    crews = off26.groupby("game_id")["official_full_name"].apply(list).to_dict()
    actual = out26.set_index("game_id")
    vals = []
    for _, r in hist.iterrows():
        gid = str(r.get("game_id"))
        if gid not in crews or gid not in actual.index:
            continue
        try:
            pt = float(r.get("predicted_total"))
            pm = float(r.get("predicted_margin"))
        except (TypeError, ValueError):
            continue
        e = _crew_effect(crews[gid], stats)
        at = float(actual.loc[gid, "actual_total"])
        am = float(actual.loc[gid, "actual_margin"])
        vals.append((abs(pt-at), abs((pt + TOTAL_WEIGHT*e["total"])-at), abs(pm-am), abs((pm + MARGIN_WEIGHT*e["margin"])-am)))
    if len(vals) < 10:
        return {"available": False, "n": len(vals), "reason": "too few matched 2026 historical predictions"}
    arr = np.array(vals)
    return {
        "available": True,
        "n": int(len(arr)),
        "total_mae_base": float(arr[:,0].mean()),
        "total_mae_ref": float(arr[:,1].mean()),
        "margin_mae_base": float(arr[:,2].mean()),
        "margin_mae_ref": float(arr[:,3].mean()),
        "total_mae_change": float(arr[:,1].mean()-arr[:,0].mean()),
        "margin_mae_change": float(arr[:,3].mean()-arr[:,2].mean()),
    }


def main() -> None:
    train_off, train_out = [], []
    for season in TRAIN_SEASONS:
        o, b = _load_season(season)
        train_off.append(o)
        train_out.append(_game_outcomes(b))
    off_train = pd.concat(train_off, ignore_index=True)
    out_train = pd.concat(train_out, ignore_index=True)
    stats, baseline = _ref_effects(off_train, out_train)

    try:
        off26, box26 = _load_season(VALID_SEASON)
        out26 = _game_outcomes(box26)
        validation = _validation(stats, off26, out26)
    except Exception as exc:
        validation = {"available": False, "reason": f"2026 validation load failed: {exc}"}

    path = Path(PREDICTION_LATEST_JSON)
    payload = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    for game in payload.get("games", []):
        gid = str(game.get("game_id"))
        crew = TODAY_CREWS.get(gid)
        if not crew:
            continue
        e = _crew_effect(crew, stats)
        total_adj = TOTAL_WEIGHT * e["total"]
        margin_adj = MARGIN_WEIGHT * e["margin"]
        base_total = float(game.get("predicted_total"))
        base_margin = float(game.get("predicted_margin"))
        ref_total = base_total + total_adj
        ref_margin = base_margin + margin_adj
        game.update({
            "referee_experiment": True,
            "referee_crew": crew,
            "referee_source": "NBA Official assignments 2026-08-24",
            "referee_history_source": "sportsdataverse WNBA officials + team boxscores 2024-2025",
            "referee_matched_count": e["matched"],
            "referee_total_effect_raw": e["total"],
            "referee_margin_effect_raw": e["margin"],
            "referee_adjusted_total": ref_total,
            "referee_adjusted_margin": ref_margin,
            "referee_adjusted_home_score": (ref_total + ref_margin)/2,
            "referee_adjusted_away_score": (ref_total - ref_margin)/2,
            "referee_total_delta": total_adj,
            "referee_margin_delta": margin_adj,
            "referee_adjustment_updated_at": now,
        })
        if game.get("next_quarter_period") == 1:
            game["referee_adjusted_q1_total"] = float(game.get("next_quarter_total")) + total_adj/4
            game["referee_adjusted_q1_margin"] = float(game.get("next_quarter_margin")) + margin_adj/4

    payload["referee_experiment_generated_at"] = now
    payload["referee_experiment_validation"] = validation
    payload["referee_experiment_baseline"] = baseline
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    Path("predictions/referee_experiment.json").write_text(json.dumps({"generated_at": now, "validation": validation, "baseline": baseline}, indent=2), encoding="utf-8")
    print(json.dumps({"validation": validation, "games": [g.get("game_id") for g in payload.get("games", []) if g.get("referee_experiment")]}, indent=2))


if __name__ == "__main__":
    main()
