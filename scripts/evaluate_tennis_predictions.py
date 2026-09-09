#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score


ESPN_ROOT = "https://site.api.espn.com/apis/site/v2/sports/tennis"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = "docs/tennis-latest.json"
OUTPUT_PATH = ROOT / "docs" / "tennis-performance.json"


def _dt(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _git_snapshots() -> list[dict[str, Any]]:
    commits = subprocess.check_output(
        ["git", "log", "--format=%H", "--", SOURCE_PATH], cwd=ROOT, text=True
    ).split()
    snapshots: list[dict[str, Any]] = []
    for commit in commits:
        try:
            raw = subprocess.check_output(
                ["git", "show", f"{commit}:{SOURCE_PATH}"], cwd=ROOT, text=True
            )
            payload = json.loads(raw)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        generated = _dt(payload.get("generated_at_utc"))
        if generated is None:
            continue
        payload["_commit"] = commit
        payload["_generated"] = generated
        snapshots.append(payload)
    return snapshots


def issued_predictions() -> list[dict[str, Any]]:
    earliest: dict[tuple[str, str], dict[str, Any]] = {}
    for payload in _git_snapshots():
        generated = payload["_generated"]
        for match in payload.get("matches", []):
            start = _dt(match.get("start_time_utc"))
            # Production evaluation must represent information visible pre-match.
            if start is None or generated >= start:
                continue
            key = (str(payload.get("target_date")), str(match.get("match_id")))
            row = {
                **match,
                "prediction_issued_at": generated.isoformat(),
                "target_date": payload.get("target_date"),
                "model_version": payload.get("model_version") or "tennis-v1",
                "prediction_commit": payload["_commit"][:8],
            }
            if key not in earliest or generated < _dt(earliest[key]["prediction_issued_at"]):
                earliest[key] = row
    return sorted(earliest.values(), key=lambda row: (row["target_date"], row["start_time_utc"], row["match_id"]))


def _actuals(dates: set[str]) -> dict[str, dict[str, Any]]:
    actuals: dict[str, dict[str, Any]] = {}
    for day in sorted(dates):
        stamp = day.replace("-", "")
        for tour_slug in ("atp", "wta"):
            response = requests.get(f"{ESPN_ROOT}/{tour_slug}/scoreboard", params={"dates": stamp}, timeout=45)
            response.raise_for_status()
            for event in response.json().get("events", []):
                for grouping in event.get("groupings", []):
                    for competition in grouping.get("competitions", []):
                        if str(competition.get("date", ""))[:10] != day:
                            continue
                        status = ((competition.get("status") or {}).get("type") or {})
                        if not status.get("completed"):
                            continue
                        players = sorted(competition.get("competitors") or [], key=lambda p: p.get("order", 99))
                        if len(players) != 2:
                            continue
                        games = []
                        for player in players:
                            values = [_finite(line.get("value")) for line in player.get("linescores") or []]
                            games.append(sum(value for value in values if value is not None))
                        actuals[str(competition.get("id"))] = {
                            "actual_player_1": (players[0].get("athlete") or {}).get("displayName"),
                            "actual_player_2": (players[1].get("athlete") or {}).get("displayName"),
                            "actual_player_1_games": games[0],
                            "actual_player_2_games": games[1],
                            "actual_margin_player_1": games[0] - games[1],
                            "actual_total_games": games[0] + games[1],
                            "actual_winner_player_1": bool(players[0].get("winner")),
                            "final_status": status.get("description") or "Final",
                        }
    return actuals


def settled_records() -> list[dict[str, Any]]:
    predictions = issued_predictions()
    actuals = _actuals({str(row["target_date"]) for row in predictions})
    records = []
    for prediction in predictions:
        actual = actuals.get(str(prediction.get("match_id")))
        if not actual:
            continue
        probability = _finite(prediction.get("player_1_win_probability"))
        margin = _finite(prediction.get("predicted_game_margin_player_1"))
        total = _finite(prediction.get("predicted_total_games"))
        if probability is None or margin is None or total is None:
            continue
        actual_win = bool(actual["actual_winner_player_1"])
        records.append({
            **{key: prediction.get(key) for key in (
                "match_id", "target_date", "start_time_utc", "tour", "tournament", "round",
                "player_1", "player_2", "player_1_rank", "player_2_rank", "model_version",
                "prediction_issued_at", "prediction_commit",
            )},
            "player_1_win_probability": probability,
            "predicted_winner": prediction.get("predicted_winner"),
            "predicted_game_margin_player_1": margin,
            "predicted_total_games": total,
            **actual,
            "winner_correct": (probability >= 0.5) == actual_win,
            "brier_score": (probability - float(actual_win)) ** 2,
            "log_loss": -(math.log(probability) if actual_win else math.log(1 - probability)),
            "margin_error": margin - actual["actual_margin_player_1"],
            "absolute_margin_error": abs(margin - actual["actual_margin_player_1"]),
            "total_error": total - actual["actual_total_games"],
            "absolute_total_error": abs(total - actual["actual_total_games"]),
        })
    return records


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"matches_evaluated": 0}
    y = np.asarray([float(row["actual_winner_player_1"]) for row in rows])
    p = np.asarray([row["player_1_win_probability"] for row in rows])
    margin_actual = np.asarray([row["actual_margin_player_1"] for row in rows])
    margin_pred = np.asarray([row["predicted_game_margin_player_1"] for row in rows])
    total_actual = np.asarray([row["actual_total_games"] for row in rows])
    total_pred = np.asarray([row["predicted_total_games"] for row in rows])
    return {
        "matches_evaluated": len(rows),
        "winner_accuracy": float(accuracy_score(y, p >= 0.5)),
        "winner_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "winner_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "winner_brier_score": float(brier_score_loss(y, p)),
        "spread_mae_games": float(mean_absolute_error(margin_actual, margin_pred)),
        "spread_bias_games": float(np.mean(margin_pred - margin_actual)),
        "total_mae_games": float(mean_absolute_error(total_actual, total_pred)),
        "total_bias_games": float(np.mean(total_pred - total_actual)),
    }


def build_payload() -> dict[str, Any]:
    records = settled_records()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["tour"]].append(row)
    by_tour = {tour: _summary(rows) for tour, rows in sorted(groups.items())}
    by_day = []
    for day in sorted({row["target_date"] for row in records}):
        day_rows = [row for row in records if row["target_date"] == day]
        by_day.append({"date": day, **_summary(day_rows)})
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "Earliest committed pre-match forecast joined to ESPN final scores by match ID",
        "summary": _summary(records),
        "by_tour": by_tour,
        "by_day": by_day,
        "matches": records,
    }


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "by_tour": payload["by_tour"]}, indent=2))


if __name__ == "__main__":
    main()
