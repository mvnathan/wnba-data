from __future__ import annotations

import io
import json
import math
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score


DATA_ROOT = "https://stats.tennismylife.org/data"
ESPN_ROOT = "https://site.api.espn.com/apis/site/v2/sports/tennis"
FEATURES = [
    "rank_log_diff", "rank_points_diff", "elo_diff", "surface_elo_diff",
    "win_rate_5_diff", "win_rate_20_diff", "game_form_5_diff", "game_form_20_diff",
    "serve_form_diff", "quality_form_diff", "rest_diff", "workload_14_diff", "experience_diff",
    "surface_hard", "surface_clay", "surface_grass", "best_of_five",
]
TOTAL_FEATURE_INDICES = [0, 1, 2, 3, 5, 7, 8, 12, 13, 14, 15, 16]


@dataclass
class CalibratedWinnerModel:
    model: Any
    calibrator: Any

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = np.clip(self.model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        logit = np.log(raw / (1 - raw)).reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(logit)[:, 1]
        return np.column_stack([1 - calibrated, calibrated])


def _symmetrize_pairs(prob: np.ndarray, margin: np.ndarray, total: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob, margin, total = prob.copy(), margin.copy(), total.copy()
    for i in range(0, len(prob) - 1, 2):
        p = (prob[i] + (1 - prob[i + 1])) / 2
        m = (margin[i] - margin[i + 1]) / 2
        t = (total[i] + total[i + 1]) / 2
        prob[i], prob[i + 1] = p, 1 - p
        margin[i], margin[i + 1] = m, -m
        total[i] = total[i + 1] = t
    return prob, margin, total


def _name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _number(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _score_games(score: Any) -> tuple[int, int] | None:
    text = str(score or "").upper()
    if not text or any(flag in text for flag in ("W/O", "DEF", "BYE")):
        return None
    winner = loser = 0
    for token in text.split():
        match = re.match(r"^(\d+)-(\d+)", token)
        if not match:
            continue
        a, b = map(int, match.groups())
        if a > 20 or b > 20:
            continue
        winner += a
        loser += b
    if winner <= loser or winner + loser < 6:
        return None
    return winner, loser


@dataclass
class PlayerState:
    elo: float = 1500.0
    surface_elo: dict[str, float] = field(default_factory=dict)
    outcomes: deque = field(default_factory=lambda: deque(maxlen=20))
    game_diffs: deque = field(default_factory=lambda: deque(maxlen=20))
    serve_points: deque = field(default_factory=lambda: deque(maxlen=20))
    quality_results: deque = field(default_factory=lambda: deque(maxlen=20))
    match_dates: deque = field(default_factory=lambda: deque(maxlen=30))
    matches: int = 0
    rank: float = np.nan
    rank_points: float = np.nan
    display_name: str = ""

    def surface_rating(self, surface: str) -> float:
        return self.surface_elo.get(surface, 1500.0)

    def win_rate(self, window: int = 20) -> float:
        values = list(self.outcomes)[-window:]
        return float(np.mean(values)) if values else 0.5

    def game_form(self, window: int = 20) -> float:
        values = list(self.game_diffs)[-window:]
        return float(np.mean(values)) if values else 0.0

    def serve_form(self) -> float:
        return float(np.mean(self.serve_points)) if self.serve_points else 0.60

    def quality_form(self) -> float:
        return float(np.mean(self.quality_results)) if self.quality_results else 0.0

    def rest_days(self, match_date: datetime | None) -> float:
        if match_date is None or not self.match_dates:
            return 7.0
        return float(np.clip((match_date - self.match_dates[-1]).days, 0, 21))

    def workload(self, match_date: datetime | None, days: int = 14) -> float:
        if match_date is None:
            return 0.0
        return float(sum(0 <= (match_date - played).days <= days for played in self.match_dates))


def _features(a: PlayerState, b: PlayerState, surface: str, best_of: int, match_date: datetime | None = None) -> list[float]:
    rank_a = a.rank if math.isfinite(a.rank) else 300.0
    rank_b = b.rank if math.isfinite(b.rank) else 300.0
    points_a = a.rank_points if math.isfinite(a.rank_points) else 0.0
    points_b = b.rank_points if math.isfinite(b.rank_points) else 0.0
    return [
        math.log1p(rank_b) - math.log1p(rank_a),
        math.log1p(points_a) - math.log1p(points_b),
        (a.elo - b.elo) / 400.0,
        (a.surface_rating(surface) - b.surface_rating(surface)) / 400.0,
        a.win_rate(5) - b.win_rate(5),
        a.win_rate(20) - b.win_rate(20),
        (a.game_form(5) - b.game_form(5)) / 10.0,
        (a.game_form(20) - b.game_form(20)) / 10.0,
        a.serve_form() - b.serve_form(),
        a.quality_form() - b.quality_form(),
        (a.rest_days(match_date) - b.rest_days(match_date)) / 21.0,
        (a.workload(match_date) - b.workload(match_date)) / 6.0,
        math.log1p(a.matches) - math.log1p(b.matches),
        float(surface == "Hard"), float(surface == "Clay"), float(surface == "Grass"),
        float(best_of == 5),
    ]


def _serve_rate(row: pd.Series, prefix: str) -> float:
    svpt = _number(row.get(f"{prefix}_svpt"), 0)
    won = _number(row.get(f"{prefix}_1stWon"), 0) + _number(row.get(f"{prefix}_2ndWon"), 0)
    return won / svpt if svpt > 0 else 0.60


def _update_elo(w: PlayerState, l: PlayerState, surface: str) -> None:
    expected = 1 / (1 + 10 ** ((l.elo - w.elo) / 400))
    change = 28 * (1 - expected)
    w.elo += change
    l.elo -= change
    ws, ls = w.surface_rating(surface), l.surface_rating(surface)
    expected_surface = 1 / (1 + 10 ** ((ls - ws) / 400))
    change_surface = 32 * (1 - expected_surface)
    w.surface_elo[surface] = ws + change_surface
    l.surface_elo[surface] = ls - change_surface


def download_history(tour: str, as_of: date) -> pd.DataFrame:
    suffix = "_wta" if tour == "WTA" else ""
    frames = []
    for year in range(as_of.year - 2, as_of.year + 1):
        url = f"{DATA_ROOT}/{year}{suffix}.csv"
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        frames.append(pd.read_csv(io.BytesIO(response.content), low_memory=False))
    data = pd.concat(frames, ignore_index=True)
    data["match_date"] = pd.to_datetime(data["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    start = pd.Timestamp(as_of - timedelta(days=730))
    end = pd.Timestamp(as_of)
    data = data[(data.match_date >= start) & (data.match_date < end)].copy()
    eligible = (pd.to_numeric(data.winner_rank, errors="coerce") <= 150) | (pd.to_numeric(data.loser_rank, errors="coerce") <= 150)
    return data[eligible].sort_values(["match_date", "tourney_id", "match_num"]).reset_index(drop=True)


def build_training(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[datetime], dict[str, PlayerState]]:
    states: dict[str, PlayerState] = defaultdict(PlayerState)
    x_rows: list[list[float]] = []
    y_win: list[int] = []
    y_margin: list[float] = []
    y_total: list[float] = []
    dates: list[datetime] = []
    for i, row in data.iterrows():
        games = _score_games(row.get("score"))
        if games is None:
            continue
        wg, lg = games
        winner_key, loser_key = _name(row.winner_name), _name(row.loser_name)
        if not winner_key or not loser_key:
            continue
        w, l = states[winner_key], states[loser_key]
        w.display_name, l.display_name = str(row.winner_name), str(row.loser_name)
        w.rank, l.rank = _number(row.winner_rank), _number(row.loser_rank)
        w.rank_points, l.rank_points = _number(row.winner_rank_points), _number(row.loser_rank_points)
        surface = str(row.get("surface") or "Hard").title()
        best_of = int(_number(row.get("best_of"), 3))
        flip = ((i + int(_number(row.get("match_num"), 0))) % 2) == 1
        a, b = (l, w) if flip else (w, l)
        played_at = row.match_date.to_pydatetime()
        feature_row = _features(a, b, surface, best_of, played_at)
        label = 0 if flip else 1
        margin = float(-(wg - lg) if flip else (wg - lg))
        x_rows.extend([feature_row, _features(b, a, surface, best_of, played_at)])
        y_win.extend([label, 1 - label])
        y_margin.extend([margin, -margin])
        y_total.extend([float(wg + lg), float(wg + lg)])
        dates.extend([played_at, played_at])
        expected = 1 / (1 + 10 ** ((l.elo - w.elo) / 400))
        w.outcomes.append(1); l.outcomes.append(0)
        w.game_diffs.append(wg - lg); l.game_diffs.append(lg - wg)
        w.serve_points.append(_serve_rate(row, "w")); l.serve_points.append(_serve_rate(row, "l"))
        w.quality_results.append(1 - expected); l.quality_results.append(-(1 - expected))
        w.match_dates.append(played_at); l.match_dates.append(played_at)
        w.matches += 1; l.matches += 1
        _update_elo(w, l, surface)
    return np.asarray(x_rows), np.asarray(y_win), np.asarray(y_margin), np.asarray(y_total), dates, states


def train_tour(tour: str, as_of: date) -> tuple[dict[str, Any], dict[str, PlayerState], dict[str, Any]]:
    data = download_history(tour, as_of)
    x, y_win, y_margin, y_total, dates, states = build_training(data)
    if len(x) < 300:
        raise RuntimeError(f"Insufficient {tour} training data: {len(x)} matches")
    split = max(2, (int(len(x) * 0.80) // 2) * 2)
    calibration_split = max(2, (int(split * 0.875) // 2) * 2)
    newest = max(dates)
    age_days = np.asarray([(newest - played).days for played in dates], dtype=float)
    sample_weight = np.power(0.5, age_days / 180.0)
    sample_weight = np.clip(sample_weight, 0.12, 1.0)
    x_total = x[:, TOTAL_FEATURE_INDICES]
    classifier = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.055, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    margin_model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=180, learning_rate=0.055, max_leaf_nodes=15, l2_regularization=1.0, random_state=43)
    total_model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=180, learning_rate=0.055, max_leaf_nodes=15, l2_regularization=1.0, random_state=44)
    classifier.fit(x[:calibration_split], y_win[:calibration_split], sample_weight=sample_weight[:calibration_split])
    calibration_raw = np.clip(classifier.predict_proba(x[calibration_split:split])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=1.0, random_state=45).fit(
        np.log(calibration_raw / (1 - calibration_raw)).reshape(-1, 1), y_win[calibration_split:split],
        sample_weight=sample_weight[calibration_split:split],
    )
    calibrated_classifier = CalibratedWinnerModel(classifier, calibrator)
    margin_model.fit(x[:split], y_margin[:split], sample_weight=sample_weight[:split])
    # Total length is structurally steadier than winner strength, so retain the
    # stable long-window feature subset instead of forcing short-term form into it.
    total_model.fit(x_total[:split], y_total[:split])
    prob = calibrated_classifier.predict_proba(x[split:])[:, 1]
    pred_margin = margin_model.predict(x[split:]); pred_total = total_model.predict(x_total[split:])
    prob, pred_margin, pred_total = _symmetrize_pairs(prob, pred_margin, pred_total)
    metrics = {
        "tour": tour, "training_matches": int(len(x) // 2), "holdout_matches": int((len(x) - split) // 2),
        "holdout_start": dates[split].date().isoformat(), "holdout_end": dates[-1].date().isoformat(),
        "winner_accuracy": round(float(accuracy_score(y_win[split:], prob >= 0.5)), 4),
        "winner_auc": round(float(roc_auc_score(y_win[split:], prob)), 4),
        "winner_log_loss": round(float(log_loss(y_win[split:], prob)), 4),
        "winner_brier_score": round(float(brier_score_loss(y_win[split:], prob)), 4),
        "spread_mae_games": round(float(mean_absolute_error(y_margin[split:], pred_margin)), 3),
        "total_mae_games": round(float(mean_absolute_error(y_total[split:], pred_total)), 3),
    }
    # Production fits use every chronologically eligible match after honest holdout scoring.
    classifier.fit(x, y_win, sample_weight=sample_weight)
    margin_model.fit(x, y_margin, sample_weight=sample_weight)
    total_model.fit(x_total, y_total)
    production_classifier = CalibratedWinnerModel(classifier, calibrator)
    return {"winner": production_classifier, "margin": margin_model, "total": total_model, "features": FEATURES, "total_feature_indices": TOTAL_FEATURE_INDICES}, states, metrics


def fetch_schedule(as_of: date) -> list[dict[str, Any]]:
    stamp = as_of.strftime("%Y%m%d")
    seen: set[str] = set()
    matches: list[dict[str, Any]] = []
    for endpoint_tour in ("atp", "wta"):
        response = requests.get(f"{ESPN_ROOT}/{endpoint_tour}/scoreboard", params={"dates": stamp}, timeout=45)
        response.raise_for_status()
        for event in response.json().get("events", []):
            for grouping in event.get("groupings", []):
                slug = grouping.get("grouping", {}).get("slug", "")
                if slug not in {"mens-singles", "womens-singles"}:
                    continue
                tour = "ATP" if slug == "mens-singles" else "WTA"
                for comp in grouping.get("competitions", []):
                    start_text = comp.get("date") or comp.get("startDate")
                    if not start_text or start_text[:10] != as_of.isoformat():
                        continue
                    competitors = comp.get("competitors") or []
                    if len(competitors) != 2 or comp.get("id") in seen:
                        continue
                    seen.add(comp.get("id"))
                    players = sorted(competitors, key=lambda c: c.get("order", 99))
                    matches.append({
                        "match_id": str(comp.get("id")), "tour": tour,
                        "tournament": event.get("name"), "round": (comp.get("round") or {}).get("displayName"),
                        "start_time_utc": start_text, "status": ((comp.get("status") or {}).get("type") or {}).get("description"),
                        "status_state": ((comp.get("status") or {}).get("type") or {}).get("state"),
                        "court": (comp.get("venue") or {}).get("court"), "venue": (comp.get("venue") or {}).get("fullName"),
                        "broadcast": comp.get("broadcast"), "best_of": int(((comp.get("format") or {}).get("regulation") or {}).get("periods") or 3),
                        "surface": "Hard" if "US Open" in str(event.get("name")) else "Hard",
                        "player_1": (players[0].get("athlete") or {}).get("displayName"),
                        "player_2": (players[1].get("athlete") or {}).get("displayName"),
                        "player_1_flag": ((players[0].get("athlete") or {}).get("flag") or {}).get("href"),
                        "player_2_flag": ((players[1].get("athlete") or {}).get("flag") or {}).get("href"),
                    })
    return sorted(matches, key=lambda m: m["start_time_utc"])


def predict_schedule(schedule: list[dict[str, Any]], bundles: dict[str, dict[str, Any]], states_by_tour: dict[str, dict[str, PlayerState]]) -> list[dict[str, Any]]:
    output = []
    for match in schedule:
        tour = match["tour"]
        states = states_by_tour[tour]
        a = states.get(_name(match["player_1"]), PlayerState(display_name=match["player_1"] or ""))
        b = states.get(_name(match["player_2"]), PlayerState(display_name=match["player_2"] or ""))
        rank_a = int(a.rank) if math.isfinite(a.rank) else None
        rank_b = int(b.rank) if math.isfinite(b.rank) else None
        eligible = (rank_a is not None and rank_a <= 150) or (rank_b is not None and rank_b <= 150)
        if not eligible:
            continue
        match_date = datetime.fromisoformat(match["start_time_utc"].replace("Z", "+00:00")).replace(tzinfo=None)
        x = np.asarray([_features(a, b, match["surface"], match["best_of"], match_date)])
        reverse_x = np.asarray([_features(b, a, match["surface"], match["best_of"], match_date)])
        bundle = bundles[tour]
        p_forward = float(bundle["winner"].predict_proba(x)[0, 1])
        p_reverse = float(bundle["winner"].predict_proba(reverse_x)[0, 1])
        p1 = (p_forward + 1 - p_reverse) / 2
        margin = (float(bundle["margin"].predict(x)[0]) - float(bundle["margin"].predict(reverse_x)[0])) / 2
        total_x = x[:, bundle.get("total_feature_indices", list(range(x.shape[1])))]
        reverse_total_x = reverse_x[:, bundle.get("total_feature_indices", list(range(reverse_x.shape[1])))]
        total = max(12.0, (float(bundle["total"].predict(total_x)[0]) + float(bundle["total"].predict(reverse_total_x)[0])) / 2)
        match.update({
            "player_1_rank": rank_a, "player_2_rank": rank_b,
            "player_1_win_probability": round(p1, 4), "player_2_win_probability": round(1 - p1, 4),
            "predicted_winner": match["player_1"] if p1 >= .5 else match["player_2"],
            "winner_confidence": round(max(p1, 1 - p1), 4),
            "predicted_game_margin_player_1": round(margin, 1),
            "predicted_total_games": round(total, 1),
            "projected_player_1_games": round((total + margin) / 2, 1),
            "projected_player_2_games": round((total - margin) / 2, 1),
        })
        output.append(match)
    return output


def run_pipeline(root: Path, as_of: date | None = None) -> dict[str, Any]:
    as_of = as_of or datetime.now(timezone.utc).date()
    bundles: dict[str, dict[str, Any]] = {}
    states: dict[str, dict[str, PlayerState]] = {}
    metrics = []
    for tour in ("ATP", "WTA"):
        bundle, tour_states, tour_metrics = train_tour(tour, as_of)
        bundles[tour], states[tour] = bundle, tour_states
        metrics.append(tour_metrics)
    schedule = fetch_schedule(as_of)
    predictions = predict_schedule(schedule, bundles, states)
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "generated_at_utc": generated, "target_date": as_of.isoformat(),
        "model_version": "tennis-v2-calibrated-recency-symmetric",
        "eligibility": "At least one player ranked in the top 150",
        "training_window": f"{(as_of - timedelta(days=730)).isoformat()} through {(as_of - timedelta(days=1)).isoformat()}",
        "data_source": "TennisMyLife historical results; ESPN schedule",
        "matches": predictions, "model_performance": metrics,
    }
    (root / "docs").mkdir(exist_ok=True); (root / "models" / "tennis").mkdir(parents=True, exist_ok=True)
    (root / "predictions" / "tennis").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "tennis-latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "predictions" / "tennis" / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for tour in ("ATP", "WTA"):
        joblib.dump(bundles[tour], root / "models" / "tennis" / f"{tour.lower()}_model.joblib", compress=3)
    (root / "models" / "tennis" / "metadata.json").write_text(json.dumps({"generated_at_utc": generated, "metrics": metrics, "features": FEATURES}, indent=2), encoding="utf-8")
    return payload
