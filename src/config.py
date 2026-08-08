from __future__ import annotations

import zoneinfo
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = REPO_ROOT / "features"
MODELS_DIR = REPO_ROOT / "models"
PREDICTIONS_DIR = REPO_ROOT / "predictions"
DOCS_DIR = REPO_ROOT / "docs"
SCRIPTS_DIR = REPO_ROOT / "scripts"

GAMES_PATH = DATA_DIR / "games.parquet"
QUARTER_SCORES_PATH = DATA_DIR / "quarter_scores.parquet"
TEAM_GAMES_PATH = DATA_DIR / "team_games.parquet"
LIVE_STATE_PATH = DATA_DIR / "live_state.json"
LIVE_SNAPSHOTS_PATH = DATA_DIR / "live_snapshots.parquet"
LAST_UPDATE_PATH = DATA_DIR / "last_update.json"
BACKFILL_PROGRESS_PATH = DATA_DIR / "backfill_progress.json"
FEATURES_PATH = FEATURES_DIR / "model_features.parquet"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_LEADERBOARD_PATH = MODELS_DIR / "model_leaderboard.csv"
PREDICTION_LATEST_JSON = PREDICTIONS_DIR / "latest.json"
PREDICTION_LATEST_CSV = PREDICTIONS_DIR / "latest.csv"
PREDICTION_HISTORY_PATH = PREDICTIONS_DIR / "prediction_history.parquet"
QUARTER_EVENTS_PATH = PREDICTIONS_DIR / "quarter_events.parquet"
DOCS_LATEST_JSON = DOCS_DIR / "latest.json"
DOCS_HISTORY_JSON = DOCS_DIR / "history.json"

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary"

DEFAULT_USER_AGENT = "wnba-data/1.0 (+https://github.com/YOUR_GITHUB_USERNAME/wnba-data)"
REQUEST_TIMEOUT = 20
REQUEST_RETRY_COUNT = 3
REQUEST_DELAY = 1.0
SEASONS_RETAINED = 3
DAILY_QUERY_OFFSETS = [-2, -1, 0, 1]
HISTORICAL_SEASON_START = (4, 1)
HISTORICAL_SEASON_END = (11, 15)
CHICAGO = zoneinfo.ZoneInfo("America/Chicago")
