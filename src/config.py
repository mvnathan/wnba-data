from __future__ import annotations

from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
GAMES_PATH = DATA_DIR / "games.parquet"
QUARTER_SCORES_PATH = DATA_DIR / "quarter_scores.parquet"
TEAM_GAMES_PATH = DATA_DIR / "team_games.parquet"
LAST_UPDATE_PATH = DATA_DIR / "last_update.json"
BACKFILL_PROGRESS_PATH = DATA_DIR / "backfill_progress.json"

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
