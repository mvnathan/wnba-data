from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

import requests

from .config import (
    DEFAULT_USER_AGENT,
    ESPN_SCOREBOARD_URL,
    ESPN_SUMMARY_URL,
    REQUEST_DELAY,
    REQUEST_RETRY_COUNT,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


class ESPNApiClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    def get_json(self, url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, REQUEST_RETRY_COUNT + 1):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object from ESPN API")
                return data
            except requests.RequestException as error:
                last_error = error
                logger.warning(
                    "ESPN API request failed (attempt %s/%s): %s",
                    attempt,
                    REQUEST_RETRY_COUNT,
                    error,
                )
                if attempt == REQUEST_RETRY_COUNT:
                    break
                time.sleep(REQUEST_DELAY * 2 ** (attempt - 1))
            except ValueError as error:
                logger.error("Invalid JSON response from ESPN API: %s", error)
                raise
        raise RuntimeError(
            "Failed to retrieve ESPN JSON after %s attempts: %s" % (REQUEST_RETRY_COUNT, last_error)
        )

    def fetch_scoreboard(self, day: date) -> dict[str, Any]:
        if not isinstance(day, date):
            raise TypeError("day must be a datetime.date")
        params = {"dates": day.strftime("%Y%m%d")}
        return self.get_json(ESPN_SCOREBOARD_URL, params=params)

    def fetch_summary(self, game_id: str) -> dict[str, Any]:
        if not isinstance(game_id, str):
            raise TypeError("game_id must be a string")
        params = {"event": game_id}
        return self.get_json(ESPN_SUMMARY_URL, params=params)
