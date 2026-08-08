#!/usr/bin/env python3
from __future__ import annotations

import json
from src.live_monitor import monitor_live_games

if __name__ == "__main__":
    result = monitor_live_games()
    print(json.dumps(result, indent=2))
