#!/usr/bin/env python3
from __future__ import annotations

import json
from src.dashboard import build_dashboard

if __name__ == "__main__":
    result = build_dashboard()
    print(json.dumps(result, indent=2))
