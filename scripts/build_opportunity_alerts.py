from __future__ import annotations

import json

from src.opportunity_alerts import build_opportunity_alerts


if __name__ == "__main__":
    result = build_opportunity_alerts()
    print(json.dumps(result, indent=2, allow_nan=False))
