import json
import sys

from src.update_data import run_daily_update


if __name__ == "__main__":
    try:
        summary = run_daily_update()
        print(json.dumps(summary, indent=2))
        if summary.get("errors") and not summary.get("games_rows"):
            sys.exit(1)
    except Exception as error:
        print("Daily update failed:", error)
        sys.exit(1)
