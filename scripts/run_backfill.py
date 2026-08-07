import argparse
import json
import sys

from src.backfill_history import run_backfill


def main() -> int:
    parser = argparse.ArgumentParser(description="Run historical WNBA backfill.")
    parser.add_argument("--seasons", nargs="+", type=int, help="Seasons to backfill.")
    parser.add_argument("--force", action="store_true", help="Force redownload even if cached.")
    args = parser.parse_args()

    seasons = args.seasons if args.seasons else []
    try:
        summary = run_backfill(seasons=seasons, force=args.force)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as error:
        print("Backfill failed:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
