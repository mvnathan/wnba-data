#!/usr/bin/env python3
from pathlib import Path

from src.tennis_model import run_pipeline


if __name__ == "__main__":
    payload = run_pipeline(Path(__file__).resolve().parents[1])
    print(f"Published {len(payload['matches'])} eligible tennis predictions for {payload['target_date']}")
    for metric in payload["model_performance"]:
        print(metric)
