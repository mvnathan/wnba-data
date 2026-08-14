#!/usr/bin/env python3

from __future__ import annotations

import json

from src.evaluate_predictions import (
    evaluate_predictions,
)


if __name__ == "__main__":
    result = evaluate_predictions()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )