#!/usr/bin/env python3
from __future__ import annotations

import argparse
from src import features


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="features/model_features.parquet")
    args = p.parse_args()
    features.build_and_save(data_root=args.data_root, out_path=args.out)


if __name__ == "__main__":
    main()
