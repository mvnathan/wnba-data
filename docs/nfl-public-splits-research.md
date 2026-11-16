# NFL public ticket and handle split research

## Purpose
Track percentages of tickets and dollars wagered by side at a named sportsbook and timestamp. Splits are **not** sportsbook-wide totals or absolute bet counts. Initial weight in predictions is **zero**.

## Record
Each observation includes `game_id`, `kickoff_utc`, `captured_at_utc`, `source`, `market` (spread, total, moneyline), `selection`, `ticket_pct`, `handle_pct`, `line`, `odds`, and later `outcome` (win, loss, push). Keep all earlier snapshots; freeze the pregame version.

## Candidate features
Ticket concentration (percentage), handle concentration, difference between handle and tickets, movement in the line since opening, and changes in splits over time. Estimate separately for spread, total, and moneyline; do not interpret a split as sharp money without further evidence.

## Historical evaluation
Obtain legally reusable timestamped historical splits at known pregame times; do not backfill from closing prices. Split chronologically by season/weeks, never randomly by snapshot. Compare no-split baseline, categorical fade-public thresholds and shrinkage/regularized fitted features. Report ATS/total hit rate, odds-adjusted ROI, CLV, confidence intervals, calibration, Brier, and incremental out-of-sample value. Account for multiple bets in a game and avoid selection bias. Do not promote signal without a sufficient sample and positive out-of-sample improvement.

## Current status
Public splits are an informational overlay and research input, NOT a validated adjustment to the NFL production model. Run `python scripts/evaluate_nfl_public_splits.py` after supplying `data/nfl-public-splits-history.json`.
