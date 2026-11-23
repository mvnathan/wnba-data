"""Read-only provider diagnostic; never prints the API key or actionable bets."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.the-odds-api.com/v4"


def get(path, params):
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY GitHub secret is missing")
    query = urllib.parse.urlencode({**params, "apiKey": key})
    req = urllib.request.Request(f"{BASE}{path}?{query}", headers={"User-Agent": "live-monitor-diagnostic/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
            headers = {k: resp.headers.get(k, "unknown") for k in (
                "x-requests-remaining", "x-requests-used", "x-requests-last")}
            return data, headers
    except urllib.error.HTTPError as exc:
        # Do not include provider response bodies or URLs: may contain secrets.
        raise RuntimeError(f"Provider HTTP {exc.code}; check plan, sport and market access") from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=["basketball_nba", "basketball_wnba"], default="basketball_nba")
    args = parser.parse_args()
    print("Checking authorized The Odds API v4 connectivity (read-only)")
    sports, usage = get("/sports", {})
    matches = [s.get("key") for s in sports if s.get("key") == args.sport]
    print("Sport listed:", bool(matches))
    print("Usage:", usage)
    if not matches:
        return
    events, usage = get(f"/sports/{args.sport}/events", {})
    print("Available event count:", len(events))
    print("Usage after event discovery:", usage)
    if not events:
        print("No current events to inspect. Market discovery deferred.")
        return
    # Inspect only one event to minimize paid credits; supported market names
    # and available bookmaker selections are not the same thing.
    event_id = events[0].get("id")
    if not event_id:
        return
    try:
        result, usage = get(f"/sports/{args.sport}/events/{event_id}/odds", {
            "regions": "us", "bookmakers": "draftkings",
            "markets": "h2h,spreads,totals", "oddsFormat": "american"})
    except RuntimeError as exc:
        print("Market inspection unavailable:", exc)
        return
    books = result.get("bookmakers", [])
    markets = sorted({m.get("key") for b in books for m in b.get("markets", []) if m.get("key")})
    print("DraftKings present for inspected event:", any(b.get("key") == "draftkings" for b in books))
    print("Available inspected market keys:", markets)
    print("Usage after market inspection:", usage)
    print("Other half/quarter/props markets require separate authorized availability checks.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        print("Diagnostic failed:", exc, file=sys.stderr)
        sys.exit(1)
