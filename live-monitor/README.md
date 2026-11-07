# Live Market Monitor — simulation-only MVP

An opt-in, selected-game research application for NBA/WNBA, designed to accommodate every DraftKings market **actually exposed by a licensed data feed**. No bet placement, scraping, or assumption that every DraftKings market is available via an API.

## Market coverage
- Discover market definitions dynamically from the provider; retain provider event/market/selection IDs, labels, lines, odds, period, settlement rules, availability and timestamps.
- All markets are **visible** even if no predictive model exists. Mark them `UNMODELED`, not as an opportunity.
- Full game, first/second half, each quarter: moneyline, spread, totals; team totals, alternate lines, player props, milestones and situational markets are supported as catalog entries. Only validated models can score them.
- Live same-game parlays require a joint-outcome model; never multiply correlated probabilities as if independent.
- Market availability differs by game, state, time and book; never promise a fixed universal list.

## Session lifecycle
1. User selects games and market categories and starts a session.
2. Server subscribes only to selected game IDs and provider-supported markets.
3. Each update is joined by event ID, market ID, selection ID and event time.
4. Quote/game freshness, suspension and event ordering are checked before any alert.
5. Only validated model probabilities are compared with no-vig market references and actual payout odds.
6. Alert is recorded prospectively with immutable input/quote/model timestamps.
7. Stop session or final whistle unsubscribes and shuts down per-game inference.

## API contract (future FastAPI service)
`POST /sessions` {game_ids, market_categories}; `DELETE /sessions/{id}`;
`GET /games`; `GET /sessions/{id}/markets`; `GET /sessions/{id}/alerts`;
`WS /sessions/{id}/stream`.

## Data access
A licensed DraftKings-covered feed is required. The browser prototype does not connect to DraftKings or produce actionable picks. Never embed API credentials in client code. Use provider-authorized interfaces, observe regional licensing and feed terms.

## Validation gates
- Prospective, timestamped shadow testing; calibration by league and market; replay tests that prohibit future leakage.
- Measure odds-to-alert latency, stale-quote false edges, suspended market rates, and quote availability.
- No automated wagering, no retroactive win attribution.
