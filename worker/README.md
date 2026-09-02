# WNBA Live Dashboard Worker

This Cloudflare Worker owns high-frequency live score delivery. A one-minute Cron Trigger refreshes a Durable Object snapshot, and dashboard requests read that snapshot at the edge.

## What it does

- Refreshes `/latest.json` every minute around actual game windows.
- Persists the latest snapshot in a Durable Object so every viewer sees the same state.
- Uses Sportradar as the primary live source and ESPN as a fallback.
- Checks both the current Chicago date and each prediction's Chicago game date so late games survive midnight rollover.
- Merges live score, period, clock, and status into the existing pregame prediction payload.
- Recomputes the existing live projection model in JavaScript using the same formulas as `src/live_predict.py`.
- Returns `Cache-Control: no-store` so the phone sees fresh data.
- Reads the canonical pregame snapshot directly from GitHub's `main` branch rather than waiting for GitHub Pages deployment.

The GitHub Pages dashboard requests the Worker snapshot every ~15 seconds and falls back to its local static snapshot if Cloudflare is unavailable. The old five-minute GitHub Actions schedule is retained only as a manual fallback, not as a recurring production job.

## Deploy with GitHub Actions

Create these repository secrets in GitHub Settings -> Secrets and variables -> Actions:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

The API token needs permission to edit Workers Scripts for the selected Cloudflare account.

Then run **Actions -> Deploy Live Dashboard Worker -> Run workflow** once. After the first deployment, changes under `worker/` deploy automatically when merged to `main`.

Wrangler will report the public `workers.dev` URL in the deployment log. Open that URL instead of the GitHub Pages URL for the near-real-time version of the dashboard.

## Local deployment alternative

From the repository root:

```bash
cd worker
npm install
npx wrangler login
npx wrangler deploy
```

No scheduled GitHub Actions workflow is required for live score delivery after the Worker is deployed.

## Runtime checks

- `/health` verifies that the Durable Object snapshot is current and refreshes it synchronously if needed.
- `/diagnostics` verifies the canonical GitHub snapshot and both live-data providers.
- `/snapshot-status` reports snapshot age, source, target date, and live-source status.
- `/latest.json` returns the dashboard payload plus Cloudflare snapshot freshness fields.
