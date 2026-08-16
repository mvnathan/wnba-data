# WNBA Live Dashboard Worker

This Cloudflare Worker removes live score delivery from GitHub Actions. The Worker runs on demand at the edge whenever the dashboard requests `latest.json`.

## What it does

- Proxies the existing GitHub Pages dashboard.
- Intercepts `/latest.json` and builds a fresh response in real time.
- Fetches the current ESPN WNBA scoreboard with cache-busting query parameters.
- Merges live score, period, clock, and status into the existing pregame prediction payload.
- Recomputes the existing live projection model in JavaScript using the same formulas as `src/live_predict.py`.
- Returns `Cache-Control: no-store` so the phone sees fresh data.

The existing dashboard already requests `latest.json` every ~15 seconds, so when the dashboard is opened through the Worker URL it no longer depends on scheduled GitHub Actions to update the score.

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

No scheduled workflow is required for live score delivery after the Worker is deployed.
