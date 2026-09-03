import liveApp from "./index.js";

const STATIC_LATEST = "https://raw.githubusercontent.com/mvnathan/wnba-data/main/docs/latest.json";
const TENNIS_LATEST = "https://raw.githubusercontent.com/mvnathan/wnba-data/main/docs/tennis-latest.json";
const SNAPSHOT_NAME = "wnba-live-singleton";
const SNAPSHOT_KEY = "latest";
const LIVE_WINDOW_BEFORE_MS = 45 * 60 * 1000;
const LIVE_WINDOW_AFTER_MS = 4 * 60 * 60 * 1000;
const SNAPSHOT_STALE_MS = 90 * 1000;

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store, no-cache, must-revalidate",
      "access-control-allow-origin": "*",
    },
  });
}

function snapshotStub(env) {
  const id = env.LIVE_SNAPSHOT.idFromName(SNAPSHOT_NAME);
  return env.LIVE_SNAPSHOT.get(id);
}

async function readSnapshot(env) {
  const response = await snapshotStub(env).fetch("https://snapshot.internal/value");
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Snapshot store returned ${response.status}`);
  return response.json();
}

async function writeSnapshot(env, data, source) {
  const record = {
    stored_at_utc: new Date().toISOString(),
    source,
    data,
  };
  const response = await snapshotStub(env).fetch("https://snapshot.internal/value", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(record),
  });
  if (!response.ok) throw new Error(`Snapshot store write returned ${response.status}`);
  return record;
}

async function fetchStaticLatest() {
  const response = await fetch(`${STATIC_LATEST}?t=${Date.now()}`, {
    headers: { "cache-control": "no-cache" },
    cf: { cacheTtl: 0, cacheEverything: false },
  });
  if (!response.ok) throw new Error(`Static latest returned ${response.status}`);
  return response.json();
}

async function fetchTennisLatest() {
  const response = await fetch(`${TENNIS_LATEST}?t=${Date.now()}`, {
    headers: { "cache-control": "no-cache" },
    cf: { cacheTtl: 0, cacheEverything: false },
  });
  if (!response.ok) throw new Error(`Tennis latest returned ${response.status}`);
  return response.json();
}

function gameStartMillis(game) {
  const value = Date.parse(String(game?.game_date_utc || ""));
  return Number.isFinite(value) ? value : null;
}

function shouldPollLive(staticLatest, nowMs = Date.now()) {
  const games = Array.isArray(staticLatest?.games) ? staticLatest.games : [];
  return games.some((game) => {
    const start = gameStartMillis(game);
    if (start === null) return false;
    return nowMs >= start - LIVE_WINDOW_BEFORE_MS && nowMs <= start + LIVE_WINDOW_AFTER_MS;
  });
}

async function refreshSnapshot(env) {
  // GitHub remains the source of pregame/model output. Cloudflare owns the
  // high-frequency live overlay only around actual game windows.
  const staticLatest = await fetchStaticLatest();

  if (!shouldPollLive(staticLatest)) {
    staticLatest.live_delivery = staticLatest.live_delivery || "cloudflare-snapshot-idle";
    staticLatest.live_source_status = staticLatest.live_source_status || "pregame";
    const record = await writeSnapshot(env, staticLatest, "github-raw-pregame");
    return { ...record, live_poll: false };
  }

  const response = await liveApp.fetch(
    new Request("https://worker.internal/api/live", {
      method: "GET",
      headers: { "cache-control": "no-cache" },
    }),
    env,
  );
  if (!response.ok) throw new Error(`Live overlay returned ${response.status}`);
  const live = await response.json();
  const record = await writeSnapshot(env, live, "cloudflare-scheduled-live");
  return { ...record, live_poll: true };
}

function snapshotAgeMs(record) {
  const stored = Date.parse(String(record?.stored_at_utc || ""));
  return Number.isFinite(stored) ? Date.now() - stored : Number.POSITIVE_INFINITY;
}

export class LiveSnapshotStore {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname !== "/value") return new Response("Not found", { status: 404 });

    if (request.method === "GET") {
      const record = await this.state.storage.get(SNAPSHOT_KEY);
      return record ? jsonResponse(record) : new Response("Not found", { status: 404 });
    }

    if (request.method === "PUT") {
      const record = await request.json();
      await this.state.storage.put(SNAPSHOT_KEY, record);
      return jsonResponse({ ok: true, stored_at_utc: record?.stored_at_utc || null });
    }

    return new Response("Method not allowed", { status: 405 });
  }
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (url.pathname === "/tennis/latest.json" || url.pathname === "/api/tennis") {
      try {
        const data = await fetchTennisLatest();
        return jsonResponse({ ...data, cloudflare_delivery: "tennis-static-proxy" });
      } catch (error) {
        return jsonResponse({ ok: false, service: "tennis-predictions", error: String(error?.message || error) }, 503);
      }
    }

    if (url.pathname === "/tennis/health") {
      try {
        const data = await fetchTennisLatest();
        return jsonResponse({
          ok: Boolean(data?.target_date && Array.isArray(data?.matches)),
          service: "tennis-predictions",
          target_date: data?.target_date || null,
          matches: Array.isArray(data?.matches) ? data.matches.length : 0,
          generated_at_utc: data?.generated_at_utc || null,
        });
      } catch (error) {
        return jsonResponse({ ok: false, service: "tennis-predictions", error: String(error?.message || error) }, 503);
      }
    }

    if (url.pathname === "/health") {
      try {
        // Force a current-version refresh. A merely fresh Durable Object record
        // could have been written by the Worker version we just replaced.
        const record = await refreshSnapshot(env);
        const ageSeconds = Math.round(snapshotAgeMs(record) / 1000);
        const ok = ageSeconds <= Math.ceil(SNAPSHOT_STALE_MS / 1000);
        return jsonResponse({
          ok,
          service: "wnba-live-dashboard",
          now: new Date().toISOString(),
          snapshot_age_seconds: ageSeconds,
          snapshot_source: record.source,
          target_date: record.data?.target_date || null,
        }, ok ? 200 : 503);
      } catch (error) {
        return jsonResponse({
          ok: false,
          service: "wnba-live-dashboard",
          error: String(error?.message || error),
        }, 503);
      }
    }

    if (url.pathname === "/snapshot-status") {
      const record = await readSnapshot(env);
      if (!record) return jsonResponse({ ok: false, snapshot: null }, 404);
      return jsonResponse({
        ok: true,
        stored_at_utc: record.stored_at_utc,
        source: record.source,
        age_seconds: Math.round(snapshotAgeMs(record) / 1000),
        target_date: record.data?.target_date || null,
        games: Array.isArray(record.data?.games) ? record.data.games.length : 0,
        live_source: record.data?.live_source || null,
        live_source_status: record.data?.live_source_status || null,
      });
    }

    if (url.pathname === "/latest.json" || url.pathname === "/api/live") {
      let record = null;
      try {
        record = await readSnapshot(env);
      } catch (error) {
        // Fall through to direct refresh if durable storage is temporarily unavailable.
      }

      if (record?.data) {
        const age = snapshotAgeMs(record);
        if (age > SNAPSHOT_STALE_MS && ctx?.waitUntil) {
          ctx.waitUntil(refreshSnapshot(env).catch(() => null));
        }
        return jsonResponse({
          ...record.data,
          cloudflare_snapshot_stored_at_utc: record.stored_at_utc,
          cloudflare_snapshot_source: record.source,
          cloudflare_snapshot_age_seconds: Math.round(age / 1000),
          cloudflare_snapshot_stale: age > SNAPSHOT_STALE_MS,
        });
      }

      try {
        const refreshed = await refreshSnapshot(env);
        return jsonResponse({
          ...refreshed.data,
          cloudflare_snapshot_stored_at_utc: refreshed.stored_at_utc,
          cloudflare_snapshot_source: refreshed.source,
          cloudflare_snapshot_age_seconds: 0,
          cloudflare_snapshot_stale: false,
        });
      } catch (error) {
        // Existing application fallback still protects availability.
        return liveApp.fetch(request, env, ctx);
      }
    }

    return liveApp.fetch(request, env, ctx);
  },

  async scheduled(controller, env, ctx) {
    ctx.waitUntil(
      refreshSnapshot(env).catch((error) => {
        console.error("Scheduled live snapshot refresh failed", error);
      }),
    );
  },
};

export { shouldPollLive };
