import liveApp from "./index.js";
import { buildAlerts, enrichTennisMarkets } from "./alerts.js";
import { createVapidKeys, sendWebPush } from "./push.js";

const STATIC_LATEST = "https://raw.githubusercontent.com/mvnathan/wnba-data/main/docs/latest.json";
const TENNIS_LATEST = "https://raw.githubusercontent.com/mvnathan/wnba-data/main/docs/tennis-latest.json";
const LIVESCORE_TENNIS = "https://prod-public-api.livescore.com/v1/api/app/date/tennis";
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

async function dispatchOpportunityAlerts(env) {
  const [wnba, tennisBase] = await Promise.all([fetchStaticLatest(), fetchTennisLatest()]);
  const tennis = await enrichTennisMarkets(tennisBase, env.ODDS_API_KEY);
  const alerts = buildAlerts(wnba, tennis);
  const response = await snapshotStub(env).fetch("https://snapshot.internal/push/dispatch", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ alerts }),
  });
  if (!response.ok) throw new Error(`Push dispatch returned ${response.status}`);
  return response.json();
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

function tennisNameKey(value) {
  return String(value || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .toLowerCase().replace(/[^a-z0-9]/g, "");
}

function tennisNameParts(value) {
  return String(value || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .toLowerCase().replace(/[^a-z0-9 ]/g, " ").trim().split(/\s+/).filter(Boolean);
}

function tennisNamesMatch(left, right) {
  if (tennisNameKey(left) === tennisNameKey(right)) return true;
  const a = tennisNameParts(left);
  const b = tennisNameParts(right);
  if (a.length < 2 || b.length < 2) return false;
  return a[0] === b[0] && a[a.length - 1] === b[b.length - 1];
}

function tennisStatus(event) {
  const code = String(event?.Eps || "");
  if (code === "FT") return { label: "Final", state: "post" };
  if (code === "NS") return { label: "Scheduled", state: "pre" };
  if (code === "Canc.") return { label: "Cancelled", state: "post" };
  if (code === "Postp.") return { label: "Postponed", state: "pre" };
  if (/^S\d+$/.test(code)) return { label: `Set ${code.slice(1)}`, state: "in" };
  return { label: code || "Scheduled", state: "in" };
}

function normalizeTennisEvent(event) {
  const player1 = event?.T1?.[0]?.Nm;
  const player2 = event?.T2?.[0]?.Nm;
  if (!player1 || !player2 || event.T1.length !== 1 || event.T2.length !== 1) return null;
  const sets = [];
  for (let set = 1; set <= 5; set += 1) {
    const one = event[`Tr1S${set}`];
    const two = event[`Tr2S${set}`];
    if (one === undefined && two === undefined) continue;
    sets.push({
      set,
      player_1: one ?? null,
      player_2: two ?? null,
      player_1_tiebreak: event[`Tr1S${set}T`] ?? null,
      player_2_tiebreak: event[`Tr2S${set}T`] ?? null,
    });
  }
  const status = tennisStatus(event);
  return {
    live_event_id: String(event.Eid || ""),
    player_1: player1,
    player_2: player2,
    player_1_sets: Number(event.Tr1 || 0),
    player_2_sets: Number(event.Tr2 || 0),
    player_1_game_points: event.Tr1G ?? null,
    player_2_game_points: event.Tr2G ?? null,
    serving_player: Number(event.Esrv || 0) || null,
    set_scores: sets,
    live_status: status.label,
    live_state: status.state,
  };
}

async function fetchLiveScoreTennis(date) {
  const stamp = String(date || "").replaceAll("-", "");
  const request = new Request(`${LIVESCORE_TENNIS}/${stamp}/0.00`, {
    headers: { "accept": "application/json", "user-agent": "Mozilla/5.0" },
  });
  const cache = caches.default;
  let response = await cache.match(request);
  if (!response) {
    response = await fetch(request, { cf: { cacheTtl: 10, cacheEverything: true } });
    if (!response.ok) throw new Error(`LiveScore tennis returned ${response.status}`);
    response = new Response(response.body, response);
    response.headers.set("cache-control", "public, max-age=10");
    await cache.put(request, response.clone());
  }
  const payload = await response.json();
  const events = [];
  for (const stage of payload?.Stages || []) {
    if (!String(stage?.Scd || "").includes("singles")) continue;
    for (const raw of stage?.Events || []) {
      const event = normalizeTennisEvent(raw);
      if (event) events.push(event);
    }
  }
  return {
    generated_at_utc: Number(payload?.Ts) ? new Date(Number(payload.Ts) * 1000).toISOString() : new Date().toISOString(),
    events,
  };
}

function mergeTennisScores(predictions, live) {
  const events = live?.events || [];
  return (predictions || []).map((match) => {
    let event = events.find((candidate) => tennisNamesMatch(candidate.player_1, match.player_1) && tennisNamesMatch(candidate.player_2, match.player_2));
    let reversed = false;
    if (!event) {
      event = events.find((candidate) => tennisNamesMatch(candidate.player_1, match.player_2) && tennisNamesMatch(candidate.player_2, match.player_1));
      reversed = Boolean(event);
    }
    if (!event) return match;
    if (!reversed) return { ...match, ...event };
    return {
      ...match,
      live_event_id: event.live_event_id,
      player_1_sets: event.player_2_sets,
      player_2_sets: event.player_1_sets,
      player_1_game_points: event.player_2_game_points,
      player_2_game_points: event.player_1_game_points,
      serving_player: event.serving_player === 1 ? 2 : event.serving_player === 2 ? 1 : null,
      set_scores: event.set_scores.map((set) => ({
        set: set.set, player_1: set.player_2, player_2: set.player_1,
        player_1_tiebreak: set.player_2_tiebreak, player_2_tiebreak: set.player_1_tiebreak,
      })),
      live_status: event.live_status,
      live_state: event.live_state,
    };
  });
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

    if (url.pathname === "/push/public-key" && request.method === "GET") {
      let keys = await this.state.storage.get("push-vapid-keys");
      if (!keys) { keys = await createVapidKeys(); await this.state.storage.put("push-vapid-keys", keys); }
      return jsonResponse({ publicKey: keys.publicKey });
    }

    if (url.pathname === "/push/subscribe" && request.method === "POST") {
      const subscription = await request.json();
      if (!subscription?.endpoint || !subscription?.keys?.p256dh || !subscription?.keys?.auth) return jsonResponse({ ok: false, error: "Invalid push subscription" }, 400);
      const subscriptions = (await this.state.storage.get("push-subscriptions")) || [];
      if (subscriptions.length >= 25 && !subscriptions.some((item) => item.endpoint === subscription.endpoint)) return jsonResponse({ ok: false, error: "Subscription limit reached" }, 429);
      const next = subscriptions.filter((item) => item.endpoint !== subscription.endpoint);
      next.push(subscription); await this.state.storage.put("push-subscriptions", next);
      return jsonResponse({ ok: true, subscribers: next.length });
    }

    if (url.pathname === "/push/unsubscribe" && request.method === "POST") {
      const { endpoint } = await request.json();
      const subscriptions = (await this.state.storage.get("push-subscriptions")) || [];
      const next = subscriptions.filter((item) => item.endpoint !== endpoint);
      await this.state.storage.put("push-subscriptions", next);
      return jsonResponse({ ok: true, subscribers: next.length });
    }

    if (url.pathname === "/push/dispatch" && request.method === "POST") {
      const { alerts = [] } = await request.json();
      let subscriptions = (await this.state.storage.get("push-subscriptions")) || [];
      const keys = await this.state.storage.get("push-vapid-keys");
      const state = (await this.state.storage.get("push-alert-state")) || {};
      if (!keys || !subscriptions.length || !alerts.length) return jsonResponse({ ok: true, sent: 0, subscribers: subscriptions.length, alerts: alerts.length });
      let sent = 0; const expired = new Set();
      for (const subscription of subscriptions) {
        const prior = state[subscription.endpoint] || {};
        for (const alert of alerts) {
          const previous = Number(prior[alert.id]);
          if (Number.isFinite(previous) && Number(alert.score) < previous + 2) continue;
          try {
            const response = await sendWebPush(subscription, alert, keys);
            if (response.status === 404 || response.status === 410) { expired.add(subscription.endpoint); break; }
            if (response.ok) { prior[alert.id] = Number(alert.score); sent += 1; }
          } catch (error) { console.error("Push delivery failed", error); }
        }
        state[subscription.endpoint] = prior;
      }
      subscriptions = subscriptions.filter((item) => !expired.has(item.endpoint));
      for (const endpoint of expired) delete state[endpoint];
      await this.state.storage.put("push-subscriptions", subscriptions);
      await this.state.storage.put("push-alert-state", state);
      return jsonResponse({ ok: true, sent, subscribers: subscriptions.length, alerts: alerts.length });
    }

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

    if (request.method === "OPTIONS" && url.pathname.startsWith("/push/")) {
      return new Response(null, { status: 204, headers: { "access-control-allow-origin": "*", "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type" } });
    }

    if (["/push/public-key", "/push/subscribe", "/push/unsubscribe"].includes(url.pathname)) {
      const origin = request.headers.get("origin");
      if (request.method === "POST" && origin !== "https://mvnathan.github.io") return jsonResponse({ ok: false, error: "Origin not allowed" }, 403);
      return snapshotStub(env).fetch(new Request(`https://snapshot.internal${url.pathname}`, request));
    }

    if (url.pathname === "/tennis/latest.json" || url.pathname === "/api/tennis") {
      try {
        const data = await enrichTennisMarkets(await fetchTennisLatest(), env.ODDS_API_KEY);
        try {
          const live = await fetchLiveScoreTennis(data.target_date);
          return jsonResponse({
            ...data,
            matches: mergeTennisScores(data.matches, live),
            live_score_generated_at_utc: live.generated_at_utc,
            live_score_source: "LiveScore",
            cloudflare_delivery: "tennis-live-overlay",
          });
        } catch (liveError) {
          return jsonResponse({
            ...data,
            live_score_error: String(liveError?.message || liveError),
            live_score_source: "prediction-feed-fallback",
            cloudflare_delivery: "tennis-static-proxy",
          });
        }
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
    ctx.waitUntil(Promise.allSettled([
      refreshSnapshot(env).catch((error) => console.error("Scheduled live snapshot refresh failed", error)),
      dispatchOpportunityAlerts(env).catch((error) => console.error("Opportunity push dispatch failed", error)),
    ]));
  },
};

export { dispatchOpportunityAlerts, mergeTennisScores, normalizeTennisEvent, shouldPollLive, tennisNameKey };
