const STATIC_BASE = "https://mvnathan.github.io/wnba-data";
const RAW_LATEST_URL = "https://raw.githubusercontent.com/mvnathan/wnba-data/main/docs/latest.json";
const ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard";
const SPORTRADAR_BASE = "https://api.sportradar.com/wnba";

const SCHEDULE_TTL_SECONDS = 600;
const LIVE_BOXSCORE_TTL_SECONDS = 20;
const FINAL_BOXSCORE_TTL_SECONDS = 21600;
const LIVE_PAYLOAD_TTL_SECONDS = 15;
const SPORTRADAR_MIN_INTERVAL_MS = 1100;

let scheduleMemory = { key: null, expires: 0, data: null, access: null };
let livePayloadMemory = { expires: 0, data: null, inFlight: null };
let lastSportradarRequestAt = 0;

function chicagoDateParts() {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  return Object.fromEntries(parts.map((p) => [p.type, p.value]));
}

function chicagoDateStamp() {
  const p = chicagoDateParts();
  return `${p.year}${p.month}${p.day}`;
}

function finiteNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function nullableNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function throttleSportradar() {
  const wait = SPORTRADAR_MIN_INTERVAL_MS - (Date.now() - lastSportradarRequestAt);
  if (wait > 0) await sleep(wait);
  lastSportradarRequestAt = Date.now();
}

function elapsedFraction(period, clock) {
  const p = Number(period);
  if (!Number.isFinite(p) || p <= 0) return 0;
  if (p > 4) return 1;
  if (clock === null || clock === undefined || clock === "") return 0;
  const text = String(clock).trim();
  let remaining;
  if (text.includes(":")) {
    const [m, s] = text.split(":", 2);
    remaining = finiteNumber(m) * 60 + finiteNumber(s);
  } else {
    remaining = finiteNumber(text);
  }
  remaining = Math.max(0, Math.min(600, remaining));
  const elapsed = (p - 1) * 600 + (600 - remaining);
  return Math.max(0, Math.min(1, elapsed / 2400));
}

function blendFinalScore(current, pregame, fraction) {
  if (fraction <= 0) return Math.max(current, pregame);
  if (fraction >= 1) return current;
  const observedFullGamePace = current / fraction;
  const observedWeight = Math.min(0.9, Math.pow(fraction, 0.75));
  const blendedRate = (1 - observedWeight) * pregame + observedWeight * observedFullGamePace;
  return Math.max(current, current + (1 - fraction) * blendedRate);
}

function liveHomeWinProbability(projectedMargin, pregameProbability, fraction, currentMargin) {
  if (fraction >= 1) return currentMargin > 0 ? 1 : currentMargin < 0 ? 0 : 0.5;
  const remaining = Math.max(0.05, 1 - fraction);
  const scale = Math.max(2.5, 10 * Math.sqrt(remaining));
  const marginProbability = 1 / (1 + Math.exp(-projectedMargin / scale));
  if (!Number.isFinite(Number(pregameProbability))) return marginProbability;
  const pre = Math.max(0, Math.min(1, Number(pregameProbability)));
  const liveWeight = Math.min(0.95, Math.pow(fraction, 0.75));
  return Math.max(0, Math.min(1, (1 - liveWeight) * pre + liveWeight * marginProbability));
}

function projectLiveGame(game, pregame) {
  const currentHome = finiteNumber(game.home_score);
  const currentAway = finiteNumber(game.away_score);
  const status = String(game.status || "").toUpperCase();
  if (status.includes("FINAL")) {
    const margin = currentHome - currentAway;
    const homeWin = margin > 0 ? 1 : margin < 0 ? 0 : 0.5;
    return {
      elapsed_fraction: 1,
      home_final: currentHome,
      away_final: currentAway,
      final_margin: margin,
      final_total: currentHome + currentAway,
      home_win_probability: homeWin,
      away_win_probability: 1 - homeWin,
    };
  }
  const fraction = elapsedFraction(game.period, game.clock);
  const projectedHome = blendFinalScore(currentHome, finiteNumber(pregame.home_score, currentHome), fraction);
  const projectedAway = blendFinalScore(currentAway, finiteNumber(pregame.away_score, currentAway), fraction);
  const margin = projectedHome - projectedAway;
  const currentMargin = currentHome - currentAway;
  const homeWin = liveHomeWinProbability(margin, pregame.home_win_probability, fraction, currentMargin);
  return {
    elapsed_fraction: fraction,
    home_final: projectedHome,
    away_final: projectedAway,
    final_margin: margin,
    final_total: projectedHome + projectedAway,
    home_win_probability: homeWin,
    away_win_probability: 1 - homeWin,
  };
}

function isLiveOrFinal(status) {
  const text = String(status || "").toUpperCase();
  return ["IN_PROGRESS", "HALFTIME", "END_PERIOD", "FINAL"].some((m) => text.includes(m));
}

function normalizeSportradarStatus(status) {
  const s = String(status || "").toLowerCase();
  if (["closed", "complete"].includes(s)) return "STATUS_FINAL";
  if (s === "halftime") return "STATUS_HALFTIME";
  if (s === "inprogress") return "STATUS_IN_PROGRESS";
  if (["delayed", "postponed", "cancelled"].includes(s)) return `STATUS_${s.toUpperCase()}`;
  return "STATUS_SCHEDULED";
}

function cleanTeamName(value) {
  return String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function teamMatches(predictionName, predictionAbbr, team) {
  const targetName = cleanTeamName(predictionName);
  const targetAbbr = cleanTeamName(predictionAbbr);
  const candidates = [team?.name, team?.market, team?.alias].map(cleanTeamName).filter(Boolean);
  return candidates.some((candidate) =>
    candidate === targetName ||
    candidate === targetAbbr ||
    candidate.endsWith(targetName) ||
    targetName.endsWith(candidate)
  );
}

function scheduleGameMatches(prediction, game) {
  return teamMatches(prediction.home_team, prediction.home_abbr, game?.home) &&
    teamMatches(prediction.away_team, prediction.away_abbr, game?.away);
}

function needsLiveBoxscore(game) {
  const status = String(game?.status || "").toLowerCase();
  return ["inprogress", "halftime"].includes(status);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} from ${url}`);
  return response.json();
}

async function fetchStaticLatest() {
  // Pull the canonical snapshot directly from the main branch. GitHub Pages can
  // lag behind a successful prediction commit, which previously allowed the
  // Durable Object to keep refreshing itself with yesterday's data.
  return fetchJson(`${RAW_LATEST_URL}?t=${Date.now()}`, {
    headers: { "Cache-Control": "no-cache" },
    cf: { cacheTtl: 0, cacheEverything: false },
  });
}

async function fetchEspnScoreboard() {
  const stamp = chicagoDateStamp();
  return fetchJson(`${ESPN_SCOREBOARD}?dates=${stamp}&_=${Date.now()}`, {
    headers: { "User-Agent": "wnba-live-dashboard/1.0", "Cache-Control": "no-cache" },
    cf: { cacheTtl: 0, cacheEverything: false },
  });
}

async function edgeCacheGet(key) {
  const cache = caches.default;
  const response = await cache.match(new Request(`https://cache.local/${key}`));
  if (!response) return null;
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function edgeCachePut(key, value, ttl) {
  const response = new Response(JSON.stringify(value), {
    headers: { "Cache-Control": `max-age=${ttl}` },
  });
  await caches.default.put(new Request(`https://cache.local/${key}`), response);
}

async function sportradarRequest(pathBuilder, apiKey, preferredAccess = null) {
  if (!apiKey) throw new Error("SPORTRADAR_API_KEY is not configured in Worker secrets");
  const accessOrder = preferredAccess === "trial" ? ["trial"] : preferredAccess === "production" ? ["production"] : ["production", "trial"];
  let lastError = null;
  for (const access of accessOrder) {
    try {
      await throttleSportradar();
      const path = pathBuilder(access);
      const url = `${SPORTRADAR_BASE}/${access}/v8/en/${path}?api_key=${encodeURIComponent(apiKey)}`;
      const data = await fetchJson(url, { cf: { cacheTtl: 0, cacheEverything: false } });
      return { data, access };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("Sportradar request failed");
}

async function fetchSportradarSchedule(apiKey) {
  const key = chicagoDateStamp();
  if (scheduleMemory.key === key && scheduleMemory.data && scheduleMemory.expires > Date.now()) {
    return { data: scheduleMemory.data, access: scheduleMemory.access, cache: "memory", key, expires: scheduleMemory.expires };
  }
  const edgeKey = `wnba-schedule-${key}`;
  const edge = await edgeCacheGet(edgeKey);
  if (edge?.data) {
    scheduleMemory = {
      key,
      data: edge.data,
      access: edge.access,
      expires: Date.now() + SCHEDULE_TTL_SECONDS * 1000,
    };
    return { ...scheduleMemory, cache: "edge" };
  }

  const yyyy = key.slice(0, 4);
  const mm = key.slice(4, 6);
  const dd = key.slice(6, 8);
  const result = await sportradarRequest((access) => `games/${yyyy}/${mm}/${dd}/schedule.json`, apiKey);
  const expires = Date.now() + SCHEDULE_TTL_SECONDS * 1000;
  scheduleMemory = { key, data: result.data, access: result.access, expires };
  await edgeCachePut(edgeKey, { data: result.data, access: result.access }, SCHEDULE_TTL_SECONDS);
  return { ...scheduleMemory, cache: "miss" };
}

async function fetchSportradarBoxscore(game, apiKey, access) {
  const gameId = String(game?.id || "");
  const status = String(game?.status || "").toLowerCase();
  const ttl = ["closed", "complete"].includes(status) ? FINAL_BOXSCORE_TTL_SECONDS : LIVE_BOXSCORE_TTL_SECONDS;
  const edgeKey = `wnba-boxscore-${gameId}`;
  const edge = await edgeCacheGet(edgeKey);
  if (edge?.data) return { data: edge.data, access: edge.access || access, cache: "edge" };

  const result = await sportradarRequest(() => `games/${gameId}/boxscore.json`, apiKey, access);
  await edgeCachePut(edgeKey, { data: result.data, access: result.access }, ttl);
  return { ...result, cache: "miss" };
}

function parseSportradarScheduleGame(game) {
  const homeScore = nullableNumber(game?.home_points);
  const awayScore = nullableNumber(game?.away_points);
  if (homeScore === null || awayScore === null) return null;
  return {
    status: normalizeSportradarStatus(game.status),
    status_detail: String(game.status || ""),
    period: null,
    clock: "",
    home_score: homeScore,
    away_score: awayScore,
  };
}

function parseSportradarBoxscore(data) {
  const game = data?.game || data || {};
  const home = game.home || data?.home || {};
  const away = game.away || data?.away || {};
  const homeScore = nullableNumber(home.points ?? game.home_points);
  const awayScore = nullableNumber(away.points ?? game.away_points);
  if (homeScore === null || awayScore === null) return null;
  return {
    status: normalizeSportradarStatus(game.status),
    status_detail: String(game.status || ""),
    period: nullableNumber(game.quarter),
    clock: game.clock_decimal || game.clock || "",
    home_score: homeScore,
    away_score: awayScore,
  };
}

function applyLiveState(prediction, live, now, source) {
  prediction.live_status = live.status;
  prediction.live_status_detail = live.status_detail;
  if (live.period !== null && live.period !== undefined) prediction.live_period = live.period;
  if (live.clock) prediction.live_clock = live.clock;
  prediction.live_home_score = live.home_score;
  prediction.live_away_score = live.away_score;
  prediction.live_updated_at = now;
  prediction.status = live.status || prediction.status;
  prediction.status_detail = live.status_detail || prediction.status_detail;
  prediction.live_source = source;

  if (isLiveOrFinal(live.status)) {
    const projection = projectLiveGame(live, prediction);
    prediction.live_projected_home_score = projection.home_final;
    prediction.live_projected_away_score = projection.away_final;
    prediction.live_predicted_margin = projection.final_margin;
    prediction.live_predicted_total = projection.final_total;
    prediction.live_home_win_probability = projection.home_win_probability;
    prediction.live_away_win_probability = projection.away_win_probability;
    prediction.live_elapsed_fraction = projection.elapsed_fraction;
    prediction.live_projection_updated_at = now;
    prediction.live_projected_winner_side = projection.final_margin > 0 ? "home" : projection.final_margin < 0 ? "away" : "pickem";
    prediction.live_projected_winner_abbr = projection.final_margin > 0 ? prediction.home_abbr : projection.final_margin < 0 ? prediction.away_abbr : "PK";
  }
}

async function overlaySportradar(latest, apiKey) {
  const schedule = await fetchSportradarSchedule(apiKey);
  const scheduleGames = Array.isArray(schedule.data?.games) ? schedule.data.games : [];
  const now = new Date().toISOString();
  let matched = 0;
  let refreshed = 0;
  let boxscoreRequests = 0;

  for (const prediction of latest.games || []) {
    const scheduledGame = scheduleGames.find((g) => scheduleGameMatches(prediction, g));
    if (!scheduledGame) continue;
    matched += 1;

    let live = parseSportradarScheduleGame(scheduledGame);

    if (needsLiveBoxscore(scheduledGame)) {
      const box = await fetchSportradarBoxscore(scheduledGame, apiKey, schedule.access);
      if (box.cache === "miss") boxscoreRequests += 1;
      live = parseSportradarBoxscore(box.data) || live;
    }

    if (!live) continue;
    applyLiveState(prediction, live, now, "Sportradar via Cloudflare Worker");
    refreshed += 1;
  }

  latest.live_generated_at_utc = now;
  latest.last_live_update_utc = refreshed ? now : latest.last_live_update_utc;
  latest.live_delivery = "cloudflare-worker";
  latest.live_source = "sportradar";
  latest.live_source_status = matched ? "fresh" : "no-matches";
  latest.live_source_matched_games = matched;
  latest.live_source_refreshed_games = refreshed;
  latest.live_source_schedule_cache = schedule.cache || "memory";
  latest.live_source_boxscore_requests = boxscoreRequests;
  delete latest.live_source_error;
  return latest;
}

async function overlayEspn(latest) {
  const scoreboard = await fetchEspnScoreboard();
  const events = Array.isArray(scoreboard.events) ? scoreboard.events : [];
  const now = new Date().toISOString();
  for (const event of events) {
    const competition = Array.isArray(event?.competitions) ? event.competitions[0] : null;
    if (!competition || !Array.isArray(competition.competitors)) continue;
    const home = competition.competitors.find((c) => c?.homeAway === "home");
    const away = competition.competitors.find((c) => c?.homeAway === "away");
    const prediction = (latest.games || []).find((g) => String(g.game_id) === String(event.id));
    if (!prediction || !home || !away) continue;
    const type = event?.status?.type || {};
    const live = {
      status: type.name || "",
      status_detail: type.detail || type.shortDetail || "",
      period: event?.status?.period ?? type.period ?? null,
      clock: event?.status?.displayClock || type.displayClock || "",
      home_score: finiteNumber(home.score),
      away_score: finiteNumber(away.score),
    };
    applyLiveState(prediction, live, now, "ESPN via Cloudflare Worker");
  }
  latest.live_generated_at_utc = now;
  latest.last_live_update_utc = now;
  latest.live_delivery = "cloudflare-worker";
  latest.live_source = "espn";
  latest.live_source_status = "fresh";
  return latest;
}

async function buildLivePayloadUncached(env) {
  const latest = await fetchStaticLatest();
  latest.games = Array.isArray(latest.games) ? latest.games : [];
  let sportradarError = null;

  try {
    return await overlaySportradar(latest, env?.SPORTRADAR_API_KEY);
  } catch (error) {
    sportradarError = String(error?.message || error);
  }

  try {
    const result = await overlayEspn(latest);
    result.live_source_error = `Sportradar failed: ${sportradarError}`;
    return result;
  } catch (error) {
    latest.live_delivery = "cloudflare-worker-fallback";
    latest.live_source = "github-raw";
    latest.live_source_status = "fallback";
    latest.live_source_error = `Sportradar failed: ${sportradarError}; ESPN failed: ${String(error?.message || error)}`;
    return latest;
  }
}

async function buildLivePayload(env) {
  if (livePayloadMemory.data && livePayloadMemory.expires > Date.now()) {
    return { ...livePayloadMemory.data, live_response_cache: "memory-hit" };
  }

  const edgeKey = `wnba-live-payload-${chicagoDateStamp()}`;
  const edge = await edgeCacheGet(edgeKey);
  if (edge?.data) {
    livePayloadMemory = { data: edge.data, expires: Date.now() + LIVE_PAYLOAD_TTL_SECONDS * 1000, inFlight: null };
    return { ...edge.data, live_response_cache: "edge-hit" };
  }

  if (livePayloadMemory.inFlight) return livePayloadMemory.inFlight;

  livePayloadMemory.inFlight = (async () => {
    try {
      const data = await buildLivePayloadUncached(env);
      livePayloadMemory.data = data;
      livePayloadMemory.expires = Date.now() + LIVE_PAYLOAD_TTL_SECONDS * 1000;
      await edgeCachePut(edgeKey, { data }, LIVE_PAYLOAD_TTL_SECONDS);
      return { ...data, live_response_cache: "miss" };
    } finally {
      livePayloadMemory.inFlight = null;
    }
  })();

  return livePayloadMemory.inFlight;
}

async function diagnostics(env) {
  const result = {
    ok: true,
    service: "wnba-live-dashboard",
    now: new Date().toISOString(),
    cache: {
      schedule_ttl_seconds: SCHEDULE_TTL_SECONDS,
      live_boxscore_ttl_seconds: LIVE_BOXSCORE_TTL_SECONDS,
      final_boxscore_ttl_seconds: FINAL_BOXSCORE_TTL_SECONDS,
      live_payload_ttl_seconds: LIVE_PAYLOAD_TTL_SECONDS,
      sportradar_min_interval_ms: SPORTRADAR_MIN_INTERVAL_MS,
    },
    static_latest: { ok: false },
    sportradar: { ok: false },
    espn: { ok: false },
  };

  try {
    const latest = await fetchStaticLatest();
    result.static_latest = {
      ok: true,
      games: Array.isArray(latest.games) ? latest.games.length : 0,
      target_date: latest.target_date || null,
      last_live_update_utc: latest.last_live_update_utc || null,
      source: "github-raw",
    };
  } catch (error) {
    result.ok = false;
    result.static_latest = { ok: false, error: String(error?.message || error) };
  }

  try {
    const schedule = await fetchSportradarSchedule(env?.SPORTRADAR_API_KEY);
    result.sportradar = {
      ok: true,
      access: schedule.access,
      games: Array.isArray(schedule.data?.games) ? schedule.data.games.length : 0,
      date: schedule.key,
      schedule_cache: schedule.cache || "memory",
      schedule_cache_expires_in_seconds: Math.max(0, Math.round((schedule.expires - Date.now()) / 1000)),
    };
  } catch (error) {
    result.sportradar = { ok: false, error: String(error?.message || error) };
  }

  // ESPN is intentionally diagnostic-only and is no longer needed for the primary live path.
  try {
    const scoreboard = await fetchEspnScoreboard();
    result.espn = { ok: true, events: Array.isArray(scoreboard.events) ? scoreboard.events.length : 0, date: chicagoDateStamp() };
  } catch (error) {
    result.espn = { ok: false, error: String(error?.message || error), date: chicagoDateStamp() };
  }

  return result;
}

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

async function proxyStatic(pathname) {
  const url = `${STATIC_BASE}${pathname === "/" ? "/" : pathname}?t=${Date.now()}`;
  const upstream = await fetch(url, { cf: { cacheTtl: 0, cacheEverything: false } });
  const headers = new Headers(upstream.headers);
  headers.set("cache-control", "no-store, no-cache, must-revalidate");
  return new Response(upstream.body, { status: upstream.status, headers });
}

const SNAPSHOT_KEY = "latest";
const SNAPSHOT_STALE_MS = 90 * 1000;

function latestGameStartMs(latest) {
  const starts = (latest?.games || [])
    .map((game) => Date.parse(game?.game_date_utc || ""))
    .filter(Number.isFinite);
  return starts.length ? Math.max(...starts) : null;
}

function firstGameStartMs(latest) {
  const starts = (latest?.games || [])
    .map((game) => Date.parse(game?.game_date_utc || ""))
    .filter(Number.isFinite);
  return starts.length ? Math.min(...starts) : null;
}

function withinLiveWindow(latest, nowMs = Date.now()) {
  const first = firstGameStartMs(latest);
  const last = latestGameStartMs(latest);
  if (first === null || last === null) return false;
  return nowMs >= first - 45 * 60 * 1000 && nowMs <= last + 4 * 60 * 60 * 1000;
}

function decorateStoredSnapshot(payload, storedAt, source) {
  const now = Date.now();
  return {
    ...payload,
    cloudflare_snapshot_stored_at_utc: storedAt,
    cloudflare_snapshot_source: source,
    cloudflare_snapshot_age_seconds: Math.max(0, Math.round((now - Date.parse(storedAt)) / 1000)),
    cloudflare_snapshot_stale: now - Date.parse(storedAt) > SNAPSHOT_STALE_MS,
  };
}

export class LiveSnapshotStore {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async getSnapshot() {
    return this.state.storage.get(SNAPSHOT_KEY);
  }

  async refresh(forceLive = false) {
    const staticLatest = await fetchStaticLatest();
    staticLatest.games = Array.isArray(staticLatest.games) ? staticLatest.games : [];
    let payload;
    let source;

    if (forceLive || withinLiveWindow(staticLatest)) {
      payload = await buildLivePayloadUncached(this.env);
      source = payload.live_source || "live-overlay";
    } else {
      payload = {
        ...staticLatest,
        live_delivery: "cloudflare-snapshot-idle",
        live_source_status: "pregame",
      };
      source = "github-raw-pregame";
    }

    const record = {
      payload,
      stored_at_utc: new Date().toISOString(),
      source,
    };
    await this.state.storage.put(SNAPSHOT_KEY, record);
    return record;
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (request.method === "POST" && url.pathname === "/refresh") {
      const forceLive = url.searchParams.get("forceLive") === "1";
      const record = await this.refresh(forceLive);
      return jsonResponse(decorateStoredSnapshot(record.payload, record.stored_at_utc, record.source));
    }

    let record = await this.getSnapshot();
    if (!record) record = await this.refresh(false);
    return jsonResponse(decorateStoredSnapshot(record.payload, record.stored_at_utc, record.source));
  }
}

function snapshotStub(env) {
  const id = env.LIVE_SNAPSHOT.idFromName("global");
  return env.LIVE_SNAPSHOT.get(id);
}

async function refreshStoredSnapshot(env, forceLive = false) {
  const stub = snapshotStub(env);
  return stub.fetch(`https://snapshot.internal/refresh?forceLive=${forceLive ? "1" : "0"}`, { method: "POST" });
}

async function readStoredSnapshot(env, ctx) {
  const stub = snapshotStub(env);
  const response = await stub.fetch("https://snapshot.internal/latest");
  const data = await response.json();
  if (data.cloudflare_snapshot_stale && ctx) {
    ctx.waitUntil(refreshStoredSnapshot(env, false));
  }
  return data;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET, OPTIONS",
          "access-control-allow-headers": "content-type",
        },
      });
    }

    if (request.method !== "GET") return jsonResponse({ error: "Method not allowed" }, 405);
    if (url.pathname === "/health") return jsonResponse({ ok: true, service: "wnba-live-dashboard", now: new Date().toISOString() });
    if (url.pathname === "/diagnostics") return jsonResponse(await diagnostics(env));

    if (url.pathname === "/latest.json" || url.pathname === "/api/live") {
      try {
        return jsonResponse(await readStoredSnapshot(env, ctx));
      } catch (error) {
        return jsonResponse({ error: "Live snapshot unavailable", detail: String(error?.message || error) }, 502);
      }
    }

    return proxyStatic(url.pathname);
  },

  async scheduled(controller, env, ctx) {
    ctx.waitUntil(refreshStoredSnapshot(env, false));
  },
};
