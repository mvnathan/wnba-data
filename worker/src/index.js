const STATIC_BASE = "https://mvnathan.github.io/wnba-data";
const ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard";
const SPORTRADAR_BASE = "https://api.sportradar.com/wnba";

let scheduleCache = { key: null, expires: 0, data: null, access: null };

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

function scheduleGameMatches(prediction, game) {
  const homeName = cleanTeamName(game?.home?.name || game?.home?.market || "");
  const awayName = cleanTeamName(game?.away?.name || game?.away?.market || "");
  return homeName === cleanTeamName(prediction.home_team) && awayName === cleanTeamName(prediction.away_team);
}

function shouldFetchBoxscore(game) {
  const status = String(game?.status || "").toLowerCase();
  if (["inprogress", "halftime", "complete"].includes(status)) return true;
  if (status === "closed") return false;
  const scheduled = Date.parse(game?.scheduled || "");
  if (!Number.isFinite(scheduled)) return false;
  const delta = Date.now() - scheduled;
  return delta > -15 * 60 * 1000 && delta < 5 * 60 * 60 * 1000;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} from ${url}`);
  return response.json();
}

async function fetchStaticLatest() {
  return fetchJson(`${STATIC_BASE}/latest.json?t=${Date.now()}`, {
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

async function sportradarRequest(pathBuilder, apiKey, preferredAccess = null) {
  if (!apiKey) throw new Error("SPORTRADAR_API_KEY is not configured in the Worker");
  const levels = preferredAccess ? [preferredAccess, preferredAccess === "trial" ? "production" : "trial"] : ["trial", "production"];
  let lastError = null;
  for (const access of levels) {
    const url = `${SPORTRADAR_BASE}/${access}/v8/en/${pathBuilder(access)}`;
    const response = await fetch(url, {
      headers: { "x-api-key": apiKey, "Cache-Control": "no-cache" },
      cf: { cacheTtl: 0, cacheEverything: false },
    });
    if (response.ok) return { data: await response.json(), access };
    lastError = new Error(`${response.status} from Sportradar ${access}`);
    if (![401, 403].includes(response.status)) throw lastError;
  }
  throw lastError || new Error("Sportradar request failed");
}

async function fetchSportradarSchedule(apiKey) {
  const p = chicagoDateParts();
  const key = `${p.year}-${p.month}-${p.day}`;
  if (scheduleCache.key === key && scheduleCache.data && scheduleCache.expires > Date.now()) return scheduleCache;
  const result = await sportradarRequest(
    () => `games/${p.year}/${p.month}/${p.day}/schedule.json`,
    apiKey,
    scheduleCache.access,
  );
  scheduleCache = { key, expires: Date.now() + 5 * 60 * 1000, data: result.data, access: result.access };
  return scheduleCache;
}

async function fetchSportradarBoxscore(gameId, apiKey, access) {
  return sportradarRequest(() => `games/${gameId}/boxscore.json`, apiKey, access);
}

function parseSportradarBoxscore(data) {
  const game = data?.game || data || {};
  const home = game.home || data?.home || {};
  const away = game.away || data?.away || {};
  const status = normalizeSportradarStatus(game.status);
  const homeScore = nullableNumber(home.points ?? game.home_points);
  const awayScore = nullableNumber(away.points ?? game.away_points);
  if (homeScore === null || awayScore === null) return null;
  return {
    status,
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
  prediction.live_period = live.period;
  prediction.live_clock = live.clock;
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
  const cache = await fetchSportradarSchedule(apiKey);
  const scheduleGames = Array.isArray(cache.data?.games) ? cache.data.games : [];
  const now = new Date().toISOString();
  let matched = 0;
  let refreshed = 0;
  for (const prediction of latest.games || []) {
    const scheduledGame = scheduleGames.find((g) => scheduleGameMatches(prediction, g));
    if (!scheduledGame) continue;
    matched += 1;
    if (!shouldFetchBoxscore(scheduledGame)) continue;
    const box = await fetchSportradarBoxscore(scheduledGame.id, apiKey, cache.access);
    const live = parseSportradarBoxscore(box.data);
    if (!live) continue;
    applyLiveState(prediction, live, now, "Sportradar via Cloudflare Worker");
    refreshed += 1;
  }
  latest.live_generated_at_utc = now;
  latest.last_live_update_utc = refreshed ? now : latest.last_live_update_utc;
  latest.live_delivery = "cloudflare-worker";
  latest.live_source = "sportradar";
  latest.live_source_status = "fresh";
  latest.live_source_matched_games = matched;
  latest.live_source_refreshed_games = refreshed;
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

async function buildLivePayload(env) {
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
    latest.live_source = "github-pages";
    latest.live_source_status = "fallback";
    latest.live_source_error = `Sportradar failed: ${sportradarError}; ESPN failed: ${String(error?.message || error)}`;
    return latest;
  }
}

async function diagnostics(env) {
  const result = {
    ok: true,
    service: "wnba-live-dashboard",
    now: new Date().toISOString(),
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
    };
  } catch (error) {
    result.ok = false;
    result.static_latest = { ok: false, error: String(error?.message || error) };
  }
  try {
    const cache = await fetchSportradarSchedule(env?.SPORTRADAR_API_KEY);
    result.sportradar = {
      ok: true,
      access: cache.access,
      games: Array.isArray(cache.data?.games) ? cache.data.games.length : 0,
      date: cache.key,
    };
  } catch (error) {
    result.sportradar = { ok: false, error: String(error?.message || error) };
  }
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

export default {
  async fetch(request, env) {
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
        return jsonResponse(await buildLivePayload(env));
      } catch (error) {
        return jsonResponse({ error: "Static dashboard data unavailable", detail: String(error?.message || error) }, 502);
      }
    }
    return proxyStatic(url.pathname);
  },
};
