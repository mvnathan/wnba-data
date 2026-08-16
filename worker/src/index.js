const STATIC_BASE = "https://mvnathan.github.io/wnba-data";
const ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard";

function chicagoDateStamp() {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  const map = Object.fromEntries(parts.map((p) => [p.type, p.value]));
  return `${map.year}${map.month}${map.day}`;
}

function finiteNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
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

function parseEvent(event) {
  const competition = Array.isArray(event?.competitions) ? event.competitions[0] : null;
  if (!competition || !Array.isArray(competition.competitors)) return null;
  const home = competition.competitors.find((c) => c?.homeAway === "home");
  const away = competition.competitors.find((c) => c?.homeAway === "away");
  if (!home || !away) return null;

  const statusInfo = event.status || {};
  const typeInfo = statusInfo.type || {};
  return {
    game_id: String(event.id || ""),
    status: typeInfo.name || "",
    status_detail: typeInfo.detail || typeInfo.shortDetail || "",
    period: statusInfo.period ?? typeInfo.period ?? null,
    clock: statusInfo.displayClock || statusInfo.clock || typeInfo.displayClock || typeInfo.clock || "",
    home_score: finiteNumber(home.score),
    away_score: finiteNumber(away.score),
    home_team_id: String(home.team?.id || ""),
    away_team_id: String(away.team?.id || ""),
  };
}

function isLiveOrFinal(status) {
  const text = String(status || "").toUpperCase();
  return ["IN_PROGRESS", "HALFTIME", "END_PERIOD", "FINAL"].some((m) => text.includes(m));
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      "User-Agent": "wnba-live-dashboard/1.0",
      "Cache-Control": "no-cache",
    },
    cf: { cacheTtl: 0, cacheEverything: false },
  });
  if (!response.ok) throw new Error(`${response.status} from ${url}`);
  return response.json();
}

async function buildLivePayload() {
  const stamp = chicagoDateStamp();
  const staticUrl = `${STATIC_BASE}/latest.json?t=${Date.now()}`;
  const espnUrl = `${ESPN_SCOREBOARD}?dates=${stamp}&_=${Date.now()}`;
  const [latest, scoreboard] = await Promise.all([fetchJson(staticUrl), fetchJson(espnUrl)]);

  const events = Array.isArray(scoreboard.events) ? scoreboard.events.map(parseEvent).filter(Boolean) : [];
  const byId = new Map(events.map((g) => [String(g.game_id), g]));
  const now = new Date().toISOString();
  const games = Array.isArray(latest.games) ? latest.games : [];

  for (const prediction of games) {
    const live = byId.get(String(prediction.game_id || ""));
    if (!live) continue;

    prediction.live_status = live.status;
    prediction.live_status_detail = live.status_detail;
    prediction.live_period = live.period;
    prediction.live_clock = live.clock;
    prediction.live_home_score = live.home_score;
    prediction.live_away_score = live.away_score;
    prediction.live_updated_at = now;
    prediction.status = live.status || prediction.status;
    prediction.status_detail = live.status_detail || prediction.status_detail;
    prediction.live_source = "ESPN via Cloudflare Worker";

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

  latest.games = games;
  latest.live_generated_at_utc = now;
  latest.last_live_update_utc = now;
  latest.live_delivery = "cloudflare-worker";
  return latest;
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
  const url = `${STATIC_BASE}${pathname === "/" ? "/" : pathname}${pathname.includes("?") ? "&" : "?"}t=${Date.now()}`;
  const upstream = await fetch(url, { cf: { cacheTtl: 0, cacheEverything: false } });
  const headers = new Headers(upstream.headers);
  headers.set("cache-control", "no-store, no-cache, must-revalidate");
  return new Response(upstream.body, { status: upstream.status, headers });
}

export default {
  async fetch(request) {
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

    if (url.pathname === "/health") {
      return jsonResponse({ ok: true, service: "wnba-live-dashboard", now: new Date().toISOString() });
    }

    if (url.pathname === "/latest.json" || url.pathname === "/api/live") {
      try {
        return jsonResponse(await buildLivePayload());
      } catch (error) {
        return jsonResponse({ error: "Live feed unavailable", detail: String(error?.message || error) }, 502);
      }
    }

    return proxyStatic(url.pathname);
  },
};
