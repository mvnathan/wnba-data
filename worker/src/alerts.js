const ODDS_ROOT = "https://api.the-odds-api.com/v4/sports";

function key(value) {
  return String(value || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .toLowerCase().replace(/[^a-z0-9]/g, "");
}

function median(values) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) return null;
  const middle = Math.floor(clean.length / 2);
  return clean.length % 2 ? clean[middle] : (clean[middle - 1] + clean[middle]) / 2;
}

function matchOdds(prediction, odds) {
  const one = key(prediction.player_1), two = key(prediction.player_2);
  return odds.find((game) => {
    const home = key(game.home_team), away = key(game.away_team);
    return (home === one && away === two) || (home === two && away === one);
  });
}

function tennisMarket(prediction, game) {
  if (!game) return {};
  const one = key(prediction.player_1);
  const spreads = [], totals = [], implied = [];
  let updated = null;
  for (const book of game.bookmakers || []) {
    updated = !updated || book.last_update > updated ? book.last_update : updated;
    for (const market of book.markets || []) {
      if (market.key === "spreads") {
        const outcome = market.outcomes?.find((item) => key(item.name) === one);
        if (outcome && Number.isFinite(Number(outcome.point))) spreads.push(-Number(outcome.point));
      } else if (market.key === "totals") {
        const outcome = market.outcomes?.find((item) => item.name === "Over");
        if (outcome && Number.isFinite(Number(outcome.point))) totals.push(Number(outcome.point));
      } else if (market.key === "h2h") {
        const outcomes = market.outcomes || [];
        const p1 = outcomes.find((item) => key(item.name) === one);
        const other = outcomes.find((item) => key(item.name) !== one);
        if (p1?.price > 1 && other?.price > 1) {
          const raw1 = 1 / Number(p1.price), raw2 = 1 / Number(other.price);
          implied.push(raw1 / (raw1 + raw2));
        }
      }
    }
  }
  return { market_margin_player_1: median(spreads), market_total_games: median(totals), market_player_1_probability: median(implied), market_updated_at: updated, market_book_count: (game.bookmakers || []).length };
}

export async function enrichTennisMarkets(payload, apiKey) {
  if (!apiKey) return { ...payload, market_data_status: [{ status: 503, error: "ODDS_API_KEY not configured" }] };
  const sports = ["tennis_atp_us_open", "tennis_wta_us_open"];
  const diagnostics = [];
  const results = await Promise.all(sports.map(async (sport) => {
    const url = new URL(`${ODDS_ROOT}/${sport}/odds`);
    url.searchParams.set("apiKey", apiKey); url.searchParams.set("regions", "us");
    url.searchParams.set("markets", "h2h,spreads,totals"); url.searchParams.set("oddsFormat", "decimal");
    const response = await fetch(url, { cf: { cacheTtl: 120, cacheEverything: true } });
    diagnostics.push({ sport, status: response.status, remaining_requests: response.headers.get("x-requests-remaining") });
    return response.ok ? response.json() : [];
  }));
  const odds = results.flat();
  return { ...payload, market_data_status: diagnostics, market_events: odds.length, matches: (payload.matches || []).map((match) => ({ ...match, ...tennisMarket(match, matchOdds(match, odds)) })) };
}

export function buildAlerts(wnba, tennis) {
  const alerts = [];
  for (const game of wnba?.games || []) {
    const matchup = `${game.away_abbr || game.away_team} @ ${game.home_abbr || game.home_team}`;
    const totalEdge = Number(game.model_market_total_edge ?? game.model_consensus_total_edge);
    const spreadEdge = Number(game.model_market_margin_edge ?? game.model_consensus_margin_edge);
    if (Number.isFinite(totalEdge) && Math.abs(totalEdge) >= 6) alerts.push({ id: `wnba:${game.game_id}:total`, sport: "WNBA", score: Math.abs(totalEdge), title: `${matchup}: ${totalEdge > 0 ? "Over" : "Under"} signal`, body: `Model ${Number(game.model_predicted_total ?? game.predicted_total).toFixed(1)} vs market ${Number(game.market_total ?? game.consensus_total).toFixed(1)} (${Math.abs(totalEdge).toFixed(1)}-point gap)`, url: "https://mvnathan.github.io/wnba-data/wnba.html" });
    if (Number.isFinite(spreadEdge) && Math.abs(spreadEdge) >= 4) alerts.push({ id: `wnba:${game.game_id}:spread`, sport: "WNBA", score: Math.abs(spreadEdge), title: `${matchup}: spread disagreement`, body: `Model-market gap is ${Math.abs(spreadEdge).toFixed(1)} points`, url: "https://mvnathan.github.io/wnba-data/wnba.html" });
  }
  for (const match of tennis?.matches || []) {
    const matchup = `${match.player_1} vs ${match.player_2}`;
    const spreadEdge = Number(match.predicted_game_margin_player_1) - Number(match.market_margin_player_1);
    const totalEdge = Number(match.predicted_total_games) - Number(match.market_total_games);
    const probabilityEdge = Number(match.player_1_win_probability) - Number(match.market_player_1_probability);
    if (Number.isFinite(spreadEdge) && Number.isFinite(Number(match.market_margin_player_1)) && Math.abs(spreadEdge) >= 4) alerts.push({ id: `tennis:${match.match_id}:spread`, sport: match.tour, score: Math.abs(spreadEdge), title: `${matchup}: spread disagreement`, body: `Model ${Number(match.predicted_game_margin_player_1).toFixed(1)} vs market ${Number(match.market_margin_player_1).toFixed(1)} (${Math.abs(spreadEdge).toFixed(1)}-game gap)`, url: "https://mvnathan.github.io/wnba-data/tennis.html" });
    if (Number.isFinite(totalEdge) && Number.isFinite(Number(match.market_total_games)) && Math.abs(totalEdge) >= 6) alerts.push({ id: `tennis:${match.match_id}:total`, sport: match.tour, score: Math.abs(totalEdge), title: `${matchup}: ${totalEdge > 0 ? "Over" : "Under"} signal`, body: `Model ${Number(match.predicted_total_games).toFixed(1)} vs market ${Number(match.market_total_games).toFixed(1)} (${Math.abs(totalEdge).toFixed(1)}-game gap)`, url: "https://mvnathan.github.io/wnba-data/tennis.html" });
    if (Number.isFinite(probabilityEdge) && Number.isFinite(Number(match.market_player_1_probability)) && Math.abs(probabilityEdge) >= .15) alerts.push({ id: `tennis:${match.match_id}:winner`, sport: match.tour, score: Math.abs(probabilityEdge) * 20, title: `${matchup}: winner disagreement`, body: `Model and market differ by ${(Math.abs(probabilityEdge) * 100).toFixed(0)} percentage points`, url: "https://mvnathan.github.io/wnba-data/tennis.html" });
  }
  return alerts;
}
