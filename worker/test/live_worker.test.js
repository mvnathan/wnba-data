import test from "node:test";
import assert from "node:assert/strict";

import {
  chicagoDateStamp,
  elapsedFraction,
  predictionDateStamps,
  projectLiveGame,
} from "../src/index.js";
import { mergeTennisScores, normalizeTennisEvent, shouldPollLive } from "../src/stateful_index.js";
import { buildAlerts } from "../src/alerts.js";
import { createVapidKeys } from "../src/push.js";

test("Chicago date stamping keeps late games on the prior date", () => {
  assert.equal(chicagoDateStamp(new Date("2026-08-31T04:30:00Z")), "20260830");
  const stamps = predictionDateStamps({
    games: [{ game_date_utc: "2026-08-31T04:30:00Z" }],
  });
  assert.ok(stamps.includes("20260830"));
});

test("live window spans the end of a late game", () => {
  const start = Date.parse("2026-08-31T03:00:00Z");
  const latest = { games: [{ game_date_utc: new Date(start).toISOString() }] };
  assert.equal(shouldPollLive(latest, start - 44 * 60 * 1000), true);
  assert.equal(shouldPollLive(latest, start + 3.5 * 60 * 60 * 1000), true);
  assert.equal(shouldPollLive(latest, start + 4.1 * 60 * 60 * 1000), false);
});

test("live projection reaches the observed final score", () => {
  assert.equal(elapsedFraction(3, "05:00"), 0.625);
  const projection = projectLiveGame(
    { status: "STATUS_FINAL", home_score: 88, away_score: 81 },
    { home_score: 84, away_score: 82, home_win_probability: 0.55 },
  );
  assert.equal(projection.final_margin, 7);
  assert.equal(projection.final_total, 169);
  assert.equal(projection.home_win_probability, 1);
});

test("tennis score overlay preserves set and serving state", () => {
  const live = normalizeTennisEvent({
    Eid: "42", T1: [{ Nm: "Carlos Alcaraz" }], T2: [{ Nm: "Jannik Sinner" }],
    Tr1: "1", Tr2: "0", Tr1S1: "6", Tr2S1: "4", Tr1S2: "2", Tr2S2: "1",
    Tr1G: "30", Tr2G: "15", Eps: "S2", Esrv: 1,
  });
  assert.equal(live.live_state, "in");
  assert.equal(live.set_scores.length, 2);
  const [merged] = mergeTennisScores(
    [{ player_1: "Carlos Alcaraz", player_2: "Jannik Sinner", predicted_winner: "Carlos Alcaraz" }],
    { events: [live] },
  );
  assert.equal(merged.player_1_sets, 1);
  assert.equal(merged.player_1_game_points, "30");
  assert.equal(merged.serving_player, 1);
});

test("balanced alert thresholds select meaningful market gaps", () => {
  const alerts = buildAlerts(
    { games: [{ game_id: "w1", away_abbr: "NY", home_abbr: "MIN", model_market_total_edge: 6.5, model_predicted_total: 166.5, market_total: 160 }] },
    { matches: [{ match_id: "t1", tour: "ATP", player_1: "A", player_2: "B", predicted_game_margin_player_1: 5, market_margin_player_1: 0.5, predicted_total_games: 35, market_total_games: 33, player_1_win_probability: .7, market_player_1_probability: .52 }] },
  );
  assert.deepEqual(alerts.map(alert => alert.id).sort(), ["tennis:t1:spread", "tennis:t1:winner", "wnba:w1:total"]);
});

test("VAPID keys are generated in browser push format", async () => {
  const keys = await createVapidKeys();
  assert.equal(keys.publicKey.length, 87);
  assert.equal(keys.privateJwk.crv, "P-256");
});
