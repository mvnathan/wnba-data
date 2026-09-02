import test from "node:test";
import assert from "node:assert/strict";

import {
  chicagoDateStamp,
  elapsedFraction,
  predictionDateStamps,
  projectLiveGame,
} from "../src/index.js";
import { shouldPollLive } from "../src/stateful_index.js";

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
