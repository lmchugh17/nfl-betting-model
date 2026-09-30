# Task 16 — First Live-Season Calibration Review

**Run date:** 2026-09-30
**Coverage:** 2026 regular season, weeks 1–3 (season opened 2026-09-09)
**Status:** Diagnostic only — no constants changed, no retrain run. This is
input for a human-led follow-up conversation about whether/how to retune.

## Data note

`scripts/reconcile_predictions.py` reports **50** completed graded games in
`prediction_results`, which is past its 40-game review threshold. Of those,
**3 are leftover 2025-season postseason games** (season=2025, weeks 20–22 —
i.e. games from the prior season's playoffs, already graded before this
task's tracking window started on 2026-09-04) and **47 are genuine 2026
regular-season games** (weeks 1, 2, 3 — 16, 16, 15 games respectively). All
analysis below uses the **47 season-2026 games** unless noted, since that's
the actual live track record this task is meant to assess. Either way, the
40-game threshold is cleared.

## Record summary (season=2026, 47 games)

| Metric | Model | Baseline |
|---|---|---|
| Moneyline (straight-up) | **27-20 (57.4%)** | Always-favorite baseline: 31-16 (66.0%) |
| Against the spread | **19-27-1 (41.3%)** | Breakeven at standard juice: ~52.4% |
| Avg margin error | **11.11 pts** | Market's own avg error: 10.16 pts |

(For reference, `reconcile_predictions.py`'s raw output across all 50 rows,
including the 3 stale 2025 games: ML 29-21 / 58%, ATS 21-28-1 / 43%, avg
margin error 10.8 pts. The season-2026-only numbers above are barely
different — the 3 extra games don't change the picture.)

## Market / Polymarket comparison

- **Margin error:** model 11.11 vs market 10.16 → model is **0.95 points
  worse** than the market's own closing-line error on the same 47 games.
  That's a bit wider than the backtest holdout's gap (0.55 pts, see
  TASKS.md Task 9), but still the same direction and same order of
  magnitude — not alarming for a 3-week live sample where variance is much
  higher than in a multi-season backtest average.
- **Brier score** (win_prob_home vs actual, joined to Polymarket's
  home_prob on the 47 games where both exist): **model 0.2321 vs
  Polymarket 0.2295** — a gap of only 0.0026. The model is very slightly
  behind Polymarket here, not ahead and not tied in any meaningful sense,
  but it's a much smaller gap than the ATS/ML numbers might suggest.
- **Moneyline accuracy vs the trivial "always take the market favorite"
  baseline:** model 57.4% vs baseline 66.0% — the model is **8.6pp behind**
  a baseline that requires no model at all. That's a bigger gap than the
  backtest's 3.5pp, but again, n=47 live games carries much more noise
  than a multi-thousand-game backtest average.

**Verdict on "behind vs beating the market":** the model is landing
**behind** the market on every axis measured (margin error, Brier, and
raw moneyline accuracy vs the favorite baseline). That's the expected,
healthy outcome for a first live stretch — it is *not* matching or beating
the market, which would have been the surprising/concerning result. Nothing
here trips the "too good to be true" alarm.

## Home-field bias check

This is the one number worth flagging for a closer look, but it turns out
to point away from a model-specific problem, not toward one:

- Model favored the home team in **31/47 games (66.0%)**.
- The market favored the home team in **31/47 games (66.0%)** — the
  *identical* rate.
- Home teams actually won **27/47 games (57.4%)**.
- Mean signed error (predicted − actual): model **+1.24**, market **+1.80**
  (positive = over-favored the home side).

So both the model and the market over-favored home teams by a similar (the
market's actually larger) margin in this sample, and they picked the home
side to win at exactly the same rate. If `HOME_ADVANTAGE_ELO=40` were
meaningfully miscalibrated relative to today's NFL, we'd expect the model's
home-favoring rate to diverge from the market's — it doesn't. This reads
much more like early-season variance (small n, plus whatever leaguewide
home-field softening reporting has noted in recent years) than a model
constant that needs adjusting.

## Constant-by-constant read

**ELO constants (`src/elo.py`): `K_FACTOR=20`, `HOME_ADVANTAGE_ELO=40`,
`SEASON_REGRESSION_FACTOR=0.80`**

- `HOME_ADVANTAGE_ELO`: see above — model's home-favoring rate matches the
  market's exactly, so no evidence this is off.
- `SEASON_REGRESSION_FACTOR` (preseason regression toward the mean): the
  model's picks track the market closely enough (same home-favorite rate,
  similar margin magnitudes — avg |predicted_margin| 5.00 vs avg
  |market_spread| 4.76) that the preseason ratings don't look wildly out of
  step with market consensus. No red flag.
- `K_FACTOR=20`: only 1–3 games have been played per team this season, so
  there's been very little opportunity for K_FACTOR to move ratings much
  either way yet. There isn't enough signal in 3 weeks to say whether it's
  too fast or too slow — this needs more weeks of data before it's a fair
  question to ask.

**Rolling-window constants (`ROLLING_WINDOW=8` in `src/epa_features.py`,
`ATS_ROLLING_WINDOW=8`)**

- With only 1–3 games played this season, an 8-game rolling window is still
  **80–90% filled with 2025 data** for every team. These features haven't
  had a chance to show whether they're too slow or too fast to react to a
  team's actual 2026 form yet — there simply isn't enough new data flowing
  through the window to observe that behavior. This is a "check again in a
  few more weeks" item, not something today's numbers can speak to either
  way.

## Plain-language verdict

Nothing in this first batch looks obviously broken. The model is
underperforming the market on every metric checked (margin error, Brier
score, and raw accuracy vs. the "always take the favorite" baseline) — that
is exactly the expected, healthy pattern for a model with no live NFL track
record yet, not a warning sign. The most attention-grabbing number is the
41.3% ATS record, which is below both the 52.4% breakeven line and a coin
flip — but margin error and Brier being roughly in line with the market
suggests the model isn't fundamentally miscalibrated, so this is most
likely ordinary small-sample variance (n=46 decided ATS games) rather than
a structural issue. It's the one line worth re-checking first as more weeks
land.

The ELO and rolling-window constants show no clear signs of being
mistuned so far, but 3 weeks (47 games, 1–3 games per team) is genuinely
too little data to stress-test `K_FACTOR` or the 8-game rolling windows in
particular — those need more weeks of real 2026 games before a retune
decision would be well-grounded. **Recommendation: keep collecting data,
re-run this review after another 4-6 weeks (or per the existing 40-game
cadence), and hold off on any constant changes or retrain until then.**
No action needed from this review beyond continued monitoring.
