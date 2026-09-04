# NFL Betting Model

Opponent-adjusted NFL predictions vs. the market. A 5-model stacked ensemble
(logistic regression, random forest, XGBoost, LightGBM, extra trees) for win
probability plus a separate XGBoost regressor for predicted margin, trained on
2016–2024 and evaluated on a held-out 2025 season, with 2026 as a live rolling
test. Personal research project, not financial advice.

Port of the [CFB Betting Model](../CFB%20Betting%20Model/). See [MAP.md](MAP.md)
for the design and [TASKS.md](TASKS.md) for the build plan.

## Status

**Task 0 complete** — repo scaffolded, environment verified, `nflreadpy` schedule
pull confirmed working (1999–2026, 46 columns, full odds coverage from 2016). Next:
Task 1, the two-DB schema.

## Data sources

- **[nflverse](https://nflreadr.nflverse.com/)** via `nflreadpy` (no API key) —
  schedules (incl. historical betting lines + weather), play-by-play (EPA),
  official injury reports, depth charts, snap counts.
- **[The Odds API](https://the-odds-api.com/)** (`americanfootball_nfl`) — live
  intra-week line-movement snapshots only. Shares the CFB account key.
- **[Open-Meteo](https://open-meteo.com/)** (no key) — forecast weather for
  upcoming outdoor games.

## Layout

- `src/` — library modules (db, clients, feature engineering, model, explain)
- `scripts/` — backfill / pull / train / predict / build-site entry points
- `data/` — `nfl_stats.db` (GitHub Actions writes), `nfl_predictions.db` (Claude
  routine writes)
- `docs/` — generated static site, served by GitHub Pages from `/docs`
- `.github/workflows/` — weekly data pull
