# NFL Betting Model — Task List

Companion to [MAP.md](MAP.md). This is the build order: each task is self-contained,
has an acceptance check, and names the CFB file it ports from. Follow
[[feedback_incremental_changes]] — build one piece, verify it against known real
data, then move to the next.

**Source of truth for data:** [nflverse](https://nflreadr.nflverse.com/) via
**`nflreadpy`** (the `nfl_data_py` package is deprecated as of 2025 — nflverse
tells new projects to use `nflreadpy`). `nflreadpy` is Polars-based and needs
Python ≥ 3.10; call `.to_pandas()` at every load boundary so the rest of the code
stays pandas like CFB.

**Verified facts baked into this plan (checked 2026-09-03):**
- `games.csv` (Lee Sharpe / `load_schedules()`) covers **1999 → present, plus the
  full upcoming schedule** (2026 Week 18 already listed with null scores/lines) —
  so it is both the training table and the upcoming-slate table.
- Exact `games.csv` columns: `game_id, season, game_type, week, gameday, weekday,
  gametime, away_team, away_score, home_team, home_score, location, result, total,
  overtime, old_game_id, gsis, nfl_detail_id, pfr, pff, espn, ftn, away_rest,
  home_rest, away_moneyline, home_moneyline, spread_line, away_spread_odds,
  home_spread_odds, total_line, under_odds, over_odds, div_game, roof, surface,
  temp, wind, away_qb_id, home_qb_id, away_qb_name, home_qb_name, away_coach,
  home_coach, referee, stadium_id, stadium`.
- Betting odds coverage: `spread_line` / `total_line` present from 1999;
  moneylines + spread/total juice **~complete from 2010 on** (2669/2670 rows in
  the 2010s), so the **2016+ window has full odds**. `temp`/`wind` populated only
  for `roof ∈ {outdoors, open}`.
- nflverse start years: pbp 1999, injuries 2009, depth_charts 2001, snap_counts
  2012, participation 2016, nextgen_stats 2016, officials 2015, rosters_weekly
  2002, ftn_charting 2022.
- `game_type` values: `REG`, `WC`, `DIV`, `CON`, `SB` (no preseason anywhere —
  not in `games.csv`, and the injury feed has no `PRE` rows either; preseason
  injuries surface on the Week 1 report).
- `week`: 1–18 regular season (1–17 before 2021), then 19=WC, 20=DIV, 21=CON,
  22=SB. 2021+ = 18 weeks / 17 games; 1999–2020 = 17 weeks / 16 games.

**Train/holdout split (updated — a year passed since MAP was written):**
train **2016–2024**, holdout **2025** (full season), **2026 as the live rolling
test**. Mirrors CFB's full-season-holdout approach.

**GitHub account / hosting:** new repo `github.com/lmchugh17/nfl-betting-model`,
Pages from `/docs` on `master`. `gh` CLI not set up on this Mac — Task 0 handles
that. Verify the Claude GitHub App can push with a manual routine run before
trusting cron (CFB lesson: a failed push is silent and lossy).

---

## Task 0 — Repo + tooling prerequisites

**Goal:** empty scaffolded repo, local Python env, `gh` working.

- `~/Documents/Luke/Claude/NFL Betting Model/` → `git init`, `.venv` (Python
  3.12), `.gitignore` (`.venv/`, `.env`, `data/*.db`, `data/*.parquet`,
  `__pycache__/`, `.DS_Store`, `models/*.pkl`; **negate** the archive CSVs:
  `!data/live_odds_archive.csv`).
- `requirements.txt`: `nflreadpy`, `pandas`, `numpy`, `scikit-learn`, `xgboost`,
  `lightgbm`, `shap`, `requests`, `python-dotenv`. (`brew install libomp` first —
  xgboost needs it on this Mac, per the CFB build.)
- `.env.example`: `ODDS_API_KEY=` (that is the only secret — nflverse + Open-Meteo
  need no key). Real `.env` reuses the CFB Odds API key.
- Install/auth `gh` CLI (`brew install gh` → `gh auth login` as `lmchugh17`).
- Create the GitHub repo (private to start), push the empty scaffold.
- **Check:** `gh repo view lmchugh17/nfl-betting-model` works; `python -c "import
  nflreadpy, xgboost, lightgbm, shap"` clean.

## Task 1 — DB schema (`src/db.py`)

**Ports from:** CFB `src/db.py`.

**Goal:** two SQLite files, schema created idempotently.

- **`data/nfl_stats.db`** — written only by GitHub Actions. Tables: `teams`,
  `games`, `team_game_epa`, `injuries`, `depth_charts`, `snap_counts`,
  `live_odds`, `game_features`.
- **`data/nfl_predictions.db`** — written only by the Claude routine. Tables:
  `predictions`; view `prediction_results`. This DB `ATTACH`es `nfl_stats.db`
  read-only for the join, or the view is built in a thin wrapper that opens both.
- `teams`: 32 rows. PK `team` (abbr, e.g. `KC`). Cols: `full_name`, `conference`
  (`AFC`/`NFC`), `division` (`AFC West` …), `franchise_id` (folds `OAK`+`LV`,
  `SD`+`LAC`, `STL`+`LAR`), `espn_id` (for optional injury cross-check).
- `games`: near 1:1 with `games.csv` (columns listed above) + derived
  `home_margin`, `home_win`, `season_type` normalised to `PRE`/`REG`/`POST`. This
  table **absorbs** CFB's separate `lines` and `game_weather` tables.
- `team_game_epa`: PK `(game_id, team)`. Cols (all "for this team in this game"):
  `off_epa_play`, `def_epa_play`, `off_pass_epa`, `off_rush_epa`,
  `off_early_down_epa`, `off_success_rate`, `def_success_rate`,
  `explosive_play_rate`, `def_explosive_rate`, `rz_td_pct`, `pressure_rate_def`,
  `plays`, `sec_per_play` (pace), `pass_rate`, `neutral_pass_rate`.
- `injuries`: PK `(season, week, team, gsis_id)`. Cols `player_name`, `position`,
  `report_status`, `practice_status`, `date_modified`, `scraped_at`. Same
  append-safe + prune design as CFB (retention plan **from day one**, not bolted
  on — architecture lesson).
- `depth_charts`: PK `(season, week, team, position, depth_team)` → `gsis_id`,
  `player_name`. `snap_counts`: PK `(game_id, pfr_player_id)` → `offense_snaps`,
  `offense_pct`, `defense_pct`.
- `live_odds`: copy CFB verbatim (PK includes `scraped_at` → append-only
  snapshots), just `sport='americanfootball_nfl'`.
- `predictions` / `prediction_results`: copy CFB **verbatim** including the SHAP
  `highlights_json` / `tldr` / `bullets_json` / `model_breakdown_json` /
  `cover_probability` / `kelly_fraction` columns and the always-live view. Add
  moneyline-pick vs spread-pick as **two separate tracked fields** (CFB added this
  later — port it in from the start). Drop CFB's `low_sample_team*` columns (no
  NFL equivalent — memory note). Keep `min_current_season_games`.
- **DONE (2026-09-03).** `src/db.py`: `get_stats_connection()` /
  `get_pred_connection()` / `init_all()`. **`prediction_results` is a per-connection
  `CREATE TEMP VIEW`, not a persistent view** — SQLite forbids a persistent view
  from referencing an ATTACHed DB, and `nfl_predictions.db` must join
  `stats.games` across the split. `get_pred_connection()` attaches `nfl_stats.db`
  as `stats` and recreates the temp view every connection (also kills the CFB
  stale-view bug for free). Verified: both files init idempotently (52-col
  `games`, 9 stats tables), cross-DB view returns correct
  `actual_margin`/`moneyline_pick_won`/`pick_covered`/`margin_error`, no
  persistent view leaks into the pred DB file.

## Task 2 — Schedules → `teams` + `games` backfill (`scripts/backfill_schedules.py`, `src/nflverse_client.py`, `src/team_names.py`)

**Ports from:** CFB `scripts/backfill.py`, `src/cfbd_client.py`, `src/team_names.py`.

**Goal:** `games` fully populated 1999–2026 (all `game_type`s), `teams` populated
with conference/division/franchise mapping.

- `src/nflverse_client.py`: thin wrappers — `schedules()`, `pbp(years)`,
  `injuries(years)`, `depth_charts(years)`, `snap_counts(years)`,
  `rosters_weekly(years)` — each returns a pandas DataFrame (`.to_pandas()` at the
  boundary). Local parquet cache under `data/cache/` so re-runs during dev don't
  re-download.
- `src/team_names.py`: generalise CFB's `build_school_only_lookup()` to any
  plain-name source. Needed because The Odds API returns full names
  ("Kansas City Chiefs"), nflverse uses abbreviations (`KC`), ESPN differs again.
  Build `ODDS_API_NAME → team` and `ESPN_NAME → team` maps. Relocation aliases:
  `{OAK, LV} → franchise LV`, `{SD, LAC} → franchise LAC`, `{STL, LAR} → franchise
  LAR`; also `LA`→`LAR` (nflverse historical quirk), `WAS`/`WSH`.
- `teams` seed: hardcode the 32-row table (abbr, name, conf, div, franchise_id) —
  it changes maybe once a decade; don't scrape it. `espn_id` via a one-off manual
  map.
- Backfill: `load_schedules()` → filter to `season >= 1999` → insert every row
  into `games` (`INSERT OR REPLACE` on `game_id`). Keep completed **and** future
  rows (future rows have null scores — that is the upcoming slate for prediction).
- Normalise `season_type`: `REG`→`REG`, `{WC,DIV,CON,SB}`→`POST`. (No preseason
  in nflverse data at all — see Task 4.)
- **DONE (2026-09-03).** `src/nflverse_client.py` (Polars→pandas wrappers,
  filesystem cache at `data/cache/`, `cache_dir` must be a `Path`),
  `src/team_names.py` (`FRANCHISE_ALIASES` STL→LA / SD→LAC / OAK→LV / LAR→LA —
  **nflverse `schedules` uses `LA` not `LAR` for the Rams**; `build_name_lookup`
  for Odds-API/Polymarket names), `scripts/backfill_schedules.py`.
  Verified: 7,548 schedule rows → 7,276 completed + 272 upcoming; 35 teams with
  correct `franchise_id` folding; season counts match nflverse (267 for 2016–19,
  269 for 2020, 285 for 2021–25); SB LVIII `2023_22_SF_KC` KC 25 SF 22 at
  Allegiant `is_dome=1` `neutral_site=1`; 2016 `SD` home games fold to franchise
  `LAC`; 2026 Week 18 rows present with null scores; **0 of 2,761** 2016+
  completed games missing `spread_line`/`total_line`, 1 missing a moneyline.
  Dome/closed/open-roof games correctly carry null `temp`/`wind`.

## Task 3 — Play-by-play → `team_game_epa` aggregation (`scripts/backfill_epa.py`, `src/epa_features.py`)

**Ports from:** nothing — new capability (CFB had only box scores). Aggregation
pattern mirrors CFB `src/box_score_features.py`.

**Goal:** `team_game_epa` populated for every 2016+ regular + postseason game.
**Raw pbp is never stored.**

- `src/epa_features.py`: given a pbp DataFrame for a set of games, return one row
  per (game, team) with the `team_game_epa` columns. Filter to `play_type ∈
  {pass, run}` for EPA/play; use `qb_epa` for the pass split; `success = epa > 0`;
  explosive = `yards_gained >= 20` (pass) / `>= 12` (rush) — **but tune these
  thresholds against the real distribution, don't assume** (weather-threshold
  lesson). "early down" = `down ∈ {1,2}`. Neutral pass rate = `wp` between .20 and
  .80 and `half_seconds_remaining > 120`.
- `scripts/backfill_epa.py`: loop seasons 2016→2026, `load_pbp([year])`, aggregate,
  write, **discard the raw frame**. ~2–3 GB transient per season; fine on the
  Actions runner, never committed.
- **DONE (2026-09-03).** `src/epa_features.py` (`aggregate_team_game_epa()`,
  standard `(pass|rush) & ~kneel & ~spike` denominator, explosive = pass ≥ 15 /
  rush ≥ 10 yds from the real 2021–23 distribution, neutral script = wp∈[.2,.8] &
  half > 2:00), `scripts/backfill_epa.py` (one season at a time, ~6 min total for
  2016–25, raw pbp discarded; clamps `end` to nflverse's `get_current_season()`
  since `load_pbp` lags `load_schedules` — 2026 pbp isn't published yet, script
  skips it cleanly).
  **`team_game_epa`: 5,522 rows, 2016–2025** (534/season pre-2021, 570 after).
  Checks (note: same-game EPA vs same-game margin is ~1.0 *by construction* —
  it's a box-score-style raw table, not a feature yet, so the real test uses
  *trailing* EPA): corr(trailing net-EPA edge, home_margin) = **0.311**,
  appropriately below the market's spread↔margin corr of **0.443** (healthy —
  a raw feature matching the market would mean leakage); corr(season net-EPA,
  season wins) = **0.856**; 2023 offense leaders SF/DAL/BUF/GB/DET and defenses
  CLE/BAL/NYJ all match reality. Nulls: `rz_td_pct` (160, games with 0 RZ trips),
  `neutral_pass_rate` (290, blowouts with no neutral-script snaps) — left NULL
  for Task 8 to handle.

## Task 4 — Injuries + snap counts + availability (`scripts/backfill_availability.py`, `scripts/scrape_injuries.py`, `src/availability.py`)

**Ports from:** CFB `scripts/scrape_injuries.py` (but NFL data is *official and
reliable*, unlike CFB's noisy ESPN scrape).

**DONE (2026-09-03).**

- `scripts/backfill_availability.py` — `players` crosswalk (nflverse
  `load_players`, gsis_id↔pfr_id, 25k rows), `injuries` (2016–2025, ~56k rows),
  `snap_counts` (2016–2025). **`snap_counts` is filtered to real role players
  (`offense_pct` or `defense_pct` ≥ 0.10)** — the 1-snap-cameo tail is ~half the
  rows and never matters for starter ID.
- **`depth_charts` was dropped entirely.** nflverse depth charts are 40+ MB,
  redundant with snap share, and have **no 2025 data** (they lag badly
  in-season). `src/availability.py` is snap-count-only; the week-1 fallback is
  the prior season's snap leaders. This kept the DB at ~15 MB instead of ~95 MB.
- **Preseason correction:** the injury feed has **no `PRE` rows** at all — the
  official report only exists in REG+POST. A player hurt in preseason first
  appears on the Week 1 report, so no separate preseason handling is needed
  (the MAP/memory note assumed PRE rows exist; they don't).
- **IR blind spot (real finding):** a season-ending injury moves a player to IR
  and they **vanish from the injury report** (Rodgers 2023 is listed only from
  Week 13, not Weeks 2–12; Burrow 2023 never appears after his Week 11 wrist).
  So `availability.py` treats "unavailable" as *listed Out/Doubtful* **OR** *a
  projected starter who didn't dress in the team's last 1–2 games* (absence of a
  `snap_counts` row = didn't dress). This catches what the report hides.
- `src/availability.py`:
  - `projected_starters(season, week, team)` — trailing snap share ≥ 0.55 over
    last 4 games (prior season for week 1).
  - `trailing_snap_share`, `recent_availability` (active/reduced/absent).
  - `injury_burden(season, week, team)` → `{burden, out_starters, detail}`.
    Position-weighted (`QB` 4.0, trench/CB ~1.2, skill ~1.0), report weight
    (Out 1.0 / Doubtful 0.6 / not-dressed 1.0). **QBs excluded** — handled by
    `qb_situation` instead, to avoid double-modeling the position.
  - `qb_situation(season, week, team, projected_qb_gsis)` → flags
    `backup_starting` (projected QB ≠ season snap leader) + the projected QB's
    trailing snap share.
- `scripts/scrape_injuries.py` — in-season re-pull of `injuries` + `snap_counts`
  for the current season (idempotent upsert; nflverse serves latest status).
  Guards a missing `date_modified` column in current-season pulls.
- Constants (snap thresholds, position weights, report weights) are starting
  points — Task 8 tunes them against holdout performance.
- **Verified:** `injury_burden` over 881 team-weeks (2020–24) mean ≈ 3.2,
  median ≈ 3.0, p90 ≈ 5.7, genuinely-healthy weeks ≈ 0 (KC 2024 wk3 = 0.0);
  Burrow 2023 wk14 caught as `not_dressed`; `qb_situation` fires
  `backup_starting=True` for 2022 SF wk15 (Purdy for Garoppolo) and 2023 CIN
  wk14 (Browning for Burrow). DB indexes added for the per-team-week lookups.

## Task 5 — Weather forecast for upcoming games (`scripts/pull_weather_forecast.py`, `src/weather_client.py`)

**Ports from:** CFB `scripts/pull_weather_forecast.py`, `src/weather_client.py` —
**near-verbatim**.

**Goal:** forecast `temp`/`wind` for upcoming **outdoor** games (nflverse
`games.csv` only fills `temp`/`wind` *after* kickoff).

- Historical weather backfill is **not needed** — `games.csv` already carries
  `temp`/`wind`/`roof` for all past games. This task is upcoming-only.
- **DONE (2026-09-04).** `src/stadiums.py` — static Python dict (not a DB table;
  dropped the planned `stadiums` table), 39 venues, keyed by **venue name**, not
  `stadium_id`. Two real reasons this had to be name-keyed: (1) an international
  "home" game keeps its usual franchise `stadium_id` while only the venue *name*
  changes — `2026_05_PHI_JAX` is `stadium_id='JAX00'` but
  `stadium='Tottenham Hotspur Stadium'` (Jacksonville's London home-game deal);
  keying on `stadium_id` would have put that game's forecast in Jacksonville.
  (2) The same physical stadium gets tagged under different `stadium_id`s
  depending which team is designated home (Tottenham Hotspur Stadium as both
  `JAX00` and `LON02`) — name-keying unifies these automatically.
  `NAME_ALIASES` folds sponsorship renames (Highmark/New Era, Huntington
  Bank/FirstEnergy, etc.) to one canonical entry.
- **Real data-quality finding:** nflverse's own `roof` column is unreliable for
  several international venues — Melbourne Cricket Ground (a fully open cricket
  ground, no roof) is tagged `'dome'`; Stade de France (open-air field) is
  tagged `'dome'`; Allianz Arena/FC Bayern Munich Stadium (the same physical
  building) is tagged `'outdoors'` in one row and `'dome'` in another. Added
  `FORCE_ROOF_OVERRIDE` — for these three venues only, `stadiums.py`'s
  hand-verified `roof_default` **overrides** `games.roof` instead of just
  filling nulls. Domestic US stadiums keep trusting nflverse's own `roof` (it
  checks out against reality there — AT&T/NRG/Lucas Oil retractable roofs
  correctly vary game to game).
  Also confirmed **`roof` is null on several upcoming rows even for known
  domestic dome/retractable stadiums** (ATL/DAL/HOU/IND/PHO) until nflverse
  finalizes the schedule closer to kickoff — `roof_default` fills these too, so
  `is_dome` is always derivable regardless of source-data gaps.
- `src/weather_client.py` (Open-Meteo forecast endpoint, ported near-verbatim,
  historical/archive functions dropped since not needed), `scripts/pull_weather_forecast.py`
  — writes straight into `games.temp`/`games.wind`/`weather_source` (no separate
  weather table, unlike CFB's `game_weather`) for rows within the horizon and
  still upcoming; a completed game gets nflverse's own recorded conditions on
  the next `backfill_schedules.py` run, overwriting the forecast automatically.
  `gametime` is confirmed **local to the stadium** (a 13:00 slate shows 13:00 for
  both Eastern- and Central-zone home teams), so `kickoff_utc_iso()` localizes
  via each stadium's IANA timezone before matching Open-Meteo's UTC hourly series.
- **Verified (2026-09-04, live run):** 31 upcoming games fall within the 16-day
  horizon; 7 had `is_dome` set via fallback/override (6 null-roof domestic +
  Melbourne); 11 outdoor games got a forecast (temp 52–99°F, wind 2–13 mph — all
  plausible for mid-September); Melbourne (season-opener, 2026-09-10, spring
  there) correctly resolved to `is_dome=0` with a 52.1°F forecast, not the
  dome nflverse's own data implies. The literal "known cold-weather Week 15"
  check isn't reachable at build time (3+ months outside the forecast horizon)
  — deferred to an in-season check once the weekly pull is live.

## Task 6 — Live odds pull (`scripts/pull_odds.py`, `src/odds_client.py`)

**Ports from:** CFB `scripts/pull_odds.py`, `src/odds_client.py` — near-verbatim,
just the sport key.

**Goal:** intra-week line-movement snapshots in `live_odds`.

- The Odds API, `sport=americanfootball_nfl`, markets `spreads,totals,h2h`,
  region `us`, **reuse the existing CFB account key**. 3 credits/pull.
- NFL slate is ~14–16 games/week (vs CFB ~60) → pulls are cheap. Cadence: Wed /
  Fri / Sat / Sun-AM (~12 credits/week, ~220/season — well within free tier,
  leaves headroom for CFB on the same key).
- Append-only into `live_odds` (PK includes `scraped_at`), team-name match via
  `src/team_names.py` Odds-API map.
- `src/spread_pricing.py` (port from CFB): real per-book median price for the
  pick side, used in the Kelly calc instead of assumed −110. For the NFL we also
  have closing juice in `games.{away,home}_spread_odds` for backtests.
- **Check:** one pull writes ~14–16 games × N books; team names 100% matched;
  a second pull hours later shows at least one line that moved.

## Task 7 — `scripts/prune_live_data.py`

**Ports from:** CFB `scripts/prune_live_data.py` — verbatim.

- Retention windows: `live_odds` 3 days past `commence_time`, `injuries` 21 days.
- Archive pruned rows to append-only `data/live_odds_archive.csv` /
  `data/injuries_archive.csv` (git-tracked via `.gitignore` negation — CFB caught
  this: Actions runners have no persistent disk, un-negated archives vanish).
- **Check:** run twice — second run is a no-op; archive CSVs grow, DB shrinks.

## Task 8 — Feature engineering (`scripts/build_features.py` + `src/` modules)

**Ports from:** CFB `scripts/build_features.py`, `src/elo.py`,
`src/opponent_adjustment.py`, `src/ats_and_situational.py`,
`src/weather_features.py`.

**Goal:** `game_features` table, one row per completed 2016+ `REG`/`POST` game.

- **`src/elo.py`** — port the formula, recalibrate (all need backtesting;
  starting points): `K_FACTOR` 40 → **20**; `SEASON_REGRESSION_FACTOR` 0.6 →
  **0.80** (low NFL roster turnover); `HOME_ADVANTAGE_ELO` 65 → **40** (~1.7 pts,
  and declining) with a **2020 no-fans downweight/flag**. Ratings key on
  `franchise_id`, not `team`.
- **`src/opponent_adjustment.py`** — port iterative SRS as-is. Less load-bearing
  (17-game formula-balanced schedule) but cheap; keep it.
- **`src/epa_features.py`** rolling form — trailing-N-game EPA aggregates per team
  (off/def EPA/play, pass vs rush, early-down, success rate, explosive rate),
  opponent-adjusted by the same SRS-style pass used for CFB box scores. Window:
  start at **8 games**, tune (17-game season affords a longer window than CFB's
  4). Season-boundary carryover regressed like ELO.
- **`src/ats_and_situational.py`** — port directly. NFL bonus: **divisional teams
  play twice/season**, so same-season H2H (Week 3 → Week 15 rematch) is strong
  signal — H2H features matter more than in CFB. Situational flags to add:
  short week (`rest <= 4`, Thursday), off bye (`rest >= 13`), **opponent** off
  bye, prime-time (from `gametime`/`weekday`), `div_game`, ≥ 2 time-zone travel
  (stadium long/long delta), international game (`location`/`stadium_id`).
- **`src/weather_features.py`** — port as-is (already sport-agnostic).
  **Re-tune adverse thresholds against NFL 2016–2025 history** — do not reuse
  CFB's 40°F/20mph/0.1in. Pull the real distribution, pick thresholds leaving
  enough same-condition history per franchise to be learnable (CFB weather
  lesson).
- **QB availability** + **injury burden** features from `src/availability.py`
  (Task 4).
- **New-HC / new-OC** flag from year-over-year `games.{home,away}_coach`.
- `scripts/build_features.py` assembles all of the above into `game_features`
  (mirrors CFB's assembler). Exclude `spread_line`/`total_line` from feature
  columns — post-hoc edge only.
- **Check:** every feature correlates with actual margin in the correct sign;
  `spread_line` benchmark strongest (~0.65–0.72), engineered features
  appropriately weaker (~0.4–0.55); no feature matches/beats the market
  (= leakage bug). Season-length logic handles 16 vs 17 game seasons.

## Task 9 — Model (`src/model.py`, `scripts/train_model.py`)

**Ports from:** CFB `src/model.py`, `scripts/train_model.py` — architecture
**verbatim**.

- Same 5-model stack (LogReg / RF / XGB / LightGBM / ExtraTrees → logistic
  meta-learner on `TimeSeriesSplit` OOF) → P(home win); separate `XGBRegressor` →
  predicted margin. `spread_line` excluded from training, used only post-hoc for
  `edge = predicted_margin - (-spread_line)`.
- Retune: `N_SPLITS` (more seasons available than CFB), tree depths for the
  larger, cleaner NFL sample.
- Split: train **2016–2024** (~2,400 games), holdout **2025** (285 games) full
  season, **2026** live rolling. Save bundle to `models/nfl_model.pkl`
  (gitignored).
- **Check (reality test vs market):** classifier accuracy should land in the
  **62–68%** range (NFL is closer to a coin flip than CFB's 76%) and **just
  behind** market-favourite accuracy; regressor MAE **just above** the market's
  (~ market ± 0.5). Landing *ahead* of the market on a first pass = leakage bug,
  not success. Compare to baselines: home-team-always (~57% modern NFL),
  ELO-only, spread-sign-only.

## Task 10 — Explanations (`src/explain.py`)

**Ports from:** CFB `src/explain.py` — **verbatim** except `_describe_feature`.

- `get_shap_contributions` (TreeSHAP on the margin regressor) unchanged.
- `_describe_feature` — add NFL feature-name cases: EPA form, QB availability
  ("Projected starter Jake Browning has a −0.05 EPA/dropback over his last 3
  starts vs the Bengals' usual +0.12"), injury burden, short week, off bye,
  divisional rematch, travel/time-zone.
- Prose (TL;DR + bullets) is **not** code — the scheduled Claude routine writes it
  natively from these facts each week (same as CFB).
- **Check:** run on a known real upset (e.g. 2023 Week 5 `Giants` were pick vs
  actual, or a QB-injury game) — highlights surface the QB/injury factor when it
  actually drove the prediction, and stay silent on it when it didn't.

## Task 11 — Live inference + predictions tracking (`scripts/predict_games.py`, `src/live_state.py`, `scripts/reconcile_predictions.py`)

**Ports from:** CFB `scripts/predict_games.py`, `src/live_state.py`,
`scripts/reconcile_predictions.py` — near-verbatim.

- `src/live_state.py` — current-state (not "entering game X") versions of every
  feature function: current ELO/SRS/rolling-EPA/ATS/availability. Watch for the
  two CFB bugs: (1) H2H must return even when the current game has no score yet;
  (2) rest-days / "last game" must scope to the current season.
- `scripts/predict_games.py` — upcoming slate from `games` (null-score rows) →
  live-state features → bundle → `edge` → cover prob (real juice from
  `spread_pricing` / `games.*_spread_odds`) → Kelly (25% fractional, **$500**
  starting paper bankroll unless changed) → SHAP highlights → `predictions`
  upsert. `--backtest` mode: exclude a target `game_id` from "completed" state to
  validate against known outcomes.
- Fix the CFB away-pick sign bug from the start: cover prob for an away pick is
  `1 - P(home covers)`.
- `scripts/reconcile_predictions.py` — W-L / ATS / avg margin error from the
  `prediction_results` view; flags when ≥ 40 completed 2026 games have
  accumulated as a retrain-review trigger.
- **Check:** backtest against the first 2–3 completed 2026 games, log hits **and**
  misses honestly; `predictions` round-trips; reconcile prints a sane record.

## Task 12 — Static site (`scripts/build_site.py`)

**Ports from:** CFB `scripts/build_site.py` — port the whole page, rebrand.

- Single self-contained dark HTML/CSS page → `docs/index.html`, no build step,
  GitHub Pages `/docs`. Sections: summary stat tiles, upcoming picks (TL;DR +
  bullets), recent results shown honestly (losses visible), per-model breakdown
  (collapsible), Kelly wager vs paper bankroll, confidence-tier badges, Weekly
  Performance table + Weekly Trends bar chart.
- Port the CFB CSS fixes: footnote-marker convention (`*`/`†`/`‡`/`§`/`¶`, scope
  `.footnote` CSS globally from the first one), marker positioning as a percentage
  *inside* `.bar-track` not a padded parent, value-label-inside-the-bar-fill.
- Footer: methodology, confidence-tier definitions, paper-bankroll/Kelly
  assumptions, "personal research, not financial advice".
- **Check:** browser-preview renders correctly including a displayed loss;
  responsive; no horizontal body scroll.

## Task 13 — Passive accuracy check: Polymarket (`src/polymarket_client.py`, `scripts/pull_polymarket.py`)

**Ports from:** CFB `src/polymarket_client.py`, `scripts/pull_polymarket.py`.

- Pull Polymarket's pre-game NFL win probability **once** per game, grade it by
  Brier score after settlement. **Never a live pick input** — a passive
  third-party benchmark shown alongside the model's own calibration.
- Needs Polymarket's NFL tag/series slug — confirm via `curl` against the live
  Gamma API, don't trust doc summaries (CFB discipline).
- **Check:** a settled 2026 game shows a Polymarket pre-game prob + a computed
  Brier score.

## Task 14 — GitHub Actions data pull (`.github/workflows/weekly_data_pull.yml`)

**Ports from:** CFB workflow — port the shape, fewer steps.

Steps: (1) nflverse refresh — schedules, pbp→`team_game_epa`, injuries, depth
charts, snap counts; (2) The Odds API pull → `live_odds`; (3) Open-Meteo forecast
for upcoming outdoor games; (4) `prune_live_data.py`; (5) rebuild `game_features`;
(6) commit + push **`data/nfl_stats.db`** + archive CSVs only.

- **Cadence:** cron **Wed / Fri / Sat / Sun 13:00 UTC** (`0 13 * * 3,5,6,0`).
  Wed = slate preview + opening lines; Fri = final injury designations; Sat/Sun =
  line movement. Note UTC vs US-Eastern DST drift in a comment (CFB did this).
- Only Actions ever writes `nfl_stats.db` → no Git LFS needed, no 403 problem.
- **Check:** `workflow_dispatch` manual run green end-to-end; commit appears;
  DB size stays < 20 MB after a full run.

## Task 15 — Scheduled Claude Code routine

**Ports from:** CFB task 11 routine.

- Runs ~1h after the **Wed** pull (write week's picks + explanations, rebuild
  site, commit `docs/` + `nfl_predictions.db`) and again after the **Fri** pull
  (refresh for final injury news / line movement).
- Explanations written natively from SHAP facts — no separate LLM API, no
  `ANTHROPIC_API_KEY`.
- **Verify GitHub App push access with a manual `RemoteTrigger run` BEFORE
  trusting the cron** — CFB lost a full pipeline run to a silent 403. Confirm the
  commit lands on `master` remote.
- **Check:** manual trigger produces a commit with real picks + prose on the
  live site; a second manual trigger cleanly updates.

## Task 16 — First live week + calibration review

- Run the full pipeline for a real 2026 week end-to-end.
- After ~40 completed 2026 games: run `reconcile_predictions.py`, review
  classifier calibration vs Polymarket, decide whether ELO/window constants need
  the first real retune (they were all starting-point guesses).
- Update [[project_nfl_betting_model]] memory with actual holdout numbers and any
  constant changes.

---

## Dependency order

```
0 → 1 → 2 → ┬→ 3 ─────────┐
            ├→ 4 ──────────┤
            ├→ 5 ──────────┤
            └→ 6 → 7 ──────┤
                           ▼
                     8 → 9 → 10 → 11 → 12 → 13
                                            │
                                    14 ─────┤
                                    15 ─────┘
                                       │
                                      16
```

Tasks 3/4/5/6 are independent once `games` exists — do them in any order or in
parallel. 8 needs 2+3+4+5. 14/15 need everything through 13.
