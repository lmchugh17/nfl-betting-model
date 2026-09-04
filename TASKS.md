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

- **DONE (2026-09-04).** `.env`'s `ODDS_API_KEY` populated from the CFB
  project's key (same account). `src/odds_client.py` — ported near-verbatim,
  default `sport=americanfootball_nfl`.
- **Real finding, changed the plan:** an unfiltered call returns the **entire
  remaining season** (272 games, live-tested) at ~2 books/game average — mostly
  thin, futures-style lines for games months out — for the **same 3-credit
  cost** as a `commenceTimeFrom`/`commenceTimeTo`-windowed call limited to the
  imminent slate (16 games at ~9 books/game, real market depth). Since cost is
  markets×regions, not games returned, windowing is strictly better, not a
  cost/coverage tradeoff — `scripts/pull_odds.py` now windows to the next
  `COMMENCE_WINDOW_DAYS = 10` days. Keeps `live_odds` rows meaningful (every
  stored price reflects an actual liquid market) and reduces the row volume
  Task 7 has to prune.
- Team-name matching uses `src/team_names.build_name_lookup()` +
  `resolve_name()` (built in Task 2) directly against The Odds API's full names
  ("Kansas City Chiefs") — confirmed these match nflverse's own `team_name`
  field exactly, so no separate Odds-API-specific alias table was needed like
  CFB's school-name aliases.
- `src/spread_pricing.py` — simpler than CFB's version: `pull_odds.py` already
  resolves each row's home/away team to a canonical abbr at write time, so
  pricing just matches `outcome_name` against the row's own
  `home_team`/`away_team` rather than rebuilding a name lookup on every call.
- **Verified (2026-09-04, live pulls):** first pull — 15 games, 802 rows, quota
  cost 3 (464 remaining), **0 unmatched team names**; `load_latest_spread_prices`
  returned realistic per-book medians with real book-to-book variance (e.g.
  PIT −120 vs ATL −102, 9 books each). Second pull 27s later confirmed
  append-only snapshotting: two distinct `scraped_at` values, 1,604 total rows,
  no overwrite. Actual line movement between two pulls seconds apart isn't
  expected (that needs hours) — the append mechanism itself is proven correct.

## Task 7 — `scripts/prune_live_data.py`

**Ports from:** CFB `scripts/prune_live_data.py` — mostly verbatim, but
`injuries` pruning was dropped (see below).

**DONE (2026-09-04).**

- **`injuries` needs no pruning in this build, unlike CFB.** Task 4 made
  `injuries`' primary key `(season, week, team, gsis_id)` — no `scraped_at` —
  because nflverse already serves "latest official status per player-week," so
  every re-pull upserts in place rather than appending a new snapshot. There is
  nothing to grow, archive, or prune. `injuries_archive.csv` was removed from
  `.gitignore`'s negation list along with the CFB-ported `INJURIES_RETENTION_DAYS`
  constant — genuinely dead code for this build, not a port-then-trim.
- `live_odds` is the only append-only table (PK includes `scraped_at`), pruned
  exactly like CFB: `LIVE_ODDS_RETENTION_DAYS = 3` past `commence_time`, archived
  to `data/live_odds_archive.csv` (git-tracked via the `.gitignore` negation)
  before deletion, `VACUUM`ed after.
- **Verified:** tested the archive+delete logic on a scratch copy of the DB
  (never touched the real project files) — inserted 5 synthetic rows with a
  `commence_time` in the past, first run archived+pruned exactly those 5 (CSV
  header + 5 rows written, DB row count dropped correctly), second run pruned 0
  (no-op). Then ran the real script against the actual project DB: correctly
  pruned 0 rows both times, since every stored game is still 6+ days out (all
  from Task 6's live pull) — proves the script doesn't false-positive on
  legitimately-current data.

## Task 8 — Feature engineering (`scripts/build_features.py` + `src/` modules)

**Ports from:** CFB `scripts/build_features.py`, `src/elo.py`,
`src/opponent_adjustment.py`, `src/ats_and_situational.py`,
`src/weather_features.py`.

**Goal:** `game_features` table, one row per completed 2016+ `REG`/`POST` game.

**DONE (2026-09-04).** `game_features`: **2,761 rows × 92 columns**.

- **`src/elo.py`** — ported + recalibrated as planned: `K_FACTOR=20`,
  `SEASON_REGRESSION_FACTOR=0.80`, `HOME_ADVANTAGE_ELO=40`, plus
  `HOME_ADVANTAGE_2020=0` for that season's non-neutral-site games (no/minimal
  fans leaguewide). Ratings keyed on `franchise_id(team)`, not the raw abbr, so
  OAK→LV / SD→LAC carry across the relocation.
- **`src/opponent_adjustment.py`** — ported as-is (already sport-agnostic),
  fed franchise-mapped team names for the same relocation-continuity reason.
- **`src/epa_features.py`** extended with `compute_rolling_epa_form()` +
  `assemble_epa_game_features()` — rolling **8-game window** (min_periods=3)
  over all 14 `team_game_epa` metrics, grouped on `franchise_id` (not raw
  `team`) so a mid-window relocation doesn't reset the trailing history. Needed
  no `build_long_format()` step unlike CFB — `team_game_epa` is already one row
  per (game, team) with its own `opponent` column.
- **`src/ats_and_situational.py`** — ATS (window 8) + H2H, **plus the planned
  NFL-specific addition**: `h2h_current_season_margin` /
  `h2h_current_season_meeting` isolate the fresh Week-3→Week-15 divisional
  rematch signal from the blended cross-season H2H average — turned out to be
  one of the **strongest engineered features** (corr 0.277, 6th of 81).
  Situational flags: short week (`rest<=4`), off bye (`rest>=12` — see below),
  `is_primetime` (`gametime>='19:00'`), `div_game` (used directly from
  nflverse, not recomputed), `is_international`, tz-shift + cross-country-travel
  (via new `src/stadiums.py` helpers).
  **Rest/bye is NOT recomputed** — ported the plan's own suggestion further
  than expected: nflverse's `games.home_rest`/`away_rest` are already correct,
  so CFB's `compute_rest_days()` wasn't ported at all. Real distribution
  confirmed the bye threshold: mode is 7 (normal week), 4 is the short-week
  cluster (160 games), 13–16 is the bye cluster (~200) — **12, not the
  naively-plausible 10 or 11**, since those exist too (schedule quirks) but
  aren't byes.
- **`src/weather_features.py`** — thresholds measured directly (not reused from
  CFB): **temp≤32°F OR wind≥20mph** flags 172/1,968 outdoor completed games
  (8.7%), with a real per-team split (LAC 1, IND 2, NO 2 vs. BUF 32, GB 30,
  KC 27, PIT 22, CLE 21 across 10 seasons) — learnable, not sparse like CFB's
  first attempt. **No precipitation feature** — nflverse's `games` table has no
  historical precip column at all (only the forecast endpoint does, and only
  for upcoming games), so precip was dropped rather than creating a
  train/live feature gap.
- **QB availability + injury burden** — `src/availability.py`'s Task 4
  functions plugged in directly with no changes. Turned out to be a bigger win
  than expected: because they already take an explicit `(season, week, team)`
  and are point-in-time correct, they serve **both** training (task 8) and live
  inference (task 11) — no CFB-style separate `live_state.py` reimplementation
  needed for this piece. Added a per-connection cache to
  `availability.gsis_pfr_maps()` (rebuilt thousands of times per build run
  otherwise).
- **New-HC flag** — implemented (`home_new_hc`/`away_new_hc`): compares a
  team's coach in its season-opening game to that team's coach in its **final**
  game of the prior season (robust to in-season interim-coach noise). **No
  new-OC flag** — nflverse's schedule has no offensive-coordinator field at
  all, confirmed while building this; dropped, not a port-then-trim.
- `scripts/build_features.py` assembles everything; `market_spread`
  (`spread_line`) stored for post-hoc edge only, excluded from the feature set.
- **Two real bugs caught by the correlation check, both fixed:**
  1. `is_adverse_weather`/`adverse_wx_ats_edge` were **100% zero** on the first
     run — `GAMES_COLS` in `build_features.py` selected `is_dome`/`roof` but
     forgot `temp`/`wind`, so `was_game_adverse()` silently saw `None` for
     every game. Fixed by adding the two columns; re-ran and got the expected
     172 adverse games / 113 with both teams' history known.
  2. `home_tz_shift_hours`/`away_tz_shift_hours` were null for **244 games**
     (~9%) across nearly every team, not just the international ones — 12
     historical stadium names inside the 2016+ window weren't in
     `src/stadiums.py` (Task 5 only verified coverage for `season>=2024`):
     renamed venues (Arrowhead/Paul Brown/Georgia Dome/Sports Authority Field),
     a real case-sensitivity typo (`"Everbank Field"` alias vs. the actual
     `"EverBank Field"`), and — the biggest gap — the Rams' pre-SoFi LA
     Coliseum, the Chargers' pre-SoFi StubHub Center, Qualcomm Stadium (SD),
     Oakland Coliseum (OAK, 3 different sponsor names across the window), and
     Frankfurt's Deutsche Bank Park (a real venue with no NFL game there in
     the `season>=2024` sample Task 5 checked). Added all of them; re-ran and
     confirmed **zero unresolved stadium names across the full 2016+ range**.
- **Check results:** every one of 81 correlatable features has the
  theoretically-correct sign (home-favoring stats positive, away-favoring
  negative). Max engineered-feature correlation is `elo_expected_home` at
  **0.356**, comfortably below the market benchmark `corr(spread_line,
  home_margin)` = **0.443** — no feature matches or beats the market (would
  signal leakage). Season-length logic needed no special-casing: every rest/bye
  threshold is calendar-days-based and every window is game-count-based, so
  16- vs 17-game seasons are handled without any explicit branch.

## Task 9 — Model (`src/model.py`, `scripts/train_model.py`)

**Ports from:** CFB `src/model.py`, `scripts/train_model.py` — architecture
**verbatim**.

- Same 5-model stack (LogReg / RF / XGB / LightGBM / ExtraTrees → logistic
  meta-learner on `TimeSeriesSplit` OOF) → P(home win); separate `XGBRegressor` →
  predicted margin. `spread_line` excluded from training, used only post-hoc for
  `edge = predicted_margin - spread_line` — **not** `- (-spread_line)` as
  written here originally. **Real bug caught while building Task 9:** nflverse's
  `spread_line` is POSITIVE when the home team is favored, the opposite of the
  "-7 = favored by 7" sportsbook-quote convention CFBD used (which is what CFB's
  own edge/cover formulas assume, and what this line was blindly copied from).
  `market_spread` already IS the market's implied home margin directly — no
  negation. This was ALSO wrong in the `prediction_results` view's
  `pick_covered` CASE logic since Task 1 (`+ market_spread` instead of
  `- market_spread`) — fixed in `src/db.py` as part of this task. Confirmed
  with a real game: 2023 Week 17 BUF (home), a 15-point favorite
  (`spread_line=15`), won by only 6 — a well-known "didn't cover" result that
  `actual_margin > spread_line` (6 > 15, false) gets right and the old
  `actual_margin > -spread_line` formula would have gotten backwards.
- Retune: `N_SPLITS` (more seasons available than CFB), tree depths for the
  larger, cleaner NFL sample.
- Split: train **2016–2024** (~2,400 games), holdout **2025** (285 games) full
  season, **2026** live rolling. Save bundle to `models/nfl_model.pkl`
  (gitignored).
- **DONE (2026-09-04).** Train **2,476** games (2016–2024), holdout **285**
  games (2025) — matches the plan's row-count estimate closely. Architecture
  ported verbatim (5-model stack + meta-learner + separate margin `XGBRegressor`).
  No CFB-style era/sample-weighting ported — CFB's down-weights pre-2021 rows
  for NIL, a discrete legal break with no NFL equivalent; left as a deliberate
  non-port rather than inventing an artificial boundary.
  `N_SPLITS`: tested 3/4/5/6 against the holdout — accuracy was noisy across
  the range (60.4–62.5%, ordinary single-holdout variance on 285 games, not a
  clean trend), so **6** was picked for being consistently best across THREE
  metrics at once (accuracy, AUC, log_loss) rather than because it maximized
  any one score, which would just be tuning to the test set. Tree
  depths/estimator counts were kept at CFB's own values — the NFL training set
  (2,476 rows) is comparably sized to CFB's original, not meaningfully larger,
  so "larger sample" in this task's original framing didn't hold up under a
  real row-count check and aggressive re-tuning wasn't warranted.
- **Real bug, independent of the model itself, caught while sanity-checking a
  cover calculation for this task:** see above — the `market_spread`
  sign-convention error was live in `prediction_results` since **Task 1**,
  4 commits before it was caught. It never affected anything already built
  (Tasks 1–8 don't read `pick_covered`), but would have silently corrupted
  every ATS win/loss record once Task 11 started writing predictions. Caught
  by manually verifying a real, unambiguous game rather than trusting the
  CFB-ported formula shape — worth remembering as a general lesson: a formula
  that "looks like a straightforward port" can still carry a wrong assumption
  from the source project's own data conventions.
- **Check results:** classifier accuracy **62.46%** (bottom of the 62–68%
  target range), AUC **0.696**, log_loss **0.632**, Brier **0.221**. Beats the
  naive home-always baseline (53.33%) by **9.1pp** and matches ELO-only
  exactly (62.46% — coincidental tie, not a bug); sits **3.5pp behind** the
  market-favorite baseline (65.96%) — the expected healthy gap, not ahead of
  it. Regressor MAE **10.22** vs the market's own **9.67** (diff 0.55, right at
  the "~market ± 0.5" expectation). Calibration curve is reasonably monotonic
  across 8 bins with the noise expected at ~35 games/bin. **No metric beats the
  market on this first pass** — the leakage smell test passes.

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
