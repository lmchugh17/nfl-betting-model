# NFL Betting Model — Build Map

Port of the [CFB Betting Model](../CFB%20Betting%20Model/) to the NFL. Same core
idea: opponent-adjusted power ratings + rolling form → a 5-model stacked ensemble
for win probability plus a separate XGBoost margin regressor, compared against the
market to compute an edge, Kelly-sized on a paper bankroll, explained from the
model's own SHAP factors, published to a static GitHub Pages site, refreshed by a
weekly GitHub Actions data pull + a scheduled Claude Code routine.

Status: **mapping**. Nothing built yet.

**Decisions locked (2026-08-30):**
1. Train window: **2016+** (modern era; relocations start here).
2. Features: **EPA + QB availability + position-weighted injury burden**, not a
   straight box-score port.
3. Storage: **two-DB split** — `nfl_stats.db` (Actions only) + `nfl_predictions.db`
   (Claude routine only).
4. Relocations: **continuous franchise** for ELO / SRS / rolling form (roster &
   coaching are the signal) — **but** stadium environment (dome/roof/surface/
   climate) always follows the *actual venue at game time*, so the weather
   features naturally capture the location change. See §4.

---

## 1. What ports directly (little or no change)

| CFB file | NFL disposition |
|---|---|
| `src/model.py` | Port as-is. Retune `N_SPLITS`, rolling-window constants. Same 5-model stack (LogReg / RF / XGB / LightGBM / ExtraTrees → logistic meta-learner on TimeSeriesSplit OOF) + separate `XGBRegressor` for margin. `market_spread` stays excluded from training features. |
| `src/explain.py` | Port. SHAP (TreeSHAP on the margin regressor) → grounded factual sentences. Only `_describe_feature` needs new feature-name cases. |
| `src/elo.py` | Port the formula. **Recalibrate constants** — see §4. Add franchise-relocation handling. |
| `src/opponent_adjustment.py` (iterative SRS) | Port as-is. Less load-bearing in the NFL (17-game, formula-balanced schedules vs 130 FBS teams with wild schedule variance) but cheap to keep and still mildly useful. |
| `src/weather_features.py` | Port as-is — already sport-agnostic per the architecture notes. **Re-tune adverse thresholds against NFL history**, don't reuse the 40°F/20mph/0.1in CFB numbers blindly. |
| `src/live_state.py` | Port. Same "current state as of now, not entering-game-X" dual-function pattern. |
| `scripts/predict_games.py` | Port. Same flow: current-state features → bundle → edge → cover prob → Kelly → SHAP highlights → `predictions` upsert. **Upgrade:** use real spread juice (we get it free for the NFL) instead of the assumed −110. |
| `scripts/reconcile_predictions.py` | Port as-is. |
| `scripts/build_site.py` | Port. Same dark single-page site: stat tiles, upcoming picks with TL;DR + bullets, recent results shown honestly, per-model breakdown, Kelly wager vs a paper bankroll, confidence-tier badges. Rebrand only. |
| `src/db.py` | Port the pattern. Schema changes in §3. |
| `prediction_results` SQL view | Port almost verbatim — `week` semantics get simpler. |
| `.github/workflows/weekly_data_pull.yml` | Port the shape. New cadence + fewer steps (§5). |
| Scheduled Claude Code routine (task 11 equivalent) | Port. Writes explanations natively from SHAP facts, rebuilds site, commits. Verify GitHub App push access with a manual run **before** trusting cron (CFB lesson — a failed push is silent and lossy). |

## 2. What gets replaced — data sources

The NFL story is **much simpler** than CFB. CFB needed CFBD + The Odds API
(historical lines) + Open-Meteo (historical weather) stitched together. For the
NFL, one source covers most of it.

### Primary: nflverse (`nflreadpy` / `nfl_data_py`) — free, no API key

| Dataset | Gives us | Replaces |
|---|---|---|
| `load_schedules()` (Lee Sharpe's `games.csv`) | Every game 1999→now: date, weekday, kickoff time, teams, scores, `game_type`, `stadium_id`, `roof`, `surface`, `temp`, `wind`, `home_rest`/`away_rest`, `div_game`, `home_qb_id`/`away_qb_id`, `home_coach`/`away_coach`, referee, **`spread_line`, `total_line`, moneylines, and the spread/total juice** | CFBD `/games` + CFBD `/lines` + Open-Meteo historical backfill + most of `coach_seasons` — all one table |
| `load_pbp()` | Play-by-play 1999→now with EPA / WPA / success / air yards / pressure | (new capability — CFB had only box scores) |
| `load_injuries()` | Official injury reports 2009→now: `report_status` (Out/Doubtful/Questionable), `practice_status`, position | CFB's noisy best-effort ESPN scrape — NFL reports are **mandatory and reliable** |
| `load_depth_charts()` | Depth charts 2001→now — identify starters | (new) |
| `load_snap_counts()` | Snap counts 2012→now — weight "is this player actually a starter" | (new) |
| `load_rosters_weekly()` | Week-by-week rosters | CFBD `/roster` |

### Still needed as separate sources

| Source | Why | Notes |
|---|---|---|
| **The Odds API** `americanfootball_nfl` | Live/upcoming line **snapshots + intra-week line movement**. nflverse only has a settled closing-ish line, post-hoc. | Reuse the existing account. NFL slate is ~14–16 games/week vs CFB's ~60, so pulls are cheap. Keep `live_odds` append-only + pruned, same as CFB. |
| **Open-Meteo forecast** | Forecast temp/wind for **upcoming** outdoor games (nflverse won't have `temp`/`wind` until after kickoff). | Historical weather backfill is **not** needed — nflverse schedule already carries it. Domes → null metrics, `is_dome=1`, same as CFB. |
| **ESPN** (optional) | Backup live scores / injury cross-check. | Only if nflverse weekly refresh lag is a problem in-season. |

### raw pbp is big — do not store it

1999–2025 pbp ≈ 1.1M plays × ~380 columns. **Never commit raw pbp to the tracked
DB.** The Actions runner pulls pbp parquet fresh, aggregates to per-team-per-game
rows, and stores only the aggregates (`team_game_epa`, ~15k rows total). Keeps the
DB tiny and git history clean.

## 3. Schema deltas from `cfb.db`

- **`teams`** — 32 rows, `team` abbreviation as PK, add `conference` (AFC/NFC) +
  `division` + `franchise_id` (folds `OAK`/`LV`, `SD`/`LAC`, `STL`/`LAR` to one
  ID). Ratings join on `franchise_id`; everything venue-bound stays on the game
  row. See §4.
- **`games`** — near 1:1 with nflverse `schedules`. Absorbs CFB's separate
  `lines` and `game_weather` tables. New cols: `game_type`, `gameday`, `weekday`,
  `gametime`, `roof`, `surface`, `stadium_id`, `div_game`, `home_rest`,
  `away_rest`, `home_qb_id`/`home_qb_name`, `away_qb_id`/`away_qb_name`,
  `home_coach`, `away_coach`, `temp`, `wind`, `spread_line`, `spread_odds` (both
  sides), `total_line`, `total_odds` (both sides), moneylines.
- **Drop** — `team_seasons.classification`, standalone `venues` (fold to a small
  `stadiums` ref or just keep fields on `games`), CFB's 400k-row
  `player_season_stats`.
- **New tables** —
  - `team_game_epa` — derived pbp aggregates per team per game (off/def EPA per
    play, pass vs rush EPA, early-down EPA, success rate, explosive-play rate,
    red-zone TD%, pressure rate, pace).
  - `injuries` — official report: `(season, week, team, gsis_id)` PK,
    `player_name`, `position`, `report_status`, `practice_status`,
    `date_modified`.
  - `depth_charts` — for starter identification.
  - `snap_counts` — optional, starter weighting.
- **`predictions` / `prediction_results`** — unchanged shape; `week` is 1–18 +
  playoff rounds.
- **`season_type` / `game_type`** — three values: `PRE`, `REG`, `POST`.
  **Preseason is tracked in `games` and fed to the injury pipeline, but excluded
  from every ELO / SRS / EPA / training computation** (architecture note — NFL
  preseason play is unrepresentative, but preseason injuries carry into the
  regular season).

## 4. NFL-specific calibration & features

### ELO recalibration (all need backtesting, these are starting points)

| Constant | CFB | NFL starting point | Why |
|---|---|---|---|
| `K_FACTOR` | 40 | ~20 | 17 games (still few, but rosters/coaching far more stable than CFB) |
| `SEASON_REGRESSION_FACTOR` | 0.6 | ~0.80 | Minimal year-over-year turnover vs CFB transfer portal — carry more forward |
| `HOME_ADVANTAGE_ELO` | 65 | ~40 (~1.7 pts) | NFL HFA is small and **declining**; 2020 (no fans) is an outlier — consider a covid-season flag or downweight |

### Franchise relocations — resolved: continuous franchise, location-aware venue

nflverse keeps historical abbreviations distinct (`OAK` ≠ `LV`, `SD` ≠ `LAC`).
Alias them to one continuous franchise ID so **ELO / SRS / rolling form / ATS /
H2H** carry across the move — the roster and coaching staff are what those
ratings are really tracking, and they don't reset when the city changes.

**But** everything venue-bound follows the actual stadium at game time, not the
franchise:
- `roof` / `surface` / `stadium_id` / `temp` / `wind` come straight off each
  game's `schedules` row, so a dome move is captured automatically.
- The adverse-weather ATS split (`weather_features.py`) keys on the game's real
  recorded conditions, so post-move home games in a dome correctly stop counting
  as adverse — the feature adapts on its own with no relocation-specific code.
- Within a 2016+ window this only spans `SD→LAC` (2016 → 2017+, San Diego
  outdoor-mild → LA, then SoFi dome from 2020) and `OAK→LV` (2016–19 → 2020+,
  Oakland outdoor → Allegiant dome). Small carryover, but the pattern is: **one
  `franchise_id` for ratings, real `stadium_id`/`roof` per game for environment.**

### Feature upgrades over a straight port

- **EPA-based form** (from pbp) instead of raw box-score yards — offensive &
  defensive EPA/play, pass vs rush split, early-down EPA, success rate, explosive
  rate. This is the single biggest model-quality upgrade available vs just porting
  CFB's yards-based rolling features.
- **QB availability** — *essential for the NFL.* At minimum a "projected starter
  is not the team's primary starter" flag from depth charts + injuries +
  `schedule.*_qb_id`. Better: a QB-value adjustment from rolling QB EPA/dropback
  (starter's recent form vs replacement level). Vegas moves lines 3–7 pts on QB
  news; the model must see it.
- **Injury burden** — count of listed-`Out` starters, position-weighted (snap
  share). Reliable here in a way it never was for CFB.
- **Situational flags** — short week (Thu, rest ≤ 4), off bye (rest ≥ 13),
  opponent off bye, prime-time (SNF/MNF), division game, cross-country travel /
  ≥ 2 time-zone shift (east-coast body clock), international game.
- **New HC / new OC** flag — year-over-year, from `schedule.*_coach`. Optional.

### ATS / situational / H2H

Port `ats_and_situational.py` directly. NFL bonus: divisional opponents play
**twice a season**, so a same-season H2H result (Week 3 → the Week 15 rematch) is
strong signal — H2H features matter more than in CFB.

### Era boundaries to handle

- 2021+ = 18 weeks / 17 games; 1999–2020 = 17 weeks / 16 games. Rolling-form and
  rest/bye logic must not assume a fixed season length.
- 2020 = no/limited fans → HFA anomaly.

## 5. Automation

### GitHub Actions data pull

Fewer steps than CFB (no CFBD, no player-stats mega-endpoint). Steps:
1. nflverse refresh — schedules, pbp → aggregate to `team_game_epa`, injuries,
   depth charts, snap counts.
2. The Odds API pull — `americanfootball_nfl`, spreads/totals/h2h, append to
   `live_odds`.
3. Open-Meteo forecast for upcoming outdoor games.
4. Prune stale `live_odds` / `injuries` snapshots → append-only archive CSVs.
5. Commit + push.

**Cadence** — NFL games are Thu / Sun / Mon. Injury practice reports land
Wed–Fri, final game-status designations Fri afternoon. Proposed: **Wed** (full
slate preview + opening lines), **Fri** (final injury designations), **Sat**,
**Sun AM**. ~4 pulls/week.

### Scheduled Claude Code routine

Runs ~1h after the Wed pull (writes the week's picks + explanations, rebuilds
site) and again after the Fri pull (refresh for final injury news / line
movement). Same as CFB: explanations written natively from SHAP facts, no
separate LLM API.

### DB architecture — decision needed

CFB ended up with one `cfb.db` + aggressive pruning + append-only CSV archives,
after reverting from Git LFS (the routine's GitHub App token 403s on the LFS batch
API). Architecture notes suggest **splitting from the start**:
- `nfl_stats.db` — large, historical; **only GitHub Actions writes it**.
- `nfl_predictions.db` — small; **only the Claude routine writes it**.

This sidesteps the LFS problem (only Actions ever pushes the big file) and avoids
merge friction between the two writers.

## 6. Hosting

- New repo: `github.com/lmchugh17/nfl-betting-model`, GitHub Pages from `/docs` on
  `master`.
- Reuse the Claude GitHub App connection already fixed for CFB. **Verify push
  access with a manual routine run before trusting the cron.**
- `gh` CLI still not set up on this Mac as of the CFB build — needed for repo
  creation.

## 7. Open decisions (before building)

Resolved — see the "Decisions locked" block at the top. Remaining to confirm:

- Reuse The Odds API account (yes, assumed).
- New repo `github.com/lmchugh17/nfl-betting-model`, GitHub Pages `/docs` (assumed).
- Train/holdout split within 2016+: proposed train 2016–2023, holdout 2024, then
  2025 as a live rolling test (mirrors CFB's full-season-holdout approach).
- Starting paper bankroll (CFB uses $500).

## 8. Task list (draft, mirrors CFB's 11)

1. Scaffold repo, `.env`, `requirements.txt` (`nflreadpy`/`nfl_data_py`, pandas,
   sklearn, xgboost, lightgbm, shap), `src/db.py` schema.
2. nflverse schedules → `teams` + `games` backfill (incl. lines + weather, which
   come free). Team-name normalization + relocation aliases.
3. pbp → `team_game_epa` aggregation.
4. Injuries + depth charts + snap counts backfill; starter identification.
5. Weather forecast pull (upcoming outdoor games).
6. The Odds API live-odds pull + `live_odds` table + pruning.
7. Feature engineering — ELO (recalibrated), SRS, EPA form, ATS/situational, H2H,
   QB availability, injury burden, weather → `game_features`.
8. Model — 5-model stack + margin regressor; train/holdout split; reality-check
   vs the market (expect to land *just behind* it).
9. Explanations — SHAP fact extraction + `_describe_feature` for NFL features.
10. Live inference + `predictions` tracking + `reconcile_predictions.py` + site.
11. GitHub Actions data pull + scheduled Claude Code routine; verify push access.
