"""SQLite schema and connection helpers for the NFL betting model.

Two databases, deliberately split (see MAP.md §5 and
[[feedback_sports_betting_model_architecture]]):

- ``nfl_stats.db``       -- historical / derived data. **Only GitHub Actions
                            writes it.** teams, games, EPA aggregates, injuries,
                            depth charts, snap counts, live-odds snapshots,
                            polymarket odds, the replace-built game_features table.
- ``nfl_predictions.db`` -- the model's own picks. **Only the scheduled Claude
                            Code routine writes it.** predictions + the always-live
                            prediction_results view.

Splitting the writers means the two files never land in the same commit, so there
is no merge friction between Actions and the routine, and the big historical file
is only ever pushed by Actions -- which sidesteps the Git-LFS 403 the CFB build
hit when the routine's GitHub App token tried to push an LFS object.

The prediction_results view lives in nfl_predictions.db but joins games from
nfl_stats.db, so get_pred_connection() ATTACHes the stats DB as ``stats`` on every
connection and the view references ``stats.games``.
"""
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATS_DB_PATH = DATA_DIR / "nfl_stats.db"
PRED_DB_PATH = DATA_DIR / "nfl_predictions.db"


# --------------------------------------------------------------------------- #
# nfl_stats.db  (GitHub Actions is the only writer)
# --------------------------------------------------------------------------- #
SCHEMA_STATS = """
CREATE TABLE IF NOT EXISTS teams (
    team TEXT PRIMARY KEY,               -- nflverse abbreviation, e.g. 'KC'
    full_name TEXT NOT NULL,
    conference TEXT,                     -- 'AFC' / 'NFC'
    division TEXT,                       -- 'AFC West' ...
    franchise_id TEXT,                   -- folds relocations: OAK+LV, SD+LAC, STL+LAR
    espn_id TEXT
);

-- Near 1:1 with nflverse load_schedules() / Lee Sharpe's games.csv (46 cols),
-- plus a handful of derived columns. Absorbs what were separate `lines` and
-- `game_weather` tables in the CFB build -- nflverse carries closing lines and
-- post-game weather on the schedule row itself. Holds completed AND future rows
-- (future rows have NULL scores -- that is the upcoming slate for prediction).
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,            -- e.g. '2024_01_BAL_KC'
    season INTEGER,
    game_type TEXT,                      -- REG / WC / DIV / CON / SB  (PRE only if loaded separately)
    season_type TEXT,                    -- derived: PRE / REG / POST
    week INTEGER,                        -- 1-18 regular (1-17 pre-2021), 19 WC .. 22 SB
    gameday TEXT,
    weekday TEXT,
    gametime TEXT,
    away_team TEXT,
    home_team TEXT,
    away_score INTEGER,
    home_score INTEGER,
    location TEXT,                       -- 'Home' / 'Neutral'
    result INTEGER,                      -- home_score - away_score (nflverse-provided)
    total INTEGER,                       -- home_score + away_score (nflverse-provided)
    overtime INTEGER,
    neutral_site INTEGER,                -- derived: location == 'Neutral'
    home_margin INTEGER,                 -- derived
    home_win INTEGER,                    -- derived
    div_game INTEGER,
    away_rest INTEGER,
    home_rest INTEGER,

    -- betting (nflverse closing-ish line; ~complete from 2010, full within 2016+)
    spread_line REAL,                    -- positive => home favored by that many
    away_spread_odds INTEGER,
    home_spread_odds INTEGER,
    total_line REAL,
    over_odds INTEGER,
    under_odds INTEGER,
    away_moneyline INTEGER,
    home_moneyline INTEGER,

    -- venue / environment (follows the ACTUAL stadium at game time, not the franchise)
    roof TEXT,                           -- outdoors / open / closed / dome
    surface TEXT,
    is_dome INTEGER,                     -- derived: roof in ('closed','dome')
    stadium_id TEXT,
    stadium TEXT,
    temp INTEGER,                        -- nflverse fills post-game for roof in (outdoors,open);
    wind INTEGER,                        --   Task 5 forecast-fills these for upcoming games
    weather_source TEXT,                 -- NULL = nflverse post-game; 'open-meteo-forecast' = pre-game fill

    -- personnel
    away_qb_id TEXT,
    home_qb_id TEXT,
    away_qb_name TEXT,
    home_qb_name TEXT,
    away_coach TEXT,
    home_coach TEXT,
    referee TEXT,

    -- nflverse cross-reference ids
    old_game_id TEXT,
    gsis INTEGER,
    nfl_detail_id TEXT,
    pfr TEXT,
    pff TEXT,
    espn TEXT,
    ftn TEXT
);

-- Derived per-team-per-game aggregates from play-by-play. Raw pbp (~1M plays x
-- ~380 cols) is NEVER stored -- the Actions runner pulls it fresh, aggregates to
-- this table (~15k rows total), and discards the raw frame. Populated for
-- REG + POST games only (preseason pbp is unrepresentative). Thresholds for
-- 'explosive' / 'success' / 'neutral' are tuned against the real distribution in
-- Task 3, not assumed.
CREATE TABLE IF NOT EXISTS team_game_epa (
    game_id TEXT NOT NULL,
    team TEXT NOT NULL,
    opponent TEXT,
    season INTEGER,
    week INTEGER,
    is_home INTEGER,
    off_epa_play REAL,
    def_epa_play REAL,
    off_pass_epa REAL,
    off_rush_epa REAL,
    off_early_down_epa REAL,
    off_success_rate REAL,
    def_success_rate REAL,
    explosive_play_rate REAL,
    def_explosive_rate REAL,
    rz_td_pct REAL,
    pressure_rate_def REAL,
    plays INTEGER,
    sec_per_play REAL,
    pass_rate REAL,
    neutral_pass_rate REAL,
    PRIMARY KEY (game_id, team)
);

-- Official NFL injury reports (nflverse load_injuries, 2009+). Mandatory and
-- reliable, unlike the CFB build's best-effort ESPN scrape. The feed only covers
-- REG + POST (no preseason report exists) -- a player hurt in preseason first
-- shows up on the Week 1 report, so no separate preseason handling is needed.
-- nflverse already serves the *latest* official status per player-week, so this
-- is an upsert (INSERT OR REPLACE), not the append-only snapshot table the CFB
-- build needed; `scraped_at` just records our last pull for staleness checks.
CREATE TABLE IF NOT EXISTS injuries (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team TEXT NOT NULL,
    gsis_id TEXT NOT NULL,
    game_type TEXT,
    player_name TEXT,
    position TEXT,
    report_status TEXT,                  -- Out / Doubtful / Questionable / (NULL = practice-report only)
    report_primary_injury TEXT,
    practice_status TEXT,
    date_modified TEXT,
    scraped_at TEXT NOT NULL,
    PRIMARY KEY (season, week, team, gsis_id)
);

-- No depth_charts table: nflverse depth charts are 40+ MB of mostly-redundant
-- rows, lag badly in-season (no 2025 data), and add little over snap share.
-- src/availability.py derives starters from snap_counts; the week-1 fallback is
-- the prior season's snap leaders.

-- gsis_id <-> pfr_id crosswalk (+ canonical name/position), from nflverse
-- load_players. Needed because snap_counts keys on pfr_player_id while injuries
-- keys on gsis_id.
CREATE TABLE IF NOT EXISTS players (
    gsis_id TEXT PRIMARY KEY,
    pfr_id TEXT,
    display_name TEXT,
    position TEXT,
    latest_team TEXT
);

-- Snap counts (nflverse load_snap_counts, 2012+) -- the working definition of
-- "starter" (trailing snap share at a position) and the weight for the
-- position-weighted injury-burden feature. Backfill keeps only rows with a real
-- role (offense_pct or defense_pct >= 0.10) -- the long tail of 1-snap cameos is
-- ~half the rows and never affects starter identification. Special-teams-only
-- players are dropped for the same reason.
CREATE TABLE IF NOT EXISTS snap_counts (
    game_id TEXT NOT NULL,
    pfr_player_id TEXT NOT NULL,
    season INTEGER,
    week INTEGER,
    team TEXT,
    player TEXT,
    position TEXT,
    offense_snaps INTEGER,
    offense_pct REAL,
    defense_snaps INTEGER,
    defense_pct REAL,
    PRIMARY KEY (game_id, pfr_player_id)
);

-- Live/upcoming line-movement snapshots from The Odds API (americanfootball_nfl).
-- Append-only (PK includes scraped_at) so intra-week movement is queryable.
-- Ported verbatim from the CFB build. Pruned by scripts/prune_live_data.py.
CREATE TABLE IF NOT EXISTS live_odds (
    odds_game_id TEXT NOT NULL,
    scraped_at TEXT NOT NULL,
    commence_time TEXT,
    home_team TEXT,
    away_team TEXT,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,
    outcome_name TEXT NOT NULL,
    price REAL,
    point REAL,
    PRIMARY KEY (odds_game_id, scraped_at, bookmaker, market, outcome_name)
);

-- Pre-game Polymarket win probability, one row per game (upsert-by-game_id --
-- always "latest known value," a pull closer to kickoff is more accurate). Used
-- ONLY for a post-game Brier-score accuracy check against the model's own
-- pre-game win_prob_home (see src/polymarket_client.py, Task 13). Never a live
-- pick or edge input.
CREATE TABLE IF NOT EXISTS polymarket_odds (
    game_id TEXT PRIMARY KEY,
    scraped_at TEXT NOT NULL,
    polymarket_event_id TEXT,
    home_prob REAL,
    away_prob REAL
);

-- No stadiums table: lat/long/timezone/roof_default live in src/stadiums.py as a
-- static Python dict keyed by venue NAME (not nflverse's stadium_id) -- see that
-- module's docstring for why (an international "home" game keeps its usual
-- franchise stadium_id but the venue name changes, e.g. 2026 Wk5 JAX@Tottenham).

-- Indexes for the availability lookups (src/availability.py hits these per
-- team-week when building features) and the EPA rolling joins.
CREATE INDEX IF NOT EXISTS ix_snaps_player   ON snap_counts (pfr_player_id, season, week);
CREATE INDEX IF NOT EXISTS ix_snaps_team     ON snap_counts (team, season, week);
CREATE INDEX IF NOT EXISTS ix_epa_team       ON team_game_epa (team, season, week);
CREATE INDEX IF NOT EXISTS ix_games_season   ON games (season, week, season_type);
"""
# Note: `game_features` is intentionally not declared here -- scripts/build_features.py
# creates it with pandas .to_sql(if_exists="replace"), same as the CFB build.


# --------------------------------------------------------------------------- #
# nfl_predictions.db  (the scheduled Claude routine is the only writer)
# --------------------------------------------------------------------------- #
SCHEMA_PRED = """
CREATE TABLE IF NOT EXISTS predictions (
    game_id TEXT PRIMARY KEY,
    predicted_at TEXT NOT NULL,
    season INTEGER,
    week INTEGER,
    season_type TEXT,
    gameday TEXT,
    gametime TEXT,
    home_team TEXT,
    away_team TEXT,

    -- model outputs
    predicted_margin REAL,              -- + => home favored, compared vs market for edge
    win_prob_home REAL,

    -- spread (ATS) pick
    market_spread REAL,                 -- spread_line snapshot at prediction time
    pick_team TEXT,                     -- the ATS pick
    edge REAL,                          -- predicted_margin - market_spread (nflverse: + = home favored)
    confidence_tier TEXT,
    cover_probability REAL,
    kelly_fraction REAL,
    spread_price INTEGER,               -- real median juice for the pick side
    spread_price_source TEXT,
    spread_price_book_count INTEGER,

    -- moneyline pick, tracked as a genuinely separate field (not implied by the
    -- ATS pick -- an ATS pick is routinely the underdog taking points)
    moneyline_pick TEXT,
    moneyline_win_prob REAL,
    moneyline_confidence_tier TEXT,

    -- explanation (prose written natively by the routine from these SHAP facts)
    highlights_json TEXT,
    tldr TEXT,
    bullets_json TEXT,
    model_breakdown_json TEXT,

    -- season-boundary confidence flag: min completed current-season games across
    -- the two teams (window-size threshold retuned for NFL's 17-game season in Task 8)
    min_current_season_games INTEGER
);
"""

# SQLite forbids a *persistent* view from referencing an attached database, so
# prediction_results is created as a TEMP VIEW on every pred connection instead
# (see get_pred_connection). A temp view is per-connection and rebuilt each
# session, which also sidesteps the CFB build's stale-view bug for free -- the
# definition can never drift from `predictions`' current columns.
VIEW_PRED = """
CREATE TEMP VIEW prediction_results AS
SELECT
    p.*,
    g.home_score, g.away_score,
    (g.home_score - g.away_score) AS actual_margin,
    CASE WHEN g.home_score > g.away_score THEN p.home_team ELSE p.away_team END AS actual_winner,
    -- Did the ATS pick also win outright? Kept for diagnostics only -- an ATS
    -- pick on a dog taking points is expected to lose this often by design. The
    -- site's straight-up record uses moneyline_pick_won.
    CASE WHEN p.pick_team = (CASE WHEN g.home_score > g.away_score THEN p.home_team ELSE p.away_team END)
         THEN 1 ELSE 0 END AS ats_pick_won_straight_up,
    CASE WHEN p.moneyline_pick IS NULL THEN NULL
         WHEN p.moneyline_pick = (CASE WHEN g.home_score > g.away_score THEN p.home_team ELSE p.away_team END)
         THEN 1 ELSE 0 END AS moneyline_pick_won,
    -- nflverse's spread_line is POSITIVE when the HOME team is favored (opposite
    -- of the common "-7 = favored by 7" sportsbook quoting convention CFB's
    -- CFBD-sourced spread used) -- so "cover" is actual_margin compared against
    -- market_spread DIRECTLY, not against its negation. Confirmed against a real
    -- game: 2023 Wk17 BUF (home) favored by 15 (spread_line=15), won by only 6 --
    -- a well-known "didn't cover" case (6 > 15 is false), which the formula below
    -- gets right and a "+ market_spread" formula would have gotten backwards.
    CASE
        WHEN p.pick_team IS NULL OR p.market_spread IS NULL THEN NULL
        WHEN (g.home_score - g.away_score) - p.market_spread = 0 THEN NULL  -- push
        WHEN p.pick_team = p.home_team THEN
            CASE WHEN (g.home_score - g.away_score) - p.market_spread > 0 THEN 1 ELSE 0 END
        WHEN p.pick_team = p.away_team THEN
            CASE WHEN (g.home_score - g.away_score) - p.market_spread < 0 THEN 1 ELSE 0 END
    END AS pick_covered,
    ABS(p.predicted_margin - (g.home_score - g.away_score)) AS margin_error
FROM predictions p
JOIN stats.games g ON p.game_id = g.game_id
WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL;
"""


def _tune_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # concurrent backfill scripts write without lock errors
    conn.execute("PRAGMA busy_timeout = 30000")


def get_stats_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATS_DB_PATH, timeout=30)
    _tune_pragmas(conn)
    return conn


def get_pred_connection(attach_stats: bool = True) -> sqlite3.Connection:
    """Connection to nfl_predictions.db. By default ATTACHes nfl_stats.db as
    ``stats`` and (re)creates the per-connection TEMP VIEW prediction_results,
    which joins stats.games. Pass attach_stats=False only for a raw write that
    never touches the view (e.g. an early init before the stats schema exists)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PRED_DB_PATH, timeout=30)
    _tune_pragmas(conn)
    if attach_stats:
        conn.execute("ATTACH DATABASE ? AS stats", (str(STATS_DB_PATH),))
        # Harmless if `predictions` doesn't exist yet -- the view is lazy; it only
        # errors when queried. init_pred_db() creates the table before anyone reads.
        try:
            conn.executescript(VIEW_PRED)
        except sqlite3.OperationalError:
            pass  # predictions table not created yet; caller is about to init it
    return conn


# Pre-launch schema churn: drop a table so executescript recreates it fresh when
# its live shape is stale. Safe only because none of these carry committed data
# yet. Each rule is (predicate over the live column set) -> drop.
def _drop_stale_tables(conn: sqlite3.Connection) -> None:
    def cols(table):
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}

    rules = {
        "depth_charts": lambda c: bool(c),                       # table removed from schema
        "stadiums": lambda c: bool(c),                           # moved to src/stadiums.py
        "injuries": lambda c: c and "report_primary_injury" not in c,
        "snap_counts": lambda c: c and "st_snaps" in c,          # rebuilt without st_* columns
    }
    for table, is_stale in rules.items():
        if is_stale(cols(table)):
            conn.execute(f"DROP TABLE {table}")


def init_stats_db() -> None:
    conn = get_stats_connection()
    try:
        _drop_stale_tables(conn)
        conn.executescript(SCHEMA_STATS)
        conn.commit()
    finally:
        conn.close()


def init_pred_db() -> None:
    # The view references stats.games, so the stats DB must exist with at least
    # its schema. In production the routine always runs after Actions has
    # committed a populated nfl_stats.db; this call just covers a fresh local
    # checkout where only the predictions DB is being initialised.
    init_stats_db()
    conn = get_pred_connection(attach_stats=True)
    try:
        conn.executescript(SCHEMA_PRED)   # persistent: predictions table
        conn.commit()
        # prediction_results is a TEMP VIEW; get_pred_connection() (re)creates it
        # on every connection. Recreate it here now that `predictions` exists so
        # this same connection can query it immediately.
        conn.execute("DROP VIEW IF EXISTS prediction_results")
        conn.executescript(VIEW_PRED)
        conn.commit()
    finally:
        conn.close()


def init_all() -> None:
    init_stats_db()
    init_pred_db()


if __name__ == "__main__":
    init_all()
    print(f"Initialized  {STATS_DB_PATH}")
    print(f"Initialized  {PRED_DB_PATH}  (ATTACHes stats for prediction_results)")
