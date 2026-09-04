"""Computes each team's CURRENT state (as of right now) for live inference --
distinct from build_features.py, which computes state "entering game X" for a
specific historical training row. Same underlying signals (ELO, SRS, EPA
rolling form), but here there's no future game_id to anchor a snapshot to: we
want each team's freshest rating/average given everything completed so far.

Ported from the CFB build's live_state.py, but narrower in scope than CFB's:
most of this project's Task 8 feature functions were built point-in-time
correct from the start and need NO live counterpart at all --
src.availability's injury_burden()/qb_situation() and
src.ats_and_situational's compute_h2h_features()/compute_situational_features()
already take an explicit (season, week, team) or work off a single game row
in-place, so calling them with the upcoming game's own (season, week) IS the
live computation. Only the batch-chronological modules (ELO, SRS, EPA rolling
form, ATS rolling form) need a "run through everything, expose the final
state" counterpart, which is what lives here.

Watches CFB's two documented live-inference bugs, both non-issues here by
construction: (1) H2H returning nothing for a scoreless target game --
ats_and_situational.compute_h2h_features() only checks home_points to decide
whether to APPEND to history, never to decide whether to RETURN a result, so
an upcoming game already gets a correct answer without any live_state
involvement. (2) rest-days not scoping to the current season -- moot here
since rest days aren't recomputed at all, see Task 8's note: nflverse's own
games.home_rest/away_rest are schedule-based and already correct for future
rows too (verified: every upcoming 2026 game already carries a real value).
"""
from collections import defaultdict

import pandas as pd

from src.elo import NFLElo
from src.epa_features import ROLL_COLS, ROLLING_WINDOW as EPA_WINDOW
from src.opponent_adjustment import PRIOR_SEASON_REGRESSION, _solve_srs

ATS_WINDOW = 8  # matches src.ats_and_situational.ATS_ROLLING_WINDOW


def compute_current_elo(completed_games: list[dict], franchise_of) -> dict:
    """completed_games: dicts with season, gameday, home_team, away_team,
    home_points, away_points, neutral_site. Returns {franchise: current_elo}."""
    elo = NFLElo()
    for g in sorted(completed_games, key=lambda g: (g["season"], g["gameday"])):
        elo.maybe_regress_for_new_season(g["season"])
        elo.update(franchise_of(g["home_team"]), franchise_of(g["away_team"]),
                   g["home_points"], g["away_points"], g["season"], bool(g["neutral_site"]))
    return dict(elo.ratings)


def compute_current_srs(completed_games: list[dict], current_season: int, franchise_of) -> dict:
    """Current-season SRS where available, else prior season's final SRS
    regressed toward the mean (same rationale as opponent_adjustment
    .compute_weekly_srs's own season-boundary handling). Regular season only."""
    def pairwise(games):
        rows = []
        for g in games:
            margin = g["home_points"] - g["away_points"]
            h, a = franchise_of(g["home_team"]), franchise_of(g["away_team"])
            rows.append((h, a, margin))
            rows.append((a, h, -margin))
        return rows

    current_season_games = [g for g in completed_games
                             if g["season"] == current_season and g["season_type"] == "REG"]
    prior_season_games = [g for g in completed_games
                           if g["season"] == current_season - 1 and g["season_type"] == "REG"]

    current_srs = _solve_srs(pairwise(current_season_games)) if current_season_games else {}
    prior_srs = _solve_srs(pairwise(prior_season_games)) if prior_season_games else {}

    result = dict(current_srs)
    for team, rating in prior_srs.items():
        if team not in result:
            result[team] = rating * (1 - PRIOR_SEASON_REGRESSION)
    return result


def compute_current_epa_form(epa_df: pd.DataFrame, franchise_of, window: int = EPA_WINDOW) -> dict:
    """epa_df: team_game_epa rows. Returns {franchise: {stat: trailing_avg}} over
    each franchise's last `window` completed games (relocation-continuous)."""
    df = epa_df.copy()
    df["franchise"] = df["team"].map(franchise_of)
    result = {}
    for franchise, group in df.groupby("franchise"):
        recent = group.sort_values(["season", "week"]).tail(window)
        if recent.empty:
            continue
        result[franchise] = {col: pd.to_numeric(recent[col], errors="coerce").mean() for col in ROLL_COLS}
    return result


def compute_current_opponent_srs(epa_df: pd.DataFrame, srs: dict, franchise_of, window: int = EPA_WINDOW) -> dict:
    """Average CURRENT SRS (not historical-at-the-time) of each franchise's last
    `window` opponents."""
    df = epa_df.copy()
    df["franchise"] = df["team"].map(franchise_of)
    df["opp_franchise"] = df["opponent"].map(franchise_of)
    result = {}
    for franchise, group in df.groupby("franchise"):
        recent = group.sort_values(["season", "week"]).tail(window)
        opp_ratings = [srs.get(opp, 0.0) for opp in recent["opp_franchise"]]
        if opp_ratings:
            result[franchise] = sum(opp_ratings) / len(opp_ratings)
    return result


def compute_current_ats_pct(ats_rows: list[dict], window: int = ATS_WINDOW) -> dict:
    """ats_rows: ats_and_situational.compute_ats_results() output (team,
    season, gameday, covered). Returns {team: (pct, n)} -- n is the actual
    decided-game count behind pct (<=window), so an explanation can say
    "last N games" precisely rather than assuming the window size always held."""
    by_team = defaultdict(list)
    for row in ats_rows:
        by_team[row["team"]].append(row)
    result = {}
    for team, rows in by_team.items():
        rows.sort(key=lambda r: (r["season"], r["gameday"]))
        decided = [r["covered"] for r in rows if r["covered"] is not None][-window:]
        if decided:
            result[team] = (sum(decided) / len(decided), len(decided))
    return result
