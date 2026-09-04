"""Against-the-spread record, head-to-head history, and game-context
situational flags (rest/bye, prime-time, divisional, international, travel).

Rest/bye is NOT recomputed here like the CFB build did -- nflverse's own
`games.home_rest`/`away_rest` (calendar days since each team's last game) are
already correct and simpler to trust than re-deriving them from start_date
deltas. Real distribution (2016+): mode is 7 (a normal week); 4 is the
short-week Thursday-game cluster (160 occurrences); 13-16 is the bye-week
cluster (~200) -- 10/11 exist too (schedule quirks, e.g. a Thu->Sun flip) but
are clearly NOT byes, so the bye threshold is 12, not 10.

All rolling stats use the same no-leakage discipline as ELO/SRS/EPA form: a
game's features only ever use data from strictly earlier games (shift-by-one
before any rolling calculation).
"""
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from src.stadiums import INTERNATIONAL_VENUES, team_home_timezone

ATS_ROLLING_WINDOW = 8    # NFL's 17-game season affords a longer window than CFB's 5 (13-game season)
BYE_WEEK_REST_DAYS = 12   # see module docstring for why 12, not the naive-looking 10/11
SHORT_WEEK_REST_DAYS = 4  # exactly the Thursday-game cluster in the real distribution
PRIMETIME_GAMETIME = "19:00"  # >= this local kickoff -> SNF/MNF/TNF-style prime-time slot
TZ_SHIFT_SIGNIFICANT_HOURS = 2


# --------------------------------------------------------------------------- #
# ATS record
# --------------------------------------------------------------------------- #
def compute_ats_results(games: list[dict]) -> list[dict]:
    """games: dicts with game_id, season, week, season_type, gameday, home_team,
    away_team, home_points, away_points, spread_line (nflverse's single closing
    line -- unlike CFB there's no multi-provider `lines` table to median across).
    Returns one row per team per game with whether that team covered
    (True/False/None for push or no line available)."""
    rows = []
    for g in games:
        spread = g.get("spread_line")
        if spread is None or g["home_points"] is None:
            continue
        cover_margin = (g["home_points"] - g["away_points"]) + spread
        home_covered = None if cover_margin == 0 else cover_margin > 0
        away_covered = None if home_covered is None else (not home_covered)
        rows.append({"game_id": g["game_id"], "season": g["season"], "week": g["week"],
                      "season_type": g["season_type"], "gameday": g["gameday"],
                      "team": g["home_team"], "covered": home_covered})
        rows.append({"game_id": g["game_id"], "season": g["season"], "week": g["week"],
                      "season_type": g["season_type"], "gameday": g["gameday"],
                      "team": g["away_team"], "covered": away_covered})
    return rows


def compute_rolling_ats_pct(ats_rows: list[dict]) -> dict:
    """Returns {(game_id, team): (trailing_ats_pct, n_games) | (None, 0)} before
    this game, using only each team's own prior games (any push is excluded
    from the denominator, matching standard ATS record convention). n_games is
    tracked alongside the pct so an explanation (task 10) can say "covered 6 of
    its last 8" precisely instead of a bare, uncountable percentage."""
    by_team = defaultdict(list)
    for row in ats_rows:
        by_team[row["team"]].append(row)

    result = {}
    for team, rows in by_team.items():
        rows.sort(key=lambda r: (r["season"], r["gameday"]))
        history = []
        for row in rows:
            decided = history[-ATS_ROLLING_WINDOW:]
            result[(row["game_id"], team)] = (
                (sum(decided) / len(decided), len(decided)) if decided else (None, 0)
            )
            if row["covered"] is not None:
                history.append(row["covered"])
    return result


# --------------------------------------------------------------------------- #
# Head-to-head
# --------------------------------------------------------------------------- #
def compute_h2h_features(games: list[dict], n_last: int = 6, min_meetings: int = 2) -> dict:
    """Returns {game_id: {h2h_home_win_pct, h2h_avg_home_margin, h2h_meetings,
    h2h_current_season_margin, h2h_current_season_meeting}} using up to the last
    n_last meetings between the two teams, from ANY prior season, before this
    game's date.

    h2h_current_season_* is an NFL-specific addition beyond the CFB port:
    divisional opponents play TWICE a season, so a Week 3 meeting is strong,
    fresh signal for the Week 15 rematch in a way no cross-season history is --
    worth surfacing as its own feature rather than relying on the model to infer
    "the most recent of these n_last meetings happened to be this season" from
    the blended average.
    """
    games_sorted = sorted(games, key=lambda g: (g["season"], g["gameday"]))
    history = defaultdict(list)  # frozenset({teamA, teamB}) -> [(season, gameday, home_team, margin)]
    result = {}

    for g in games_sorted:
        pair_key = frozenset({g["home_team"], g["away_team"]})
        past = history[pair_key][-n_last:]

        this_season = [p for p in past if p[0] == g["season"]]
        if this_season:
            season, gameday, past_home, margin = this_season[-1]
            result_current = margin if past_home == g["home_team"] else -margin
        else:
            result_current = None

        if len(past) < min_meetings:
            result[g["game_id"]] = {
                "h2h_home_win_pct": None, "h2h_avg_home_margin": None, "h2h_meetings": len(past),
                "h2h_current_season_margin": result_current,
                "h2h_current_season_meeting": int(result_current is not None),
            }
        else:
            wins, margins = 0, []
            for _, _, past_home, margin in past:
                normalized = margin if past_home == g["home_team"] else -margin
                margins.append(normalized)
                if normalized > 0:
                    wins += 1
            result[g["game_id"]] = {
                "h2h_home_win_pct": wins / len(past),
                "h2h_avg_home_margin": sum(margins) / len(margins),
                "h2h_meetings": len(past),
                "h2h_current_season_margin": result_current,
                "h2h_current_season_meeting": int(result_current is not None),
            }

        if g["home_points"] is not None:
            history[pair_key].append((g["season"], g["gameday"], g["home_team"],
                                       g["home_points"] - g["away_points"]))

    return result


# --------------------------------------------------------------------------- #
# Situational flags (rest, prime-time, divisional, international, travel)
# --------------------------------------------------------------------------- #
def _utc_offset_hours(tz_name: str | None, date_str: str | None) -> float | None:
    if not tz_name or not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=ZoneInfo(tz_name))
        return dt.utcoffset().total_seconds() / 3600
    except (ValueError, TypeError):
        return None


def _wrap_signed_12(hours: float) -> float:
    """Wraps a raw UTC-offset difference to (-12, 12] -- the shorter way around
    the clock. A naive venue_offset - team_offset subtraction is only correct
    for offsets within 12h of each other; e.g. LA (UTC-7) to Melbourne
    (UTC+10) is a raw 17h, which reads as an absurd "17 time zones" -- the real
    body-clock shift is the 7h the other way (24 - 17)."""
    return ((hours + 12) % 24) - 12


def compute_situational_features(g: dict, venue_tz: str | None) -> dict:
    """g: a games row dict (needs home_rest, away_rest, gametime, weekday,
    div_game, stadium, gameday, home_team, away_team). venue_tz: the game's
    actual venue timezone (from src.stadiums.lookup(g['stadium'])['timezone']).
    Returns the situational feature dict for this one game."""
    home_rest, away_rest = g.get("home_rest"), g.get("away_rest")
    gametime = g.get("gametime") or ""

    home_tz = team_home_timezone(g["home_team"])
    away_tz = team_home_timezone(g["away_team"])
    gameday = g.get("gameday")
    venue_offset = _utc_offset_hours(venue_tz, gameday)
    home_team_offset = _utc_offset_hours(home_tz, gameday)
    away_team_offset = _utc_offset_hours(away_tz, gameday)

    home_tz_shift = (_wrap_signed_12(venue_offset - home_team_offset)
                     if (venue_offset is not None and home_team_offset is not None) else None)
    away_tz_shift = (_wrap_signed_12(venue_offset - away_team_offset)
                      if (venue_offset is not None and away_team_offset is not None) else None)

    return {
        "home_short_week": int(home_rest is not None and home_rest <= SHORT_WEEK_REST_DAYS),
        "away_short_week": int(away_rest is not None and away_rest <= SHORT_WEEK_REST_DAYS),
        "home_off_bye": int(home_rest is not None and home_rest >= BYE_WEEK_REST_DAYS),
        "away_off_bye": int(away_rest is not None and away_rest >= BYE_WEEK_REST_DAYS),
        "is_primetime": int(bool(gametime) and gametime >= PRIMETIME_GAMETIME),
        "div_game": int(bool(g.get("div_game"))),
        "is_international": int(g.get("stadium") in INTERNATIONAL_VENUES),
        "home_tz_shift_hours": home_tz_shift,
        "away_tz_shift_hours": away_tz_shift,
        "away_cross_country_travel": int(away_tz_shift is not None and abs(away_tz_shift) >= TZ_SHIFT_SIGNIFICANT_HOURS),
    }
