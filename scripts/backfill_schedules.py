"""Backfill `teams` and `games` from the nflverse schedule.

Usage:  .venv/bin/python scripts/backfill_schedules.py [start_season] [end_season]

Defaults to 1999 .. current. The model only trains on 2016+, but the full history
is pulled so ELO/SRS have a long burn-in before the training window opens, and so
the upcoming slate (future rows, NULL scores) is present for prediction.

One nflverse call (`load_schedules`) fills what took three stitched sources in the
CFB build: games + closing betting lines + post-game weather.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import nflverse_client as nv
from src.db import get_stats_connection, init_stats_db
from src.team_names import franchise_id

DEFAULT_START = 1999

# Every column copied straight from the schedule frame (same name in `games`).
_PASSTHROUGH = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "away_team", "home_team", "away_score", "home_score", "location", "result",
    "total", "overtime", "div_game", "away_rest", "home_rest",
    "spread_line", "away_spread_odds", "home_spread_odds",
    "total_line", "over_odds", "under_odds", "away_moneyline", "home_moneyline",
    "roof", "surface", "stadium_id", "stadium", "temp", "wind",
    "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name",
    "away_coach", "home_coach", "referee",
    "old_game_id", "gsis", "nfl_detail_id", "pfr", "pff", "espn", "ftn",
]

_GAMES_COLUMNS = _PASSTHROUGH + [
    "season_type", "neutral_site", "is_dome", "home_margin", "home_win", "weather_source",
]


def _clean(v):
    """NaN / NaT / '' -> None; leave everything else."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def build_teams(conn, sched):
    """32 current franchises + 3 historical relocation abbrs (SD/STL/OAK), from
    the abbreviations that actually appear in the schedule, enriched with
    conference/division from nflverse `load_teams`."""
    ref = nv.teams().set_index("team_abbr")
    used = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    rows = 0
    for abbr in used:
        r = ref.loc[abbr] if abbr in ref.index else None
        full_name = (r["team_name"] if r is not None else abbr)
        conf = (r["team_conf"] if r is not None else None)
        div = (r["team_division"] if r is not None else None)
        conn.execute(
            """INSERT INTO teams (team, full_name, conference, division, franchise_id, espn_id)
               VALUES (?, ?, ?, ?, ?, NULL)
               ON CONFLICT(team) DO UPDATE SET
                   full_name=excluded.full_name, conference=excluded.conference,
                   division=excluded.division, franchise_id=excluded.franchise_id""",
            (abbr, full_name, conf, div, franchise_id(abbr)),
        )
        rows += 1
    return rows


def upsert_games(conn, sched):
    placeholders = ", ".join("?" for _ in _GAMES_COLUMNS)
    sql = f"INSERT OR REPLACE INTO games ({', '.join(_GAMES_COLUMNS)}) VALUES ({placeholders})"

    for g in sched.to_dict("records"):
        rec = {c: _clean(g.get(c)) for c in _PASSTHROUGH}

        rec["season_type"] = "REG" if g.get("game_type") == "REG" else "POST"
        rec["neutral_site"] = int(g.get("location") == "Neutral")

        roof = rec["roof"]
        rec["is_dome"] = int(roof in ("dome", "closed")) if roof is not None else None

        hs, as_ = rec["home_score"], rec["away_score"]
        if hs is not None and as_ is not None:
            rec["home_margin"] = int(hs) - int(as_)
            rec["home_win"] = int(int(hs) > int(as_))
        else:
            rec["home_margin"] = rec["home_win"] = None

        rec["weather_source"] = None  # nflverse post-game; Task 5 sets 'open-meteo-forecast' for upcoming games

        conn.execute(sql, tuple(rec[c] for c in _GAMES_COLUMNS))


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_START
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None

    init_stats_db()
    sched = nv.schedules()
    sched = sched[sched["season"] >= start]
    if end is not None:
        sched = sched[sched["season"] <= end]
    print(f"schedule rows {start}..{end or 'current'}: {len(sched)}")

    conn = get_stats_connection()
    try:
        n_teams = build_teams(conn, sched)
        conn.commit()
        upsert_games(conn, sched)
        conn.commit()

        completed = conn.execute("SELECT COUNT(*) FROM games WHERE home_score IS NOT NULL").fetchone()[0]
        upcoming = conn.execute("SELECT COUNT(*) FROM games WHERE home_score IS NULL").fetchone()[0]
        by_type = dict(conn.execute("SELECT season_type, COUNT(*) FROM games GROUP BY season_type"))
        miss_spread = conn.execute(
            "SELECT COUNT(*) FROM games WHERE season >= 2016 AND home_score IS NOT NULL AND spread_line IS NULL"
        ).fetchone()[0]
        print(f"\nteams: {n_teams}")
        print(f"games: {completed} completed + {upcoming} upcoming   by season_type: {by_type}")
        print(f"2016+ completed games missing spread_line: {miss_spread}")
        print("\nrelocation check (franchise_id):")
        for row in conn.execute(
            "SELECT team, franchise_id, conference, division FROM teams "
            "WHERE team IN ('SD','LAC','STL','LA','OAK','LV') ORDER BY franchise_id, team"
        ):
            print("  ", row)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
