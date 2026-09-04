"""Backfill `injuries`, `snap_counts`, and the `players` crosswalk from nflverse.

Usage:  .venv/bin/python scripts/backfill_availability.py [start_season] [end_season]

Defaults to 2016 .. latest available. injuries start 2009, snap_counts 2012;
both lag load_schedules, so the season list is clamped to nflverse's current
data season. snap_counts is filtered to players with a real role
(offense_pct or defense_pct >= 0.10) -- the 1-snap-cameo tail is ~half the rows
and never matters for starter identification.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import nflverse_client as nv
from src.db import get_stats_connection, init_stats_db

DEFAULT_START = 2016
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


def _int(v):
    return None if pd.isna(v) else int(v)


def _str(v):
    return None if pd.isna(v) else str(v)


def backfill_players(conn):
    p = nflreadpy.load_players().to_pandas()
    p = p[p["gsis_id"].notna()]
    n = 0
    for r in p.to_dict("records"):
        conn.execute(
            """INSERT OR REPLACE INTO players (gsis_id, pfr_id, display_name, position, latest_team)
               VALUES (?, ?, ?, ?, ?)""",
            (r["gsis_id"], _str(r.get("pfr_id")), _str(r.get("display_name")),
             _str(r.get("position")), _str(r.get("latest_team"))),
        )
        n += 1
    return n


def backfill_injuries(conn, seasons):
    try:
        df = nv.injuries(seasons)
    except ValueError as e:
        print(f"  injuries: {e}")
        return 0
    df = df[df["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
    # nflverse serves the latest status per player-week already, but guard anyway.
    # date_modified is absent from some current-season pulls -- fall back to week order.
    sort_key = "date_modified" if "date_modified" in df.columns else "week"
    df = df.sort_values(sort_key).drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
    if "date_modified" not in df.columns:
        df = df.assign(date_modified=None)
    if "report_primary_injury" not in df.columns:
        df = df.assign(report_primary_injury=None)
    n = 0
    for r in df.to_dict("records"):
        conn.execute(
            """INSERT OR REPLACE INTO injuries
               (season, week, team, gsis_id, game_type, player_name, position,
                report_status, report_primary_injury, practice_status, date_modified, scraped_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_int(r["season"]), _int(r["week"]), r["team"], r["gsis_id"], r.get("game_type"),
             _str(r.get("full_name")), _str(r.get("position")), _str(r.get("report_status")),
             _str(r.get("report_primary_injury")), _str(r.get("practice_status")),
             _str(r.get("date_modified")), NOW),
        )
        n += 1
    return n


def backfill_snap_counts(conn, seasons):
    try:
        df = nv.snap_counts(seasons)
    except ValueError as e:
        print(f"  snap_counts: {e}")
        return 0
    df = df[df["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
    df = df[(df["offense_pct"].fillna(0) >= 0.10) | (df["defense_pct"].fillna(0) >= 0.10)]
    n = 0
    for r in df.to_dict("records"):
        conn.execute(
            """INSERT OR REPLACE INTO snap_counts
               (game_id, pfr_player_id, season, week, team, player, position,
                offense_snaps, offense_pct, defense_snaps, defense_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (r["game_id"], r["pfr_player_id"], _int(r["season"]), _int(r["week"]), r.get("team"),
             _str(r.get("player")), _str(r.get("position")),
             _int(r.get("offense_snaps")), r.get("offense_pct"),
             _int(r.get("defense_snaps")), r.get("defense_pct")),
        )
        n += 1
    return n


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_START
    requested_end = int(sys.argv[2]) if len(sys.argv) > 2 else datetime.now().year
    # nflverse rejects the WHOLE call if any season is past its data -- these feeds
    # all lag load_schedules, so clamp to nflverse's current data season.
    data_end = int(nflreadpy.get_current_season())
    end = min(requested_end, data_end)
    if end < requested_end:
        print(f"note: nflverse availability data only through {end}; "
              f"{end + 1}+ will be picked up once published")
    seasons = list(range(start, end + 1))

    init_stats_db()
    conn = get_stats_connection()
    try:
        print("players crosswalk...")
        np_ = backfill_players(conn); conn.commit()
        print(f"  {np_} players")
        print(f"injuries {start}-{end}...")
        ni = backfill_injuries(conn, seasons); conn.commit()
        print(f"  {ni} rows")
        print(f"snap_counts {start}-{end} (role players only)...")
        ns = backfill_snap_counts(conn, seasons); conn.commit()
        print(f"  {ns} rows")

        print("\ncoverage by season:")
        for row in conn.execute("""
            SELECT s.season,
              (SELECT COUNT(*) FROM injuries i WHERE i.season=s.season) inj,
              (SELECT COUNT(*) FROM snap_counts c WHERE c.season=s.season) snaps
            FROM (SELECT DISTINCT season FROM snap_counts UNION SELECT DISTINCT season FROM injuries) s
            ORDER BY s.season
        """):
            print(f"  {row[0]}: injuries={row[1]:>5}  snap_rows={row[2]:>6}")

        # spot check: 2023 NYJ Aaron Rodgers -- healthy wk1, then Out wk2+ (Achilles)
        print("\n2023 NYJ Aaron Rodgers injury timeline (first 4 listed weeks):")
        for row in conn.execute("""
            SELECT week, report_status, report_primary_injury FROM injuries
            WHERE season=2023 AND team='NYJ' AND player_name LIKE '%Rodgers%' ORDER BY week LIMIT 4
        """):
            print("  wk", row)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
