"""In-season refresh of the official injury report (and current-season snap
counts / depth charts, if published).

Run by GitHub Actions on the weekly cadence. Unlike the CFB build's ESPN scrape,
this is just a re-pull of nflverse's official feed for the current season --
nflverse serves the latest status per player-week, so it's an idempotent upsert.

Usage:  .venv/bin/python scripts/scrape_injuries.py [season]
"""
import sys
from pathlib import Path

import nflreadpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_availability import backfill_injuries, backfill_snap_counts
from src.db import get_stats_connection, init_stats_db


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else int(nflreadpy.get_current_season())

    init_stats_db()
    conn = get_stats_connection()
    try:
        ni = backfill_injuries(conn, [season]); conn.commit()
        ns = backfill_snap_counts(conn, [season]); conn.commit()
        print(f"{season}: injuries={ni}  snap_counts={ns}")

        latest = conn.execute(
            "SELECT MAX(week) FROM injuries WHERE season = ?", (season,)
        ).fetchone()[0]
        n_out = conn.execute(
            "SELECT COUNT(*) FROM injuries WHERE season=? AND week=? AND report_status='Out'",
            (season, latest),
        ).fetchone()[0]
        print(f"latest week on report: {latest}  ({n_out} players Out)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
