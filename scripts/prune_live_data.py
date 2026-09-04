"""Archives and prunes `live_odds`, the one snapshot table that grows
unboundedly -- it keys its primary key on `scraped_at`, so every pull writes a
brand-new snapshot instead of updating existing rows. Measured 2026-09-04: one
windowed pull (Task 6) wrote 802 rows; at a Wed/Fri/Sat/Sun-AM cadence that's
~3,200 rows/week of pure accumulation if never pruned. Run this after every
scripts/pull_odds.py.

Unlike the CFB build, `injuries` here does NOT need this treatment: nflverse's
injury feed is upserted (INSERT OR REPLACE on (season, week, team, gsis_id),
no scraped_at in the primary key -- see src/db.py) because nflverse itself
already serves "latest official status per player-week," not a raw scrape we
need to snapshot over time. There is nothing to prune.

Stale live_odds rows are archived to an append-only CSV
(data/live_odds_archive.csv -- explicitly un-ignored in .gitignore and
committed, since GitHub Actions runners have no persistent disk between runs)
before being removed from SQLite, rather than deleted outright: line-movement
history and real per-book spread pricing (games.spread_line only has the point,
never the price) are both plausible future features. Appending text lines to a
CSV also compresses far better in git history than re-committing a rewritten
binary SQLite blob on every pull.

Usage: .venv/bin/python scripts/prune_live_data.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_stats_connection, init_stats_db

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Once a game's kicked off there's no more line-shopping value in its odds -- a
# few days of buffer covers any lag between a game finishing and this script's
# next scheduled run.
LIVE_ODDS_RETENTION_DAYS = 3


def _archive_and_prune(conn, table: str, time_col: str, cutoff_iso: str, archive_path: Path) -> int:
    stale = pd.read_sql(f"SELECT * FROM {table} WHERE {time_col} IS NOT NULL AND {time_col} < ?",
                         conn, params=(cutoff_iso,))
    if stale.empty:
        return 0
    write_header = not archive_path.exists()
    stale.to_csv(archive_path, mode="a", header=write_header, index=False)
    conn.execute(f"DELETE FROM {table} WHERE {time_col} IS NOT NULL AND {time_col} < ?", (cutoff_iso,))
    conn.commit()
    return len(stale)


def main():
    init_stats_db()
    conn = get_stats_connection()
    try:
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=LIVE_ODDS_RETENTION_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

        n_odds = _archive_and_prune(conn, "live_odds", "commence_time", cutoff,
                                     DATA_DIR / "live_odds_archive.csv")
        print(f"live_odds: archived + pruned {n_odds} row(s) for games before {cutoff}")

        if n_odds:
            conn.execute("VACUUM")
            print("Vacuumed database to reclaim freed space.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
