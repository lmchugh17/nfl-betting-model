"""Backfill `team_game_epa` from play-by-play.

Usage:  .venv/bin/python scripts/backfill_epa.py [start_season] [end_season]

Defaults to 2016 .. current (EPA form is a training feature and the model window
opens in 2016; earlier pbp exists back to 1999 if ever needed for ELO/EPA
burn-in). One season at a time so the raw pbp frame (~50k rows x 372 cols) is
loaded, aggregated to ~570 rows, and dropped before the next season.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import nflverse_client as nv
from src.db import get_stats_connection, init_stats_db
from src.epa_features import aggregate_team_game_epa

DEFAULT_START = 2016

_COLS = [
    "game_id", "team", "opponent", "season", "week", "is_home",
    "off_epa_play", "def_epa_play", "off_pass_epa", "off_rush_epa",
    "off_early_down_epa", "off_success_rate", "def_success_rate",
    "explosive_play_rate", "def_explosive_rate", "rz_td_pct",
    "pressure_rate_def", "plays", "sec_per_play", "pass_rate", "neutral_pass_rate",
]


def _latest_pbp_season() -> int:
    """nflreadpy.load_pbp rejects seasons past this -- it lags load_schedules,
    which already carries the upcoming (unplayed) season."""
    import nflreadpy
    return int(nflreadpy.get_current_season())


def write_season(conn, df) -> int:
    ph = ", ".join("?" for _ in _COLS)
    sql = f"INSERT OR REPLACE INTO team_game_epa ({', '.join(_COLS)}) VALUES ({ph})"
    for r in df.to_dict("records"):
        conn.execute(sql, tuple(None if (isinstance(r[c], float) and r[c] != r[c]) else r[c] for c in _COLS))
    return len(df)


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_START
    requested_end = int(sys.argv[2]) if len(sys.argv) > 2 else _latest_pbp_season()
    end = min(requested_end, _latest_pbp_season())
    if end < requested_end:
        print(f"note: pbp not available past {end}; the {end + 1}+ season(s) will be "
              f"picked up once nflverse publishes their play-by-play")

    init_stats_db()
    conn = get_stats_connection()
    try:
        total = 0
        for season in range(start, end + 1):
            try:
                pbp = nv.pbp([season])
            except ValueError as e:
                print(f"{season}: {e}; skipping")
                continue
            if pbp.empty:
                print(f"{season}: no pbp yet, skipping")
                continue
            agg = aggregate_team_game_epa(pbp)
            n = write_season(conn, agg)
            conn.commit()
            total += n
            print(f"{season}: {len(pbp):>6} plays -> {n} team-game rows")
            del pbp, agg

        print(f"\ntotal team_game_epa rows: {total}")
        # sanity: 2023 offensive-EPA leaders should look like real 2023 (SF, MIA, DAL, BUF, KC...)
        print("\n2023 top-5 season off_epa_play (min 10 games):")
        for row in conn.execute("""
            SELECT team, ROUND(AVG(off_epa_play), 3) e, COUNT(*) n
            FROM team_game_epa WHERE season = 2023 GROUP BY team HAVING n >= 10
            ORDER BY e DESC LIMIT 5
        """):
            print("  ", row)
        print("\n2023 top-5 defenses (lowest def_epa_play allowed):")
        for row in conn.execute("""
            SELECT team, ROUND(AVG(def_epa_play), 3) e, COUNT(*) n
            FROM team_game_epa WHERE season = 2023 GROUP BY team HAVING n >= 10
            ORDER BY e ASC LIMIT 5
        """):
            print("  ", row)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
