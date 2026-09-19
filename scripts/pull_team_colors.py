"""Writes data/team_colors.json -- each current team's primary and alternate color --
for the site's team-color theme picker (scripts/build_site.py). Run by hand when a
team rebrands; the site build itself makes no network call.

Source: nflverse's team table (nflreadpy.load_teams(), free, no key). It also lists
retired franchise codes (OAK, SD, STL) and a duplicate LAR row, so only teams that
appear in the latest season's schedule are kept -- 32, keyed by the same
abbreviations games/teams use (e.g. 'LA' for the Rams).

Usage: .venv/bin/python scripts/pull_team_colors.py
"""
import json
import sys
from pathlib import Path

import nflreadpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_stats_connection

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "team_colors.json"


def main():
    conn = get_stats_connection()
    current = {row[0] for row in conn.execute("""
        SELECT home_team FROM games WHERE season = (SELECT MAX(season) FROM games)
        UNION SELECT away_team FROM games WHERE season = (SELECT MAX(season) FROM games)
    """)}
    full_names = dict(conn.execute("SELECT team, full_name FROM teams").fetchall())
    conn.close()

    teams = nflreadpy.load_teams().to_pandas()
    colors = {}
    for _, t in teams.iterrows():
        if t["team_abbr"] not in current:
            continue
        colors[t["team_abbr"]] = {
            "name": full_names.get(t["team_abbr"]) or t["team_name"],
            "color": t["team_color"].lower(),
            "alternate_color": t["team_color2"].lower() if isinstance(t["team_color2"], str) else None,
        }

    missing = current - colors.keys()
    if missing:
        sys.exit(f"No nflverse colors for: {sorted(missing)}")
    OUTPUT_PATH.write_text(json.dumps(dict(sorted(colors.items())), indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH} ({len(colors)} teams)")


if __name__ == "__main__":
    main()
