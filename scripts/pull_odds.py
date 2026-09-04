"""Pull current NFL odds from The Odds API and store as a timestamped snapshot.

Meant to run multiple times a week (Wed/Fri/Sat/Sun-AM per the agreed cadence)
so line movement is visible, not just a single closing line. Costs 3 credits
per run (spreads + totals + h2h markets, us region) regardless of the
commence-time window -- ~12 credits/week, ~220/season, well under the free
tier's 500/month and leaving headroom to keep sharing the account with the CFB
model.

Windowed to the next COMMENCE_WINDOW_DAYS days: confirmed live that an
unfiltered call returns the ENTIRE remaining season (272 games) at ~2
books/game average (mostly thin futures-style lines for games months out) for
the exact same 3-credit cost as a windowed call limited to the imminent slate
(16 games at ~9 books/game -- real market depth). Windowing is strictly better:
same cost, far less live_odds bloat, and every stored row reflects an actual
liquid market instead of a placeholder line.

Usage: .venv/bin/python scripts/pull_odds.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import nflverse_client as nv
from src.db import get_stats_connection, init_stats_db
from src.odds_client import OddsAPIClient
from src.team_names import build_name_lookup, resolve_name

COMMENCE_WINDOW_DAYS = 10  # covers the current NFL week (games run Thu-Mon) with margin


def main():
    init_stats_db()
    client = OddsAPIClient()
    conn = get_stats_connection()
    scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    window_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    window_end = (datetime.now(timezone.utc) + timedelta(days=COMMENCE_WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        lookup = build_name_lookup(nv.teams())
        games, quota = client.get_odds(commence_time_from=window_start, commence_time_to=window_end)
        print(f"Pulled odds for {len(games)} games (next {COMMENCE_WINDOW_DAYS}d). "
              f"Quota remaining: {quota['remaining']} (this call cost {quota['last_cost']}).")

        unmatched_teams = set()
        rows_written = 0
        for game in games:
            home_abbr = resolve_name(game["home_team"], lookup)
            away_abbr = resolve_name(game["away_team"], lookup)
            if home_abbr is None:
                unmatched_teams.add(game["home_team"])
            if away_abbr is None:
                unmatched_teams.add(game["away_team"])

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        conn.execute(
                            """INSERT OR REPLACE INTO live_odds
                               (odds_game_id, scraped_at, commence_time, home_team, away_team,
                                home_team_abbr, away_team_abbr, bookmaker, market, outcome_name, price, point)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                game["id"], scraped_at, game.get("commence_time"),
                                game["home_team"], game["away_team"], home_abbr, away_abbr,
                                bookmaker["key"], market["key"], outcome["name"],
                                outcome.get("price"), outcome.get("point"),
                            ),
                        )
                        rows_written += 1
        conn.commit()

        print(f"Wrote {rows_written} odds rows (snapshot: {scraped_at}).")
        if unmatched_teams:
            print(f"WARN: {len(unmatched_teams)} team names didn't match: {sorted(unmatched_teams)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
