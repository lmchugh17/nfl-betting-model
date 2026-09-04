"""Pulls pre-game Polymarket win probabilities for upcoming games and writes/
refreshes their polymarket_odds rows -- a passive accuracy benchmark, not a
live pick source (see src/polymarket_client.py's docstring for the scope
decision).

Meant to run on the same weekly cadence as the odds pull. Re-running this each
cycle deliberately overwrites the earlier (less accurate, further from
kickoff) probability via polymarket_odds' PK (game_id only, no timestamp) --
same "latest known value" convention as games.temp/wind
(pull_weather_forecast.py).

Usage: .venv/bin/python scripts/pull_polymarket.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_stats_connection, init_stats_db
from src.polymarket_client import extract_moneyline, fetch_nfl_events
from src.team_names import build_name_lookup, resolve_name
from src import nflverse_client as nv

# Games more than ~2 weeks out rarely have a Polymarket game-level market open
# yet (CFB's own empirical finding; confirmed here too -- the Week 1 slate,
# ~5 days out, already had full markets, matching CFB's "~1 week before
# kickoff" observation).
HORIZON_DAYS = 16


def main():
    init_stats_db()
    conn = get_stats_connection()
    scraped_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        now = datetime.now(timezone.utc)
        start_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = (now + timedelta(days=HORIZON_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"Fetching Polymarket NFL game events ({start_iso} to {end_iso})...")
        try:
            events = fetch_nfl_events(start_iso, end_iso)
        except Exception as e:
            print(f"WARN: Polymarket fetch failed entirely: {e} -- nothing to write this run, "
                  "next scheduled pull will retry.")
            return
        print(f"Got {len(events)} game event(s).")

        lookup = build_name_lookup(nv.teams())
        games = conn.execute(
            """SELECT game_id, home_team, away_team, gameday FROM games
               WHERE home_score IS NULL AND gameday BETWEEN ? AND ?""",
            (start_iso[:10], end_iso[:10]),
        ).fetchall()
        # Match our games to Polymarket events by (home_abbr, away_abbr) pair,
        # not by date/slug string-matching -- more robust to timezone/slug quirks.
        games_by_teams = {}
        for game_id, home, away, gameday in games:
            games_by_teams[(home, away)] = game_id

        matched, unmatched_teams = 0, set()
        for event in events:
            ml = extract_moneyline(event)
            if ml is None:
                continue
            abbr_a = resolve_name(ml["team_a"], lookup)
            abbr_b = resolve_name(ml["team_b"], lookup)
            if abbr_a is None:
                unmatched_teams.add(ml["team_a"])
            if abbr_b is None:
                unmatched_teams.add(ml["team_b"])
            if abbr_a is None or abbr_b is None:
                continue
            # Polymarket's outcome order isn't guaranteed to be home-first --
            # resolve against our own games table, which knows which team is home.
            game_id = games_by_teams.get((abbr_a, abbr_b))
            if game_id is not None:
                home_prob, away_prob = ml["prob_a"], ml["prob_b"]
            else:
                game_id = games_by_teams.get((abbr_b, abbr_a))
                if game_id is None:
                    continue
                home_prob, away_prob = ml["prob_b"], ml["prob_a"]
            conn.execute(
                """INSERT OR REPLACE INTO polymarket_odds
                   (game_id, scraped_at, polymarket_event_id, home_prob, away_prob)
                   VALUES (?, ?, ?, ?, ?)""",
                (game_id, scraped_at, str(event.get("id")), home_prob, away_prob),
            )
            matched += 1
        conn.commit()

        print(f"Matched and wrote {matched} game(s) to polymarket_odds.")
        if unmatched_teams:
            print(f"WARN: {len(unmatched_teams)} Polymarket team name(s) didn't match: {sorted(unmatched_teams)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
