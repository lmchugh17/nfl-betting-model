"""Forecast-fills `games.temp` / `games.wind` for upcoming outdoor games.

Open-Meteo's forecast endpoint only covers ~16 days ahead, so games further out
get no fill yet -- correctly falls back to "no weather signal" until a later
pull gets close enough. Meant to run on the same weekly cadence as the odds
pull; each run overwrites the previous (less accurate, further-out) forecast,
which is the point -- `games` has one row per game (no timestamp dimension),
so this is "latest known value," not a time series, same convention CFB's
game_weather table used.

Also backfills `is_dome` for any upcoming row where nflverse left `roof` null
(confirmed on the 2026 schedule at several stadiums, incl. every international
venue before the schedule is finalized) -- falls back to
src.stadiums.STADIUMS[...]['roof_default'] so the dome/outdoor decision doesn't
silently break for exactly the games furthest in the future.

Once a game's date passes, backfill_schedules.py's next run overwrites this
forecast with nflverse's own post-game recorded conditions -- this script never
needs to "fix" a completed game itself.

Usage: .venv/bin/python scripts/pull_weather_forecast.py
"""
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import stadiums
from src.db import get_stats_connection, init_stats_db
from src.weather_client import FORECAST_HORIZON_DAYS, fetch_forecast, nearest_hour, polite_sleep

MAX_CONSECUTIVE_FAILURES = 10  # circuit breaker -- see src/weather_client.py docstring


def kickoff_utc_iso(gameday: str, gametime: str, tz_name: str) -> str | None:
    try:
        local = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(tz_name))
    except (ValueError, TypeError):
        return None
    return local.astimezone(timezone.utc).isoformat()


def main():
    init_stats_db()
    conn = get_stats_connection()
    try:
        now = datetime.now(timezone.utc)
        horizon_date = (now + timedelta(days=FORECAST_HORIZON_DAYS)).strftime("%Y-%m-%d")
        today = now.strftime("%Y-%m-%d")

        rows = conn.execute(
            """SELECT game_id, stadium, roof, gameday, gametime FROM games
               WHERE home_score IS NULL AND gameday BETWEEN ? AND ?""",
            (today, horizon_date),
        ).fetchall()
        print(f"{len(rows)} upcoming game(s) within the {FORECAST_HORIZON_DAYS}-day forecast horizon.")

        outdoor_by_venue = defaultdict(list)   # (lat, lon) -> [(game_id, kickoff_utc_iso)]
        dome_fixed = 0
        for game_id, stadium_name, roof, gameday, gametime in rows:
            info = stadiums.lookup(stadium_name)
            if info is None:
                print(f"  WARN: unrecognized stadium {stadium_name!r} for {game_id} -- skipping")
                continue
            forced = stadium_name in stadiums.FORCE_ROOF_OVERRIDE
            effective_roof = info["roof_default"] if (forced or roof is None) else roof
            if roof is None or forced:
                conn.execute(
                    "UPDATE games SET is_dome = ? WHERE game_id = ?",
                    (int(effective_roof in ("closed", "dome")), game_id),
                )
                dome_fixed += 1
            if effective_roof in ("closed", "dome"):
                continue
            kickoff = kickoff_utc_iso(gameday, gametime, info["timezone"])
            if kickoff is None:
                continue
            outdoor_by_venue[(info["lat"], info["lon"])].append((game_id, kickoff))
        conn.commit()
        if dome_fixed:
            print(f"Set is_dome for {dome_fixed} row(s) (null `roof`, or a known-unreliable venue tag).")

        venues = list(outdoor_by_venue.items())
        random.shuffle(venues)  # spread a circuit-breaker trip across venues run to run, not always the same tail
        print(f"Fetching forecast weather for {len(venues)} outdoor venue(s) "
              f"covering {sum(len(g) for _, g in venues)} game(s)...", flush=True)

        fetched, skipped, consecutive_failures = 0, 0, 0
        for i, ((lat, lon), games) in enumerate(venues, 1):
            try:
                hourly = fetch_forecast(lat, lon)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"  [{i}/{len(venues)}] WARN: forecast fetch failed for ({lat},{lon}): {e}", flush=True)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    remaining = len(venues) - i
                    print(f"  {consecutive_failures} venues in a row failed -- stopping early "
                          f"({remaining} venue(s) skipped, will retry next pull).", flush=True)
                    break
                continue
            for game_id, kickoff in games:
                weather = nearest_hour(hourly, kickoff)
                if weather is None:
                    skipped += 1
                    continue
                conn.execute(
                    "UPDATE games SET temp = ?, wind = ?, weather_source = 'open-meteo-forecast' WHERE game_id = ?",
                    (weather["temp_f"], weather["wind_mph"], game_id),
                )
                fetched += 1
            conn.commit()
            polite_sleep()

        print(f"Wrote forecast weather for {fetched} game(s). "
              f"{skipped} fell outside the forecast's hourly range (kickoff at the {FORECAST_HORIZON_DAYS}-day edge).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
