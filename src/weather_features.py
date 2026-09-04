"""Team-level ATS performance in adverse-weather games, plus the "is this game
itself in adverse weather" flag. Same no-leakage discipline as ELO/SRS/EPA/ATS:
a team's split only ever uses its own strictly prior games.

Differences from the CFB build's version (which this is ported from):

- **No precipitation signal.** nflverse's `games` table carries `temp`/`wind`
  for completed games but no precipitation field at all -- Open-Meteo's forecast
  endpoint (src/weather_client.py) can return it for UPCOMING games, but there
  is no historical equivalent for the ~2,700 already-completed training games,
  so this build can't use precip as a training feature without a systematic gap
  between train and live. Dropped rather than half-implemented.
- **Thresholds measured directly against 2016-2025 NFL weather, not reused from
  CFB.** temp<=32F OR wind>=20mph flags 172 of 1,968 outdoor completed games
  (8.7%) -- literal freezing-or-high-wind, and (unlike CFB's initial 6-game
  problem) the NFL's much larger and more geographically spread outdoor slate
  gives this a real per-team split worth learning from: warm/dome-adjacent
  teams (LAC 1, IND 2, NO 2, ARI 3) barely ever see it, classic cold-weather
  outdoor teams (BUF 32, GB 30, KC 27, PIT 22, CLE 21) see it routinely across
  the 10-season backfill.
- **Weather lives directly on `games.temp`/`wind`/`is_dome`**, not a separate
  `game_weather` table -- see src/db.py and Task 5.
"""
from collections import defaultdict

ADVERSE_TEMP_F = 32
ADVERSE_WIND_MPH = 20
ADVERSE_WX_ROLLING_WINDOW = 8  # matches the general rolling windows here -- adverse games are still rare
MIN_ADVERSE_GAMES = 2         # below this, treat the team's split as unknown rather than trust a noisy small sample


def is_adverse(temp_f, wind_mph) -> bool:
    if temp_f is not None and temp_f <= ADVERSE_TEMP_F:
        return True
    if wind_mph is not None and wind_mph >= ADVERSE_WIND_MPH:
        return True
    return False


def was_game_adverse(g: dict) -> bool:
    """g: a games row dict with is_dome, temp, wind."""
    if g.get("is_dome"):
        return False  # domes/closed roofs are climate-controlled -- never adverse
    return is_adverse(g.get("temp"), g.get("wind"))


def compute_adverse_wx_ats_pct(ats_rows: list[dict], games_by_id: dict) -> dict:
    """Training-time: returns {(game_id, team): (trailing_adverse_wx_ats_pct, n) |
    (None, n)} before this game -- n is the actual adverse-weather game count
    behind the pct (even when below MIN_ADVERSE_GAMES, so an explanation, task
    10, can say precisely why a team's split is being treated as unknown, e.g.
    "only 1 prior cold-weather game"). ats_rows: ats_and_situational
    .compute_ats_results output (needs game_id, season, gameday, team,
    covered). games_by_id: {game_id: games row dict}."""
    by_team = defaultdict(list)
    for row in ats_rows:
        by_team[row["team"]].append(row)

    result = {}
    for team, rows in by_team.items():
        rows.sort(key=lambda r: (r["season"], r["gameday"]))
        history = []
        for row in rows:
            decided = history[-ADVERSE_WX_ROLLING_WINDOW:]
            result[(row["game_id"], team)] = (
                (sum(decided) / len(decided), len(decided)) if len(decided) >= MIN_ADVERSE_GAMES
                else (None, len(decided))
            )
            g = games_by_id.get(row["game_id"])
            if g is not None and was_game_adverse(g) and row["covered"] is not None:
                history.append(row["covered"])
    return result
