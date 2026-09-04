"""Real per-book spread pricing from live_odds, when available -- nflverse's
`games.spread_line` (the market_spread used for training and the edge calc) only
has the spread NUMBER, never its price. live_odds does, via scripts/pull_odds.py,
and per-book price genuinely varies (e.g. -110/-105/-101/-115/-111 across books on
the same side of the same game) -- worth using instead of the standing
-110-both-sides assumption whenever it's actually available.

Simpler than the CFB build's version: pull_odds.py already resolves each row's
home/away team name to a canonical abbr at write time, so this module just
matches outcome_name against the row's own home_team/away_team (no repeated
name-lookup rebuild needed here).
"""
from collections import defaultdict


def load_latest_spread_prices(conn) -> dict:
    """Returns {team_abbr: (median_price, book_count)} using only the most recent
    live_odds pull (scraped_at) so pricing reflects the current market rather than
    blending stale and fresh snapshots across the week."""
    latest = conn.execute("SELECT MAX(scraped_at) FROM live_odds").fetchone()[0]
    if latest is None:
        return {}
    rows = conn.execute(
        """SELECT outcome_name, price, home_team, away_team, home_team_abbr, away_team_abbr
           FROM live_odds WHERE market = 'spreads' AND scraped_at = ? AND price IS NOT NULL""",
        (latest,),
    ).fetchall()

    by_team = defaultdict(list)
    for outcome_name, price, home_team, away_team, home_abbr, away_abbr in rows:
        if outcome_name == home_team and home_abbr:
            by_team[home_abbr].append(price)
        elif outcome_name == away_team and away_abbr:
            by_team[away_abbr].append(price)

    result = {}
    for abbr, prices in by_team.items():
        prices.sort()
        n = len(prices)
        median = prices[n // 2] if n % 2 else (prices[n // 2 - 1] + prices[n // 2]) / 2
        result[abbr] = (round(median), n)
    return result


def get_spread_price(spread_prices: dict, team_abbr: str | None) -> tuple[int | None, int]:
    """Returns (median_price, book_count) for team_abbr, or (None, 0) if
    unavailable -- caller falls back to an assumed price (e.g. -110)."""
    if team_abbr is None:
        return None, 0
    return spread_prices.get(team_abbr, (None, 0))
