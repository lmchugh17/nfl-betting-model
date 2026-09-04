"""The Odds API client. Free tier: 500 credits/month, cost = markets x regions
per call. Ported from the CFB build near-verbatim -- same account, same client,
just a different default sport key."""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.the-odds-api.com/v4"


class OddsAPIClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY not set (check .env)")

    def get_odds(self, sport: str = "americanfootball_nfl", regions: str = "us",
                 markets: str = "spreads,totals,h2h",
                 commence_time_from: str | None = None,
                 commence_time_to: str | None = None) -> tuple[list, dict]:
        """Returns (games, quota_info). quota_info has 'remaining' and 'used' from
        response headers. Cost is markets x regions per call regardless of how
        many games/bookmakers come back -- confirmed live: unfiltered returns
        every game with any posted line (272 games, ~2 books avg, mostly thin
        futures-style lines for games months out) for the SAME 3 credits as a
        commence_time-windowed call (16 games, 9 books avg -- real market depth),
        so narrowing the window is a strictly better default, not a cost tradeoff."""
        params = {"apiKey": self.api_key, "regions": regions, "markets": markets,
                  "oddsFormat": "american", "dateFormat": "iso"}
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        resp = requests.get(
            f"{BASE_URL}/sports/{sport}/odds",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        quota = {
            "remaining": resp.headers.get("x-requests-remaining"),
            "used": resp.headers.get("x-requests-used"),
            "last_cost": resp.headers.get("x-requests-last"),
        }
        return resp.json(), quota
