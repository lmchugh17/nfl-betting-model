"""Polymarket's public Gamma API (no auth needed for read-only market data).

Scope: a PASSIVE POST-GAME ACCURACY CHECK, not live edge detection -- same
principle as the CFB build. This project's spread and moneyline picks already
cover the "generate an actionable pick" role; Polymarket here only answers,
after the fact, whether the model's pre-game win probability was closer to the
truth than Polymarket's crowd-sourced one was.

Endpoints confirmed empirically (2026-09-04), not from docs:
- GET /events?tag_slug=nfl&closed=false&end_date_min=...&end_date_max=...&limit=100&offset=N
  Returns a MIX of NFL-tagged content, not just game moneylines -- season-long
  props ("Tush Push banned for 2026?"), player futures, Week-1-starting-QB
  questions, and "Season Series Winner" bets (a division-rivalry prop, NOT a
  single-game market -- easy to mistake for a real game by its "X vs. Y" title)
  all carry the same `nfl` tag. The actual per-game markets are the ones that
  ALSO carry a `games` tag (confirmed: "Patriots vs. Seahawks", "49ers vs.
  Rams", etc. all have tags=['sports','nfl','games']) -- filtering on that is
  what isolates real game events, no equivalent filtering step existed in the
  CFB build since CFBD's `cfb` tag was apparently game-only already.
- Each qualifying event embeds a `markets` array with ~100+ entries (spread at
  every line, totals, quarter/half splits, player props, exact-margin bets).
  The base moneyline market is still the one whose `question` exactly equals
  the event's own `title` ("Patriots vs. Seahawks") -- same CFB convention,
  confirmed still holds for NFL events.
- That market's `outcomes` (JSON-encoded list of two team names, e.g.
  ["Patriots","Seahawks"] -- short nickname only, not "New England Patriots")
  and `outcomePrices` (JSON-encoded probability strings, index-matched) are
  the actual win probabilities.
"""
import json

import requests

BASE_URL = "https://gamma-api.polymarket.com"
PAGE_SIZE = 100
REQUEST_TIMEOUT_S = 10  # same reasoning as src/weather_client.py -- fail fast, not 30s
GAME_EVENT_TAG = "games"  # distinguishes a real per-game market from nfl-tagged props/futures


def fetch_nfl_events(start_iso: str, end_iso: str) -> list[dict]:
    """Returns raw per-game NFL moneyline-market events whose market resolves
    (endDate) within [start_iso, end_iso). Paginated internally; already
    filtered to events carrying the `games` tag (drops props/futures/season-
    series bets that also carry the broader `nfl` tag)."""
    events = []
    offset = 0
    while True:
        resp = requests.get(
            f"{BASE_URL}/events",
            params={
                "tag_slug": "nfl", "closed": "false", "limit": PAGE_SIZE, "offset": offset,
                "end_date_min": start_iso, "end_date_max": end_iso,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        page = resp.json()
        events.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return [e for e in events if GAME_EVENT_TAG in [t.get("slug") for t in e.get("tags", [])]]


_NOT_MONEYLINE_OUTCOMES = {"over", "under", "yes", "no"}
_NOT_MONEYLINE_QUESTION_MARKERS = ("spread:", "1h ", "2h ", "1q ", "2q ", "3q ", "4q ", "o/u")


def _looks_like_moneyline(m: dict, outcomes: list) -> bool:
    q = (m.get("question") or "").lower()
    if any(marker in q for marker in _NOT_MONEYLINE_QUESTION_MARKERS):
        return False
    return not any(o.strip().lower() in _NOT_MONEYLINE_OUTCOMES for o in outcomes)


def _parse_market(m: dict) -> dict | None:
    try:
        outcomes = json.loads(m["outcomes"])
        prices = [float(p) for p in json.loads(m["outcomePrices"])]
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if len(outcomes) != 2 or len(prices) != 2:
        return None
    return {"team_a": outcomes[0], "team_b": outcomes[1], "prob_a": prices[0], "prob_b": prices[1]}


def extract_moneyline(event: dict) -> dict | None:
    """Returns {'team_a', 'team_b', 'prob_a', 'prob_b'} for the event's base
    moneyline market, or None if it's missing/malformed. team_a/prob_a and
    team_b/prob_b are index-matched pairs, not home/away -- caller resolves that.

    Primary match: the market whose `question` exactly equals the event's own
    `title` -- true for the overwhelming majority of events (confirmed on real
    regular-season events, e.g. "Patriots vs. Seahawks" == "Patriots vs.
    Seahawks"). Falls back to the first market that "looks like" a moneyline
    (not a spread/total/quarter-or-half split, outcomes aren't Over/Under/
    Yes/No) when that exact match fails -- needed for the Super Bowl
    specifically, confirmed empirically: Polymarket titles that event with
    CITY names ("Seattle vs. New England", presumably to sidestep "Super
    Bowl" trademark enforcement -- other NFL sportsbooks use "the Big Game"
    the same way) while its own moneyline market question still uses
    NICKNAMES ("Seahawks vs. Patriots") -- the one game per season where the
    exact-match assumption this was ported from (CFB, and this build's own
    regular-season testing) doesn't hold.
    """
    markets = event.get("markets", [])
    title = event.get("title")
    for m in markets:
        if m.get("question") == title:
            parsed = _parse_market(m)
            if parsed:
                return parsed
    for m in markets:
        parsed = _parse_market(m)
        if parsed and _looks_like_moneyline(m, [parsed["team_a"], parsed["team_b"]]):
            return parsed
    return None
