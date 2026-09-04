"""Thin wrapper around nflreadpy.

nflreadpy returns Polars frames; every function here hands back a pandas
DataFrame (``.to_pandas()``) so the rest of the codebase stays pandas like the
CFB build. Filesystem caching is pointed at ``data/cache/`` so repeated dev runs
don't re-download -- CI runners start clean so they always fetch fresh.

nfl_data_py is deprecated (nflverse redirects new projects to nflreadpy), so this
is the only NFL data client.
"""
from pathlib import Path

import nflreadpy
import pandas as pd
from nflreadpy.config import update_config

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

update_config(
    cache_mode="filesystem",
    cache_dir=_CACHE_DIR,      # must be a Path, not str
    cache_duration=6 * 3600,   # 6h: long enough for a dev session, short enough to catch mid-week nflverse refreshes
)


def _pd(polars_df) -> pd.DataFrame:
    return polars_df.to_pandas()


def schedules() -> pd.DataFrame:
    """Every game 1999 -> present PLUS the full upcoming schedule (future rows
    have NULL scores/lines). One table = games + closing lines + post-game
    weather + QB ids + coaches. 46 columns."""
    return _pd(nflreadpy.load_schedules())


def teams() -> pd.DataFrame:
    """Team reference: abbr, name, nickname, conference, division, colors, logos.
    Includes historical relocation abbrs (OAK, SD, STL) alongside current ones."""
    return _pd(nflreadpy.load_teams())


def pbp(seasons: list[int]) -> pd.DataFrame:
    """Play-by-play with EPA/WPA/success/air-yards. Large (~1M plays x ~380 cols
    across all seasons) -- pass an explicit season list and aggregate + discard,
    never store raw."""
    return _pd(nflreadpy.load_pbp(seasons))


def injuries(seasons: list[int]) -> pd.DataFrame:
    """Official injury reports, 2009+. report_status in {Out, Doubtful,
    Questionable}; includes preseason ('PRE') rows."""
    return _pd(nflreadpy.load_injuries(seasons))


def depth_charts(seasons: list[int]) -> pd.DataFrame:
    """Depth charts, 2001+ (depth_team == 1 is the starter)."""
    return _pd(nflreadpy.load_depth_charts(seasons))


def snap_counts(seasons: list[int]) -> pd.DataFrame:
    """Snap counts, 2012+ (offense/defense/special-teams snap share per player per game)."""
    return _pd(nflreadpy.load_snap_counts(seasons))


def rosters_weekly(seasons: list[int]) -> pd.DataFrame:
    """Week-by-week rosters, 2002+."""
    return _pd(nflreadpy.load_rosters_weekly(seasons))


def clear_cache() -> None:
    nflreadpy.clear_cache()
