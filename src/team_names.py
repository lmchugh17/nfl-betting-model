"""Team-abbreviation handling: franchise continuity across relocations, plus
name->abbr matching for external sources that don't use nflverse abbreviations
(The Odds API's "Kansas City Chiefs", Polymarket's plain names).

Generalises the CFB build's ``team_names.py`` -- same idea (one shared lookup so
the mappings don't drift), different sport.
"""
import unicodedata

# nflverse keeps historical abbreviations distinct in `schedules`:
#   Rams     STL (1999-2015) -> LA  (2016+)
#   Chargers SD  (1999-2016) -> LAC (2017+)
#   Raiders  OAK (1999-2019) -> LV  (2020+)
# Alias each to ONE continuous franchise id so ELO / SRS / rolling form / ATS /
# H2H carry across the move -- the roster and coaching staff are what those
# ratings track, and they don't reset when the city changes. Everything
# venue-bound (roof/surface/temp/wind/stadium_id) stays on the game row and
# follows the actual stadium, so a dome move is captured with no extra code.
#
# The canonical id is the *current* nflverse abbreviation. (Some nflverse
# datasets historically emitted "LAR" for the Rams; `schedules` uses "LA", so
# LA is canonical and LAR folds into it.)
FRANCHISE_ALIASES = {
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
    "LAR": "LA",
}

# The 32 current franchises (canonical abbreviations).
TEAM_ABBRS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
}


def franchise_id(abbr: str) -> str:
    """Continuous franchise id for a (possibly historical) team abbreviation."""
    return FRANCHISE_ALIASES.get(abbr, abbr)


def normalize(name: str) -> str:
    """Lowercase, strip diacritics/punctuation/whitespace -- for fuzzy name matching."""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "".join(ch for ch in stripped.lower() if ch.isalnum())


# Manual overrides for external-source names that don't cleanly reduce to a
# nickname or "City Nickname" match. Purely additive.
_NAME_OVERRIDES = {
    "washington football team": "WAS",
    "washington redskins": "WAS",
    "las vegas raiders": "LV",
    "oakland raiders": "LV",       # franchise-folded
    "san diego chargers": "LAC",
    "st. louis rams": "LA",
    "st louis rams": "LA",
    "la rams": "LA",
    "los angeles rams": "LA",
    "la chargers": "LAC",
}


def build_name_lookup(teams_df) -> dict:
    """{normalized external name -> canonical abbr} from an nflverse teams frame
    (columns team_abbr, team_name, team_nick). Covers full name, nickname, and
    "city nick"; relocation abbrs fold to the current franchise."""
    lookup = {}
    for _, row in teams_df.iterrows():
        abbr = franchise_id(row["team_abbr"])
        for key in (row.get("team_name"), row.get("team_nick")):
            if isinstance(key, str) and key:
                lookup[normalize(key)] = abbr
    lookup.update({normalize(k): v for k, v in _NAME_OVERRIDES.items()})
    return lookup


def resolve_name(name: str, lookup: dict) -> str | None:
    """External name -> canonical abbr, or None if unmatched (caller logs it)."""
    return lookup.get(normalize(name))
