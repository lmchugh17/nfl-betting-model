"""Static NFL stadium reference: lat/long + timezone + default roof, for the
Open-Meteo forecast pull (Task 5) and later situational features (Task 8:
cross-country travel / time-zone shift).

Keyed by **venue name**, not nflverse's `stadium_id`. Two reasons:

1. `stadium_id` is a per-franchise code (e.g. `JAX00`) that does NOT change for
   an international "home" game played somewhere else -- 2026 Week 5
   (`2026_05_PHI_JAX`) has `stadium_id='JAX00'` but `stadium='Tottenham Hotspur
   Stadium'`, because Jacksonville's multi-year London home-game deal keeps
   their normal stadium_id while nflverse updates the venue *name* to the real
   location. Keying on stadium_id would put that game's forecast in Jacksonville
   instead of London.
2. The venue name also correctly unifies venues nflverse tags under different
   stadium_ids depending on which team is "home" (Tottenham Hotspur Stadium
   appears as both `JAX00` and `LON02` across different games) -- keying on name
   collapses these back to one real place automatically.

Renames of the SAME physical stadium (sponsorship changes) are folded via
NAME_ALIASES so historical rows still resolve.

roof_default is used when a game's own `roof` column is null -- which happens
for future rows at several of these stadiums (nflverse doesn't always fill roof
ahead of time, confirmed on the 2026 schedule for ATL/DAL/HOU/IND/PHO/international
venues) -- so the dome/outdoor decision doesn't silently break for exactly the
games furthest in the future, which is when the forecast pull matters most.

FORCE_ROOF_OVERRIDE names a small set of venues where nflverse's own `roof`
value is confirmed unreliable and should be IGNORED even when present, not just
used as a null fallback: checked across every historical row, Melbourne Cricket
Ground (a fully open cricket ground, no roof, ever) is tagged 'dome'; Stade de
France (open-air field, partial seating canopy only) is tagged 'dome'; and the
Allianz Arena / FC Bayern Munich Stadium pair -- the SAME physical building --
is tagged 'outdoors' in one row and 'dome' in another. Domestic US stadiums are
NOT in this set because nflverse's roof tagging for them checks out against
known reality (AT&T/NRG/Lucas Oil retractable roofs correctly vary by game).
"""
import unicodedata

# {canonical name: {lat, lon, timezone (IANA), roof_default}}
# roof_default: 'outdoors' | 'closed' | 'dome' (climate-controlled, no forecast needed)
STADIUMS = {
    "Mercedes-Benz Stadium":            {"lat": 33.7554, "lon": -84.4008, "timezone": "America/New_York", "roof_default": "closed"},
    "M&T Bank Stadium":                 {"lat": 39.2780, "lon": -76.6227, "timezone": "America/New_York", "roof_default": "outdoors"},
    "Gillette Stadium":                 {"lat": 42.0909, "lon": -71.2643, "timezone": "America/New_York", "roof_default": "outdoors"},
    "Highmark Stadium":                 {"lat": 42.7738, "lon": -78.7870, "timezone": "America/New_York", "roof_default": "outdoors"},
    "Bank of America Stadium":          {"lat": 35.2258, "lon": -80.8528, "timezone": "America/New_York", "roof_default": "outdoors"},
    "Soldier Field":                    {"lat": 41.8623, "lon": -87.6167, "timezone": "America/Chicago",  "roof_default": "outdoors"},
    "Paycor Stadium":                   {"lat": 39.0954, "lon": -84.5160, "timezone": "America/New_York", "roof_default": "outdoors"},
    "Huntington Bank Field":            {"lat": 41.5061, "lon": -81.6995, "timezone": "America/New_York", "roof_default": "outdoors"},
    "AT&T Stadium":                     {"lat": 32.7473, "lon": -97.0945, "timezone": "America/Chicago",  "roof_default": "closed"},
    "Empower Field at Mile High":       {"lat": 39.7439, "lon": -105.0201, "timezone": "America/Denver",   "roof_default": "outdoors"},
    "Ford Field":                       {"lat": 42.3400, "lon": -83.0456, "timezone": "America/New_York", "roof_default": "dome"},
    "Lambeau Field":                    {"lat": 44.5013, "lon": -88.0622, "timezone": "America/Chicago",  "roof_default": "outdoors"},
    "NRG Stadium":                      {"lat": 29.6847, "lon": -95.4107, "timezone": "America/Chicago",  "roof_default": "closed"},
    "Lucas Oil Stadium":                {"lat": 39.7601, "lon": -86.1639, "timezone": "America/New_York", "roof_default": "closed"},
    "EverBank Stadium":                 {"lat": 30.3239, "lon": -81.6373, "timezone": "America/New_York", "roof_default": "outdoors"},
    "GEHA Field at Arrowhead Stadium":  {"lat": 39.0489, "lon": -94.4839, "timezone": "America/Chicago",  "roof_default": "outdoors"},
    "SoFi Stadium":                     {"lat": 33.9535, "lon": -118.3392, "timezone": "America/Los_Angeles", "roof_default": "dome"},
    "Hard Rock Stadium":                {"lat": 25.9580, "lon": -80.2389, "timezone": "America/New_York", "roof_default": "outdoors"},
    "U.S. Bank Stadium":                {"lat": 44.9735, "lon": -93.2575, "timezone": "America/Chicago",  "roof_default": "dome"},
    "Nissan Stadium":                   {"lat": 36.1665, "lon": -86.7713, "timezone": "America/Chicago",  "roof_default": "outdoors"},
    "Caesars Superdome":                {"lat": 29.9511, "lon": -90.0812, "timezone": "America/Chicago",  "roof_default": "dome"},
    "MetLife Stadium":                  {"lat": 40.8135, "lon": -74.0745, "timezone": "America/New_York", "roof_default": "outdoors"},
    "Lincoln Financial Field":          {"lat": 39.9008, "lon": -75.1675, "timezone": "America/New_York", "roof_default": "outdoors"},
    "State Farm Stadium":               {"lat": 33.5276, "lon": -112.2626, "timezone": "America/Phoenix", "roof_default": "closed"},
    "Acrisure Stadium":                 {"lat": 40.4468, "lon": -80.0158, "timezone": "America/New_York", "roof_default": "outdoors"},
    "Lumen Field":                      {"lat": 47.5952, "lon": -122.3316, "timezone": "America/Los_Angeles", "roof_default": "outdoors"},
    "Levi's Stadium":                   {"lat": 37.4033, "lon": -121.9694, "timezone": "America/Los_Angeles", "roof_default": "outdoors"},
    "Raymond James Stadium":            {"lat": 27.9759, "lon": -82.5033, "timezone": "America/New_York", "roof_default": "outdoors"},
    "Allegiant Stadium":                {"lat": 36.0909, "lon": -115.1833, "timezone": "America/Los_Angeles", "roof_default": "dome"},
    "Northwest Stadium":                {"lat": 38.9077, "lon": -76.8645, "timezone": "America/New_York", "roof_default": "outdoors"},
    # international
    "Wembley Stadium":                  {"lat": 51.5560, "lon": -0.2795, "timezone": "Europe/London", "roof_default": "outdoors"},
    "Tottenham Hotspur Stadium":        {"lat": 51.6043, "lon": -0.0664, "timezone": "Europe/London", "roof_default": "outdoors"},
    "Allianz Arena":                    {"lat": 48.2188, "lon": 11.6247, "timezone": "Europe/Berlin", "roof_default": "outdoors"},
    "Estadio Santiago Bernabeu":        {"lat": 40.4531, "lon": -3.6883, "timezone": "Europe/Madrid", "roof_default": "outdoors"},
    "Estadio Banorte":                  {"lat": 19.3029, "lon": -99.1505, "timezone": "America/Mexico_City", "roof_default": "outdoors"},
    "Melbourne Cricket Ground":         {"lat": -37.8199, "lon": 144.9834, "timezone": "Australia/Melbourne", "roof_default": "outdoors"},
    "Stade de France":                  {"lat": 48.9245, "lon": 2.3601, "timezone": "Europe/Paris", "roof_default": "outdoors"},
    "Maracana Stadium":                 {"lat": -22.9121, "lon": -43.2302, "timezone": "America/Sao_Paulo", "roof_default": "outdoors"},
    "Arena Corinthians":                {"lat": -23.5453, "lon": -46.4742, "timezone": "America/Sao_Paulo", "roof_default": "outdoors"},
    "Deutsche Bank Park":               {"lat": 50.0685, "lon": 8.6455, "timezone": "Europe/Berlin", "roof_default": "outdoors"},
    "Twickenham Stadium":               {"lat": 51.4548, "lon": -0.3407, "timezone": "Europe/London", "roof_default": "outdoors"},
    # retired/temporary venues still inside the 2016+ training window
    "Los Angeles Memorial Coliseum":    {"lat": 34.0141, "lon": -118.2879, "timezone": "America/Los_Angeles", "roof_default": "outdoors"},  # LA Rams' home 2016-2019, pre-SoFi
    "Oakland Coliseum":                 {"lat": 37.7516, "lon": -122.2005, "timezone": "America/Los_Angeles", "roof_default": "outdoors"},  # OAK's home through 2019, pre-LV
    "Qualcomm Stadium":                 {"lat": 32.7831, "lon": -117.1196, "timezone": "America/Los_Angeles", "roof_default": "outdoors"},  # SD's home through 2016, pre-LAC
    "Dignity Health Sports Park":       {"lat": 33.8644, "lon": -118.2611, "timezone": "America/Los_Angeles", "roof_default": "outdoors"},  # LAC's home 2017-2019, pre-SoFi
}

# Venues where games.roof is confirmed unreliable -- see module docstring.
FORCE_ROOF_OVERRIDE = {"Melbourne Cricket Ground", "Stade de France", "Allianz Arena"}

# Every non-US venue in STADIUMS -- for the international-game situational flag
# (task 8). All nine currently host only "home" games for a designated franchise
# (never a true neutral-market game for two other teams), so this list is
# maintained by hand rather than derived from a country field nflverse doesn't provide.
INTERNATIONAL_VENUES = {
    "Wembley Stadium", "Tottenham Hotspur Stadium", "Allianz Arena",
    "Estadio Santiago Bernabeu", "Estadio Banorte", "Melbourne Cricket Ground",
    "Stade de France", "Maracana Stadium", "Arena Corinthians",
}

# Each of the 32 current franchises' normal home venue -> STADIUMS key, for the
# time-zone-travel situational feature (task 8): a team's OWN timezone is looked
# up here and compared against the actual game venue's timezone (which already
# correctly reflects an international "home" game per the module docstring).
TEAM_HOME_STADIUM = {
    "ARI": "State Farm Stadium", "ATL": "Mercedes-Benz Stadium", "BAL": "M&T Bank Stadium",
    "BUF": "Highmark Stadium", "CAR": "Bank of America Stadium", "CHI": "Soldier Field",
    "CIN": "Paycor Stadium", "CLE": "Huntington Bank Field", "DAL": "AT&T Stadium",
    "DEN": "Empower Field at Mile High", "DET": "Ford Field", "GB": "Lambeau Field",
    "HOU": "NRG Stadium", "IND": "Lucas Oil Stadium", "JAX": "EverBank Stadium",
    "KC": "GEHA Field at Arrowhead Stadium", "LA": "SoFi Stadium", "LAC": "SoFi Stadium",
    "LV": "Allegiant Stadium", "MIA": "Hard Rock Stadium", "MIN": "U.S. Bank Stadium",
    "NE": "Gillette Stadium", "NO": "Caesars Superdome", "NYG": "MetLife Stadium",
    "NYJ": "MetLife Stadium", "PHI": "Lincoln Financial Field", "PIT": "Acrisure Stadium",
    "SEA": "Lumen Field", "SF": "Levi's Stadium", "TB": "Raymond James Stadium",
    "TEN": "Nissan Stadium", "WAS": "Northwest Stadium",
    # historical relocation-era abbrs still inside the 2016+ training window
    "SD": "Qualcomm Stadium", "OAK": "Oakland Coliseum",
}


def team_home_timezone(franchise_abbr: str) -> str | None:
    stadium_name = TEAM_HOME_STADIUM.get(franchise_abbr)
    info = STADIUMS.get(stadium_name) if stadium_name else None
    return info["timezone"] if info else None

# Sponsorship/name changes for the SAME physical building -> canonical STADIUMS key.
NAME_ALIASES = {
    "New Era Field": "Highmark Stadium",
    "Ralph Wilson Stadium": "Highmark Stadium",
    "FirstEnergy Stadium": "Huntington Bank Field",
    "Cleveland Browns Stadium": "Huntington Bank Field",
    "Reliant Stadium": "NRG Stadium",
    "TIAA Bank Stadium": "EverBank Stadium",
    "EverBank Field": "EverBank Stadium",
    "Arrowhead Stadium": "GEHA Field at Arrowhead Stadium",
    "Georgia Dome": "Mercedes-Benz Stadium",
    "Paul Brown Stadium": "Paycor Stadium",
    "Sports Authority Field at Mile High": "Empower Field at Mile High",
    "Ring Central Coliseum": "Oakland Coliseum",
    "RingCentral Coliseum": "Oakland Coliseum",
    "O.co Coliseum": "Oakland Coliseum",
    "Oakland-Alameda County Coliseum": "Oakland Coliseum",
    "StubHub Center": "Dignity Health Sports Park",
    "Mercedes-Benz Superdome": "Caesars Superdome",
    "Louisiana Superdome": "Caesars Superdome",
    "New Meadowlands Stadium": "MetLife Stadium",
    "Giants Stadium": "MetLife Stadium",
    "University of Phoenix Stadium": "State Farm Stadium",
    "FedExField": "Northwest Stadium",
    "Jack Kent Cooke Stadium": "Northwest Stadium",
    "Heinz Field": "Acrisure Stadium",
    "Three Rivers Stadium": "Acrisure Stadium",
    "CenturyLink Field": "Lumen Field",
    "Qwest Field": "Lumen Field",
    "Seahawks Stadium": "Lumen Field",
    "Tottenham Stadium": "Tottenham Hotspur Stadium",
    "Bernabeu": "Estadio Santiago Bernabeu",
    "Azteca Stadium": "Estadio Banorte",
    "FC Bayern Munich Stadium": "Allianz Arena",
}


def _norm(name: str) -> str:
    return unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().strip()


def lookup(stadium_name: str) -> dict | None:
    """Venue name (as it appears in games.stadium, any historical spelling) ->
    {lat, lon, timezone, roof_default}, or None if genuinely unrecognized."""
    if not stadium_name:
        return None
    if stadium_name in STADIUMS:
        return STADIUMS[stadium_name]
    canonical = NAME_ALIASES.get(stadium_name)
    if canonical:
        return STADIUMS.get(canonical)
    # last resort: accent/whitespace-insensitive match
    target = _norm(stadium_name)
    for name, info in STADIUMS.items():
        if _norm(name) == target:
            return info
    return None
