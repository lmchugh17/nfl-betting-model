"""Open-Meteo client (free, no API key) -- forecast endpoint only.

Historical weather is NOT needed here (unlike the CFB build): nflverse's
`games` schedule already carries `temp`/`wind` for every completed game. This
client exists purely to forecast-fill upcoming outdoor games, ported
near-verbatim from the CFB build's src/weather_client.py.
"""
import time

import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_FIELDS = "temperature_2m,precipitation,windspeed_10m"
UNIT_PARAMS = {
    "temperature_unit": "fahrenheit",
    "windspeed_unit": "mph",
    "precipitation_unit": "inch",
    "timezone": "UTC",
}

# Short on purpose: pull_weather_forecast.py makes one call per venue in a loop.
# A slow/stuck request needs to fail fast, not eat 30s each -- the CFB build
# confirmed GitHub Actions' shared runner IPs can see Open-Meteo hang far longer
# than a normal network (likely rate-limiting/throttling aimed at datacenter/CI
# traffic on this free, unauthenticated API).
REQUEST_TIMEOUT_S = 10
FORECAST_HORIZON_DAYS = 16  # Open-Meteo's forecast endpoint cap


def _hourly_by_timestamp(payload: dict) -> dict:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    result = {}
    for i, ts in enumerate(times):
        result[ts] = {
            "temp_f": hourly.get("temperature_2m", [None] * len(times))[i],
            "wind_mph": hourly.get("windspeed_10m", [None] * len(times))[i],
            "precip_in": hourly.get("precipitation", [None] * len(times))[i],
        }
    return result


def fetch_forecast(lat: float, lon: float) -> dict:
    """Returns {'YYYY-MM-DDTHH:00': {temp_f, wind_mph, precip_in}} for the next
    ~16 days at (lat, lon)."""
    params = {"latitude": lat, "longitude": lon, "hourly": HOURLY_FIELDS,
              "forecast_days": FORECAST_HORIZON_DAYS, **UNIT_PARAMS}
    resp = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return _hourly_by_timestamp(resp.json())


def nearest_hour(hourly: dict, kickoff_iso_utc: str):
    """kickoff_iso_utc like '2026-11-08T18:00:00+00:00' -> match against the
    hourly dict's 'YYYY-MM-DDTHH:00' keys."""
    key = kickoff_iso_utc[:13] + ":00"
    return hourly.get(key)


def polite_sleep():
    time.sleep(0.2)  # stay well under Open-Meteo's fair-use rate limits
