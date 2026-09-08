"""
Current weather via Open-Meteo - free, no API key required, unlike most
weather APIs.

On location and privacy: with no latitude/longitude configured this can fall
back to IP-based geolocation, which means sending a request to a third party
(ipapi.co) that reveals the user's IP address and gets their approximate
location back. In an app that describes itself as local-first, that shouldn't
happen silently as a side effect of asking about the weather - so it is
opt-in via `weather.allow_ip_geolocation: true`. Without it, and without
coordinates, the tool says what's missing instead of guessing.

Offline behaviour: the last successful reading is cached to disk and served when
the network is gone, marked `stale` with the time it was taken. A local-first
assistant that goes blank the moment the wi-fi drops is the wrong shape - "7 degrees,
as of 40 minutes ago" is useful, and it is honest as long as the age travels with
it. Nothing here ever presents a cached reading as current. The resolved
coordinates are cached alongside it for the same reason: IP geolocation needs the
network too, so without them an offline machine could not even say where the
reading it already has came from.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from core.atomic_write import atomic_write_text
from core.config_loader import get_settings, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.weather")

CACHE_PATH = "./data/weather_cache.json"

# Open-Meteo's numeric weather codes, condensed to plain English.
_WEATHER_CODES: Dict[int, str] = {
    0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    85: "Light snow showers", 86: "Snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}


class LocationUnavailable(RuntimeError):
    """No coordinates configured and IP geolocation not permitted."""


def _cache_path():
    return resolve_path(CACHE_PATH)


def _load_cache() -> Optional[Dict[str, Any]]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Ignoring unreadable weather cache ({e}).")
        return None
    return data if isinstance(data, dict) and data.get("reading") else None


def _save_cache(reading: Dict[str, Any], location: Dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(path, json.dumps({
            "reading": reading, "location": location, "fetched_at": time.time(),
        }, indent=2))
    except OSError as e:
        # A cache that can't be written is not a reason to fail the reading the
        # caller already has in hand.
        logger.warning(f"Couldn't write the weather cache ({e}).")


def _as_stale(cache: Dict[str, Any]) -> Dict[str, Any]:
    """The cached reading, labelled with its age. Never presented as current."""
    age = max(0.0, time.time() - float(cache.get("fetched_at", 0)))
    return {
        **cache["reading"],
        "stale": True,
        "fetched_at": cache.get("fetched_at"),
        "age_minutes": round(age / 60),
    }


async def _resolve_location() -> Dict[str, Any]:
    cfg = get_settings().get("weather", {})
    if cfg.get("latitude") is not None and cfg.get("longitude") is not None:
        return {"lat": cfg["latitude"], "lon": cfg["longitude"], "label": cfg.get("location_label", "Configured location")}

    if not cfg.get("allow_ip_geolocation"):
        raise LocationUnavailable(
            "No location is set for weather. Add weather.latitude and weather.longitude "
            "via /settings, or set weather.allow_ip_geolocation: true to let Leti look up "
            "your approximate location from your IP address (this sends a request to "
            "ipapi.co, a third party)."
        )

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get("https://ipapi.co/json/")
            resp.raise_for_status()
            data = resp.json()
            return {
                "lat": data["latitude"], "lon": data["longitude"],
                "label": f"{data.get('city', 'Unknown')}, {data.get('region_code', '')}".strip(", "),
            }
    except Exception:
        # Looking up where we are needs the network as much as the forecast does.
        # Offline, the coordinates from the last successful lookup are the best
        # answer available - and without them there is nothing to be offline WITH.
        cached = _load_cache()
        if cached and cached.get("location"):
            return cached["location"]
        raise


async def _fetch_current_weather() -> tuple:
    """(reading, location) straight from the network. Raises if it can't."""
    location = await _resolve_location()
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": location["lat"], "longitude": location["lon"],
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "temperature_unit": get_settings().get("weather", {}).get("unit", "celsius"),
        })
        resp.raise_for_status()
        data = resp.json()

    current = data.get("current", {})
    code = current.get("weather_code", 0)
    unit_symbol = "°F" if get_settings().get("weather", {}).get("unit") == "fahrenheit" else "°C"
    return {
        "location": location["label"],
        "temperature": current.get("temperature_2m"),
        "unit": unit_symbol,
        "condition": _WEATHER_CODES.get(code, "Unknown"),
        "humidity_percent": current.get("relative_humidity_2m"),
        # Open-Meteo returns wind in km/h regardless of temperature_unit, so this
        # stays km/h even when temperatures are in Fahrenheit.
        "wind_kph": current.get("wind_speed_10m"),
    }, location


async def get_current_weather() -> Dict[str, Any]:
    """The current weather, or the last one we had, labelled with its age.

    A network failure falls back to the cache rather than to nothing: this runs on
    a desktop that closes its laptop lid and moves between networks, and going
    blank on the dashboard every time that happens is worse than an hour-old
    number that says how old it is. `stale` and `age_minutes` are on the reading
    so no caller can mistake one for the other.

    LocationUnavailable is deliberately NOT caught: "you never told me where you
    are" is a thing the user can fix, and answering it with a cached reading from
    somewhere else would be worse than saying so.
    """
    try:
        reading, location = await _fetch_current_weather()
    except LocationUnavailable:
        raise
    except Exception as e:
        cached = _load_cache()
        if cached:
            logger.info(f"Weather unavailable ({type(e).__name__}); serving the cached reading.")
            return _as_stale(cached)
        raise

    _save_cache(reading, location)
    return {**reading, "stale": False, "fetched_at": time.time(), "age_minutes": 0}


class GetWeatherTool(BaseTool):
    name = "get_weather"
    description = (
        "Get the current weather. Uses IP-based location automatically unless a location "
        "is configured.\n"
        "If the network is unavailable this returns the last reading instead of failing, "
        "with `stale: true` and `age_minutes`. When it does, say how old it is rather "
        "than reporting it as the weather right now."
    )
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        try:
            return ToolResult(success=True, output=await get_current_weather())
        except LocationUnavailable as e:
            return ToolResult(success=False, error=str(e))
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"Weather API error: {e.response.status_code}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
