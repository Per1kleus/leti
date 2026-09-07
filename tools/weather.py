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
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from core.config_loader import get_settings
from tools.base import BaseTool, ToolParameter, ToolResult

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

    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get("https://ipapi.co/json/")
        resp.raise_for_status()
        data = resp.json()
        return {
            "lat": data["latitude"], "lon": data["longitude"],
            "label": f"{data.get('city', 'Unknown')}, {data.get('region_code', '')}".strip(", "),
        }


async def get_current_weather() -> Dict[str, Any]:
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
    }


class GetWeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather. Uses IP-based location automatically unless a location is configured."
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
