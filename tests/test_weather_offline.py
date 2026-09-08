"""The weather panel when the network isn't there.

A local-first assistant that blanks its dashboard the moment the wi-fi drops is
the wrong shape: the laptop lid closes, the machine moves between networks, and
"19 degrees, 40 minutes ago" is more use than an empty box. What makes that
honest rather than a lie is that the age travels with the reading, so these
tests are as much about the labelling as the fallback.
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

import tools.weather as weather


@pytest.fixture
def cache(monkeypatch, tmp_path):
    path = tmp_path / "weather_cache.json"
    monkeypatch.setattr(weather, "_cache_path", lambda: path)
    return path


READING = {
    "location": "Athens, GR", "temperature": 19.6, "unit": "°C",
    "condition": "Partly cloudy", "humidity_percent": 52, "wind_kph": 11.0,
}
LOCATION = {"lat": 37.98, "lon": 23.72, "label": "Athens, GR"}


def _online(monkeypatch, reading=READING):
    async def fetch():
        return reading, LOCATION
    monkeypatch.setattr(weather, "_fetch_current_weather", fetch)


def _offline(monkeypatch, error=None):
    async def fetch():
        raise error or httpx.ConnectError("Network is unreachable")
    monkeypatch.setattr(weather, "_fetch_current_weather", fetch)


@pytest.mark.asyncio
async def test_a_live_reading_is_marked_current_and_cached(cache, monkeypatch):
    _online(monkeypatch)
    result = await weather.get_current_weather()
    assert result["stale"] is False and result["age_minutes"] == 0
    assert result["temperature"] == 19.6
    assert cache.exists(), "a live reading has to be saved, or there is nothing to fall back to"


@pytest.mark.asyncio
async def test_the_last_reading_is_served_when_the_network_is_gone(cache, monkeypatch):
    _online(monkeypatch)
    await weather.get_current_weather()

    _offline(monkeypatch)
    result = await weather.get_current_weather()
    assert result["temperature"] == 19.6
    assert result["condition"] == "Partly cloudy"
    assert result["location"] == "Athens, GR"


@pytest.mark.asyncio
async def test_a_cached_reading_always_says_how_old_it_is(cache, monkeypatch):
    """The whole difference between a useful fallback and a wrong number."""
    _online(monkeypatch)
    await weather.get_current_weather()

    stored = json.loads(cache.read_text())
    stored["fetched_at"] -= 95 * 60
    cache.write_text(json.dumps(stored))

    _offline(monkeypatch)
    result = await weather.get_current_weather()
    assert result["stale"] is True
    assert result["age_minutes"] == 95


@pytest.mark.asyncio
async def test_with_no_cache_the_failure_is_still_a_failure(cache, monkeypatch):
    """Nothing is invented. No reading has ever been taken, so there is none."""
    _offline(monkeypatch)
    with pytest.raises(httpx.ConnectError):
        await weather.get_current_weather()


@pytest.mark.asyncio
async def test_an_unset_location_is_not_answered_from_the_cache(cache, monkeypatch):
    """'You never told me where you are' is a thing the user can fix, and a
    reading from wherever they used to be is not the answer to it."""
    _online(monkeypatch)
    await weather.get_current_weather()

    _offline(monkeypatch, error=weather.LocationUnavailable("No location is set for weather."))
    with pytest.raises(weather.LocationUnavailable):
        await weather.get_current_weather()


@pytest.mark.asyncio
async def test_a_fresh_reading_replaces_a_stale_one(cache, monkeypatch):
    """Coming back online has to actually come back online."""
    _online(monkeypatch)
    await weather.get_current_weather()
    _offline(monkeypatch)
    assert (await weather.get_current_weather())["stale"] is True

    _online(monkeypatch, {**READING, "temperature": 24.1, "condition": "Clear"})
    fresh = await weather.get_current_weather()
    assert fresh["stale"] is False and fresh["temperature"] == 24.1


@pytest.mark.asyncio
async def test_an_unreadable_cache_is_ignored_rather_than_crashing(cache, monkeypatch):
    cache.write_text("{ not json")
    _offline(monkeypatch)
    with pytest.raises(httpx.ConnectError):
        await weather.get_current_weather()


@pytest.mark.asyncio
async def test_the_cached_coordinates_survive_losing_ip_geolocation(cache, monkeypatch):
    """Looking up where you are needs the network as much as the forecast does.
    Without the saved coordinates an offline machine couldn't even say where the
    reading it already has came from."""
    _online(monkeypatch)
    await weather.get_current_weather()
    assert json.loads(cache.read_text())["location"] == LOCATION

    monkeypatch.setattr(weather, "get_settings",
                        lambda: {"weather": {"allow_ip_geolocation": True}})

    async def dead_geoip(*a, **k):
        raise httpx.ConnectError("Network is unreachable")

    monkeypatch.setattr(httpx.AsyncClient, "get", dead_geoip)
    assert await weather._resolve_location() == LOCATION


def test_the_tool_tells_the_model_a_reading_can_be_stale():
    """Otherwise Leti reports an hour-old temperature as the weather right now."""
    description = weather.GetWeatherTool().description
    assert "stale" in description and "age_minutes" in description
