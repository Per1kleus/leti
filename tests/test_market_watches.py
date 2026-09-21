"""Advanced Watch & Act: the market, the news, and how fast attention is moving.

The old watches were about this machine and could be checked by asking the
operating system. These three are about the world, which means somebody else's
API, somebody else's reporting, and a great deal more ways to be confidently
wrong. So most of what is asserted here is restraint:

  the same event is reported once, however many outlets carry it and however many
  times the watch is evaluated;

  a provider that fails, rate-limits or returns nothing is never read as "it
  didn't happen";

  ten watches on four symbols are four symbols' worth of requests, not ten;

  the model is not called to decide whether a condition is true - that is
  arithmetic, and it happens in Python;

  and nothing here predicts anything. A trend watch says what attention is doing
  and how sure it is, or it says it has not seen enough yet.

The provider is faked at the HTTP boundary rather than at the module boundary, so
what is under test is the real snapshot parsing, the real cache, the real batching
and the real error mapping.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

import httpx  # noqa: E402

from core import signals, task_manager, watches  # noqa: E402
from tools import scheduler, trading_platform  # noqa: E402


# --- A market that answers ----------------------------------------------------------

def bars(count=25, close=91.0, volume=2_000_000):
    return [{"o": close - 1, "h": close + 1, "l": close - 2, "c": close,
             "v": volume, "t": f"2026-09-{i + 1:02d}"} for i in range(count)]


def snapshot(price=100.0, previous_close=92.0, open_=94.0, volume=9_000_000):
    return {"latestTrade": {"p": price, "t": "2026-09-13T14:00:00Z"},
            "dailyBar": {"o": open_, "h": price + 1, "l": open_ - 1, "c": price,
                         "v": volume, "t": "2026-09-13"},
            "prevDailyBar": {"o": previous_close - 2, "h": previous_close + 2,
                             "l": previous_close - 3, "c": previous_close,
                             "v": 3_000_000, "t": "2026-09-12"}}


class FakeMarket:
    """Alpaca, as far as anything under test can tell. Counts its own requests."""

    def __init__(self):
        self.stocks = {}
        self.crypto = {}
        self.bars = bars()
        self.requests = []
        self.fail_with = None

    async def get(self, base, path, params=None):
        self.requests.append((path, dict(params or {})))
        if self.fail_with is not None:
            raise self.fail_with
        if "crypto/us/snapshots" in path:
            wanted = (params or {}).get("symbols", "").split(",")
            return {"snapshots": {s: self.crypto[s] for s in wanted if s in self.crypto}}
        if "stocks/snapshots" in path:
            wanted = (params or {}).get("symbols", "").split(",")
            return {s: self.stocks[s] for s in wanted if s in self.stocks}
        if "crypto/us/bars" in path:
            return {"bars": {(params or {}).get("symbols", ""): self.bars}}
        if "bars" in path:
            return {"bars": self.bars}
        return {}

    @property
    def symbols_asked_for(self):
        asked = []
        for path, params in self.requests:
            if "snapshots" in path:
                asked.extend(params.get("symbols", "").split(","))
        return asked


@pytest.fixture
def market(monkeypatch):
    fake = FakeMarket()
    fake.stocks["TTWO"] = snapshot()
    fake.stocks["AAPL"] = snapshot(price=200.0, previous_close=199.0, open_=199.5)
    fake.crypto["BTC/USD"] = snapshot(price=60_000.0, previous_close=65_000.0, open_=64_800.0,
                                      volume=1_200.0)
    monkeypatch.setattr(trading_platform, "_get", fake.get)
    monkeypatch.setattr(trading_platform, "market_settings",
                        lambda: {"api_key": "key", "api_secret": "secret"})
    trading_platform.clear_market_cache()
    yield fake
    trading_platform.clear_market_cache()


@pytest.fixture
def searches(monkeypatch):
    """The web search tool, answering with whatever a test puts in `articles`."""
    box = {"articles": [], "calls": 0, "fail": None}

    async def fake_run_searches(queries, max_results=5):
        box["calls"] += 1
        if box["fail"]:
            raise box["fail"]
        return {"results": list(box["articles"]), "failed_queries": []}

    import tools.web_search as web_search

    monkeypatch.setattr(web_search, "run_searches", fake_run_searches)
    return box


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(watches, "store_path", lambda: tmp_path / "watches.json")
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "scheduled.json")
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")


def article(title, domain, url=None):
    return {"title": title, "url": url or f"https://{domain}/story", "domain": domain,
            "snippet": title, "found_by": ["q"]}


def market_watch(**condition):
    base = {"symbol": "TTWO", "metric": "change_percent", "comparison": "abs_above",
            "threshold": 5.0, "window": 20}
    base.update(condition)
    return watches.save_new(watches.create("m", "market", base, cooldown_minutes=0))


# --- Market: the measurements -------------------------------------------------------

def test_a_stock_watch_fires_on_the_move_it_was_given(market):
    watch = market_watch()                       # 92 -> 100 is +8.7%
    assert watches.check(watch["id"])["outcome"] == "triggered"

    fired = watches.get_watch(watch["id"])
    why = fired["history"][-1]
    assert "8.696" in json.dumps(why) or "8.7" in json.dumps(why)
    assert why["evidence"]["feed"] == "iex", "the feed the figure came from is recorded"


def test_a_crypto_watch_reads_the_crypto_endpoint(market):
    watch = market_watch(symbol="BTC/USD", threshold=5.0)   # 65000 -> 60000 is -7.7%
    assert watches.check(watch["id"])["outcome"] == "triggered"
    assert any("crypto" in path for path, _ in market.requests)
    assert watches.get_watch(watch["id"])["condition_state"]["evidence"]["feed"] == "crypto"


def test_a_move_the_other_way_still_counts_when_the_question_was_how_much(market):
    """'Tell me if it moves more than 5%' is not 'tell me if it goes up 5%'. Getting
    this wrong means silence through a crash."""
    watch = market_watch(symbol="BTC/USD")
    value = watches.get_watch(watch["id"])
    assert watches.check(value["id"])["outcome"] == "triggered"
    assert watches.get_watch(watch["id"])["condition_state"]["last_value"] < 0


def test_a_smaller_move_than_asked_for_is_not_an_event(market):
    watch = market_watch(symbol="AAPL")          # 199 -> 200 is +0.5%
    assert watches.check(watch["id"])["outcome"] == "false"


def test_a_volume_spike_is_todays_volume_against_its_own_average(market):
    watch = market_watch(metric="volume_ratio", comparison="above", threshold=3.0)
    assert watches.check(watch["id"])["outcome"] == "triggered"
    evidence = watches.get_watch(watch["id"])["condition_state"]["evidence"]
    assert evidence["today_volume"] == 9_000_000
    assert evidence["average_volume"] == 2_000_000
    assert evidence["value"] == 4.5


def test_historical_metrics_are_measured_against_real_bars(market):
    market.bars = bars(count=25, close=80.0)
    value, evidence = signals.market_value("TTWO", "sma_gap_percent", window=20)
    assert evidence["moving_average"] == 80.0
    assert round(value, 2) == 25.0                # 100 against an 80 average
    assert round(signals.market_value("TTWO", "momentum_percent", 20)[0], 2) == 25.0


def test_a_breakout_is_measured_against_the_windows_own_high(market):
    market.bars = bars(count=25, close=95.0)      # highs of 96, price is 100
    value, evidence = signals.market_value("TTWO", "high_breakout_percent", window=20)
    assert evidence["window_high"] == 96.0
    assert round(value, 3) == round((100 - 96) / 96 * 100, 3)


def test_volatility_is_a_number_not_an_adjective(market):
    market.bars = [{"o": 90, "h": 92, "l": 88, "c": 90 + (5 if i % 2 else -5),
                    "v": 1_000, "t": f"d{i}"} for i in range(21)]
    value, evidence = signals.market_value("TTWO", "volatility_percent", window=20)
    assert value > 5 and evidence["daily_returns_used"] >= 19


# --- Market: when the provider does not cooperate -----------------------------------

@pytest.mark.parametrize("status,kind,fragment", [
    (401, "auth", "not authorised"),
    (403, "subscription", "subscription"),
    (429, "rate_limit", "rate-limited"),
    (404, "unknown_symbol", "does not know"),
    (503, "outage", "trouble"),
])
def test_each_provider_failure_is_named_rather_than_guessed(market, status, kind, fragment):
    body = "subscription required" if status == 403 else "nope"
    market.fail_with = httpx.HTTPStatusError(
        "x", request=httpx.Request("GET", "https://data.alpaca.markets/x"),
        response=httpx.Response(status, text=body))
    with pytest.raises(trading_platform.MarketDataError) as raised:
        asyncio.run(trading_platform.get_snapshots(["TTWO"]))
    assert raised.value.kind == kind
    assert fragment in str(raised.value).lower()


def test_a_provider_failure_is_never_read_as_the_condition_being_false(market):
    watch = market_watch()
    market.fail_with = httpx.HTTPStatusError(
        "x", request=httpx.Request("GET", "https://data.alpaca.markets/x"),
        response=httpx.Response(429, text="slow down"))

    outcome = watches.check(watch["id"])
    assert outcome["outcome"] == "error"
    stored = watches.get_watch(watch["id"])
    assert stored["condition_was_true"] is False
    assert stored["enabled"] is True, "one rate limit is not a reason to stop watching"
    assert "rate-limited" in stored["last_error"]
    assert stored["history"][-1]["problem"]


def test_an_unknown_symbol_says_so_instead_of_reporting_no_movement(market):
    watch = market_watch(symbol="NOTREAL")
    outcome = watches.check(watch["id"])
    assert outcome["outcome"] == "error"
    assert "NOTREAL" in outcome["error"]


def test_no_credentials_is_a_stated_problem_not_a_silent_watch(market, monkeypatch):
    monkeypatch.setattr(trading_platform, "market_settings", lambda: {})
    watch = market_watch()
    outcome = watches.check(watch["id"])
    assert outcome["outcome"] == "error"
    assert "no alpaca api key" in outcome["error"].lower()
    assert "Connections" in outcome["error"]


def test_a_symbol_with_almost_no_history_is_refused_not_extrapolated(market):
    market.bars = bars(count=2)
    with pytest.raises(signals.SignalError) as raised:
        signals.market_value("TTWO", "volatility_percent", window=20)
    assert raised.value.kind == "no_data"


def test_a_malformed_provider_response_does_not_become_a_price(market):
    market.stocks["TTWO"] = {"latestTrade": {"p": "not a number"}, "dailyBar": "nonsense"}
    with pytest.raises(signals.SignalError):
        signals.market_value("TTWO", "change_percent")


# --- Market: one connection, shared ------------------------------------------------

def test_many_watches_on_a_few_symbols_are_a_few_requests(market):
    for symbol in ("TTWO", "AAPL", "TTWO", "BTC/USD", "AAPL", "TTWO"):
        market_watch(symbol=symbol, threshold=99.0)
    pending = [w for w in watches.load_watches() if watches.due(w)]
    assert len(pending) == 6

    watches.prefetch(pending)
    for watch in pending:
        watches.check(watch["id"])

    snapshot_calls = [p for p, _ in market.requests if "snapshots" in p]
    assert len(snapshot_calls) == 2, f"one per asset class, got {market.requests}"
    assert sorted(market.symbols_asked_for) == ["AAPL", "BTC/USD", "TTWO"]


def test_the_market_cache_does_not_grow_without_limit(market):
    for n in range(trading_platform.MAX_CACHED_SYMBOLS + 40):
        trading_platform._SNAPSHOTS[f"S{n}"] = (time.time(), {"symbol": f"S{n}"})
        trading_platform._remember({f"S{n}": {"symbol": f"S{n}"}}, time.time())
    assert len(trading_platform._SNAPSHOTS) <= trading_platform.MAX_CACHED_SYMBOLS


def test_a_stale_cache_entry_is_refetched_rather_than_served(market, monkeypatch):
    asyncio.run(trading_platform.get_snapshots(["TTWO"]))
    first = len(market.requests)
    trading_platform._SNAPSHOTS["TTWO"] = (
        time.time() - trading_platform.SNAPSHOT_CACHE_SECONDS - 1,
        trading_platform._SNAPSHOTS["TTWO"][1])
    asyncio.run(trading_platform.get_snapshots(["TTWO"]))
    assert len(market.requests) == first + 1


def test_nothing_holds_a_socket_open(market):
    """A watch is evaluated and the process is idle again. No stream, no reconnect
    loop, no background task per feed - which is the whole reason watches are
    scheduler-driven rather than event-driven. (All three modules mention the
    word in comments explaining why they don't; what matters is the calls.)"""
    import ast

    import core.signals as signals_module

    for module in (trading_platform, signals_module, watches):
        tree = ast.parse(open(module.__file__).read())
        imported = {n.name.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for n in node.names}
        imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom)}
        assert "websockets" not in imported, f"{module.__name__} imports a stream client"
        called = {node.func.attr if isinstance(node.func, ast.Attribute)
                  else getattr(node.func, "id", "")
                  for node in ast.walk(tree) if isinstance(node, ast.Call)}
        for forbidden in ("ws_connect", "connect", "Thread", "create_task", "sleep"):
            assert forbidden not in called, f"{module.__name__} calls {forbidden}()"
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)], \
            f"{module.__name__} has a loop of its own"


# --- News: one event, reported once -------------------------------------------------

def news_watch(**condition):
    base = {"topic": "GTA VI", "min_sources": 1, "require_confirmation": False}
    base.update(condition)
    return watches.save_new(watches.create("n", "news_event", base, cooldown_minutes=0))


def test_a_new_event_is_an_event(searches):
    searches["articles"] = [article("Rockstar announces GTA VI release date", "reuters.com")]
    watch = news_watch()
    assert watches.check(watch["id"])["outcome"] == "triggered"
    evidence = watches.get_watch(watch["id"])["history"][-1]["evidence"]
    assert evidence["new_events"][0]["sources"] == ["reuters.com"]


def test_the_same_event_from_ten_outlets_is_one_event(searches):
    searches["articles"] = [
        article("Rockstar announces GTA VI release date", "reuters.com"),
        article("GTA VI release date announced by Rockstar", "bbc.co.uk"),
        article("Rockstar announces the GTA VI release date at last", "theverge.com"),
    ]
    watch = news_watch()
    watches.check(watch["id"])
    events = watches.get_watch(watch["id"])["history"][-1]["evidence"]["new_events"]
    assert len(events) == 1, [e["headline"] for e in events]
    assert events[0]["independent_sources"] == 3


def test_the_same_story_tomorrow_is_not_news_again(searches):
    searches["articles"] = [article("Rockstar announces GTA VI release date", "reuters.com")]
    watch = news_watch()
    assert watches.check(watch["id"])["outcome"] == "triggered"

    # The condition goes false (nothing new), then the same story is found again.
    searches["articles"] = []
    assert watches.check(watch["id"])["outcome"] == "cleared"
    searches["articles"] = [
        article("Rockstar announces GTA VI release date", "reuters.com"),
        article("Rockstar announces GTA VI release date", "ign.com"),
    ]
    assert watches.check(watch["id"])["outcome"] == "false", "it re-reported a known event"


def test_a_genuinely_new_development_still_gets_through(searches):
    searches["articles"] = [article("Rockstar announces GTA VI release date", "reuters.com")]
    watch = news_watch()
    watches.check(watch["id"])
    searches["articles"] = [
        article("Rockstar announces GTA VI release date", "reuters.com"),
        article("GTA VI delayed to November, Take-Two tells investors", "bloomberg.com"),
    ]
    assert watches.check(watch["id"])["outcome"] == "triggered"
    headline = watches.get_watch(watch["id"])["history"][-1]["evidence"]["new_events"][0]
    assert "delayed" in headline["headline"].lower()


def test_a_condition_that_stays_true_is_not_reported_every_time(searches):
    searches["articles"] = [article("Suspect arrested over the Rockstar leak", "bbc.co.uk"),
                            article("Police arrest suspect in Rockstar leak", "reuters.com")]
    watch = news_watch(topic="Rockstar leak")
    assert watches.check(watch["id"])["outcome"] == "triggered"
    assert watches.check(watch["id"])["outcome"] != "triggered"


def test_a_cooldown_holds_a_second_event_back(searches):
    watch = watches.save_new(watches.create(
        "n", "news_event", {"topic": "GTA VI"}, cooldown_minutes=60))
    searches["articles"] = [article("GTA VI trailer two arrives", "reuters.com")]
    assert watches.check(watch["id"])["outcome"] == "triggered"
    searches["articles"] = []
    watches.check(watch["id"])
    searches["articles"] = [article("Take-Two confirms GTA VI marketing push", "bloomberg.com")]
    assert watches.check(watch["id"])["outcome"] == "cooling_down"


def test_a_search_that_fails_is_not_an_absence_of_news(searches):
    searches["fail"] = RuntimeError("the search engine timed out")
    watch = news_watch()
    outcome = watches.check(watch["id"])
    assert outcome["outcome"] == "error"
    assert watches.get_watch(watch["id"])["condition_was_true"] is False


# --- News: how much a source is worth ------------------------------------------------

@pytest.mark.parametrize("domain,expected", [
    ("reuters.com", "major"), ("www.bbc.co.uk", "major"),
    ("justice.gov", "regulator"), ("europol.europa.eu", "regulator"),
    ("x.com", "social"), ("reddit.com", "social"),
    ("some-blog.example", "secondary"),
])
def test_sources_are_sorted_by_what_they_are(domain, expected):
    assert signals.source_tier(domain) == expected


def test_a_domain_the_user_named_as_official_outranks_the_list():
    assert signals.source_tier("rockstargames.com") == "secondary"
    assert signals.source_tier("rockstargames.com", ["rockstargames.com"]) == "official"


def test_one_social_post_is_never_confirmation(searches):
    searches["articles"] = [article("The hacker was caught, apparently", "x.com")]
    watch = news_watch(topic="the leak investigation", require_confirmation=True)
    assert watches.check(watch["id"])["outcome"] == "false"


def test_two_independent_major_outlets_are(searches):
    searches["articles"] = [
        article("Police charge suspect over the Rockstar leak", "reuters.com"),
        article("Suspect charged in Rockstar leak case", "bbc.co.uk"),
    ]
    watch = news_watch(topic="the leak investigation", require_confirmation=True)
    assert watches.check(watch["id"])["outcome"] == "triggered"
    event = watches.get_watch(watch["id"])["history"][-1]["evidence"]["new_events"][0]
    assert event["support"] == "confirmed"
    assert "independent" in event["why"]


def test_an_event_below_the_source_bar_is_counted_but_not_reported(searches):
    searches["articles"] = [article("Rumour: GTA VI delayed again", "some-blog.example")]
    watch = news_watch(min_sources=3)
    assert watches.check(watch["id"])["outcome"] == "false"
    state = watches.get_watch(watch["id"])["condition_state"]
    assert state["evidence"]["below_the_bar"] == 1


def test_the_notification_carries_the_sources_behind_it(searches):
    searches["articles"] = [article("Take-Two announces a GTA VI delay", "bloomberg.com"),
                            article("GTA VI delayed, Take-Two says", "reuters.com")]
    watch = news_watch()
    watches.check(watch["id"])
    event = watches.get_watch(watch["id"])["history"][-1]["evidence"]["new_events"][0]
    assert set(event["sources"]) == {"bloomberg.com", "reuters.com"}
    assert event["articles"][0]["url"].startswith("https://")


def test_what_has_been_seen_is_remembered_but_bounded(searches):
    watch = news_watch()
    for n in range(signals.MAX_SEEN_EVENTS + 30):
        searches["articles"] = [article(f"Distinct headline number {n} about it", "reuters.com")]
        watches.check(watch["id"])
    seen = watches.get_watch(watch["id"])["condition_state"]["seen"]
    assert len(seen) <= signals.MAX_SEEN_EVENTS


# --- Trend: an estimate that says it is one ------------------------------------------

def trend_watch(**condition):
    base = {"topic": "GTA VI", "acceleration": 1.6, "min_activity": 1.0}
    base.update(condition)
    return watches.save_new(watches.create("t", "trend", base, cooldown_minutes=0))


SUBJECTS = ["release date", "a leaked map", "casting rumours", "the soundtrack",
            "preorder pages", "an investor call", "a delay", "the trailer",
            "review embargoes", "collector editions", "mod support", "server tests",
            "a livestream", "box art", "a patch", "voice actors"]


def observe(watch_id, searches, count, wave=0):
    """`count` genuinely different stories - the grouping would collapse one story
    told sixteen ways into a single event, which is what it is for."""
    searches["articles"] = [
        article(f"{SUBJECTS[n % len(SUBJECTS)]} {wave}{n} stirs interest",
                f"outlet{n}.example")
        for n in range(count)]
    return watches.check(watch_id)


def test_a_trend_says_nothing_until_it_has_seen_enough(searches):
    watch = trend_watch()
    for _ in range(signals.MIN_TREND_SAMPLES - 1):
        assert observe(watch["id"], searches, 9)["outcome"] == "false"
    evidence = watches.get_watch(watch["id"])["condition_state"]["evidence"]
    assert evidence["confidence"] == "none"
    assert "observation" in evidence["not_yet"]
    assert evidence["ratio"] is None, "it produced a rate from one period"


def test_accelerating_attention_is_detected(searches):
    watch = trend_watch()
    for _ in range(3):
        observe(watch["id"], searches, 1)
    outcome = None
    for _ in range(3):
        outcome = observe(watch["id"], searches, 12)
    evidence = watches.get_watch(watch["id"])["condition_state"]["evidence"]
    assert evidence["phase"].startswith("accelerating"), evidence
    assert evidence["ratio"] > 1.6
    assert watches.get_watch(watch["id"])["condition_was_true"] is True


def test_declining_attention_is_not_an_event(searches):
    watch = trend_watch()
    for _ in range(3):
        observe(watch["id"], searches, 12)
    for _ in range(3):
        observe(watch["id"], searches, 1)
    evidence = watches.get_watch(watch["id"])["condition_state"]["evidence"]
    assert evidence["phase"] == "cooling"
    assert watches.get_watch(watch["id"])["condition_was_true"] is False


def test_a_trend_never_claims_to_know_what_happens_next(searches):
    watch = trend_watch()
    for _ in range(6):
        observe(watch["id"], searches, 8)
    evidence = watches.get_watch(watch["id"])["condition_state"]["evidence"]
    text = json.dumps(evidence).lower()
    assert "estimate" in text
    assert evidence["confidence"] in ("low", "moderate", "moderate-to-high")
    assert evidence["samples"] >= signals.MIN_TREND_SAMPLES
    for forbidden in ("will peak", "prediction", "guaranteed", "certain"):
        assert forbidden not in text.replace("not a forecast", "")


def test_confidence_rises_with_evidence_and_never_beyond_it(searches):
    assert signals._confidence(2, 9, True) == "none"
    assert signals._confidence(6, 1, True) == "low"
    assert signals._confidence(9, 3, True) == "moderate"
    assert signals._confidence(20, 9, True) == "moderate-to-high"


def test_a_failed_search_leaves_a_gap_rather_than_a_zero(searches, market):
    watch = trend_watch(symbol="TTWO")
    observe(watch["id"], searches, 5)
    searches["fail"] = RuntimeError("search unavailable")
    watches.check(watch["id"])
    series = watches.get_watch(watch["id"])["condition_state"]["series"]
    assert "events" not in series[-1], "a failed search was recorded as no interest"
    assert series[-1]["volume_ratio"] > 0, "the signal that did work was still recorded"
    assert watches.get_watch(watch["id"])["condition_state"]["evidence"]["gaps"]


def test_a_trend_with_no_signal_at_all_is_an_error(searches, market):
    watch = trend_watch(symbol="TTWO")
    searches["fail"] = RuntimeError("search unavailable")
    market.fail_with = httpx.HTTPStatusError(
        "x", request=httpx.Request("GET", "https://x"), response=httpx.Response(503))
    assert watches.check(watch["id"])["outcome"] == "error"


# --- Several signals, one watch ------------------------------------------------------

def combined_watch(mode="all"):
    return watches.save_new(watches.create("c", "combined", {
        "mode": mode,
        "parts": [
            {"condition_type": "market",
             "condition": {"symbol": "TTWO", "metric": "change_percent",
                           "comparison": "abs_above", "threshold": 5.0}},
            {"condition_type": "news_event", "condition": {"topic": "GTA VI"}},
        ]}, cooldown_minutes=0))


def test_all_means_all(market, searches):
    watch = combined_watch("all")
    searches["articles"] = []
    assert watches.check(watch["id"])["outcome"] == "false", "it fired on the market alone"

    searches["articles"] = [article("Rockstar announces GTA VI release date", "reuters.com")]
    assert watches.check(watch["id"])["outcome"] == "triggered"
    signals_seen = watches.get_watch(watch["id"])["history"][-1]["evidence"]["signals"]
    assert [s["signal"] for s in signals_seen] == ["market", "news_event"]
    assert all(s["met"] for s in signals_seen)


def test_any_means_any(market, searches):
    watch = combined_watch("any")
    searches["articles"] = []
    assert watches.check(watch["id"])["outcome"] == "triggered"


def test_each_part_keeps_its_own_memory(market, searches):
    watch = combined_watch("any")
    searches["articles"] = [article("Rockstar announces GTA VI release date", "reuters.com")]
    watches.check(watch["id"])
    state = watches.get_watch(watch["id"])["condition_state"]
    assert state["parts"][1]["seen"], "the news part forgot what it had seen"


def test_a_part_that_cannot_be_measured_stops_an_all_watch(market, searches):
    watch = combined_watch("all")
    searches["fail"] = RuntimeError("search unavailable")
    outcome = watches.check(watch["id"])
    assert outcome["outcome"] == "error", "an unmeasurable signal was counted as false"


def test_a_combined_watch_is_bounded(market, searches):
    problems = watches.validate(watches.create("c", "combined", {
        "mode": "all",
        "parts": [{"condition_type": "market",
                   "condition": {"symbol": "X", "metric": "price",
                                 "comparison": "above", "threshold": 1}}] * 9}))
    assert any("at most" in p for p in problems)


def test_a_combined_watch_cannot_contain_another_one():
    problems = watches.validate(watches.create("c", "combined", {
        "mode": "all",
        "parts": [{"condition_type": "combined", "condition": {}},
                  {"condition_type": "combined", "condition": {}}]}))
    assert any("cannot be one of" in p for p in problems)


# --- The watches that existed before this ---------------------------------------------

def test_a_price_watch_saved_before_any_of_this_still_works(market):
    """price_below is now a market condition with metric=price, evaluated by the
    same arithmetic as everything else. A watch saved last month does not know
    that and should not have to."""
    watch = watches.save_new(watches.create(
        "old one", "price_below", {"symbol": "TTWO", "price": 150.0}, cooldown_minutes=0))
    assert watches.check(watch["id"])["outcome"] == "triggered"        # last price is 100
    assert watches.get_watch(watch["id"])["condition_state"]["last_value"] == 100.0

    above = watches.save_new(watches.create(
        "old two", "price_above", {"symbol": "TTWO", "price": 150.0}, cooldown_minutes=0))
    assert watches.check(above["id"])["outcome"] == "false"


def test_a_price_watch_is_evaluated_where_it_is_actually_called_from(market):
    """It used to call asyncio.run() from inside the async tool that evaluates it,
    which raises every single time - so price watches never once worked when the
    scheduler ran them. They are evaluated from a worker loop now."""
    from tools.watch_tools import CheckWatchesTool

    watches.save_new(watches.create(
        "old one", "price_below", {"symbol": "TTWO", "price": 150.0}, cooldown_minutes=0))
    result = asyncio.run(CheckWatchesTool().run())
    assert result.output["errors"] == []
    assert [t["name"] for t in result.output["triggered"]] == ["old one"]


# --- The watch, as a thing the user manages ------------------------------------------

@pytest.mark.asyncio
async def test_a_sentence_becomes_a_structured_watch(market):
    """'Watch TTWO and tell me if it moves more than 5%' is one tool call with four
    values in it - not a standing instruction the model re-reads every time."""
    from tools.watch_tools import CreateWatchTool

    result = await CreateWatchTool().run(
        name="TTWO 5%", condition_type="market", symbol="ttwo", threshold=5)
    assert result.success
    stored = watches.get_watch(result.output["watch"]["id"])
    assert stored["condition"] == {"symbol": "TTWO", "metric": "change_percent",
                                   "comparison": "abs_above", "threshold": 5.0,
                                   "window": signals.DEFAULT_WINDOW_DAYS}
    assert stored["action"] == "notify"


@pytest.mark.asyncio
async def test_a_watch_with_no_subject_is_a_question_not_a_guess():
    from tools.watch_tools import CreateWatchTool

    result = await CreateWatchTool().run(name="that stock", condition_type="market")
    assert result.success is False
    assert "which symbol" in result.error.lower()
    assert watches.load_watches() == []

    topic = await CreateWatchTool().run(name="that story", condition_type="news_event")
    assert topic.success is False and "what topic" in topic.error.lower()


@pytest.mark.asyncio
async def test_two_signals_can_be_one_watch(market, searches):
    from tools.watch_tools import CreateWatchTool

    result = await CreateWatchTool().run(
        name="TTWO and GTA VI", condition_type="market", symbol="TTWO", threshold=5,
        topic="GTA VI", also_watch="news", combine="all")
    assert result.success
    stored = watches.get_watch(result.output["watch"]["id"])
    assert stored["condition_type"] == "combined"
    assert [p["condition_type"] for p in stored["condition"]["parts"]] == ["market", "news_event"]


@pytest.mark.asyncio
async def test_the_advanced_kinds_do_not_check_every_five_minutes(market):
    from tools.watch_tools import CreateWatchTool

    news = await CreateWatchTool().run(name="n", condition_type="news_event", topic="GTA VI")
    assert news.output["watch"]["every"] == "30.0 min"


@pytest.mark.asyncio
async def test_a_watch_without_market_credentials_says_so_at_once(market, monkeypatch):
    """Better than five failed evaluations and a disabled watch an hour later."""
    from tools.watch_tools import CreateWatchTool

    monkeypatch.setattr(trading_platform, "market_settings", lambda: {})
    result = await CreateWatchTool().run(name="w", condition_type="market",
                                         symbol="TTWO", threshold=5)
    assert result.success
    assert "no Alpaca API key" in result.output["warning"]
    assert "Connections" in result.output["warning"]


@pytest.mark.asyncio
async def test_the_feed_behind_a_figure_is_never_implied_to_be_the_whole_market(market):
    from tools.watch_tools import CreateWatchTool

    result = await CreateWatchTool().run(name="w", condition_type="market",
                                         symbol="TTWO", threshold=5)
    assert "IEX" in result.output["warning"] and "whole US market" in result.output["warning"]
    assert trading_platform.feed_report()["stock_feed"] == "iex"


@pytest.mark.asyncio
async def test_a_watch_can_be_changed_without_losing_what_it_has_seen(searches):
    from tools.watch_tools import ManageWatchTool

    watch = news_watch()
    searches["articles"] = [article("Rockstar announces GTA VI release date", "reuters.com")]
    watches.check(watch["id"])
    seen_before = watches.get_watch(watch["id"])["condition_state"]["seen"]
    assert seen_before

    result = await ManageWatchTool().run(watch["id"], "update", interval_minutes=120,
                                         cooldown_minutes=90, name="GTA VI, slower")
    assert result.success
    after = watches.get_watch(watch["id"])
    assert (after["interval_minutes"], after["cooldown_minutes"]) == (120.0, 90.0)
    assert after["name"] == "GTA VI, slower"
    assert after["condition_state"]["seen"] == seen_before, "it forgot what it had reported"


@pytest.mark.asyncio
async def test_a_change_that_would_break_a_watch_is_refused(market):
    from tools.watch_tools import ManageWatchTool

    watch = market_watch()
    result = await ManageWatchTool().run(watch["id"], "update", interval_minutes="soon")
    assert result.success is False
    assert watches.get_watch(watch["id"])["interval_minutes"] == 5.0, "it half-applied"


@pytest.mark.asyncio
async def test_pause_resume_and_remove(market):
    from tools.watch_tools import ManageWatchTool

    watch = market_watch()
    assert (await ManageWatchTool().run(watch["id"], "disable")).success
    assert watches.get_watch(watch["id"])["enabled"] is False
    assert watches.due(watches.get_watch(watch["id"])) is False

    assert (await ManageWatchTool().run(watch["id"], "enable")).success
    assert watches.get_watch(watch["id"])["enabled"] is True

    assert (await ManageWatchTool().run(watch["id"], "delete")).success
    assert watches.get_watch(watch["id"]) is None


@pytest.mark.asyncio
async def test_an_expired_watch_stops_and_says_why(market):
    from tools.watch_tools import CheckWatchesTool

    watch = market_watch()
    watch["expires_at"] = time.time() - 60
    watches._replace(watch)

    result = await CheckWatchesTool().run()
    assert result.output["expired"] == ["m"]
    stopped = watches.get_watch(watch["id"])
    assert stopped["enabled"] is False
    assert "Expired" in stopped["disabled_reason"]


@pytest.mark.asyncio
async def test_a_watch_can_be_inspected_on_its_own(market):
    from tools.watch_tools import ListWatchesTool

    watch = market_watch()
    watch["reason"] = "because the GTA VI announcement is coming"
    watches._replace(watch)
    watches.check(watch["id"])

    detail = (await ListWatchesTool().run(watch_id=watch["id"])).output["watch"]
    assert detail["why_it_exists"] == "because the GTA VI announcement is coming"
    assert detail["type"] == "market"
    assert detail["current_signals"]["feed"] == "iex"
    assert detail["history"][-1]["why"]
    assert detail["source"] == "Alpaca market data (iex feed)"
    assert detail["next_evaluation"]

    missing = await ListWatchesTool().run(watch_id="nope")
    assert missing.success is False


@pytest.mark.asyncio
async def test_several_watches_are_one_scheduled_row_however_many_there_are(market, searches):
    from tools.watch_tools import CreateWatchTool

    for n in range(6):
        await CreateWatchTool().run(name=f"w{n}", condition_type="market",
                                    symbol="TTWO", threshold=n + 1)
    rows = [t for t in scheduler.load_tasks() if t["name"] == watches.SCHEDULER_TASK_NAME]
    assert len(rows) == 1
    assert len(watches.load_watches()) == 6


@pytest.mark.asyncio
async def test_check_watches_reports_the_evidence_not_just_the_fact(market):
    from tools.watch_tools import CheckWatchesTool

    market_watch()
    result = await CheckWatchesTool().run()
    reported = result.output["triggered"][0]
    assert reported["why"] and "8.6" in reported["why"]
    assert reported["source"] == "Alpaca market data (iex feed)"
    assert reported["evidence"]["feed"] == "iex"
    assert "feed's, not the whole market's" in result.output["how_to_report"]


# --- Detection is not permission -----------------------------------------------------

@pytest.mark.asyncio
async def test_a_watch_that_wants_something_done_starts_a_task_and_nothing_else(market, monkeypatch):
    """'If it drops 10%, sell it' - the watch may notice the drop. Selling is a task,
    run by the orchestrator, authorised call by call."""
    from tools import watch_tools

    started = []
    monkeypatch.setattr(watch_tools, "_RUNNER", None)
    watches.save_new(watches.create(
        "sell on a drop", "market",
        {"symbol": "TTWO", "metric": "change_percent", "comparison": "abs_above",
         "threshold": 5.0},
        action="start_task", action_target="sell my TTWO position", cooldown_minutes=0))

    real_create = task_manager.create_task

    def watched_create(**kwargs):
        started.append(kwargs)
        return real_create(**kwargs)

    monkeypatch.setattr(task_manager, "create_task", watched_create)
    result = await watch_tools.CheckWatchesTool().run()

    assert len(started) == 1
    assert started[0]["objective"] == "sell my TTWO position"
    assert result.output["tasks_started"][0]["status"] == "queued", "it ran something itself"


def test_nothing_in_the_watch_path_authorises_anything():
    """Not one file added or changed for this may decide an action is allowed."""
    import ast

    for module in ("core/watches.py", "core/signals.py", "tools/watch_tools.py",
                   "tools/trading_platform.py"):
        tree = ast.parse(open(module).read())
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert "authorize" not in called, f"{module} calls authorize()"
        assert "set_unattended" not in called, f"{module} changes unattended mode"
        assert "execute_tool" not in called, f"{module} runs a tool itself"


@pytest.mark.asyncio
async def test_a_market_order_is_still_a_confirmed_action_whoever_asked_for_it():
    """The watch can see the drop. Placing the order is external and is confirmed
    like any other external act - a watch is not a way round that."""
    from core.safety_guard import ConfirmationDenied, RiskTier, SafetyGuard

    guard = SafetyGuard()
    assert guard.get_tier("place_paper_order") == RiskTier.EXTERNAL
    assert guard.requires_confirmation(RiskTier.EXTERNAL)

    guard.confirmation_callback = lambda prompt: False
    with pytest.raises(ConfirmationDenied):
        await guard.authorize("place_paper_order", {"symbol": "TTWO", "qty": 10, "side": "sell"})


@pytest.mark.asyncio
async def test_an_unattended_run_cannot_be_talked_into_trading():
    """A watch firing at 3am runs its task unattended, where external acts are
    refused outright rather than confirmed by nobody."""
    from core.safety_guard import ConfirmationDenied, SafetyGuard

    guard = SafetyGuard()
    guard.set_unattended(True)
    with pytest.raises(ConfirmationDenied) as refused:
        await guard.authorize("place_paper_order", {"symbol": "TTWO", "qty": 1, "side": "sell"})
    assert "nobody to confirm it" in str(refused.value)


def test_every_watch_and_market_tool_has_exactly_one_permission_entry():
    from core.config_loader import get_permissions

    entries = get_permissions()["tools"]
    for name in ("create_watch", "list_watches", "manage_watch", "check_watches",
                 "get_market_quote", "evaluate_trading_strategy", "place_paper_order"):
        assert name in entries, f"{name} has no action class"
    assert entries["check_watches"]["action"] == "read"
    assert entries["create_watch"]["action"] == "modify"


# --- Cost --------------------------------------------------------------------------

def test_deciding_whether_a_condition_is_true_never_calls_the_model():
    """The arithmetic is arithmetic. The model reads the evidence afterwards, when
    there is something to say; it is not in the loop that decides."""
    import ast

    for module in ("core/watches.py", "core/signals.py"):
        source = open(module).read()
        for forbidden in ("llm_client", "ollama", "generate(", "chat("):
            assert forbidden not in source, f"{module} mentions {forbidden}"
        tree = ast.parse(source)
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)]


def test_only_the_watches_that_are_due_are_evaluated(market, monkeypatch):
    looked_at = []
    for n in range(4):
        watch = market_watch(threshold=90 + n)
        if n < 2:
            watch["last_checked_at"] = time.time()          # just checked
            watches._replace(watch)

    real_check = watches.check
    monkeypatch.setattr(watches, "check",
                        lambda wid, now=None: looked_at.append(wid) or real_check(wid, now))
    pending = [w for w in watches.load_watches() if watches.due(w)]
    for watch in pending:
        watches.check(watch["id"])
    assert len(looked_at) == 2


def test_a_watch_keeps_a_bounded_history_rather_than_the_data_it_read(market, searches):
    watch = news_watch()
    for n in range(watches.MAX_HISTORY + 15):
        searches["articles"] = [article(f"A wholly separate development {n} occurs",
                                        "reuters.com")]
        watches.check(watch["id"])
    stored = watches.get_watch(watch["id"])
    assert len(stored["history"]) <= watches.MAX_HISTORY
    assert len(json.dumps(stored)) < 60_000, "the watch store is becoming a data store"


def test_evidence_too_large_to_keep_is_summarised_not_stored(market):
    huge = {"evidence": {"bars": ["x" * 100 for _ in range(200)]}}
    trimmed = watches._trim(huge["evidence"])
    assert len(json.dumps(trimmed)) <= watches.MAX_EVIDENCE_CHARS + 200
    assert "shortened" in json.dumps(trimmed)


def test_the_trend_series_does_not_grow_without_limit(searches):
    watch = trend_watch()
    for wave in range(20):
        observe(watch["id"], searches, 3, wave=wave)
    series = watches.get_watch(watch["id"])["condition_state"]["series"]
    assert len(series) <= signals.MAX_SERIES_POINTS


@pytest.mark.asyncio
async def test_a_slow_provider_does_not_freeze_the_interface(market, searches, monkeypatch):
    """Watches wait on somebody else's API, and the interface is on this event
    loop. The sweep runs in one worker thread - one for the sweep, not one per
    watch - so a search that takes three seconds does not stop the screen.

    What this asserts is the MECHANISM, not a stopwatch reading. An earlier
    version required no heartbeat gap to exceed 0.1s against a 0.15s provider
    call, which is a real property measured by a proxy that a loaded machine can
    trip: inside a full 1,700-test run, one missed 10ms wakeup fails it while the
    sweep is behaving perfectly. So it now checks the two things that are
    actually true when the loop is free - every provider call happened off the
    main thread, and the loop went on ticking throughout - and keeps a stall
    ceiling set to what a genuinely frozen loop would look like rather than to
    what a busy one might.
    """
    from tools.watch_tools import CheckWatchesTool

    calling_threads = []

    async def slow_searches(queries, max_results=5):
        calling_threads.append(threading.current_thread())
        await asyncio.sleep(0.15)
        return {"results": [article("Something happened at last", "reuters.com")],
                "failed_queries": []}

    import tools.web_search as web_search

    monkeypatch.setattr(web_search, "run_searches", slow_searches)
    for n in range(3):
        news_watch(topic=f"topic {n}")

    ticks = []

    async def heartbeat():
        while True:
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)

    beat = asyncio.get_running_loop().create_task(heartbeat())
    started = time.perf_counter()
    try:
        result = await CheckWatchesTool().run()
    finally:
        beat.cancel()
    swept = time.perf_counter() - started

    assert result.output["checked"] == 3
    # The mechanism: nothing waited on a provider from the thread the interface
    # and the voice loop live on.
    assert calling_threads, "no provider call was made"
    main = threading.main_thread()
    assert all(t is not main for t in calling_threads), (
        "a provider call ran on the event loop's thread")
    # And the consequence: the loop kept running while it happened. Three 0.15s
    # calls take about 0.45s, which is ~45 ticks at 10ms; a frozen loop manages
    # one or two.
    assert len(ticks) > 10, "the loop stopped while watches were evaluated"
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert max(gaps) < swept / 2, (
        f"the loop stalled for {max(gaps):.2f}s of a {swept:.2f}s sweep")


def test_one_worker_thread_for_the_sweep_not_one_per_provider_call(market, searches):
    """run_blocking only borrows a thread when it is called from inside a running
    loop. Off the loop - which is where the sweep runs - it uses the calling
    thread, so a sweep of twenty watches is one thread, not twenty."""
    import threading

    from tools.watch_tools import CheckWatchesTool

    for n in range(4):
        market_watch(threshold=90 + n)
        news_watch(topic=f"topic {n}")

    before = threading.active_count()
    peak = []

    real_check = watches.check

    def counting_check(watch_id, now=None):
        peak.append(threading.active_count())
        return real_check(watch_id, now)

    watches.check, checked = counting_check, None
    try:
        checked = asyncio.run(CheckWatchesTool().run())
    finally:
        watches.check = real_check

    assert checked.output["checked"] == 8
    # The executor's own worker, and nothing else per watch.
    assert max(peak) <= before + 2, f"threads grew to {max(peak)} from {before}"


# --- Options: only if the account is actually entitled to them -----------------------

def test_an_options_symbol_is_not_mistaken_for_a_share(market):
    assert trading_platform.is_option("AAPL241220C00150000")
    assert trading_platform.option_underlying("AAPL241220C00150000") == "AAPL"
    assert not trading_platform.is_option("TTWO")
    assert not trading_platform.is_option("BTC/USD")


@pytest.mark.asyncio
async def test_an_options_request_without_the_subscription_says_which_subscription(market):
    """The wrong answer is 'no data for that contract', which reads as 'it does not
    exist'. A missing OPRA entitlement is a different sentence."""
    from tools.trading_platform import GetQuoteTool

    async def refuse(base, path, params=None):
        if "options" in path:
            raise httpx.HTTPStatusError(
                "x", request=httpx.Request("GET", "https://data.alpaca.markets" + path),
                response=httpx.Response(403, text="subscription does not permit options"))
        return await market.get(base, path, params)

    market_get, trading_platform._get = market.get, refuse
    try:
        result = await GetQuoteTool().run(["TTWO", "AAPL241220C00150000"])
    finally:
        trading_platform._get = market_get

    assert result.success
    assert result.output["quotes"]["TTWO"]["price"] == 100.0
    contract = result.output["options"]["AAPL241220C00150000"]
    assert contract["problem"] == "subscription"
    assert "subscription" in contract["unavailable"].lower()
    assert "AAPL241220C00150000" not in result.output["no_data_for"]


@pytest.mark.asyncio
async def test_options_come_back_when_the_account_does_have_them(market):
    from tools.trading_platform import GetQuoteTool

    async def with_options(base, path, params=None):
        if "options" in path:
            return {"snapshots": {"AAPL241220C00150000": {"latestTrade": {"p": 3.25}}}}
        return await market.get(base, path, params)

    market_get, trading_platform._get = market.get, with_options
    try:
        result = await GetQuoteTool().run(["AAPL241220C00150000"])
    finally:
        trading_platform._get = market_get
    assert result.output["options"]["AAPL241220C00150000"]["AAPL241220C00150000"]["latestTrade"]["p"] == 3.25


def test_what_the_feed_report_claims_is_only_what_is_configured(market):
    report = trading_platform.feed_report()
    assert report["credentials_configured"] is True
    assert report["stock_feed"] == "iex"
    assert "not the whole US market" in report["stock_feed_note"]
    assert "OPRA" in report["options"]
    assert "not used" in report["streaming"]
    # The key itself is never in anything this returns.
    assert "secret" not in json.dumps(report).lower()


# --- Cross-domain: several signals, one answer ---------------------------------------
#
# "Tell me when at least two of these three move" is a real question about a
# market, a story and the attention around it, and neither all-of-them nor
# any-of-them asks it.

def at_least_watch(minimum=2):
    return watches.save_new(watches.create("cross", "combined", {
        "mode": "at_least", "minimum": minimum,
        "parts": [
            {"condition_type": "market",
             "condition": {"symbol": "TTWO", "metric": "change_percent",
                           "comparison": "abs_above", "threshold": 5.0}},
            {"condition_type": "news_event", "condition": {"topic": "GTA VI"}},
            {"condition_type": "trend", "condition": {"topic": "GTA VI",
                                                      "acceleration": 1.6}},
        ]}, cooldown_minutes=0))


def test_one_signal_out_of_three_is_not_two(market, searches):
    searches["articles"] = []
    watch = at_least_watch(minimum=2)          # only the market moves
    assert watches.check(watch["id"])["outcome"] == "false"


def test_two_signals_out_of_three_is(market, searches):
    searches["articles"] = [article("Rockstar announces GTA VI release date", "reuters.com")]
    watch = at_least_watch(minimum=2)
    assert watches.check(watch["id"])["outcome"] == "triggered"

    evidence = watches.get_watch(watch["id"])["history"][-1]["evidence"]
    assert evidence["mode"] == "at_least"
    met = [s["signal"] for s in evidence["signals"] if s.get("met")]
    assert set(met) == {"market", "news_event"}
    # And the ones that did NOT fire are still shown, which is what makes the
    # trigger explicable rather than just true.
    assert len(evidence["signals"]) == 3


def test_enough_unmeasurable_signals_to_change_the_answer_is_not_an_answer(market, searches):
    """Two of three, with two of them failing, is not "no" - it is "cannot tell",
    and saying no would be reading a provider outage as a fact about the world."""
    searches["fail"] = RuntimeError("search unavailable")
    watch = at_least_watch(minimum=2)

    outcome = watches.check(watch["id"])
    assert outcome["outcome"] == "error"
    assert "could not be measured" in outcome["error"]
    assert watches.get_watch(watch["id"])["condition_was_true"] is False


def test_a_failure_that_cannot_change_the_answer_does_not_stop_the_watch(market, searches):
    """One failed signal when one is already enough: the answer is the same either
    way, so it is given rather than withheld."""
    searches["fail"] = RuntimeError("search unavailable")
    watch = at_least_watch(minimum=1)          # the market alone satisfies it
    assert watches.check(watch["id"])["outcome"] == "triggered"


def test_a_minimum_bigger_than_the_signals_is_refused():
    problems = watches.validate(watches.create("c", "combined", {
        "mode": "at_least", "minimum": 5,
        "parts": [{"condition_type": "market",
                   "condition": {"symbol": "X", "metric": "price",
                                 "comparison": "above", "threshold": 1}},
                  {"condition_type": "news_event", "condition": {"topic": "t"}}]}))
    assert any("between 1 and 2" in p for p in problems)


def test_a_cross_domain_watch_reads_as_a_sentence(market, searches):
    watch = at_least_watch(minimum=2)
    reads = watches.describe_condition(watches.get_watch(watch["id"]))
    assert reads.startswith("at least 2 of:")
    assert "TTWO" in reads and "GTA VI" in reads


@pytest.mark.asyncio
async def test_three_signals_can_be_asked_for_in_one_sentence(market, searches):
    """The brief's example: watch the stock, the news and the interest, and tell
    me when at least two of them move."""
    from tools.watch_tools import CreateWatchTool

    result = await CreateWatchTool().run(
        name="TTWO, GTA VI and the interest", condition_type="market", symbol="TTWO",
        threshold=5, topic="GTA VI", also_watch=["news", "trend"],
        combine="at_least", minimum=2)

    assert result.success
    stored = watches.get_watch(result.output["watch"]["id"])
    assert stored["condition_type"] == "combined"
    assert stored["condition"]["minimum"] == 2
    assert [p["condition_type"] for p in stored["condition"]["parts"]] == [
        "market", "news_event", "trend"]


@pytest.mark.asyncio
async def test_a_single_extra_signal_still_works_as_it_did(market, searches):
    from tools.watch_tools import CreateWatchTool

    result = await CreateWatchTool().run(
        name="two signals", condition_type="market", symbol="TTWO", threshold=5,
        topic="GTA VI", also_watch="news")
    assert result.success
    stored = watches.get_watch(result.output["watch"]["id"])
    assert stored["condition"]["mode"] == "all"
    assert len(stored["condition"]["parts"]) == 2


@pytest.mark.asyncio
async def test_the_same_signal_twice_is_refused(market):
    from tools.watch_tools import CreateWatchTool

    result = await CreateWatchTool().run(
        name="x", condition_type="market", symbol="TTWO", threshold=5,
        also_watch=["market"])
    assert result.success is False
    assert "different kinds" in result.error


@pytest.mark.asyncio
async def test_a_combined_watch_still_asks_for_what_it_needs(market):
    """Folding in a news signal needs a topic, and inventing one would be worse
    than asking for it."""
    from tools.watch_tools import CreateWatchTool

    result = await CreateWatchTool().run(
        name="x", condition_type="market", symbol="TTWO", threshold=5,
        also_watch=["news"])
    assert result.success is False
    assert "topic" in result.error.lower()
