"""What a watch can measure: the market, the news, and how fast attention is moving.

core/watches.py owns the store, the schedule and the transition rules. This owns
the three kinds of thing those rules can be written about, and it is deliberately
the only place any of them is computed.

Three principles run through all of it.

  Deterministic first. Every condition here is arithmetic on numbers a provider
  returned - a percentage, a ratio, a count. Nothing here calls the model. The
  model reads the evidence afterwards, when there is something to tell the user;
  it is not in the loop that decides whether anything happened.

  Missing data is not a false condition. Every signal either produces a number or
  raises. "Alpaca is rate-limiting us" and "TTWO did not move" are different
  answers, and a monitor that confuses them is worse than no monitor.

  Analysis is labelled as analysis. The market signals are measurements. The news
  signals are reports, with the sources that carried them and how independent they
  are. The trend signals are estimates, and they say so in the same breath: the
  strongest thing this can honestly say is that attention is accelerating, never
  that a peak is coming on Tuesday.

Nothing here holds a connection open, runs a loop or starts a thread that outlives
the call. Watches are evaluated when the scheduler says so, several at a time,
sharing one market request per asset class (see tools/trading_platform.py) and one
search per topic.

On streaming: Alpaca offers a market-data WebSocket, and this does not use it. A
socket earns its keep when the consumer needs sub-second ticks. Leti's watches are
evaluated at most once a minute against daily and minute bars, so a socket would
buy nothing measurable and cost a permanently open connection, a reconnect loop
and a background task per feed - the three things the watch architecture exists to
avoid. Batched snapshot requests give the same answer at these intervals.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("leti.signals")


class SignalError(Exception):
    """A signal could not be measured, and why. Never means 'condition false'."""

    def __init__(self, message: str, kind: str = "unavailable"):
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------------- #
# Running one coroutine from the synchronous evaluator
# --------------------------------------------------------------------------- #

def run_blocking(coro, timeout: float = 45.0):
    """Run an async provider call from watch evaluation, which is synchronous.

    Watches are evaluated from check_watches, which is itself an async tool, so
    asyncio.run() raises "cannot be called from a running event loop" - which is
    how price watches used to fail every time they were evaluated by the
    scheduler. A worker thread with its own loop is the standard answer: it lasts
    exactly as long as the request and nothing is left running afterwards.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.wait_for(coro, timeout))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="leti-signal") as pool:
        return pool.submit(lambda: asyncio.run(asyncio.wait_for(coro, timeout))).result(
            timeout=timeout + 5)


# --------------------------------------------------------------------------- #
# Market
# --------------------------------------------------------------------------- #
#
# One condition shape for every market question, rather than a tool per question:
# a metric, a comparison and a number. "Tell me if TTWO moves more than 5%" is
# change_percent / abs_above / 5, and "tell me if volume spikes" is volume_ratio /
# above / 3. The alternative - price_watch, percent_watch, volume_watch,
# breakout_watch - is four names for one piece of arithmetic.

MARKET_METRICS: Dict[str, str] = {
    "price": "the last traded price",
    "change_percent": "today's move against yesterday's close, in percent",
    "change": "today's move against yesterday's close, in currency",
    "volume": "shares or coins traded today",
    "volume_ratio": "today's volume against its recent average (a spike is > 2)",
    "gap_percent": "today's open against yesterday's close, in percent",
    "high_breakout_percent": "the last price above the recent high, in percent",
    "low_breakout_percent": "the last price below the recent low, in percent",
    "sma_gap_percent": "the last price above (+) or below (-) its moving average, in percent",
    "momentum_percent": "the move over the whole window, in percent",
    "volatility_percent": "the standard deviation of daily returns in the window, in percent",
}

# abs_above is the one that makes "moves more than 5%" a single condition rather
# than two watches for up and down.
COMPARISONS = ("above", "below", "abs_above", "abs_below")

DEFAULT_WINDOW_DAYS = 20
MAX_WINDOW_DAYS = 200
NEEDS_HISTORY = frozenset({"volume_ratio", "high_breakout_percent", "low_breakout_percent",
                           "sma_gap_percent", "momentum_percent", "volatility_percent"})


def market_value(symbol: str, metric: str, window: int = DEFAULT_WINDOW_DAYS,
                 snapshot: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
    """The metric's current value, and the evidence behind it.

    snapshot is passed in when a batch of watches has already fetched it, so
    several watches on one symbol cost one request between them.
    """
    from tools.trading_platform import MarketDataError, get_bars, get_snapshots

    metric = str(metric or "").strip()
    if metric not in MARKET_METRICS:
        raise SignalError(f"'{metric}' is not a market metric. Known: "
                          f"{', '.join(sorted(MARKET_METRICS))}.", kind="unsupported")
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        raise SignalError("That watch has no symbol to look up.", kind="misconfigured")
    window = max(2, min(int(window or DEFAULT_WINDOW_DAYS), MAX_WINDOW_DAYS))

    try:
        if snapshot is None:
            snapshot = (run_blocking(get_snapshots([symbol])) or {}).get(symbol)
        if snapshot is None:
            raise SignalError(f"No market data came back for {symbol}.", kind="no_data")
        bars = (run_blocking(get_bars(symbol, "1Day", window + 1))
                if metric in NEEDS_HISTORY else [])
    except MarketDataError as e:
        raise SignalError(str(e), kind=e.kind)
    except SignalError:
        raise
    except Exception as e:
        raise SignalError(f"Couldn't read market data for {symbol}: {e}", kind="unavailable")

    evidence: Dict[str, Any] = {"symbol": symbol, "metric": metric,
                                "feed": snapshot.get("feed"), "price": snapshot.get("price"),
                                "as_of": snapshot.get("as_of")}

    def need(value, what):
        if value is None:
            raise SignalError(f"{symbol}: {what} is not in the data the feed returned "
                              f"(feed: {snapshot.get('feed')}).", kind="no_data")
        return float(value)

    if metric == "price":
        value = need(snapshot.get("price"), "a last price")
    elif metric == "change_percent":
        value = need(snapshot.get("change_percent"), "today's change")
        evidence["previous_close"] = (snapshot.get("previous_daily_bar") or {}).get("close")
    elif metric == "change":
        value = need(snapshot.get("change"), "today's change")
        evidence["previous_close"] = (snapshot.get("previous_daily_bar") or {}).get("close")
    elif metric == "volume":
        value = need(snapshot.get("volume"), "today's volume")
    elif metric == "gap_percent":
        value = need(snapshot.get("gap_percent"), "today's open against yesterday's close")
    else:
        value = _historical_metric(metric, symbol, snapshot, bars, window, evidence)

    evidence["value"] = round(value, 6)
    return value, evidence


def _historical_metric(metric: str, symbol: str, snapshot: Dict[str, Any],
                       bars: List[Dict[str, Any]], window: int,
                       evidence: Dict[str, Any]) -> float:
    """The metrics that need history. Bars are oldest first and include today."""
    if len(bars) < 3:
        raise SignalError(
            f"{symbol}: only {len(bars)} day(s) of history came back, which is not "
            "enough to compare against. A newly listed symbol looks like this.",
            kind="no_data")

    closes = [b["close"] for b in bars]
    volumes = [b["volume"] for b in bars]
    price = snapshot.get("price") or closes[-1]
    evidence["window_days"] = window
    evidence["bars_used"] = len(bars)

    if metric == "volume_ratio":
        today = snapshot.get("volume")
        if today is None:
            today = volumes[-1]
        earlier = [v for v in volumes[:-1] if v > 0]
        if not earlier:
            raise SignalError(f"{symbol}: no earlier volume to compare today's against.",
                              kind="no_data")
        average = sum(earlier) / len(earlier)
        evidence.update({"today_volume": today, "average_volume": round(average, 2)})
        return float(today) / average

    if metric in ("high_breakout_percent", "low_breakout_percent"):
        highs = [b["high"] for b in bars[:-1]]
        lows = [b["low"] for b in bars[:-1]]
        if metric == "high_breakout_percent":
            reference = max(highs)
            evidence["window_high"] = reference
            return (price - reference) / reference * 100.0
        reference = min(lows)
        evidence["window_low"] = reference
        return (reference - price) / reference * 100.0

    if metric == "sma_gap_percent":
        average = sum(closes[-window:]) / len(closes[-window:])
        evidence["moving_average"] = round(average, 4)
        return (price - average) / average * 100.0

    if metric == "momentum_percent":
        first = closes[0]
        evidence["window_open"] = first
        return (price - first) / first * 100.0

    if metric == "volatility_percent":
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                   for i in range(1, len(closes)) if closes[i - 1]]
        if len(returns) < 2:
            raise SignalError(f"{symbol}: not enough history to measure volatility.",
                              kind="no_data")
        mean = sum(returns) / len(returns)
        deviation = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
        evidence["daily_returns_used"] = len(returns)
        return deviation * 100.0

    raise SignalError(f"'{metric}' has no implementation.", kind="unsupported")


def compare(value: float, comparison: str, threshold: float) -> bool:
    comparison = str(comparison or "above").lower()
    if comparison == "above":
        return value > threshold
    if comparison == "below":
        return value < threshold
    if comparison == "abs_above":
        return abs(value) > abs(threshold)
    if comparison == "abs_below":
        return abs(value) < abs(threshold)
    raise SignalError(f"'{comparison}' is not a comparison. Use: {', '.join(COMPARISONS)}.",
                      kind="misconfigured")


def describe_market(condition: Dict[str, Any]) -> str:
    metric = condition.get("metric", "price")
    comparison = str(condition.get("comparison", "above"))
    threshold = condition.get("threshold")
    word = {"above": "above", "below": "below",
            "abs_above": "moving more than", "abs_below": "moving less than"}.get(comparison, comparison)
    unit = "%" if metric.endswith("_percent") else ("x" if metric.endswith("_ratio") else "")
    return f"{condition.get('symbol', '?')} {metric.replace('_', ' ')} {word} {threshold}{unit}"


# --------------------------------------------------------------------------- #
# News and events
# --------------------------------------------------------------------------- #
#
# The hard part is not finding articles, it is not reporting the same thing twice.
# Ten outlets covering one arrest is one event. So articles are grouped by what
# they say rather than counted, each group gets a fingerprint, and a fingerprint
# that has been reported is never reported again.

MAJOR_OUTLETS = frozenset({
    "reuters.com", "apnews.com", "ap.org", "bbc.com", "bbc.co.uk", "ft.com",
    "wsj.com", "bloomberg.com", "cnbc.com", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "npr.org", "aljazeera.com", "dw.com", "economist.com",
    "politico.com", "axios.com", "cnn.com", "abcnews.go.com", "nbcnews.com",
    "cbsnews.com", "forbes.com", "time.com", "latimes.com", "independent.co.uk",
    "telegraph.co.uk", "thetimes.co.uk", "lemonde.fr", "spiegel.de", "elpais.com",
    "kathimerini.gr", "nikkei.com", "scmp.com", "theverge.com", "arstechnica.com",
    "wired.com", "techcrunch.com", "engadget.com", "eurogamer.net", "ign.com",
    "gamespot.com", "polygon.com", "pcgamer.com", "gamesindustry.biz",
    "bleepingcomputer.com", "krebsonsecurity.com", "therecord.media",
    "securityweek.com", "darkreading.com",
})

SOCIAL_DOMAINS = frozenset({
    "x.com", "twitter.com", "reddit.com", "facebook.com", "instagram.com",
    "tiktok.com", "youtube.com", "t.me", "telegram.org", "threads.net",
    "mastodon.social", "bsky.app", "4chan.org", "medium.com", "substack.com",
    "quora.com", "discord.com", "linkedin.com",
})

REGULATOR_SUFFIXES = (".gov", ".gov.uk", ".europa.eu", ".mil", ".police.uk", ".gov.au",
                      ".gc.ca", ".govt.nz", ".gov.in", ".gov.gr")

# What each tier is worth when deciding whether something is confirmed or merely
# going around. Ordered, not weighted: two blogs are not a wire service.
TIER_STRENGTH = {"official": 3, "regulator": 3, "major": 2, "secondary": 1, "social": 0}

MAX_SEEN_EVENTS = 120           # fingerprints kept per watch - small strings
MAX_SERIES_POINTS = 60          # attention history per watch
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "is", "are",
    "was", "were", "be", "been", "has", "have", "had", "it", "its", "this",
    "that", "with", "by", "as", "at", "from", "after", "over", "new", "says",
    "said", "report", "reports", "reported", "news", "update", "updates",
    "what", "how", "why", "when", "will", "can", "could", "may", "about",
})


def source_tier(domain: str, official_domains: Optional[List[str]] = None) -> str:
    """What kind of source this is. Structural where possible, listed where not."""
    domain = (domain or "").lower().lstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    for official in (official_domains or []):
        official = str(official).lower().lstrip(".")
        if domain == official or domain.endswith("." + official):
            return "official"
    if any(domain.endswith(suffix) for suffix in REGULATOR_SUFFIXES):
        return "regulator"
    if domain in MAJOR_OUTLETS or any(domain.endswith("." + d) for d in MAJOR_OUTLETS):
        return "major"
    if domain in SOCIAL_DOMAINS or any(domain.endswith("." + d) for d in SOCIAL_DOMAINS):
        return "social"
    return "secondary" if domain else "social"


def _terms(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if w not in STOPWORDS]


def _overlap(a: List[str], b: List[str]) -> float:
    """Jaccard on the distinctive words of two headlines."""
    first, second = set(a), set(b)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def fingerprint(terms: List[str]) -> str:
    """A stable id for an event, from the words its coverage agrees on.

    The five most distinctive words in alphabetical order: two outlets writing
    "Police arrest suspect in Rockstar leak" and "Suspect arrested over Rockstar
    leak, police say" land on the same fingerprint, which is what stops the second
    one being a new event tomorrow.
    """
    core = sorted(set(terms))[:5]
    return hashlib.sha256("|".join(core).encode("utf-8")).hexdigest()[:16] if core else ""


def group_into_events(articles: List[Dict[str, Any]], official_domains=None,
                      similarity: float = 0.34) -> List[Dict[str, Any]]:
    """Articles about one thing, grouped into one event with its sources."""
    events: List[Dict[str, Any]] = []
    for article in articles:
        headline = f"{article.get('title') or ''} {article.get('snippet') or ''}"
        terms = _terms(article.get("title") or "") or _terms(headline)
        if not terms:
            continue
        domain = (article.get("domain") or urlparse(article.get("url") or "").netloc).lower()
        tier = source_tier(domain, official_domains)
        for event in events:
            if _overlap(terms, event["_terms"]) >= similarity:
                event["_terms"] = sorted(set(event["_terms"]) & set(terms)) or event["_terms"]
                if domain and domain not in event["domains"]:
                    event["domains"].append(domain)
                    event["tiers"].append(tier)
                if len(event["articles"]) < 6:
                    event["articles"].append({"title": article.get("title"),
                                              "url": article.get("url"),
                                              "domain": domain, "tier": tier})
                break
        else:
            events.append({
                "_terms": terms,
                "headline": article.get("title") or "",
                "domains": [domain] if domain else [],
                "tiers": [tier],
                "articles": [{"title": article.get("title"), "url": article.get("url"),
                              "domain": domain, "tier": tier}],
            })

    for event in events:
        tiers = event.pop("tiers")
        event["fingerprint"] = fingerprint(event["_terms"])
        event["independent_sources"] = len(set(event["domains"]))
        event["best_source"] = max(tiers, key=lambda t: TIER_STRENGTH[t]) if tiers else "social"
        event["strength"] = _strength(event["domains"], tiers)
        event["terms"] = event.pop("_terms")[:8]
    return events


def _strength(domains: List[str], tiers: List[str]) -> Dict[str, Any]:
    """How well supported this event is, and the words to use about it.

    Nothing here is "confirmed" because Leti checked; it is confirmed because
    several independent outlets that are not each other carried it, or an official
    or regulator source did. One post on a social platform is never enough - that
    is the difference between "the hacker was caught" and "somebody said so".
    """
    independent = len(set(domains))
    credible = [t for t in tiers if TIER_STRENGTH[t] >= 2]
    official = [t for t in tiers if TIER_STRENGTH[t] >= 3]
    if official:
        level, wording = "confirmed", "reported by an official or regulatory source"
    elif len(set(credible)) and independent >= 2 and len(credible) >= 2:
        level, wording = "confirmed", f"carried by {independent} independent major outlets"
    elif credible:
        level, wording = "reported", "carried by one major outlet so far"
    elif independent >= 2:
        level, wording = "reported", f"carried by {independent} secondary sources"
    else:
        level, wording = "unverified", "a single unverified source"
    return {"level": level, "why": wording, "independent_sources": independent,
            "credible_sources": len(credible)}


def search_topic(topic: str, extra_queries: Optional[List[str]] = None,
                 max_results: int = 8) -> List[Dict[str, Any]]:
    """Recent coverage of a topic, through the search tool Leti already has."""
    from tools.web_search import run_searches

    queries = [q for q in ([f"{topic} news", topic] + list(extra_queries or [])) if q]
    try:
        found = run_blocking(run_searches(queries[:4], max_results=max_results))
    except Exception as e:
        raise SignalError(f"Couldn't search for '{topic}': {e}", kind="unavailable")
    results = found.get("results") or []
    if not results and found.get("failed_queries"):
        raise SignalError(
            f"Every search for '{topic}' failed: "
            f"{found['failed_queries'][0].get('error', 'no reason given')}", kind="unavailable")
    return results


def relevant(event: Dict[str, Any], must_include: List[str], exclude: List[str]) -> bool:
    text = " ".join([event.get("headline", "")] + event.get("terms", [])).lower()
    if any(str(word).lower() in text for word in exclude if str(word).strip()):
        return False
    return all(str(word).lower() in text for word in must_include if str(word).strip())


def news_events(condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """New, relevant, sufficiently-supported events about a topic.

    True on the arrival of an event that has not been reported before and that
    clears the bar the watch set. The seen list is what makes the second telling
    of the same story silence rather than a notification.
    """
    topic = str(condition.get("topic", "")).strip()
    if not topic:
        raise SignalError("That news watch has no topic.", kind="misconfigured")

    minimum = max(1, int(condition.get("min_sources", 1) or 1))
    require_confirmation = bool(condition.get("require_confirmation"))
    official_domains = list(condition.get("official_domains") or [])
    must_include = list(condition.get("must_include") or [])
    exclude = list(condition.get("exclude") or [])

    articles = search_topic(topic, condition.get("extra_queries"))
    events = group_into_events(articles, official_domains)
    seen = list(state.get("seen") or [])
    seen_set = set(seen)

    qualifying, ignored = [], 0
    for event in events:
        if not relevant(event, must_include, exclude):
            continue
        if event["independent_sources"] < minimum:
            ignored += 1
            continue
        if require_confirmation and event["strength"]["level"] != "confirmed":
            ignored += 1
            continue
        if event["fingerprint"] in seen_set:
            continue
        qualifying.append(event)

    # Everything relevant that was looked at is remembered, not only what fired:
    # an event that was below the bar today and above it tomorrow is still the same
    # event, and the point of the fingerprint is that it is only news once.
    for event in events:
        if event["fingerprint"] and event["fingerprint"] not in seen_set:
            seen.append(event["fingerprint"])
            seen_set.add(event["fingerprint"])

    series = list(state.get("series") or [])
    series.append({"at": time.time(), "events": len(events),
                   "sources": len({d for e in events for d in e["domains"]})})

    new_state = {
        "seen": seen[-MAX_SEEN_EVENTS:],
        "series": series[-MAX_SERIES_POINTS:],
        # Which event this is, so two different developments on two consecutive
        # evaluations are two notifications rather than one and a silence.
        "event_key": "+".join(e["fingerprint"] for e in qualifying[:3]) or None,
        "last_checked_events": len(events),
        "evidence": {
            "topic": topic,
            "events_found": len(events),
            "below_the_bar": ignored,
            "new_events": [{
                "headline": e["headline"],
                "sources": e["domains"][:5],
                "independent_sources": e["independent_sources"],
                "support": e["strength"]["level"],
                "why": e["strength"]["why"],
                "articles": e["articles"][:3],
            } for e in qualifying[:3]],
            "how_to_report": (
                "Say what the sources say, name them, and use the support level as "
                "written: 'confirmed' means independent outlets or an official source "
                "carried it, 'reported' means it is going around, 'unverified' means "
                "one source said so and nothing else has."),
        },
    }
    return bool(qualifying), new_state


def describe_news(condition: Dict[str, Any]) -> str:
    topic = condition.get("topic", "?")
    bar = []
    if int(condition.get("min_sources", 1) or 1) > 1:
        bar.append(f"{condition['min_sources']}+ independent sources")
    if condition.get("require_confirmation"):
        bar.append("confirmed reporting only")
    return f"news about {topic}" + (f" ({', '.join(bar)})" if bar else "")


# --------------------------------------------------------------------------- #
# Trend and attention
# --------------------------------------------------------------------------- #
#
# This is the analytical one, and the one with the most ways to mislead. It does
# not predict. It measures how fast the other signals are changing and says which
# phase that looks like, with the sample count attached so a confident-sounding
# sentence can always be checked against how little it is based on.

MIN_TREND_SAMPLES = 4           # below this there is no rate to compare, only noise
ACCELERATING_RATIO = 1.6        # recent activity against the window before it
COOLING_RATIO = 0.6


def _phase(ratio: Optional[float], recent_mean: float, samples: int) -> Tuple[str, str]:
    if ratio is None:
        return "unknown", "not enough history yet to compare one period against another"
    if recent_mean <= 0.5:
        return "quiet", "almost nothing is being published about it"
    if ratio >= ACCELERATING_RATIO * 1.5:
        return "accelerating sharply", f"recent activity is {ratio:.1f}x the period before"
    if ratio >= ACCELERATING_RATIO:
        return "accelerating", f"recent activity is {ratio:.1f}x the period before"
    if ratio <= COOLING_RATIO:
        return "cooling", f"recent activity is {ratio:.1f}x the period before"
    return "steady", f"recent activity is {ratio:.1f}x the period before, which is flat"


def _confidence(samples: int, sources: int, consistent: bool) -> str:
    """How much weight the estimate can carry. Never higher than 'moderate' on a
    handful of samples, whatever the numbers look like."""
    if samples < MIN_TREND_SAMPLES:
        return "none"
    if samples >= 12 and sources >= 4 and consistent:
        return "moderate-to-high"
    if samples >= 8 and sources >= 2:
        return "moderate"
    return "low"


def trend(condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Whether attention around something is accelerating.

    Every number in here is counted from what was actually observed on previous
    evaluations - this is why a trend watch says nothing useful for its first few
    hours and says so plainly rather than guessing from one sample.
    """
    topic = str(condition.get("topic", "")).strip()
    symbol = str(condition.get("symbol", "")).strip().upper()
    if not topic and not symbol:
        raise SignalError("A trend watch needs a topic, a symbol, or both.",
                          kind="misconfigured")

    point: Dict[str, Any] = {"at": time.time()}
    sources = 0
    problems = []

    if topic:
        try:
            events = group_into_events(search_topic(topic, condition.get("extra_queries")),
                                       condition.get("official_domains"))
            point["events"] = len(events)
            sources = len({d for e in events for d in e["domains"]})
            point["sources"] = sources
        except SignalError as e:
            # One failed search is a gap in the series, not a reading of zero:
            # recording a zero would read as "interest collapsed".
            problems.append(str(e))

    if symbol:
        try:
            value, evidence = market_value(symbol, "volume_ratio",
                                           int(condition.get("window", 20) or 20))
            point["volume_ratio"] = round(value, 3)
            point["price"] = evidence.get("price")
        except SignalError as e:
            problems.append(str(e))

    if "events" not in point and "volume_ratio" not in point:
        raise SignalError("; ".join(problems) or "no trend signal could be measured",
                          kind="unavailable")

    series = [p for p in (state.get("series") or []) if isinstance(p, dict)]
    series.append(point)
    series = series[-MAX_SERIES_POINTS:]

    measured = [p for p in series if "events" in p or "volume_ratio" in p]
    samples = len(measured)

    def activity(p):
        parts = [p.get("events"), p.get("volume_ratio")]
        values = [float(v) for v in parts if isinstance(v, (int, float))]
        return sum(values) / len(values) if values else 0.0

    ratio = recent_mean = earlier_mean = None
    if samples >= MIN_TREND_SAMPLES:
        half = max(2, samples // 2)
        recent = [activity(p) for p in measured[-half:]]
        earlier = [activity(p) for p in measured[:-half]] or [0.0]
        recent_mean = sum(recent) / len(recent)
        earlier_mean = sum(earlier) / len(earlier)
        # A zero baseline is real: something nobody was writing about now being
        # written about is the sharpest acceleration there is. It is expressed as a
        # bounded ratio rather than as infinity.
        ratio = (recent_mean / earlier_mean) if earlier_mean > 0.05 else (
            min(10.0, recent_mean * 4) if recent_mean > 0 else 1.0)

    phase, why = _phase(ratio, recent_mean or 0.0, samples)
    threshold = float(condition.get("acceleration", ACCELERATING_RATIO) or ACCELERATING_RATIO)
    minimum_activity = float(condition.get("min_activity", 1.0) or 1.0)
    is_true = bool(ratio is not None and ratio >= threshold
                   and (recent_mean or 0) >= minimum_activity)
    consistent = bool(measured[-2:] and all(activity(p) > 0 for p in measured[-2:]))

    new_state = {
        "series": series,
        "evidence": {
            "topic": topic or None,
            "symbol": symbol or None,
            "phase": phase,
            "why": why,
            "ratio": round(ratio, 3) if ratio is not None else None,
            "recent_activity": round(recent_mean, 2) if recent_mean is not None else None,
            "earlier_activity": round(earlier_mean, 2) if earlier_mean is not None else None,
            "samples": samples,
            "independent_sources": sources,
            "confidence": _confidence(samples, sources, consistent),
            "gaps": problems or None,
            "this_is_an_estimate": (
                "These are measured signals - how much is being published and how "
                "heavily the symbol is trading - not a forecast. Report it as what "
                "attention is doing now, with the confidence and the sample count. "
                "Never give a date for a peak; nothing here can know one."),
        },
    }
    if samples < MIN_TREND_SAMPLES:
        new_state["evidence"]["not_yet"] = (
            f"{samples} observation(s) so far; {MIN_TREND_SAMPLES} are needed before "
            "one period can be compared with another. Until then this watch reports "
            "nothing rather than guessing.")
    return is_true, new_state


def describe_trend(condition: Dict[str, Any]) -> str:
    what = " and ".join(x for x in [condition.get("topic"), condition.get("symbol")] if x)
    return f"attention around {what} accelerating"
