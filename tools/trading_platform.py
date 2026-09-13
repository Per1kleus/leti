"""
Trading platform integration - Alpaca paper (simulated) trading only.

IMPORTANT / BY DESIGN: the base URLs below are hardcoded to Alpaca's paper
endpoints and are NOT read from user settings. This is intentional: the
user asked for a sandbox, so this module cannot be pointed at a live
trading endpoint no matter what gets put in config/settings.yaml. If real
capital execution is ever wanted, that's a deliberate separate decision
requiring a new, explicitly-live-labeled module - not a config toggle here.

None of this is financial advice; the strategies below are simple,
well-known technical rules (SMA crossover, RSI threshold) provided as-is
for experimentation in a simulated account.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import re as _re
import time as _time

import httpx

from core.config_loader import get_settings
from tools.base import BaseTool, ToolParameter, ToolResult

PAPER_TRADING_BASE = "https://paper-api.alpaca.markets"
PAPER_DATA_BASE = "https://data.alpaca.markets"


def _creds() -> Dict[str, str]:
    settings = get_settings()
    trading = settings.get("trading", {})
    key, secret = trading.get("api_key", ""), trading.get("api_secret", "")
    if not key or not secret:
        raise RuntimeError(
            "No trading.api_key/api_secret in config/settings.yaml. Create a free Alpaca "
            "PAPER account (not a live account) and put its paper API key/secret there."
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _watchlist() -> List[str]:
    return get_settings().get("trading", {}).get("watchlist", [])


async def _get(base: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(base_url=base, headers=_creds(), timeout=15) as client:
        resp = await client.get(path, params=params or {})
        resp.raise_for_status()
        return resp.json()


async def _post(base: str, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(base_url=base, headers=_creds(), timeout=15) as client:
        resp = await client.post(path, json=json_body)
        resp.raise_for_status()
        return resp.json()


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol or symbol.upper().endswith(("USD", "USDT")) and len(symbol) > 5


# --------------------------------------------------------------------------- #
# Market data
#
# This is the one market-data client in Leti. The trading tools above use it, and
# so does the watch system (core/signals.py) - a watch does not open its own
# connection, and ten watches on five symbols are one HTTP request per asset class
# per evaluation, not ten.
#
# Three things this is careful about, because getting them wrong is how a monitor
# lies:
#
#   The feed is named, never assumed. A free Alpaca account gets IEX, which is one
#   exchange's view of the tape rather than the whole market; SIP needs a paid
#   subscription. Everything that comes back out of here says which feed it came
#   from, so "AAPL is at 190.12" can be read as "on IEX, at 14:03".
#
#   A failure has a kind. "Rate limited" and "that symbol does not exist" and "your
#   key is not subscribed to this feed" are different problems with different
#   answers, and a watch that treats any of them as "the condition is false" is
#   worse than a watch that stops. MarketDataError carries the kind; nothing here
#   ever returns a price it did not receive.
#
#   Nothing is held open. There is no socket, no thread and no background task:
#   the scheduler asks, this answers, and the process is idle again. See
#   core/signals.py for why streaming would buy nothing at these intervals.
# --------------------------------------------------------------------------- #

STOCK_FEEDS = ("iex", "sip", "delayed_sip", "otc", "boats", "overnight")
DEFAULT_STOCK_FEED = "iex"          # what a free Alpaca account actually gets
SNAPSHOT_CACHE_SECONDS = 30.0       # one evaluation cycle, not a store
MAX_CACHED_SYMBOLS = 200
MAX_SYMBOLS_PER_REQUEST = 100


class MarketDataError(Exception):
    """Market data could not be read, and why.

    kind is one of: unconfigured, auth, subscription, rate_limit, unknown_symbol,
    no_data, outage, malformed, network. The caller decides what to do with it -
    but "the condition is false" is never one of the options.
    """

    def __init__(self, message: str, kind: str = "outage", symbol: str = ""):
        super().__init__(message)
        self.kind = kind
        self.symbol = symbol


def market_settings() -> Dict[str, Any]:
    return get_settings().get("trading", {}) or {}


def stock_feed() -> str:
    """The configured stock feed, or IEX - which is what a free key can read."""
    feed = str(market_settings().get("data_feed", "") or "").strip().lower()
    return feed if feed in STOCK_FEEDS else DEFAULT_STOCK_FEED


def have_credentials() -> bool:
    """Whether a key is configured. Never returns or logs the key itself."""
    settings = market_settings()
    return bool(settings.get("api_key")) and bool(settings.get("api_secret"))


def feed_report() -> Dict[str, Any]:
    """What this installation can actually read, without claiming more.

    Deliberately does not call the API: it says what is configured, and what that
    configuration is known to include. Whether the account is subscribed to SIP is
    something only a request can answer, and it answers it with a 403 that
    _raise_for names as a subscription problem.
    """
    feed = stock_feed()
    return {
        "provider": "alpaca",
        "credentials_configured": have_credentials(),
        "stock_feed": feed,
        "stock_feed_note": (
            "IEX is one exchange's view of the tape, not the whole US market. "
            "Quotes and volume are IEX's own and will differ from consolidated "
            "(SIP) figures." if feed == "iex" else
            f"'{feed}' needs the matching Alpaca subscription; requests fail with a "
            "subscription error if the account does not have it."),
        "crypto": "v1beta3 crypto (US) - no separate subscription",
        "options": ("v1beta1 options - only if the account has an OPRA subscription; "
                    "Leti reports the provider's refusal rather than guessing"),
        "streaming": "not used - see core/signals.py",
    }


def _raise_for(error: Exception, symbols: str = "") -> None:
    """Turn a provider failure into a MarketDataError with a kind and a reason."""
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        # The body can carry the provider's own explanation. It never carries the
        # key - that goes in a header - but it is truncated anyway.
        detail = (error.response.text or "")[:200]
        if status in (401, 403):
            kind = "subscription" if "subscription" in detail.lower() else "auth"
            raise MarketDataError(
                ("The market data key is not authorised for that request"
                 f" ({status}). {detail}" if kind == "auth" else
                 f"That data needs an Alpaca subscription this account does not have. {detail}"),
                kind=kind, symbol=symbols)
        if status == 429:
            raise MarketDataError("Alpaca rate-limited the request; it will be retried "
                                  "on the next evaluation.", kind="rate_limit", symbol=symbols)
        if status == 404:
            raise MarketDataError(f"Alpaca does not know the symbol(s) {symbols}.",
                                  kind="unknown_symbol", symbol=symbols)
        if status >= 500:
            raise MarketDataError(f"Alpaca returned {status}; the provider is having "
                                  "trouble.", kind="outage", symbol=symbols)
        raise MarketDataError(f"Alpaca refused the request ({status}). {detail}",
                              kind="outage", symbol=symbols)
    if isinstance(error, RuntimeError) and "api_key" in str(error):
        raise MarketDataError(str(error), kind="unconfigured", symbol=symbols)
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        raise MarketDataError(f"Couldn't reach Alpaca: {error}", kind="network", symbol=symbols)
    raise MarketDataError(f"Market data failed: {error}", kind="outage", symbol=symbols)


# symbol -> (fetched_at, snapshot). One evaluation cycle's worth, bounded: two
# hundred symbols of small dicts, not a time series. core/signals.py keeps the
# history it needs in the watch's own state, where it is visible and persisted.
_SNAPSHOTS: Dict[str, Any] = {}


def clear_market_cache() -> None:
    _SNAPSHOTS.clear()


def _cached(symbols: List[str], now: float) -> Dict[str, Any]:
    fresh = {}
    for symbol in symbols:
        entry = _SNAPSHOTS.get(symbol)
        if entry and (now - entry[0]) < SNAPSHOT_CACHE_SECONDS:
            fresh[symbol] = entry[1]
    return fresh


def _remember(snapshots: Dict[str, Any], now: float) -> None:
    for symbol, snapshot in snapshots.items():
        _SNAPSHOTS[symbol] = (now, snapshot)
    if len(_SNAPSHOTS) > MAX_CACHED_SYMBOLS:
        for symbol in sorted(_SNAPSHOTS, key=lambda s: _SNAPSHOTS[s][0])[:len(_SNAPSHOTS) // 2]:
            _SNAPSHOTS.pop(symbol, None)


def _bar(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """One OHLCV bar in Leti's names rather than Alpaca's single letters."""
    if not isinstance(raw, dict):
        return None
    try:
        return {"open": float(raw["o"]), "high": float(raw["h"]), "low": float(raw["l"]),
                "close": float(raw["c"]), "volume": float(raw.get("v", 0) or 0),
                "at": raw.get("t")}
    except (KeyError, TypeError, ValueError):
        return None


def _snapshot_from(symbol: str, raw: Dict[str, Any], feed: str) -> Dict[str, Any]:
    """Alpaca's snapshot shape, reduced to what a condition can be written against."""
    daily = _bar(raw.get("dailyBar"))
    previous = _bar(raw.get("prevDailyBar"))
    minute = _bar(raw.get("minuteBar"))
    trade = raw.get("latestTrade") or {}
    try:
        price = float(trade.get("p"))
    except (TypeError, ValueError):
        price = daily["close"] if daily else (minute["close"] if minute else None)

    snapshot = {
        "symbol": symbol,
        "price": price,
        "feed": feed,
        "daily_bar": daily,
        "previous_daily_bar": previous,
        "minute_bar": minute,
        "as_of": trade.get("t") or (daily or {}).get("at"),
    }
    if price is not None and previous and previous["close"]:
        snapshot["change"] = price - previous["close"]
        snapshot["change_percent"] = (price - previous["close"]) / previous["close"] * 100.0
    if daily:
        snapshot["volume"] = daily["volume"]
        if previous and previous["close"]:
            snapshot["gap_percent"] = (daily["open"] - previous["close"]) / previous["close"] * 100.0
    return snapshot


async def get_snapshots(symbols: List[str], use_cache: bool = True) -> Dict[str, Any]:
    """Latest price, today's bar and yesterday's, for many symbols in one request.

    Batched on purpose: this is what keeps ten watches on the same handful of
    symbols from being ten round trips. Stocks and crypto go to different Alpaca
    endpoints, so that is two requests at most however many symbols are asked for.
    """
    wanted = [s.strip().upper() for s in symbols if str(s).strip()]
    if not wanted:
        return {}
    if not have_credentials():
        raise MarketDataError(
            "No Alpaca API key is configured, so Leti cannot read market data. Add one "
            "under Connections (the 'trading' section) - a free paper account is enough.",
            kind="unconfigured")

    now = _time.time()
    found = _cached(wanted, now) if use_cache else {}
    missing = [s for s in wanted if s not in found]
    if not missing:
        return found

    feed = stock_feed()
    stocks = [s for s in missing
              if not _is_crypto(s) and not is_option(s)][:MAX_SYMBOLS_PER_REQUEST]
    crypto = [s for s in missing if _is_crypto(s)][:MAX_SYMBOLS_PER_REQUEST]
    fetched: Dict[str, Any] = {}

    if stocks:
        try:
            data = await _get(PAPER_DATA_BASE, "/v2/stocks/snapshots",
                              {"symbols": ",".join(stocks), "feed": feed})
        except Exception as e:
            _raise_for(e, ", ".join(stocks))
        raw = data.get("snapshots", data) if isinstance(data, dict) else {}
        for symbol in stocks:
            entry = raw.get(symbol)
            if isinstance(entry, dict):
                fetched[symbol] = _snapshot_from(symbol, entry, feed)

    if crypto:
        try:
            data = await _get(PAPER_DATA_BASE, "/v1beta3/crypto/us/snapshots",
                              {"symbols": ",".join(crypto)})
        except Exception as e:
            _raise_for(e, ", ".join(crypto))
        raw = (data or {}).get("snapshots", {})
        for symbol in crypto:
            entry = raw.get(symbol)
            if isinstance(entry, dict):
                fetched[symbol] = _snapshot_from(symbol, entry, "crypto")

    _remember(fetched, now)
    found.update(fetched)
    unknown = [s for s in wanted if s not in found]
    if unknown and not found:
        raise MarketDataError(
            f"No market data came back for {', '.join(unknown)}. Check the symbol"
            + (" - crypto pairs look like BTC/USD." if any("/" not in u for u in unknown) else "."),
            kind="unknown_symbol", symbol=", ".join(unknown))
    return found


async def get_bars(symbol: str, timeframe: str = "1Day", limit: int = 30) -> List[Dict[str, Any]]:
    """Historical OHLCV for one symbol, oldest first."""
    if not have_credentials():
        raise MarketDataError(
            "No Alpaca API key is configured, so Leti cannot read market history.",
            kind="unconfigured", symbol=symbol)
    symbol = symbol.strip().upper()
    try:
        if _is_crypto(symbol):
            data = await _get(PAPER_DATA_BASE, "/v1beta3/crypto/us/bars",
                              {"symbols": symbol, "timeframe": timeframe, "limit": limit})
            raw = (data.get("bars") or {}).get(symbol, [])
        else:
            data = await _get(PAPER_DATA_BASE, f"/v2/stocks/{symbol}/bars",
                              {"timeframe": timeframe, "limit": limit, "feed": stock_feed()})
            raw = data.get("bars") or []
    except Exception as e:
        _raise_for(e, symbol)
    bars = [b for b in (_bar(item) for item in raw) if b]
    if not bars:
        raise MarketDataError(
            f"No {timeframe} history came back for {symbol}. Newly listed symbols and "
            "closed markets can both look like this.", kind="no_data", symbol=symbol)
    return bars


# An OCC option symbol: underlying, expiry, C or P, then the strike - AAPL241220C00150000.
OPTION_SYMBOL = _re.compile(r"^(?P<underlying>[A-Z]{1,6})\d{6}[CP]\d{8}$")


def is_option(symbol: str) -> bool:
    return bool(OPTION_SYMBOL.match(str(symbol or "").strip().upper()))


def option_underlying(symbol: str) -> str:
    match = OPTION_SYMBOL.match(str(symbol or "").strip().upper())
    return match.group("underlying") if match else ""


async def get_option_snapshot(underlying: str) -> Dict[str, Any]:
    """Options chain for an underlying - only if the account is subscribed.

    Not wrapped in a "maybe it works" guess: if Alpaca says the account has no
    options entitlement, that refusal is what comes back, named as one.
    """
    if not have_credentials():
        raise MarketDataError("No Alpaca API key is configured.", kind="unconfigured")
    try:
        data = await _get(PAPER_DATA_BASE, f"/v1beta1/options/snapshots/{underlying.upper()}",
                          {"limit": 50})
    except Exception as e:
        _raise_for(e, underlying)
    snapshots = (data or {}).get("snapshots") or {}
    if not snapshots:
        raise MarketDataError(f"No options data came back for {underlying.upper()}.",
                              kind="no_data", symbol=underlying)
    return snapshots


class GetWatchlistTool(BaseTool):
    name = "get_trading_watchlist"
    description = "Show the symbols currently being monitored (stocks/crypto)."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={"watchlist": _watchlist()})


class SetWatchlistTool(BaseTool):
    name = "set_trading_watchlist"
    description = "Replace the list of symbols Leti monitors (e.g. ['AAPL','BTC/USD'])."
    parameters: List[ToolParameter] = [
        ToolParameter(name="symbols", type="array", items_type="string", description="Symbols to monitor."),
    ]

    async def run(self, symbols: List[str], **kwargs) -> ToolResult:
        # This used to return success while doing nothing but printing YAML for the
        # user to edit by hand - so the model reported the watchlist as changed when
        # it wasn't, against the system prompt's "never claim to have done something
        # you did not actually do". It now writes to config/settings.local.yaml, the
        # override file the /settings command already owns, which leaves the
        # commented settings.yaml template untouched and takes effect immediately.
        from core.config_loader import reload_settings
        from core.settings_editor import _load_overrides, _save_overrides, _deep_set

        cleaned = [str(s).strip().upper() for s in symbols if str(s).strip()]
        if not cleaned:
            return ToolResult(success=False, error="No symbols given.")

        try:
            overrides = _load_overrides()
            _deep_set(overrides, ["trading"], {"watchlist": cleaned})
            _save_overrides(overrides)
            reload_settings()
        except OSError as e:
            return ToolResult(success=False, error=f"Couldn't save the watchlist: {e}")

        return ToolResult(
            success=True,
            output={"watchlist": cleaned, "saved_to": "config/settings.local.yaml"},
        )


class GetQuoteTool(BaseTool):
    name = "get_market_quote"
    description = (
        "The price of a stock or crypto symbol right now, with today's change, volume and "
        "the day's range. Several symbols come back in one request. Says which data feed "
        "the figures came from, because a free Alpaca key reads IEX rather than the whole "
        "market. Read-only. For a standing 'tell me when it moves', use create_watch."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="symbols", type="array", items_type="string", description="Symbols to quote."),
    ]

    async def run(self, symbols: List[str], **kwargs) -> ToolResult:
        # An options symbol is a different endpoint and a different entitlement, so
        # it is separated before anything is fetched. Answering "no data for that"
        # would read as "that contract does not exist", which is not what a missing
        # OPRA subscription means.
        wanted = [s for s in (symbols or []) if str(s).strip()]
        options = [s for s in wanted if is_option(s)]
        chains: Dict[str, Any] = {}
        for contract in options[:4]:
            try:
                chains[contract.upper()] = await get_option_snapshot(option_underlying(contract))
            except MarketDataError as e:
                chains[contract.upper()] = {"unavailable": str(e), "problem": e.kind}

        ordinary = [s for s in wanted if not is_option(s)]
        try:
            snapshots = await get_snapshots(ordinary) if ordinary else {}
        except MarketDataError as e:
            if not chains:
                return ToolResult(success=False, error=str(e), output={"problem": e.kind})
            snapshots = {}
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        quotes = {symbol: {
            "price": snap.get("price"),
            "change": round(snap["change"], 4) if snap.get("change") is not None else None,
            "change_percent": (round(snap["change_percent"], 3)
                               if snap.get("change_percent") is not None else None),
            "volume": snap.get("volume"),
            "day": snap.get("daily_bar"),
            "timestamp": snap.get("as_of"),
            "feed": snap.get("feed"),
        } for symbol, snap in snapshots.items()}
        missing = [s.upper() for s in ordinary if s.upper() not in quotes]
        return ToolResult(success=True, output={
            "quotes": quotes,
            "options": chains or None,
            "no_data_for": missing,
            "feed": feed_report(),
        })


async def _get_closes(symbol: str, limit: int = 60, timeframe: str = "1Day") -> List[float]:
    """Closing prices, oldest first - the strategy tools' view of get_bars."""
    return [b["close"] for b in await get_bars(symbol, timeframe=timeframe, limit=limit)]


def _sma(closes: List[float], window: int) -> Optional[float]:
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def _rsi(closes: List[float], window: int = 14) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    gains, losses = [], []
    for i in range(-window, 0):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


class EvaluateStrategyTool(BaseTool):
    name = "evaluate_trading_strategy"
    description = (
        "Run a simple technical strategy (SMA crossover or RSI threshold) against recent price "
        "history for a symbol and return a buy/sell/hold signal. Read-only - does not place any "
        "order. Not financial advice; these are basic, well-known technical rules for a paper account."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="symbol", type="string", description="Symbol to evaluate."),
        ToolParameter(
            name="strategy", type="string", enum=["sma_crossover", "rsi_threshold"],
            description="Which strategy to apply.",
        ),
        ToolParameter(name="short_window", type="number", required=False, description="Short SMA window (default 10)."),
        ToolParameter(name="long_window", type="number", required=False, description="Long SMA window (default 30)."),
        ToolParameter(name="rsi_window", type="number", required=False, description="RSI window (default 14)."),
        ToolParameter(name="oversold", type="number", required=False, description="RSI oversold threshold (default 30)."),
        ToolParameter(name="overbought", type="number", required=False, description="RSI overbought threshold (default 70)."),
    ]

    async def run(
        self, symbol: str, strategy: str,
        short_window: int = 10, long_window: int = 30, rsi_window: int = 14,
        oversold: int = 30, overbought: int = 70, **kwargs
    ) -> ToolResult:
        try:
            closes = await _get_closes(symbol, limit=max(long_window, rsi_window) + 5)
            if not closes:
                return ToolResult(success=False, error=f"No price history returned for {symbol}.")

            if strategy == "sma_crossover":
                short = _sma(closes, short_window)
                long_ = _sma(closes, long_window)
                if short is None or long_ is None:
                    return ToolResult(success=False, error="Not enough history for requested SMA windows.")
                signal = "buy" if short > long_ else "sell" if short < long_ else "hold"
                return ToolResult(success=True, output={
                    "symbol": symbol, "strategy": strategy, "signal": signal,
                    "short_sma": round(short, 4), "long_sma": round(long_, 4),
                })

            elif strategy == "rsi_threshold":
                rsi = _rsi(closes, rsi_window)
                if rsi is None:
                    return ToolResult(success=False, error="Not enough history for requested RSI window.")
                signal = "buy" if rsi < oversold else "sell" if rsi > overbought else "hold"
                return ToolResult(success=True, output={
                    "symbol": symbol, "strategy": strategy, "signal": signal, "rsi": round(rsi, 2),
                })

            return ToolResult(success=False, error=f"Unknown strategy: {strategy}")
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"Market data API error: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetPositionsTool(BaseTool):
    name = "get_paper_positions"
    description = "List current open positions in the Alpaca PAPER (simulated) account."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        try:
            data = await _get(PAPER_TRADING_BASE, "/v2/positions")
            return ToolResult(success=True, output=data)
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"Trading API error: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class PlacePaperOrderTool(BaseTool):
    name = "place_paper_order"
    description = (
        "Place a simulated (paper account, no real money) market order. Risky: requires "
        "confirmation. This can NEVER touch a live trading account - the endpoint is hardcoded "
        "to Alpaca's paper API regardless of any other configuration."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="symbol", type="string", description="Symbol to trade."),
        ToolParameter(name="qty", type="number", description="Quantity of shares/coins."),
        ToolParameter(name="side", type="string", enum=["buy", "sell"], description="Order side."),
    ]

    async def run(self, symbol: str, qty: float, side: str, **kwargs) -> ToolResult:
        try:
            order = {
                "symbol": symbol, "qty": str(qty), "side": side,
                "type": "market", "time_in_force": "gtc",
            }
            data = await _post(PAPER_TRADING_BASE, "/v2/orders", order)
            return ToolResult(success=True, output={
                "note": "Executed in PAPER (simulated) account only - no real funds involved.",
                "order": data,
            })
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"Trading API error: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
