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
        # Note: this only updates the in-memory/cached settings object via config file rewrite
        # is out of scope here to avoid clobbering the YAML file's comments/formatting; instead
        # this tool reports what to persist and the orchestrator/user should confirm the edit
        # via write_file on config/settings.yaml, which already goes through SafetyGuard.
        return ToolResult(
            success=True,
            output=(
                f"To persist this watchlist, update the 'trading.watchlist' key in "
                f"config/settings.yaml to: {symbols}"
            ),
        )


class GetQuoteTool(BaseTool):
    name = "get_market_quote"
    description = "Get the latest price for one or more stock/crypto symbols from the paper market data feed."
    parameters: List[ToolParameter] = [
        ToolParameter(name="symbols", type="array", items_type="string", description="Symbols to quote."),
    ]

    async def run(self, symbols: List[str], **kwargs) -> ToolResult:
        try:
            quotes = {}
            for sym in symbols:
                if _is_crypto(sym):
                    data = await _get(PAPER_DATA_BASE, f"/v1beta3/crypto/us/latest/trades", {"symbols": sym})
                    trade = data.get("trades", {}).get(sym, {})
                    quotes[sym] = {"price": trade.get("p"), "timestamp": trade.get("t")}
                else:
                    data = await _get(PAPER_DATA_BASE, f"/v2/stocks/{sym}/trades/latest")
                    trade = data.get("trade", {})
                    quotes[sym] = {"price": trade.get("p"), "timestamp": trade.get("t")}
            return ToolResult(success=True, output=quotes)
        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, error=f"Market data API error: {e.response.status_code} {e.response.text[:200]}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


async def _get_closes(symbol: str, limit: int = 60, timeframe: str = "1Day") -> List[float]:
    if _is_crypto(symbol):
        data = await _get(
            PAPER_DATA_BASE, f"/v1beta3/crypto/us/bars",
            {"symbols": symbol, "timeframe": timeframe, "limit": limit},
        )
        bars = data.get("bars", {}).get(symbol, [])
    else:
        data = await _get(
            PAPER_DATA_BASE, f"/v2/stocks/{symbol}/bars",
            {"timeframe": timeframe, "limit": limit},
        )
        bars = data.get("bars", [])
    return [b["c"] for b in bars]


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
