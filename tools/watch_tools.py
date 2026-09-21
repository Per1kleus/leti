"""Tools for watching something and acting when it changes.

Not the same thing as the social watches in tools/social_media.py, despite the
shared word. A social watch is a content feed with a cursor: it remembers the
last item id it saw for a channel, subreddit or page, and answers "what is new
since I last looked" when something asks it. It is pull-based and it never acts.

A watch here is a condition plus an action: CPU above a threshold, a file
changed, a URL back up. It is evaluated on Leti's scheduler, it fires on the
transition into true rather than for as long as the condition holds, and what it
does when it fires goes through the task manager - and therefore the
orchestrator and SafetyGuard. Neither one can be expressed as the other, so they
stay separate; see core/watches.py for the condition machinery.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from core import signals, task_manager, watches
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.watches")

_RUNNER: Optional[Any] = None


def set_runner(runner) -> None:
    global _RUNNER
    _RUNNER = runner


class CreateWatchTool(BaseTool):
    name = "create_watch"
    description = (
        "Watch something and tell the user when it happens - a stock or crypto symbol, a "
        "topic in the news, how fast attention around something is growing, or this "
        "machine. Use for 'watch TTWO and tell me if it moves more than 5%', 'tell me if "
        "Bitcoin drops 5%', 'watch this stock for a volume spike', 'watch the news about "
        "this game and tell me when something big happens', 'tell me when the hype starts "
        "accelerating', 'tell me if my CPU stays above 90%', 'tell me when this page "
        "changes'. Leti checks on its own schedule and reports the moment something "
        "BECOMES true, once - not every time it looks.\n"
        "market needs symbol, metric, comparison and threshold - a volume spike is "
        "volume_ratio above 2, and 'moves more than 5%' is change_percent abs_above 5. "
        "news_event needs a topic and how strong the reporting must be. trend needs a "
        "topic and/or symbol and measures whether attention is accelerating, which is "
        "always an estimate, never a prediction. also_watch combines two of these in one "
        "watch: 'moves unusually AND there is major news', or combine=at_least with "
        "minimum=2 for 'when at least two of these three signals move'. Ask which "
        "symbol or topic if the user did not say. Leti CANNOT watch a calendar."
    )
    parameters = [
        ToolParameter(name="name", type="string", description="Short name for the watch."),
        ToolParameter(name="condition_type", type="string",
                      description="What kind of thing to watch.",
                      enum=list(watches.OFFERED_CONDITION_TYPES)),
        ToolParameter(name="symbol", type="string", required=False,
                      description="For market/trend: e.g. TTWO, AAPL, BTC/USD."),
        ToolParameter(name="metric", type="string", required=False,
                      description="For market: which measurement (default change_percent).",
                      enum=list(signals.MARKET_METRICS)),
        ToolParameter(name="comparison", type="string", required=False,
                      description="For market: above, below, abs_above ('either direction').",
                      enum=list(signals.COMPARISONS)),
        ToolParameter(name="threshold", type="number", required=False,
                      description="For market: the number the metric is compared against."),
        ToolParameter(name="window", type="number", required=False,
                      description="For market/trend: days of history to measure against "
                                  "(default 20)."),
        ToolParameter(name="topic", type="string", required=False,
                      description="For news_event/trend: the subject, e.g. 'GTA VI' or "
                                  "'the Rockstar leak investigation'."),
        ToolParameter(name="min_sources", type="number", required=False,
                      description="For news_event: independent sources needed before it "
                                  "counts. Default 1; use 2+ for anything serious."),
        ToolParameter(name="require_confirmation", type="boolean", required=False,
                      description="For news_event: only report events carried by major or "
                                  "official sources. Use it for arrests, charges, deaths or "
                                  "official decisions - anywhere being wrong matters."),
        ToolParameter(name="official_domains", type="array", items_type="string", required=False,
                      description="For news_event: domains that count as official, e.g. "
                                  "['rockstargames.com']."),
        ToolParameter(name="also_watch", type="array", items_type="string", required=False,
                      description="Other signals to fold in: news, trend, market.",
                      enum=["news", "trend", "market"]),
        ToolParameter(name="combine", type="string", required=False,
                      description="With also_watch: all (default), any, or at_least.",
                      enum=list(watches.COMBINE_MODES)),
        ToolParameter(name="minimum", type="number", required=False,
                      description="With combine=at_least: how many must be true, e.g. 2."),
        ToolParameter(name="percent", type="number", required=False,
                      description="For cpu_above: the threshold."),
        ToolParameter(name="for_minutes", type="number", required=False,
                      description="For cpu_above: how long it must stay above before it counts."),
        ToolParameter(name="path", type="string", required=False,
                      description="For file_changed: the file to watch."),
        ToolParameter(name="url", type="string", required=False,
                      description="For url_changed / url_available: the address."),
        ToolParameter(name="sender", type="string", required=False,
                      description="For email_from: the address or name to look for."),
        ToolParameter(name="action", type="string", required=False,
                      description="notify (default), run_workflow, or start_task.",
                      enum=list(watches.ACTIONS)),
        ToolParameter(name="action_target", type="string", required=False,
                      description="Workflow id, or for start_task what the task should do."),
        ToolParameter(name="interval_minutes", type="number", required=False,
                      description="How often to check. Sensible defaults per kind."),
        ToolParameter(name="expires_in_days", type="number", required=False,
                      description="Stop after this many days - only when the user gave a "
                                  "date or a period. Never invent one."),
        ToolParameter(name="reason", type="string", required=False,
                      description="Why the user wants this, in their words."),
        ToolParameter(name="project", type="string", required=False,
                      description="Project this belongs to, if any."),
    ]

    async def run(self, name: str, condition_type: str, percent: float = 90,
                  for_minutes: float = 0, path: str = "", url: str = "",
                  sender: str = "", symbol: str = "", metric: str = "",
                  comparison: str = "", threshold: Any = None, window: Any = None,
                  topic: str = "", min_sources: Any = None,
                  require_confirmation: bool = False, official_domains: Any = None,
                  also_watch: Any = None, combine: str = "all", minimum: Any = None,
                  action: str = "notify", action_target: str = "",
                  interval_minutes: Any = None, expires_in_days: Any = None,
                  reason: str = "", project: str = "", **kwargs) -> ToolResult:
        arguments = dict(percent=percent, for_minutes=for_minutes, path=path, url=url,
                         sender=sender, symbol=symbol, metric=metric,
                         comparison=comparison, threshold=threshold, window=window,
                         topic=topic, min_sources=min_sources,
                         require_confirmation=require_confirmation,
                         official_domains=official_domains)
        missing = _what_is_missing(condition_type, arguments)
        if missing:
            return ToolResult(success=False, error=missing)

        condition = _condition_for(condition_type, arguments)
        extra = [str(s).strip().lower() for s in _as_list(also_watch) if str(s).strip()]
        if extra:
            parts = [{"condition_type": condition_type, "condition": condition}]
            for wanted in extra:
                kind = {"news": "news_event", "trend": "trend", "market": "market"}.get(wanted)
                if kind is None:
                    return ToolResult(success=False, error=(
                        f"'{wanted}' is not a signal to combine. Use news, trend or market."))
                if any(p["condition_type"] == kind for p in parts):
                    return ToolResult(success=False, error=(
                        f"also_watch lists '{wanted}' twice, or the watch is already that "
                        "kind. Combine different kinds of signal."))
                missing = _what_is_missing(kind, arguments)
                if missing:
                    return ToolResult(success=False, error=missing)
                parts.append({"condition_type": kind,
                              "condition": _condition_for(kind, arguments)})
            mode = (combine or "all").lower()
            condition = {"mode": mode, "parts": parts[:watches.MAX_COMBINED_PARTS]}
            if mode == "at_least":
                condition["minimum"] = int(_number(minimum, 2))
            condition_type = "combined"

        every = float(interval_minutes) if interval_minutes else watches.DEFAULT_INTERVALS.get(
            condition_type, watches.DEFAULT_INTERVAL_MINUTES)
        expires_at = None
        if expires_in_days:
            try:
                expires_at = time.time() + float(expires_in_days) * 86_400
            except (TypeError, ValueError):
                return ToolResult(success=False, error="expires_in_days has to be a number.")

        watch = watches.create(name, condition_type, condition, action or "notify",
                               action_target, every, project=project, expires_at=expires_at)
        if reason:
            watch["reason"] = str(reason)[:300]
        problems = watches.validate(watch)
        if problems:
            return ToolResult(success=False, error="That watch isn't valid.",
                              output={"problems": problems})

        watches.save_new(watch)
        described = watches.describe(watch, detail=True)
        return ToolResult(success=True, output={
            "watch": described,
            "note": (f"Watching {described['watching']}; when it happens Leti will "
                     f"{described['action']}. Checked every {described['every']}."),
            "warning": _setup_warning(watch),
        })


def _as_list(value) -> List[Any]:
    """One signal or several. The model sends either, and both mean the same."""
    if value is None or value == "":
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _number(value, fallback):
    try:
        return float(value) if value is not None and value != "" else fallback
    except (TypeError, ValueError):
        return fallback


def _condition_for(kind: str, a: Dict[str, Any]) -> Dict[str, Any]:
    """The structured condition behind a sentence like 'if TTWO moves 5%'."""
    if kind == "cpu_above":
        return {"percent": a["percent"], "for_minutes": a["for_minutes"]}
    if kind == "file_changed":
        return {"path": a["path"]}
    if kind in ("url_changed", "url_available"):
        return {"url": a["url"]}
    if kind == "email_from":
        return {"sender": a["sender"]}
    if kind == "market":
        return {"symbol": str(a["symbol"]).upper(),
                "metric": a["metric"] or "change_percent",
                # A bare "tell me if it moves more than 5%" means either direction;
                # that is what people mean and the wrong guess here is a watch that
                # stays silent through a crash.
                "comparison": a["comparison"] or ("abs_above" if (a["metric"] or
                                                  "change_percent").endswith("_percent")
                                                  else "above"),
                "threshold": _number(a["threshold"], 5.0),
                "window": int(_number(a["window"], signals.DEFAULT_WINDOW_DAYS))}
    if kind == "news_event":
        return {"topic": a["topic"],
                "min_sources": int(_number(a["min_sources"], 1)),
                "require_confirmation": bool(a["require_confirmation"]),
                "official_domains": [str(d) for d in (a["official_domains"] or [])]}
    if kind == "trend":
        return {"topic": a["topic"], "symbol": str(a["symbol"] or "").upper(),
                "window": int(_number(a["window"], signals.DEFAULT_WINDOW_DAYS)),
                "official_domains": [str(d) for d in (a["official_domains"] or [])]}
    return {}


def _what_is_missing(kind: str, a: Dict[str, Any]) -> str:
    """The question to ask back, rather than a watch on a guess.

    "Watch that stock" has no symbol in it, and inventing one is worse than asking.
    """
    if kind == "market" and not str(a["symbol"]).strip():
        return "Which symbol should Leti watch? Say the ticker, e.g. TTWO or BTC/USD."
    if kind == "news_event" and not str(a["topic"]).strip():
        return "What topic should Leti follow? Name the company, person, game or story."
    if kind == "trend" and not (str(a["topic"]).strip() or str(a["symbol"]).strip()):
        return "What should Leti measure attention around - a topic, a symbol, or both?"
    if kind == "file_changed" and not str(a["path"]).strip():
        return "Which file should Leti watch?"
    if kind in ("url_changed", "url_available") and not str(a["url"]).strip():
        return "Which page should Leti watch?"
    if kind == "email_from" and not str(a["sender"]).strip():
        return "Whose mail should Leti look for?"
    return ""


def _setup_warning(watch: Dict[str, Any]) -> Optional[str]:
    """Whether this watch can actually run, said now rather than after five failures."""
    kinds = {watch.get("condition_type")}
    kinds.update(p.get("condition_type") for p in
                 (watch.get("condition") or {}).get("parts") or [])
    if not ({"market", "trend", "price_below", "price_above"} & kinds):
        return None
    try:
        from tools.trading_platform import feed_report, have_credentials
    except Exception:
        return None
    if not have_credentials():
        return ("This watch needs market data and no Alpaca API key is configured, so it "
                "will fail until one is added under Connections (the 'trading' section). "
                "A free paper account is enough.")
    report = feed_report()
    if report.get("stock_feed") == "iex" and "/" not in str(
            (watch.get("condition") or {}).get("symbol", "")):
        return ("Prices come from Alpaca's IEX feed, which is one exchange rather than "
                "the whole US market - figures can differ a little from what a broker "
                "shows. SIP needs a paid Alpaca subscription.")
    return None


class ListWatchesTool(BaseTool):
    name = "list_watches"
    description = (
        "Show what Leti is watching: each condition, what it will do, whether it is on, "
        "when it was last checked, when it last triggered and when it next will. Use for "
        "'what are you watching', or before changing or removing one. Give a watch_id to "
        "see one watch in full - what it is measuring right now, why it exists, and why "
        "it fired the last time it did."
    )
    parameters = [
        ToolParameter(name="watch_id", type="string", required=False,
                      description="One watch, in full. Leave blank for all of them."),
    ]

    async def run(self, watch_id: str = "", **kwargs) -> ToolResult:
        if watch_id:
            watch = watches.get_watch(watch_id)
            if watch is None:
                return ToolResult(success=False, error=f"No watch with id '{watch_id}'.")
            return ToolResult(success=True, output={"watch": watches.describe(watch, detail=True)})

        all_watches = [watches.describe(w) for w in watches.load_watches()]
        return ToolResult(success=True, output={
            "count": len(all_watches),
            "active": [w for w in all_watches if w["enabled"]],
            "paused": [w for w in all_watches if not w["enabled"]],
        })


class ManageWatchTool(BaseTool):
    name = "manage_watch"
    description = (
        # "remove" rather than "delete" on purpose: "delete" is the word that
        # distinguishes the file tools, and every extra tool that claims it makes
        # "delete the old backup folder" a little less obviously about files.
        "Pause a watch, resume it, remove it, or change it - how often it checks, how "
        "long it waits between reports, its threshold or topic, when it should stop, "
        "and what it does when it fires. Changing keeps everything the watch has already "
        "seen, so raising a threshold does not make it re-report old news. Find the id "
        "with list_watches first."
    )
    parameters = [
        ToolParameter(name="watch_id", type="string", description="Which watch."),
        ToolParameter(name="action", type="string",
                      description="enable, disable, delete, or update.",
                      enum=["enable", "disable", "delete", "update"]),
        ToolParameter(name="interval_minutes", type="number", required=False,
                      description="For update: how often to check."),
        ToolParameter(name="cooldown_minutes", type="number", required=False,
                      description="For update: the quiet period after it reports."),
        ToolParameter(name="threshold", type="number", required=False,
                      description="For update: a market watch's number."),
        ToolParameter(name="topic", type="string", required=False,
                      description="For update: a news or trend watch's subject."),
        ToolParameter(name="expires_in_days", type="number", required=False,
                      description="For update: stop watching after this many days from now."),
        ToolParameter(name="notify_action", type="string", required=False,
                      description="For update: what it does when it fires.",
                      enum=list(watches.ACTIONS)),
        ToolParameter(name="action_target", type="string", required=False,
                      description="For update: the workflow id or what the task should do."),
        ToolParameter(name="name", type="string", required=False,
                      description="For update: rename it."),
    ]

    async def run(self, watch_id: str, action: str, interval_minutes: Any = None,
                  cooldown_minutes: Any = None, threshold: Any = None, topic: str = "",
                  expires_in_days: Any = None, notify_action: str = "",
                  action_target: str = "", name: str = "", **kwargs) -> ToolResult:
        action = str(action or "").lower()
        if action == "delete":
            if not watches.delete(watch_id):
                return ToolResult(success=False, error=f"No watch with id '{watch_id}'.")
            watches.ensure_scheduled()
            return ToolResult(success=True, output={"deleted": watch_id})

        if action == "update":
            condition = {}
            if threshold is not None:
                condition["threshold"] = threshold
            if topic:
                condition["topic"] = topic
            changes: Dict[str, Any] = {
                "interval_minutes": interval_minutes,
                "cooldown_minutes": cooldown_minutes,
                "name": name or None,
                "action": notify_action or None,
                "action_target": action_target or None,
                "condition": condition or None,
            }
            if expires_in_days is not None:
                try:
                    changes["expires_at"] = time.time() + float(expires_in_days) * 86_400
                except (TypeError, ValueError):
                    return ToolResult(success=False, error="expires_in_days has to be a number.")
            updated, problems = watches.update(watch_id, changes)
            if problems:
                return ToolResult(success=False, error="That change isn't valid.",
                                  output={"problems": problems})
            return ToolResult(success=True, output={"watch": watches.describe(updated, detail=True)})

        if action not in ("enable", "disable"):
            return ToolResult(success=False,
                              error=f"'{action}' is not enable, disable, delete or update.")
        watch = watches.set_enabled(watch_id, action == "enable")
        if watch is None:
            return ToolResult(success=False, error=f"No watch with id '{watch_id}'.")
        watches.ensure_scheduled()
        return ToolResult(success=True, output={"watch": watches.describe(watch)})


class CheckWatchesTool(BaseTool):
    name = "check_watches"
    description = (
        "Evaluate every watch that is due and report which conditions have just become "
        "true, with the numbers and sources behind each one. Leti's scheduler calls this "
        "on its own; call it directly only if the user asks whether anything has happened "
        "yet. A watch that was already true last time is not reported again."
    )
    parameters = []

    async def run(self, **kwargs) -> ToolResult:
        # Evaluating means waiting on somebody else's API, and the interface is on
        # this event loop. So the whole sweep happens in one worker thread - one
        # for the sweep, not one per watch - and the loop stays responsive while a
        # search takes its three seconds. It also means the provider calls inside
        # find no running loop and need no thread of their own.
        checked, triggered, errors, expired = await asyncio.get_running_loop().run_in_executor(
            None, self._evaluate_due)

        # Anything a triggered watch wants DONE goes back through the ordinary
        # route: a task, run by the orchestrator, authorised call by call. Seeing a
        # condition is not permission to act on it.
        started = []
        for item in triggered:
            if item["action_type"] == "start_task" and item["action_target"]:
                task = task_manager.create_task(
                    objective=item["action_target"],
                    steps=[item["action_target"]],
                    name=f"From watch: {item['name']}")
                if _RUNNER is not None:
                    _RUNNER.start_in_background(task["id"])
                started.append(task_manager.describe(task))
                # What the watch's action actually did, written back onto the
                # watch - see core/watches.py's verify_action. Started is not
                # finished, and the record says so.
                watches.record_action(item.get("watch_id") or item.get("id"),
                                      {"task_id": task["id"]})
            elif item["action_type"] == "notify":
                watches.record_action(item.get("watch_id") or item.get("id"), {})

        return ToolResult(success=True, output={
            "checked": checked,
            "triggered": triggered,
            "tasks_started": started,
            "errors": errors,
            "expired": expired,
            "how_to_report": (
                "Say what happened, the number or the source that made it happen, and "
                "when. Market figures are the feed's, not the whole market's. News "
                "events are what the named sources reported - keep the support level "
                "('confirmed', 'reported', 'unverified') as it is given. Trend readings "
                "are estimates from the signals listed; give the confidence with them "
                "and never turn one into a prediction with a date on it."
                if triggered else "Nothing to report."),
        })

    def _evaluate_due(self):
        """Every watch that is due, evaluated once. Runs off the event loop."""
        expired = [w["name"] for w in watches.expire_due_watches()]
        pending = [w for w in watches.load_watches() if watches.due(w)]
        # One market request for every symbol about to be looked at, before any of
        # them is evaluated. Without this, five watches on TTWO are five requests.
        watches.prefetch(pending)

        triggered, checked, errors = [], 0, []
        for watch in pending:
            checked += 1
            outcome = watches.check(watch["id"])
            if outcome.get("outcome") == "triggered":
                fired = outcome["watch"]
                last = (fired.get("history") or [{}])[-1]
                triggered.append({
                    "watch_id": fired.get("id"),
                    "name": fired["name"],
                    "condition": watches.describe_condition(fired),
                    "why": last.get("why"),
                    "evidence": last.get("evidence"),
                    "source": watches.provider(fired),
                    "at": last.get("at"),
                    "action": watches.describe_action(fired),
                    "action_type": fired.get("action"),
                    "action_target": fired.get("action_target"),
                    "project": fired.get("project"),
                })
            elif outcome.get("outcome") == "error":
                errors.append({"watch": watch["name"], "error": outcome.get("error"),
                               "still_watching": not outcome.get("disabled")})
        return checked, triggered, errors, expired
