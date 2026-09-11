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

import logging
from typing import Any, Dict, List, Optional

from core import task_manager, watches
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.watches")

_RUNNER: Optional[Any] = None


def set_runner(runner) -> None:
    global _RUNNER
    _RUNNER = runner


class CreateWatchTool(BaseTool):
    name = "create_watch"
    description = (
        "Watch something and tell the user when it happens. Use for 'tell me if my CPU "
        "stays above 90% for 5 minutes', 'let me know when this file changes', 'tell me "
        "when this page changes', 'tell me when the site comes back up'. Leti checks on "
        "a schedule and reports the moment the condition BECOMES true, once - not every "
        "time it looks. Condition types: cpu_above (percent, optional for_minutes), "
        "file_changed (path), url_changed (url), url_available (url), email_from (sender - "
        "needs the mail settings the email tools use), price_below and price_above "
        "(symbol and price - need the market data key the trading tools use). Leti "
        "CANNOT watch a calendar: it can create events but has no tool that reads one "
        "back, so say that plainly rather than pretending to watch it."
    )
    parameters = [
        ToolParameter(name="name", type="string", description="Short name for the watch."),
        ToolParameter(name="condition_type", type="string",
                      description="cpu_above, file_changed, url_changed, or url_available.",
                      enum=list(watches.CONDITION_TYPES)),
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
        ToolParameter(name="symbol", type="string", required=False,
                      description="For price_below / price_above: e.g. AAPL or BTC/USD."),
        ToolParameter(name="price", type="number", required=False,
                      description="For price_below / price_above: the threshold."),
        ToolParameter(name="action", type="string", required=False,
                      description="notify (default), run_workflow, or start_task.",
                      enum=list(watches.ACTIONS)),
        ToolParameter(name="action_target", type="string", required=False,
                      description="Workflow id, or for start_task what the task should do."),
        ToolParameter(name="interval_minutes", type="number", required=False,
                      description="How often to check. Default 5."),
        ToolParameter(name="project", type="string", required=False,
                      description="Project this belongs to, if any."),
    ]

    async def run(self, name: str, condition_type: str, percent: float = 90,
                  for_minutes: float = 0, path: str = "", url: str = "",
                  sender: str = "", symbol: str = "", price: float = 0,
                  action: str = "notify", action_target: str = "",
                  interval_minutes: float = watches.DEFAULT_INTERVAL_MINUTES,
                  project: str = "", **kwargs) -> ToolResult:
        condition: Dict[str, Any] = {}
        if condition_type == "cpu_above":
            condition = {"percent": percent, "for_minutes": for_minutes}
        elif condition_type == "file_changed":
            condition = {"path": path}
        elif condition_type in ("url_changed", "url_available"):
            condition = {"url": url}
        elif condition_type == "email_from":
            condition = {"sender": sender}
        elif condition_type in ("price_below", "price_above"):
            condition = {"symbol": symbol, "price": price}

        watch = watches.create(name, condition_type, condition, action or "notify",
                               action_target, interval_minutes, project=project)
        problems = watches.validate(watch)
        if problems:
            return ToolResult(success=False, error="That watch isn't valid.",
                              output={"problems": problems})

        watches.save_new(watch)
        described = watches.describe(watch)
        return ToolResult(success=True, output={
            "watch": described,
            "note": (f"Watching {described['watching']}; when it happens Leti will "
                     f"{described['action']}. Checked every {described['every']}."),
        })


class ListWatchesTool(BaseTool):
    name = "list_watches"
    description = (
        "Show what Leti is watching: each condition, what it will do, whether it is on, "
        "when it was last checked and last triggered. Use for 'what are you watching', "
        "or before disabling or deleting one."
    )
    parameters = []

    async def run(self, **kwargs) -> ToolResult:
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
        "Turn a watch on or off, or remove it. Find its id with list_watches first."
    )
    parameters = [
        ToolParameter(name="watch_id", type="string", description="Which watch."),
        ToolParameter(name="action", type="string", description="enable, disable, or remove.",
                      enum=["enable", "disable", "delete"]),
    ]

    async def run(self, watch_id: str, action: str, **kwargs) -> ToolResult:
        action = str(action or "").lower()
        if action == "delete":
            if not watches.delete(watch_id):
                return ToolResult(success=False, error=f"No watch with id '{watch_id}'.")
            watches.ensure_scheduled()
            return ToolResult(success=True, output={"deleted": watch_id})

        if action not in ("enable", "disable"):
            return ToolResult(success=False, error=f"'{action}' is not enable, disable or delete.")
        watch = watches.set_enabled(watch_id, action == "enable")
        if watch is None:
            return ToolResult(success=False, error=f"No watch with id '{watch_id}'.")
        watches.ensure_scheduled()
        return ToolResult(success=True, output={"watch": watches.describe(watch)})


class CheckWatchesTool(BaseTool):
    name = "check_watches"
    description = (
        "Evaluate every watch that is due and report which conditions have just become "
        "true. Leti's scheduler calls this on its own; call it directly only if the user "
        "asks whether anything has happened yet. A watch that was already true last time "
        "is not reported again."
    )
    parameters = []

    async def run(self, **kwargs) -> ToolResult:
        triggered, checked, errors = [], 0, []
        for watch in watches.load_watches():
            if not watches.due(watch):
                continue
            checked += 1
            outcome = watches.check(watch["id"])
            if outcome.get("outcome") == "triggered":
                fired = outcome["watch"]
                triggered.append({
                    "name": fired["name"],
                    "condition": watches.describe_condition(fired),
                    "action": watches.describe_action(fired),
                    "action_type": fired.get("action"),
                    "action_target": fired.get("action_target"),
                    "project": fired.get("project"),
                })
            elif outcome.get("outcome") == "error":
                errors.append({"watch": watch["name"], "error": outcome.get("error")})

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

        return ToolResult(success=True, output={
            "checked": checked,
            "triggered": triggered,
            "tasks_started": started,
            "errors": errors,
            "note": ("Tell the user about anything in 'triggered'."
                     if triggered else "Nothing to report."),
        })
