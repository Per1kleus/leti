"""Tools for objectives that run on their own: start one, watch it, control it.

These are ordinary registered tools. The model reaches them the same way it
reaches every other tool - through the router, through the safety guard - and the
steps they run go back through the orchestrator, so nothing here is a side door
around the path a typed request takes.

The plan comes from the model, as arguments to start_autonomous_task. That is
deliberate: the reasoning model is already reading the request, so asking it for
a list of steps costs nothing extra, and once the steps are written down the
running of them is plain deterministic code.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core import task_manager
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.autonomous")

_RUNNER: Optional[Any] = None


def set_runner(runner) -> None:
    """main.py registers the live runner once the orchestrator exists."""
    global _RUNNER
    _RUNNER = runner


class StartAutonomousTaskTool(BaseTool):
    name = "start_autonomous_task"
    description = (
        "Start a long-running task that works through several steps on its own, for an "
        "objective too big for one reply - researching a list of companies, gathering "
        "data and writing it to a file, anything that is really a sequence of jobs. "
        "YOU write the plan: give the objective and an ordered list of plain-language "
        "steps, each one a single instruction you could have been asked directly. The "
        "task keeps its own state, survives a restart, and can be paused, resumed or "
        "cancelled. Steps that need permission you cannot give while working alone "
        "(sending mail, deleting things) stop the task and wait for the user. Use this "
        "instead of trying to do everything in one answer."
    )
    parameters = [
        ToolParameter(name="objective", type="string",
                      description="What the finished task should have achieved."),
        ToolParameter(name="steps", type="array", items_type="string",
                      description="Ordered plain-language steps, each a single instruction."),
        ToolParameter(name="name", type="string", required=False,
                      description="Short name for the task, e.g. 'Greek engineering research'."),
    ]

    async def run(self, objective: str, steps: List[str], name: str = "",
                  **kwargs) -> ToolResult:
        try:
            task = task_manager.create_task(objective, steps, name)
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

        started = False
        if _RUNNER is not None:
            started = _RUNNER.start_in_background(task["id"])
        return ToolResult(success=True, output={
            "task": task_manager.describe(task),
            "started": started,
            "note": ("Running now; ask for its progress at any time."
                     if started else
                     "Queued. It will run when a task runner is available."),
        })


class ListAutonomousTasksTool(BaseTool):
    name = "list_autonomous_tasks"
    description = (
        "Show long-running tasks and how far they have got: which step they are on, "
        "how many are done, whether one is paused, waiting for approval or finished. "
        "Use this for 'what are you working on', 'how is that going', or before "
        "pausing or cancelling something so you know which task is meant."
    )
    parameters = [
        ToolParameter(name="include_finished", type="boolean", required=False,
                      description="Also show completed, failed and cancelled tasks."),
    ]

    async def run(self, include_finished: bool = False, **kwargs) -> ToolResult:
        tasks = task_manager.load_tasks()
        if not include_finished:
            tasks = [t for t in tasks if t.get("status") in task_manager.ACTIVE_STATUSES]
        return ToolResult(success=True, output={
            "count": len(tasks),
            "tasks": [task_manager.describe(t) for t in tasks],
        })


class ControlAutonomousTaskTool(BaseTool):
    name = "control_autonomous_task"
    description = (
        "Pause, resume or cancel a long-running task, or approve one that stopped to "
        "ask permission. Name the task with a word or two from what it is doing ('the "
        "research task') or its id. If more than one task could be meant this says so "
        "and lists them rather than picking one - ask the user which."
    )
    parameters = [
        ToolParameter(name="action", type="string",
                      description="pause, resume, cancel, or approve.",
                      enum=["pause", "resume", "cancel", "approve"]),
        ToolParameter(name="which", type="string", required=False,
                      description="A word from the task's name, or its id. Blank if only one is active."),
    ]

    async def run(self, action: str, which: str = "", **kwargs) -> ToolResult:
        action = str(action or "").strip().lower()
        if action not in ("pause", "resume", "cancel", "approve"):
            return ToolResult(success=False,
                              error=f"'{action}' is not one of pause, resume, cancel, approve.")

        candidates = task_manager.find_active(which)
        if not candidates:
            return ToolResult(success=False, error="No task is currently active.")
        if len(candidates) > 1:
            # Never guess between two live tasks - ask.
            return ToolResult(success=False, error="More than one task could be meant.",
                              output={"ambiguous": True,
                                      "tasks": [task_manager.describe(t) for t in candidates],
                                      "note": "Ask the user which of these they mean."})

        task = candidates[0]
        task_id = task["id"]

        if action == "pause":
            updated = task_manager.pause(task_id)
            if updated is None:
                return ToolResult(success=False,
                                  error=f"'{task['name']}' is {task['status']}; it can't be paused.")
            return ToolResult(success=True, output={"task": task_manager.describe(updated)})

        if action == "cancel":
            updated = task_manager.cancel(task_id)
            if updated is None:
                return ToolResult(success=False, error=f"'{task['name']}' has already finished.")
            return ToolResult(success=True, output={"task": task_manager.describe(updated)})

        # resume and approve are the same move - the step that stopped is retried,
        # and the guard is asked again with the user present this time.
        updated = task_manager.resume(task_id)
        if updated is None:
            return ToolResult(success=False,
                              error=f"'{task['name']}' is {task['status']}; there is nothing to resume.")
        started = _RUNNER.start_in_background(task_id) if _RUNNER is not None else False
        return ToolResult(success=True, output={
            "task": task_manager.describe(updated),
            "started": started,
            "note": ("Resumed." if started else "Marked ready to resume."),
        })
