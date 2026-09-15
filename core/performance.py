"""How much work this turn is worth, given what it is and what the machine has left.

Leti's expensive habits were fixed years of commits ago: tool routing already
stops the whole registry being sent, the activity log is pushed rather than
polled, the scheduler is one row, nothing captures the screen on its own, and the
watches evaluate in one sweep. What was missing is the thing that ties them
together - a single place that says "this turn is a one-line question, do not
spend a complex turn's budget on it".

That is all this is. It is not a scheduler, not a monitor, and not a loop: a
function, called once per turn, that reads three cheap facts and answers with a
budget. It costs one non-blocking psutil sample, which is the same sample the
diagnostics panel already takes, and two file reads that were happening anyway.

The rule it follows is the one that matters: fewer tools and less optional
context for a small request, everything for a large one, and under genuine
resource pressure the OPTIONAL work is shed - never a capability. Leti does not
stop being able to do something because the CPU is busy; it stops doing the
extras that were only ever a nicety, says so in the log, and does the actual job.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("leti.performance")

IDLE = "idle"
SIMPLE = "simple"
NORMAL = "normal"
COMPLEX = "complex"
COMPUTER_USE = "computer_use"
PRESSURE = "pressure"

# What "busy" means. Deliberately high: a machine at 70% is a machine doing work,
# which is the normal state of a computer somebody is using. Shedding optional
# work below this would make Leti worse for no reason.
CPU_PRESSURE = 90.0
RAM_PRESSURE = 92.0

# Tool budgets per mode. None means "whatever routing chose" - the router is
# already the thing that decides, and this only ever tightens its answer for the
# smallest requests. Never below core.tool_router.MIN_TOOLS, which that module
# enforces itself.
BUDGETS: Dict[str, Optional[int]] = {
    IDLE: None,
    SIMPLE: 12,
    NORMAL: None,
    COMPLEX: None,
    COMPUTER_USE: None,
    PRESSURE: 10,
}


@dataclass
class Mode:
    """What this turn may spend. Every field is advisory; nothing here refuses."""

    name: str = NORMAL
    tool_budget: Optional[int] = None
    recall_memory: bool = True          # long-term vector recall
    derive_visuals: bool = True         # artifact shapes from tool results
    reason: str = ""

    @property
    def under_pressure(self) -> bool:
        return self.name == PRESSURE


def _resources() -> Dict[str, float]:
    """CPU and memory, without blocking. Unknown reads as "not under pressure"."""
    try:
        import psutil

        # interval=None is the load since the last call: no sample window, no wait.
        return {"cpu": float(psutil.cpu_percent(interval=None)),
                "ram": float(psutil.virtual_memory().percent)}
    except Exception:
        return {"cpu": 0.0, "ram": 0.0}


def _work_in_flight() -> Dict[str, bool]:
    """Whether something long-running is already going, from the stores that
    already hold that state. No new bookkeeping and nothing to keep in sync."""
    busy = {"task": False, "computer_use": False}
    try:
        from core import task_manager

        busy["task"] = any(t.get("status") == task_manager.RUNNING
                           for t in task_manager.load_tasks())
    except Exception:
        pass
    try:
        from core import computer_use

        # An errand nobody has touched for the idle timeout is not in flight -
        # it is abandoned. Counting one would quietly put every later turn in
        # computer-use mode until something else opened a session.
        cutoff = time.time() - computer_use.SESSION_IDLE_TIMEOUT
        busy["computer_use"] = any(s.started_at >= cutoff
                                   for s in computer_use.active_sessions())
    except Exception:
        pass
    return busy


def for_turn(intent: Any = None, measure: bool = True) -> Mode:
    """The budget for one turn. Never raises: the fallback is the ordinary mode."""
    try:
        return _for_turn(intent, measure)
    except Exception as e:
        logger.debug(f"Adaptive mode failed ({e}); using the ordinary budget.")
        return Mode(reason="mode detection failed; nothing restricted")


def _for_turn(intent: Any, measure: bool) -> Mode:
    resources = _resources() if measure else {"cpu": 0.0, "ram": 0.0}
    pressured = resources["cpu"] >= CPU_PRESSURE or resources["ram"] >= RAM_PRESSURE

    complexity = getattr(intent, "complexity", None)
    shape = getattr(intent, "shape", "")
    busy = _work_in_flight()

    if busy["computer_use"]:
        name = COMPUTER_USE
    elif complexity == "complex" or busy["task"]:
        name = COMPLEX
    elif shape == "chat":
        name = IDLE
    elif complexity == "simple":
        name = SIMPLE
    else:
        name = NORMAL

    # Pressure narrows the budget but never the capability: the mode below still
    # routes, still runs every tool it chose, and still answers the question. What
    # it gives up is the two pieces of work that were optional to begin with.
    if pressured:
        budget = BUDGETS[PRESSURE] if name in (IDLE, SIMPLE, NORMAL) else BUDGETS[name]
        return Mode(name=PRESSURE, tool_budget=budget, recall_memory=False,
                    derive_visuals=False,
                    reason=(f"cpu {resources['cpu']:.0f}%, memory {resources['ram']:.0f}% - "
                            "optional recall and visuals skipped for this turn"))

    return Mode(name=name, tool_budget=BUDGETS[name],
                # A greeting needs no memory search: there is nothing to recall
                # against "hello", and the search is a vector query either way.
                recall_memory=name != IDLE,
                derive_visuals=name != IDLE,
                reason=f"{name} turn")
