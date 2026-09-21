"""Holding on to what the user is actually trying to achieve.

A request and a goal are different things. "What's the weather" is a request:
it starts, it finishes, nothing survives it. "Get the Q3 pipeline to 200k" is a
goal: it outlives the turn, it has work attached to it, and the useful question
next Tuesday is not "what did you ask" but "where is it and what is next".

Leti already has every part of that. core/business_goals.py holds goals and their
honest progress basis. core/task_manager.py holds multi-step work, its plan, what
each step did and what is next. tools/projects.py holds the durable workspace.
core/workflows.py holds the recurring version. What was missing was the shape
that ties them together for a person reading it:

    GOAL -> PLAN -> ACTIONS -> RESULTS -> PROGRESS -> NEXT ACTION

That is what this module produces. It owns no store, creates nothing, and
persists nothing: every line below is read out of a store that already had it.

THE GATE. Most requests are not goals, and treating them as goals is its own
failure - it buries a one-line answer under a plan, a checkpoint and a progress
report. worth_tracking() below is deliberately hard to satisfy: several distinct
stages, or an explicit ask to keep going. "Write me an email" does not qualify
and must not.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core import intent as intent_reader

logger = logging.getLogger("leti.objectives")

# What a goal's progress rests on - core/business_goals.py's vocabulary, reused
# so "40%" means the same thing wherever it is printed.
MEASURED = "measured"
BY_TASKS = "by_tasks"
UNMEASURABLE = "unmeasurable"
BY_STEPS = "by_steps"           # a task's own plan: steps finished out of steps

# A request has to be doing at least this many genuinely different things before
# it is worth carrying across turns. Two is a sequence; one is an errand.
MIN_STAGES = 2

_EXPLICITLY_ONGOING = (
    "keep going until", "until it is done", "until it's done", "over the next",
    "every day until", "work towards", "work toward", "my goal is", "the goal is",
    "i want to reach", "target of", "by the end of the",
)


def worth_tracking(intent: intent_reader.Intent, text: str = "") -> Dict[str, Any]:
    """Should this request become something that outlives the turn?

    Deterministic, and biased hard towards no. The reasoning is returned either
    way so a "no" can be argued with rather than just experienced.
    """
    lowered = str(text or "").lower()
    explicit = next((phrase for phrase in _EXPLICITLY_ONGOING if phrase in lowered), None)
    if explicit:
        return {"track": True, "why": f"the request says so ('{explicit}')",
                "as": "goal"}
    if intent.is_complex and len(intent.stages) >= MIN_STAGES:
        return {"track": True,
                "why": (f"{len(intent.stages)} distinct stages "
                        f"({', '.join(intent.stages[:4])})"),
                "as": "task"}
    return {"track": False,
            "why": ("this is one request, not an objective - answer it and stop"
                    if not intent.is_complex else
                    "it is involved, but it is still one thing"),
            "as": None}


# --------------------------------------------------------------------------- #
# The shape
# --------------------------------------------------------------------------- #

def from_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """GOAL..NEXT ACTION for one autonomous task, read from the task itself."""
    from core import task_manager

    steps = task.get("steps", [])
    done = [s for s in steps if s.get("status") == "done"]
    current = task.get("current_step", 0)
    next_step = steps[current] if 0 <= current < len(steps) else None
    blocked = task.get("status") == task_manager.WAITING_FOR_USER

    return {
        "kind": "task",
        "id": task.get("id"),
        "goal": task.get("objective") or task.get("name"),
        "success_criteria": task.get("success_criteria"),
        "plan": [s.get("instruction", "") for s in steps],
        "actions": [{"step": s.get("n", i + 1), "did": s.get("instruction", ""),
                     "status": s.get("status"),
                     "recoveries": s.get("recoveries", 0)}
                    for i, s in enumerate(steps) if s.get("status") != "pending"],
        "results": [{"step": s.get("n", i + 1), "result": (s.get("result") or "")[:300]}
                    for i, s in enumerate(steps) if s.get("result")],
        "progress": {
            "basis": BY_STEPS,
            "percent": round(len(done) / len(steps) * 100.0, 1) if steps else None,
            "how": (f"{len(done)} of {len(steps)} steps finished. Steps finished is not "
                    "the same as the objective achieved - that is what "
                    "success_criteria is checked against at the end."),
        },
        "next_action": (
            f"WAITING FOR YOU: {task.get('blocked_reason')}" if blocked
            else next_step.get("instruction") if next_step
            else "nothing - every step is done"),
        "status": task.get("status"),
        "verified": (task.get("verification") or {}).get("passed"),
    }


def from_goal(goal: Dict[str, Any]) -> Dict[str, Any]:
    """The same shape for a business goal, over core/business_goals.py."""
    from core import business_goals, task_manager

    measured = business_goals.progress(goal)
    linked_ids = list(goal.get("tasks") or [])
    tasks = [t for t in task_manager.load_tasks() if t.get("id") in linked_ids]
    active = [t for t in tasks if t.get("status") in task_manager.ACTIVE_STATUSES]
    waiting = [t for t in tasks if t.get("status") == task_manager.WAITING_FOR_USER]

    if waiting:
        next_action = (f"WAITING FOR YOU: {waiting[0].get('blocked_reason') or 'approval'} "
                       f"(task '{waiting[0].get('name')}')")
    elif active:
        next_action = f"task '{active[0].get('name')}' is running"
    elif measured["basis"] == UNMEASURABLE:
        next_action = ("give this goal a target value, record a result, or link the "
                       "tasks that pursue it - otherwise nothing can say where it is")
    else:
        next_action = "no work is attached to this goal right now"

    return {
        "kind": "goal",
        "id": goal.get("id"),
        "goal": goal.get("name"),
        "success_criteria": goal.get("target") or None,
        "plan": [t.get("name") for t in tasks] or [],
        "actions": [{"did": t.get("name"), "status": t.get("status")} for t in tasks],
        "results": list(goal.get("results") or [])[-5:],
        "progress": measured,
        "next_action": next_action,
        "status": goal.get("status"),
        "project": goal.get("project") or None,
    }


def current(limit: int = 5) -> List[Dict[str, Any]]:
    """Everything Leti is currently carrying, newest first.

    Reads the task store and, when Business Mode's goals exist, those too. Both
    reads are local JSON files; neither is a search and neither is cached here.
    """
    out: List[Dict[str, Any]] = []
    try:
        from core import task_manager

        for task in task_manager.load_tasks():
            if task.get("status") in task_manager.ACTIVE_STATUSES:
                out.append(from_task(task))
    except Exception as e:
        logger.debug(f"Couldn't read tasks for the objective view: {e}")

    try:
        from core import business_goals

        for goal in business_goals.load_goals():
            if goal.get("status") == business_goals.ACTIVE:
                out.append(from_goal(goal))
    except Exception as e:
        logger.debug(f"Couldn't read goals for the objective view: {e}")

    out.sort(key=lambda o: o.get("status") == "waiting_for_user", reverse=True)
    return out[:limit]


def summarise(objectives: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One paragraph a person can read, and the counts behind it."""
    items = objectives if objectives is not None else current()
    if not items:
        return {"count": 0, "waiting_on_you": 0,
                "text": "Nothing is being carried across turns right now."}
    waiting = [o for o in items if str(o.get("next_action", "")).startswith("WAITING")]
    lines = []
    for objective in items:
        percent = (objective.get("progress") or {}).get("percent")
        where = f"{percent}%" if percent is not None else "progress not measurable"
        lines.append(f"- {objective['goal']} ({where}) - next: {objective['next_action']}")
    return {
        "count": len(items),
        "waiting_on_you": len(waiting),
        "text": "\n".join(lines),
        "at": time.time(),
    }


def turn_note(limit: int = 2) -> str:
    """One short block for the model, when something is genuinely in flight.

    Empty when nothing is, which is nearly every turn - the same discipline
    core/proactive.py follows, and for the same reason.
    """
    try:
        items = current(limit=limit)
    except Exception:
        return ""
    if not items:
        return ""
    summary = summarise(items)
    return ("Objectives carried from earlier (do not restart them, and do not claim "
            f"progress the plan does not show):\n{summary['text']}")
