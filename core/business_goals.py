"""Goal -> Project -> Task -> Result, made out of things that already exist.

This is a structure layer, not a store. A goal is a record in the business file
Leti already keeps (tools/business.py, same path, same atomic write, same
load/save), and what it holds is mostly references: the names of projects it is
pursued through, the ids of tasks in the task manager that belong to it, and the
results somebody actually measured.

Nothing here is a second task manager or a second project system. A "project" on
a goal is a project name that Project Memory already knows; a "task" is an id
from core/task_manager.py. Deleting the goal does not touch either of them, and
the goal knowing about a task is not the goal owning it.

The rule the whole file is bent around is the one about progress. A percentage is
only honest when there is something to count and something to count towards. So
progress is reported four ways - by measured result against a target, by task
completion, by neither, or not at all - and the answer always says which of those
it is. A goal with no target and no linked tasks reports that its progress cannot
be determined, because that is the truth and "0%" is not.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.business.goals")

ACTIVE, PAUSED, COMPLETED, BLOCKED = "active", "paused", "completed", "blocked"
STATUSES = (ACTIVE, PAUSED, COMPLETED, BLOCKED)

MAX_GOALS = 40
MAX_LINKS = 20
MAX_RESULTS = 20
MAX_TEXT = 200
NEGLECTED_DAYS = 21


def _records():
    from tools.business import load_records, save_records

    return load_records, save_records


def load_goals() -> List[Dict[str, Any]]:
    load, _ = _records()
    try:
        return list(load().get("goal", []))
    except Exception as e:
        logger.warning(f"Couldn't read goals ({e}); treating as none.")
        return []


def _save(goals: List[Dict[str, Any]]) -> None:
    load, save = _records()
    records = load()
    records["goal"] = goals[:MAX_GOALS]
    save(records)


def get(goal_id: str) -> Optional[Dict[str, Any]]:
    return next((g for g in load_goals() if g.get("id") == goal_id), None)


def find(hint: str) -> List[Dict[str, Any]]:
    """Goals matching a phrase. Several matches is an answer, not a problem."""
    words = [w for w in str(hint or "").lower().split() if len(w) > 2]
    if not words:
        return load_goals()
    return [g for g in load_goals()
            if any(w in str(g.get("name", "")).lower() for w in words)]


def create(name: str, target: str = "", measure: str = "",
           target_value: Optional[float] = None, due: str = "") -> Dict[str, Any]:
    """A new goal. `measure` and `target_value` are what make progress checkable.

    Without them the goal still works - it just reports honestly that its progress
    cannot be measured, rather than inventing a number for it.
    """
    text = str(name or "").strip()
    if not text:
        raise ValueError("A goal needs a name - what is it you are trying to achieve?")
    goals = load_goals()
    if len(goals) >= MAX_GOALS:
        raise ValueError(f"There are already {len(goals)} goals; close some before adding more.")

    goal = {
        "id": uuid.uuid4().hex[:10],
        "name": text[:MAX_TEXT],
        "target": str(target or "")[:MAX_TEXT] or None,
        "measure": str(measure or "")[:MAX_TEXT] or None,
        "target_value": float(target_value) if target_value is not None else None,
        "due": str(due or "")[:40] or None,
        "status": ACTIVE,
        "projects": [],
        "tasks": [],
        "results": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    goals.append(goal)
    _save(goals)
    return goal


def _replace(goal: Dict[str, Any]) -> Dict[str, Any]:
    goal["updated_at"] = time.time()
    goals = load_goals()
    for i, existing in enumerate(goals):
        if existing.get("id") == goal["id"]:
            goals[i] = goal
            _save(goals)
            return goal
    goals.append(goal)
    _save(goals)
    return goal


def set_status(goal_id: str, status: str, reason: str = "") -> Optional[Dict[str, Any]]:
    goal = get(goal_id)
    if goal is None:
        return None
    if status not in STATUSES:
        raise ValueError(f"'{status}' is not a status. Use: {', '.join(STATUSES)}.")
    goal["status"] = status
    goal["status_reason"] = str(reason or "")[:MAX_TEXT] or None
    return _replace(goal)


def link(goal_id: str, project: str = "", task_id: str = "") -> Optional[Dict[str, Any]]:
    """Point a goal at a project or a task that already exists.

    Both are references. The project lives in Project Memory and the task in the
    task manager; this records that they belong to this goal and nothing more.
    """
    goal = get(goal_id)
    if goal is None:
        return None
    if project:
        name = str(project)[:MAX_TEXT]
        if name not in goal["projects"]:
            goal["projects"] = (goal["projects"] + [name])[:MAX_LINKS]
    if task_id:
        from core import task_manager

        if task_manager.get_task(task_id) is None:
            raise ValueError(f"There is no task with id '{task_id}' to link.")
        if task_id not in goal["tasks"]:
            goal["tasks"] = (goal["tasks"] + [task_id])[:MAX_LINKS]
    return _replace(goal)


def record_result(goal_id: str, value: Optional[float] = None, note: str = "",
                  at: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Something measured. This is what makes a goal's progress real rather than felt."""
    goal = get(goal_id)
    if goal is None:
        return None
    if value is None and not str(note or "").strip():
        raise ValueError("A result needs a number, a note, or both.")
    goal["results"] = (goal["results"] + [{
        "value": float(value) if value is not None else None,
        "note": str(note or "")[:MAX_TEXT] or None,
        "at": at if at is not None else time.time(),
    }])[-MAX_RESULTS:]
    return _replace(goal)


def delete(goal_id: str) -> bool:
    goals = load_goals()
    remaining = [g for g in goals if g.get("id") != goal_id]
    if len(remaining) == len(goals):
        return False
    _save(remaining)
    return True


# --------------------------------------------------------------------------- #
# Progress, which is where it would be easy to lie
# --------------------------------------------------------------------------- #

MEASURED = "measured"          # a real number against a real target
BY_TASKS = "by_tasks"          # how many linked tasks are done
UNMEASURABLE = "unmeasurable"  # nothing to count, and no pretending otherwise


def progress(goal: Dict[str, Any]) -> Dict[str, Any]:
    """How far along a goal is, and on what basis - or that it cannot be said.

    Deliberately refuses to produce a percentage from nothing. A goal with no
    target value and no linked tasks has no denominator, and inventing one is the
    single most misleading thing a business tool can do.
    """
    latest = _latest_value(goal)
    target = goal.get("target_value")

    if target is not None and latest is not None:
        try:
            percent = max(0.0, min(100.0, (latest / float(target)) * 100.0)) if target else None
        except ZeroDivisionError:
            percent = None
        return {"basis": MEASURED, "percent": round(percent, 1) if percent is not None else None,
                "current": latest, "target": target,
                "measure": goal.get("measure"),
                "how": (f"{latest} of {target} {goal.get('measure') or ''}".strip()
                        + " - a measured result against the target you set."),
                "counted_from": "results recorded on this goal"}

    tasks = _linked_task_states(goal)
    if tasks["total"]:
        percent = tasks["done"] / tasks["total"] * 100.0
        return {"basis": BY_TASKS, "percent": round(percent, 1),
                "current": tasks["done"], "target": tasks["total"],
                "how": (f"{tasks['done']} of {tasks['total']} linked tasks are done. This "
                        "counts work completed, which is not the same as the goal being "
                        "achieved."),
                "counted_from": "the task manager",
                "blocked_tasks": tasks["blocked"]}

    missing = []
    if target is None:
        missing.append("no target value to measure against")
    if latest is None and target is not None:
        missing.append("no result has been recorded yet")
    if not goal.get("tasks"):
        missing.append("no tasks are linked to it")
    return {"basis": UNMEASURABLE, "percent": None,
            "how": ("Progress cannot be determined: " + "; ".join(missing) + ". "
                    "Give the goal a target value and record results, or link the tasks "
                    "that pursue it."),
            "counted_from": None}


def _latest_value(goal: Dict[str, Any]) -> Optional[float]:
    for result in reversed(goal.get("results") or []):
        if result.get("value") is not None:
            return float(result["value"])
    return None


def _linked_task_states(goal: Dict[str, Any]) -> Dict[str, Any]:
    """What the task manager says about the tasks this goal points at."""
    from core import task_manager

    done = blocked = total = 0
    missing = []
    for task_id in goal.get("tasks") or []:
        task = task_manager.get_task(task_id)
        if task is None:
            missing.append(task_id)
            continue
        total += 1
        if task.get("status") == task_manager.COMPLETED:
            done += 1
        elif task.get("status") in (task_manager.WAITING_FOR_USER, task_manager.FAILED):
            blocked += 1
    return {"done": done, "blocked": blocked, "total": total, "missing": missing}


def describe(goal: Dict[str, Any]) -> Dict[str, Any]:
    tasks = _linked_task_states(goal)
    return {
        "id": goal.get("id"),
        "name": goal.get("name"),
        "status": goal.get("status"),
        "target": goal.get("target"),
        "due": goal.get("due"),
        "projects": goal.get("projects") or [],
        "tasks": {"linked": len(goal.get("tasks") or []), **tasks},
        "latest_result": _latest_value(goal),
        "results_recorded": len(goal.get("results") or []),
        "progress": progress(goal),
    }


def needs_attention(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Goals that are blocked, or that nothing has happened to in weeks.

    Both are things a person would want to be told. Neither is inferred from
    anything cleverer than a status field and a timestamp.
    """
    now = now if now is not None else time.time()
    flagged = []
    for goal in load_goals():
        if goal.get("status") in (COMPLETED, PAUSED):
            continue
        tasks = _linked_task_states(goal)
        idle_days = (now - float(goal.get("updated_at", now))) / 86_400
        why = []
        if goal.get("status") == BLOCKED:
            why.append(goal.get("status_reason") or "marked blocked")
        if tasks["blocked"]:
            why.append(f"{tasks['blocked']} linked task(s) blocked or failed")
        if idle_days >= NEGLECTED_DAYS:
            why.append(f"nothing recorded for {int(idle_days)} days")
        if why:
            flagged.append({"goal": goal.get("name"), "goal_id": goal.get("id"),
                            "status": goal.get("status"), "why": "; ".join(why)})
    return flagged[:MAX_LINKS]
