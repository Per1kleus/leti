"""Objectives that outlive a single turn.

An ordinary request is one turn: the user asks, the orchestrator runs whatever
tools it needs, an answer comes back. "Research twenty companies and put them in
a spreadsheet" is not that shape - it is a dozen turns, it survives a restart, and
somewhere in the middle it may need permission the user is not there to give.

This keeps that state. It is deliberately not an executor: every step runs through
orchestrator.handle_user_input, exactly as if the user had typed it, so tool
routing, the safety guard, tool-call parsing and tool execution are all the ones
that already exist. Nothing here calls a tool. Nothing here decides whether a call
is allowed.

Where the permission goes:

    step -> orchestrator -> tool router -> SafetyGuard -> tool

SafetyGuard already knows what an unattended run may do - reading, computing and
writing, per scheduler.unattended_allows - and refuses external and critical
actions with ConfirmationDenied when nobody is present to approve them. A task
that hits one does not fail and does not proceed: it stops at that step, keeps
everything it has done, and waits for the user. That is the whole approval story,
and it is the existing one.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.atomic_write import atomic_write_text
from core.config_loader import resolve_path
from core import recovery, task_history, world_state

logger = logging.getLogger("leti.tasks")

STORE_PATH = "./data/autonomous_tasks.json"
MAX_STEPS = 25
DEFAULT_MAX_ATTEMPTS = 2          # one retry; a step that fails twice is not transient
# Self-recovery: after the plain retries are spent, a step whose failure looks
# recoverable gets this many attempts at a DIFFERENT approach. Small on purpose -
# the failure mode of automatic recovery is a machine that will not admit defeat.
MAX_RECOVERIES_PER_STEP = 2
MAX_RECOVERY_SECONDS = 300        # a step that has been recovering for five minutes has failed
MAX_RESULT_CHARS = 4000
MAX_TASKS_KEPT = 50

QUEUED = "queued"
RUNNING = "running"
PAUSED = "paused"
WAITING_FOR_USER = "waiting_for_user"
FAILED = "failed"
COMPLETED = "completed"
CANCELLED = "cancelled"

# Waiting on somebody else's machine rather than on the user - a build, a
# download, a provider. Distinct from WAITING_FOR_USER because the answer to
# "which task needs me" must not include it, and distinct from RUNNING because
# nothing is being attempted.
WAITING_FOR_EXTERNAL = "waiting_for_external"
# A step that failed and is being attempted a different way. Visible as its own
# state so "retrying" never reads as ordinary progress - see core/recovery.py.
RETRYING = "retrying"

# The in-between states. A task passes through one of these while a control it
# was given takes effect, so the interface and the runner agree about what is
# happening during the gap between asking and it being true.
#
# They exist because the controls are COOPERATIVE: pause and cancel do not
# interrupt a step that is already running, so there is a real interval where
# the task has been told to stop and has not stopped yet. Showing PAUSED during
# that interval would be a lie, and showing RUNNING would hide the request.
PAUSING = "pausing"
CANCELLING = "cancelling"
RESUMING = "resuming"

TRANSITIONAL_STATUSES = (PAUSING, CANCELLING, RESUMING)
ACTIVE_STATUSES = (QUEUED, RUNNING, PAUSED, WAITING_FOR_USER, WAITING_FOR_EXTERNAL,
                   RETRYING) + TRANSITIONAL_STATUSES
FINISHED_STATUSES = (FAILED, COMPLETED, CANCELLED)

# Which statuses each control may be applied from. One table rather than a
# condition per function, so the state machine is a thing you can read.
ALLOWED_FROM = {
    "pause": (QUEUED, RUNNING, RETRYING, WAITING_FOR_EXTERNAL),
    "resume": (PAUSED, PAUSING, WAITING_FOR_USER, WAITING_FOR_EXTERNAL, FAILED),
    "cancel": ACTIVE_STATUSES,
    "retry": tuple(st for st in ACTIVE_STATUSES + FINISHED_STATUSES
                   if st not in (RUNNING, COMPLETED)),
    "skip": (PAUSED, WAITING_FOR_USER, WAITING_FOR_EXTERNAL, FAILED, RETRYING, QUEUED),
    "approve": (WAITING_FOR_USER,),
    "reject": (WAITING_FOR_USER,),
}


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #

def store_path():
    return resolve_path(STORE_PATH)


def load_tasks() -> List[Dict[str, Any]]:
    path = store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Couldn't read {path} ({e}); starting with no tasks.")
        return []
    return data if isinstance(data, list) else []


def save_tasks(tasks: List[Dict[str, Any]]) -> None:
    # Finished tasks are history; keeping every one of them forever turns a small
    # state file into a slow one.
    trimmed = [t for t in tasks if t.get("status") in ACTIVE_STATUSES]
    finished = [t for t in tasks if t.get("status") in FINISHED_STATUSES]
    finished.sort(key=lambda t: t.get("updated_at", 0))
    keep = trimmed + finished[-MAX_TASKS_KEPT:]
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(keep, indent=2))


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    return next((t for t in load_tasks() if t.get("id") == task_id), None)


def _replace(task: Dict[str, Any]) -> None:
    tasks = load_tasks()
    for index, existing in enumerate(tasks):
        if existing.get("id") == task["id"]:
            task["updated_at"] = time.time()
            tasks[index] = task
            save_tasks(tasks)
            return
    tasks.append(task)
    save_tasks(tasks)


# --------------------------------------------------------------------------- #
# Creating and describing
# --------------------------------------------------------------------------- #

def create_task(objective: str, steps: List[str], name: str = "",
                project: str = "", expected: Optional[List[str]] = None,
                success_criteria: str = "") -> Dict[str, Any]:
    """A new task, queued. Steps are the plan; nothing runs until start().

    `expected` is what each step should have produced, in the same order as the
    steps - the answer to "how would you know that worked". `success_criteria` is
    the same question about the whole task, and it is what the verification pass
    at the end checks against. Both are optional: a task without them behaves
    exactly as tasks did before, which is what every existing caller relies on.
    """
    if not str(objective or "").strip():
        raise ValueError("A task needs an objective.")
    cleaned = [str(s).strip() for s in (steps or []) if str(s).strip()]
    if not cleaned:
        raise ValueError("A task needs at least one step.")
    if len(cleaned) > MAX_STEPS:
        raise ValueError(f"That is {len(cleaned)} steps; the limit is {MAX_STEPS}.")
    outcomes = [str(e).strip() for e in (expected or [])]

    now = time.time()
    task = {
        "id": uuid.uuid4().hex[:12],
        "name": (name or objective)[:80],
        "objective": objective,
        # Which project this belongs to, so reopening one shows what was already
        # done for it. A name, not a copy of the project.
        "project": project or None,
        "status": QUEUED,
        "current_step": 0,
        "steps": [{"n": i + 1, "instruction": text, "status": "pending",
                   "attempts": 0, "result": None, "error": None,
                   "expected": outcomes[i] if i < len(outcomes) else None}
                  for i, text in enumerate(cleaned)],
        # What finished means for this task, and how many times it has been
        # checked. A task with no criteria is finished when its steps are, which
        # is what every task did before this existed.
        "success_criteria": str(success_criteria or "").strip() or None,
        "verification": None,
        "verify_rounds": 0,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "result": None,
        "error": None,
        "blocked_reason": None,
    }
    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)
    return task


def _note(step: Dict[str, Any], text: str) -> None:
    """Write what happened into the step, so a recovery is visible rather than
    a gap between a failure the user saw and a success they cannot explain."""
    step.setdefault("history", []).append(
        {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "note": text})
    step["history"] = step["history"][-10:]


def progress(task: Dict[str, Any]) -> Dict[str, Any]:
    """Real progress, counted from the steps. Never an invented percentage."""
    steps = task.get("steps", [])
    done = sum(1 for s in steps if s.get("status") == "done")
    current = task.get("current_step", 0)
    return {
        "steps_total": len(steps),
        "steps_done": done,
        "current_step": min(current + 1, len(steps)) if steps else 0,
        "current_instruction": (steps[current]["instruction"]
                                if 0 <= current < len(steps) else None),
    }


def describe(task: Dict[str, Any]) -> Dict[str, Any]:
    """A task as the model and the interface should see it."""
    p = progress(task)
    return {
        "id": task["id"],
        "name": task.get("name"),
        "objective": task.get("objective"),
        "status": task.get("status"),
        "step": f"{p['current_step']}/{p['steps_total']}" if p["steps_total"] else "0/0",
        "steps_done": p["steps_done"],
        "steps_total": p["steps_total"],
        "current": p["current_instruction"],
        "project": task.get("project"),
        "recoveries": sum(s.get("recoveries", 0) for s in task.get("steps", [])),
        "step_history": [note for s in task.get("steps", []) for note in s.get("history", [])],
        "blocked_reason": task.get("blocked_reason"),
        "error": task.get("error"),
        "result": task.get("result"),
        "success_criteria": task.get("success_criteria"),
        "verified": (task.get("verification") or {}).get("passed"),
        "verification_said": (task.get("verification") or {}).get("said"),
        "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(task.get("created_at", 0))),
    }


STEP_STATES = ("pending", "running", "recovering", "done", "failed", "blocked",
               "skipped")


def detail(task: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the interface's task panel shows, read from the task itself.

    Separate from describe() on purpose: describe() goes into tool output and
    therefore into the model's context, where a full step list would be tokens
    spent on something the model already knows it planned. This goes to a panel,
    where the whole point is seeing every step.
    """
    base = describe(task)
    steps = task.get("steps", [])
    base["steps"] = [{
        "n": s.get("n", i + 1),
        "instruction": s.get("instruction", ""),
        "kind": s.get("kind", "work"),
        "expected": s.get("expected"),
        "status": s.get("status", "pending"),
        "attempts": s.get("attempts", 0),
        "recoveries": s.get("recoveries", 0),
        "error": s.get("error"),
        # A preview, not the result: a step that wrote a report should not put the
        # report in a panel row.
        "result": (s.get("result") or "")[:400] or None,
        "history": s.get("history", [])[-4:],
    } for i, s in enumerate(steps)]
    done = [s for s in steps if s.get("status") == "done"]
    current = task.get("current_step", 0)
    # Everything a panel needs to say where this is, computed from the steps that
    # are already here. No second progress model: these are the same steps
    # describe() counted, named the way a person reads them.
    base["percent"] = round(len(done) / len(steps) * 100.0, 1) if steps else None
    base["completed_steps"] = [s.get("instruction", "") for s in done]
    base["next_step"] = (steps[current + 1].get("instruction")
                         if 0 <= current + 1 < len(steps) else None)
    base["failed_step"] = next((s.get("instruction") for s in steps
                                if s.get("status") == "failed"), None)
    base["recovering"] = any(s.get("status") == "recovering" for s in steps)
    base["awaiting_approval"] = task.get("status") == WAITING_FOR_USER
    base["approval_request"] = task.get("blocked_reason") if base["awaiting_approval"] else None
    # The one line the interface puts in front of the user when it is their turn.
    base["waiting_for_you"] = (
        (f"Leti needs your answer before it can carry on: "
         f"{task.get('blocked_reason') or 'it needs your approval for the next step'}.")
        if base["awaiting_approval"] else None)
    base["can"] = {
        "pause": task.get("status") in (QUEUED, RUNNING),
        "resume": task.get("status") in (PAUSED, WAITING_FOR_USER, FAILED),
        "cancel": task.get("status") not in FINISHED_STATUSES,
        "retry": task.get("status") in (FAILED, PAUSED, WAITING_FOR_USER, CANCELLED),
        "approve": task.get("status") == WAITING_FOR_USER,
    }
    return base


def find_active(hint: str = "") -> List[Dict[str, Any]]:
    """Active tasks matching a phrase, for "pause the research task".

    Returns every active task when there is no hint, and every match when there is
    more than one - the caller asks which, rather than this guessing.
    """
    active = [t for t in load_tasks() if t.get("status") in ACTIVE_STATUSES]
    words = [w for w in str(hint or "").lower().split() if len(w) > 2]
    if not words:
        return active
    exact = [t for t in active if t["id"] == hint.strip()]
    if exact:
        return exact
    return [t for t in active
            if any(w in f"{t.get('name','')} {t.get('objective','')}".lower() for w in words)] or active


# --------------------------------------------------------------------------- #
# Control - all cooperative, all persisted
# --------------------------------------------------------------------------- #

# How each status reads in the interface's activity log. The status itself is
# unchanged and still owned here - this is only the wording.
_STATUS_WORDS = {
    QUEUED: "queued",
    RUNNING: "started",
    PAUSED: "paused",
    PAUSING: "stopping after this step",
    RESUMING: "resuming",
    CANCELLING: "cancelling",
    RETRYING: "retrying a step that failed",
    WAITING_FOR_USER: "waiting for your approval",
    WAITING_FOR_EXTERNAL: "waiting on something outside this machine",
    FAILED: "failed",
    COMPLETED: "completed",
    CANCELLED: "cancelled",
}


def _set_status(task_id: str, status: str, **fields) -> Optional[Dict[str, Any]]:
    task = get_task(task_id)
    if task is None:
        return None
    task["status"] = status
    task.update(fields)
    if status in FINISHED_STATUSES:
        task["completed_at"] = time.time()
    _replace(task)
    _mirror_to_activity(task, status)
    # The record that outlives the store's own trim - see core/task_history.py.
    # Written on every transition, not only at the end, so a task interrupted by
    # a restart still has one. It never raises into here.
    task_history.record(task)
    return task


def spoken_progress(task: Dict[str, Any], index: Optional[int] = None) -> str:
    """Where a task is, in one sentence short enough to hear.

    Read entirely from the task's own steps, so what is said out loud and what the
    interface's task panel shows come from the same place and cannot drift. Says
    what is true and nothing more: a step that is recovering is described as
    trying again, not as progress.
    """
    steps = task.get("steps", [])
    index = task.get("current_step", 0) if index is None else index
    total = len(steps)
    position = min(index + 1, total) if total else 0
    name = task.get("name") or "the task"
    step = steps[index] if 0 <= index < total else {}
    doing = (step.get("instruction") or "").strip()
    short = (doing[:60].rsplit(" ", 1)[0] + "...") if len(doing) > 60 else doing

    if step.get("status") == "recovering":
        return f"Still on {name}: step {position} of {total} did not work, trying another way."
    if task.get("status") == WAITING_FOR_USER:
        return (f"{name} is waiting for you: "
                + (task.get("blocked_reason") or "it needs your approval to carry on") + ".")
    if short:
        return f"Still working on {name}: step {position} of {total}, {short}"
    return f"Still working on {name}: step {position} of {total}."


def _mirror_step(task: Dict[str, Any], index: int) -> None:
    """Progress, for the same log and with the same guarantee: a task never fails
    over a log line. Read from the step that is already being marked running."""
    try:
        from core import diagnostics

        steps = task.get("steps", [])
        diagnostics.record_activity(
            "task",
            f"{task.get('name') or 'Task'} - step {index + 1} of {len(steps)}: "
            f"{steps[index].get('instruction', '')[:80]}",
            task_id=task.get("id"), step=index + 1,
        )
    except Exception:
        logger.debug("Couldn't mirror step progress to the activity log.")


def _mirror_to_activity(task: Dict[str, Any], status: str) -> None:
    """Tell the activity log a status changed. Writes nothing and decides nothing.

    Every status transition already passes through _set_status, so one call here
    covers all of them without a second place that knows what a task's states are.
    Wrapped because a task must never fail over a log line.
    """
    try:
        from core import diagnostics

        diagnostics.record_activity(
            "task",
            f"Task '{task.get('name') or task.get('objective', 'task')}' "
            f"{_STATUS_WORDS.get(status, status)}",
            task_id=task.get("id"), status=status,
        )
    except Exception:
        logger.debug("Couldn't mirror a task status to the activity log.")


# Tasks that have been told to stop, in memory, checked by the runner before it
# does anything further. The STORE is still the source of truth - this is the
# fast path, so a stop lands between two steps rather than after the next
# reload - and everything here is mirrored into the store by the functions below.
#
# In memory on purpose: after a restart nothing is running, so there is nothing
# for a stop request to still apply to, and the store already says CANCELLED.
_stop_requested: Set[str] = set()


def stop_requested(task_id: str) -> bool:
    """Has this task been told to stop? The runner's cooperative check."""
    if str(task_id) in _stop_requested:
        return True
    task = get_task(task_id)
    return bool(task and task.get("status") in (CANCELLING, CANCELLED, PAUSING))


def clear_stop(task_id: str) -> None:
    _stop_requested.discard(str(task_id))


def pause(task_id: str) -> Optional[Dict[str, Any]]:
    """Stop after the step that is running.

    A running step is not interrupted: half a tool call is a worse state to
    leave behind than one extra step, and an external action that has already
    left this machine cannot be recalled by changing a status. So a running task
    goes to PAUSING, which says exactly that, and the runner moves it to PAUSED
    when the step it was in the middle of comes back.
    """
    task = get_task(task_id)
    if task is None or task.get("status") not in ALLOWED_FROM["pause"]:
        return None
    if task.get("status") == RUNNING:
        _stop_requested.add(str(task_id))
        return _set_status(task_id, PAUSING,
                           blocked_reason="Pausing after the step that is running.")
    return _set_status(task_id, PAUSED, blocked_reason=None)


def resume(task_id: str) -> Optional[Dict[str, Any]]:
    """Back in the queue, from wherever it stopped. Nothing already done is redone."""
    task = get_task(task_id)
    if task is None or task.get("status") not in ALLOWED_FROM["resume"]:
        return None
    clear_stop(task_id)
    return _set_status(task_id, RESUMING, blocked_reason=None, error=None)


def cancel(task_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
    """Stop, and keep everything already done.

    A cancelled task is not a finished one: its steps keep their own statuses so
    the history says what actually happened. A task that is mid-step goes to
    CANCELLING first and the runner finalises it - which is the truth, because
    the step that is running will finish, and claiming CANCELLED while a tool
    call is in flight would be the status lying about the world.

    What this CAN promise is that nothing further starts: the runner checks
    stop_requested() before every remaining step and before the verification
    pass, so cancellation prevents future actions even though it cannot undo the
    one already under way.
    """
    task = get_task(task_id)
    if task is None or task.get("status") in FINISHED_STATUSES:
        return None
    _stop_requested.add(str(task_id))
    if task.get("status") == RUNNING:
        return _set_status(task_id, CANCELLING,
                           blocked_reason=reason or "Cancelling after the step in flight.")
    return finalise_cancel(task_id, reason)


def finalise_cancel(task_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
    """Move a cancelling task to CANCELLED, recording what already ran.

    Called by the runner when the step in flight comes back, and directly by
    cancel() when nothing was running. The record of what was already executed
    is the point: a cancelled task that does not say what it did before it
    stopped is worse than one that was never cancelled.
    """
    task = get_task(task_id)
    if task is None or task.get("status") in FINISHED_STATUSES:
        return task
    executed = [s.get("n", i + 1) for i, s in enumerate(task.get("steps", []))
                if s.get("status") == "done"]
    already = (f"Steps already carried out before cancelling: "
               f"{', '.join(str(n) for n in executed)}." if executed
               else "Nothing had been carried out yet.")
    clear_stop(task_id)
    return _set_status(task_id, CANCELLED,
                       blocked_reason=((reason + " " if reason else "") + already).strip(),
                       cancelled_after_steps=executed)


def skip_step(task_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
    """Leave the step that stopped undone and move to the next one.

    The step is marked `skipped`, not `done`: a plan with a hole in it has to
    look like a plan with a hole in it, both in the task panel and in the
    history, and the final result says which steps were never carried out.
    """
    task = get_task(task_id)
    if task is None or task.get("status") not in ALLOWED_FROM["skip"]:
        return None
    steps = task.get("steps", [])
    if not steps:
        return None
    index = min(task.get("current_step", 0), len(steps) - 1)
    step = steps[index]
    step["status"] = "skipped"
    step["error"] = None
    step.pop("recovery_instruction", None)
    _note(step, f"skipped by the user{': ' + reason if reason else ''}")
    task["current_step"] = index + 1
    task.setdefault("skipped_steps", []).append(step.get("n", index + 1))
    _replace(task)
    clear_stop(task_id)
    return _set_status(task_id, RESUMING, blocked_reason=None, error=None)


def retry(task_id: str) -> Optional[Dict[str, Any]]:
    """Try the step that stopped, again, from the beginning of that step.

    Only that step: everything before it keeps its result, which is the whole
    reason a task records per-step state. The attempt counters reset so the
    ordinary retry and recovery budgets apply afresh rather than being already
    spent on the failure the user just looked at.
    """
    task = get_task(task_id)
    if task is None or task.get("status") in (RUNNING, COMPLETED):
        return None
    steps = task.get("steps", [])
    index = next((i for i, s in enumerate(steps)
                  if s.get("status") in ("failed", "blocked", "running", "recovering")), None)
    if index is None:
        index = min(task.get("current_step", 0), max(0, len(steps) - 1))
    if not steps:
        return None
    step = steps[index]
    _note(step, f"retried by the user after: {step.get('error') or step.get('status')}")
    step["status"] = "pending"
    step["attempts"] = 0
    step["recoveries"] = 0
    step["error"] = None
    step.pop("recovery_instruction", None)
    step.pop("first_attempt_at", None)
    task["current_step"] = index
    _replace(task)
    return _set_status(task_id, QUEUED, blocked_reason=None, error=None)


def approve(task_id: str) -> Optional[Dict[str, Any]]:
    """The user has said yes to the action the task stopped on.

    This does not authorise anything itself. It marks ONE step as carrying the
    user's approval and hands it back to the runner, which passes it to the
    orchestrator as the same pre-approval SafetyGuard already understands from a
    spoken "yes, go ahead". The guard still classifies the action, still refuses
    to let an irreversible one ride on approval given in advance, and still writes
    the audit line. There is no path here that runs a tool.
    """
    task = get_task(task_id)
    if task is None or task.get("status") != WAITING_FOR_USER:
        return None
    task["approved_step"] = task.get("current_step", 0)
    task["approved_at"] = time.time()
    _replace(task)
    return _set_status(task_id, QUEUED, blocked_reason=None, error=None)


def reject(task_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
    """The user has said no. The task stops and says why, rather than skipping on."""
    task = get_task(task_id)
    if task is None or task.get("status") != WAITING_FOR_USER:
        return None
    asked = task.get("blocked_reason") or "the action it stopped on"
    return cancel(task_id, reason or f"You declined: {asked}")


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

# The check at the end, and how many times it may ask for a correction. Small on
# purpose: a task that cannot satisfy its own criteria in two rounds has a
# problem the user needs to know about, not one more attempt.
MAX_VERIFY_ROUNDS = 2
VERIFIED = re.compile(r"\bVERIFIED\b", re.I)
NOT_VERIFIED = re.compile(r"\bNOT[\s_-]*VERIFIED\b", re.I)


def verification_step(task: Dict[str, Any]) -> Dict[str, Any]:
    """The step that checks the work against what the user actually asked for.

    An ordinary step, deliberately: it runs through the orchestrator like every
    other one, so it is subject to the same routing, the same tool set and the
    same SafetyGuard. Checking a file means opening it with the file tools, and
    if checking something needs permission the task stops and waits exactly as it
    would anywhere else.
    """
    return {
        "n": len(task.get("steps", [])) + 1,
        "kind": "verify",
        "instruction": (
            "Check whether the task is actually finished, then answer.\n"
            f"Objective: {task.get('objective')}\n"
            f"It counts as done when: {task.get('success_criteria')}\n"
            "Verify it by looking, not by remembering: open the file you wrote and "
            "read it, re-run the check, list the folder. A file existing is not the "
            "same as a file with the right contents in it.\n"
            "Reply with the word VERIFIED and one sentence of evidence if every part "
            "of that is true. Otherwise reply NOT VERIFIED and say exactly which part "
            "is missing or wrong. Do not fix anything in this step."),
        "status": "pending", "attempts": 0, "result": None, "error": None,
        "expected": "a VERIFIED or NOT VERIFIED answer with evidence",
    }


def correction_step(task: Dict[str, Any], problem: str) -> Dict[str, Any]:
    """One attempt at the thing the check said was missing."""
    return {
        "n": len(task.get("steps", [])) + 1,
        "kind": "correct",
        "instruction": (
            f"The check on this task found a problem: {problem}\n"
            f"Objective: {task.get('objective')}\n"
            "Fix that specific thing and nothing else. If fixing it needs permission "
            "you do not have, stop and say so rather than working around it."),
        "status": "pending", "attempts": 0, "result": None, "error": None,
        "expected": "the problem the check named, fixed",
    }


def read_verdict(answer: str) -> Tuple[bool, str]:
    """Whether the checking step said the work is done, and why it said so.

    NOT VERIFIED is looked for first: "not verified" contains "verified", and
    reading it the other way round would turn every failed check into a pass.
    An answer that says neither is not a pass either - a task is finished when
    something says it is, never when nothing said it was not.
    """
    text = str(answer or "")
    if NOT_VERIFIED.search(text):
        return False, text.strip()[:MAX_RESULT_CHARS]
    if VERIFIED.search(text):
        return True, text.strip()[:MAX_RESULT_CHARS]
    return False, ("the check did not answer VERIFIED or NOT VERIFIED; treating that "
                   "as unverified. It said: " + text.strip()[:600])


class TaskRunner:
    """Runs one task's steps through the orchestrator, one at a time.

    Not a scheduler and not an executor: it decides which step is next and hands
    the step's text to the orchestrator exactly as a user's message. Everything
    after that - which tools are offered, whether a call is permitted, how it runs -
    is the path that already existed.
    """

    def __init__(self, orchestrator, safety_guard=None,
                 notify: Optional[Callable[[str], Any]] = None):
        self.orchestrator = orchestrator
        self.safety_guard = safety_guard
        self.notify = notify
        # When each task last had something said about it out loud. In memory,
        # one float per task, so a long task does not narrate every step.
        self._last_progress_spoken: Dict[str, float] = {}
        self._current: Optional[str] = None
        # Strong references: asyncio only keeps weak ones, so a background task
        # without this can be garbage collected mid-run.
        self._background: set = set()

    def start_in_background(self, task_id: str) -> bool:
        """Run a task without making the caller wait for it.

        The tool that starts a task returns immediately - a request that took the
        length of the whole task to answer would not be a background task at all.
        One at a time: the orchestrator serialises turns anyway, so a second task
        would only interleave itself with the first.
        """
        if self._current is not None:
            logger.info(f"A task is already running; {task_id} stays queued.")
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        background = loop.create_task(self.run(task_id))
        self._background.add(background)
        background.add_done_callback(self._background.discard)
        return True

    async def run(self, task_id: str) -> Dict[str, Any]:
        """Drive a task to a stopping point: done, blocked, paused or failed."""
        task = get_task(task_id)
        if task is None:
            return {"task_id": task_id, "status": "missing"}
        # RESUMING is a startable state: it is what resume() and skip_step() leave
        # behind, and the runner picking it up is what turns it into RUNNING.
        if task.get("status") not in (QUEUED, RUNNING, RESUMING):
            return describe(task)

        self._current = task_id
        clear_stop(task_id)
        _set_status(task_id, RUNNING)
        # What this task believes about the world while it runs - see
        # core/world_state.py. In memory, one entry, dropped below whatever
        # happens; it informs a recovery instead of a recovery having to guess.
        world_state.begin(task_id, request=task.get("objective", ""),
                          task_id=task_id, project=task.get("project") or None,
                          goal=task.get("success_criteria") or None)
        # An unattended run is what the guard already understands: reading,
        # computing and writing proceed; sending and deleting stop and wait.
        restore_unattended = self._enter_unattended()
        try:
            while True:
                # The cooperative stop, checked before anything else happens.
                # This is what makes cancellation prevent FUTURE actions: the
                # step in flight finishes, and nothing after it starts.
                if self._settle_stop(task_id):
                    break
                task = get_task(task_id)
                # RETRYING is running - it is what a step looks like while it is
                # being attempted a different way, and it exists so that "trying
                # again" never reads as ordinary progress. Anything else here
                # means the task was paused or cancelled between steps.
                if task is None or task.get("status") not in (RUNNING, RETRYING):
                    break
                index = task.get("current_step", 0)
                steps = task.get("steps", [])
                if index >= len(steps):
                    # Every step is done. That is not the same as the task being
                    # done, and for a task that said what done means it is checked
                    # rather than assumed. A cancelled task does not get verified:
                    # verification is more work, and it was told to stop.
                    if not self._verify_or_finish(task_id):
                        break
                    continue
                if not await self._run_step(task_id, index):
                    break
        finally:
            restore_unattended()
            self._current = None
            world_state.forget(task_id)
            # Nothing is ever left mid-transition by the runner: whatever else
            # happened, a task that was told to pause or cancel ends up in the
            # state it was told to be in.
            try:
                self._settle_stop(task_id)
            except Exception:
                logger.exception("Couldn't settle a pending stop")
        return describe(get_task(task_id) or {})

    def _settle_stop(self, task_id: str) -> bool:
        """Turn a pending stop into the state it asked for. True means stop now.

        The only place PAUSING becomes PAUSED and CANCELLING becomes CANCELLED,
        so the interval where a task has been told to stop and has not stopped
        yet has exactly one exit.
        """
        task = get_task(task_id)
        if task is None:
            return True
        status = task.get("status")
        if status == CANCELLING or (str(task_id) in _stop_requested and
                                    status not in (PAUSING, PAUSED)):
            finalise_cancel(task_id)
            return True
        if status == PAUSING:
            clear_stop(task_id)
            executed = [s.get("n", i + 1) for i, s in enumerate(task.get("steps", []))
                        if s.get("status") == "done"]
            _set_status(task_id, PAUSED, blocked_reason=(
                f"Paused after step {max(executed)}." if executed
                else "Paused before anything ran."))
            return True
        if status == PAUSED:
            return True
        return False

    def _enter_unattended(self) -> Callable[[], None]:
        guard = self.safety_guard
        if guard is None or not hasattr(guard, "set_unattended"):
            return lambda: None
        previous = getattr(guard, "_unattended", False)
        guard.set_unattended(True)
        return lambda: guard.set_unattended(previous)

    async def _run_step(self, task_id: str, index: int) -> bool:
        """One step. True to keep going, False to stop here."""
        task = get_task(task_id)
        step = task["steps"][index]
        step["attempts"] = step.get("attempts", 0) + 1
        step.setdefault("first_attempt_at", time.time())
        step["status"] = "running"
        if task.get("status") == RETRYING:
            # Back to RUNNING for the duration of the attempt: RETRYING describes
            # the gap between a failure and the next try, not the try itself.
            task["status"] = RUNNING
        # One shot: the approval is consumed as the step starts, so a step that
        # fails for some other reason cannot quietly re-use the user's yes.
        approved = task.pop("approved_step", None) == index
        if approved:
            _note(step, "running with the approval you gave in the interface")
        _replace(task)
        _mirror_step(task, index)
        self._announce_progress(task, index)

        instruction = (
            f"[Autonomous task '{task['name']}', step {index + 1} of "
            f"{len(task['steps'])}. Objective: {task['objective']}]\n{step['instruction']}"
        )
        if step.get("expected"):
            instruction += (f"\nThis step is done when: {step['expected']}. Check that "
                            "before reporting it finished.")
        if step.get("recovery_instruction"):
            instruction += f"\n\n{step['recovery_instruction']}"
        # Written down BEFORE the step runs: an expectation formed after seeing
        # the answer is a description, not a check.
        world_state.expect(task_id, step.get("expected") or step.get("instruction", ""))
        try:
            answer = await self.orchestrator.handle_user_input(
                instruction, preapproved=approved)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return self._step_failed(task_id, index, e)

        task = get_task(task_id)
        if task is None:
            return False
        step = task["steps"][index]
        if task.get("status") in (CANCELLED, CANCELLING, PAUSING) or stop_requested(task_id):
            # Stopped while this step was running. The step finished anyway - the
            # answer is in hand - so it is recorded as finished and the task
            # stops. Leaving it as "running" would show a step frozen mid-flight
            # forever and lose work that actually happened, and claiming it never
            # ran would be the record lying about what this machine did.
            step["status"] = "done"
            step["result"] = (answer or "")[:MAX_RESULT_CHARS]
            _note(step, "finished, and the task stopped before the next step"
                        if task.get("status") == PAUSING
                        else "finished, but the task was cancelled before the next step")
            # The step really did finish, so the cursor really does move past it.
            # Leaving it where it was would mean a resumed task repeating a step
            # whose side effects have already happened - the exact thing pausing
            # between steps exists to avoid.
            task["current_step"] = index + 1
            _replace(task)
            world_state.observe(task_id, answer or "")
            return False
        step["status"] = "done"
        step["result"] = (answer or "")[:MAX_RESULT_CHARS]
        world_state.observe(task_id, answer or "")
        world_state.succeeded(task_id, step.get("instruction", "")[:120])
        if step.get("recoveries"):
            _note(step, f"recovery {step['recoveries']} succeeded")
        step["error"] = None
        step.pop("recovery_instruction", None)
        task["current_step"] = index + 1
        _replace(task)
        return True

    def _verify_or_finish(self, task_id: str) -> bool:
        """PLAN -> EXECUTE -> VERIFY -> CORRECT -> VERIFY -> COMPLETE, in the store.

        Returns True when another step was queued and the loop should keep going,
        False when the task has reached its end - completed, or failed with the
        reason the check gave. Nothing here executes anything: it appends the next
        step and lets the ordinary loop run it through the orchestrator.
        """
        task = get_task(task_id)
        if task is None:
            return False
        if not task.get("success_criteria"):
            self._finish(task_id)
            return False

        last = (task.get("steps") or [])[-1] if task.get("steps") else None
        if last and last.get("kind") == "verify":
            passed, why = read_verdict(last.get("result") or "")
            task["verification"] = {"passed": passed, "said": why,
                                    "round": task.get("verify_rounds", 0)}
            if passed:
                _replace(task)
                self._finish(task_id)
                return False
            if task.get("verify_rounds", 0) >= MAX_VERIFY_ROUNDS:
                _replace(task)
                _set_status(task_id, FAILED,
                            error=f"The work did not meet its own success criteria: {why}")
                self._announce(f"'{task['name']}' finished its steps but did not pass its "
                               f"own check: {why[:200]}")
                return False
            task["steps"].append(correction_step(task, why))
            task["verify_rounds"] = task.get("verify_rounds", 0) + 1
            _replace(task)
            return True

        # First arrival, or the step after a correction: check it.
        task["steps"].append(verification_step(task))
        _replace(task)
        return True

    def _step_failed(self, task_id: str, index: int, error: Exception) -> bool:
        """A step raised. Approval is not failure; everything else might retry."""
        task = get_task(task_id)
        step = task["steps"][index]
        message = str(error)

        if _needs_approval(error):
            # The guard refused because nobody was there to say yes. The task keeps
            # everything it has done and waits, rather than failing or proceeding.
            step["status"] = "blocked"
            step["error"] = message
            _replace(task)
            # Approved once and still refused means the guard classed it as
            # irreversible, which approval given in advance deliberately does not
            # cover. Saying that is more use than offering the same button again.
            already_approved = any("approval you gave" in n.get("note", "")
                                   for n in step.get("history", []))
            _set_status(task_id, WAITING_FOR_USER, blocked_reason=(
                f"Step {index + 1} needs your approval: {message}"
                + ("\n\nYou already approved this once. Leti will not let an "
                   "irreversible action run on approval given in advance - do this "
                   "step while you are here, or take it out of the task."
                   if already_approved else "")))
            self._announce(f"'{task['name']}' is waiting for your approval on step "
                           f"{index + 1} of {len(task['steps'])}.")
            return False

        if step["attempts"] < DEFAULT_MAX_ATTEMPTS:
            # Retry once. A bounded count, never a loop that can spin.
            step["status"] = "pending"
            step["error"] = message
            _note(step, f"attempt {step['attempts']} failed: {message}")
            _replace(task)
            logger.info(f"Task {task_id} step {index + 1} failed ({message}); retrying.")
            return True

        # The plain retries are spent. If the failure looks like something that
        # might succeed a different way - a site that was briefly down, a request
        # that timed out - the step is re-run ONCE more with the failure written
        # into it, so the model tries another approach rather than the same one.
        # It is still an ordinary request: the orchestrator runs it, the guard
        # authorises it, and nothing here executes anything.
        recoveries = step.get("recoveries", 0)
        elapsed = time.time() - step.get("first_attempt_at", time.time())
        # What to try, from core/recovery.py: the diagnosis and the different
        # approach, or the reason there is no point trying one. It decides;
        # this enforces. The budget stays here because the step state is here.
        decision = recovery.plan(
            error,
            attempts_used=recoveries,
            max_attempts=MAX_RECOVERIES_PER_STEP,
            expected=step.get("expected") or "",
            observed=(step.get("result") or "")[:500],
            previous_failures=step.get("failure_fingerprints") or [],
            cancelled=stop_requested(task_id),
        )
        step.setdefault("failure_fingerprints", []).append(decision["fingerprint"])
        step["failure_fingerprints"] = step["failure_fingerprints"][-6:]
        # Every attempt is on the record, recovered or not - diagnostics must
        # never have to guess how many times something was tried.
        step.setdefault("recovery_log", []).append({
            "at": time.time(), "kind": decision["kind"],
            "decision": decision["decision"], "diagnosis": decision["diagnosis"],
            "error": message[:200],
        })
        step["recovery_log"] = step["recovery_log"][-8:]

        if decision["recover"] and elapsed < MAX_RECOVERY_SECONDS:
            step["recoveries"] = recoveries + 1
            step["attempts"] = 0                  # the retries reset for the new approach
            step["status"] = "recovering"
            step["error"] = message
            step["recovery_instruction"] = (
                world_state.recovery_note(task_id, error) + " "
                + recovery.instruction(decision))
            # The step must be re-checked after a recovered attempt, not assumed
            # fixed because it did not raise the second time.
            step["verify_after_recovery"] = decision.get("verify")
            _note(step, f"recovery {step['recoveries']} attempted after: {message}")
            _replace(task)
            _set_status(task_id, RETRYING, error=None)
            logger.info(f"Task {task_id} step {index + 1}: recovery "
                        f"{step['recoveries']} after {message}")
            return True

        # No recovery, or the budget or the clock is spent. What the user is
        # told is the five things core/recovery.py names, not "it failed".
        escalation = decision.get("escalation") or {}
        if elapsed >= MAX_RECOVERY_SECONDS and decision["recover"]:
            escalation = {**escalation,
                          "why_it_stopped": "it had been recovering for too long"}
        step["status"] = "failed"
        step["error"] = message
        step["escalation"] = escalation
        _replace(task)
        remaining = [s.get("n", i + 1) for i, s in enumerate(task.get("steps", []))
                     if s.get("status") in ("pending", "recovering")]
        # Deliberately FAILED, never WAITING_FOR_USER. The case where the user
        # CAN unblock a step - the guard asking for a confirmation nobody was
        # there to give - is handled above by _needs_approval, and it has a
        # button behind it. Everything reaching here is something no approval
        # fixes: a forbidden path, a missing package, an unknown cause. Parking
        # one of those in WAITING_FOR_USER would show an Approve button that
        # cannot work, which is worse than saying it failed.
        _set_status(
            task_id, FAILED,
            error=f"Step {index + 1} failed: {message}",
            escalation={**escalation, "still_incomplete": remaining,
                        "what_to_tell_the_user": decision.get("say")})
        self._announce(f"'{task['name']}' stopped at step {index + 1}: {message}")
        return False

    def _finish(self, task_id: str) -> None:
        task = get_task(task_id)
        # The work's own result, not the checker's verdict on it: "VERIFIED, the
        # file has all five laptops in it" is how we know the task is done, and
        # the report the user asked for is what they wanted back.
        last = next((s["result"] for s in reversed(task.get("steps", []))
                     if s.get("result") and s.get("kind") not in ("verify", "correct")), None)
        if last is None:
            last = next((s["result"] for s in reversed(task.get("steps", []))
                         if s.get("result")), None)
        _set_status(task_id, COMPLETED, result=last)
        checked = (task.get("verification") or {}).get("passed")
        self._announce(f"'{task['name']}' is done."
                       + (" It passed its own check." if checked else ""))

    # Spoken progress on a long task.
    #
    # There is no second voice pipeline here and no loop: this fires on the step
    # transition that was happening anyway, and the line it produces goes down the
    # SAME notify path the approval and failure announcements already use, which
    # in voice mode is the one speak_callback and in the interface is the chat.
    # The interface's own task panel reads the same task state, so the two never
    # disagree about where a task is.
    #
    # Two conditions keep it from becoming chatter: a task has to be long enough
    # for progress to be a question, and enough time has to have passed since the
    # last thing Leti said about it. A four-step task that finishes in ten seconds
    # says nothing at all until it is done.
    SPOKEN_PROGRESS_AFTER_SECONDS = 45
    MIN_STEPS_FOR_SPOKEN_PROGRESS = 3

    def _announce_progress(self, task: Dict[str, Any], index: int) -> None:
        """One short line about where a long task has got to, or nothing."""
        try:
            steps = task.get("steps", [])
            if len(steps) < self.MIN_STEPS_FOR_SPOKEN_PROGRESS or index == 0:
                return
            now = time.time()
            since = now - self._last_progress_spoken.get(task["id"], 0.0)
            if since < self.SPOKEN_PROGRESS_AFTER_SECONDS:
                return
            self._last_progress_spoken[task["id"]] = now
            self._announce(spoken_progress(task, index))
        except Exception:
            logger.debug("Couldn't announce task progress; the task is unaffected.")

    def _announce(self, message: str) -> None:
        if not self.notify:
            return
        try:
            result = self.notify(message)
            if asyncio.iscoroutine(result):
                asyncio.get_event_loop().create_task(result)
        except Exception:
            logger.exception("Couldn't deliver a task notification")


def is_recoverable(error: Exception) -> bool:
    """Whether a different approach is worth one attempt.

    The judgement is core/world_state.py's: it classifies why something failed,
    and only the kinds where a different approach could plausibly work unlock this
    budget. One classifier rather than two, so "why did that fail" gets the same
    answer whether a task is asking or a turn is.

    Never true for an approval stop or a hard block: those are answers, not
    failures, and retrying them differently would be trying to get around them.
    """
    from core import world_state

    return world_state.unlocks_recovery(error)


def _needs_approval(error: Exception) -> bool:
    """Whether this is the guard asking for a human, rather than a real failure."""
    from core.safety_guard import ConfirmationDenied, PermissionDenied

    if isinstance(error, ConfirmationDenied):
        return True
    if isinstance(error, PermissionDenied):
        return False                    # blocked outright; approval would not help
    return "unattended" in str(error).lower()


def recover_interrupted() -> List[Dict[str, Any]]:
    """Tasks left mid-run by a restart. Marked paused, never silently resumed.

    A process that died in the middle of a step cannot know whether that step's
    side effects happened, so the task stops and says so instead of repeating it.
    """
    tasks = load_tasks()
    recovered = []
    for task in tasks:
        status = task.get("status")
        if status == RUNNING:
            task["status"] = PAUSED
            task["blocked_reason"] = (
                "Leti restarted while this task was running. Nothing was repeated - "
                "resume it when you are ready.")
        elif status == CANCELLING:
            # It had been told to stop and the runner never got to finish the
            # job. It is cancelled; that was the instruction.
            task["status"] = CANCELLED
            task["completed_at"] = time.time()
            task["blocked_reason"] = (
                (task.get("blocked_reason") or "Cancelled.")
                + " Leti restarted before the step in flight came back, so whether "
                  "that one step completed is not known.")
        elif status in (PAUSING, RESUMING):
            # Neither ever survives a restart: nothing is running for a pause to
            # take effect on, and nothing picked the resume up.
            task["status"] = PAUSED
            task["blocked_reason"] = (
                "Leti restarted before this took effect. Resume it when you are ready.")
        else:
            continue
        task["updated_at"] = time.time()
        recovered.append(task)
    if recovered:
        save_tasks(tasks)
        for task in recovered:
            task_history.record(task)
    return recovered
