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
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from core.atomic_write import atomic_write_text
from core.config_loader import resolve_path

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

ACTIVE_STATUSES = (QUEUED, RUNNING, PAUSED, WAITING_FOR_USER)
FINISHED_STATUSES = (FAILED, COMPLETED, CANCELLED)


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
                project: str = "") -> Dict[str, Any]:
    """A new task, queued. Steps are the plan; nothing runs until start()."""
    if not str(objective or "").strip():
        raise ValueError("A task needs an objective.")
    cleaned = [str(s).strip() for s in (steps or []) if str(s).strip()]
    if not cleaned:
        raise ValueError("A task needs at least one step.")
    if len(cleaned) > MAX_STEPS:
        raise ValueError(f"That is {len(cleaned)} steps; the limit is {MAX_STEPS}.")

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
                   "attempts": 0, "result": None, "error": None}
                  for i, text in enumerate(cleaned)],
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
        "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(task.get("created_at", 0))),
    }


STEP_STATES = ("pending", "running", "recovering", "done", "failed", "blocked")


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
        "status": s.get("status", "pending"),
        "attempts": s.get("attempts", 0),
        "recoveries": s.get("recoveries", 0),
        "error": s.get("error"),
        # A preview, not the result: a step that wrote a report should not put the
        # report in a panel row.
        "result": (s.get("result") or "")[:400] or None,
        "history": s.get("history", [])[-4:],
    } for i, s in enumerate(steps)]
    base["awaiting_approval"] = task.get("status") == WAITING_FOR_USER
    base["approval_request"] = task.get("blocked_reason") if base["awaiting_approval"] else None
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
    WAITING_FOR_USER: "waiting for your approval",
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
    return task


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


def pause(task_id: str) -> Optional[Dict[str, Any]]:
    """Stop after the step that is running. A running step is not interrupted -
    half a tool call is a worse state to leave behind than one extra step."""
    task = get_task(task_id)
    if task is None or task.get("status") not in (QUEUED, RUNNING):
        return None
    return _set_status(task_id, PAUSED)


def resume(task_id: str) -> Optional[Dict[str, Any]]:
    """Back to queued, from wherever it stopped. Nothing already done is redone."""
    task = get_task(task_id)
    if task is None or task.get("status") not in (PAUSED, WAITING_FOR_USER, FAILED):
        return None
    return _set_status(task_id, QUEUED, blocked_reason=None, error=None)


def cancel(task_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
    """Stop, and keep everything already done. A cancelled task is not a finished
    one: its steps keep their own statuses so the history says what happened."""
    task = get_task(task_id)
    if task is None or task.get("status") in FINISHED_STATUSES:
        return None
    return _set_status(task_id, CANCELLED, blocked_reason=reason or None)


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
        if task.get("status") not in (QUEUED, RUNNING):
            return describe(task)

        self._current = task_id
        _set_status(task_id, RUNNING)
        # An unattended run is what the guard already understands: reading,
        # computing and writing proceed; sending and deleting stop and wait.
        restore_unattended = self._enter_unattended()
        try:
            while True:
                task = get_task(task_id)
                if task is None or task.get("status") != RUNNING:
                    break                       # paused or cancelled between steps
                index = task.get("current_step", 0)
                steps = task.get("steps", [])
                if index >= len(steps):
                    self._finish(task_id)
                    break
                if not await self._run_step(task_id, index):
                    break
        finally:
            restore_unattended()
            self._current = None
        return describe(get_task(task_id) or {})

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
        # One shot: the approval is consumed as the step starts, so a step that
        # fails for some other reason cannot quietly re-use the user's yes.
        approved = task.pop("approved_step", None) == index
        if approved:
            _note(step, "running with the approval you gave in the interface")
        _replace(task)
        _mirror_step(task, index)

        instruction = (
            f"[Autonomous task '{task['name']}', step {index + 1} of "
            f"{len(task['steps'])}. Objective: {task['objective']}]\n{step['instruction']}"
        )
        if step.get("recovery_instruction"):
            instruction += f"\n\n{step['recovery_instruction']}"
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
        if task.get("status") == CANCELLED:
            # Cancelled while this step was running. The step finished anyway -
            # the answer is in hand - so it is recorded as finished and the task
            # stops. Leaving it as "running" would show a step frozen mid-flight
            # forever and lose work that actually happened.
            step["status"] = "done"
            step["result"] = (answer or "")[:MAX_RESULT_CHARS]
            _note(step, "finished, but the task was cancelled before the next step")
            _replace(task)
            return False
        step["status"] = "done"
        step["result"] = (answer or "")[:MAX_RESULT_CHARS]
        if step.get("recoveries"):
            _note(step, f"recovery {step['recoveries']} succeeded")
        step["error"] = None
        step.pop("recovery_instruction", None)
        task["current_step"] = index + 1
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
        if (is_recoverable(error) and recoveries < MAX_RECOVERIES_PER_STEP
                and elapsed < MAX_RECOVERY_SECONDS):
            step["recoveries"] = recoveries + 1
            step["attempts"] = 0                  # the retries reset for the new approach
            step["status"] = "recovering"
            step["error"] = message
            step["recovery_instruction"] = (
                f"The previous attempt failed: {message}. Try a different way of doing "
                "this same step - an alternative source, address or tool. Do not repeat "
                "the approach that just failed, and do not skip the step.")
            _note(step, f"recovery {step['recoveries']} attempted after: {message}")
            _replace(task)
            logger.info(f"Task {task_id} step {index + 1}: recovery "
                        f"{step['recoveries']} after {message}")
            return True

        step["status"] = "failed"
        step["error"] = message
        _replace(task)
        _set_status(task_id, FAILED, error=f"Step {index + 1} failed: {message}")
        self._announce(f"'{task['name']}' failed at step {index + 1}: {message}")
        return False

    def _finish(self, task_id: str) -> None:
        task = get_task(task_id)
        last = next((s["result"] for s in reversed(task.get("steps", []))
                     if s.get("result")), None)
        _set_status(task_id, COMPLETED, result=last)
        self._announce(f"'{task['name']}' is done.")

    def _announce(self, message: str) -> None:
        if not self.notify:
            return
        try:
            result = self.notify(message)
            if asyncio.iscoroutine(result):
                asyncio.get_event_loop().create_task(result)
        except Exception:
            logger.exception("Couldn't deliver a task notification")


# Failures worth trying differently: something that was briefly unavailable, timed
# out, or refused once. Deliberately a short list - treating everything as
# recoverable is how a task spends an afternoon failing in new ways.
_RECOVERABLE = (
    "timeout", "timed out", "temporarily", "unavailable", "connection",
    "network", "unreachable", "rate limit", "too many requests", "503", "502",
    "504", "429", "reset by peer", "try again",
)


def is_recoverable(error: Exception) -> bool:
    """Whether a different approach is worth one attempt.

    Never true for an approval stop or a hard block: those are answers, not
    failures, and retrying them differently would be trying to get around them.
    """
    from core.safety_guard import ConfirmationDenied, PermissionDenied

    if isinstance(error, (ConfirmationDenied, PermissionDenied)):
        return False
    text = str(error).lower()
    return any(marker in text for marker in _RECOVERABLE)


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
        if task.get("status") == RUNNING:
            task["status"] = PAUSED
            task["blocked_reason"] = (
                "Leti restarted while this task was running. Nothing was repeated - "
                "resume it when you are ready.")
            task["updated_at"] = time.time()
            recovered.append(task)
    if recovered:
        save_tasks(tasks)
    return recovered
