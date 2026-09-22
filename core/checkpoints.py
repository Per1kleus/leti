"""The last thing Leti actually knew, written down before it could be lost.

A task that was running when the process died leaves a status saying RUNNING and
nothing else. From that, two very different situations look identical: the step
had not started yet, and the step had sent an email and the process died before
the answer came back. Treating the second as the first resends the email.

So every task carries a compact checkpoint, and the checkpoint records the one
thing status cannot: what state the step in flight was in.

    NOT_STARTED          nothing has been attempted
    STARTED              it was attempted and the outcome is not yet known
    VERIFIED             it finished and was checked
    UNKNOWN_AFTER_CRASH  it was STARTED when the process disappeared

That last one is the whole point. It is not "failed" - failing is a thing that
was observed. It is not "done" either. It is the honest answer, and what
happens next depends on whether repeating the step could do damage.

WHERE IT LIVES. Inside the task, in data/autonomous_tasks.json, written by
core/task_manager.py's existing save path, which already goes through
core/atomic_write.py. There is no second store and no new database: a checkpoint
is a field on a record that was already being written at exactly the moments a
checkpoint should be taken.

WHAT IT DOES NOT HOLD. No screenshots, no accessibility trees, no model context,
no conversation. A step's text is truncated, results are a few hundred
characters, and anything larger is referenced rather than copied. The whole
thing is a few hundred bytes.

GENERATIONS. Each process gets an id when it starts. A checkpoint carries the id
of the process that wrote it, so on startup "written by a process that is gone"
is a comparison rather than a guess - no marker file to clean up, no heartbeat,
nothing to leave behind when the power cuts.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from core import verification

logger = logging.getLogger("leti.checkpoints")

# How a step's execution stood when the checkpoint was written.
#
# VERIFIED is core/verification.py's, imported rather than restated: a step that
# finished and was checked means the same thing here as it does everywhere else
# in Leti, and two constants spelling it would be two things to keep in step.
NOT_STARTED = "NOT_STARTED"
STARTED = "STARTED"
VERIFIED = verification.VERIFIED
UNKNOWN_AFTER_CRASH = "UNKNOWN_AFTER_CRASH"

EXECUTION_STATES = (NOT_STARTED, STARTED, VERIFIED, UNKNOWN_AFTER_CRASH)

# What a restart decided about one interrupted task.
READY_TO_RESUME = "READY_TO_RESUME"   # safe, and nothing irreversible is in doubt
NEEDS_YOU = "NEEDS_YOU"               # a person has to say what happened
WAITING = "WAITING"                   # it was already waiting; it still is
STOPPED = "STOPPED"                   # it was cancelled, or cannot be continued

RECOVERY_STATES = (READY_TO_RESUME, NEEDS_YOU, WAITING, STOPPED)

MAX_STEP_CHARS = 200
MAX_RESULT_CHARS = 300

# This process. Made once, at import, from nothing that touches the disk.
_GENERATION = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def generation() -> str:
    """The id of the process that is running now."""
    return _GENERATION


def new_generation_for_tests() -> str:
    """Pretend to be a different process. Only the tests call this."""
    global _GENERATION
    _GENERATION = f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    return _GENERATION


def _short(text: Any, limit: int = MAX_STEP_CHARS) -> Optional[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    return cleaned if len(cleaned) <= limit else cleaned[:limit - 1] + "…"


def write(task: Dict[str, Any], execution: str = NOT_STARTED,
          note: str = "") -> Dict[str, Any]:
    """Build the checkpoint for a task as it stands. Pure; saves nothing.

    The caller persists it by putting it on the task and saving the task, which
    is how it ends up going through the atomic write that already existed.
    """
    if execution not in EXECUTION_STATES:
        raise ValueError(f"{execution!r} is not one of {EXECUTION_STATES}")

    steps = task.get("steps") or []
    index = task.get("current_step", 0)
    current = steps[index] if 0 <= index < len(steps) else None
    verification = task.get("verification") or {}

    held: List[Dict[str, Any]] = []
    try:
        from core import resources

        held = [{"resource": h["resource"], "mode": h["mode"]}
                for h in resources.held_by(str(task.get("id") or ""))]
    except Exception:
        held = []

    return {
        "task_id": task.get("id"),
        "goal": _short(task.get("objective"), 300),
        "status": task.get("status"),
        "step_index": index,
        "step": _short(current.get("instruction")) if current else None,
        "execution": execution,
        # Steps that finished AND were checked. The recovery boundary: anything
        # before this is known to have happened, anything after it is not.
        "verified_steps": [s.get("n", i + 1) for i, s in enumerate(steps)
                           if s.get("status") == "done"],
        "failed_steps": [s.get("n", i + 1) for i, s in enumerate(steps)
                         if s.get("status") == "failed"],
        "recoveries": sum(s.get("recoveries", 0) for s in steps),
        "last_verification": ({"passed": verification.get("passed"),
                               "said": _short(verification.get("said"), MAX_RESULT_CHARS)}
                              if verification else None),
        "resources": held,
        "irreversible": _looks_irreversible(current),
        "generation": _GENERATION,
        "at": time.time(),
        "note": _short(note, MAX_RESULT_CHARS),
    }


def _looks_irreversible(step: Optional[Dict[str, Any]]) -> bool:
    """Could repeating this step do something that cannot be taken back?

    core/computer_use.py already owns the list of words that mean "this leaves
    the machine and is somebody else's problem if it is wrong" - send, submit,
    delete, pay, publish. Reused rather than restated, because two lists of
    what counts as irreversible would drift and the drift would be silent.
    """
    if not step:
        return False
    try:
        from core.computer_use import is_consequential

        return bool(is_consequential(step.get("instruction") or ""))
    except Exception:
        # Fail closed: unable to tell means treat it as irreversible, because
        # the cost of being wrong the other way is a second email.
        return True


def of(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The checkpoint on a task, if it has one and the one it has is readable."""
    checkpoint = task.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("task_id"):
        return None
    if checkpoint.get("execution") not in EXECUTION_STATES:
        return None
    return checkpoint


def from_another_process(checkpoint: Optional[Dict[str, Any]]) -> bool:
    """Was this written by a process that is no longer running?

    A checkpoint carrying a different generation than this process was written
    by one that has gone - there is no other way for it to exist. A checkpoint
    with no generation at all predates checkpointing and is treated the same
    way, because it cannot be shown to belong to this process.
    """
    if not checkpoint:
        return False
    return checkpoint.get("generation") != _GENERATION


# --------------------------------------------------------------------------- #
# What a restart should do about one interrupted task
# --------------------------------------------------------------------------- #

def assess(task: Dict[str, Any], checkpoint: Optional[Dict[str, Any]] = None,
           permitted: Optional[bool] = None,
           connections_ready: Optional[bool] = None) -> Dict[str, Any]:
    """Whether this task can safely carry on, and if not, what is needed.

    Every condition has to hold. Any one of them failing is NEEDS_YOU, which is
    a stop, not a failure - the task is intact and a person has to say what to
    do. There is deliberately no path from here to "resume anyway".
    """
    checkpoint = checkpoint if checkpoint is not None else of(task)
    blockers: List[str] = []
    unknown_effect = False

    if checkpoint is None:
        return {
            "state": NEEDS_YOU,
            "resume": False,
            "why": ["there is no checkpoint, so what it had already done is not known"],
            "unknown_external_effect": False,
            "last_verified_step": None,
            "explain": ("This task was interrupted and Leti has no record of how far "
                        "it got. Tell it what to do rather than letting it guess."),
        }

    execution = checkpoint.get("execution")
    verified = checkpoint.get("verified_steps") or []
    last_verified = max(verified) if verified else None

    if execution in (STARTED, UNKNOWN_AFTER_CRASH):
        # The step was in flight. Whether it happened is not knowable from here.
        unknown_effect = True
        if checkpoint.get("irreversible"):
            blockers.append(
                "the step in flight could have sent, deleted or paid something, and "
                "whether it did is unknown - repeating it could do it twice")
        else:
            blockers.append(
                "the step in flight did not finish, and whether it did anything is "
                "unknown")

    if permitted is False:
        blockers.append("a permission it needs has changed since it was interrupted")
    if connections_ready is False:
        blockers.append("a connection it needs is not available")

    if blockers:
        return {
            "state": NEEDS_YOU,
            "resume": False,
            "why": blockers,
            "unknown_external_effect": unknown_effect,
            "last_verified_step": last_verified,
            "explain": _explain(task, checkpoint, blockers, last_verified),
        }

    return {
        "state": READY_TO_RESUME,
        "resume": True,
        "why": [],
        "unknown_external_effect": False,
        "last_verified_step": last_verified,
        "explain": (
            f"Interrupted after step {last_verified} finished and was checked. "
            "Nothing was in flight, so carrying on from the next step repeats "
            "nothing." if last_verified else
            "Interrupted before anything had been attempted, so starting it "
            "repeats nothing."),
    }


def _explain(task: Dict[str, Any], checkpoint: Dict[str, Any],
             blockers: List[str], last_verified: Optional[int]) -> str:
    name = task.get("name") or task.get("objective") or "A task"
    where = (f"It got as far as step {last_verified}, which finished and was checked."
             if last_verified else "Nothing had been confirmed finished.")
    step = checkpoint.get("step")
    doing = f" It was in the middle of: {step}." if step else ""
    return (f"'{name}' was interrupted when Leti stopped. {where}{doing} "
            f"It cannot carry on by itself because " + "; ".join(blockers) + ".")


def describe(task: Dict[str, Any]) -> Dict[str, Any]:
    """One interrupted task, for diagnostics and for telling the user."""
    checkpoint = of(task)
    verdict = assess(task, checkpoint)
    return {
        "task": task.get("id"),
        "name": task.get("name") or task.get("objective"),
        "state": verdict["state"],
        "resume": verdict["resume"],
        "last_verified_step": verdict["last_verified_step"],
        "execution_at_crash": (checkpoint or {}).get("execution"),
        "unknown_external_effect": verdict["unknown_external_effect"],
        "held_resources": (checkpoint or {}).get("resources") or [],
        "why": verdict["why"],
        "explain": verdict["explain"],
        "checkpoint_age_seconds": (round(time.time() - checkpoint["at"], 1)
                                   if checkpoint and checkpoint.get("at") else None),
    }
