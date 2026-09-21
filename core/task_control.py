"""Turning "stop" into the right task stopping.

Phase 3's plumbing, kept out of the orchestrator because it is a decision -
WHICH task did they mean - and decisions about which entity is meant belong
with the other ones.

It owns nothing. core/task_manager.py owns the lifecycle and every transition
here goes through its functions; core/entities.py owns the choose / ask /
nothing-found rule and this hands it the candidates. What this adds is the one
thing neither of them had: the mapping from a spoken command to a task, with
the same three honest outcomes.

    one candidate            -> do it
    several candidates       -> say which, and ask
    none                     -> say there is nothing to stop

STOP IS DIFFERENT, and deliberately so. When the user says "stop" with no
qualifier and several tasks are running, stopping all of them is the safe
reading: a stop that asks a question while an action it was meant to prevent
goes ahead is a stop that did not work. Every other control asks.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from core import entities, intent as intent_reader, task_history, task_manager

logger = logging.getLogger("leti.task_control")

# Which task_manager function each spoken command runs. Nothing is implemented
# here; this is a lookup from a word to the existing transition.
_MOVES: Dict[str, Callable[..., Any]] = {
    intent_reader.STOP: task_manager.cancel,
    intent_reader.PAUSE: task_manager.pause,
    intent_reader.RESUME: task_manager.resume,
    intent_reader.RETRY: task_manager.retry,
    intent_reader.SKIP: task_manager.skip_step,
}

# Controls that make sense for a task that has already finished. Only one does:
# you can retry a failed task, and nothing else can be done to a finished one.
_WORKS_ON_FINISHED = (intent_reader.RETRY,)


def _candidates(action: str, hint: str) -> List[Dict[str, Any]]:
    """The tasks a control could apply to, narrowed by what it can apply to."""
    tasks = task_manager.load_tasks()
    if action in _WORKS_ON_FINISHED:
        allowed = task_manager.ALLOWED_FROM.get("retry", ())
        pool = [t for t in tasks if t.get("status") in allowed]
    else:
        key = {intent_reader.STOP: "cancel"}.get(action, action)
        allowed = task_manager.ALLOWED_FROM.get(key, task_manager.ACTIVE_STATUSES)
        pool = [t for t in tasks if t.get("status") in allowed]
    if not hint:
        return pool
    exact = [t for t in pool if t.get("id") == hint]
    if exact:
        return exact
    # Which one they meant is core/entities.py's question. Tasks are given to it
    # in the shape it reads - a name and whatever else distinguishes them.
    outcome = entities.choose(
        [{"name": f"{t.get('name') or ''} {t.get('objective') or ''}".strip(),
          "id": t.get("id"), "status": t.get("status"),
          "updated_at": t.get("updated_at")} for t in pool],
        hint, entities.gather(), kind="task")
    if outcome.get("resolved"):
        return [t for t in pool if t.get("id") == outcome["resolved"].get("id")]
    chosen_ids = {c.get("id") for c in outcome.get("candidates") or []}
    return [t for t in pool if t.get("id") in chosen_ids] or []


def _summarise(task: Dict[str, Any]) -> str:
    detail = task_manager.detail(task)
    where = f"step {detail['step']}" if detail.get("steps_total") else "no steps"
    return f"'{detail['name'] or detail['objective']}' ({detail['status']}, {where})"


def apply(command: Dict[str, Any],
          runner: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Carry out one lifecycle command, or say why it could not be.

    Returns None when the command is not one this handles, so the caller falls
    through to an ordinary turn. Never raises: a control that explodes must not
    take the turn with it.
    """
    try:
        return _apply(command, runner)
    except Exception as e:
        logger.exception("A lifecycle command failed")
        return {"handled": True, "ok": False,
                "answer": f"I couldn't do that to the task: {e}"}


def _apply(command: Dict[str, Any], runner: Optional[Any]) -> Optional[Dict[str, Any]]:
    action = command.get("action")
    hint = (command.get("hint") or "").strip()

    if action == intent_reader.STATUS:
        return {"handled": True, "ok": True, "answer": status_report()}
    if action == intent_reader.WAITING:
        return {"handled": True, "ok": True, "answer": waiting_report()}

    move = _MOVES.get(action)
    if move is None:
        return None

    candidates = _candidates(action, hint)

    if not candidates:
        return {"handled": True, "ok": False, "answer": _nothing_to_do(action, hint)}

    # "Stop" with nothing named and several things running stops all of them.
    # Asking which one, while the thing it was meant to prevent carries on, is a
    # stop that did not work.
    stop_everything = (action == intent_reader.STOP
                       and (command.get("everything") or not hint))
    if len(candidates) > 1 and not stop_everything:
        listed = "; ".join(_summarise(t) for t in candidates[:5])
        return {"handled": True, "ok": False, "needs_choice": True,
                "answer": (f"There is more than one task that could be {action}ed - "
                           f"which one? {listed}")}

    done, failed = [], []
    for task in (candidates if stop_everything else candidates[:1]):
        before = task_manager.detail(task)
        result = move(task["id"])
        if result is None:
            failed.append(f"{_summarise(task)} cannot be {action}ed from "
                          f"{task.get('status')}")
            continue
        done.append({"task": task["id"], "was": before["status"],
                     "now": result.get("status"),
                     "steps_done": before.get("steps_done"),
                     "summary": _summarise(result)})

    # Resume and retry need something to pick the task up again. Saying a task
    # resumed when nothing is going to run it would be exactly the kind of false
    # success this whole layer exists to prevent.
    started = []
    if action in (intent_reader.RESUME, intent_reader.RETRY, intent_reader.SKIP):
        for entry in done:
            if runner is not None and runner.start_in_background(entry["task"]):
                started.append(entry["task"])

    return {"handled": True, "ok": bool(done), "moved": done, "started": started,
            "answer": _say(action, done, failed, started)}


def _say(action: str, done: List[Dict[str, Any]], failed: List[str],
         started: List[str]) -> str:
    if not done:
        return ("I couldn't do that: " + "; ".join(failed)) if failed else \
            "There was nothing to do that to."
    lines = []
    for entry in done:
        line = f"{entry['summary']} - {entry['was']} → {entry['now']}"
        if entry.get("steps_done"):
            line += f", {entry['steps_done']} step(s) already carried out"
        lines.append(line)
    answer = f"{action.capitalize()}: " + "; ".join(lines) + "."
    if action == intent_reader.STOP:
        answer += (" Nothing further will start. Anything already sent or written "
                   "before now has still happened.")
    if action in (intent_reader.RESUME, intent_reader.RETRY, intent_reader.SKIP) \
            and not started:
        answer += (" It is queued, but nothing is running it right now - it will "
                   "pick up when the runner is next free.")
    if failed:
        answer += " " + "; ".join(failed) + "."
    return answer


def _nothing_to_do(action: str, hint: str) -> str:
    active = [t for t in task_manager.load_tasks()
              if t.get("status") in task_manager.ACTIVE_STATUSES]
    if hint and active:
        return (f"Nothing matching '{hint}' can be {action}ed. Active right now: "
                + "; ".join(_summarise(t) for t in active[:5]) + ".")
    if hint:
        return f"Nothing matching '{hint}' is running, so there is nothing to {action}."
    return f"Nothing is running, so there is nothing to {action}."


def status_report() -> str:
    """What Leti is doing, in the order a person would want to hear it."""
    tasks = task_manager.load_tasks()
    active = [t for t in tasks if t.get("status") in task_manager.ACTIVE_STATUSES]
    if not active:
        recent = task_history.recent(limit=1)
        if recent:
            described = task_history.describe(recent[0])
            return (f"Nothing is running. The last thing was {described['name'] or 'a task'}"
                    f" ({described['status']}, {described['finished'] or 'recently'}).")
        return "Nothing is running."

    needs_you = [t for t in active if t.get("status") == task_manager.WAITING_FOR_USER]
    waiting = [t for t in active
               if t.get("status") == task_manager.WAITING_FOR_EXTERNAL]
    running = [t for t in active if t.get("status") in
               (task_manager.RUNNING, task_manager.QUEUED, task_manager.RETRYING,
                task_manager.RESUMING)]
    paused = [t for t in active if t.get("status") in
              (task_manager.PAUSED, task_manager.PAUSING)]

    counts = [f"{len(active)} active task{'s' if len(active) != 1 else ''}"]
    if needs_you:
        counts.append(f"{len(needs_you)} needs you")
    if waiting:
        counts.append(f"{len(waiting)} waiting")
    if running:
        counts.append(f"{len(running)} running")
    if paused:
        counts.append(f"{len(paused)} paused")

    lines = [", ".join(counts) + "."]
    for task in needs_you + running + waiting + paused:
        detail = task_manager.detail(task)
        line = f"- {detail['name'] or detail['objective']}: {detail['status']}"
        if detail.get("steps_total"):
            line += f", step {detail['step']}"
        if detail.get("current"):
            line += f" - {detail['current']}"
        if detail.get("waiting_for_you"):
            line += f" - {detail['waiting_for_you']}"
        lines.append(line)
    return "\n".join(lines)


def waiting_report() -> str:
    """What is blocked, and on what. Only the things a person can act on."""
    tasks = task_manager.load_tasks()
    needs_you = [t for t in tasks if t.get("status") == task_manager.WAITING_FOR_USER]
    external = [t for t in tasks
                if t.get("status") == task_manager.WAITING_FOR_EXTERNAL]
    if not needs_you and not external:
        return "Nothing is waiting on anything."
    lines = []
    for task in needs_you:
        detail = task_manager.detail(task)
        lines.append(f"- {detail['name'] or detail['objective']}: "
                     f"{detail.get('approval_request') or 'needs your approval'}")
    for task in external:
        detail = task_manager.detail(task)
        lines.append(f"- {detail['name'] or detail['objective']}: waiting on something "
                     f"outside this machine - {detail.get('blocked_reason') or 'unspecified'}")
    head = ("Waiting for you:" if needs_you
            else "Nothing needs you. Waiting on something else:")
    return head + "\n" + "\n".join(lines)
