"""Things worth telling the user without being asked - and nothing more than telling.

Everything here is already known to Leti. A task stopped and is waiting for an
answer (core/task_manager.py). A watch fired and nobody has heard about it yet
(core/watches.py). A GUI errand was left open with something consequential in it
that was never checked (core/computer_use.py). A scheduled task is about to run
(tools/scheduler.py). Four stores that exist, read on a turn that was happening
anyway.

What this adds is the judgement about whether to bring one up, and the discipline
about what "bring up" means:

    SUGGEST -> ASK -> the user says yes -> the ordinary path -> SafetyGuard

This module cannot execute anything. It has no tool, it calls nothing, and the
most consequential thing it produces is a sentence. When the user says yes to one
of its suggestions, what happens next is an ordinary turn with an ordinary
request in it, authorised call by call exactly as if they had thought of it
themselves. That is the whole design: proactivity is about noticing, and noticing
has never been permission.

There is no loop, no timer and no thread. Items are assembled when something is
already asking - the interface refreshing a panel, or a turn being built - and a
quiet period keeps the same thing from being raised twice in a row.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.proactive")

# How much Leti is allowed to volunteer, quietest first. The level is a setting
# (core/settings_editor.py, "proactive"), and OFF means this module produces
# nothing at all rather than producing things nobody shows.
OFF = "off"
SUGGESTIONS = "suggestions"          # only while the user is already talking
NOTIFICATIONS = "notifications"      # low-risk facts may appear on their own
ACTIVE = "active"                    # and an offer of the obvious next step
LEVELS = (OFF, SUGGESTIONS, NOTIFICATIONS, ACTIVE)
DEFAULT_LEVEL = NOTIFICATIONS

_RANK = {level: i for i, level in enumerate(LEVELS)}

QUIET_SECONDS = 600         # the same item is not raised twice within ten minutes
MAX_ITEMS = 3               # three things is a prompt; ten is a feed
SOON_SECONDS = 1_800        # "about to run" means the next half hour
RECENT_TRIGGER_SECONDS = 3_600

# What has been raised, and when. In process and bounded: this is a quiet period,
# not a record, and the stores it reads are the ones that remember things. A
# restart may surface one item a second time, which is the honest cost of not
# building a second notification store to avoid it.
_raised: Dict[str, float] = {}


def level() -> str:
    """How proactive the user has asked Leti to be."""
    try:
        from core.config_loader import get_settings

        chosen = str(get_settings().get("proactive", {}).get("level", DEFAULT_LEVEL)).lower()
        return chosen if chosen in LEVELS else DEFAULT_LEVEL
    except Exception:
        return DEFAULT_LEVEL


def allows(needed: str, current: Optional[str] = None) -> bool:
    current = current or level()
    return _RANK.get(current, 0) >= _RANK.get(needed, 99)


def _fresh(key: str, now: float) -> bool:
    """Whether this is worth raising again yet. Prunes as it goes."""
    for old in [k for k, at in _raised.items() if now - at > QUIET_SECONDS * 4]:
        _raised.pop(old, None)
    last = _raised.get(key)
    return last is None or (now - last) > QUIET_SECONDS


def mark_raised(items: List[Dict[str, Any]], now: Optional[float] = None) -> None:
    """Remember that these were shown, so the next turn does not repeat them."""
    now = now if now is not None else time.time()
    for item in items or []:
        if item.get("key"):
            _raised[item["key"]] = now


def items(now: Optional[float] = None, current_level: Optional[str] = None) -> List[Dict[str, Any]]:
    """What is worth mentioning, most important first. Never raises."""
    try:
        return _items(now if now is not None else time.time(), current_level or level())
    except Exception as e:
        logger.debug(f"Proactive items unavailable ({e}).")
        return []


def _items(now: float, current: str) -> List[Dict[str, Any]]:
    if current == OFF:
        return []

    found: List[Dict[str, Any]] = []
    found.extend(_waiting_tasks(now))
    found.extend(_triggered_watches(now))
    found.extend(_unfinished_errands(now))
    if allows(NOTIFICATIONS, current):
        found.extend(_work_due_soon(now))

    # An approval is the only thing that is actually blocking the user; it goes
    # first however many watches fired.
    order = {"approval": 0, "watch": 1, "errand": 2, "schedule": 3}
    found.sort(key=lambda i: (order.get(i["kind"], 9), -i.get("at", 0)))
    visible = [i for i in found if allows(i["needs_level"], current) and _fresh(i["key"], now)]
    return visible[:MAX_ITEMS]


def _waiting_tasks(now: float) -> List[Dict[str, Any]]:
    """A task that stopped to ask. The user is the only thing that can move it."""
    from core import task_manager

    out = []
    for task in task_manager.load_tasks():
        if task.get("status") != task_manager.WAITING_FOR_USER:
            continue
        out.append({
            "kind": "approval",
            "key": f"approval:{task['id']}",
            "at": task.get("updated_at", 0),
            "title": f"'{task.get('name')}' is waiting for your approval.",
            "detail": task.get("blocked_reason") or "A step needs permission to continue.",
            "why": "The task stopped at that step rather than working around it.",
            "suggestion": "Approve it in the task panel and it carries on from there.",
            # Saying that something is waiting is not doing it. Approving is the
            # user's act, in the interface, and the step then runs through the
            # orchestrator and the guard like every other step.
            "needs_approval": True,
            "needs_level": SUGGESTIONS,
            "task_id": task["id"],
        })
    return out


def _triggered_watches(now: float) -> List[Dict[str, Any]]:
    """A watch that fired recently, with the evidence it fired on."""
    from core import watches

    out = []
    for watch in watches.load_watches():
        fired = watch.get("last_triggered_at")
        if not fired or (now - fired) > RECENT_TRIGGER_SECONDS:
            continue
        history = watch.get("history") or []
        last = history[-1] if history else {}
        why = last.get("why") or watches.describe_condition(watch)
        out.append({
            "kind": "watch",
            "key": f"watch:{watch['id']}:{int(fired)}",
            "at": fired,
            "title": f"Your '{watch.get('name')}' watch triggered.",
            "detail": why,
            "why": f"Watching {watches.describe_condition(watch)} - {watches.provider(watch)}.",
            "suggestion": "Ask about it and Leti will look into what happened.",
            "needs_approval": False,
            # A watch firing is a fact, and a fact the user asked to be told.
            "needs_level": NOTIFICATIONS,
            "watch_id": watch["id"],
        })
    return out


def _unfinished_errands(now: float) -> List[Dict[str, Any]]:
    """A GUI errand left open with something consequential nobody looked at."""
    from core import computer_use

    out = []
    for session in computer_use.active_sessions():
        unverified = session.unverified_actions()
        if not unverified:
            continue
        out.append({
            "kind": "errand",
            "key": f"errand:{session.id}",
            "at": session.started_at,
            "title": f"The errand '{session.goal}' has something unchecked.",
            "detail": f"{len(unverified)} action(s) that change something were never "
                      "looked at afterwards.",
            "why": "Leti does not report those as done without seeing the result.",
            "suggestion": "Ask Leti to look at the screen and say where it got to.",
            "needs_approval": False,
            "needs_level": SUGGESTIONS,
            "session_id": session.id,
        })
    return out


def _work_due_soon(now: float) -> List[Dict[str, Any]]:
    """A scheduled task about to run. Read from the scheduler's own store.

    Deliberately only scheduled work. "You have a meeting in 30 minutes" would
    need to READ a calendar, and Leti can create calendar events but has no tool
    that reads one back - so that particular suggestion is one Leti cannot
    honestly make, and it does not pretend to.
    """
    from core.watches import SCHEDULER_TASK_NAME
    from tools import scheduler

    out = []
    for task in scheduler.load_tasks():
        due = task.get("next_run")
        if not task.get("enabled") or not due:
            continue
        if not (now <= due <= now + SOON_SECONDS):
            continue
        if task.get("name") == SCHEDULER_TASK_NAME:
            continue                      # Leti's own watch sweep is not news
        out.append({
            "kind": "schedule",
            "key": f"schedule:{task.get('id')}:{int(due)}",
            "at": due,
            "title": f"'{task.get('name')}' is scheduled to run at "
                     f"{time.strftime('%H:%M', time.localtime(due))}.",
            "detail": str(task.get("instruction", ""))[:200],
            "why": "It is on the schedule you set.",
            "suggestion": "Nothing to do - it runs on its own.",
            "needs_approval": False,
            "needs_level": NOTIFICATIONS,
        })
    return out


def turn_note(found: List[Dict[str, Any]], current: Optional[str] = None) -> str:
    """One line for the model, when there is something worth mentioning.

    Never an instruction to act. At every level this says what is true and, at the
    most forward level, that Leti may OFFER - which is a question, and a question
    is not an action.
    """
    current = current or level()
    if not found or current == OFF:
        return ""
    lines = [f"- {i['title']} {i['detail']}".strip() for i in found]
    note = ("Worth mentioning to the user, briefly and only if it fits what they are "
            "asking about:\n" + "\n".join(lines))
    if allows(ACTIVE, current):
        note += ("\nYou may offer to look into one of these. Offer - do not start. "
                 "Anything you would actually do still needs them to say yes, and still "
                 "goes through the ordinary permission checks when they do.")
    else:
        note += "\nMention them; do not act on them unless the user asks."
    return note
