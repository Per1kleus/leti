"""What a task is actually touching, as opposed to what its plan said it would.

core/task_conflicts.py reads a plan and predicts what it will need. That is a
useful early answer and it is a guess: a plan that says "tidy up the project"
never mentions report.md, and two tasks can therefore both start and both write
it. This is the other half - the authoritative one - and it works by watching
what tools are actually called.

    PLAN DECLARATION  (task_conflicts, a prediction)
  + RUNTIME OBSERVATION (here, the truth)
  ->  RESOURCE SET -> CONFLICT CHECK -> ACTION

The ledger is one dictionary in memory. It is not a database and it is not
persisted: a lock held by a process that no longer exists is not a lock, so a
restart starts with nothing held, which is the only correct answer. What a
crashed task HAD is recorded in its checkpoint, where it belongs.

ACQUISITION IS ATOMIC BY CONSTRUCTION. acquire() checks and records in one
synchronous function with no await anywhere inside it, so two coroutines on the
same event loop cannot both succeed - the second one cannot run until the first
has returned. That is the whole fix, and it needs no thread and no lock.

IT GRANTS NOTHING. A resource lock is about two tasks not ruining each other's
work. It is not permission: SafetyGuard still authorises every call, and holding
a lock on a file does not make writing it allowed.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("leti.resources")

# How a resource is being used. Ordered weakest first; the table below decides
# which pairs can coexist.
READ = "read"            # any number at once
WRITE = "write"          # one writer, and no readers
EXCLUSIVE = "exclusive"  # one holder, whatever anybody else wants it for
CONTROL = "control"      # one holder driving something with one of it

MODES = (READ, WRITE, EXCLUSIVE, CONTROL)

# Can a holder in mode A coexist with a requester in mode B? Two readers can;
# nothing else can. Deliberately conservative: the cost of a needless wait is a
# few seconds, and the cost of being wrong is two tasks writing the same file.
_COMPATIBLE = {
    (READ, READ): True,
}


def compatible(held: str, wanted: str) -> bool:
    return _COMPATIBLE.get((held, wanted), False)


# A resource nobody is going to fight over is not worth tracking, and tracking
# everything would mean a lock per log line. These are the kinds where
# concurrency really does corrupt something.
FILE = "file"
DIRECTORY = "dir"
APPLICATION = "app"
SCREEN = "screen"          # there is one, and it has one keyboard
CONNECTION = "connection"  # an account: mail, calendar, GitHub
PROCESS = "process"
BROWSER = "browser"        # one session, one set of cookies
STORE = "store"            # a JSON store Leti itself owns

KINDS = (FILE, DIRECTORY, APPLICATION, SCREEN, CONNECTION, PROCESS, BROWSER, STORE)

# How long a lock may be held before it is reported as suspicious. Nothing is
# force-released on a timer - that would be a background loop, and a task that
# legitimately takes an hour is not a bug - but a lock older than this shows up
# in diagnostics so a stuck task is visible.
LONG_HELD_SECONDS = 900


def identity(kind: str, name: str) -> str:
    """One canonical string for one resource.

    Files are resolved to an absolute real path so that ./report.md,
    ~/proj/report.md and a symlink to it are one resource rather than three -
    which is the difference between detecting a conflict and missing it.
    """
    cleaned = str(name or "").strip()
    if kind in (FILE, DIRECTORY) and cleaned:
        try:
            cleaned = os.path.realpath(os.path.expanduser(cleaned))
        except (OSError, ValueError):
            pass
    elif kind in (APPLICATION, BROWSER, CONNECTION, STORE):
        cleaned = cleaned.lower()
    return f"{kind}:{cleaned}" if cleaned else kind


@dataclass
class Hold:
    """One task's claim on one resource."""
    resource: str
    mode: str
    task_id: str
    since: float = field(default_factory=time.time)
    reason: str = ""

    def age(self, now: Optional[float] = None) -> float:
        return (now if now is not None else time.time()) - self.since

    def describe(self) -> Dict[str, Any]:
        return {"resource": self.resource, "mode": self.mode,
                "task": self.task_id, "held_for_seconds": round(self.age(), 1),
                "reason": self.reason or None,
                "long_held": self.age() > LONG_HELD_SECONDS}


# resource -> the holds on it. One dictionary, in this process, never written
# to disk. A list rather than a single hold because several readers coexist.
_held: Dict[str, List[Hold]] = {}
# Who is waiting for what, so the interface can say so. Purely descriptive:
# nothing here schedules, and the waiting task's own status is the real state.
_waiting: Dict[str, List[Dict[str, Any]]] = {}


def clear() -> None:
    """Drop every lock. Startup and the tests call this; nothing else should.

    Startup is the important one: locks held by a process that has gone are not
    locks, and a restart that inherited them would deadlock against a ghost.
    """
    _held.clear()
    _waiting.clear()


def holders(resource: str) -> List[Hold]:
    return list(_held.get(resource, []))


def conflict_for(resource: str, mode: str,
                 task_id: str) -> Optional[Hold]:
    """The hold that stops this request, or None. Pure; changes nothing."""
    for hold in _held.get(resource, []):
        if hold.task_id == task_id:
            continue                      # a task never conflicts with itself
        if not compatible(hold.mode, mode):
            return hold
    return None


def acquire(resource: str, mode: str, task_id: str,
            reason: str = "") -> Dict[str, Any]:
    """Claim a resource, or say who has it. Atomic with respect to other tasks.

    There is deliberately no `await` anywhere in this function. Python's event
    loop cannot interleave two coroutines inside it, so the check and the record
    are one operation from every other task's point of view - which is the whole
    race this exists to close, solved without a thread or a lock.
    """
    if mode not in MODES:
        raise ValueError(f"{mode!r} is not one of {MODES}")
    resource, task_id = str(resource), str(task_id)

    blocker = conflict_for(resource, mode, task_id)
    if blocker is not None:
        _note_waiting(resource, mode, task_id)
        return {
            "acquired": False,
            "resource": resource,
            "wanted": mode,
            "held_by": blocker.task_id,
            "held_as": blocker.mode,
            "since_seconds": round(blocker.age(), 1),
            "explain": _explain(resource, mode, blocker),
        }

    # Upgrading an existing hold (read -> write on the same file by the same
    # task) replaces it rather than stacking, so release is one call and a task
    # cannot leak a hold by acquiring twice.
    existing = [h for h in _held.get(resource, []) if h.task_id == task_id]
    for hold in existing:
        _held[resource].remove(hold)
    _held.setdefault(resource, []).append(
        Hold(resource=resource, mode=mode, task_id=task_id, reason=reason))
    _stop_waiting(resource, task_id)
    return {"acquired": True, "resource": resource, "mode": mode, "task": task_id}


def release(resource: str, task_id: str) -> bool:
    """Let go of one resource. True if this task was actually holding it."""
    resource, task_id = str(resource), str(task_id)
    holds = _held.get(resource)
    if not holds:
        return False
    remaining = [h for h in holds if h.task_id != task_id]
    released = len(remaining) != len(holds)
    if remaining:
        _held[resource] = remaining
    else:
        _held.pop(resource, None)
    _stop_waiting(resource, task_id)
    return released


def release_all(task_id: str) -> List[str]:
    """Everything this task held. Called when it stops, however it stops.

    "However it stops" is the point: completed, failed, cancelled, paused into
    a state that should not hold locks, or abandoned by a crash. A resource that
    is only released on the happy path is a resource that deadlocks on the
    unhappy one.
    """
    task_id = str(task_id)
    released = []
    for resource in list(_held):
        if release(resource, task_id):
            released.append(resource)
    for resource in list(_waiting):
        _stop_waiting(resource, task_id)
    return released


def held_by(task_id: str) -> List[Dict[str, Any]]:
    task_id = str(task_id)
    return [h.describe() for holds in _held.values() for h in holds
            if h.task_id == task_id]


def _note_waiting(resource: str, mode: str, task_id: str) -> None:
    queue = _waiting.setdefault(resource, [])
    if not any(entry["task"] == task_id for entry in queue):
        queue.append({"task": task_id, "mode": mode, "since": time.time()})


def _stop_waiting(resource: str, task_id: str) -> None:
    queue = _waiting.get(resource)
    if not queue:
        return
    remaining = [entry for entry in queue if entry["task"] != task_id]
    if remaining:
        _waiting[resource] = remaining
    else:
        _waiting.pop(resource, None)


def waiting_for(resource: str) -> List[Dict[str, Any]]:
    return list(_waiting.get(resource, []))


def _explain(resource: str, mode: str, blocker: Hold) -> str:
    kind, _, name = resource.partition(":")
    what = name or kind
    if kind == SCREEN:
        return ("Another task is driving the screen and keyboard. There is one of "
                "each, so two tasks using them at once do not do two things.")
    if kind == APPLICATION:
        return (f"Another task is driving {what}. Whichever clicks second is "
                "clicking in a window the other one moved.")
    if kind == FILE:
        return (f"Another task is using {what} and one of them writes to it. "
                "Running both can lose one task's work entirely.")
    if kind == CONNECTION:
        return (f"Another task is using the {what} connection.")
    return f"Another task holds {resource} as {blocker.mode}."


def describe_conflict(outcome: Dict[str, Any],
                      owner_name: str = "") -> str:
    """The sentence a person reads. Names the resource, the owner and the wait."""
    owner = owner_name or outcome.get("held_by") or "another task"
    resource = outcome.get("resource", "a resource")
    _, _, name = str(resource).partition(":")
    return (f"{owner} is currently using {name or resource} "
            f"({outcome.get('held_as')}). This task needs "
            f"{outcome.get('wanted')} access to the same thing, so it is waiting. "
            f"{outcome.get('explain', '')}").strip()


# --------------------------------------------------------------------------- #
# Working out what a tool is about to touch
#
# The instrumentation point is the tool boundary and nowhere else. No Python
# tracing, no filesystem watcher, no scanning: the arguments a tool was called
# with already say what it is about to use, and this reads them.
#
# A tool that is not in this table touches nothing worth tracking, which is the
# honest default - inventing a resource for every call would put a lock around
# get_weather.
# --------------------------------------------------------------------------- #

# tool -> (argument names to look at, kind, mode). The first argument present
# wins, so a tool that names its path differently still resolves.
#
# A name here that is not a real tool is worse than an omission: the tool that
# replaced it holds no resource, so two tasks writing the same store are not
# serialised and nothing says why. create_document, add_business_data and
# update_business_data were all gone. Checked against the registry by
# tests/test_architecture_and_performance.py.
_TOOL_RESOURCES: Dict[str, Tuple[Tuple[str, ...], str, str]] = {
    "read_file":        (("path", "file_path"), FILE, READ),
    "read_document":    (("path", "file_path"), FILE, READ),
    "inspect_document": (("path", "file_path"), FILE, READ),
    "write_file":       (("path", "file_path"), FILE, WRITE),
    "delete_file":      (("path", "file_path"), FILE, WRITE),
    "list_files":       (("path", "directory", "folder"), DIRECTORY, READ),
    "run_code":         (("path", "file_path"), FILE, READ),
    "launch_app":       (("app_name", "name"), APPLICATION, CONTROL),
    "close_app":        (("window_title", "app_name", "name"), APPLICATION, CONTROL),
    "focus_window":     (("window_title", "app_name", "name"), APPLICATION, CONTROL),
    "mouse_click":      ((), SCREEN, CONTROL),
    "keyboard_type":    ((), SCREEN, CONTROL),
    "keyboard_hotkey":  ((), SCREEN, CONTROL),
    "read_screen":      ((), SCREEN, READ),
    "browser_read_page":  ((), BROWSER, CONTROL),
    "browser_click":      ((), BROWSER, CONTROL),
    "browser_fill_form":  ((), BROWSER, CONTROL),
    "send_email":       ((), CONNECTION, CONTROL),
    "list_new_emails":  ((), CONNECTION, READ),
    "schedule_meeting": ((), CONNECTION, CONTROL),
    "record_business_data": ((), STORE, WRITE),
    "update_lead":          ((), STORE, WRITE),
    "list_business_data":   ((), STORE, READ),
}

# The fixed name a tool with no naming argument uses. "the screen" is one
# resource whoever is touching it.
_FIXED_NAMES = {
    SCREEN: "", BROWSER: "default", CONNECTION: "", STORE: "business",
}

# Which connection each tool uses, so two mail tools share a resource and a mail
# tool and a calendar tool do not.
_CONNECTION_OF = {
    "send_email": "email", "list_new_emails": "email",
    "schedule_meeting": "calendar",
}

# A move has two ends and both of them matter.
_TWO_ENDED = {
    "move_file": (("source_path", "source"), ("destination_path", "destination")),
}


def _first(arguments: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def for_tool(tool_name: str,
             arguments: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """What this call is about to touch. Empty for most tools, and that is right.

    Never raises: a resource reading that fails must not stop a tool running.
    The cost of missing one is the behaviour Leti had before runtime tracking;
    the cost of an exception here would be a tool that does not run at all.
    """
    arguments = arguments if isinstance(arguments, dict) else {}
    name = str(tool_name or "")
    try:
        if name in _TWO_ENDED:
            source_keys, destination_keys = _TWO_ENDED[name]
            found = []
            source = _first(arguments, source_keys)
            destination = _first(arguments, destination_keys)
            if source:
                found.append({"resource": identity(FILE, source), "mode": WRITE})
            if destination:
                found.append({"resource": identity(FILE, destination), "mode": WRITE})
            return found

        spec = _TOOL_RESOURCES.get(name)
        if spec is None:
            return []
        keys, kind, mode = spec
        if keys:
            value = _first(arguments, keys)
            if not value:
                return []
        else:
            value = _CONNECTION_OF.get(name) if kind == CONNECTION \
                else _FIXED_NAMES.get(kind, "")
        return [{"resource": identity(kind, value or ""), "mode": mode}]
    except Exception as e:
        logger.debug(f"Couldn't read resources for {name}: {e}")
        return []


def snapshot() -> Dict[str, Any]:
    """Everything held and everything waiting. Read-only, for diagnostics."""
    return {
        "held": [hold.describe() for holds in _held.values() for hold in holds],
        "waiting": [{"resource": resource, **entry}
                    for resource, queue in _waiting.items() for entry in queue],
        "long_held": [hold.describe() for holds in _held.values() for hold in holds
                      if hold.age() > LONG_HELD_SECONDS],
    }
