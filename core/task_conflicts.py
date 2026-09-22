"""Two tasks, one file. Which one waits.

Phase 8's only genuinely new idea. Everything else about running several tasks
is core/task_manager.py's - the store, the statuses, the runner, the history -
but a second concurrent task introduces a question a single-task runner never
had to answer: can these two run at the same time without ruining each other's
work.

The answer is deliberately conservative and deliberately declarative. A task
says what it needs; two tasks needing the same exclusive thing do not run
together; the second one waits and says why. Nothing here inspects what a task
is actually doing, because that would be guessing, and guessing wrong here means
two tasks writing the same file.

WHAT A RESOURCE IS. A string. `file:/home/u/report.md`, `app:blender`,
`project:Turbine`, `screen`. The point is that they compare by equality, so
declaring one costs nothing and getting one slightly wrong costs a needless
wait rather than a corrupted file - which is the right direction to be wrong in.

WHAT IS EXCLUSIVE. The screen and the keyboard, because there is one of each and
two tasks taking turns with the mouse is not two tasks. A file being written. An
application being driven. Reading is not exclusive: any number of tasks may read
the same file.

THIS SCHEDULES NOTHING. It answers "may this start" and "what is it waiting
for". core/task_manager.py does the waiting.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Set, Tuple

logger = logging.getLogger("leti.conflicts")

# Resource kinds, and whether holding one excludes everybody else.
READ = "read"          # any number at once
WRITE = "write"        # one at a time
CONTROL = "control"    # one at a time, and nothing else may read it either

MODES = (READ, WRITE, CONTROL)

# The one resource there is exactly one of. A task driving the interface and a
# task typing into it are the same task or they are a mess.
SCREEN = "screen"

# What a step's text says it will touch. Deliberately shallow: a path, an app
# name, the screen. Anything it cannot see, it does not claim.
_FILE = re.compile(
    r"(?:^|[\s\"'`(])(?P<path>(?:~|\.{0,2}/)[\w./\-]{2,120}"
    r"|[\w\-]{1,60}\.(?:py|js|ts|md|txt|csv|json|ya?ml|html|css|pdf|docx|xlsx|sql))",
    re.I)
_WRITES = re.compile(
    r"\b(write|save|edit|modify|update|append|delete|remove|move|rename|replace|"
    r"overwrite|create|generate|export|commit|patch|fix)\b", re.I)
_APP = re.compile(
    r"\b(?:open|launch|close|focus|use|in|drive|control)\s+(?:the\s+)?"
    r"(?P<app>blender|chrome|firefox|safari|edge|word|excel|powerpoint|outlook|"
    r"spotify|slack|discord|terminal|vscode|code|photoshop|illustrator|notepad|"
    r"finder|explorer|calculator|mail|calendar|browser)\b", re.I)
_SCREEN = re.compile(
    r"\b(click|type into|keyboard|mouse|screen|window|scroll|drag|press|"
    r"screenshot|gui)\b", re.I)


def declare(task: Dict[str, Any]) -> List[Dict[str, str]]:
    """What a task says it needs, read off its own plan.

    A best-effort reading, and one that only ever ADDS caution: a resource it
    fails to spot means two tasks may collide the way they always could, and a
    resource it spots wrongly means one waits for no reason. Neither is new
    damage, and the second is the cheap one.

    A task may also carry an explicit `resources` list, which is taken as given -
    that is how a caller says something this cannot see from the text.
    """
    declared: Dict[str, str] = {}

    for entry in (task.get("resources") or []):
        if isinstance(entry, dict) and entry.get("name"):
            declared[str(entry["name"])] = str(entry.get("mode") or WRITE)
        elif isinstance(entry, str) and entry.strip():
            declared[entry.strip()] = WRITE

    text = " ".join(filter(None, [
        str(task.get("objective") or ""),
        " ".join(str(s.get("instruction") or "") for s in task.get("steps") or []),
    ]))

    writing = bool(_WRITES.search(text))
    for match in _FILE.finditer(text):
        path = match.group("path").rstrip(".,;:")
        name = f"file:{path}"
        # A file that is written anywhere in the plan is held for writing for
        # the whole task: a plan that reads it in step one and rewrites it in
        # step four cannot share it with anybody in between.
        declared.setdefault(name, WRITE if writing else READ)
        if writing:
            declared[name] = WRITE

    for match in _APP.finditer(text):
        declared[f"app:{match.group('app').lower()}"] = CONTROL

    if _SCREEN.search(text):
        declared[SCREEN] = CONTROL

    if task.get("project"):
        declared.setdefault(f"project:{task['project']}", READ)

    return [{"name": name, "mode": mode} for name, mode in sorted(declared.items())]


def _held(tasks: Iterable[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """resource -> (mode, task id) for everything currently being held."""
    held: Dict[str, Tuple[str, str]] = {}
    for task in tasks:
        for resource in (task.get("declared_resources") or declare(task)):
            name, mode = resource["name"], resource["mode"]
            # The strongest claim wins when one task declares a thing twice.
            existing = held.get(name)
            if existing is None or _rank(mode) > _rank(existing[0]):
                held[name] = (mode, task.get("id"))
    return held


def _rank(mode: str) -> int:
    return {READ: 0, WRITE: 1, CONTROL: 2}.get(mode, 1)


def conflicts(task: Dict[str, Any],
              running: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What stops this task starting while those are running.

    Empty means nothing does. Each entry names the resource, the mode each side
    wants it in, and which task is holding it - enough to tell the user which
    one to stop without them having to work it out.
    """
    wanted = task.get("declared_resources") or declare(task)
    others = [t for t in running if t.get("id") != task.get("id")]
    held = _held(others)
    found = []
    for resource in wanted:
        name, mode = resource["name"], resource["mode"]
        holder = held.get(name)
        if holder is None:
            continue
        holder_mode, holder_id = holder
        # Two readers are not a conflict. Anything else involving a write or a
        # control is.
        if mode == READ and holder_mode == READ:
            continue
        found.append({
            "resource": name,
            "wanted_as": mode,
            "held_as": holder_mode,
            "held_by": holder_id,
            "why": _why(name, mode, holder_mode),
        })
    return found


def _why(name: str, wanted: str, held: str) -> str:
    if name == SCREEN:
        return ("There is one screen and one keyboard. Two tasks driving them at "
                "once do not do two things; they do one broken thing.")
    if name.startswith("app:"):
        return (f"Both tasks want to drive {name[4:]}. Whichever clicks second is "
                "clicking in a window the other one moved.")
    if name.startswith("file:"):
        if CONTROL in (wanted, held) or WRITE in (wanted, held):
            return (f"Both tasks touch {name[5:]} and at least one of them writes to "
                    "it. Running them together can lose one task's work entirely.")
    return f"Both tasks need {name} and at least one of them changes it."


def _runtime_conflicts(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What this task's declaration wants that somebody is ACTUALLY holding.

    The declaration is a prediction; core/resources.py is the truth. A task
    whose plan says it will write report.md must not start while another task
    has report.md open, even if that other task never declared it - which is the
    whole point of tracking what tools really touch.
    """
    try:
        from core import resources
    except Exception:
        return []
    found = []
    for entry in (task.get("declared_resources") or declare(task)):
        name, mode = entry["name"], entry["mode"]
        # The declaration's vocabulary and the ledger's are the same three words
        # for the overlapping cases; a declared CONTROL is a runtime CONTROL.
        blocker = resources.conflict_for(name, mode, str(task.get("id") or ""))
        if blocker is None:
            continue
        found.append({
            "resource": name, "wanted_as": mode, "held_as": blocker.mode,
            "held_by": blocker.task_id, "runtime": True,
            "why": (f"Another task has {name} open right now "
                    f"({blocker.mode}), whatever its plan said it would need."),
        })
    return found


def may_start(task: Dict[str, Any],
              running: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Can this task start right now? The whole answer, including why not.

    Two sources, and they answer different questions. The declarations predict
    what each running task will need; the runtime ledger says what is open at
    this instant. A conflict from either is a conflict.
    """
    found = conflicts(task, running) + _runtime_conflicts(task)
    if not found:
        return {"ok": True, "conflicts": [], "wait_for": []}
    return {
        "ok": False,
        "conflicts": found,
        "wait_for": sorted({c["held_by"] for c in found if c["held_by"]}),
        "explain": ("This task needs something another task is using: "
                    + "; ".join(f"{c['resource']} (held by {c['held_by']})"
                                for c in found[:4])
                    + ". It will start when that one finishes, or you can stop "
                      "the other one."),
    }


def describe(task: Dict[str, Any]) -> Dict[str, Any]:
    """What this task holds, for the panel and for diagnostics."""
    resources = task.get("declared_resources") or declare(task)
    return {
        "resources": resources,
        "exclusive": [r["name"] for r in resources if r["mode"] in (WRITE, CONTROL)],
        "shared": [r["name"] for r in resources if r["mode"] == READ],
    }


def all_conflicts(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every pair of tasks that cannot run together, for the diagnostics panel."""
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for task in tasks:
        for clash in conflicts(task, tasks):
            key = tuple(sorted([str(task.get("id")), str(clash["held_by"])])) + \
                (clash["resource"],)
            if key in seen:
                continue
            seen.add(key)
            out.append({"between": [task.get("id"), clash["held_by"]], **clash})
    return out
