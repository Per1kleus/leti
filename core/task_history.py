"""What happened to a task after it stopped mattering to the runner.

core/task_manager.py owns tasks while they are alive: the steps, the statuses,
the retries, the approvals. It also trims - a state file that keeps every task
forever becomes a slow state file - and the trim is why "what happened to that
task from yesterday" used to have no answer.

So this is the other half of the SAME system, not a second one. It owns no
lifecycle, starts nothing, runs nothing and decides nothing. It is a compact,
append-only record written by task_manager on the transitions task_manager was
already making, and read back when somebody asks about a task that is no longer
in flight.

COMPACT IS THE POINT. A record is a summary, not an archive: step instructions
are truncated, results are a few hundred characters, and the conversation that
produced the task is not copied here at all - it is in the session log, which is
where conversation belongs. A history that stores everything is a history nobody
can load.

AND IT DOES NOT PRETEND. A record that is missing the thing being asked about
says so. "Continue the task from yesterday" against a record with no remaining
plan is answered with what IS known and a question, never with a confident
resumption of something Leti cannot actually describe.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from core.atomic_write import atomic_write_text
from core.config_loader import resolve_path

logger = logging.getLogger("leti.task_history")

STORE_PATH = "./data/task_history.json"

# Bounded by construction. Two hundred finished tasks is months of ordinary use
# and a file of a few hundred kilobytes; beyond that the oldest go.
MAX_RECORDS = 200
MAX_STEP_CHARS = 160
MAX_RESULT_CHARS = 600
MAX_STEPS_RECORDED = 40
MAX_RECOVERIES_RECORDED = 20


def store_path():
    return resolve_path(STORE_PATH)


def load() -> List[Dict[str, Any]]:
    """Every record, newest last. A corrupt file is empty, never an exception.

    A history that can raise is a history that can stop Leti starting, and the
    thing it protects is a convenience.
    """
    path = store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning(f"Couldn't read {path} ({e}); continuing with no task history.")
        return []
    if not isinstance(data, list):
        logger.warning(f"{path} does not hold a list; continuing with no task history.")
        return []
    # A single malformed record must not cost the other hundred and ninety-nine.
    return [r for r in data if isinstance(r, dict) and r.get("task_id")]


def _save(records: List[Dict[str, Any]]) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(records[-MAX_RECORDS:], indent=2))


def _short(text: Any, limit: int = MAX_STEP_CHARS) -> Optional[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    return cleaned if len(cleaned) <= limit else cleaned[:limit - 1] + "…"


def summarise(task: Dict[str, Any]) -> Dict[str, Any]:
    """One task, compressed into a record. Reads the task; changes nothing."""
    steps = task.get("steps") or []
    recoveries = [
        {"step": s.get("n", i + 1), "attempts": s.get("recoveries", 0),
         "after": _short(s.get("error"), 120)}
        for i, s in enumerate(steps) if s.get("recoveries")
    ][:MAX_RECOVERIES_RECORDED]

    verification = task.get("verification") or {}
    current = task.get("current_step", 0)
    last_step = steps[min(current, len(steps) - 1)] if steps else {}

    return {
        "task_id": task.get("id"),
        "name": task.get("name") or None,
        "request": _short(task.get("objective"), 400),
        "goal": _short(task.get("success_criteria"), 300),
        "created_at": task.get("created_at"),
        "finished_at": task.get("completed_at") or task.get("updated_at"),
        "status": task.get("status"),
        "plan": [_short(s.get("instruction")) for s in steps[:MAX_STEPS_RECORDED]],
        "completed_steps": [s.get("n", i + 1) for i, s in enumerate(steps)
                            if s.get("status") == "done"],
        "failed_steps": [s.get("n", i + 1) for i, s in enumerate(steps)
                         if s.get("status") == "failed"],
        "skipped_steps": [s.get("n", i + 1) for i, s in enumerate(steps)
                          if s.get("status") == "skipped"],
        "last_step": {"n": last_step.get("n"),
                      "instruction": _short(last_step.get("instruction")),
                      "status": last_step.get("status")} if last_step else None,
        "recoveries": recoveries,
        "verification": ({"passed": verification.get("passed"),
                          "said": _short(verification.get("said"), 300)}
                         if verification else None),
        "result": _short(task.get("result"), MAX_RESULT_CHARS),
        "reason": _short(task.get("blocked_reason") or task.get("error"), 300),
        "waiting_for": _short(task.get("blocked_reason"), 200)
                       if task.get("status") == "waiting_for_user" else None,
        "project": task.get("project") or None,
        "entities": list(task.get("entities") or [])[:8],
    }


def record(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Write (or replace) one task's record. Never raises into its caller.

    A task must never fail because its epitaph could not be written, so every
    failure here is logged and swallowed. Replacing rather than appending keeps
    one record per task however many times a task changes state.
    """
    try:
        if not isinstance(task, dict) or not task.get("id"):
            return None
        entry = summarise(task)
        records = [r for r in load() if r.get("task_id") != entry["task_id"]]
        records.append(entry)
        _save(records)
        return entry
    except Exception as e:
        logger.warning(f"Couldn't record task history ({e}); the task is unaffected.")
        return None


def get(task_id: str) -> Optional[Dict[str, Any]]:
    return next((r for r in load() if r.get("task_id") == str(task_id)), None)


def recent(limit: int = 10) -> List[Dict[str, Any]]:
    """The most recently finished tasks, newest first."""
    records = sorted(load(), key=lambda r: r.get("finished_at") or 0, reverse=True)
    return records[:max(0, int(limit))]


def find(hint: str = "", since: Optional[float] = None,
         limit: int = 10) -> List[Dict[str, Any]]:
    """Records matching a phrase, newest first. Every match, never a guess.

    With no hint this is recent(): the caller asks which one rather than this
    picking. The words are matched against the name, the request and the plan,
    which is where a person's description of a task actually lives.
    """
    records = recent(limit=MAX_RECORDS)
    if since is not None:
        records = [r for r in records if (r.get("finished_at") or 0) >= since]
    words = [w for w in str(hint or "").lower().split() if len(w) > 2]
    if not words:
        return records[:limit]
    exact = [r for r in records if r.get("task_id") == str(hint).strip()]
    if exact:
        return exact
    scored = []
    for record_ in records:
        haystack = " ".join(filter(None, [
            record_.get("name") or "", record_.get("request") or "",
            " ".join(p or "" for p in (record_.get("plan") or []))])).lower()
        hits = sum(1 for w in words if w in haystack)
        if hits:
            scored.append((hits, record_))
    scored.sort(key=lambda pair: -pair[0])
    return [r for _, r in scored[:limit]]


def describe(record_: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A record turned into an answer, including what it cannot answer.

    `resumable` is the honest bit. A record knows what the plan was and which
    steps finished, but a record whose plan is empty cannot be continued from -
    and saying so is the difference between continuity and confabulation.
    """
    if not record_:
        return {"found": False,
                "problem": "There is no record of that task.",
                "resumable": False}

    plan = [p for p in (record_.get("plan") or []) if p]
    done = set(record_.get("completed_steps") or [])
    remaining = [p for i, p in enumerate(plan, start=1) if i not in done]
    missing = []
    if not plan:
        missing.append("the plan was not recorded")
    if record_.get("status") is None:
        missing.append("the final status was not recorded")

    return {
        "found": True,
        "task_id": record_.get("task_id"),
        "name": record_.get("name"),
        "request": record_.get("request"),
        "status": record_.get("status"),
        "finished": _stamp(record_.get("finished_at")),
        "steps_done": len(done),
        "steps_total": len(plan) or None,
        "remaining_steps": remaining,
        "failed_steps": record_.get("failed_steps") or [],
        "recoveries": record_.get("recoveries") or [],
        "verification": record_.get("verification"),
        "result": record_.get("result"),
        "reason": record_.get("reason"),
        "waiting_for": record_.get("waiting_for"),
        "project": record_.get("project"),
        # Enough to carry on with: a plan, and something still left in it.
        "resumable": bool(remaining) and not missing,
        "cannot_say": missing or None,
        "how_to_report": (
            "Say what the record actually holds. If resumable is false, do not offer "
            "to continue it - say what is known and ask what they want done instead."),
    }


def _stamp(value: Any) -> Optional[str]:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(value))


def forget(task_id: str = "") -> int:
    """Drop one record, or all of them. Returns how many went."""
    records = load()
    keep = ([r for r in records if r.get("task_id") != str(task_id)] if task_id else [])
    _save(keep)
    return len(records) - len(keep)
