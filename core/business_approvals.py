"""Show the work before doing it, and do only what was agreed.

"Follow up with every lead nobody has contacted for thirty days" is eight emails.
The failure mode is obvious and expensive: eight messages leave before anybody
sees what they say, two of them to leads with no address, one of them to somebody
who replied yesterday. So a batch like that is prepared first, shown, and then
executed one item at a time - and only the items that were approved.

What this is NOT, and the distinction matters more than anything else here: it is
not a permission system. Approving a batch does not authorise anything. Every
action in it still goes through the tool that performs it and therefore through
SafetyGuard, which asks about sending an email exactly as it would if the user
had asked for that one email directly. Approval here means "yes, these are the
right eight" - authorisation still means what it always meant.

That is why nothing in this file executes. It holds a list, marks items approved
or rejected, and hands back the approved ones for the ordinary tools to perform.
There is no send, no write, no authorize, and a test asserts all three.

The queue lives in process and is released with the mode. A batch nobody acted on
is not a promise Leti should still be keeping tomorrow.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.business.approvals")

PROPOSED, APPROVED, REJECTED, DONE, FAILED, SKIPPED = (
    "proposed", "approved", "rejected", "done", "failed", "skipped")

MAX_BATCHES = 4
MAX_ITEMS = 50
MAX_TEXT = 300

# What a batch is allowed to propose. Named rather than free-form so a batch
# cannot smuggle in an action type nobody reviewed - and every one of these is a
# call to a tool that already exists, with the permission class it already has.
ACTION_KINDS = {
    "send_email": "send an email (external - SafetyGuard asks before each one)",
    "update_lead": "change a lead's stage or details (a write)",
    "schedule_meeting": "create a calendar event (external)",
    "create_task": "start a task through the task manager",
    "record_result": "record a measured result against a goal",
}

_batches: Dict[str, Dict[str, Any]] = {}


def release() -> None:
    """Leaving Business Mode. An unacted batch is not kept."""
    _batches.clear()


def _trim() -> None:
    """Keep the newest few. Written without a loop on purpose: nothing in the
    business modules runs repeatedly, and a flat assertion of that is worth more
    than one with an exception in it."""
    if len(_batches) <= MAX_BATCHES:
        return
    newest = sorted(_batches, key=lambda k: -_batches[k]["created_at"])[:MAX_BATCHES]
    for old_id in [k for k in _batches if k not in newest]:
        _batches.pop(old_id, None)


def propose(what: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare a batch and return the preview. Nothing happens to anything.

    Each item needs a `kind` from ACTION_KINDS and enough detail for a person to
    judge it. An item that cannot be performed - a lead with no email address -
    is kept in the batch and marked as such rather than quietly dropped, because
    "two of these cannot be sent" is exactly what the preview is for.
    """
    cleaned: List[Dict[str, Any]] = []
    for index, item in enumerate(list(items or [])[:MAX_ITEMS]):
        kind = str(item.get("kind", "")).strip()
        entry = {
            "n": index + 1,
            "kind": kind,
            "about": str(item.get("about", ""))[:MAX_TEXT],
            "detail": {k: v for k, v in item.items()
                       if k not in ("kind", "about", "blocked_because")},
            "status": PROPOSED,
        }
        if kind not in ACTION_KINDS:
            entry["status"] = SKIPPED
            entry["blocked_because"] = (
                f"'{kind}' is not an action this can propose. Known: "
                f"{', '.join(sorted(ACTION_KINDS))}.")
        elif item.get("blocked_because"):
            entry["status"] = SKIPPED
            entry["blocked_because"] = str(item["blocked_because"])[:MAX_TEXT]
        cleaned.append(entry)

    batch = {
        "id": "batch-" + uuid.uuid4().hex[:8],
        "what": str(what or "")[:MAX_TEXT],
        "items": cleaned,
        "created_at": time.time(),
        "decided": False,
    }
    _batches[batch["id"]] = batch
    _trim()
    return preview(batch["id"])


def get(batch_id: str) -> Optional[Dict[str, Any]]:
    return _batches.get(batch_id)


def preview(batch_id: str) -> Dict[str, Any]:
    """What would happen, in the words a person needs to decide - and nothing done."""
    batch = _batches.get(batch_id)
    if batch is None:
        return {"error": f"No batch '{batch_id}'. Propose one first."}

    actionable = [i for i in batch["items"] if i["status"] in (PROPOSED, APPROVED)]
    blocked = [i for i in batch["items"] if i["status"] == SKIPPED]
    kinds: Dict[str, int] = {}
    for item in actionable:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1

    return {
        "batch": batch["id"],
        "what": batch["what"],
        "found": len(batch["items"]),
        "can_be_done": len(actionable),
        "cannot_be_done": [{"n": i["n"], "about": i["about"],
                            "why": i.get("blocked_because")} for i in blocked],
        "by_kind": {k: {"count": v, "means": ACTION_KINDS.get(k, "unknown")}
                    for k, v in sorted(kinds.items())},
        "items": [{"n": i["n"], "kind": i["kind"], "about": i["about"],
                   "status": i["status"]} for i in batch["items"]],
        "nothing_has_happened": True,
        "next": ("Show this to the user and ask. Then approve the numbers they agreed to; "
                 "each one is still performed by its own tool and still asks for "
                 "permission where that tool always did."),
    }


def decide(batch_id: str, approve: Optional[List[int]] = None,
           reject: Optional[List[int]] = None, all_of_them: bool = False) -> Dict[str, Any]:
    """Mark items approved or rejected. Still does not do anything to anything."""
    batch = _batches.get(batch_id)
    if batch is None:
        return {"error": f"No batch '{batch_id}'."}

    approved_numbers = set(approve or [])
    rejected_numbers = set(reject or [])
    for item in batch["items"]:
        if item["status"] == SKIPPED:
            continue
        if item["n"] in rejected_numbers:
            item["status"] = REJECTED
        elif all_of_them or item["n"] in approved_numbers:
            item["status"] = APPROVED
    batch["decided"] = True

    return {
        "batch": batch["id"],
        "approved": [i["n"] for i in batch["items"] if i["status"] == APPROVED],
        "rejected": [i["n"] for i in batch["items"] if i["status"] == REJECTED],
        "still_undecided": [i["n"] for i in batch["items"] if i["status"] == PROPOSED],
        "nothing_has_happened": True,
        "next": ("Perform the approved items with the ordinary tools, one at a time. "
                 "Approval here is 'these are the right ones' - it is not permission, "
                 "and each action is authorised where it always was."),
    }


def approved_items(batch_id: str) -> List[Dict[str, Any]]:
    """The items to perform, in order. The caller performs them; this cannot."""
    batch = _batches.get(batch_id)
    if batch is None:
        return []
    return [dict(i) for i in batch["items"] if i["status"] == APPROVED]


def record_outcome(batch_id: str, n: int, ok: bool, detail: str = "") -> Dict[str, Any]:
    """What actually happened to one item, once a tool has actually done it.

    Reported by the caller after the fact rather than assumed: an item is done
    because a tool said so, never because it was approved.
    """
    batch = _batches.get(batch_id)
    if batch is None:
        return {"error": f"No batch '{batch_id}'."}
    for item in batch["items"]:
        if item["n"] == n:
            item["status"] = DONE if ok else FAILED
            item["outcome"] = str(detail or "")[:MAX_TEXT]
            return {"batch": batch_id, "n": n, "status": item["status"]}
    return {"error": f"No item {n} in {batch_id}."}


def report(batch_id: str) -> Dict[str, Any]:
    """What succeeded, what failed, what was never attempted."""
    batch = _batches.get(batch_id)
    if batch is None:
        return {"error": f"No batch '{batch_id}'."}

    def numbers(status):
        return [{"n": i["n"], "about": i["about"], "outcome": i.get("outcome")}
                for i in batch["items"] if i["status"] == status]

    done, failed = numbers(DONE), numbers(FAILED)
    return {
        "batch": batch_id,
        "what": batch["what"],
        "done": done,
        "failed": failed,
        "rejected": numbers(REJECTED),
        "could_not_be_done": numbers(SKIPPED),
        "never_attempted": numbers(APPROVED) + numbers(PROPOSED),
        "summary": (f"{len(done)} done, {len(failed)} failed, "
                    f"{len([i for i in batch['items'] if i['status'] == APPROVED])} "
                    "approved but not yet attempted"),
        "how_to_report": ("Say what actually happened per item. An approved item that was "
                          "never attempted has not been done, and a failure is a failure - "
                          "do not round either of them up."),
    }


def cancel(batch_id: str) -> Dict[str, Any]:
    batch = _batches.pop(batch_id, None)
    if batch is None:
        return {"error": f"No batch '{batch_id}'."}
    outstanding = [i["n"] for i in batch["items"] if i["status"] in (PROPOSED, APPROVED)]
    return {"cancelled": batch_id, "dropped": outstanding,
            "note": "Nothing outstanding was performed."}
