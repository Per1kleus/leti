"""Watch something, and act when it changes - on the scheduler Leti already has.

A watch is a condition plus what to do when it becomes true. The hard parts are
not the checking; they are these three:

  It must not spam. A condition that stays true for ten minutes is one event, not
  one per evaluation. Each watch remembers whether the condition was already true
  and fires only on the transition into it, with a cooldown behind that.

  An error is not a condition. A monitor that cannot reach what it watches records
  a failure and says nothing - reading "the site is down" as "the site changed"
  would be worse than not watching at all.

  Detecting is not permission. A watch that sees an email arrive has learnt a fact,
  not earned the right to reply to it. Actions run through the task manager and
  therefore through the orchestrator and SafetyGuard, which stops an unattended
  external action exactly as it does anywhere else.

There is no second scheduler and no loop here. One row in tools/scheduler.py's
store evaluates every due watch, so an installation with no watches costs nothing
at all and one with ten costs one wake-up.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.atomic_write import atomic_write_text
from core.config_loader import resolve_path

logger = logging.getLogger("leti.watches")

STORE_PATH = "./data/watches.json"
SCHEDULER_TASK_NAME = "Leti watch evaluation"
DEFAULT_INTERVAL_MINUTES = 5
MIN_INTERVAL_MINUTES = 1
DEFAULT_COOLDOWN_MINUTES = 30
DISABLE_AFTER_FAILURES = 5
MAX_HISTORY = 20

# Conditions this can decide on its own, with no model and no tool call. Anything
# outside this list is not pretended at: see NEEDS_A_TASK below.
#
# email_from and price_below/price_above reuse what the email and trading tools
# already connect to - the same IMAP settings and the same market data helper - so
# they work exactly when those tools do and fail the same way when they don't.
#
# Calendar conflicts are NOT here. Leti can CREATE meetings over CalDAV but has no
# tool that reads a calendar back, so there is nothing to evaluate a conflict
# against. A watch that pretended to check would be worse than not offering one.
CONDITION_TYPES = ("cpu_above", "file_changed", "url_changed", "url_available",
                   "email_from", "price_below", "price_above")

# Asked for often enough to be worth refusing by name rather than with a generic
# "unsupported", so the answer explains itself.
UNSUPPORTED = {
    "calendar_conflict": ("Leti can create calendar events but has no tool that reads "
                          "a calendar back, so it cannot check for conflicts. Watching "
                          "for that would mean inventing an answer."),
}

ACTIONS = ("notify", "run_workflow", "start_task")


def store_path():
    return resolve_path(STORE_PATH)


def load_watches() -> List[Dict[str, Any]]:
    path = store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Couldn't read {path} ({e}); starting with no watches.")
        return []
    return data if isinstance(data, list) else []


def save_watches(watches: List[Dict[str, Any]]) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(watches, indent=2))


def get_watch(watch_id: str) -> Optional[Dict[str, Any]]:
    return next((w for w in load_watches() if w.get("id") == watch_id), None)


def _replace(watch: Dict[str, Any]) -> None:
    watches = load_watches()
    for i, existing in enumerate(watches):
        if existing.get("id") == watch["id"]:
            watches[i] = watch
            save_watches(watches)
            return
    watches.append(watch)
    save_watches(watches)


# --------------------------------------------------------------------------- #
# Creating
# --------------------------------------------------------------------------- #

def create(name: str, condition_type: str, condition: Dict[str, Any],
           action: str = "notify", action_target: str = "",
           interval_minutes: float = DEFAULT_INTERVAL_MINUTES,
           cooldown_minutes: float = DEFAULT_COOLDOWN_MINUTES,
           project: str = "", expires_at: Optional[float] = None) -> Dict[str, Any]:
    """A new watch, enabled. Validation is the caller's job - see validate()."""
    now = time.time()
    return {
        "id": uuid.uuid4().hex[:12],
        "name": (name or condition_type)[:80],
        "condition_type": condition_type,
        "condition": dict(condition or {}),
        "action": action,
        "action_target": action_target,
        "interval_minutes": max(MIN_INTERVAL_MINUTES, float(interval_minutes)),
        "cooldown_minutes": max(0.0, float(cooldown_minutes)),
        "project": project or None,
        "enabled": True,
        "expires_at": expires_at,
        "created_at": now,
        "last_checked_at": None,
        "last_triggered_at": None,
        # The anti-spam state: what the condition was last time we looked.
        "condition_was_true": False,
        "failures": 0,
        "last_error": None,
        "disabled_reason": None,
        "history": [],
    }


def validate(watch: Dict[str, Any]) -> List[str]:
    """Everything wrong with this watch. Empty means it can be saved."""
    errors: List[str] = []
    kind = watch.get("condition_type")
    if kind in UNSUPPORTED:
        return [UNSUPPORTED[kind]]
    if kind not in CONDITION_TYPES:
        return [f"'{kind}' is not something Leti can watch on its own. "
                f"Supported: {', '.join(CONDITION_TYPES)}."]

    condition = watch.get("condition") or {}
    if kind == "cpu_above":
        try:
            percent = float(condition.get("percent", 0))
            if not 1 <= percent <= 100:
                errors.append("cpu_above needs a percent between 1 and 100.")
        except (TypeError, ValueError):
            errors.append("cpu_above needs a numeric percent.")
        if float(condition.get("for_minutes", 0) or 0) < 0:
            errors.append("for_minutes cannot be negative.")
    elif kind == "file_changed":
        if not str(condition.get("path", "")).strip():
            errors.append("file_changed needs a path.")
    elif kind == "email_from":
        if not str(condition.get("sender", "")).strip():
            errors.append("email_from needs a sender to look for.")
    elif kind in ("price_below", "price_above"):
        if not str(condition.get("symbol", "")).strip():
            errors.append(f"{kind} needs a symbol, e.g. AAPL.")
        try:
            float(condition.get("price"))
        except (TypeError, ValueError):
            errors.append(f"{kind} needs a numeric price.")
    elif kind in ("url_changed", "url_available"):
        url = str(condition.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            errors.append(f"{kind} needs an http:// or https:// url.")

    if watch.get("action") not in ACTIONS:
        errors.append(f"'{watch.get('action')}' is not an action. Use: {', '.join(ACTIONS)}.")
    if watch.get("action") in ("run_workflow", "start_task") and not watch.get("action_target"):
        errors.append(f"A '{watch['action']}' action needs a target.")
    if float(watch.get("interval_minutes", 0)) < MIN_INTERVAL_MINUTES:
        errors.append(f"Evaluate no more often than every {MIN_INTERVAL_MINUTES} minute(s).")
    return errors


def save_new(watch: Dict[str, Any]) -> Dict[str, Any]:
    watches = load_watches()
    watches.append(watch)
    save_watches(watches)
    ensure_scheduled()
    return watch


def describe(watch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": watch["id"],
        "name": watch.get("name"),
        "watching": describe_condition(watch),
        "action": describe_action(watch),
        "enabled": bool(watch.get("enabled")),
        "project": watch.get("project"),
        "every": f"{watch.get('interval_minutes')} min",
        "last_checked": _stamp(watch.get("last_checked_at")),
        "last_triggered": _stamp(watch.get("last_triggered_at")),
        "currently_true": bool(watch.get("condition_was_true")),
        "failures": watch.get("failures", 0),
        "disabled_reason": watch.get("disabled_reason"),
        "recent": watch.get("history", [])[-5:],
    }


def _stamp(value) -> Optional[str]:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(value)) if value else None


def describe_condition(watch: Dict[str, Any]) -> str:
    kind, c = watch.get("condition_type"), watch.get("condition") or {}
    if kind == "cpu_above":
        held = c.get("for_minutes")
        return (f"CPU above {c.get('percent')}%"
                + (f" for {held} minutes" if held else ""))
    if kind == "file_changed":
        return f"changes to {c.get('path')}"
    if kind == "url_changed":
        return f"changes at {c.get('url')}"
    if kind == "url_available":
        return f"{c.get('url')} becoming reachable"
    if kind == "email_from":
        return f"unread mail from {c.get('sender')}"
    if kind == "price_below":
        return f"{c.get('symbol')} falling below {c.get('price')}"
    if kind == "price_above":
        return f"{c.get('symbol')} rising above {c.get('price')}"
    return str(kind)


def describe_action(watch: Dict[str, Any]) -> str:
    action = watch.get("action")
    if action == "notify":
        return "tell you"
    if action == "run_workflow":
        return f"run the workflow '{watch.get('action_target')}'"
    if action == "start_task":
        return f"start a task: {watch.get('action_target')}"
    return str(action)


def set_enabled(watch_id: str, enabled: bool) -> Optional[Dict[str, Any]]:
    watch = get_watch(watch_id)
    if watch is None:
        return None
    watch["enabled"] = bool(enabled)
    if enabled:
        watch["failures"] = 0
        watch["disabled_reason"] = None
    _replace(watch)
    return watch


def delete(watch_id: str) -> bool:
    before = load_watches()
    after = [w for w in before if w.get("id") != watch_id]
    if len(after) == len(before):
        return False
    save_watches(after)
    return True


# --------------------------------------------------------------------------- #
# Evaluating - deterministic, no model, no tool call
# --------------------------------------------------------------------------- #

class ConditionError(Exception):
    """The monitor could not tell. Never the same thing as 'condition is false'."""


def evaluate_condition(watch: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """(is_true, state_to_remember). Raises ConditionError when it cannot tell."""
    kind = watch.get("condition_type")
    condition = watch.get("condition") or {}
    state = dict(watch.get("condition_state") or {})

    if kind == "cpu_above":
        return _cpu_above(condition, state)
    if kind == "file_changed":
        return _file_changed(condition, state)
    if kind in ("url_changed", "url_available"):
        return _url(kind, condition, state)
    if kind == "email_from":
        return _email_from(condition, state)
    if kind in ("price_below", "price_above"):
        return _price(kind, condition, state)
    raise ConditionError(f"'{kind}' has no evaluator.")


def _email_from(condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Unread mail from a particular sender.

    Reuses the mail tool's own settings and IMAP session rather than a second
    configuration, and reads in peek mode exactly as list_new_emails does - a watch
    must not mark the user's mail as read behind them.
    """
    from tools.email_client import _ImapSession, _decode, _email_settings

    sender = str(condition.get("sender", "")).lower()
    try:
        cfg = _email_settings()
        with _ImapSession(cfg) as conn:
            status, data = conn.search(None, "UNSEEN")
            if status != "OK":
                raise ConditionError("the mail server refused the search")
            uids = data[0].split()[-50:]
            seen_uids = set(state.get("matched_uids") or [])
            matched = []
            for uid in reversed(uids):
                status, headers = conn.fetch(uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)])")
                if status != "OK" or not headers or not headers[0]:
                    continue
                text = _decode(headers[0][1].decode("utf-8", errors="replace"))
                if sender in text.lower():
                    matched.append(uid.decode() if isinstance(uid, bytes) else str(uid))
    except ConditionError:
        raise
    except Exception as e:
        raise ConditionError(f"couldn't check the mailbox: {e}")

    fresh = [uid for uid in matched if uid not in seen_uids]
    # True only while something NEW is sitting there, so one message is one event.
    return bool(fresh), {"matched_uids": matched[:50]}


def _price(kind: str, condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """A symbol's last close against a threshold, via the trading tool's own data
    helper - so it needs the same Alpaca key and fails the same way without it."""
    import asyncio as _asyncio

    from tools.trading_platform import _get_closes

    symbol = str(condition.get("symbol", "")).upper()
    try:
        threshold = float(condition.get("price"))
    except (TypeError, ValueError):
        raise ConditionError("the watch has no numeric price to compare against")

    try:
        closes = _asyncio.run(_get_closes(symbol, limit=1))
    except RuntimeError:
        raise ConditionError("price checks cannot run inside a running event loop here")
    except Exception as e:
        raise ConditionError(f"couldn't get a price for {symbol}: {e}")

    if not closes:
        raise ConditionError(f"no price data came back for {symbol}")
    last = float(closes[-1])
    hit = last < threshold if kind == "price_below" else last > threshold
    return hit, {"last_price": last}


def _cpu_above(condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    try:
        import psutil

        # interval=None returns the load since the last call, which is what a
        # periodic check wants - and costs nothing, unlike a blocking sample.
        percent = psutil.cpu_percent(interval=None)
    except Exception as e:
        raise ConditionError(f"couldn't read CPU usage: {e}")

    threshold = float(condition.get("percent", 90))
    hold_minutes = float(condition.get("for_minutes", 0) or 0)
    now = time.time()

    if percent < threshold:
        return False, {"above_since": None, "last_percent": percent}

    above_since = state.get("above_since") or now
    held_for = (now - above_since) / 60.0
    return held_for >= hold_minutes, {"above_since": above_since, "last_percent": percent}


def _file_changed(condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    from pathlib import Path

    path = Path(str(condition.get("path", ""))).expanduser()
    try:
        stat = path.stat()
    except OSError as e:
        raise ConditionError(f"couldn't read {path}: {e}")

    signature = f"{stat.st_mtime_ns}:{stat.st_size}"
    previous = state.get("signature")
    # The first look establishes a baseline; it is not a change.
    return (previous is not None and signature != previous), {"signature": signature}


def _url(kind: str, condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    import httpx

    url = str(condition.get("url", ""))
    try:
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            response = client.get(url)
    except Exception as e:
        if kind == "url_available":
            # Unreachable is a real answer for this condition, not a failure.
            return False, {"last_status": None}
        raise ConditionError(f"couldn't fetch {url}: {e}")

    if kind == "url_available":
        return response.status_code < 400, {"last_status": response.status_code}

    if response.status_code >= 400:
        raise ConditionError(f"{url} returned {response.status_code}")
    digest = hashlib.sha256(response.content).hexdigest()
    previous = state.get("digest")
    return (previous is not None and digest != previous), {"digest": digest}


def due(watch: Dict[str, Any], now: Optional[float] = None) -> bool:
    now = now if now is not None else time.time()
    if not watch.get("enabled"):
        return False
    if watch.get("expires_at") and now >= watch["expires_at"]:
        return False
    last = watch.get("last_checked_at")
    if last is None:
        return True
    return (now - last) >= float(watch.get("interval_minutes", 5)) * 60.0


def check(watch_id: str, now: Optional[float] = None) -> Dict[str, Any]:
    """Evaluate one watch and update its state. Returns what happened.

    The transition is the whole anti-spam story: a condition that was already true
    last time is not news, however long it stays true.
    """
    now = now if now is not None else time.time()
    watch = get_watch(watch_id)
    if watch is None:
        return {"watch_id": watch_id, "outcome": "missing"}

    was_true = bool(watch.get("condition_was_true"))
    try:
        is_true, state = evaluate_condition(watch)
    except Exception as e:
        watch["failures"] = watch.get("failures", 0) + 1
        watch["last_error"] = str(e)
        watch["last_checked_at"] = now
        if watch["failures"] >= DISABLE_AFTER_FAILURES:
            watch["enabled"] = False
            watch["disabled_reason"] = (
                f"Disabled after {watch['failures']} failed checks. Last error: {e}")
        _replace(watch)
        # An error never counts as the condition being met.
        return {"watch_id": watch_id, "outcome": "error", "error": str(e),
                "disabled": not watch["enabled"]}

    watch["condition_state"] = state
    watch["last_checked_at"] = now
    watch["failures"] = 0
    watch["last_error"] = None
    watch["condition_was_true"] = is_true

    if not is_true:
        _replace(watch)
        return {"watch_id": watch_id, "outcome": "false" if not was_true else "cleared"}

    if was_true:
        _replace(watch)
        return {"watch_id": watch_id, "outcome": "still_true"}

    cooldown = float(watch.get("cooldown_minutes", 0)) * 60.0
    last_fired = watch.get("last_triggered_at")
    if last_fired and (now - last_fired) < cooldown:
        _replace(watch)
        return {"watch_id": watch_id, "outcome": "cooling_down"}

    watch["last_triggered_at"] = now
    watch.setdefault("history", []).append(
        {"at": _stamp(now), "condition": describe_condition(watch)})
    watch["history"] = watch["history"][-MAX_HISTORY:]
    _replace(watch)
    # Mirrored to the interface's activity log. Nothing about the watch changes
    # here, and a failure to log can never stop a watch from firing.
    try:
        from core import diagnostics

        diagnostics.record_activity(
            "watch", f"Watch triggered - {describe_condition(watch)}",
            watch_id=watch_id,
        )
    except Exception:
        logger.debug("Couldn't mirror a watch trigger to the activity log.")
    return {"watch_id": watch_id, "outcome": "triggered", "watch": watch}


def ensure_scheduled(interval_minutes: Optional[float] = None) -> Optional[str]:
    """One row in the existing scheduler that evaluates every due watch.

    One row, not one per watch: the point is that a machine with no watches wakes
    up for nothing and a machine with ten wakes up once.
    """
    try:
        from tools import scheduler

        tasks = scheduler.load_tasks()
        existing = next((t for t in tasks if t.get("name") == SCHEDULER_TASK_NAME), None)
        watches = [w for w in load_watches() if w.get("enabled")]

        if not watches:
            if existing:
                scheduler.save_tasks([t for t in tasks if t is not existing])
            return None

        every = interval_minutes or min(
            [float(w.get("interval_minutes", DEFAULT_INTERVAL_MINUTES)) for w in watches])
        every = max(MIN_INTERVAL_MINUTES, every)

        if existing:
            existing["every_minutes"] = every
            existing["enabled"] = True
            existing["next_run"] = scheduler.compute_next_run(existing)
            scheduler.save_tasks(tasks)
            return existing["id"]

        task = {
            "id": uuid.uuid4().hex[:12],
            "name": SCHEDULER_TASK_NAME,
            "instruction": ("Check Leti's watches with the check_watches tool and act on "
                            "anything it reports as triggered."),
            "schedule_type": "interval",
            "every_minutes": every,
            "enabled": True,
            "created_at": time.time(),
            "history": [],
        }
        task["next_run"] = scheduler.compute_next_run(task)
        tasks.append(task)
        scheduler.save_tasks(tasks)
        return task["id"]
    except Exception:
        logger.exception("Couldn't register the watch evaluation task")
        return None
