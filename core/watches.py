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
# market, news_event and trend are the advanced three; see core/signals.py for
# what each one measures. They are condition types rather than separate watch
# systems on purpose - one store, one scheduler row, one set of transition rules.
CONDITION_TYPES = ("cpu_above", "file_changed", "url_changed", "url_available",
                   "email_from", "price_below", "price_above",
                   "market", "news_event", "trend", "combined")

# What create_watch offers. price_below and price_above are still evaluated - old
# watches use them and are not rewritten - but they are a market condition with
# metric=price, so offering both would be offering the same capability twice under
# two names and asking the model to choose.
OFFERED_CONDITION_TYPES = ("market", "news_event", "trend", "cpu_above",
                           "file_changed", "url_changed", "url_available", "email_from")

# The advanced types default to looking less often than a CPU check: a news search
# every five minutes is rude to the search engine and tells you nothing a
# quarter-hourly one does not.
DEFAULT_INTERVALS = {"news_event": 30.0, "trend": 60.0, "market": 15.0, "combined": 30.0}
MAX_COMBINED_PARTS = 4

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

    elif kind == "market":
        errors.extend(_market_problems(condition))
    elif kind == "news_event":
        if not str(condition.get("topic", "")).strip():
            errors.append("A news watch needs a topic - what or who to follow.")
        if int(condition.get("min_sources", 1) or 1) < 1:
            errors.append("min_sources cannot be below 1.")
    elif kind == "trend":
        if not str(condition.get("topic", "")).strip() and not str(condition.get("symbol", "")).strip():
            errors.append("A trend watch needs a topic, a symbol, or both.")
    elif kind == "combined":
        parts = condition.get("parts") or []
        if len(parts) < 2:
            errors.append("A combined watch needs at least two signals to combine.")
        if len(parts) > MAX_COMBINED_PARTS:
            errors.append(f"A combined watch takes at most {MAX_COMBINED_PARTS} signals.")
        if str(condition.get("mode", "all")).lower() not in ("all", "any"):
            errors.append("A combined watch's mode is 'all' or 'any'.")
        for part in parts[:MAX_COMBINED_PARTS]:
            inner = part.get("condition_type")
            if inner in ("combined", None) or inner not in CONDITION_TYPES:
                errors.append(f"'{inner}' cannot be one of a combined watch's signals.")
                continue
            errors.extend(validate({"condition_type": inner,
                                    "condition": part.get("condition") or {},
                                    "action": "notify",
                                    "interval_minutes": MIN_INTERVAL_MINUTES}))

    if watch.get("action") not in ACTIONS:
        errors.append(f"'{watch.get('action')}' is not an action. Use: {', '.join(ACTIONS)}.")
    if watch.get("action") in ("run_workflow", "start_task") and not watch.get("action_target"):
        errors.append(f"A '{watch['action']}' action needs a target.")
    if float(watch.get("interval_minutes", 0)) < MIN_INTERVAL_MINUTES:
        errors.append(f"Evaluate no more often than every {MIN_INTERVAL_MINUTES} minute(s).")
    return errors


def _market_problems(condition: Dict[str, Any]) -> List[str]:
    from core import signals

    problems = []
    if not str(condition.get("symbol", "")).strip():
        problems.append("A market watch needs a symbol, e.g. TTWO or BTC/USD.")
    metric = str(condition.get("metric", "")).strip()
    if metric not in signals.MARKET_METRICS:
        problems.append(f"'{metric}' is not a market metric. Use one of: "
                        f"{', '.join(sorted(signals.MARKET_METRICS))}.")
    if str(condition.get("comparison", "above")).lower() not in signals.COMPARISONS:
        problems.append(f"'{condition.get('comparison')}' is not a comparison. Use: "
                        f"{', '.join(signals.COMPARISONS)}.")
    try:
        float(condition.get("threshold"))
    except (TypeError, ValueError):
        problems.append("A market watch needs a numeric threshold.")
    return problems


def save_new(watch: Dict[str, Any]) -> Dict[str, Any]:
    watches = load_watches()
    watches.append(watch)
    save_watches(watches)
    ensure_scheduled()
    return watch


# Where each kind of watch gets its numbers, so the interface and the model can
# both say where a claim came from rather than implying Leti knows it directly.
PROVIDERS = {
    "market": "Alpaca market data", "price_below": "Alpaca market data",
    "price_above": "Alpaca market data", "news_event": "web search",
    "trend": "web search + Alpaca market data", "email_from": "your mailbox (IMAP)",
    "cpu_above": "this machine", "file_changed": "this machine",
    "url_changed": "the page itself", "url_available": "the page itself",
    "combined": "several - see the signals",
}


def provider(watch: Dict[str, Any]) -> str:
    kind = watch.get("condition_type")
    if kind == "market" or kind in ("price_below", "price_above"):
        from tools.trading_platform import stock_feed

        symbol = str((watch.get("condition") or {}).get("symbol", ""))
        return ("Alpaca crypto data" if "/" in symbol
                else f"Alpaca market data ({stock_feed()} feed)")
    return PROVIDERS.get(kind, "unknown")


def next_evaluation(watch: Dict[str, Any]) -> Optional[float]:
    if not watch.get("enabled"):
        return None
    last = watch.get("last_checked_at")
    if last is None:
        return time.time()
    return last + float(watch.get("interval_minutes", DEFAULT_INTERVAL_MINUTES)) * 60.0


def describe(watch: Dict[str, Any], detail: bool = False) -> Dict[str, Any]:
    """What the interface and the model see. detail adds why it last fired."""
    described = {
        "id": watch["id"],
        "name": watch.get("name"),
        "watching": describe_condition(watch),
        "action": describe_action(watch),
        "enabled": bool(watch.get("enabled")),
        "project": watch.get("project"),
        "every": f"{watch.get('interval_minutes')} min",
        "last_checked": _stamp(watch.get("last_checked_at")),
        "last_triggered": _stamp(watch.get("last_triggered_at")),
        "next_evaluation": _stamp(next_evaluation(watch)),
        "expires": _stamp(watch.get("expires_at")),
        "source": provider(watch),
        "currently_true": bool(watch.get("condition_was_true")),
        "failures": watch.get("failures", 0),
        "last_error": watch.get("last_error"),
        "disabled_reason": watch.get("disabled_reason"),
        "recent": watch.get("history", [])[-5:],
    }
    if detail:
        described.update({
            "type": watch.get("condition_type"),
            "condition": watch.get("condition"),
            "cooldown": f"{watch.get('cooldown_minutes')} min",
            "why_it_exists": watch.get("reason") or None,
            # The signals as they were last measured - what the condition is being
            # decided on right now, not a summary of it.
            "current_signals": (watch.get("condition_state") or {}).get("evidence"),
            "history": watch.get("history", []),
        })
    return described


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
    if kind == "market":
        from core import signals

        return signals.describe_market(c)
    if kind == "news_event":
        from core import signals

        return signals.describe_news(c)
    if kind == "trend":
        from core import signals

        return signals.describe_trend(c)
    if kind == "combined":
        joiner = " and " if str(c.get("mode", "all")).lower() == "all" else " or "
        return joiner.join(
            describe_condition({"condition_type": p.get("condition_type"),
                                "condition": p.get("condition") or {}})
            for p in (c.get("parts") or [])[:MAX_COMBINED_PARTS]) or "nothing yet"
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
    if kind == "market":
        return _market(condition, state)
    if kind == "news_event":
        return _news(condition, state)
    if kind == "trend":
        return _trend(condition, state)
    if kind == "combined":
        return _combined(condition, state)
    raise ConditionError(f"'{kind}' has no evaluator.")


def _market(condition: Dict[str, Any], state: Dict[str, Any],
            snapshot: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
    """One metric against one threshold, through core/signals.py.

    Every market condition goes through here, price_below and price_above
    included: there is one piece of arithmetic and one place a provider failure is
    turned into "Leti could not tell" rather than "it did not happen".
    """
    from core import signals

    try:
        value, evidence = signals.market_value(
            condition.get("symbol", ""), condition.get("metric", "price"),
            int(condition.get("window", signals.DEFAULT_WINDOW_DAYS) or signals.DEFAULT_WINDOW_DAYS),
            snapshot=snapshot)
        hit = signals.compare(value, condition.get("comparison", "above"),
                              float(condition.get("threshold")))
    except signals.SignalError as e:
        raise ConditionError(str(e))
    except (TypeError, ValueError) as e:
        raise ConditionError(f"that market watch is not configured properly: {e}")

    evidence["threshold"] = condition.get("threshold")
    evidence["comparison"] = condition.get("comparison", "above")
    evidence["reading"] = describe_condition(
        {"condition_type": "market", "condition": condition}) + f" - now {value:,.3f}"
    return hit, {"last_value": round(value, 6), "evidence": evidence}


def _news(condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    from core import signals

    try:
        return signals.news_events(condition, state)
    except signals.SignalError as e:
        raise ConditionError(str(e))


def _trend(condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    from core import signals

    try:
        return signals.trend(condition, state)
    except signals.SignalError as e:
        raise ConditionError(str(e))


def _combined(condition: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Several signals under one watch, combined with all or any.

    Each part keeps its own state - a news part still remembers which events it has
    already reported even on an evaluation where the market part was false - and a
    part that cannot be measured fails the whole watch rather than being counted as
    false, because "the search engine timed out" is not "there was no news".
    """
    mode = str(condition.get("mode", "all")).lower()
    parts = list(condition.get("parts") or [])[:MAX_COMBINED_PARTS]
    part_states = list(state.get("parts") or [])
    results, evidence, new_states = [], [], []

    for index, part in enumerate(parts):
        inner_state = part_states[index] if index < len(part_states) else {}
        inner = {"condition_type": part.get("condition_type"),
                 "condition": part.get("condition") or {},
                 "condition_state": inner_state}
        try:
            is_true, updated = evaluate_condition(inner)
        except ConditionError as e:
            # 'any' can still be satisfied by another part; 'all' cannot be
            # decided at all, so it stops rather than guessing.
            if mode == "all":
                raise
            results.append(False)
            new_states.append(inner_state)
            evidence.append({"signal": part.get("condition_type"), "value": None,
                             "problem": str(e)})
            continue
        results.append(is_true)
        new_states.append(updated)
        evidence.append({"signal": part.get("condition_type"),
                         "met": is_true,
                         "what": describe_condition(inner),
                         "detail": (updated or {}).get("evidence")})

    if not results:
        raise ConditionError("none of that watch's signals could be measured")
    met = all(results) if mode == "all" else any(results)
    keys = [s.get("event_key") for s in new_states if isinstance(s, dict) and s.get("event_key")]
    return met, {"parts": new_states,
                 "event_key": "+".join(keys) or None,
                 "evidence": {"mode": mode, "signals": evidence, "met": met}}


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
    """The original price threshold watch, expressed as a market condition.

    Kept as its own condition type because watches saved before market conditions
    existed still say price_below, and because "tell me when it falls under 120" is
    worth a name. It is not a second implementation: it builds the market
    condition and hands it to the same evaluator.
    """
    try:
        threshold = float(condition.get("price"))
    except (TypeError, ValueError):
        raise ConditionError("the watch has no numeric price to compare against")
    return _market({"symbol": condition.get("symbol", ""), "metric": "price",
                    "comparison": "below" if kind == "price_below" else "above",
                    "threshold": threshold}, state)


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


MAX_EVIDENCE_CHARS = 4_000


def _trim(evidence: Any) -> Any:
    """Keep the evidence compact. History explains a trigger; it is not a data store."""
    if evidence is None:
        return None
    text = json.dumps(evidence, default=str)
    if len(text) <= MAX_EVIDENCE_CHARS:
        return evidence
    return {"summary": text[:MAX_EVIDENCE_CHARS],
            "note": "shortened - watch history records why, not the whole reading"}


def _summarise_evidence(state: Dict[str, Any]) -> Optional[str]:
    """One line a person can read, from whichever signal produced the trigger."""
    evidence = (state or {}).get("evidence") or {}
    if evidence.get("reading"):
        return evidence["reading"]
    if evidence.get("new_events"):
        first = evidence["new_events"][0]
        return (f"{first.get('headline')} - {first.get('support')}, "
                f"{first.get('why')}")
    if evidence.get("phase"):
        return (f"attention is {evidence['phase']} ({evidence.get('why')}), "
                f"confidence {evidence.get('confidence')}")
    if evidence.get("signals"):
        met = [s.get("what") for s in evidence["signals"] if s.get("met")]
        return " and ".join(m for m in met if m) or None
    return None


def expire_due_watches(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Turn off watches that have reached their expiry, saying why.

    Without this an expired watch sits enabled and never due, which reads in the
    interface as "monitoring" for something nobody is monitoring any more.
    """
    now = now if now is not None else time.time()
    expired = []
    watches = load_watches()
    changed = False
    for watch in watches:
        if (watch.get("enabled") and watch.get("expires_at")
                and now >= float(watch["expires_at"])):
            watch["enabled"] = False
            watch["disabled_reason"] = (
                f"Expired on {_stamp(watch['expires_at'])}, as asked.")
            expired.append(watch)
            changed = True
    if changed:
        save_watches(watches)
        ensure_scheduled()
    return expired


UPDATABLE = ("name", "interval_minutes", "cooldown_minutes", "expires_at",
             "action", "action_target", "condition", "project", "reason")


def update(watch_id: str, changes: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Change a watch in place, keeping its id, its history and what it has seen.

    Editing rather than deleting and recreating is the difference between "raise
    the threshold to 8%" and "forget every event you have already told me about".
    """
    watch = get_watch(watch_id)
    if watch is None:
        return None, [f"No watch with id '{watch_id}'."]

    updated = dict(watch)
    for key, value in (changes or {}).items():
        if key not in UPDATABLE or value is None:
            continue
        if key == "condition" and isinstance(value, dict):
            # Merged, not replaced: "check it every hour instead" should not wipe
            # the symbol out of the condition.
            merged = dict(updated.get("condition") or {})
            merged.update({k: v for k, v in value.items() if v is not None})
            updated["condition"] = merged
        elif key in ("interval_minutes", "cooldown_minutes"):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None, [f"{key} has to be a number."]
            updated[key] = max(MIN_INTERVAL_MINUTES if key == "interval_minutes" else 0.0,
                               number)
        else:
            updated[key] = value

    problems = validate(updated)
    if problems:
        return None, problems
    _replace(updated)
    ensure_scheduled()
    return updated, []


def prefetch(due_watches: List[Dict[str, Any]]) -> None:
    """Warm one market request for every symbol about to be evaluated.

    Ten watches on four symbols are one HTTP request between them rather than ten:
    the snapshots land in the market client's short-lived cache and each watch then
    reads its own metric out of what is already there. Failing here is not an
    error - each watch will fetch what it needs and report its own problem.
    """
    symbols = set()
    for watch in due_watches:
        for condition in _market_conditions(watch):
            symbol = str(condition.get("symbol", "")).strip()
            if symbol:
                symbols.add(symbol.upper())
    if not symbols:
        return
    try:
        from core.signals import run_blocking
        from tools.trading_platform import get_snapshots

        run_blocking(get_snapshots(sorted(symbols)))
    except Exception as e:
        logger.debug(f"Couldn't prefetch market data ({e}); watches will fetch their own.")


def _market_conditions(watch: Dict[str, Any]):
    kind, condition = watch.get("condition_type"), watch.get("condition") or {}
    if kind in ("market", "price_below", "price_above", "trend"):
        yield condition
    elif kind == "combined":
        for part in (condition.get("parts") or [])[:MAX_COMBINED_PARTS]:
            yield from _market_conditions(part)


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
        watch.setdefault("history", []).append(
            {"at": _stamp(now), "problem": str(e)[:300],
             "note": "recoverable - the watch stays on and tries again"
                     if watch["enabled"] else watch["disabled_reason"]})
        watch["history"] = watch["history"][-MAX_HISTORY:]
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

    # The transition rule, and the one thing it has to get right for both shapes of
    # condition. A LEVEL - a price above 120, CPU over 90% - is one event for as
    # long as it holds, so true-after-true is silence. A STREAM - news about a
    # topic - is a different thing each time: an arrest today and a charge tomorrow
    # are two events, and treating the second as "still true" would lose it. So an
    # evaluator that deals in discrete events names the one that made the condition
    # true, and a different name is a new event. Level conditions name nothing and
    # behave exactly as they always did.
    event_key = state.get("event_key")
    if was_true and (event_key is None or event_key == watch.get("last_event_key")):
        _replace(watch)
        return {"watch_id": watch_id, "outcome": "still_true"}

    cooldown = float(watch.get("cooldown_minutes", 0)) * 60.0
    last_fired = watch.get("last_triggered_at")
    if last_fired and (now - last_fired) < cooldown:
        _replace(watch)
        return {"watch_id": watch_id, "outcome": "cooling_down"}

    watch["last_triggered_at"] = now
    watch["last_event_key"] = event_key
    # The evidence, not the data: the numbers the condition turned on and the
    # sources behind them, which is what "why did this fire" needs and what a
    # stored copy of every bar it read would bury.
    watch.setdefault("history", []).append(
        {"at": _stamp(now), "condition": describe_condition(watch),
         "why": _summarise_evidence(state), "evidence": _trim(state.get("evidence"))})
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
