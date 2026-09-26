"""What Leti is doing and what it costs, measured or honestly absent.

Every number here is either read from something that already exists - the
registry, the task store, the watch store, psutil - or from a timing another part
of the application recorded while doing its ordinary work. Nothing is measured
FOR this panel: no probe requests, no model calls, no screenshots, no background
loop. A metric that cannot be obtained that way reports "Unavailable" rather than
a plausible-looking invention, because a made-up number in a diagnostics panel is
worse than a missing one.

The timings come from record_*, which the orchestrator and router call on the way
past. They are kept in a small ring in memory and never written to disk.

The same record_* calls also feed the activity ring the interface's RT-LOG reads.
That ring owns no state either - it mirrors transitions the task manager, the
watch checker and the agent state machine make anyway, so the log says what Leti
is doing without anything having to be asked twice or polled for.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("leti.diagnostics")

UNAVAILABLE = "Unavailable"
_KEEP = 20                    # a short history; this is a panel, not a time series

_turns: Deque[Dict[str, Any]] = deque(maxlen=_KEEP)
_routings: Deque[Dict[str, Any]] = deque(maxlen=_KEEP)
_tools: Deque[Dict[str, Any]] = deque(maxlen=_KEEP)

# ---------------------------------------------------------------------------- #
# The activity feed behind the interface's RT-LOG.
#
# This is NOT a second status system. Nothing here decides, stores or owns any
# state: task status still lives in core/task_manager.py, watch state in
# core/watches.py, and the agent state machine in the orchestrator. This is a
# write-only mirror of transitions those parts make anyway, phrased for a person,
# kept in a bounded ring, and handed to whoever is listening.
#
# It is push-based on purpose. A log the interface has to poll for is a timer that
# runs whether or not anything happened; this one costs exactly one function call
# per thing that actually did.
# ---------------------------------------------------------------------------- #
_ACTIVITY_KEEP = 60           # bounded: the panel shows a tail, not a history
_activity: Deque[Dict[str, Any]] = deque(maxlen=_ACTIVITY_KEEP)
_activity_listeners: List[Any] = []


def record_activity(kind: str, message: str, **detail: Any) -> None:
    """Note one thing Leti just did, in words a person can read.

    Never raises into its caller: every call site is a piece of real work that
    must not fail because a log line could not be delivered.
    """
    entry: Dict[str, Any] = {"at": time.time(), "kind": kind, "message": message}
    if detail:
        entry.update(detail)
    _activity.append(entry)
    for listener in list(_activity_listeners):
        try:
            listener(entry)
        except Exception:
            logger.debug("An activity listener raised; dropping it from this entry.")


def on_activity(listener: Any) -> None:
    """Subscribe to activity as it happens (gui/api.py pushes it to the HUD)."""
    _activity_listeners.append(listener)


def recent_activity() -> List[Dict[str, Any]]:
    """The tail, for a client that connected after some of it happened."""
    return list(_activity)


def _humanise(tool_name: str) -> str:
    return tool_name.replace("_", " ").strip().capitalize() or tool_name


def record_routing(tool_count: int, total: int, seconds: float,
                   names: Optional[List[str]] = None) -> None:
    _routings.append({"at": time.time(), "exposed": tool_count, "total": total,
                      "seconds": seconds, "names": list(names or [])[:12]})
    record_activity("routing", f"Planning - {tool_count} of {total} tools in scope")


_contexts: Deque[Dict[str, Any]] = deque(maxlen=_KEEP)


def record_context(report: Dict[str, Any]) -> None:
    """What core/context_engine.py assembled for one turn, and what it left out.

    A mirror of a decision that was made anyway, like every other record_* here.
    Nothing is measured for it and nothing is re-read; the report is the object
    the engine already built.
    """
    _contexts.append({"at": time.time(), **(report or {})})
    included = len((report or {}).get("included") or [])
    record_activity("context",
                    f"Context - {included} source(s), {(report or {}).get('chars', 0)} chars")


def context_section() -> Dict[str, Any]:
    """The last turn's context decisions, for the diagnostics panel."""
    last = _last(_contexts)
    if not last:
        return {"status": UNAVAILABLE,
                "detail": "No turn has assembled context since Leti started."}
    return {
        "status": "ok",
        "sources_used": last.get("included", []),
        "sources_skipped": last.get("skipped", []),
        "dropped_for_budget": last.get("dropped_for_budget", []),
        "only_a_tool_can_reach": last.get("only_a_tool_can_reach", []),
        "chars": last.get("chars"),
        "budget_chars": last.get("budget_chars"),
        "seconds": last.get("seconds"),
    }


def record_turn(seconds: float, answer_chars: int = 0) -> None:
    _turns.append({"at": time.time(), "seconds": seconds, "chars": answer_chars})
    record_activity("turn", f"Answer ready ({seconds:.1f}s)")


def record_tool(name: str, seconds: float, success: bool) -> None:
    _tools.append({"at": time.time(), "name": name, "seconds": seconds, "success": success})
    record_activity(
        "tool",
        f"{_humanise(name)} - {'done' if success else 'failed'} ({seconds:.1f}s)",
        tool=name, success=bool(success),
    )


def reset() -> None:
    _turns.clear(); _routings.clear(); _tools.clear(); _activity.clear()
    _contexts.clear(); _intents.clear()


def _last(source: Deque[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return source[-1] if source else None


def model_section() -> Dict[str, Any]:
    """What the model is configured as, and whether Ollama answered recently.

    Availability is NOT probed here: asking Ollama on every panel refresh is the
    kind of cost this panel exists to avoid noticing. It reports what the last
    real request found.
    """
    try:
        from core.config_loader import get_settings

        ollama = get_settings().get("ollama", {})
        info: Dict[str, Any] = {
            "reasoning_model": ollama.get("reasoning_model", UNAVAILABLE),
            "vision_model": ollama.get("vision_model", UNAVAILABLE),
            "embedding_model": ollama.get("embedding_model", UNAVAILABLE),
            "context_size": ollama.get("num_ctx", UNAVAILABLE),
            "host": ollama.get("host", UNAVAILABLE),
        }
    except Exception as e:
        logger.warning(f"Couldn't read the model settings ({e}).")
        return {"reasoning_model": UNAVAILABLE, "context_size": UNAVAILABLE,
                "connected": UNAVAILABLE}

    last = _last(_turns)
    info["connected"] = (True if last else UNAVAILABLE)
    info["connected_note"] = ("A request completed at "
                              + time.strftime("%H:%M:%S", time.localtime(last["at"]))
                              if last else
                              "No request has been made yet this session.")
    return info


def performance_section() -> Dict[str, Any]:
    """Timings taken while doing the real work, or nothing.

    Time to first token and tokens per second are NOT here: Leti's requests are
    not streamed, so there is no first-token moment to time and no token count to
    divide by. Reporting a figure derived from characters would be a guess wearing
    a unit, so it says so instead.
    """
    turn, routing, tool = _last(_turns), _last(_routings), _last(_tools)
    return {
        "time_to_first_token": UNAVAILABLE,
        "time_to_first_token_note": ("Responses are not streamed, so there is no "
                                     "first-token moment to measure."),
        "tokens_per_second": UNAVAILABLE,
        "tokens_per_second_note": ("The model server does not report token counts "
                                   "to Leti, and a figure from character counts "
                                   "would be a guess."),
        "last_response_seconds": round(turn["seconds"], 2) if turn else UNAVAILABLE,
        "mean_response_seconds": (round(sum(t["seconds"] for t in _turns) / len(_turns), 2)
                                  if _turns else UNAVAILABLE),
        "responses_measured": len(_turns),
        "last_routing_ms": round(routing["seconds"] * 1000, 3) if routing else UNAVAILABLE,
        "last_tool": (f"{tool['name']} ({tool['seconds']:.2f}s)" if tool else UNAVAILABLE),
        "tools_measured": len(_tools),
    }


def resources_section() -> Dict[str, Any]:
    """CPU, memory, and the GPU only if something can actually be asked.

    No nvidia-smi call from here: it costs the better part of a second and this
    panel may refresh. VRAM is reported as unavailable unless a GPU reading was
    already taken elsewhere.
    """
    out: Dict[str, Any] = {"cpu_percent": UNAVAILABLE, "ram_percent": UNAVAILABLE,
                           "gpu": UNAVAILABLE, "vram": UNAVAILABLE,
                           "kv_cache": UNAVAILABLE}
    try:
        import psutil

        process = psutil.Process()
        # interval=None is the load since the last call: no blocking sample.
        out["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
        memory = psutil.virtual_memory()
        out["ram_percent"] = round(memory.percent, 1)
        out["ram_used_gb"] = round((memory.total - memory.available) / (1024 ** 3), 1)
        out["ram_total_gb"] = round(memory.total / (1024 ** 3), 1)
        out["leti_ram_mb"] = round(process.memory_info().rss / (1024 ** 2), 1)
        out["threads"] = process.num_threads()
    except Exception as e:
        logger.warning(f"Couldn't read resource usage ({e}).")

    out["gpu_note"] = ("Reading the GPU costs a subprocess call, so it is not done "
                       "to fill this panel. The first-launch hardware check reports it.")
    out["kv_cache_note"] = ("Ollama does not report its cache use to Leti. The context "
                            "size above is what it is configured to reserve.")
    return out


def tools_section(registry: Any = None) -> Dict[str, Any]:
    routing = _last(_routings)
    return {
        "registered": len(registry.names()) if registry is not None else UNAVAILABLE,
        "exposed_last_request": routing["exposed"] if routing else UNAVAILABLE,
        "routing_ms": round(routing["seconds"] * 1000, 3) if routing else UNAVAILABLE,
        "last_selected": routing["names"] if routing else [],
        "recent_tool_calls": [
            {"name": t["name"], "seconds": round(t["seconds"], 2), "success": t["success"]}
            for t in list(_tools)[-5:]
        ],
    }


def autonomy_section() -> Dict[str, Any]:
    try:
        from core import task_manager

        tasks = task_manager.load_tasks()
    except Exception:
        return {"active": UNAVAILABLE, "queued": UNAVAILABLE}

    active = [t for t in tasks if t.get("status") == task_manager.RUNNING]
    queued = [t for t in tasks if t.get("status") == task_manager.QUEUED]
    waiting = [t for t in tasks if t.get("status") == task_manager.WAITING_FOR_USER]
    current = active[0] if active else None
    return {
        "running": task_manager.describe(current) if current else None,
        "queued": len(queued),
        "waiting_for_approval": len(waiting),
        "recovery_attempts": sum(
            sum(s.get("recoveries", 0) for s in t.get("steps", [])) for t in tasks),
        "recent": [task_manager.describe(t) for t in tasks[-5:]],
    }


def watches_section() -> Dict[str, Any]:
    try:
        from core import watches

        all_watches = watches.load_watches()
    except Exception:
        return {"active": UNAVAILABLE}

    described = [watches.describe(w) for w in all_watches]
    next_due = None
    try:
        from tools import scheduler

        row = next((t for t in scheduler.load_tasks()
                    if t.get("name") == watches.SCHEDULER_TASK_NAME), None)
        if row and row.get("next_run"):
            next_due = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["next_run"]))
    except Exception:
        pass

    return {
        "active": sum(1 for w in described if w["enabled"]),
        "paused": sum(1 for w in described if not w["enabled"]),
        "next_evaluation": next_due or UNAVAILABLE,
        "failing": [w for w in described if w["failures"]],
        "watches": described,
    }


_intents: Deque[Dict[str, Any]] = deque(maxlen=_KEEP)


def record_intent(reading: Dict[str, Any]) -> None:
    """How the last request was read. A mirror of a decision already made."""
    _intents.append({"at": time.time(), **(reading or {})})


def intent_section() -> Dict[str, Any]:
    """What the Intent Layer made of the last few requests."""
    last = _last(_intents)
    if not last:
        return {"status": UNAVAILABLE,
                "detail": "No request has been read since Leti started."}
    return {
        "status": "ok",
        "kind": last.get("kind"),
        "confidence": last.get("confidence"),
        "shape": last.get("shape"),
        "side_effect_requested": last.get("side_effect_requested"),
        "tool_possibly_required": last.get("tool_possibly_required"),
        "clarification_required": last.get("clarification_required"),
        "belongs_to_existing_task": last.get("belongs_to_existing_task"),
        "recent": [{"kind": r.get("kind"), "confidence": r.get("confidence")}
                   for r in list(_intents)[-5:]],
    }


def tasks_section() -> Dict[str, Any]:
    """Every task, what it is waiting for, and anything that cannot run yet.

    Read-only, from core/task_manager.py's own store and the conflict view - no
    second task state and nothing computed here that the task does not already
    say about itself.
    """
    try:
        from core import task_conflicts, task_manager
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": f"tasks could not be read: {e}"}
    try:
        tasks = task_manager.load_tasks()
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": f"the task store could not be read: {e}"}

    active = [t for t in tasks if t.get("status") in task_manager.ACTIVE_STATUSES]
    described = [task_manager.detail(t) for t in active]
    return {
        "status": "ok",
        "active": len(active),
        "needs_you": sum(1 for t in described if t.get("awaiting_approval")),
        "waiting_external": sum(
            1 for t in active
            if t.get("status") == task_manager.WAITING_FOR_EXTERNAL),
        "retrying": sum(1 for t in active
                        if t.get("status") == task_manager.RETRYING),
        "blocked_by_conflict": [
            {"task": t.get("id"), "waiting_for": t.get("waiting_for_tasks"),
             "why": t.get("blocked_reason")}
            for t in active if t.get("waiting_for_tasks")],
        "conflicts": task_conflicts.all_conflicts(active),
        "tasks": [{
            "id": t["id"], "name": t.get("name"), "status": t.get("status"),
            "step": t.get("step"), "percent": t.get("percent"),
            "current": t.get("current"), "next": t.get("next_step"),
            "recoveries": t.get("recoveries"), "recovering": t.get("recovering"),
            "verified": t.get("verified"), "holds": t.get("holds"),
            "waiting_for_you": t.get("waiting_for_you"),
        } for t in described],
    }


def watches_health() -> Dict[str, Any]:
    """Which watches are stale, failing, or could act. Read-only."""
    try:
        from core import watches
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": str(e)}
    try:
        described = [watches.describe(w) for w in watches.load_watches()]
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": str(e)}
    return {
        "status": "ok",
        "total": len(described),
        "stale": [w["name"] for w in described if w.get("stale")],
        "failing": [w["name"] for w in described if w.get("failures")],
        "can_act": [w["name"] for w in described if w.get("acts_outside_leti")],
        "action_problems": [{"watch": w["name"], "problem": w["action_problem"]}
                            for w in described if w.get("action_problem")],
    }


def resource_locks_section() -> Dict[str, Any]:
    """What is held, by whom, and who is waiting. Read-only; holds nothing.

    Named for locks rather than resources because resources_section() above is
    already CPU and memory - two different meanings of the word, and one of them
    had the name first.
    """
    try:
        from core import resources
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": str(e)}
    try:
        found = resources.snapshot()
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": str(e)}
    return {
        "status": "ok",
        "held": found["held"],
        "waiting": found["waiting"],
        "long_held": found["long_held"],
        "note": ("Locks exist so two pieces of work do not ruin each other's. "
                 "They are not permissions and grant nothing."),
    }


def recovery_section() -> Dict[str, Any]:
    """Tasks the previous session left behind, and why each can or cannot resume."""
    try:
        from core import checkpoints, task_manager
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": str(e)}
    try:
        tasks = task_manager.load_tasks()
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": f"the task store could not be read: {e}"}

    interrupted = [t for t in tasks if t.get("recovery")]
    return {
        "status": "ok",
        "generation": checkpoints.generation(),
        "interrupted": len(interrupted),
        "ready_to_resume": sum(1 for t in interrupted
                               if t["recovery"].get("state") == checkpoints.READY_TO_RESUME),
        "needs_you": sum(1 for t in interrupted
                         if t["recovery"].get("state") == checkpoints.NEEDS_YOU),
        "unknown_external_effects": [
            {"task": t.get("id"), "name": t.get("name"),
             "why": t["recovery"].get("why")}
            for t in interrupted if t["recovery"].get("unknown_external_effect")],
        "tasks": [checkpoints.describe(t) for t in interrupted][:10],
    }


def computer_use_section() -> Dict[str, Any]:
    """Open GUI sessions and anything in them nobody checked. Read-only."""
    try:
        from core import computer_use
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": str(e)}
    try:
        sessions = computer_use.active_sessions()
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": str(e)}
    try:
        from core import ui_targets

        how = ui_targets.capabilities()
    except Exception:
        how = {"best": None, "method": None}
    return {
        "status": "ok",
        "open_sessions": len(sessions),
        # How targets are resolved on this machine, and what that is worth. No
        # tree, no screenshot, no element list - one line.
        "resolution": {"provider": how.get("best"), "method": how.get("method"),
                       "confidence": how.get("note")},
        "sessions": [{
            "id": s.id, "goal": s.goal, "steps": s.steps_taken,
            "mismatches": s.mismatches,
            "application": (s.context or {}).get("window"),
            "resolved_target": (s.resolved.name if getattr(s, "resolved", None)
                                else None),
            "observed_seconds_ago": (round(time.time() - s.observed_at, 1)
                                     if s.observed_at else None),
            "unverified": s.unverified_actions(),
        } for s in sessions],
    }


def output_section() -> Dict[str, Any]:
    """How the current answer is being delivered: spoken, held, shown.

    Says what is held and never what it says. The words of an answer are the
    conversation, and a diagnostics panel is not where the conversation belongs -
    so this reports a character count and a list of visual KINDS, and no text.
    """
    try:
        from core import math_render, transcript
    except Exception as e:
        return {"status": UNAVAILABLE, "detail": str(e)}
    held = transcript.section()
    return {**held,
            "transcript_default": "hidden",
            "math_elements_allowed": len(math_render.ELEMENTS),
            "note": ("Leti speaks the answer and holds the text. Asking to see it "
                     "shows the same words - nothing is generated again.")}


def snapshot(registry: Any = None) -> Dict[str, Any]:
    """Everything the panel shows, in one cheap call."""
    return {
        "taken_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_section(),
        "performance": performance_section(),
        "resources": resources_section(),
        "tools": tools_section(registry),
        "autonomy": autonomy_section(),
        "watches": watches_section(),
        # The upgrade layers, all of them views of state that already exists.
        "intent": intent_section(),
        "tasks": tasks_section(),
        "watch_health": watches_health(),
        "computer_use": computer_use_section(),
        "resource_locks": resource_locks_section(),
        "recovery": recovery_section(),
        "output": output_section(),
    }


# ---------------------------------------------------------------------------- #
# The full check
#
# snapshot() above is the panel: everything it reports was already measured while
# Leti did its ordinary work, so refreshing it costs nothing. This is the other
# thing - what "run full Leti diagnostics" means. It is ON DEMAND and nothing
# else: no timer, no background monitor, no periodic sweep. It runs because
# somebody asked, it finishes, and nothing keeps running afterwards.
#
# Three rules shape every check below.
#
# NOTHING DESTRUCTIVE, NOTHING WRITTEN. A check may read a store, count what is
# in it, and look at configuration. No check deletes, creates, sends, moves or
# edits anything, and none of them touches user data. Checking that the task
# store works does not mean creating a task in it.
#
# NOT TESTED IS A RESULT. A subsystem that could not be exercised reports
# NOT TESTED, and a subsystem with no account configured reports NOT CONFIGURED.
# Neither is PASS. A diagnostics report that says everything is fine because it
# did not look is worse than no diagnostics report.
#
# NO NETWORK UNLESS ASKED. The default check contacts nothing: an account's state
# comes from core/connections.py, which reads settings. `reach_out=True` is the
# caller explicitly accepting that a few checks will take seconds, and even then
# every call is a read.
# ---------------------------------------------------------------------------- #

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"
NOT_CONFIGURED = "NOT CONFIGURED"
NOT_AVAILABLE = "NOT AVAILABLE"
NOT_TESTED = "NOT TESTED"

CHECK_STATES = (PASS, WARNING, FAIL, NOT_CONFIGURED, NOT_AVAILABLE, NOT_TESTED)

# Worst first, for the overall verdict. NOT CONFIGURED and NOT TESTED deliberately
# outrank PASS: a report whose summary says "healthy" while three subsystems were
# never exercised is the failure this ordering exists to prevent.
_CHECK_SEVERITY = {FAIL: 5, WARNING: 4, NOT_AVAILABLE: 3, NOT_CONFIGURED: 2,
                   NOT_TESTED: 1, PASS: 0}


def _check(name: str, state: str, detail: str, **extra: Any) -> Dict[str, Any]:
    out = {"subsystem": name, "state": state, "detail": detail}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def _check_core() -> Dict[str, Any]:
    """Settings load, the data directory is writable, the stores parse."""
    try:
        from core.config_loader import get_settings, resolve_path
    except Exception as e:
        return _check("Core", FAIL, f"Leti's own configuration could not be imported: {e}")
    try:
        settings = get_settings()
    except Exception as e:
        return _check("Core", FAIL, f"config/settings.yaml could not be read: {e}")
    missing = [key for key in ("ollama", "memory", "safety") if key not in settings]
    if missing:
        return _check("Core", FAIL,
                      f"Settings are missing required section(s): {', '.join(missing)}.")
    try:
        data_dir = resolve_path("./data")
        writable = data_dir.exists() and os.access(data_dir, os.W_OK)
    except Exception as e:
        return _check("Core", WARNING, f"The data directory could not be checked: {e}")
    if not writable:
        return _check("Core", FAIL,
                      "The data directory is missing or not writable, so nothing Leti "
                      "records will survive.", path=str(data_dir))
    return _check("Core", PASS, "Settings load and the data directory is writable.",
                  path=str(data_dir))


def _check_model(reach_out: bool = False) -> Dict[str, Any]:
    try:
        from core.config_loader import get_settings

        ollama = get_settings().get("ollama", {})
    except Exception as e:
        return _check("Model", NOT_AVAILABLE, f"Model settings could not be read: {e}")
    model = ollama.get("reasoning_model")
    if not model:
        return _check("Model", NOT_CONFIGURED,
                      "No reasoning model is set. Leti cannot answer anything without one.")
    last = _last(_turns)
    if last:
        return _check("Model", PASS,
                      f"'{model}' answered a real request at "
                      + time.strftime("%H:%M:%S", time.localtime(last["at"])) + ".",
                      context_size=ollama.get("num_ctx"))
    return _check("Model", NOT_TESTED,
                  f"'{model}' is configured. Nothing has asked it anything this session, "
                  "so whether it answers is untested.",
                  context_size=ollama.get("num_ctx"))


def _check_ollama(reach_out: bool = False) -> Dict[str, Any]:
    """Is the model server there? Only asked when the caller accepts the cost."""
    try:
        from core.config_loader import get_settings

        host = get_settings().get("ollama", {}).get("host") or "http://localhost:11434"
    except Exception as e:
        return _check("Ollama", NOT_AVAILABLE, f"The host could not be read: {e}")
    if not reach_out:
        last = _last(_turns)
        if last:
            return _check("Ollama", PASS,
                          f"A request to {host} completed this session.", host=host)
        return _check("Ollama", NOT_TESTED,
                      f"Configured at {host}. Not contacted - run diagnostics with "
                      "connection checks to actually ask it.", host=host)
    try:
        import urllib.request

        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5) as response:
            ok = response.status == 200
        return (_check("Ollama", PASS, f"{host} answered.", host=host) if ok
                else _check("Ollama", FAIL, f"{host} answered with an error.", host=host))
    except Exception as e:
        return _check("Ollama", FAIL,
                      f"{host} could not be reached ({e}). Nothing that needs the model "
                      "will work until it is running.", host=host)


def _check_tools(registry: Any) -> Dict[str, Any]:
    if registry is None:
        return _check("Tool registry", NOT_TESTED,
                      "No registry was handed to the check, so the tools were not looked at.")
    try:
        names = list(registry.names())
    except Exception as e:
        return _check("Tool registry", FAIL, f"The registry could not be read: {e}")
    if not names:
        return _check("Tool registry", FAIL, "No tools are registered.")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        return _check("Tool registry", FAIL,
                      f"{len(duplicates)} tool name(s) are registered twice: "
                      + ", ".join(duplicates[:5]), registered=len(names))
    return _check("Tool registry", PASS, f"{len(names)} tools registered, no duplicates.",
                  registered=len(names))


def _check_context_window(registry: Any) -> Dict[str, Any]:
    """Does the tool list still fit num_ctx?

    main.py works this out at startup and logs a warning, which was the whole of
    the answer and is not enough: logging goes to a console, and the Windows
    launchers hide that console the moment Leti's own window appears. The one
    warning the code went out of its way to add - because "nothing anywhere said
    so" the first time this happened - was therefore invisible in the normal way
    of starting Leti. It is a check as well now, so it reaches the panel and
    "run full Leti diagnostics" rather than only a log nobody is looking at.

    The arithmetic is main.py's, kept in step by tests/test_docs_match_code.py.
    """
    try:
        from core.config_loader import get_settings

        num_ctx = int(get_settings().get("ollama", {}).get("num_ctx", 0))
    except Exception as e:
        return _check("Context window", NOT_AVAILABLE,
                      f"ollama.num_ctx could not be read: {e}")
    if not num_ctx:
        return _check("Context window", NOT_CONFIGURED,
                      "ollama.num_ctx is not set, so how much Leti can hold in mind is "
                      "whatever Ollama defaults to.")
    if registry is None:
        return _check("Context window", NOT_TESTED,
                      f"num_ctx is {num_ctx:,}; without the registry the tool list could "
                      "not be measured against it.", context_size=num_ctx)

    try:
        import json

        from core import modes

        widest, tool_tokens = "", 0
        for name in modes.MODES:
            tokens = len(json.dumps(registry.schemas_for(modes.visible_tools(registry, name)))) // 4
            if tokens > tool_tokens:
                widest, tool_tokens = name, tokens
    except Exception as e:
        return _check("Context window", NOT_TESTED,
                      f"The tool list could not be measured ({e}).", context_size=num_ctx)

    # The rest of a turn: system prompt, personality, profile, recalled memories,
    # the rolling buffer, and every tool result appended during the loop. The same
    # figure main.py reserves.
    headroom = 6000
    needed = tool_tokens + headroom
    if needed > num_ctx:
        return _check("Context window", FAIL,
                      f"{widest} mode's tools are about {tool_tokens:,} tokens and "
                      f"ollama.num_ctx is {num_ctx:,}. Ollama truncates instead of "
                      f"erroring, so tools are going missing and Leti will seem to have "
                      f"forgotten them. Raise num_ctx to at least {needed:,} in "
                      f"config/settings.yaml.",
                      context_size=num_ctx, tool_tokens=tool_tokens, needs=needed)
    spare = num_ctx - needed
    if spare < headroom // 2:
        return _check("Context window", WARNING,
                      f"{widest} mode's whole tool list is about {tool_tokens:,} tokens of "
                      f"{num_ctx:,}, leaving {spare:,} beyond what one turn can need. "
                      f"An ordinary turn is nowhere near this - routing sends a fraction of "
                      f"the list - but the turn routing cannot narrow sends all of it, and "
                      f"that turn is close to the edge. Adding tools without raising "
                      f"num_ctx is what tips it over; nothing needs doing today.",
                      context_size=num_ctx, tool_tokens=tool_tokens, needs=needed)
    return _check("Context window", PASS,
                  f"{widest} mode's tools are about {tool_tokens:,} tokens of {num_ctx:,}, "
                  f"with {spare:,} spare beyond a normal turn.",
                  context_size=num_ctx, tool_tokens=tool_tokens, needs=needed)


def _check_permissions(registry: Any) -> Dict[str, Any]:
    try:
        from core.config_loader import get_permissions

        permissions = get_permissions() or {}
    except Exception as e:
        return _check("Permissions", FAIL, f"config/permissions.yaml could not be read: {e}")
    classes = permissions.get("tools") or permissions.get("tool_permissions") or permissions
    if not isinstance(classes, dict) or not classes:
        return _check("Permissions", FAIL,
                      "No tool permissions are defined, so SafetyGuard has nothing to "
                      "classify calls by.")
    if registry is None:
        return _check("Permissions", NOT_TESTED,
                      f"{len(classes)} entries defined; without the registry they could "
                      "not be checked against the tools that exist.")
    try:
        names = set(registry.names())
    except Exception as e:
        return _check("Permissions", NOT_TESTED, f"The registry could not be read: {e}")
    unlisted = sorted(names - set(classes))
    if unlisted:
        return _check("Permissions", FAIL,
                      f"{len(unlisted)} registered tool(s) have no permission entry, so "
                      "their risk is undefined: " + ", ".join(unlisted[:5]),
                      unlisted=unlisted[:20])
    return _check("Permissions", PASS,
                  f"Every one of the {len(names)} registered tools has a permission entry.")


def _check_memory() -> Dict[str, Any]:
    try:
        from core.config_loader import resolve_path, get_settings

        path = resolve_path(get_settings()["memory"]["sqlite_path"])
    except Exception as e:
        return _check("Memory", NOT_AVAILABLE, f"The memory settings could not be read: {e}")
    if not path.exists():
        return _check("Memory", NOT_TESTED,
                      "The conversation log has not been created yet - it appears on the "
                      "first turn.", path=str(path))
    try:
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(path)) as conn:
            turns = conn.execute("SELECT COUNT(*) FROM conversation_log").fetchone()[0]
    except Exception as e:
        return _check("Memory", FAIL, f"The conversation log could not be read: {e}",
                      path=str(path))
    return _check("Memory", PASS, f"The conversation log holds {turns} turn(s).",
                  path=str(path))


def _check_project_memory() -> Dict[str, Any]:
    try:
        from tools.projects import get_active_project, list_projects

        projects = list_projects()
        active = get_active_project()
    except Exception as e:
        return _check("Project Memory", FAIL, f"Projects could not be read: {e}")
    if not projects:
        return _check("Project Memory", PASS,
                      "No projects exist yet, which is a valid state, not a problem.")
    return _check("Project Memory", PASS,
                  f"{len(projects)} project(s)"
                  + (f", '{active}' open." if active else ", none open."))


def _check_mode(name: str, label: str, registry: Any = None) -> Dict[str, Any]:
    """A specialised mode is checked by whether its own tools are reachable in it.

    Needs the live registry: building a second one here would mean constructing
    every tool again, which is real work and real imports for a check. Without it
    the honest answer is NOT TESTED.
    """
    try:
        from core import modes
    except Exception as e:
        return _check(label, NOT_AVAILABLE, f"The mode system could not be read: {e}")
    if registry is None:
        return _check(label, NOT_TESTED,
                      "No registry was handed to the check, so this mode's tools were "
                      "not looked at.")
    try:
        visible = modes.visible_tools(registry, name)
    except Exception as e:
        return _check(label, FAIL, f"The mode's tool list could not be built: {e}")
    expected = (modes.CODING_ONLY_TOOLS if name == modes.CODING
                else modes.BUSINESS_ONLY_TOOLS)
    missing = sorted(expected - visible)
    if missing:
        return _check(label, FAIL,
                      f"{len(missing)} of this mode's own tools are not visible in it: "
                      + ", ".join(missing))
    return _check(label, PASS,
                  f"{len(visible)} tools visible, including all {len(expected)} that only "
                  "this mode has.", tools_visible=len(visible))


def _check_connection(capability: str, label: str) -> Dict[str, Any]:
    """One account's state, from settings. Contacts nothing - see core/connections.py."""
    try:
        from core import connections

        state = connections.status(capability)
    except Exception as e:
        return _check(label, NOT_AVAILABLE, f"The connection could not be read: {e}")
    mapping = {
        connections.NOT_CONFIGURED: NOT_CONFIGURED,
        connections.CONFIGURED: NOT_TESTED,
        connections.WORKING: PASS,
        connections.FAILING: FAIL,
    }
    detail = state.get("detail", "")
    if mapping[state["state"]] == NOT_TESTED:
        detail += " Nothing here contacts it, so it is set up but untested."
    return _check(label, mapping[state["state"]], detail.strip(),
                  fix=state.get("fix"))


def _check_voice() -> Dict[str, Any]:
    """Can the voice stack load? Imports only - nothing is recorded or spoken."""
    missing = []
    for module, what in (("faster_whisper", "speech to text"),
                         ("pyttsx3", "speech out"),
                         ("openwakeword", "the wake word")):
        try:
            __import__(module)
        except Exception:
            missing.append(f"{what} ({module})")
    if len(missing) == 3:
        return _check("Voice", NOT_AVAILABLE,
                      "None of the voice packages are installed, so Leti is text-only. "
                      "That is a valid way to run it.")
    if missing:
        return _check("Voice", WARNING,
                      "Part of the voice stack is missing: " + "; ".join(missing))
    return _check("Voice", NOT_TESTED,
                  "Every voice package imports. Whether the microphone and speakers "
                  "work was not tested - that would mean recording and making a noise.")


def _check_computer_use() -> Dict[str, Any]:
    try:
        __import__("pyautogui")
    except Exception as e:
        return _check("Computer Use", NOT_AVAILABLE,
                      f"Screen and input control is not available here ({type(e).__name__}). "
                      "On a headless machine that is expected.")
    try:
        from core import computer_use

        open_sessions = len(computer_use.active_sessions())
    except Exception as e:
        return _check("Computer Use", FAIL, f"The session tracker failed: {e}")
    return _check("Computer Use", NOT_TESTED,
                  "The libraries load. Nothing was clicked or typed to test it, and "
                  f"{open_sessions} session(s) are open.")


def _check_scheduler() -> Dict[str, Any]:
    """The scheduled-task store, and whether the OS will wake Leti for it."""
    try:
        from tools import scheduler

        jobs = scheduler.load_tasks()
    except Exception as e:
        return _check("Scheduler", FAIL, f"The scheduler store could not be read: {e}")

    installed = None
    try:
        from core import system_scheduler

        installed = bool(system_scheduler.status().get("installed"))
    except Exception:
        installed = None                   # unknown, and reported as unknown below

    enabled = [j for j in jobs if j.get("enabled", True)]
    if not jobs:
        return _check("Scheduler", PASS, "No scheduled work, which is a valid state.",
                      wakes_leti_up=installed)
    overdue = [j for j in enabled
               if j.get("next_run") and j["next_run"] < time.time() - 3600]
    failing = [j for j in jobs if j.get("consecutive_failures", 0) >= 3]
    if failing:
        return _check("Scheduler", WARNING,
                      f"{len(failing)} scheduled task(s) have failed three or more times "
                      "in a row: " + ", ".join(str(j.get("name")) for j in failing[:5]),
                      wakes_leti_up=installed)
    if overdue:
        return _check("Scheduler", WARNING,
                      f"{len(overdue)} of {len(enabled)} enabled task(s) are more than an "
                      "hour overdue"
                      + (". Nothing is installed to wake Leti up for them."
                         if installed is False else
                         ". Leti only runs them while it is open."),
                      overdue=[j.get("name") for j in overdue][:5],
                      wakes_leti_up=installed)
    return _check("Scheduler", PASS,
                  f"{len(enabled)} enabled scheduled task(s), none overdue"
                  + ("; the OS is set to wake Leti for them." if installed
                     else "; nothing is installed to wake Leti up, so they run only "
                          "while it is open." if installed is False else "."),
                  wakes_leti_up=installed)


def _check_workflows() -> Dict[str, Any]:
    try:
        from core import workflows

        saved = workflows.load_workflows()
    except Exception as e:
        return _check("Workflows", FAIL, f"The workflow store could not be read: {e}")
    if not saved:
        return _check("Workflows", PASS, "No workflows saved, which is a valid state.")
    broken = []
    for workflow in saved:
        try:
            errors, _ = workflows.validate(workflow)
            if errors:
                broken.append(workflow.get("name") or workflow.get("id"))
        except Exception:
            broken.append(workflow.get("name") or workflow.get("id"))
    if broken:
        return _check("Workflows", WARNING,
                      f"{len(broken)} of {len(saved)} workflow(s) no longer validate: "
                      + ", ".join(str(b) for b in broken[:5]))
    return _check("Workflows", PASS, f"{len(saved)} workflow(s), all still valid.")


def _check_connections_overall() -> Dict[str, Any]:
    try:
        from core import connections

        counts = connections.summary()
    except Exception as e:
        return _check("Connections", NOT_AVAILABLE, f"Connections could not be read: {e}")
    if counts["failing"]:
        return _check("Connections", FAIL,
                      f"{counts['failing']} connection(s) failed the last time something "
                      "used them.", **counts)
    if counts["connected"] == 0:
        return _check("Connections", NOT_CONFIGURED,
                      f"None of the {counts['total']} connections are set up. Everything "
                      "that needs an account will say so rather than work.", **counts)
    return _check("Connections", PASS,
                  f"{counts['connected']} of {counts['total']} connections are set up "
                  f"({counts['not_configured']} are not).", **counts)


def full_check(registry: Any = None, reach_out: bool = False) -> Dict[str, Any]:
    """Every subsystem, checked on demand, reading only.

    `registry` is the live ToolRegistry when the caller has one - without it the
    tool and permission checks report NOT TESTED rather than guessing.
    `reach_out` lets the model-server check actually contact it; everything else
    stays local either way.

    Never raises: a check that blows up becomes a FAIL row naming itself, because
    a diagnostics run that dies is the least useful possible outcome.
    """
    started = time.perf_counter()
    runners = [
        ("Core", lambda: _check_core()),
        ("Model", lambda: _check_model(reach_out)),
        ("Ollama", lambda: _check_ollama(reach_out)),
        ("Tool registry", lambda: _check_tools(registry)),
        ("Permissions", lambda: _check_permissions(registry)),
        ("Context window", lambda: _check_context_window(registry)),
        ("Memory", lambda: _check_memory()),
        ("Project Memory", lambda: _check_project_memory()),
        ("Coding Mode", lambda: _check_mode("coding", "Coding Mode", registry)),
        ("Business Mode", lambda: _check_mode("business", "Business Mode", registry)),
        ("Calendar", lambda: _check_connection("calendar", "Calendar")),
        ("GitHub", lambda: _check_connection("github", "GitHub")),
        ("Voice", lambda: _check_voice()),
        ("Computer Use", lambda: _check_computer_use()),
        ("Scheduler", lambda: _check_scheduler()),
        ("Workflows", lambda: _check_workflows()),
        ("Connections", lambda: _check_connections_overall()),
    ]

    checks: List[Dict[str, Any]] = []
    for name, run in runners:
        try:
            checks.append(run())
        except Exception as e:
            logger.debug(f"Diagnostic '{name}' raised: {e}")
            checks.append(_check(name, FAIL, f"The check itself failed: {e}"))

    counts = {state: sum(1 for c in checks if c["state"] == state) for state in CHECK_STATES}
    worst = max((c["state"] for c in checks), key=lambda s: _CHECK_SEVERITY.get(s, 0),
                default=NOT_TESTED)
    untested = [c["subsystem"] for c in checks
                if c["state"] in (NOT_TESTED, NOT_CONFIGURED, NOT_AVAILABLE)]

    record_activity("diagnostics",
                    f"Full check - {counts[PASS]} pass, {counts[FAIL]} fail, "
                    f"{len(untested)} not tested")
    return {
        "taken_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.perf_counter() - started, 3),
        "contacted_anything": bool(reach_out),
        "overall": worst,
        "counts": counts,
        "checks": checks,
        "not_tested": untested,
        "how_to_read": (
            "PASS means the check ran and passed. NOT TESTED means it did not run - it "
            "is not a pass. NOT CONFIGURED means nothing is set up, which is often fine "
            "and is never the same as working. Nothing here changed, sent or deleted "
            "anything."),
    }


def full_check_text(report: Optional[Dict[str, Any]] = None, registry: Any = None) -> str:
    """The same report as lines a person can read or hear."""
    report = report if report is not None else full_check(registry)
    lines = [f"Leti diagnostics - {report['overall']} overall "
             f"({report['seconds']}s, nothing was changed)."]
    for check in report["checks"]:
        lines.append(f"  {check['state']:<16} {check['subsystem']}: {check['detail']}")
    if report["not_tested"]:
        lines.append("Not actually tested: " + ", ".join(report["not_tested"]) + ".")
    return "\n".join(lines)
