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
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("leti.diagnostics")

UNAVAILABLE = "Unavailable"
_KEEP = 20                    # a short history; this is a panel, not a time series

_turns: Deque[Dict[str, Any]] = deque(maxlen=_KEEP)
_routings: Deque[Dict[str, Any]] = deque(maxlen=_KEEP)
_tools: Deque[Dict[str, Any]] = deque(maxlen=_KEEP)


def record_routing(tool_count: int, total: int, seconds: float,
                   names: Optional[List[str]] = None) -> None:
    _routings.append({"at": time.time(), "exposed": tool_count, "total": total,
                      "seconds": seconds, "names": list(names or [])[:12]})


def record_turn(seconds: float, answer_chars: int = 0) -> None:
    _turns.append({"at": time.time(), "seconds": seconds, "chars": answer_chars})


def record_tool(name: str, seconds: float, success: bool) -> None:
    _tools.append({"at": time.time(), "name": name, "seconds": seconds, "success": success})


def reset() -> None:
    _turns.clear(); _routings.clear(); _tools.clear()


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
    }
