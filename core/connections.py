"""Which outside services Leti can currently reach, asked cheaply.

Leti already has a Connections Manager: core/settings_editor.py's sections with
kind "connection" are it, the credentials live in config/settings.local.yaml at
0600, and /settings is how they get there. This module does not hold any of that
and never will. It is a VIEW that answers one question the rest of the code kept
asking badly:

    before doing something that needs the internet or an account, is that
    account even set up - and did it work the last time anybody tried?

Asked badly, that question becomes a network round trip on every turn. So:

  * configured() reads settings. No network, no credentials returned, microseconds.
  * last_outcome() reads a small in-memory memo of what happened the last time a
    real call was made - recorded by the code that was making the call anyway,
    the same way core/diagnostics.py records timings. Nothing is probed for it.
  * Nothing here ever calls out. A function that "checks" a connection by
    contacting it would put a round trip in front of every turn that mentions
    email, which is exactly the cost this exists to avoid.

What a caller gets is a status, not a secret. No value from a connection's
configuration is ever returned, logged or put in a prompt by anything in this
file - only whether a field is set.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.connections")

# The states a connection can be in, weakest first.
NOT_CONFIGURED = "not configured"     # nothing set up; an attempt will fail
CONFIGURED = "configured"             # set up, never exercised this session
WORKING = "working"                   # a real call succeeded recently
FAILING = "failing"                   # the last real call failed

# How long a remembered outcome stays interesting. Longer than a turn, shorter
# than a session: "it worked an hour ago" is not evidence about now.
OUTCOME_TTL_SECONDS = 900

# Capability -> the settings section that configures it, and what it is for.
# Section names are settings_editor's, so adding a connection there is the only
# place it has to be added.
CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "email": {"section": "email", "does": "reading and sending mail",
              "tools": ("send_email", "list_new_emails", "send_meeting_invite_email")},
    "calendar": {"section": "calendar", "does": "reading and creating calendar events",
                 "tools": ("schedule_meeting", "check_calendar_availability",
                           "business_calendar")},
    "zoom": {"section": "zoom", "does": "creating Zoom meeting links",
             "tools": ("create_video_meeting_link",)},
    "teams": {"section": "teams", "does": "creating Teams meeting links",
              "tools": ("create_video_meeting_link",)},
    "github": {"section": "github", "does": "reading repositories and opening pull requests",
               "tools": ("github",)},
    "trading": {"section": "trading", "does": "market data and paper trading",
                "tools": ("get_market_data", "place_paper_order", "check_watches")},
    "youtube": {"section": "youtube", "does": "reading YouTube channels",
                "tools": ("get_social_content", "check_social_watches")},
    "reddit": {"section": "reddit", "does": "reading Reddit",
               "tools": ("get_social_content",)},
}

# tool name -> capability, built once from the table above.
_TOOL_NEEDS: Dict[str, str] = {
    tool: name for name, spec in CAPABILITIES.items() for tool in spec["tools"]
}

# The last real outcome per capability: {capability: (at, ok, detail)}. Bounded by
# construction - one entry per capability, in memory, never written to disk.
_outcomes: Dict[str, Dict[str, Any]] = {}


def record_outcome(capability: str, ok: bool, detail: str = "") -> None:
    """What happened the last time something actually used this connection.

    Called by the code that made the call, on its way past. Never raises into it:
    a failed memo must not turn a successful call into a failed one.
    """
    try:
        name = str(capability or "").strip().lower()
        if name not in CAPABILITIES:
            return
        # A detail can carry a server's error text, which can carry anything. It is
        # truncated and never re-sent anywhere that a credential could be echoed
        # from: callers pass reasons, not payloads.
        _outcomes[name] = {"at": time.time(), "ok": bool(ok),
                           "detail": str(detail or "")[:200]}
    except Exception:
        logger.debug("Couldn't record a connection outcome; ignoring.")


def record_tool_outcome(tool_name: str, ok: bool, detail: str = "") -> None:
    """The same, addressed by tool name - for callers that know the tool, not
    the capability. A tool that needs no connection is a no-op."""
    capability = _TOOL_NEEDS.get(str(tool_name or ""))
    if capability:
        record_outcome(capability, ok, detail)


def forget_outcomes() -> None:
    """Only the tests and a reconfiguration call this."""
    _outcomes.clear()


def _is_configured(section: str) -> Optional[bool]:
    """True/False, or None when settings could not be read at all."""
    try:
        from core import settings_editor

        return settings_editor.section_is_configured(section)
    except KeyError:
        return False
    except Exception as e:
        logger.debug(f"Couldn't read the connections settings: {e}")
        return None


def status(capability: str, now: Optional[float] = None) -> Dict[str, Any]:
    """One capability's state. Reads settings and a memo; contacts nothing."""
    name = str(capability or "").strip().lower()
    spec = CAPABILITIES.get(name)
    if spec is None:
        return {"capability": name, "state": NOT_CONFIGURED,
                "problem": f"'{capability}' is not a connection Leti knows about.",
                "known": sorted(CAPABILITIES)}

    configured = _is_configured(spec["section"])
    if configured is None:
        return {"capability": name, "does": spec["does"], "state": NOT_CONFIGURED,
                "detail": "Leti could not read its own settings, so it cannot say "
                          "whether this is set up.",
                "fix": f"Open /settings and check the '{spec['section']}' section."}
    if not configured:
        return {"capability": name, "does": spec["does"], "state": NOT_CONFIGURED,
                "detail": f"No {name} account is set up, so {spec['does']} will fail.",
                "fix": f"Add it under Connections (the '{spec['section']}' section)."}

    now = now if now is not None else time.time()
    memo = _outcomes.get(name)
    if memo and (now - memo["at"]) <= OUTCOME_TTL_SECONDS:
        ago = int(now - memo["at"])
        if memo["ok"]:
            return {"capability": name, "does": spec["does"], "state": WORKING,
                    "detail": f"A real {name} call succeeded {ago}s ago."}
        return {"capability": name, "does": spec["does"], "state": FAILING,
                "detail": f"The last {name} call failed {ago}s ago"
                          + (f": {memo['detail']}" if memo["detail"] else "."),
                "fix": f"Check the '{spec['section']}' section under Connections."}
    return {"capability": name, "does": spec["does"], "state": CONFIGURED,
            "detail": f"{name.capitalize()} is set up. Nothing has used it recently, "
                      "so whether it works right now is untested."}


def available(capability: str) -> bool:
    """Is it worth trying? True for configured and for working; False for neither.

    FAILING is deliberately still True: one failure is not proof the account is
    gone, and refusing to try would turn a transient error into a permanent one.
    The caller that wants to warn first reads status() instead.
    """
    return status(capability).get("state") in (CONFIGURED, WORKING, FAILING)


def needed_by(tool_name: str) -> Optional[str]:
    """The capability a tool needs, or None if it needs no account."""
    return _TOOL_NEEDS.get(str(tool_name or ""))


def warning_for(tool_name: str) -> str:
    """One line to say before running a tool whose connection is not there.

    Empty for everything else, which is nearly every call: a tool that needs no
    connection, or one whose connection is set up, produces no text and costs the
    lookup above and nothing more.
    """
    capability = needed_by(tool_name)
    if not capability:
        return ""
    state = status(capability)
    if state["state"] == NOT_CONFIGURED:
        return (f"No {capability} connection is set up, so this will fail. "
                f"{state.get('fix', '')}").strip()
    if state["state"] == FAILING:
        return f"The last {capability} call failed. {state.get('detail', '')}".strip()
    return ""


def overview() -> List[Dict[str, Any]]:
    """Every capability and its state - for diagnostics and the interface."""
    return [status(name) for name in sorted(CAPABILITIES)]


def summary() -> Dict[str, Any]:
    """Counts, for a panel that does not want eight rows."""
    states = [s["state"] for s in overview()]
    return {
        "connected": sum(1 for s in states if s in (CONFIGURED, WORKING)),
        "failing": sum(1 for s in states if s == FAILING),
        "not_configured": sum(1 for s in states if s == NOT_CONFIGURED),
        "total": len(states),
    }
