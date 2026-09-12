"""Driving the GUI, but only when nothing better will do.

Leti can already open an application, focus a window, click a point, type, press a
shortcut, read the screen and drive a browser. Those tools exist and are already
authorised. So this is not another way to control the computer - it is the thing
that decides whether the computer should be controlled that way at all, and that
keeps a sequence of clicks honest once it starts.

Three jobs, and deliberately no fourth:

  Choose the layer. A dedicated tool beats browser automation, and browser
  automation beats moving the mouse. Clicking through a web page Leti could have
  read directly is slower, less reliable and impossible to verify.

  Verify. A GUI step acts on what is on screen, so a sequence that does not look
  before it clicks is guessing. A session requires a check before it acts and
  after anything that matters, and stops when what it sees is not what it expected
  rather than clicking on into an unknown window.

  Bound it. A step budget, and no repeating an identical action - the failure mode
  of GUI automation is a loop that clicks the same wrong pixel forever.

It executes nothing. Every actual click and keystroke is the existing tool, called
by the orchestrator, authorised by SafetyGuard exactly as if the user had asked
for it directly - which is what stops the GUI becoming a way around the rules that
govern the equivalent tool. Sending mail by clicking Send is still sending mail.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("leti.computer_use")

MAX_STEPS = 20                 # a GUI errand, not an afternoon
MAX_IDENTICAL_ACTIONS = 2      # twice is a retry; three times is a loop
SESSION_IDLE_TIMEOUT = 600     # a forgotten session should not stay open
# How old a look at the screen may be before it is no longer a reason to click.
# Separate from, and much shorter than, the session timeout: a session can sit for
# ten minutes while the model thinks, but a window that was in front of you a
# minute and a half ago is not evidence about where the mouse should go now.
OBSERVATION_MAX_AGE = 90
MAX_PLAN_STEPS = 12            # a plan, not a program

# Actions whose result is somebody else's problem if it is wrong: they leave the
# screen and go somewhere. One of these must be verified before anything else
# happens, and a session that ends with one unverified says so.
_CONSEQUENTIAL = re.compile(
    r"\b(send|submit|confirm|delete|remove|buy|purchase|pay|order|publish|post|"
    r"install|uninstall|overwrite|replace|sign|accept|apply|save|upload|share|"
    r"transfer|discard)\b", re.I)


def is_consequential(action: str) -> bool:
    """Whether an action is one that cannot simply be looked at again afterwards."""
    return bool(_CONSEQUENTIAL.search(str(action or "")))

# Requests a dedicated tool already covers. Reaching for the mouse here would be
# slower and less reliable, and would lose the tool's own error reporting.
_BETTER_WITH_A_TOOL = (
    (r"\b(read|open|write|delete|move|list)\b.*\bfile\b", "the file tools"),
    (r"\bsearch (the )?web\b|\bgoogle\b|\blook up\b", "web_search"),
    (r"\bsend (an? )?email\b", "send_email"),
    (r"\b(create|make|add|book|schedule)\b.*\b(calendar )?(event|meeting|appointment)\b",
     "schedule_meeting"),
    # File FORMATS only. "open the invoice page and download it" is browser work,
    # and an earlier version of this pattern claimed it for the document tools
    # because it mentioned an invoice.
    (r"\b(read|summari[sz]e|compare|extract)\b.*\b(pdf|docx|word document|"
     r"spreadsheet|xlsx|csv)s?\b", "the document tools"),
    (r"\badd\b.*\b(to my )?(to-?do|task list)\b", "add_todo_item"),
    (r"\bweather\b", "get_weather"),
    (r"\brun\b.*\b(command|shell|script)\b", "run_shell_command"),
    (r"\bscreenshot\b|\bwhat('s| is) on (my |the )?screen\b", "read_screen"),
)

# Browser work the existing automation can do without touching the mouse.
_BETTER_IN_THE_BROWSER = (
    (r"\bgo to\b.*\b(https?://|www\.|\.com|\.org|\.net)\b", "browser navigation"),
    (r"\bfill (in |out )?(the )?form\b", "browser_fill_form"),
    (r"\bread\b.*\b(page|website|site|article)\b", "browser_read_page"),
    # Getting something off a website is browser work before it is mouse work:
    # navigating and reading a page directly is verifiable, and clicking through
    # one is not.
    (r"\b(download|save|fetch)\b.*\b(from|on|off)\b.*"
     r"\b(site|website|page|portal|url|https?://|www\.)\b", "browser navigation"),
    (r"\b(find|get|look for|locate)\b.*\bon (the |this |their )?"
     r"(site|website|page|portal)\b", "browser navigation"),
)

LAYER_TOOL = "tool"
LAYER_BROWSER = "browser"
LAYER_GUI = "gui"


def choose_layer(request: str) -> Dict[str, Any]:
    """Which layer should handle this: a tool, the browser, or the GUI.

    Deterministic and cheap - no model call to decide whether to use the model's
    other tools. It advises rather than forbids: the caller is the reasoning model,
    which can see things a pattern cannot, but it has to be told there is a better
    way before it decides to ignore it.
    """
    text = str(request or "").lower()

    for pattern, suggestion in _BETTER_WITH_A_TOOL:
        if re.search(pattern, text):
            return {"layer": LAYER_TOOL, "suggestion": suggestion,
                    "reason": (f"This is what {suggestion} is for. A dedicated tool is "
                               "faster than the GUI, reports its own errors, and does not "
                               "depend on what happens to be on screen.")}

    for pattern, suggestion in _BETTER_IN_THE_BROWSER:
        if re.search(pattern, text):
            return {"layer": LAYER_BROWSER, "suggestion": suggestion,
                    "reason": (f"This is browser work, and {suggestion} can do it without "
                               "moving the mouse - reading a page directly is more reliable "
                               "than clicking through it.")}

    return {"layer": LAYER_GUI, "suggestion": None,
            "reason": ("No dedicated tool covers this, so driving the interface is "
                       "reasonable. Look at the screen before each step and check the "
                       "result after anything that matters.")}


class Session:
    """One bounded GUI errand: what it is for, what it has done, what it saw."""

    def __init__(self, goal: str, session_id: str = "", plan: Optional[List[str]] = None):
        self.goal = goal
        self.id = session_id or f"gui-{int(time.time())}"
        self.started_at = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.last_observation: Optional[str] = None
        self.observed_at: Optional[float] = None
        self.closed = False
        self.closed_reason: Optional[str] = None
        # The errand written down before it starts: "open the site", "find the
        # invoice", "download it", "move it into the project". It is a list of
        # intentions, not a program - nothing here executes a plan step. What it
        # buys is a place to say where the errand has got to, which is what turns
        # a run of clicks into something a person can follow and stop.
        wanted = [str(t).strip() for t in (plan or []) if str(t).strip()][:MAX_PLAN_STEPS]
        self.plan: List[Dict[str, Any]] = [
            {"n": i + 1, "what": text, "status": "pending", "note": None}
            for i, text in enumerate(wanted)
        ]

    # -- state ------------------------------------------------------------------

    @property
    def steps_taken(self) -> int:
        return len(self.steps)

    @property
    def steps_left(self) -> int:
        return max(0, MAX_STEPS - self.steps_taken)

    @property
    def plan_index(self) -> int:
        """The plan step being worked on, or the length of the plan when done.

        A step that was abandoned when the session stopped is not the step being
        worked on - nothing is.
        """
        for i, step in enumerate(self.plan):
            if step["status"] == "pending":
                return i
        return len(self.plan)

    def plan_progress(self) -> Dict[str, Any]:
        """Where the errand has got to. Counted, never estimated."""
        done = sum(1 for s in self.plan if s["status"] == "done")
        abandoned = sum(1 for s in self.plan if s["status"] == "abandoned")
        current = self.plan[self.plan_index] if self.plan_index < len(self.plan) else None
        return {
            "planned": len(self.plan),
            "done": done,
            "abandoned": abandoned,
            "finished": bool(self.plan) and done == len(self.plan),
            # No plan means no progress to report, and "Working..." is the honest
            # thing to say rather than a fraction with an invented denominator.
            "position": f"{done}/{len(self.plan)}" if self.plan else "working",
            "current": current["what"] if current else None,
            "steps": [dict(s) for s in self.plan],
        }

    def complete_plan_step(self, note: str = "") -> Optional[Dict[str, Any]]:
        """Mark the plan step that is being worked on as done. The caller says so;
        nothing infers it from a click, because a click is not an outcome."""
        index = self.plan_index
        if index >= len(self.plan):
            return None
        self.plan[index]["status"] = "done"
        self.plan[index]["note"] = (note or "")[:200] or None
        _announce(self, f"{self.plan[index]['what']} - done")
        return dict(self.plan[index])

    def unverified_actions(self) -> List[Dict[str, Any]]:
        """Consequential steps nobody looked at afterwards.

        Clicking Send and never checking is the one failure a GUI session can hide
        completely, so it is reported rather than left to be assumed either way.
        """
        return [{"n": s["n"], "action": s["action"], "detail": s["detail"]}
                for s in self.steps if s.get("consequential") and not s.get("verified")]

    def observe(self, what_is_on_screen: str) -> None:
        """Record one look at the screen. Nothing captures on its own."""
        self.last_observation = what_is_on_screen or ""
        self.observed_at = time.time()
        # Anything consequential that was waiting to be checked has now been
        # looked at; whether it WORKED is expectation_holds' answer, not this one.
        for step in reversed(self.steps):
            if step.get("consequential") and not step.get("verified"):
                step["verified"] = True
                break

    def expectation_holds(self, expectation: str) -> Tuple[bool, str]:
        """Whether the last look matches what was expected.

        Plain substring matching over the words of the expectation, deliberately:
        a fuzzy match here would be a confident wrong answer about what is on the
        user's screen, and the whole point of checking is not to guess.
        """
        if self.last_observation is None:
            return False, "nothing has been looked at yet"
        wanted = [w for w in re.findall(r"[a-z0-9]+", str(expectation).lower()) if len(w) > 2]
        if not wanted:
            return False, "no expectation was given to check"
        seen = self.last_observation.lower()
        missing = [w for w in wanted if w not in seen]
        if missing:
            return False, f"expected to see {', '.join(missing)} on screen, and did not"
        return True, "the screen matches what was expected"

    # -- acting -----------------------------------------------------------------

    def may_act(self, action: str, detail: str = "") -> Tuple[bool, str]:
        """Whether the next step is allowed. Refusing is the useful answer."""
        if self.closed:
            return False, f"this session is closed ({self.closed_reason})"
        if self.steps_taken >= MAX_STEPS:
            return False, (f"this session has already taken {MAX_STEPS} steps; stop and "
                           "tell the user where it got to")
        if self.observed_at is None:
            return False, ("nothing has been looked at yet - check the screen before "
                           "acting on it")
        if time.time() - self.observed_at > OBSERVATION_MAX_AGE:
            return False, (f"the last look at the screen is more than "
                           f"{OBSERVATION_MAX_AGE}s old; look again before acting on it")

        signature = f"{action}:{detail}"
        repeats = sum(1 for s in self.steps if s["signature"] == signature)
        if repeats >= MAX_IDENTICAL_ACTIONS:
            return False, (f"'{action}' with the same target has already been tried "
                           f"{repeats} times and the screen has not changed as expected - "
                           "stop rather than repeating it")
        return True, "ok"

    def record(self, action: str, detail: str = "") -> Dict[str, Any]:
        step = {"n": self.steps_taken + 1, "action": action, "detail": detail,
                "signature": f"{action}:{detail}", "at": time.time(),
                "observed": self.last_observation is not None,
                "consequential": is_consequential(f"{action} {detail}"),
                "verified": False}
        self.steps.append(step)
        _announce(self, f"{action}" + (f" - {detail}" if detail and detail != action else ""))
        # A step invalidates the last look: the screen has moved on.
        self.observed_at = None
        return step

    def close(self, reason: str = "finished") -> Dict[str, Any]:
        self.closed = True
        self.closed_reason = reason
        # A plan step left "pending" on a session that has stopped reads as work
        # still to come. It is not: nothing is going to do it. Marking it
        # abandoned is the same honesty a cancelled task's steps get - the record
        # says what happened, and nothing is left looking like it is in progress.
        for step in self.plan:
            if step["status"] != "done":
                step["status"] = "abandoned"
                step["note"] = step["note"] or f"the session stopped: {reason}"
        _announce(self, f"session closed - {reason}")
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        unverified = self.unverified_actions()
        return {
            "session_id": self.id,
            "goal": self.goal,
            "steps_taken": self.steps_taken,
            "steps_left": self.steps_left,
            "closed": self.closed,
            "closed_reason": self.closed_reason,
            "progress": self.plan_progress(),
            "actions": [{"n": s["n"], "action": s["action"], "detail": s["detail"],
                         "consequential": s.get("consequential", False),
                         "verified": s.get("verified", False)}
                        for s in self.steps],
            "unverified_actions": unverified,
            "warning": (f"{len(unverified)} action(s) that change something were never "
                        "checked afterwards. Say so rather than reporting them as done."
                        if unverified else None),
            "plan_incomplete": ([s["what"] for s in self.plan if s["status"] == "abandoned"]
                                if self.closed else []),
        }


def _announce(session: "Session", message: str) -> None:
    """Put one line in the interface's activity log. Decides nothing, and a
    failure to log can never stop or change what is happening on screen."""
    try:
        from core import diagnostics

        diagnostics.record_activity("computer", message, session_id=session.id)
    except Exception:
        logger.debug("Couldn't mirror a computer-use step to the activity log.")


_SESSIONS: Dict[str, Session] = {}


def open_session(goal: str, plan: Optional[List[str]] = None) -> Session:
    """Start an errand. Old sessions are dropped rather than accumulating."""
    cutoff = time.time() - SESSION_IDLE_TIMEOUT
    for stale in [k for k, s in _SESSIONS.items()
                  if s.closed or s.started_at < cutoff]:
        _SESSIONS.pop(stale, None)
    session = Session(goal, plan=plan)
    _SESSIONS[session.id] = session
    _announce(session, f"started: {goal[:80]}")
    return session


def get_session(session_id: str) -> Optional[Session]:
    return _SESSIONS.get(session_id)


def active_sessions() -> List[Session]:
    return [s for s in _SESSIONS.values() if not s.closed]
