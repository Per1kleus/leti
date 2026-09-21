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
# How many unexpected screens end a session. The first is allowed to be a dialog
# that had not finished drawing; the second is the errand being somewhere else
# than Leti thinks it is, which is when to stop. Neither one ever acts.
MAX_MISMATCHES = 2
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


# Words that describe almost any screen. They carry no information about WHICH
# screen is in front of Leti, so requiring them verbatim rejects correct screens
# for wording - the failure mode that makes people stop writing expectations.
FURNITURE = frozenset({
    "the", "and", "with", "showing", "shows", "should", "window", "dialog", "screen",
    "page", "tab", "panel", "view", "app", "application", "open", "opened", "visible",
    "displayed", "currently", "menu", "box", "area", "section", "list", "item",
})


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


# --------------------------------------------------------------------------- #
# Naming what to act on, rather than where to click
#
# A coordinate is the weakest possible description of a target: it is wrong the
# moment the window moves, the font changes or the list scrolls, and it cannot
# be checked afterwards because a pixel has no identity. A name can be - "the
# Settings button", "the tab called General" - and a name can be looked for in
# what was actually read off the screen.
#
# So this is the layer between "click Settings" and a click: it says whether the
# thing being aimed at is actually visible, and it refuses when it is not. It
# does not capture, click or type; every one of those is still the existing
# tool, authorised by SafetyGuard the ordinary way, which is what stops driving
# the interface becoming a way around the rules that govern the equivalent tool.
# --------------------------------------------------------------------------- #

# How a target was described, weakest last. Reported so a session that had to
# fall back to coordinates says so rather than looking the same as one that did
# not.
BY_NAME = "by name"
BY_TEXT = "by visible text"
BY_ROLE = "by role and label"
BY_COORDINATES = "by coordinates"

TARGET_PRECEDENCE = (BY_NAME, BY_TEXT, BY_ROLE, BY_COORDINATES)

# The words people use for the things they point at. Used to pull a target out
# of a plain instruction - "click the Save button" - without asking the model to
# re-state what it already said.
_CONTROL_WORDS = (
    "button", "link", "tab", "menu", "menu item", "field", "box", "checkbox",
    "dropdown", "option", "icon", "toggle", "switch", "row", "cell", "entry",
    "item",
)

_TARGET_PATTERNS = (
    # click the "Save" button / click the Save button / click on Save
    re.compile(r"\b(?:click|press|tap|select|choose|open|activate)\s+"
               r"(?:on\s+)?(?:the\s+)?[\"\u2018\u2019\u201c\u201d']?"
               r"(?P<label>[\w][\w \-/&.]{0,48}?)[\"\u2018\u2019\u201c\u201d']?\s+"
               r"(?P<control>" + "|".join(_CONTROL_WORDS) + r")\b", re.I),
    re.compile(r"\b(?:click|press|tap|select|choose|activate)\s+"
               r"(?:on\s+)?(?:the\s+)?[\"\u2018\u2019\u201c\u201d']"
               r"(?P<label>[^\"\u2018\u2019\u201c\u201d']{1,48})"
               r"[\"\u2018\u2019\u201c\u201d']", re.I),
    re.compile(r"\b(?:type|enter|put)\s+.{0,40}?\b(?:in|into)\s+(?:the\s+)?"
               r"(?P<label>[\w][\w \-/&.]{0,48}?)\s+"
               r"(?P<control>" + "|".join(_CONTROL_WORDS) + r")\b", re.I),
)

_COORDINATES = re.compile(r"\b(?:at|to)?\s*\(?\s*(?P<x>\d{1,5})\s*,\s*(?P<y>\d{1,5})\s*\)?")


def describe_target(instruction: str) -> Dict[str, Any]:
    """What this instruction is aiming at, and how well it is described.

    Never raises and never guesses a label out of thin air: an instruction with
    nothing nameable in it comes back with target None, which is itself the
    useful answer - it says the step is about to act on a coordinate.
    """
    text = str(instruction or "")
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(text)
        if match:
            label = (match.groupdict().get("label") or "").strip(" \"'\u2018\u2019\u201c\u201d")
            if label:
                control = (match.groupdict().get("control") or "").strip().lower()
                return {"target": label, "control": control or None,
                        "how": BY_ROLE if control else BY_NAME,
                        "instruction": text[:200]}
    coordinates = _COORDINATES.search(text)
    if coordinates:
        return {"target": None, "control": None, "how": BY_COORDINATES,
                "at": [int(coordinates.group("x")), int(coordinates.group("y"))],
                "instruction": text[:200],
                "why_weak": ("A coordinate cannot be checked afterwards and is wrong "
                             "the moment anything moves. Name what is being clicked "
                             "if the screen shows a name for it.")}
    return {"target": None, "control": None, "how": None, "instruction": text[:200]}


def _visible_label(label: str, observation: str) -> Tuple[bool, str]:
    """Is this label actually in what was read off the screen?

    Whole-label first, then every significant word of it. A label whose words
    are all present but scattered is reported as a weaker match rather than as
    the same thing, because "Save" and "Save as" are different buttons.
    """
    seen = str(observation or "").lower()
    wanted = str(label or "").strip().lower()
    if not wanted:
        return False, "no label to look for"
    if not seen:
        return False, "nothing has been read off the screen"
    if wanted in seen:
        return True, f"'{label}' is on screen"
    words = [w for w in re.findall(r"[a-z0-9]+", wanted)
             if len(w) > 2 and w not in FURNITURE]
    if words and all(w in seen for w in words):
        return True, (f"every word of '{label}' is on screen, though not together - "
                      "check it is the right one before acting")
    missing = [w for w in words if w not in seen] or [wanted]
    return False, f"'{label}' is not on screen (missing: {', '.join(missing[:4])})"


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
        # How many times the screen has not been what was expected. See mismatch().
        self.mismatches = 0
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

        The one concession to real interfaces is FURNITURE below. "The Settings
        window, General tab" and "Settings - General" are the same screen, and
        failing the second because it does not contain the word "window" stops a
        session for a synonym rather than for a mismatch. The words that say WHICH
        screen this is - Settings, General - are still all required, and a screen
        missing any of them still fails. Only the words that would be true of
        almost any screen are optional, and an expectation made entirely of those
        is not an expectation at all.
        """
        if self.last_observation is None:
            return False, "nothing has been looked at yet"
        words = [w for w in re.findall(r"[a-z0-9]+", str(expectation).lower()) if len(w) > 2]
        wanted = [w for w in words if w not in FURNITURE]
        if not wanted:
            return False, ("that expectation is only generic words"
                           if words else "no expectation was given to check")
        seen = self.last_observation.lower()
        missing = [w for w in wanted if w not in seen]
        if missing:
            return False, f"expected to see {', '.join(missing)} on screen, and did not"
        return True, "the screen matches what was expected"

    def mismatch(self) -> bool:
        """The screen was not what was expected. Whether to look again or stop.

        A real interface is not always where it was a second ago: a dialog that
        had not finished drawing, a page still loading, a notification on top of
        the window. Ending the errand for that loses work that was going fine,
        and doing it twice in a row is not that - it is the errand being lost.

        So the FIRST mismatch spends the look and asks for another, and the second
        stops the session. What does not change is the part that matters: this
        clears observed_at, so may_act refuses until the screen has been looked at
        again, and nothing can act on a screen that was not what was expected.
        Recovering means looking again, never clicking anyway.
        """
        self.mismatches += 1
        self.observed_at = None
        if self.mismatches >= MAX_MISMATCHES:
            return False
        _announce(self, "the screen was not what was expected - looking again")
        return True

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

        # If the step names something, that something has to be on the screen
        # that was just read. This is the difference between driving an interface
        # and clicking blindly: a button that is not there is not a button that
        # will be there after the click.
        aimed_at = describe_target(f"{action} {detail}".strip())
        if aimed_at.get("target"):
            visible, why = _visible_label(aimed_at["target"], self.last_observation or "")
            if not visible:
                return False, (f"{why}. Do not click where it used to be - look again, "
                               "and if it is still not there say so rather than guessing.")
        return True, "ok"

    def aim(self, instruction: str) -> Dict[str, Any]:
        """Work out what a step is aiming at and whether it can be seen.

        The IDENTIFY TARGET step, separated from acting on purpose: a caller can
        ask before it commits, and the answer is the same one may_act will give.
        """
        found = describe_target(instruction)
        if found.get("target"):
            visible, why = _visible_label(found["target"], self.last_observation or "")
            found["visible"] = visible
            found["evidence"] = why
        elif found.get("how") == BY_COORDINATES:
            found["visible"] = None
            found["evidence"] = ("A coordinate cannot be confirmed against what is on "
                                 "screen. Nothing here can say whether it is the right "
                                 "place to click.")
        else:
            found["visible"] = None
            found["evidence"] = ("This step does not name anything to aim at. Say what "
                                 "is being clicked so it can be checked.")
        found["prefer"] = (
            "Name the control if the screen shows a name for it; coordinates are the "
            "last resort, not the first."
            if found.get("how") in (None, BY_COORDINATES) else None)
        return found

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
            "unexpected_screens": self.mismatches,
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
