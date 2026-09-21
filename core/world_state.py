"""What Leti believed was true, so it can notice when it stopped being true.

A long action is a sequence of small ones, and every small one is taken on the
strength of something observed earlier: the window that was in front, the file
that was there, the record that said "proposal". Most of the time that belief
survives the next step. When it does not - the dialog closed, the page
navigated, somebody touched the mouse, the token expired - an agent with no
memory of what it believed cannot tell the difference between "that did not
work" and "the world is not where I left it", and those need opposite responses.

So this keeps one small record per active piece of work:

    mode, task, goal, project, where it is (app/page), the last action that
    worked, what the next one expects, what was actually seen, the request it
    all came from, and the entities already resolved for it.

The loop it supports is the one Leti already runs in Computer Use, generalised:

    OBSERVE -> UNDERSTAND -> ACT -> OBSERVE AGAIN -> COMPARE -> CONTINUE or RECOVER

WHAT THIS IS NOT. Not a store: it is a dict in memory, one entry per active
task, replaced rather than appended to, and gone when the process ends. Not a
second task manager: core/task_manager.py owns status, steps, retries and the
recovery budget, and this is consulted by it rather than competing with it. Not
a second Computer Use session: core/computer_use.py still owns the screen, its
expectations and its mismatch counter, and expectation_holds() below is that
comparison lifted out so a file operation or an API call can use the same one.
Not a background anything: every function here runs because a step ran.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.world_state")

# --------------------------------------------------------------------------- #
# Why something did not work
#
# The classification matters because the right response differs: a permission
# problem must not be retried differently (that is trying to get around it), a
# stale observation must be re-observed rather than re-attempted, and an
# ambiguous state must stop and ask rather than pick.
# --------------------------------------------------------------------------- #

WRONG_ACTION = "wrong action"                   # it ran, it did the wrong thing
TOOL_FAILURE = "tool failure"                   # the tool itself errored
ENVIRONMENT_CHANGED = "environment changed"     # the world moved under the plan
STALE_OBSERVATION = "stale observation"         # acting on something no longer true
PERMISSION_PROBLEM = "permission problem"       # not allowed, or not approved
MISSING_INFORMATION = "missing information"     # cannot proceed without an answer
DEPENDENCY_PROBLEM = "dependency problem"       # something it needs is not installed
EXTERNAL_SERVICE = "external service failure"   # somebody else's machine
USER_INTERVENTION = "user intervention"         # the person did something
AMBIGUOUS_STATE = "ambiguous state"             # cannot tell what is true
UNKNOWN = "unknown"                             # say so rather than pick

KINDS = (WRONG_ACTION, TOOL_FAILURE, ENVIRONMENT_CHANGED, STALE_OBSERVATION,
         PERMISSION_PROBLEM, MISSING_INFORMATION, DEPENDENCY_PROBLEM,
         EXTERNAL_SERVICE, USER_INTERVENTION, AMBIGUOUS_STATE, UNKNOWN)

# What to do about each. `retry_differently` is the only field that unlocks the
# self-recovery budget core/task_manager.py already owns - everything else either
# stops for a person or re-observes first.
RESPONSES: Dict[str, Dict[str, Any]] = {
    WRONG_ACTION: {"retry_differently": True, "re_observe": True,
                   "do": "Look at what actually happened, then take a different action - "
                         "not the same one again."},
    TOOL_FAILURE: {"retry_differently": True, "re_observe": False,
                   "do": "Read the real error. If it names something fixable, fix that; "
                         "otherwise try a different tool for the same goal."},
    ENVIRONMENT_CHANGED: {"retry_differently": True, "re_observe": True,
                          "do": "The plan was made for a state that no longer holds. "
                                "Re-observe and re-plan from what is there now."},
    STALE_OBSERVATION: {"retry_differently": True, "re_observe": True,
                        "do": "Observe again before acting; the last observation is too "
                              "old to act on."},
    PERMISSION_PROBLEM: {"retry_differently": False, "re_observe": False,
                         "do": "Stop. Say what was refused and what permission it needs. "
                               "Never work around it."},
    MISSING_INFORMATION: {"retry_differently": False, "re_observe": False,
                          "do": "Stop and ask the one question that unblocks it. Do not "
                                "guess the missing value."},
    DEPENDENCY_PROBLEM: {"retry_differently": True, "re_observe": False,
                         "do": "Say what is missing and offer to install it, rather than "
                               "working around its absence."},
    EXTERNAL_SERVICE: {"retry_differently": True, "re_observe": False,
                       "do": "Somebody else's service failed. One retry is reasonable; "
                             "after that report it as their outage, not as a result."},
    USER_INTERVENTION: {"retry_differently": False, "re_observe": True,
                        "do": "The user acted. Stop, look at where things are now, and "
                              "ask before continuing."},
    AMBIGUOUS_STATE: {"retry_differently": False, "re_observe": True,
                      "do": "Leti cannot tell what is true. Observe again; if it is still "
                            "unclear, ask rather than assume."},
    UNKNOWN: {"retry_differently": False, "re_observe": True,
              "do": "The cause is not known. Say that plainly rather than guessing at "
                    "a fix."},
}

# The kinds that unlock core/task_manager.py's self-recovery budget - a second,
# different attempt at the same step while nobody is watching.
#
# Deliberately narrower than RESPONSES' `retry_differently`. That field is advice
# to a model inside a turn, with a person there to see what happens. This is what
# an UNATTENDED task may do on its own, and the failure mode of automatic recovery
# is a machine that spends an afternoon failing in new ways. A tool that raised a
# traceback gets reported, not re-attempted in the dark.
RECOVERABLE_KINDS = frozenset({EXTERNAL_SERVICE, ENVIRONMENT_CHANGED,
                               STALE_OBSERVATION, WRONG_ACTION})


def unlocks_recovery(problem: Any) -> bool:
    """May an unattended task try this step a different way?

    Never for an approval stop or a hard block: those are answers, not failures,
    and retrying them differently would be trying to get around them.
    """
    return classify(problem)["kind"] in RECOVERABLE_KINDS


# Markers in an error's text, most specific first. Order matters: "permission
# denied connecting to host" is a connection problem wearing the word permission,
# and a token that expired is an external service saying no, not a local block.
_MARKERS = (
    (PERMISSION_PROBLEM, ("permissiondenied", "confirmationdenied", "permissionerror",
                          "permission denied", "not authorized", "unauthorized",
                          "forbidden", "403", "operation not permitted",
                          "access is denied", "declined", "denied by the user",
                          "requires confirmation", "needs approval")),
    (DEPENDENCY_PROBLEM, ("modulenotfounderror", "no module named", "command not found",
                          "is not installed", "importerror", "executable not found")),
    (EXTERNAL_SERVICE, ("503", "502", "504", "429", "rate limit", "too many requests",
                        "service unavailable", "bad gateway", "upstream", "timed out",
                        "timeout", "connection reset", "unreachable", "temporarily")),
    (MISSING_INFORMATION, ("missing required", "no value for", "is required",
                           "not configured", "no credentials", "which one did you mean")),
    (STALE_OBSERVATION, ("stale", "no longer on screen", "element not found",
                         "window not found", "no such window", "file has changed")),
    (ENVIRONMENT_CHANGED, ("no longer exists", "was moved", "already exists",
                           "state has changed", "not in the expected state")),
    (TOOL_FAILURE, ("traceback", "exception", "error:", "failed to")),
)


def classify(problem: Any, expected: str = "", observed: str = "") -> Dict[str, Any]:
    """Why something did not work, and what that means for what to do next.

    Takes an exception or a string. Never raises, and never reports a cause it
    did not find evidence for: UNKNOWN is a real answer here and is used.
    """
    text = str(problem or "").strip()
    lowered = text.lower()
    type_name = type(problem).__name__.lower() if isinstance(problem, BaseException) else ""
    haystack = f"{type_name} {lowered}"

    kind = UNKNOWN
    if not text:
        # Nothing failed, but the result was not what was expected. That is a
        # different thing from an error and must not be filed as one.
        if expected and observed:
            kind = WRONG_ACTION
    else:
        for candidate, markers in _MARKERS:
            if any(marker in haystack for marker in markers):
                kind = candidate
                break

    if kind is UNKNOWN and expected and observed:
        holds, _ = expectation_holds(expected, observed)
        if not holds:
            kind = WRONG_ACTION

    response = RESPONSES[kind]
    return {
        "kind": kind,
        "problem": text[:400] or None,
        "expected": expected or None,
        "observed": observed or None,
        "retry_differently": response["retry_differently"],
        "re_observe": response["re_observe"],
        "do": response["do"],
        "certain": kind is not UNKNOWN,
    }


def expectation_holds(expected: str, observed: str) -> tuple:
    """Does what was seen contain what was expected?

    The same comparison core/computer_use.py makes about a screen, lifted out so
    a file, an API response or a record can be compared the same way. Words of
    the expectation that appear in the observation; a majority has to land, and
    an empty expectation is not a passed check.
    """
    from core.computer_use import FURNITURE

    wanted = [w for w in str(expected or "").lower().split()
              if len(w) > 2 and w not in FURNITURE]
    if not wanted:
        return False, "nothing specific was expected, so nothing was confirmed"
    seen = str(observed or "").lower()
    hits = [w for w in wanted if w in seen]
    if len(hits) * 2 >= len(wanted):
        return True, f"{len(hits)} of {len(wanted)} expected things are there"
    missing = [w for w in wanted if w not in seen]
    return False, f"not there: {', '.join(missing[:5])}"


# --------------------------------------------------------------------------- #
# The record itself
# --------------------------------------------------------------------------- #

@dataclass
class State:
    """What Leti believes about one piece of work right now.

    Every field is optional, and absent means "not known" rather than "not the
    case". A world state that quietly filled its own gaps would be the thing it
    exists to prevent.
    """
    key: str
    request: str = ""
    mode: str = "default"
    task_id: Optional[str] = None
    goal: Optional[str] = None
    project: Optional[str] = None
    location: Optional[str] = None            # the app, page or directory in play
    last_success: Optional[str] = None
    expected: Optional[str] = None
    observed: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    observed_at: float = 0.0
    updated_at: float = field(default_factory=time.time)

    def age(self, now: Optional[float] = None) -> float:
        """How long since the last observation, in seconds. Infinite if never."""
        if not self.observed_at:
            return float("inf")
        return (now if now is not None else time.time()) - self.observed_at

    def describe(self) -> Dict[str, Any]:
        return {
            "request": self.request or None, "mode": self.mode, "task": self.task_id,
            "goal": self.goal, "project": self.project, "where": self.location,
            "last_successful_action": self.last_success,
            "expected_next": self.expected, "last_observed": self.observed,
            "entities": self.entities or None,
            "observation_age_seconds": (None if self.observed_at == 0
                                        else round(self.age(), 1)),
        }


# One entry per active piece of work. Bounded by how many tasks can be active at
# once, cleared by forget(), and never written to disk - a belief about where a
# window was is worthless after a restart and dangerous if trusted.
_states: Dict[str, State] = {}

# An observation older than this is not evidence about now. Deliberately short:
# the cost of looking again is one screenshot or one read, and the cost of acting
# on a stale belief is the wrong click.
STALE_AFTER_SECONDS = 90


def begin(key: str, request: str = "", **fields: Any) -> State:
    """Start (or restart) the record for one piece of work."""
    state = State(key=str(key), request=str(request or ""))
    for name, value in fields.items():
        if hasattr(state, name):
            setattr(state, name, value)
    if state.mode == "default":
        try:
            from core import modes

            state.mode = modes.current()
        except Exception:
            pass
    _states[state.key] = state
    return state


def get(key: str) -> Optional[State]:
    return _states.get(str(key))


def forget(key: str = "") -> None:
    """Drop one record, or all of them. Called when work finishes."""
    if key:
        _states.pop(str(key), None)
    else:
        _states.clear()


def active() -> List[Dict[str, Any]]:
    return [s.describe() for s in _states.values()]


def observe(key: str, what: str, location: str = "",
            now: Optional[float] = None) -> Dict[str, Any]:
    """Record what is actually there. The O in both halves of the loop."""
    state = _states.get(str(key))
    if state is None:
        state = begin(key)
    state.observed = str(what or "")[:2000]
    state.observed_at = now if now is not None else time.time()
    state.updated_at = state.observed_at
    if location:
        state.location = str(location)[:200]
    return state.describe()


def expect(key: str, what: str) -> Dict[str, Any]:
    """Say what the next action should produce, BEFORE taking it.

    Written down first on purpose: an expectation formed after seeing the result
    is not an expectation, it is a description.
    """
    state = _states.get(str(key)) or begin(key)
    state.expected = str(what or "")[:400]
    state.updated_at = time.time()
    return state.describe()


def succeeded(key: str, action: str) -> Dict[str, Any]:
    state = _states.get(str(key)) or begin(key)
    state.last_success = str(action or "")[:200]
    state.expected = None
    state.updated_at = time.time()
    return state.describe()


def remember_entity(key: str, role: str, value: Any) -> None:
    """Which lead "the customer" turned out to be, so the next step does not
    resolve it again - and does not resolve it DIFFERENTLY."""
    state = _states.get(str(key)) or begin(key)
    state.entities[str(role)] = value
    state.updated_at = time.time()


def compare(key: str, observed: str = "", now: Optional[float] = None) -> Dict[str, Any]:
    """Is the world where the last action said it would be?

    The COMPARE step. Returns a verdict in core/verification.py's vocabulary, so
    "the screen is where I expected" reads the same as "the file holds what I
    wrote" - which is the whole point of having one vocabulary.
    """
    from core import verification

    state = _states.get(str(key))
    if state is None:
        return {"verdict": verification.NOT_APPLICABLE,
                "detail": "Nothing is being tracked under that name.",
                "continue": False}
    if observed:
        observe(key, observed, now=now)
    if not state.expected:
        return {"verdict": verification.NOT_APPLICABLE,
                "detail": "Nothing was expected of the last action, so there is nothing "
                          "to compare against.",
                "continue": True, "state": state.describe()}
    if not state.observed:
        return {"verdict": verification.NOT_VERIFIED,
                "detail": "Nothing has been observed since the action, so whether it "
                          "worked is unknown.",
                "continue": False, "next": "observe before continuing",
                "state": state.describe()}

    age = state.age(now)
    if age > STALE_AFTER_SECONDS:
        return {"verdict": verification.NOT_VERIFIED,
                "detail": f"The last observation is {int(age)}s old, which is too old to "
                          "act on.",
                "continue": False, "next": "observe again",
                "classification": classify("stale observation"),
                "state": state.describe()}

    holds, why = expectation_holds(state.expected, state.observed)
    if holds:
        return {"verdict": verification.VERIFIED, "detail": why, "continue": True,
                "state": state.describe()}
    return {"verdict": verification.FAILED, "detail": why, "continue": False,
            "classification": classify("", expected=state.expected,
                                       observed=state.observed),
            "state": state.describe()}


def recovery_note(key: str, problem: Any) -> str:
    """One line telling the next attempt what changed and what not to repeat.

    Fed into the instruction core/task_manager.py already builds for a recovery
    attempt, rather than becoming a second recovery mechanism.
    """
    state = _states.get(str(key))
    found = classify(problem,
                     expected=(state.expected or "") if state else "",
                     observed=(state.observed or "") if state else "")
    parts = [f"The previous attempt did not work: {found['kind']}."]
    if state and state.last_success:
        parts.append(f"The last thing that did work was: {state.last_success}.")
    if state and state.expected:
        parts.append(f"It expected: {state.expected}.")
    if state and state.observed:
        parts.append(f"What was actually there: {state.observed[:200]}.")
    parts.append(found["do"])
    if not found["certain"]:
        parts.append("The cause is not known - do not invent one.")
    return " ".join(parts)
