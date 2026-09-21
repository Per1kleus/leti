"""What to try after something did not work, and when to stop trying.

Leti already knows WHY things fail: core/world_state.py classifies a failure
into one of eleven kinds and says whether a different approach could plausibly
help. Leti already knows how to check whether something worked:
core/verification.py. And core/task_manager.py already owns the budget - two
attempts, five minutes - and the step state they apply to.

What was missing was the bit in between: given a classified failure, WHAT
should the next attempt do differently, and is there any point. That is all
this module is:

    FAILURE -> CLASSIFY -> DIAGNOSE -> RECOVERY OPTION -> ACT -> VERIFY
            -> CONTINUE or ESCALATE

CLASSIFY is world_state's. ACT is the orchestrator's, through the ordinary
step. VERIFY is verification's. This module owns DIAGNOSE and RECOVERY OPTION,
and it owns them as text: it returns a description of what to try, and returns
nothing at all when there is nothing worth trying.

IT EXECUTES NOTHING. No tool call, no retry loop, no timer, no thread. It is a
pure function from a failure to a suggestion, which is what makes the budget
enforceable somewhere else and what stops recovery becoming a second executor.

FAIL CLOSED. Every kind that is not explicitly recoverable escalates. A failure
Leti cannot classify is escalated with "the cause is not known" rather than
retried in the dark, and a refusal is never retried at all - a different way of
doing something the user or the guard said no to is the same thing they said no
to.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core import world_state

logger = logging.getLogger("leti.recovery")

# The ceiling, wherever a caller has not got one of its own.
# core/task_manager.py's MAX_RECOVERIES_PER_STEP is the one that actually binds
# an unattended task and is deliberately smaller; this is the absolute limit
# that nothing may exceed.
MAX_ATTEMPTS = 3

CONTINUE = "continue"       # try again, differently
ESCALATE = "escalate"       # stop and tell the user
ASK = "ask"                 # stop and ask one question

# What a different attempt should actually DO, per failure kind. Only the kinds
# that unlock the recovery budget appear here; everything else escalates by
# falling off the end, which is the fail-closed direction.
_OPTIONS: Dict[str, Dict[str, Any]] = {
    world_state.ENVIRONMENT_CHANGED: {
        "diagnosis": "the world is not where the plan expected it to be",
        "try": ("Find where it is now before acting: list the folder, re-read the "
                "record, or look at the screen again. Then redo this step against "
                "what is actually there."),
        "verify": "the thing the step was about is where the new attempt says it is",
    },
    world_state.STALE_OBSERVATION: {
        "diagnosis": "Leti acted on something it had seen too long ago",
        "try": ("Observe again first - read the screen, re-read the file - and only "
                "then repeat the action against what is there now."),
        "verify": "the observation is fresh and the action matched it",
    },
    world_state.WRONG_ACTION: {
        "diagnosis": "the action ran and did not produce what was expected",
        "try": ("Do not repeat it. Look at what actually happened, work out which "
                "part of the expectation failed, and take a different action for "
                "the same goal."),
        "verify": "the expectation that failed now holds",
    },
    world_state.EXTERNAL_SERVICE: {
        "diagnosis": "somebody else's service failed or was too slow",
        "try": ("One more attempt is reasonable. If a different source or endpoint "
                "would answer the same question, use that instead of the one that "
                "just failed."),
        "verify": "the call returned real data rather than an error",
    },
}

# Kinds where a retry would be wrong, with what to say instead. These escalate
# or ask; none of them consumes an attempt.
_NO_RETRY: Dict[str, Dict[str, Any]] = {
    world_state.PERMISSION_PROBLEM: {
        "decision": ESCALATE,
        "diagnosis": "this was refused, not broken",
        "say": ("Say what was refused and what permission it would need. Never try "
                "a different route to the same action - a way around a refusal is "
                "the thing that was refused."),
    },
    world_state.MISSING_INFORMATION: {
        "decision": ASK,
        "diagnosis": "something needed is not known and guessing it would change the result",
        "say": "Ask the one question that unblocks this. Do not guess the missing value.",
    },
    world_state.AMBIGUOUS_STATE: {
        "decision": ASK,
        "diagnosis": "Leti cannot tell which of several things is meant or true",
        "say": "Say what the possibilities are and ask which one. Never pick.",
    },
    world_state.DEPENDENCY_PROBLEM: {
        "decision": ESCALATE,
        "diagnosis": "something this needs is not installed",
        "say": ("Name what is missing and offer to install it. Do not work around "
                "its absence with something that does a different job."),
    },
    world_state.USER_INTERVENTION: {
        "decision": ASK,
        "diagnosis": "the user did something while this was running",
        "say": "Say where things are now and ask before continuing.",
    },
    world_state.TOOL_FAILURE: {
        "decision": ESCALATE,
        "diagnosis": "the tool itself errored",
        "say": ("Report the real error. A tool that raised is not a transient "
                "condition to be retried in the dark while nobody is watching."),
    },
    world_state.UNKNOWN: {
        "decision": ESCALATE,
        "diagnosis": "the cause is not known",
        "say": ("Say that plainly. Do not invent a cause and do not try something "
                "at random to see if it helps."),
    },
}

# A failure that comes back identical is not a transient one. Repeating the same
# attempt against the same error is the loop this exists to prevent, so the
# second identical failure ends recovery whatever the budget says.
IDENTICAL_FAILURE_LIMIT = 2


def _fingerprint(problem: Any) -> str:
    """Enough of a failure to recognise it coming back. Type and first words."""
    name = type(problem).__name__ if isinstance(problem, BaseException) else ""
    return f"{name}:{' '.join(str(problem or '').lower().split()[:8])}"


def plan(problem: Any, attempts_used: int = 0, max_attempts: int = MAX_ATTEMPTS,
         expected: str = "", observed: str = "",
         previous_failures: Optional[List[str]] = None,
         cancelled: bool = False) -> Dict[str, Any]:
    """What to do about one failure. Never raises; never executes anything.

    `attempts_used` is how many recoveries this step has already had, which the
    caller owns. `previous_failures` are the fingerprints of what went wrong
    before, so an identical failure can end this rather than consume the budget.
    """
    try:
        found = world_state.classify(problem, expected=expected, observed=observed)
    except Exception as e:                                   # pragma: no cover - guard
        logger.debug(f"Classification failed inside recovery planning: {e}")
        found = {"kind": world_state.UNKNOWN, "certain": False,
                 "do": "The cause is not known.", "problem": str(problem or "")[:400]}

    kind = found["kind"]
    fingerprint = _fingerprint(problem)
    seen_before = list(previous_failures or []).count(fingerprint)

    base = {
        "kind": kind,
        "certain": found.get("certain", False),
        "problem": found.get("problem"),
        "fingerprint": fingerprint,
        "attempts_used": attempts_used,
        "attempts_allowed": min(max_attempts, MAX_ATTEMPTS),
    }

    # A cancelled task recovers from nothing. The user said stop, and trying a
    # different way is still trying.
    if cancelled:
        return {**base, "decision": ESCALATE, "recover": False,
                "diagnosis": "the task was cancelled while this step was failing",
                "say": "Report what had happened when it was cancelled. Do not retry.",
                "escalation": _escalation(kind, found, attempts_used,
                                          "it was cancelled")}

    if kind in _NO_RETRY:
        spec = _NO_RETRY[kind]
        return {**base, "decision": spec["decision"], "recover": False,
                "diagnosis": spec["diagnosis"], "say": spec["say"],
                "escalation": _escalation(kind, found, attempts_used,
                                          spec["diagnosis"])}

    if seen_before >= IDENTICAL_FAILURE_LIMIT - 1:
        return {**base, "decision": ESCALATE, "recover": False,
                "diagnosis": "the same failure has come back unchanged",
                "say": ("Say that the same thing failed the same way twice. Something "
                        "about the approach is wrong, not about the moment."),
                "escalation": _escalation(kind, found, attempts_used,
                                          "the identical failure repeated")}

    if attempts_used >= min(max_attempts, MAX_ATTEMPTS):
        return {**base, "decision": ESCALATE, "recover": False,
                "diagnosis": "the recovery budget for this step is spent",
                "say": ("Say what was tried and stop. More attempts at this point are "
                        "a machine that will not admit defeat."),
                "escalation": _escalation(kind, found, attempts_used,
                                          "the retry limit was reached")}

    option = _OPTIONS.get(kind)
    if option is None:
        # Fail closed: an unrecognised but non-refusal kind still escalates.
        return {**base, "decision": ESCALATE, "recover": False,
                "diagnosis": found.get("do", "the cause is not known"),
                "say": "Report what failed rather than trying something at random.",
                "escalation": _escalation(kind, found, attempts_used,
                                          "there is no known way to recover from this")}

    return {**base, "decision": CONTINUE, "recover": True,
            "diagnosis": option["diagnosis"],
            "try": option["try"],
            "verify": option["verify"],
            "say": None}


def _escalation(kind: str, found: Dict[str, Any], attempts: int,
                why: str) -> Dict[str, Any]:
    """What the user is told when recovery stops. Five specific things."""
    return {
        "what_failed": found.get("problem") or "the step did not do what it should have",
        "classified_as": kind,
        "what_was_attempted": (f"{attempts} automatic recovery attempt(s)"
                               if attempts else "no automatic recovery was attempted"),
        "why_it_stopped": why,
        "user_intervention_required": kind in (
            world_state.PERMISSION_PROBLEM, world_state.MISSING_INFORMATION,
            world_state.AMBIGUOUS_STATE, world_state.USER_INTERVENTION,
            world_state.DEPENDENCY_PROBLEM),
    }


def instruction(decision: Dict[str, Any], step_text: str = "") -> str:
    """The recovery plan as one paragraph for the next attempt at a step.

    Text, not behaviour: the orchestrator runs the step exactly as it always
    does, with this prepended, and every tool it reaches for meets SafetyGuard
    the ordinary way.
    """
    if not decision.get("recover"):
        return ""
    parts = [f"The previous attempt failed: {decision['kind']} - "
             f"{decision['diagnosis']}."]
    if decision.get("problem"):
        parts.append(f"What it said: {decision['problem']}")
    parts.append(decision["try"])
    parts.append("Try a different way of doing this same step - an alternative source, "
                 "address or tool.")
    parts.append(f"This attempt counts as recovered only when {decision['verify']}. "
                 "Do not repeat the approach that just failed, and do not skip the step.")
    if not decision.get("certain"):
        parts.append("The cause is not certain - do not invent one.")
    return " ".join(parts)
