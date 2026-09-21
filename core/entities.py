"""Which one they meant, when the name alone does not settle it.

core/business.py has resolved references by name overlap since Business Mode
existed, and name overlap gets "Acme" right and "the customer" wrong. What a
person actually uses to disambiguate is everything around the name: they said it
two turns ago, it is the project that is open, it is the only one with a meeting
tomorrow, it is the one at the stage they were just complaining about.

This module is those signals. It does not replace the resolver - business.resolve
still owns the three outcomes and this is what it consults before choosing.

WHAT THIS IS NOT. There is no embedding here, no similarity model, no learned
ranking and no language understanding. Every signal below is a deterministic
match against something already recorded: a word, a date, an id, a stage. That
is worth saying plainly because "context-aware resolution" reads like semantics,
and claiming semantics that the code cannot do would be the same failure as
guessing which customer was meant.

THE RULE THAT DOES NOT BEND. One candidate with enough support wins. Two
candidates with comparable support is a question for the user, never a coin
flip - and support close enough to be inside the margin counts as comparable.
Nothing supported at all is "I cannot find that", not "here is the nearest
thing". choose() below is the only place that decides, so there is one copy of
that rule.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("leti.entities")

# Each signal: how much it is worth, and how it reads in a "because..." line.
# The weights are deliberately coarse. They exist to say "two independent things
# point here and only one points there", not to express confidence to two decimal
# places, and a scale fine enough to rank near-ties would be a scale fine enough
# to resolve them - which is the thing this must not do.
SIGNALS: Dict[str, Dict[str, Any]] = {
    "name": {"weight": 2, "because": "the name matches"},
    "alias": {"weight": 3, "because": "a known short name or alias matches"},
    "conversation": {"weight": 2, "because": "it came up earlier in this conversation"},
    "project": {"weight": 2, "because": "it belongs to the project that is open"},
    "recent": {"weight": 1, "because": "it was worked on recently"},
    "when": {"weight": 3, "because": "the date lines up"},
    "relationship": {"weight": 2, "because": "the company, stage or role matches"},
    "task": {"weight": 2, "because": "an active task refers to it"},
}

# Enough to act on: at least one signal beyond a single weak word match.
MIN_SUPPORT = 3
# And clearly ahead: a winner inside this margin of the runner-up is a question,
# not an answer. One whole signal's worth.
MARGIN = 2

RECENT_DAYS = 7
MAX_CONVERSATION_MESSAGES = 8
MAX_CANDIDATES_SHOWN = 6

# Words that cannot identify anything: grammar, and the nouns people use to
# refer to a thing WITHOUT naming it. "the customer" and "that lead" have to come
# out empty, because a reference with nothing distinctive in it is a question
# about which one, not a search for a name. core/business.py used to keep its own
# copy of this list; it delegates here now so both halves agree on what "bare"
# means.
_STOPWORDS = frozenset({
    "the", "a", "an", "my", "our", "this", "that", "for", "with", "about", "from",
    "and", "of", "to", "in", "on", "at", "is", "was", "it", "they", "them", "please",
    "one", "who", "what", "which", "we", "i", "me", "are", "working", "previous",
    "last", "customer", "client", "lead", "company", "contact", "person", "goal",
    "objective", "project", "task", "meeting", "call", "appointment", "quotation",
    "quote", "proposal", "deal", "opportunity",
})

# "tomorrow"/"today"/"this week" and friends. Deliberately a small, literal list:
# a date parser that guesses is a date parser that resolves the wrong meeting.
_WHEN_WORDS = {
    "today": (0, 1), "tonight": (0, 1), "tomorrow": (1, 2),
    "yesterday": (-1, 0), "this week": (0, 7), "next week": (7, 14),
}

_conversation_source: Optional[Callable[[], List[Dict[str, Any]]]] = None


def use_conversation_source(source: Optional[Callable[[], List[Dict[str, Any]]]]) -> None:
    """Point this at the conversation buffer that already exists.

    A function reference, not a copy: memory/session_memory.py stays the one place
    the conversation lives, and this reads it when a resolution is happening and
    at no other time. main.py and gui/api.py wire it once at startup; the tests
    set it to a list and back.
    """
    global _conversation_source
    _conversation_source = source


def _recent_conversation() -> List[str]:
    if _conversation_source is None:
        return []
    try:
        messages = _conversation_source() or []
    except Exception as e:
        logger.debug(f"Couldn't read the conversation for entity context: {e}")
        return []
    texts = [str(m.get("content", "")) for m in messages[-MAX_CONVERSATION_MESSAGES:]
             if isinstance(m, dict)]
    return [t for t in texts if t]


@dataclass
class Context:
    """What is true around the request, read from stores that already hold it.

    Every field can be empty, and empty means "this signal was not available"
    rather than "this signal says no" - a resolution that had no conversation to
    read should not look the same as one where the conversation pointed elsewhere.
    """
    now: float = field(default_factory=time.time)
    conversation: List[str] = field(default_factory=list)
    project: Optional[str] = None
    mode: str = "default"
    task_texts: List[str] = field(default_factory=list)
    unavailable: List[str] = field(default_factory=list)

    @property
    def conversation_text(self) -> str:
        return " \n".join(self.conversation).lower()


def gather(now: Optional[float] = None) -> Context:
    """Build the context. Reads only local state; contacts nothing; never raises."""
    context = Context(now=now if now is not None else time.time())

    context.conversation = _recent_conversation()
    if not context.conversation:
        context.unavailable.append("nothing has been said in this conversation yet")

    try:
        from core import modes

        context.mode = modes.current()
    except Exception:
        context.unavailable.append("the current mode could not be read")

    try:
        from tools.projects import get_active_project

        context.project = get_active_project()
    except Exception:
        context.unavailable.append("the active project could not be read")

    try:
        from core import task_manager

        context.task_texts = [
            f"{t.get('name', '')} {t.get('objective', '')}"
            for t in task_manager.load_tasks()
            if t.get("status") in task_manager.ACTIVE_STATUSES
        ]
    except Exception:
        context.unavailable.append("active tasks could not be read")

    return context


def significant_words(text: str) -> List[str]:
    """The words in a reference that could identify something."""
    words = re.findall(r"[a-z0-9][a-z0-9'&.-]*", str(text or "").lower())
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS]


def _aliases_of(name: str) -> List[str]:
    """Short forms a person would actually type for this name.

    First word, initials, and the acronym of a multi-word name. Not a nickname
    database: "Bob" for "Robert" is a fact about English, and inventing one here
    would be the module guessing.
    """
    parts = [p for p in re.split(r"[\s,]+", str(name or "").strip()) if p]
    if not parts:
        return []
    out = [parts[0].lower()]
    if len(parts) > 1:
        out.append("".join(p[0] for p in parts if p).lower())
        out.append(parts[-1].lower())
    return [a for a in out if len(a) > 1]


def _when_window(reference: str, now: float) -> Optional[tuple]:
    lowered = str(reference or "").lower()
    for phrase, (start_days, end_days) in _WHEN_WORDS.items():
        if phrase in lowered:
            day = 86400.0
            midnight = now - (now % day)
            return (midnight + start_days * day, midnight + end_days * day)
    return None


def signals_for(candidate: Dict[str, Any], reference: str,
                context: Optional[Context] = None) -> List[Dict[str, Any]]:
    """Every signal that points at this candidate, with why.

    The candidate is whatever the resolver produced: it must have a `name`, and
    anything else it carries (company, stage, project, starts, updated_at, id) is
    used when present and skipped when not.
    """
    context = context or Context()
    words = significant_words(reference)
    name = str(candidate.get("name") or "")
    lowered_name = name.lower()
    found: List[Dict[str, Any]] = []

    def add(signal: str, detail: str, times: int = 1) -> None:
        spec = SIGNALS[signal]
        found.append({"signal": signal, "weight": spec["weight"] * max(1, times),
                      "because": f"{spec['because']} ({detail})"})

    # Every word of the reference that lands in the name counts: "Beta Co" is a
    # stronger claim on a lead called Beta Co than "Beta" is, and a full-name
    # match has to be able to win outright.
    matched_words = [w for w in words if w in lowered_name]
    if matched_words:
        add("name", ", ".join(matched_words[:3]), times=len(matched_words))

    aliases = _aliases_of(name)
    alias_hits = [w for w in words if w in aliases and w not in lowered_name.split()]
    if alias_hits:
        add("alias", ", ".join(alias_hits[:2]))

    if name and context.conversation_text:
        # The whole name, not a word of it: "the project" appearing in the
        # conversation says nothing about which project.
        if lowered_name and lowered_name in context.conversation_text:
            add("conversation", f"'{name}' was mentioned")

    if context.project and str(candidate.get("project") or "") == context.project:
        add("project", context.project)
    elif context.project and lowered_name and lowered_name == context.project.lower():
        add("project", "it is the open project")

    updated = candidate.get("updated_at") or candidate.get("created_at")
    if isinstance(updated, (int, float)) and updated > 0:
        age_days = (context.now - float(updated)) / 86400.0
        if 0 <= age_days <= RECENT_DAYS:
            add("recent", f"{int(age_days)}d ago")

    window = _when_window(reference, context.now)
    starts = candidate.get("starts") or candidate.get("due")
    if window and isinstance(starts, (int, float)) and window[0] <= float(starts) < window[1]:
        add("when", "it falls in the period named")

    related = " ".join(str(candidate.get(k) or "")
                       for k in ("company", "stage", "status", "role", "tags")).lower()
    relationship_hits = [w for w in words if w and w in related]
    if relationship_hits:
        add("relationship", ", ".join(relationship_hits[:3]))

    if name and context.task_texts:
        if any(lowered_name in text.lower() for text in context.task_texts if text):
            add("task", "an active task names it")

    return found


def support(candidate: Dict[str, Any], reference: str,
            context: Optional[Context] = None) -> Dict[str, Any]:
    """How strongly the context points at this one candidate."""
    found = signals_for(candidate, reference, context)
    return {"score": sum(s["weight"] for s in found), "signals": found,
            "because": [s["because"] for s in found]}


def choose(candidates: Sequence[Dict[str, Any]], reference: str,
           context: Optional[Context] = None,
           kind: str = "") -> Dict[str, Any]:
    """The one they meant, the question to ask, or nothing found.

    Exactly three outcomes, and this is the only function that picks between
    them:

      resolved   one candidate has MIN_SUPPORT and leads the next by MARGIN
      ask        two or more are close enough that choosing would be a guess
      nothing    no candidate has any support at all
    """
    context = context or Context()
    what = f"'{reference}'" + (f" ({kind})" if kind else "")
    words = significant_words(reference)

    scored = []
    for candidate in candidates:
        result = support(candidate, reference, context)
        scored.append({**candidate, "support": result["score"],
                       "because": result["because"]})
    scored.sort(key=lambda c: -c["support"])

    # A reference with something distinctive in it ("Wakanda Industries") rules
    # out everything that matches none of it. A bare one ("the customer") rules
    # out nothing, because there is nothing in it to rule anything out WITH -
    # every entity of that kind stays in play and the context below is what
    # narrows it.
    pointed_at = [c for c in scored if c["support"] > 0]
    in_play = pointed_at if words else list(scored)
    if not in_play:
        return {
            "resolved": None, "candidates": [],
            "problem": f"Nothing matches {what}.",
            "looked_at": len(scored),
            "context_missing": context.unavailable or None,
        }

    # Exactly one candidate that anything points at, with nothing pointing
    # anywhere else, is not a choice between candidates - it is the answer. So is
    # one candidate full stop. The threshold and the margin below exist for the
    # case this module is really about: several plausible ones at once, where
    # picking is the failure.
    only = (pointed_at[0] if len(pointed_at) == 1 else
            in_play[0] if len(in_play) == 1 else None)
    if only is not None:
        return {
            "resolved": only, "candidates": in_play[:MAX_CANDIDATES_SHOWN],
            "how": "; ".join(only["because"]) or "it is the only one",
            "context_missing": context.unavailable or None,
        }

    supported = in_play
    top = supported[0]
    runner_up = supported[1]["support"] if len(supported) > 1 else 0
    if top["support"] >= MIN_SUPPORT and (top["support"] - runner_up) >= MARGIN:
        return {
            "resolved": top,
            "candidates": supported[:MAX_CANDIDATES_SHOWN],
            "how": "; ".join(top["because"]) or "one match",
            "context_missing": context.unavailable or None,
        }

    return {
        "resolved": None,
        "candidates": supported[:MAX_CANDIDATES_SHOWN],
        "ask": (f"{what} could be any of these - which one? "
                + "; ".join(f"{c.get('name')}"
                            + (f" ({'; '.join(c['because'][:2])})" if c["because"] else "")
                            for c in supported[:MAX_CANDIDATES_SHOWN])),
        "why_not_resolved": (
            "several candidates are supported about equally"
            if top["support"] - runner_up < MARGIN
            else "nothing points strongly enough at any one of them"),
        "context_missing": context.unavailable or None,
    }


def explain() -> Dict[str, Any]:
    """What resolution actually uses, for a diagnostics panel or a sceptical user."""
    return {
        "method": "deterministic signal matching, not semantic similarity",
        "signals": {name: spec["because"] for name, spec in SIGNALS.items()},
        "rule": (f"resolve when one candidate scores at least {MIN_SUPPORT} and leads "
                 f"the next by {MARGIN}; otherwise ask; never choose between two "
                 "comparable candidates"),
        "not_used": ["embeddings", "a language model", "learned ranking",
                     "a nickname database"],
    }
