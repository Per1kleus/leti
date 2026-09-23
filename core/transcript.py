"""What Leti just said, kept so it can be shown without being said again.

Leti speaks its answers. That is the point of it, and it is why the answer is no
longer printed into the session log the moment it is produced: a spoken
assistant that also prints everything it says is a chat window that happens to
make noise, and the reading eye wins every time. So the transcript is hidden by
default and the voice is the output.

Hidden is not gone. The answer that was spoken is held right here, complete, and
"show me the text" hands it back. That request costs nothing: no model call, no
regeneration, no second opinion about what was said - the words the user is
shown are the exact words that were spoken, because they are the same string.

WHAT THIS IS NOT

It is not memory. core/memory keeps the conversation; this keeps one response
and the state of the panel showing it. It is not a second store either - nothing
here is written to disk, because the question it answers ("is the current answer
on screen?") stops being meaningful the moment Leti restarts.

It also holds the last visual result, for the same reason and with the same
rule: "show that graph again" should show THAT graph, not compute a new one.
A visual is remembered only while the turn that made it is still the current
one; after that it is honestly gone, and saying so beats showing something
stale.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.transcript")

# How long a remembered visual is still "that graph". Past this the user has
# moved on and "show it again" is more likely to mean something new. Generous:
# the cost of keeping one payload is a few kilobytes, and the cost of getting
# this wrong is recomputing something expensive.
VISUAL_MAX_AGE_SECONDS = 900

# The longest answer kept whole. Past this the text is still shown, truncated,
# with the panel saying so - an answer this long is a document, and the document
# path already exists for that.
MAX_TEXT_CHARS = 20000

# How many visuals one turn may leave behind. A turn that made six charts is a
# turn whose "show it again" is ambiguous, and ambiguity is answered by asking.
MAX_VISUALS = 6


@dataclass
class Response:
    """One answer, and what looking at it would mean."""

    text: str = ""
    at: float = field(default_factory=time.time)
    visible: bool = False
    visuals: List[Dict[str, Any]] = field(default_factory=list)
    spoken: bool = False

    def age(self, now: Optional[float] = None) -> float:
        return (time.time() if now is None else now) - self.at


# The current response. One per process, because there is one conversation and
# one screen; a second surface connected to the same Leti is looking at the same
# answer, which is the behaviour gui/api.py's broadcast already assumes.
_current = Response()


def clear() -> None:
    """Forget the current response. Tests, and a session that has ended."""
    global _current
    _current = Response()


def begin() -> Response:
    """Start a fresh response for a turn that is about to run.

    Called when an ordinary turn begins - before the tools run, so that what
    they put on screen is collected against THIS answer rather than the last
    one. The previous response is dropped: "show me the text" means the text of
    what you just heard, and a stack of old ones would make that question
    ambiguous for no benefit.

    A turn answered without the model - showing, hiding, a mode command - never
    calls this, which is what stops "show me the text" from clearing the very
    answer it was asked to show.
    """
    global _current
    _current = Response()
    return _current


def said(text: str) -> Response:
    """Record what the turn ended up answering, on the response `begin` started.

    Separate from `begin` because the visuals arrive first: a chart is drawn by
    a tool call halfway through the turn, and the words come after it.
    """
    _current.text = str(text or "")[:MAX_TEXT_CHARS]
    _current.at = time.time()
    return _current


def current() -> Response:
    return _current


def mark_spoken() -> None:
    """The voice finished with this answer. Read by the tests that prove hiding
    the text does not touch the speaking, and by diagnostics."""
    _current.spoken = True


def add_visual(visual: Dict[str, Any]) -> None:
    """Remember something that was put on screen this turn.

    Only what a tool or the artifact rules actually produced - this decides
    nothing about whether a panel opens (core/artifacts.should_open does, and
    still does). It only means that if one did, "show it again" can find it.
    """
    if not isinstance(visual, dict) or not visual.get("type"):
        return
    if len(_current.visuals) >= MAX_VISUALS:
        return
    _current.visuals.append(visual)


def visuals(kind: str = "") -> List[Dict[str, Any]]:
    """The visuals this turn produced, newest last. `kind` filters by type.

    Empty once they are older than VISUAL_MAX_AGE_SECONDS: a payload that old
    is not what "that graph" means any more, and the caller says so rather than
    showing something from another conversation.
    """
    if _current.age() > VISUAL_MAX_AGE_SECONDS:
        return []
    if not kind:
        return list(_current.visuals)
    return [v for v in _current.visuals if v.get("type") == kind]


# --------------------------------------------------------------------------- #
# Showing and hiding
#
# Both are pure state changes over the response that already exists. Neither
# generates, regenerates, re-runs or asks anything - which is the guarantee the
# whole feature rests on, and the one the tests check most directly.
# --------------------------------------------------------------------------- #

def reveal() -> Dict[str, Any]:
    """Show the current answer. Returns what the surface should display."""
    if not _current.text:
        return {"shown": False, "text": "",
                "answer": "There is nothing to show yet - ask me something first."}
    _current.visible = True
    return {"shown": True, "text": _current.text,
            "answer": "Here it is." if _current.spoken else "Here is the text."}


def hide() -> Dict[str, Any]:
    """Hide the current answer.

    Hiding is a fact about a panel and nothing else. The text stays, the speech
    is not stopped, the task is not cancelled and memory is untouched - a
    guarantee that is easy to state and easy to break, so it is stated here and
    tested directly.
    """
    was = _current.visible
    _current.visible = False
    return {"shown": False, "was_visible": was, "answer": "Hidden."}


def is_visible() -> bool:
    return bool(_current.visible)


def section() -> Dict[str, Any]:
    """For the diagnostics panel. Says what is held, never what it says: an
    answer's text is the conversation, and a diagnostics dump is not where the
    conversation belongs."""
    return {
        "status": "ok",
        "holding": bool(_current.text),
        "characters": len(_current.text),
        "visible": _current.visible,
        "spoken": _current.spoken,
        "visuals": [v.get("type") for v in _current.visuals],
        "age_seconds": round(_current.age(), 1) if _current.text else 0.0,
    }
