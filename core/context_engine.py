"""What this request actually needs to know, and nothing else.

Leti has a lot of context available: the conversation, what it remembers about
the user, the open project's instructions and files, the business workspace, the
coding workspace, active tasks, goals, what is waiting for approval, what a watch
just found. Assembling all of it every turn is easy and wrong. It costs tokens on
turns that needed none of it, it pushes the tool schemas towards the edge of the
window, and it buries the two lines that mattered in forty that did not.

So a turn goes through six steps, all of them deterministic and none of them
involving a model:

    UNDERSTAND   core/intent.py has already read the request (tens of microseconds)
    ENTITIES     what it names - a file, a person, a project, a symbol
    SOURCES      which of the sources below could possibly help THIS request
    RETRIEVE     ask only those, and only the ones that are local and cheap
    RANK         drop what came back empty, order what did not
    PACKAGE      fit it in the budget, worst-first, and say what was left out

WHAT THIS IS NOT. There is no new memory here. No vector database - the one in
memory/vector_store.py is a source this consults, like the others. No index, no
embeddings computed on the way past, no background process building anything, no
second planner and no second model. Every source below reads a store that already
exists, when a request looks like it needs it, and not otherwise.

WHAT IT WILL NOT FETCH. A source that lives on somebody else's machine - the
calendar, the web, a CRM behind an API - is never fetched here, however relevant
it looks. Putting a network round trip in front of every turn is exactly the cost
this module exists to avoid, and the tools already reach those things when the
model asks. What the engine does instead is NAME them: the package says "this
request looks like it needs the calendar, which is a tool call", which is honest
and also more useful than a silent five-second delay.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from core import intent as intent_reader

logger = logging.getLogger("leti.context")

# How much room the assembled context may take, in characters.
#
# Roughly six thousand tokens at four characters each. The window is 28,672 and
# the tool schemas are the other big tenant, so this is a ceiling that a normal
# turn never approaches (most turns assemble two or three thousand characters) and
# that a pathological one - a project with long instructions, a full buffer, a
# proactive queue - cannot sail past without something being dropped on purpose
# and said out loud.
BUDGET_CHARS = 24_000

# The conversation is never trimmed below this many messages. Trimming history is
# the biggest saving available and also the easiest way to break a conversation,
# so the floor is generous and the full buffer is used whenever the request points
# backwards at all.
MIN_HISTORY_MESSAGES = 8


@dataclass
class Request:
    """One turn, as the engine understands it before retrieving anything."""
    text: str
    intent: intent_reader.Intent
    mode: str = "default"
    allow_optional: bool = True       # core/performance.py's budget for this turn
    first_turn_of_session: bool = False

    @property
    def lowered(self) -> str:
        return self.text.lower()

    def mentions(self, pattern: str) -> bool:
        return bool(re.search(pattern, self.lowered, re.I))

    def has_capability(self, *names: str) -> bool:
        return any(n in self.intent.capabilities for n in names)


@dataclass
class Piece:
    """One retrieved piece of context, before it is packaged."""
    source: str
    text: str
    priority: int                    # 1 keeps longest; higher goes first when trimming
    role: str = "system"

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class Package:
    """The assembled context, plus a truthful account of how it was assembled."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    included: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    needs_tools: List[Dict[str, str]] = field(default_factory=list)
    chars: int = 0
    seconds: float = 0.0

    def report(self) -> Dict[str, Any]:
        return {
            "included": list(self.included),
            "skipped": [{"source": name, "why": why} for name, why in self.skipped],
            "dropped_for_budget": list(self.dropped),
            "only_a_tool_can_reach": list(self.needs_tools),
            "chars": self.chars,
            "budget_chars": BUDGET_CHARS,
            "seconds": round(self.seconds, 4),
        }


# --------------------------------------------------------------------------- #
# Which sources a request could possibly need
#
# Each predicate answers one question - "could this request want the project's
# files?" - from the request alone, before anything is read. They are deliberately
# generous: the cost of consulting a local store that turns out to be empty is
# microseconds, and the cost of NOT consulting one that mattered is a wrong
# answer. What they exist to exclude is the large, always-on stuff: a project's
# file list on a weather question, a vector search on "what is 12 times 40".
# --------------------------------------------------------------------------- #

# A request that points at something already said. The engine's own reading, on
# top of intent.refers_back, because history is the one source where being wrong
# is expensive in both directions.
_BACKWARD = re.compile(
    r"\b(it|its|that|those|these|them|they|the (?:first|second|third|last|other|same)|"
    r"again|also|instead|as well|too|previous|earlier|you said|you just|before|"
    r"continue|carry on|keep going|what about|and now)\b", re.I)

_ABOUT_THE_USER = re.compile(
    r"\b(i|me|my|mine|myself|we|our|us|remember|prefer|preference|favourite|favorite|"
    r"usually|always|never|my name|call me)\b", re.I)

_ABOUT_FILES = re.compile(
    r"\b(file|files|folder|directory|document|pdf|docx|csv|xlsx|spreadsheet|script|"
    r"code|module|function|class|repo|repository|project|notebook|save|open|read|"
    r"write|edit|\w+\.(?:py|js|ts|md|txt|csv|json|yaml|yml|pdf|docx|xlsx))\b", re.I)

_ABOUT_WORK_IN_FLIGHT = re.compile(
    r"\b(task|tasks|running|progress|status|how far|finished|done yet|still|"
    r"waiting|approve|approval|cancel|stop it|pause)\b", re.I)

_ABOUT_TIME = re.compile(
    r"\b(today|tomorrow|tonight|yesterday|this week|next week|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|calendar|meeting|appointment|"
    r"schedule|diary|free|busy|available)\b", re.I)

_ABOUT_PEOPLE = re.compile(
    r"\b(contact|contacts|email|e-mail|call|message|invite|send .{0,20}to|lead|leads|"
    r"customer|client|supplier|colleague)\b", re.I)

_ABOUT_THE_WEB = re.compile(
    r"\b(search|google|look up|latest|news|current|today's|price of|who won|"
    r"what happened|website|url|http)\b", re.I)

_ABOUT_THE_SCREEN = re.compile(
    r"\b(screen|window|click|button|type into|scroll|drag|what am i looking at|"
    r"this app|on my desktop)\b", re.I)


def refers_backwards(request: Request) -> bool:
    """Does this request depend on what came before it?

    Three independent signals, any one of which is enough, because the failure
    mode here is asymmetric: keeping history that was not needed costs tokens,
    and dropping history that was needed costs the conversation.
    """
    if request.intent.refers_back:
        return True
    if _BACKWARD.search(request.text):
        return True
    # A very short request is almost always a continuation: "yes", "the second
    # one", "do it", "why?".
    return len(request.text.split()) <= 4


# --------------------------------------------------------------------------- #
# Retrieval
#
# Every fetcher returns text or None. None means "there was nothing", which is
# reported as a skip with its reason - a source that returned nothing and a source
# that was never asked look different in the report, on purpose. None of them
# raise: a context source that can fail a turn is worse than a missing one.
# --------------------------------------------------------------------------- #

def _personality(request: Request) -> Optional[str]:
    from tools.personality import describe_personality

    return describe_personality() or None


def _mode_note(request: Request) -> Optional[str]:
    from core import modes

    return modes.system_note() or None


def _project(request: Request) -> Optional[str]:
    """The open project's instructions, and its files only when files are in play.

    The instructions are how the user told Leti to work in this project and are
    short; the file list is neither. A weather question inside an open project
    used to carry forty filenames.
    """
    from tools.projects import get_active_project, project_context

    active = get_active_project()
    if not active:
        return None
    wants_files = bool(_ABOUT_FILES.search(request.text)) or request.has_capability(
        intent_reader.FILE_TASK, intent_reader.RESEARCH)
    return project_context(active, max_files=40 if wants_files else 0,
                           request=request.text) or None


def _user_profile(request: Request) -> Optional[str]:
    from tools.user_profile import profile_summary

    text = profile_summary()
    return f"What you know about the user so far:\n{text}" if text else None


def _continuity(request: Request, referents: List[str]) -> Optional[str]:
    return intent_reader.continuity_note(request.intent, referents) or None


def _objective(request: Request) -> Optional[str]:
    return intent_reader.system_note(request.intent) or None


def _proactive(request: Request) -> Optional[str]:
    from core import proactive

    pending = proactive.items()
    note = proactive.turn_note(pending)
    if note:
        proactive.mark_raised(pending)
    return note or None


def _objectives(request: Request) -> Optional[str]:
    """What Leti is still carrying, when the request sounds like it is about that.

    core/objectives.py composes this from the task store and the goal store - the
    ones that already exist - in the GOAL -> ... -> NEXT ACTION shape, and says
    nothing at all when nothing is in flight, which is nearly always.
    """
    from core import objectives

    return objectives.turn_note() or None


def _missing_answer(request: Request) -> Optional[str]:
    if not request.intent.question_to_ask:
        return None
    return ("Something needed to answer this is missing and guessing it would change "
            f"the result. Ask: \"{request.intent.question_to_ask}\" - unless the "
            "conversation above already answers it, in which case carry on.")


def _connection_warnings(request: Request) -> Optional[str]:
    """Which accounts this request will need and has not got.

    Reads core/connections.py, which reads settings. No network call, no
    credential, and nothing at all to say when everything the request needs is
    set up - which is the common case and costs one dictionary lookup per
    capability.
    """
    from core import connections

    wanted: List[str] = []
    if _ABOUT_PEOPLE.search(request.text) or request.has_capability(
            intent_reader.COMMUNICATION):
        wanted.append("email")
    if _ABOUT_TIME.search(request.text):
        wanted.append("calendar")
    if not wanted:
        return None
    missing = []
    for capability in dict.fromkeys(wanted):
        state = connections.status(capability)
        if state["state"] == connections.NOT_CONFIGURED:
            missing.append(f"- {capability}: {state['detail']} {state.get('fix', '')}".strip())
        elif state["state"] == connections.FAILING:
            missing.append(f"- {capability}: {state['detail']}")
    if not missing:
        return None
    return ("Before trying, know that these are not available:\n" + "\n".join(missing)
            + "\nSay so plainly rather than attempting it and reporting a failure, and "
              "never report missing access as an empty result.")


def _watch_catchup(request: Request) -> Optional[str]:
    if not request.first_turn_of_session:
        return None
    return ("This is the start of a new session. If any social media watches are "
            "configured, call check_social_watches now, before anything else, to catch "
            "the user up on anything missed since last time - regardless of how long "
            "it's been. If it reports nothing new (or no watches exist), just continue "
            "with the user's actual request without mentioning the check.")


# name -> (when to ask it, how to ask it, priority). Priority is what gets
# dropped first if the package does not fit: higher number, dropped sooner.
# Nothing with priority 1 is ever dropped.
SOURCES: List[Tuple[str, Callable[[Request], bool], Callable[[Request], Optional[str]], int]] = [
    ("mode", lambda r: True, _mode_note, 1),
    ("personality", lambda r: True, _personality, 3),
    ("connections", lambda r: True, _connection_warnings, 2),
    ("missing information", lambda r: bool(r.intent.question_to_ask), _missing_answer, 1),
    ("objective", lambda r: r.intent.is_complex, _objective, 2),
    ("project memory",
     lambda r: True, _project, 4),
    ("what Leti knows about you",
     lambda r: bool(_ABOUT_THE_USER.search(r.text)) or r.intent.shape == intent_reader.CHAT,
     _user_profile, 4),
    ("objectives in flight",
     lambda r: bool(_ABOUT_WORK_IN_FLIGHT.search(r.text)) or r.intent.is_complex,
     _objectives, 3),
    ("proactive", lambda r: r.allow_optional, _proactive, 5),
    ("session start", lambda r: r.first_turn_of_session, _watch_catchup, 2),
]

# Sources that exist but live somewhere else. Never fetched here; named in the
# package so a request that needs one is not silently answered without it.
EXTERNAL_SOURCES: List[Tuple[str, Callable[[Request], bool], str]] = [
    ("calendar", lambda r: bool(_ABOUT_TIME.search(r.text)),
     "a calendar tool - do not answer from memory, and an unreadable calendar is "
     "not an empty one"),
    ("the web", lambda r: bool(_ABOUT_THE_WEB.search(r.text)),
     "web_search or browser_read_page"),
    ("the screen", lambda r: bool(_ABOUT_THE_SCREEN.search(r.text)),
     "read_screen"),
    ("contacts and CRM", lambda r: bool(_ABOUT_PEOPLE.search(r.text)),
     "resolve_contact, and list_business_data for records"),
]


# --------------------------------------------------------------------------- #
# Building the package
# --------------------------------------------------------------------------- #

async def build(request: Request, system_prompt: str,
                history: List[Dict[str, Any]],
                referents: Optional[List[str]] = None,
                recall: Optional[Callable[[str], Awaitable[List[Dict[str, Any]]]]] = None,
                budget_chars: int = BUDGET_CHARS) -> Package:
    """Assemble the context for one turn.

    Never raises. Every individual source is guarded, and the worst case is a
    package holding the system prompt and the conversation - which is exactly what
    Leti had before this module existed.
    """
    started = time.perf_counter()
    package = Package()
    pieces: List[Piece] = []

    for name, wanted, fetch, priority in SOURCES:
        try:
            if not wanted(request):
                package.skipped.append((name, "this request does not need it"))
                continue
            text = fetch(request)
        except Exception as e:
            package.skipped.append((name, f"could not be read ({e})"))
            logger.debug(f"Context source '{name}' failed: {e}")
            continue
        if not text:
            package.skipped.append((name, "nothing to say"))
            continue
        pieces.append(Piece(source=name, text=text, priority=priority))

    # What the request points back at, from the list Leti itself produced. Only
    # when it points back at all: a fresh request has no antecedent and a note
    # about one would be an invitation to invent it.
    backwards = refers_backwards(request)
    if backwards:
        try:
            note = _continuity(request, list(referents or []))
            if note:
                pieces.append(Piece(source="what 'it' refers to", text=note, priority=1))
        except Exception as e:
            package.skipped.append(("what 'it' refers to", f"could not be read ({e})"))
    else:
        package.skipped.append(("what 'it' refers to", "this request stands on its own"))

    # Long-term recall is a vector search over memory/vector_store.py - the store
    # that already exists, searched when a request could plausibly be about
    # something remembered, and skipped when it plainly is not. It is also the
    # first thing core/performance.py drops when the machine is under pressure.
    if recall is not None and request.allow_optional and _worth_recalling(request):
        try:
            remembered = await recall(request.text)
            if remembered:
                pieces.append(Piece(
                    source="long-term memory",
                    text="Relevant things you remember about the user:\n"
                         + "\n".join(f"- {m['text']}" for m in remembered),
                    priority=4))
            else:
                package.skipped.append(("long-term memory", "nothing relevant remembered"))
        except Exception as e:
            package.skipped.append(("long-term memory", f"search failed ({e})"))
    else:
        package.skipped.append((
            "long-term memory",
            "not worth a search for this request" if request.allow_optional
            else "skipped: the machine is under load"))

    for name, wanted, how in EXTERNAL_SOURCES:
        try:
            if wanted(request):
                package.needs_tools.append({"source": name, "reach_it_with": how})
        except Exception:
            continue

    # RANK and PACKAGE. Trimming is worst-priority-first and each drop is named,
    # because context that vanished silently is context nobody can debug.
    history = list(history or [])
    kept_history = history if backwards else history[-max(MIN_HISTORY_MESSAGES,
                                                          len(history) // 2):]
    if len(kept_history) < len(history):
        package.skipped.append((
            "older conversation",
            f"{len(history) - len(kept_history)} earlier message(s) left out - this "
            "request does not refer back"))

    fixed = len(system_prompt) + sum(len(m.get("content") or "") for m in kept_history)
    pieces.sort(key=lambda p: (p.priority, -p.chars))
    room = max(0, budget_chars - fixed)
    used, kept = 0, []
    for piece in pieces:
        if piece.priority > 1 and used + piece.chars > room:
            package.dropped.append(f"{piece.source} ({piece.chars} chars)")
            continue
        kept.append(piece)
        used += piece.chars

    package.messages = [{"role": "system", "content": system_prompt}]
    package.messages.extend({"role": p.role, "content": p.text} for p in kept)
    package.messages.extend(kept_history)
    package.included = [p.source for p in kept]
    package.chars = fixed + used
    package.seconds = time.perf_counter() - started
    return package


_WORTH_RECALLING = re.compile(
    r"\b(i|me|my|mine|we|our|remember|remind|prefer|preference|usually|favourite|"
    r"favorite|last time|before|again|my name|who am i|what do you know|told you|"
    r"like|hate|birthday|anniversary|goal|plan)\b", re.I)


def _worth_recalling(request: Request) -> bool:
    """Is a vector search over what Leti remembers likely to help?

    Long-term memory holds durable facts about the person: names, preferences,
    goals, things they said once and expect to be carried. A request with nothing
    personal in it - "convert 40 psi to bar", "what does this error mean" - is not
    going to be improved by one, and the search is the most expensive thing in
    this module.
    """
    if request.intent.shape == intent_reader.CHAT:
        return True
    return bool(_WORTH_RECALLING.search(request.text))
