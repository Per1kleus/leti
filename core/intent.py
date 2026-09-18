"""What the user is actually asking for, worked out before anything is sent.

Leti already had two pieces of this. core/tool_router.py reads a request to decide
which tools to show, and core/intent_signals.py reads it for "the user already
said yes". Neither answers the question this one does: what SHAPE is this request,
and does answering it well need more than one turn?

The answer costs no model call. That is the design constraint, not an accident: a
classifier model in front of every turn would double the latency of "what time is
it" to make "research five laptops and put them in a file" go slightly better.
Everything here is regex and word lists over the text the user typed, and it runs
in tens of microseconds.

What it produces is used three ways, all of them cheap:

  The router is told how simple the request is, so a one-line question is not
  shown thirty tools (core/tool_router.py, budget=).

  The orchestrator adds ONE short system line for a genuinely complex request,
  naming the stages the request implies and reminding the model to finish with a
  check. A simple request gets nothing added at all, which is the point.

  A request that refers to something earlier ("compare the first three") is
  recognised as doing so, and a request whose ambiguity would change the answer
  gets a question rather than a guess.

None of it decides anything on its own. It cannot run a tool, it cannot grant
permission, and every judgement it makes is a hint to a model that is still free
to do something else.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

# --------------------------------------------------------------------------- #
# The shapes a request comes in
# --------------------------------------------------------------------------- #
CHAT = "chat"                     # a greeting; no work implied
QUESTION = "question"             # answerable from what the model knows
RETRIEVAL = "retrieval"           # look something up and report it
FILE_TASK = "file_task"
COMMUNICATION = "communication"
COMPUTER_USE = "computer_use"
MONITORING = "monitoring"
RESEARCH = "research"
MULTI_STEP = "multi_step"

SIMPLE, NORMAL, COMPLEX = "simple", "normal", "complex"

# Words that put a request in a shape. Deliberately small and concrete: a long
# list of near-synonyms makes everything match everything, which is how a
# classifier stops classifying.
_SIGNALS = {
    RESEARCH: r"\b(research|investigate|find out|look into|compare|comparison|best|"
              r"cheapest|options|alternatives|review|survey|shortlist|recommend)\b",
    FILE_TASK: r"\b(file|files|document|documents|pdf|docx|xlsx|csv|spreadsheet|report|"
               r"folder|directory|save|write it to|put .{0,20}(in|into) a|read)\b",
    COMMUNICATION: r"\b(email|e-mail|mail|message|send|reply|forward|invite|meeting|"
                   r"schedule a call|contact)\b",
    COMPUTER_USE: r"\b(click|type into|screen|window|open the app|launch|button|"
                  r"settings menu|drag|scroll)\b",
    MONITORING: r"\b(watch|monitor|notify me|tell me when|alert me|keep an eye|"
                r"let me know when)\b",
    RETRIEVAL: r"\b(what is|what's|who is|when is|where is|how much|price of|weather|"
               r"show me|list|status)\b",
}

# Most specific first. A request that is several of these is named by the
# narrowest one that fits: "email the report" is a communication task that
# happens to involve a report, not a file task that happens to mention mail.
_SHAPE_PRECEDENCE = (MONITORING, COMPUTER_USE, COMMUNICATION, FILE_TASK,
                     RESEARCH, RETRIEVAL)

# A request that implies work happening somewhere other than this machine. Used
# only to say "this will probably need your approval", never to grant it.
_EXTERNAL = re.compile(
    r"\b(send|email|e-mail|post|publish|deploy|delete|remove|buy|sell|order|pay|"
    r"transfer|sign up|subscribe|reply to|message)\b", re.I)

# Sequencing words: the difference between one job and several. "and then" is the
# clearest, but a bare "and" between two verbs counts too.
_STAGE_SPLIT = re.compile(
    r"\b(?:and then|then|after that|afterwards|followed by|next,|,\s*then|"
    r"and (?=\w+\s+(?:them|it|the|a|into|to)\b))", re.I)

# The stages a verb implies, in the order they have to happen. This is what turns
# "find the five best laptops, compare them and put the results in a file" into
# research -> compare -> write -> verify without asking anybody.
_STAGE_VERBS = [
    ("research", r"\b(research|find|search|look up|look for|investigate|gather|collect)\b"),
    ("compare", r"\b(compare|comparison|rank|shortlist|pick the|choose between|"
                r"evaluate|weigh up|which (is|one)|side by side)\b"),
    ("compute", r"\b(calculate|compute|analyse|analyze|plot|chart|model|simulate)\b"),
    ("write", r"\b(write|create|produce|generate|save|export|put .{0,20}(in|into)|"
              r"make a (file|report|document|spreadsheet))\b"),
    ("send", r"\b(send|email|e-mail|message|post|share with)\b"),
]

# References back into the conversation. These are the reason "compare the first
# three" is answerable at all, and the reason a wrong guess at one is expensive.
# Anaphora that is always a reference back: there is no reading of "compare
# those" that is about something the user has not already mentioned.
_ANAPHORA = re.compile(
    r"\b(those|these|them|that one|the winner|the same|the other one|"
    r"(number|option|item) (one|two|three|\d+))\b", re.I)

# "the first three", "the top two" - a reference only when nothing follows it.
# "the best laptops for MATLAB" names what it wants; "compare the best three"
# points at a list that already exists.
_ORDINAL_REFERENCE = re.compile(
    r"\b(the|its)\s+(first|second|third|last|top|best|cheapest|other)"
    r"(\s+(one|two|three|four|five|few|ones|options?|results?))?\s*(?=[.,;!?]|$)", re.I)

_BARE_PRONOUN = re.compile(r"^\W*(do|put|send|save|open|compare|use|run)\s+(it|that|this|them)\b", re.I)

# Things that make a request ambiguous in a way that CHANGES the answer, each with
# the question to ask instead of guessing. A request missing something harmless -
# no filename for a file Leti is about to invent - is not in here.
#
# The second element is what counts as the answer being already known, and it is a
# function rather than a pattern for a reason. It used to be "any two-to-five
# letter capitalised word" for the symbol case, which meant a conversation
# containing PDF, EUR, OK or USB silently suppressed "which symbol do you mean?".
# An antecedent has to actually look like the thing, in context.

_CRYPTO = re.compile(r"\b(bitcoin|btc|ethereum|eth|solana|doge)\b", re.I)
_MARKET_WORD = re.compile(r"\b(stock|stocks|share|shares|ticker|symbol|price|market|"
                          r"trading|traded|nasdaq|nyse|crypto|coin)\b", re.I)
_PAIR = re.compile(r"\b[A-Z]{2,5}/[A-Z]{3,4}\b")
_UPPER_TOKEN = re.compile(r"\b[A-Z]{2,5}\b")

# Capitalised words that are never a ticker. Without this, a conversation that
# said PDF, EUR, OK or USB looked like one that had already named a symbol, and
# "watch that stock" stopped asking which. Short and concrete on purpose: the
# cost of a wrong entry is one unnecessary question, and the cost of leaving the
# list out entirely was no question at all.
_NOT_TICKERS = frozenset({
    "PDF", "CSV", "TSV", "XLS", "DOC", "TXT", "JSON", "XML", "HTML", "HTTP", "URL",
    "API", "SQL", "USB", "SSD", "HDD", "RAM", "CPU", "GPU", "VRAM", "OS", "PC", "TV",
    "AI", "ID", "UI", "IP", "OK", "AM", "PM", "FAQ", "PIN", "GB", "MB", "KB", "TB",
    "EUR", "USD", "GBP", "JPY", "CHF", "UK", "US", "EU", "USA", "CEO", "CTO", "HR",
    "PR", "QA", "VPN", "DNS", "SSH", "FTP", "CLI", "GUI", "IDE", "RSS", "PNG", "JPG",
    "SVG", "MP3", "MP4", "ZIP", "OCR", "STT", "TTS", "LLM", "MATLAB", "IMAP", "SMTP",
})


def _names_a_symbol(text: str) -> bool:
    """Whether this text actually names a tradeable thing.

    A capitalised acronym on its own is not one - "read the PDF", "under 1100 EUR"
    and "the USB drive" all contain one, and treating any of them as a symbol
    suppressed the question that makes "watch that stock" answerable. A pair
    (BTC/USD) or a coin by name always counts; a bare acronym counts when it is
    not one of the common ones, or when a market word is in the same breath.
    """
    if _PAIR.search(text) or _CRYPTO.search(text):
        return True
    tokens = [tok for tok in _UPPER_TOKEN.findall(text) if tok not in _NOT_TICKERS]
    if tokens:
        return True
    return bool(_UPPER_TOKEN.search(text) and _MARKET_WORD.search(text))


def _names_a_file(text: str) -> bool:
    """A filename or a path. "the project" is where a file might be, not which one."""
    return bool(_FILENAME.search(text))


def _names_a_person(text: str) -> bool:
    return bool(_ADDRESSEE.search(text))


_FILENAME = re.compile(r"[\w~][\w/\\.-]*\.(?:pdf|docx|xlsx|csv|txt|md|json|py|xls|pptx)\b", re.I)
_ADDRESSEE = re.compile(r"\S+@\S+\.\w+|\bcontact\b", re.I)

_MISSING = [
    (re.compile(r"\b(that|this|the) (stock|share|ticker|symbol)\b", re.I),
     _names_a_symbol, "Which symbol do you mean?"),
    (re.compile(r"\b(that|this|the) (file|document|report|spreadsheet)\b", re.I),
     _names_a_file, "Which file do you mean?"),
    (re.compile(r"\bemail\s+(him|her|them)\b", re.I),
     _names_a_person, "Who should Leti email?"),
]

_QUANTITY = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)
_MONEY = re.compile(r"(?:[€$£]\s?\d[\d.,]*|\d[\d.,]*\s?(?:eur|usd|gbp|euros?|dollars?|pounds?))", re.I)
_LIMIT = re.compile(r"\b(under|below|less than|at most|no more than|over|above|"
                    r"at least|within|before|by)\b\s*\S+", re.I)
_PATH = re.compile(r"[\w~][\w/\\.-]*\.(?:pdf|docx|xlsx|csv|txt|md|json|py|xls|pptx)\b", re.I)
_URL = re.compile(r"https?://\S+", re.I)
_TICKER = re.compile(r"\b[A-Z]{2,5}\b")
_QUOTED = re.compile(r"[\"'“]([^\"'”]{2,60})[\"'”]")

MAX_ENTITIES = 8


@dataclass
class Intent:
    """A reading of one request. Every field is a hint, never an instruction."""

    shape: str = QUESTION
    complexity: str = SIMPLE
    capabilities: List[str] = field(default_factory=list)
    stages: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output: str = "answer"
    needs_confirmation: bool = False
    refers_back: bool = False
    question_to_ask: str = ""

    @property
    def is_complex(self) -> bool:
        return self.complexity == COMPLEX

    def as_objective(self) -> Dict[str, Any]:
        """The structured objective, for a task or for the system note below."""
        return {"goal": self.shape, "stages": self.stages, "entities": self.entities,
                "constraints": self.constraints, "output": self.output,
                "needs_confirmation": self.needs_confirmation}


def read(text: str, history: Optional[Sequence[Dict[str, Any]]] = None) -> Intent:
    """Read one request. Never raises: an unreadable request is a simple one."""
    try:
        return _read(text or "", history or ())
    except Exception:            # a hint that fails is not a turn that fails
        return Intent()


def _read(text: str, history: Sequence[Dict[str, Any]]) -> Intent:
    lowered = text.lower()
    intent = Intent()

    from core.tool_router import _PURE_CHAT     # one definition of "just chat"

    if _PURE_CHAT.match(text.strip()):
        return Intent(shape=CHAT, complexity=SIMPLE)

    matched = {name for name, pattern in _SIGNALS.items()
               if re.search(pattern, lowered, re.I)}
    shapes = [name for name in _SHAPE_PRECEDENCE if name in matched]
    intent.capabilities = sorted(matched)

    intent.stages = [stage for stage, pattern in _STAGE_VERBS
                     if re.search(pattern, lowered, re.I)]
    clauses = [c for c in _STAGE_SPLIT.split(text) if c and c.strip()]

    # Complexity is about how many DIFFERENT things have to happen, not about how
    # long the sentence is. Two stages, or an explicit "and then", is the line.
    distinct = len(intent.stages)
    if distinct >= 2 or len(clauses) > 2:
        intent.complexity = COMPLEX
        intent.shape = MULTI_STEP
    elif distinct == 1 or shapes:
        intent.complexity = NORMAL
        intent.shape = shapes[0] if shapes else QUESTION
    else:
        intent.complexity = SIMPLE
        intent.shape = QUESTION

    # A short question with no stage in it stays simple even if it mentions a
    # capability: "what's the price of AAPL" is one lookup.
    if intent.complexity == NORMAL and not intent.stages and len(text.split()) <= 8:
        intent.complexity = SIMPLE

    if RESEARCH in shapes and intent.complexity == SIMPLE:
        intent.complexity = NORMAL
    if MONITORING in shapes:
        intent.shape = MONITORING
        intent.output = "watch"
    if "write" in intent.stages or re.search(_SIGNALS[FILE_TASK], lowered, re.I):
        intent.output = "file" if "write" in intent.stages else intent.output
    if "send" in intent.stages:
        intent.output = "message"

    intent.needs_confirmation = bool(_EXTERNAL.search(text))
    intent.entities = _entities(text)
    intent.constraints = _constraints(text)
    # A request that goes and finds the things it then talks about supplies its
    # own antecedent: "find five laptops and compare them" is not pointing at an
    # earlier turn, and treating it as though it were would put a question in
    # front of a request that could not be clearer.
    self_contained = "research" in intent.stages
    intent.refers_back = bool(
        (not self_contained and (_ANAPHORA.search(text) or _BARE_PRONOUN.match(text)))
        or _ORDINAL_REFERENCE.search(text))
    intent.question_to_ask = _what_is_missing(text, history)
    return intent


def _entities(text: str) -> List[str]:
    """The concrete things named in the request - paths, links, quoted names."""
    found: List[str] = []
    for pattern in (_URL, _PATH, _QUOTED, _MONEY):
        for match in pattern.findall(text):
            value = match if isinstance(match, str) else match[0]
            if value and value not in found:
                found.append(value.strip())
    for ticker in _TICKER.findall(text):
        # An acronym in a sentence about markets, not every capitalised word.
        if ticker not in found and re.search(r"\b(stock|share|ticker|price|market|watch)\b",
                                             text, re.I):
            found.append(ticker)
    return found[:MAX_ENTITIES]


def _constraints(text: str) -> List[str]:
    out = []
    for match in _LIMIT.finditer(text):
        out.append(match.group(0).strip())
    quantity = _QUANTITY.search(text)
    if quantity and re.search(r"\b(best|top|first|find|list|compare)\b", text, re.I):
        out.append(f"how many: {quantity.group(0)}")
    return out[:4]


def _what_is_missing(text: str, history: Sequence[Dict[str, Any]]) -> str:
    """The one question worth asking, or nothing.

    Only asks when the missing piece would change the answer AND nothing in the
    recent conversation supplies it - "watch that stock" after two turns about
    TTWO is not ambiguous, and asking anyway is its own kind of unhelpful.
    """
    recent = " ".join(str(m.get("content", "")) for m in list(history)[-6:])
    for trigger, already_named, question in _MISSING:
        if trigger.search(text) and not already_named(text) and not already_named(recent):
            return question
    return ""


# --------------------------------------------------------------------------- #
# What the model is told about it
# --------------------------------------------------------------------------- #

def system_note(intent: Intent) -> str:
    """One short line for a complex request. Empty for everything else.

    Deliberately small. This is a nudge towards finishing the job and checking the
    result, not a plan - the model writes the plan, as it already does when it
    calls start_autonomous_task.
    """
    if not intent.is_complex or not intent.stages:
        return ""
    stages = " -> ".join(intent.stages + ["verify"])
    lines = [f"This request has several parts: {stages}. Work through them in order "
             "and do not report it finished until you have checked the result - a file "
             "that exists is not the same as a file with the right contents in it."]
    if intent.constraints:
        lines.append("Constraints the user gave: " + "; ".join(intent.constraints) + ".")
    if intent.needs_confirmation:
        lines.append("Part of this reaches outside this machine, so it will stop for "
                     "the user's confirmation. That is expected; do not work around it.")
    lines.append("If it is genuinely several turns of work, start_autonomous_task with "
                 "the steps and the success criteria is the right shape for it.")
    return " ".join(lines)


def continuity_note(intent: Intent, referents: Sequence[str]) -> str:
    """What "the first three" refers to, when there is something to point at."""
    if not intent.refers_back:
        return ""
    if referents:
        listed = "; ".join(f"{i + 1}. {r}" for i, r in enumerate(referents))
        return ("The user is referring back to what you listed a moment ago, in this "
                f"order: {listed}. Use that ordering rather than re-deriving it.")
    return ("The user's request refers to something from earlier in this conversation. "
            "If you cannot tell from the messages above exactly what they mean, ask "
            "which one rather than picking for them.")
