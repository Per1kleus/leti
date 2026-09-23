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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# Entering a mode is a COMMAND, not a request that needs judgement - and the rule
# that matters most about it is that nothing else may trigger it. "Check my
# customer emails" is a business task and must stay in Default Mode; "enter
# business mode" is the only kind of thing that changes modes.
#
# So it is matched here, deterministically, rather than left to the model to
# notice. A regex cannot be talked into switching by a sentence full of invoices
# and pipelines, which is exactly the failure this has to be immune to. It also
# costs nothing: no schema, no model call, and it works identically by voice.
_MODE_WORD = r"(?P<mode>coding|business|dev|developer|software|default|normal|assistant)"
_MODE_COMMANDS = (
    # "enter coding mode", "switch to business mode", "let's work in business mode"
    re.compile(rf"\b(?:enter|activate|start|go(?:\s+in)?to|switch\s+(?:to|into)|"
               rf"turn\s+on|use|work\s+in|let'?s\s+work\s+in|put\s+(?:you|yourself)\s+in)"
               rf"\s+(?:the\s+)?{_MODE_WORD}\s+mode\b", re.I),
    # "open my business workspace"
    re.compile(rf"\b(?:open|start|enter)\s+(?:my\s+|the\s+)?{_MODE_WORD}\s+workspace\b", re.I),
    # "coding mode on"
    re.compile(rf"\b{_MODE_WORD}\s+mode\s+(?:on|please)\b", re.I),
)
_MODE_EXIT = re.compile(
    r"\b(?:exit|leave|quit|stop|end|turn\s+off|close)\s+(?:the\s+)?"
    r"(?:coding|business|dev|developer|software|this|that)?\s*mode\b|"
    r"\b(?:back\s+to|return\s+to)\s+(?:the\s+)?(?:default|normal)(?:\s+mode)?\b", re.I)

_MODE_ALIASES = {"coding": "coding", "dev": "coding", "developer": "coding",
                 "software": "coding", "business": "business",
                 "default": "default", "normal": "default", "assistant": "default"}


_MODE_QUESTION = re.compile(
    r"\b(?:what|which)\s+mode\b|\bwhat\s+mode\s+are\s+you\s+in\b|"
    r"\b(?:are\s+you|am\s+i)\s+in\s+\w+\s+mode\b", re.I)


def asks_which_mode(text: str) -> bool:
    """A question about the current mode, as opposed to a command to change it."""
    return bool(isinstance(text, str) and "mode" in text.lower()
                and _MODE_QUESTION.search(text))


# Exactly these, alone, and nothing that merely starts with them: "exit the
# browser" is not a mode command.
_BARE_EXIT = re.compile(r"^(?:exit|leave|go back|back to normal)[.!]?$", re.I)
DEFAULT_MODE_NAME = "default"


def mode_command(text: str) -> Optional[str]:
    """The mode this request explicitly asks for, or None.

    None is the answer for every ordinary request, including every business or
    coding TASK. Doing business work is not asking for Business Mode, and this is
    the function that keeps those two apart.
    """
    # A cheap guard so the ordinary request costs one substring check, not four
    # regexes. "back to default" carries none of the obvious words, so it is in
    # here too rather than being silently unmatchable.
    if not isinstance(text, str):
        return None
    lowered = text.lower()

    # A bare "exit" while a specialised mode is on. The mode description tells the
    # user to say exactly that, so it has to work - and it is scoped to a
    # specialised mode on purpose: in Default Mode there is nothing to exit, and a
    # one-word command that does something in one state and nothing in another is
    # better than one that might mean "stop what you are doing".
    if _BARE_EXIT.match(lowered.strip()):
        try:
            from core import modes

            return DEFAULT_MODE_NAME if modes.current() != DEFAULT_MODE_NAME else None
        except Exception:
            return None

    if not any(hint in lowered for hint in ("mode", "workspace", "back to", "return to")):
        return None
    for pattern in _MODE_COMMANDS:
        match = pattern.search(text)
        if match:
            return _MODE_ALIASES.get(match.group("mode").lower())
    if _MODE_EXIT.search(text):
        return "default"
    return None


# "Run full Leti diagnostics" is a command, like a mode command, and is answered
# the same way: matched deterministically here, before any model call, and handled
# by the orchestrator itself.
#
# It is deliberately NOT a tool. Every registered tool's schema sits in Default
# Mode's fallback, which is already close to the context window, and a diagnostics
# report is a fixed thing Leti knows how to produce - there is nothing for the
# model to decide. This way it costs zero tokens, works by voice, and cannot be
# triggered by a sentence that merely mentions health.
_DIAGNOSTICS = re.compile(
    r"\b(?:run|do|perform|start|give me)\s+(?:a\s+|the\s+|my\s+)?"
    r"(?:full\s+|complete\s+|whole\s+|deep\s+|thorough\s+|quick\s+)?(?:leti\s+)?"
    r"(?:diagnostics?|system\s+check|health\s+check|self[\s-]?check|self[\s-]?test)\b"
    r"|\b(?:leti\s+)?diagnostics\b\s*$"
    r"|\bcheck\s+(?:your|leti'?s)\s+(?:own\s+)?health\b", re.I)

# The same command, asking it to actually contact the model server. Costs seconds,
# so it happens only when asked for in these words.
_DIAGNOSTICS_DEEP = re.compile(
    r"\b(?:deep|thorough|including connections?|with connections?|contact|"
    r"actually (?:test|check|try))\b", re.I)


def asks_for_diagnostics(text: str) -> Optional[Dict[str, Any]]:
    """A request for a full system check, or None for everything else.

    None for every ordinary request, including one that complains something is
    broken: "my email isn't working" is a problem to look into, not a command to
    run sixteen checks.
    """
    if not isinstance(text, str):
        return None
    lowered = text.lower()
    if not any(hint in lowered for hint in
               ("diagnos", "system check", "health", "self-check", "self check",
                "self-test", "self test")):
        return None
    if not _DIAGNOSTICS.search(text):
        return None
    return {"reach_out": bool(_DIAGNOSTICS_DEEP.search(text))}


# --------------------------------------------------------------------------- #
# Asking to see, or to stop seeing
#
# Leti speaks its answers and does not print them (see core/transcript.py). That
# only works if asking for the text is RELIABLE, so this reads it the way mode
# commands and lifecycle commands are read: deterministically, here, with no
# model round trip. A request the model has to agree is a request is a request
# that sometimes gets an essay instead of the text.
#
# Deliberately narrow. These fire only on a short sentence that is entirely
# about showing or hiding, so "show me how to write a for loop" is a question
# about for loops and "can you show the file contents" is a request for a file.
# --------------------------------------------------------------------------- #

SHOW_TEXT = "show_text"
HIDE_TEXT = "hide_text"
SHOW_MATH = "show_math"
SHOW_LAST_VISUAL = "show_last_visual"

TRANSCRIPT_ACTIONS = (SHOW_TEXT, HIDE_TEXT, SHOW_MATH, SHOW_LAST_VISUAL)

# What the thing being asked for is called. Split by what it refers to, because
# "show me the equation" and "show me the answer" want different panels.
_TEXT_NOUN = (r"(?:the\s+|that\s+|your\s+|my\s+)?"
              r"(?:text|transcript|answer|reply|response|words|explanation|"
              r"writing|wrote|said|everything you said|what you said|"
              r"full answer|whole answer|written answer)")
_MATH_NOUN = (r"(?:the\s+|that\s+|those\s+)?"
              r"(?:equation|equations|formula|formulas|formulae|math|maths|"
              r"mathematics|derivation|working|calculation)")
_VISUAL_NOUN = (r"(?:the\s+|that\s+|those\s+)?"
                r"(?:graph|graphs|chart|charts|plot|plots|image|images|picture|"
                r"pictures|photo|photos|diagram|diagrams|visual|visuals|figure)")

_SHOW = r"(?:show|display|open|reveal|print|bring up|put up|pull up)"
_HIDE = r"(?:hide|close|dismiss|clear|take down|put away|get rid of)"
_LEAD = r"^\W*(?:please\s+|leti,?\s+|just\s+|ok(?:ay)?,?\s+|can you\s+|could you\s+|"          r"would you\s+|i want to\s+|i\'d like to\s+|let me\s+)*"
_TAIL = r"(?:\s+(?:please|now|again|too|as well))?[\s,.!?]*$"

_TRANSCRIPT = (
    # Hiding first: "close the transcript" must never be read as showing it.
    (HIDE_TEXT, re.compile(_LEAD + _HIDE + r"\s+" + _TEXT_NOUN + _TAIL, re.I)),
    (HIDE_TEXT, re.compile(_LEAD + r"(?:stop showing|don\'t show)\s+" + _TEXT_NOUN + _TAIL, re.I)),
    (SHOW_MATH, re.compile(_LEAD + _SHOW + r"\s+(?:me\s+|us\s+)?(?:only\s+)?" + _MATH_NOUN + _TAIL, re.I)),
    (SHOW_LAST_VISUAL, re.compile(_LEAD + _SHOW + r"\s+(?:me\s+|us\s+)?(?:only\s+)?" + _VISUAL_NOUN + _TAIL, re.I)),
    (SHOW_TEXT, re.compile(_LEAD + _SHOW + r"\s+(?:me\s+|us\s+)?(?:all\s+of\s+)?" + _TEXT_NOUN + _TAIL, re.I)),
    # "let me read that", "let me see what you said" - asking to read, not to be told.
    (SHOW_TEXT, re.compile(_LEAD + r"(?:read|see)\s+" + _TEXT_NOUN + _TAIL, re.I)),
    # "let me read that" on its own. Only after "let me"/"can i" and only for
    # READ: a bare "show me that" could mean anything on screen, but nobody
    # reads a graph.
    (SHOW_TEXT, re.compile(
        r"^\W*(?:please\s+|leti,?\s+)?(?:let me|can i|could i|i want to|i\'d like to)\s+"
        r"read\s+(?:that|it|this)" + _TAIL, re.I)),
)

# The longest a request may be and still be only about showing something. Past
# this there is a question in there too, and answering it is the ordinary turn.
TRANSCRIPT_MAX_CHARS = 60


def transcript_command(text: str) -> Optional[str]:
    """SHOW_TEXT / HIDE_TEXT / SHOW_MATH / SHOW_LAST_VISUAL, or None.

    None for every ordinary request, which is almost all of them. The caller
    answers a match from what it already has - see core/transcript.py - so a
    false positive costs a panel and a false negative costs a model call; the
    patterns above are tuned for the second, which is the cheaper mistake.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped or len(stripped) > TRANSCRIPT_MAX_CHARS:
        return None
    for action, pattern in _TRANSCRIPT:
        if pattern.match(stripped):
            return action
    return None


# --------------------------------------------------------------------------- #
# What KIND of thing the user just said
#
# `shape` above is about what a request is ABOUT - files, mail, the screen - and
# is what the router and the context engine read. This is the other question,
# which nothing used to ask: what is the user DOING by saying it. A remark, a
# question, an order, a correction, a cancellation.
#
# The two are orthogonal and both are kept. "Open the project" and "maybe we
# should open the project" have the same shape and want completely different
# treatment, and the difference is here rather than in the shape.
#
# Everything below is regex and word lists over the text, in tens of
# microseconds. There is no model call: a classifier in front of every message
# would double the cost of the cheapest turns to answer a question that the
# words themselves already settle.
# --------------------------------------------------------------------------- #

CONVERSATION = "conversation"              # a remark; nothing is being asked for
INFORMATION_REQUEST = "information_request"  # look something up and tell me
COMMAND = "command"                        # do this, now, one action
SINGLE_STEP_TASK = "single_step_task"      # one job, possibly a few tool calls
MULTI_STEP_TASK = "multi_step_task"        # several distinct jobs in order
WATCH_REQUEST = "watch_request"            # keep looking, tell me when
CORRECTION = "correction"                  # that was wrong; here is what I meant
CANCELLATION = "cancellation"              # stop, and do not continue
CONTINUATION = "continuation"              # pick up what was already happening
CLARIFICATION = "clarification"            # answering, or asking for, a question
# QUESTION is defined above and is a kind as well as a shape: "what is a turbine"
# is one thing to answer whichever way you look at it.

KINDS = (CONVERSATION, QUESTION, INFORMATION_REQUEST, COMMAND, SINGLE_STEP_TASK,
         MULTI_STEP_TASK, WATCH_REQUEST, CORRECTION, CANCELLATION, CONTINUATION,
         CLARIFICATION)

# Kinds that are ABOUT something already in flight rather than about new work.
# The orchestrator routes these to the task they refer to instead of to the model.
LIFECYCLE_KINDS = (CANCELLATION, CONTINUATION)

# Kinds where nothing should be done. Not "nothing needs doing" - nothing SHOULD
# be, because the user did not ask for anything to be.
NO_ACTION_KINDS = (CONVERSATION,)

# How sure a "this is only a remark" reading has to be before it is allowed to
# withhold tools. The recognised shapes - a greeting, a reaction, a hedge - come
# back at 0.85 and above; the fallback, which is only "nothing pointed
# anywhere", comes back at 0.5 and keeps its tools. See Intent.wants_action.
CONFIDENT_CONVERSATION = 0.8

# A remark. Not a request, not a question, not an instruction - a person
# reacting to what was just said. The failure this prevents is the expensive
# one: "that's interesting" becoming a web search.
#
# Anchored to the whole message on purpose. "Interesting - now open Spotify" is
# not a remark, and treating it as one loses the request.
_REACTION = re.compile(
    r"^\W*(?:"
    r"(?:that'?s|that is|this is|that was|it'?s|it is)\s+"
    r"(?:really\s+|very\s+|quite\s+|so\s+|pretty\s+)?"
    r"(?:interesting|cool|nice|neat|clever|useful|helpful|great|good|bad|odd|weird|"
    r"strange|annoying|frustrating|true|fair|right|wrong|surprising|impressive|"
    r"amazing|terrible|awful|funny|sad|unfortunate)"
    r"|(?:i\s+)?(?:see|understand|agree|know|thought so|get it|like that)"
    r"|makes sense|fair enough|good to know|got it|noted|understood|of course|"
    r"no worries|never mind then|"
    r"(?:wow|huh|oh|ah|hmm+|yikes|ouch|damn|nice one|well done|good job|"
    r"interesting|lovely|brilliant|exactly|indeed|true|agreed|same here)"
    r")"
    r"[\s,.!?\u2026]*$", re.I)

# Hedged suggestions. "Maybe we should open the project" names an action and
# asks for none: the hedge IS the request, and it is a request for an opinion.
# Acting on it is the model deciding the user was being coy, which they were not.
_HEDGE = re.compile(
    r"^\W*(?:"
    r"maybe|perhaps|possibly|probably|i wonder|i was wondering|i think we|"
    r"(?:we|you|i)\s+(?:could|might|may|should probably)|"
    r"it (?:might|may|could) be (?:worth|good|better|a good idea)|"
    r"(?:would|wouldn'?t) it be|"
    r"(?:do you think|what do you think|any thoughts|thoughts on)|"
    r"(?:i'?m|im) (?:thinking|considering|wondering)"
    r")\b", re.I)

# An explicit instruction beats a hedge in the same sentence: "maybe later, but
# open it now" is an order with a hedge in front of it.
_OVERRIDE_HEDGE = re.compile(
    r"\b(?:just do it|do it now|go ahead|please do|right now|now(?:,| )?\s*(?:open|run|"
    r"start|send|do)\b|actually,? (?:open|run|start|send|do)\b)", re.I)

# Stop. The highest-priority reading there is, and deliberately the narrowest -
# it takes precedence over a running task, so a false positive costs work.
_CANCELLATION = re.compile(
    r"^\W*(?:please\s+|leti,?\s+|now\s+|just\s+|ok(?:ay)?,?\s+)?"
    r"(?:"
    r"stop|halt|abort|cancel|quit|"
    r"(?:stop|cancel|abort|kill|end)\s+(?:it|that|this|them|everything|all of it|"
    r"the\s+[\w-]+(?:\s+[\w-]+)?|my\s+[\w-]+(?:\s+[\w-]+)?)|"
    r"never ?mind|nevermind|forget (?:it|that|about it)|drop it|leave it|"
    r"don'?t(?:\s+do)?\s+(?:that|it|bother)"
    r")"
    r"[\s,.!?]*$", re.I)

# Carry on with something that already exists.
_CONTINUATION = re.compile(
    r"^\W*(?:"
    r"continue|resume|carry on|keep going|go on|pick (?:it |that )?up|"
    r"(?:continue|resume|finish|carry on with) (?:the|that|my|it)\b.*|"
    r"where (?:were|was) (?:we|you|it)|what (?:were|was) (?:you|we) doing|"
    r"back to (?:that|it|the)\b.*"
    r")[\s,.!?]*$", re.I)

# "No, I meant the other one." A correction points at something already said and
# says it was read wrong. It is never new work on its own.
_CORRECTION = re.compile(
    r"^\W*(?:"
    r"no,?\s+(?:i (?:meant|said)|not|that'?s not|wrong)|"
    r"(?:i (?:meant|said))\b|"
    r"that'?s (?:not (?:what|right|it)|wrong)|"
    r"not (?:that|those|it|what i)\b|"
    r"wrong (?:one|file|person|thing|project)\b|"
    r"actually,?\s+i\b|"
    r"i didn'?t (?:mean|say|ask)"
    r")", re.I)

# Asking for, or giving, a clarification.
_CLARIFICATION = re.compile(
    r"^\W*(?:"
    r"what do you mean|which (?:one|of them|did you)|"
    r"(?:can you )?(?:be more specific|clarify|explain what you)|"
    r"i don'?t (?:understand|follow)|"
    r"(?:the|i mean(?:t)? the) (?:first|second|third|last|other) one"
    r")\b", re.I)

# An imperative aimed at Leti. The leading verb is the signal; a sentence that
# starts with one is an instruction in English, and one that does not, is not.
_IMPERATIVE = re.compile(
    r"^\W*(?:please\s+|now\s+|leti,?\s+|can you\s+|could you\s+|would you\s+|"
    r"i(?:'| a)?m going to need you to\s+|i need you to\s+|i want you to\s+)?"
    r"(?:go\s+)?(?:open|close|run|start|stop|launch|send|write|read|create|make|"
    r"delete|remove|move|copy|save|find|search|look|check|show|list|tell|give|"
    r"add|update|edit|fix|build|install|download|upload|click|type|press|set|"
    r"turn|enable|disable|schedule|book|call|email|message|post|publish|deploy|"
    r"analyse|analyze|compare|research|summarise|summarize|explain|draft|"
    r"generate|convert|rename|switch|pause|resume|retry|cancel|watch|monitor|"
    r"notify|remind|play|restart|"
    r"remember|forget|note|record|log|track|keep|"
    r"calculate|compute|count|measure|estimate|translate|"
    r"plan|review|test|verify|confirm|sort|filter|merge|split|clean|"
    r"print|describe|draw|plot|chart|sketch|clear|reset|sync|refresh|"
    r"try|use|pick|choose|select|apply|attach|extract|import|export)\b", re.I)

# An interrogative. A question mark is the clearest signal; a leading question
# word is nearly as good.
_INTERROGATIVE = re.compile(
    r"^\W*(?:what|who|when|where|why|how|which|is|are|was|were|do|does|did|can|"
    r"could|should|would|will|has|have|had|am)\b", re.I)

# Words that point at a piece of work already in flight.
_ABOUT_A_TASK = re.compile(
    r"\b(?:that task|the task|this task|the job|that job|it|the download|the build|"
    r"the research|the analysis|yesterday'?s|from yesterday|earlier|the last one|"
    r"the one (?:you|we) (?:were|was|started)|what (?:you|we) (?:were|was) doing)\b",
    re.I)


# Clause boundaries, for finding an instruction that arrives behind something
# else. Deliberately punctuation and the plainest sequencing words only.
_CLAUSE_SPLIT = re.compile(r"\s*(?:[-\u2013\u2014;.!?]+|,|\bthen\b|\band then\b)\s*")

# What makes an answer depend on the world rather than on knowledge. A question
# with one of these in it cannot be answered from the model alone.
_NEEDS_CURRENT_DATA = re.compile(
    r"\b(price|cost|worth|weather|status|now|today|tonight|tomorrow|current|currently|"
    r"latest|recent|how much|how many|show me|list|when is|when does|schedule|"
    r"calendar|stock|shares|news|running|open|installed|left|remaining|my\s+\w+)\b",
    re.I)


def _classify(text: str, intent: "Intent") -> Tuple[str, float]:
    """What kind of thing this is, and how sure that reading is.

    Order is the whole design. Cancellation comes first because "stop" has to
    outrank everything, including a sentence that also looks like an
    instruction. Conversation comes before the imperative test because "maybe we
    should open it" contains "open" and is not an order.
    """
    stripped = str(text or "").strip()
    if not stripped:
        return CONVERSATION, 1.0

    from core.tool_router import _PURE_CHAT

    # Clause by clause, for the same reason the imperative test is: "cool,
    # cancel the download" is a cancellation with a reaction in front of it.
    # Both patterns require a clause to be ENTIRELY the command, so "don't
    # cancel that" and "stop worrying about it" do not match either of them.
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(stripped) if c.strip()]
    if any(_CANCELLATION.match(c) for c in clauses):
        return CANCELLATION, 1.0
    if any(_CONTINUATION.match(c) for c in clauses):
        return CONTINUATION, 0.95
    if _PURE_CHAT.match(stripped) or _REACTION.match(stripped):
        return CONVERSATION, 1.0
    if _HEDGE.match(stripped) and not _OVERRIDE_HEDGE.search(stripped):
        # A hedged sentence that ends in a question mark is still a question -
        # "do you think we should open it?" wants an answer, not silence.
        return (QUESTION if stripped.endswith("?") else CONVERSATION), 0.85
    if _CORRECTION.match(stripped):
        return CORRECTION, 0.9
    if _CLARIFICATION.match(stripped):
        return CLARIFICATION, 0.85

    if intent.shape == MONITORING:
        return WATCH_REQUEST, 0.9

    # An instruction can arrive behind a remark: "interesting - now open
    # Spotify" is an order with a reaction in front of it, and reading the whole
    # string from position zero loses the order entirely. So each clause gets
    # the test, and one imperative clause makes the message an instruction.
    imperative = any(_IMPERATIVE.match(clause.strip())
                     for clause in _CLAUSE_SPLIT.split(stripped) if clause.strip())
    interrogative = stripped.endswith("?") or bool(_INTERROGATIVE.match(stripped))

    if intent.complexity == COMPLEX and len(intent.stages) >= 2:
        return MULTI_STEP_TASK, 0.9 if imperative else 0.7

    if imperative:
        # A one-action instruction is a command; anything with a stage verb in it
        # is a job that will take a few calls.
        if intent.stages or intent.shape in (RESEARCH, FILE_TASK, COMMUNICATION):
            return SINGLE_STEP_TASK, 0.85
        return COMMAND, 0.9

    if interrogative:
        # "What is the price of AAPL" needs a lookup; "what is a turbine" does
        # not. The difference is whether the answer depends on the state of the
        # world right now, which _NEEDS_CURRENT_DATA is the test for - the
        # RETRIEVAL shape is too broad for it, because "what is" puts every
        # question in that shape.
        if _NEEDS_CURRENT_DATA.search(stripped) or intent.shape == RESEARCH:
            return INFORMATION_REQUEST, 0.8
        return QUESTION, 0.85

    # Nothing pointed anywhere. A declarative sentence that named no action is a
    # remark; one that named an action without asking for it is also a remark,
    # and both are safer read that way than as work.
    return CONVERSATION, 0.5


# --------------------------------------------------------------------------- #
# Lifecycle commands
#
# "Stop.", "Pause.", "Resume.", "Retry.", "Skip this step.", "What are you
# waiting for?" - things said ABOUT work already running rather than requests
# for new work. Matched here, deterministically, before any model call, for the
# same reason mode commands are: a running task must not depend on a model
# agreeing that "stop" means stop.
#
# The target hint is whatever came after the verb. It is passed to the existing
# task resolution rather than being resolved here - which task is meant is a
# question core/task_manager.py and core/entities.py already answer.
# --------------------------------------------------------------------------- #

STOP = "stop"
PAUSE = "pause"
RESUME = "resume"
RETRY = "retry"
SKIP = "skip"
STATUS = "status"
WAITING = "waiting"

LIFECYCLE_ACTIONS = (STOP, PAUSE, RESUME, RETRY, SKIP, STATUS, WAITING)

_LIFECYCLE = (
    # Stop first, always. It outranks everything else that could be read out of
    # the same sentence, because the cost of a missed stop is an action nobody
    # wanted and the cost of a spurious one is a task that has to be resumed.
    (STOP, re.compile(
        r"^\W*(?:please\s+|leti,?\s+|just\s+|now\s+|ok(?:ay)?,?\s+)?"
        r"(?:stop|halt|abort|cancel|quit|kill)"
        r"(?:\s+(?:it|that|this|them|everything|all of it|all tasks|"
        r"the\s+[\w-]+(?:\s+[\w-]+)?|my\s+[\w-]+(?:\s+[\w-]+)?))?"
        r"[\s,.!?]*$", re.I)),
    (PAUSE, re.compile(
        r"^\W*(?:please\s+|leti,?\s+|just\s+)?"
        r"(?:pause|hold|wait|hang on|suspend)"
        r"(?:\s+(?:it|that|this|on|a moment|"
        r"the\s+[\w-]+(?:\s+[\w-]+)?))?"
        r"[\s,.!?]*$", re.I)),
    (RESUME, re.compile(
        r"^\W*(?:please\s+|leti,?\s+|ok(?:ay)?,?\s+|now\s+)?"
        r"(?:resume|continue|carry on|keep going|go on|unpause|pick up)"
        r"(?:\s+(?:it|that|this|with\s+\w+|"
        r"(?:the|my|that)\s+[\w-]+(?:\s+[\w-]+)?))?"
        r"[\s,.!?]*$", re.I)),
    (RETRY, re.compile(
        r"^\W*(?:please\s+|leti,?\s+|just\s+)?"
        r"(?:retry|try again|try it again|do it again|run it again|"
        r"retry\s+(?:it|that|this|the\s+[\w-]+(?:\s+[\w-]+)?))"
        r"[\s,.!?]*$", re.I)),
    (SKIP, re.compile(
        r"^\W*(?:please\s+|leti,?\s+|just\s+)?"
        r"(?:skip|skip (?:it|this|that|this step|the step|ahead)|"
        r"move on|next step|go to the next)"
        r"[\s,.!?]*$", re.I)),
    (WAITING, re.compile(
        r"^\W*(?:what|who|why)\s+(?:are|is)\s+(?:you|it|that)\s+"
        r"(?:waiting (?:for|on)|blocked (?:on|by)|stuck (?:on|at))"
        r"[\s,.!?]*$", re.I)),
    (STATUS, re.compile(
        r"^\W*(?:"
        r"what are you (?:doing|working on|up to)|"
        r"(?:show|list)(?: me)?(?: my| the)? (?:active |running |current )?tasks|"
        r"what(?:'s| is) running|what tasks|"
        r"which task(?:s)? (?:needs?|wants?|requires?) (?:me|my attention|input)|"
        r"what (?:needs|wants) (?:me|my attention)|"
        r"(?:任务)|"
        r"task status|status of (?:my |the )?tasks?"
        r")[\s,.!?]*$", re.I)),
)


def lifecycle_command(text: str) -> Optional[Dict[str, Any]]:
    """A command about work already in flight, or None.

    None for every ordinary request. The hint is the words after the verb, left
    exactly as the user said them so the existing task resolution can do what it
    already does with "the research task".
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return None
    for action, pattern in _LIFECYCLE:
        match = pattern.match(stripped)
        if match:
            hint = re.sub(
                r"^\W*(?:please|leti,?|just|now|ok(?:ay)?,?)?\s*"
                r"(?:stop|halt|abort|cancel|quit|kill|pause|hold|wait|hang on|suspend|"
                r"resume|continue|carry on|keep going|go on|unpause|pick up|retry|"
                r"try again|skip|move on)\s*",
                "", stripped, flags=re.I)
            hint = re.sub(r"^(?:it|that|this|them|the|my|with|on)\b\s*", "", hint,
                          flags=re.I).strip(" ,.!?")
            # status and waiting are about every task, not one of them, so a
            # hint scraped off the question would only mislead the resolver.
            if action in (STATUS, WAITING):
                hint = ""
            return {"action": action, "hint": hint,
                    "everything": bool(re.search(r"\b(everything|all of it|all tasks)\b",
                                                 stripped, re.I))}
    return None


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

    # The Intent Layer. `shape` says what the request is about; these say what
    # the user is doing by making it, and what may follow from that.
    kind: str = CONVERSATION
    confidence: float = 0.0
    side_effect: bool = False          # something outside this machine is asked for
    may_need_tool: bool = True         # a tool could plausibly be required
    needs_clarification: bool = False  # proceeding would mean guessing
    about_task: bool = False           # this refers to work already in flight

    @property
    def is_complex(self) -> bool:
        return self.complexity == COMPLEX

    @property
    def wants_action(self) -> bool:
        """Did the user actually ask for something to happen?

        The one question the execution path needs answered before it offers a
        tool - and the answer is deliberately biased towards yes.

        Withholding tools is a real power, so it needs POSITIVE EVIDENCE that
        this was a remark: a greeting, a recognised reaction, a hedged
        suggestion. It is never inferred from the absence of evidence that
        something was a command, because that depends on a verb list being
        complete and no verb list ever is. An earlier version did infer it, and
        "remember that I prefer metric units", "calculate 40 psi in bar" and
        "test the auth module" all came out as remarks with no tools - the
        failure in the dangerous direction, where a real instruction is silently
        disarmed and the user cannot tell why.

        So the fallback reading, the one with nothing to go on, keeps its tools.
        A remark that slips through is shown tools it will not use, which is
        what Leti did before this layer existed and costs nothing.
        """
        return not (self.kind in NO_ACTION_KINDS
                    and self.confidence >= CONFIDENT_CONVERSATION)

    def as_dict(self) -> Dict[str, Any]:
        """The Intent Layer's structured output, for diagnostics and tests."""
        return {
            "kind": self.kind,
            "confidence": round(self.confidence, 2),
            "shape": self.shape,
            "complexity": self.complexity,
            "side_effect_requested": self.side_effect,
            "tool_possibly_required": self.may_need_tool,
            "clarification_required": self.needs_clarification,
            "belongs_to_existing_task": self.about_task,
            "entities": list(self.entities),
            "stages": list(self.stages),
        }

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
        # An unreadable request is treated as an ordinary one that may need a
        # tool, NOT as a remark: failing closed here would mean a parser bug
        # silently making Leti refuse to act.
        return Intent(kind=QUESTION, confidence=0.0, may_need_tool=True)


def _read(text: str, history: Sequence[Dict[str, Any]]) -> Intent:
    lowered = text.lower()
    intent = Intent()

    from core.tool_router import _PURE_CHAT     # one definition of "just chat"

    if _PURE_CHAT.match(text.strip()):
        return Intent(shape=CHAT, complexity=SIMPLE, kind=CONVERSATION,
                      confidence=1.0, may_need_tool=False)

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

    # The Intent Layer, last, because it reads what the shape pass worked out.
    intent.kind, intent.confidence = _classify(text, intent)
    intent.side_effect = bool(_EXTERNAL.search(text)) and intent.wants_action
    intent.needs_clarification = bool(intent.question_to_ask) or (
        intent.kind == CLARIFICATION)
    intent.about_task = bool(_ABOUT_A_TASK.search(text)) or intent.kind in LIFECYCLE_KINDS
    # A remark needs no tool, and neither does a question about the world. Both
    # are deliberately conservative readings: the cost of withholding tools from
    # something that turns out to want one is the model saying so and the user
    # rephrasing; the cost of the reverse is Leti acting on a passing thought.
    # A hint for the report, not a gate: wants_action above is what decides.
    intent.may_need_tool = intent.wants_action and intent.kind not in (
        QUESTION, CORRECTION, CLARIFICATION)
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
