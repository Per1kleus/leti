"""Which tools this request should be shown - not which it is allowed to run.

Every tool schema was sent on every call, and on every iteration of the
tool-calling loop. Measured: 103 tools serialise to 63,410 characters, roughly
16,000-18,000 tokens, before the system prompt, personality, profile, recalled
memories or the conversation get any of a 24,576-token window. This picks a
relevant subset instead.

The registry stays the source of truth. Tools are grouped by the module they are
defined in, because that grouping already exists and is already correct -
tools/email_client.py is the email tools, tools/browser.py is the browser ones -
so it cannot drift out of step with the code the way a hand-maintained list of
tool names would. A named category then bundles the modules that belong together
for the purposes of a request, so "email John about tomorrow's meeting" can reach
contacts and the calendar as well as the mailbox.

Matching is deterministic and local: the request's words are scored against each
tool's own name and description, weighted so that a word shared by half the
registry counts for little and a rare one counts for a lot. There is no second
model, no network call and no embedding index - a router that costs a language
model call would spend more time than the context it saves.

Three rules keep it from ever costing Leti an ability:

  - Breadth over precision. The unit of exposure is a whole module, never a
    single tool, and a floor keeps the set generous. Exposing ten tools instead
    of five is not worth one failed request.
  - Uncertainty broadens. A weak match adds the rest of its category rather than
    narrowing, and a module this file has never heard of is always exposed.
  - Failure falls all the way back. Anything unexpected - an exception, an empty
    result, routing switched off - returns the complete registry, which is
    exactly the behaviour that existed before this file.

It answers one question and no others. Whether a call is permitted stays with
SafetyGuard, which sees every call regardless of how the tool was chosen.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

logger = logging.getLogger("leti.tool_router")

# The smallest set worth sending. Below this the saving is noise and the risk of
# having left out the one tool that mattered is not.
MIN_TOOLS = 12
# A module has to clear this share of the best module's score to be included, so
# a strong match brings its neighbours and a weak field does not bring everything.
# Swept against a set of representative requests: 0.34 and 0.45 were equally
# correct but broader (30.1 and 26.2 tools on average), and 0.65 started missing
# a tool. 0.55 was the widest setting that still got every request right.
RELATIVE_THRESHOLD = 0.55
# Below this the request did not really match anything and the answer is "broaden".
ABSOLUTE_THRESHOLD = 1.6

# Words that carry no signal about which capability is wanted.
#
# Worth being thorough here: a filler word that survives is weighted like any
# other, and because it is rare among tool DESCRIPTIONS it can score high. "about"
# was missing, appears in seven tool descriptions, and on its own routed "tell me
# a joke about computers" into the system and security tools - the request's only
# real word, "computers", did not match anything at all.
_STOPWORDS = frozenset("""
a about above after again all also am an and another any anything are around as
at back be because been before being below between both but by can could did do
does doing done down during each either else even every few for from further get
gets give go going had has have having he her here hers him his how i if in into
is it its itself just let like make many may me might more most much must my need
needs no nor not now of off on once one only or other others ought our out over
own per please put same say see she should since so some something such take tell
than that the their them then there these they thing things this those though
through to too under until up upon us use used uses using very want wants was way
we well were what when where whether which while who whom why will with within
without would you your yours
""".split())

_WORD = re.compile(r"[a-z0-9]+")

# Categories bundle the modules that a single request tends to need together.
# Membership is by MODULE, so adding a tool to an existing module needs no change
# here at all. A module missing from this table is always exposed - see
# _uncategorised_modules - and a test asserts the table still covers the registry
# so that stays a safety net rather than a habit.
CATEGORIES: Dict[str, List[str]] = {
    "files":         ["tools.file_manager", "tools.backup_restore"],
    "system":        ["tools.os_control", "tools.shell_runner", "tools.system_health",
                      "tools.vision"],
    "security":      ["tools.system_defense", "tools.network_security",
                      "tools.backup_restore"],
    "web":           ["tools.web_search", "tools.browser", "tools.image_search",
                      "tools.social_login"],
    "communication": ["tools.email_client", "tools.contacts", "tools.meeting_scheduler"],
    "scheduling":    ["tools.scheduler", "tools.meeting_scheduler", "tools.todo_list"],
    "social":        ["tools.social_media", "tools.social_login"],
    "development":   ["tools.coding", "tools.projects"],
    "data":          ["tools.data_analysis", "tools.engineering"],
    "business":      ["tools.business", "tools.venture_scout", "tools.trading_platform"],
    "personal":      ["tools.user_profile", "tools.personality", "tools.contacts"],
    "everyday":      ["tools.weather", "tools.todo_list", "tools.sketch"],
}

# Always sent, whatever the request. Searching the web is the capability a model
# reaches for when nothing else fits, and it is two tools.
ALWAYS_ON_MODULES = ("tools.web_search",)

# A message that is only a greeting or an acknowledgement needs no tools at all.
# Deliberately anchored to the WHOLE message: "thanks" is chat, "thanks, now open
# spotify" is not, and the cost of getting this wrong is a request Leti cannot act
# on. Anything with more in it than this goes down the ordinary path.
_PURE_CHAT = re.compile(
    r"^\W*(hi|hey|hello|yo|sup|thanks|thank you|thx|ty|ok|okay|cool|nice|great|"
    r"good (morning|afternoon|evening|night)|bye|goodbye|see you|night|"
    r"how are you|how's it going|what's up)"
    r"[\s,.!?]*(leti)?[\s,.!?]*$",
    re.I,
)


@dataclass
class Routing:
    """What was chosen, and honestly how it was chosen."""

    tool_names: List[str]
    categories: List[str] = field(default_factory=list)
    confident: bool = True
    full_fallback: bool = False
    no_tools: bool = False
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.tool_names)


# --------------------------------------------------------------------------- #
# The index, built once per registry
# --------------------------------------------------------------------------- #

class _Index:
    """Tool terms and their weights. Built once and reused for every request."""

    def __init__(self, registry: Any):
        self.module_of: Dict[str, str] = {}
        self.terms_of: Dict[str, Set[str]] = {}
        self.tools_in: Dict[str, List[str]] = {}

        for name in registry.names():
            tool = registry.get(name)
            module = type(tool).__module__
            self.module_of[name] = module
            self.tools_in.setdefault(module, []).append(name)
            self.terms_of[name] = _terms(f"{name} {getattr(tool, 'description', '')}")

        # Inverse document frequency: a word on half the tools says almost nothing
        # about which one is wanted; a word on one tool says a great deal.
        total = max(1, len(self.terms_of))
        seen: Dict[str, int] = {}
        for terms in self.terms_of.values():
            for term in terms:
                seen[term] = seen.get(term, 0) + 1
        self.weight = {term: math.log(total / count) for term, count in seen.items()}

        self.category_of: Dict[str, List[str]] = {}
        for category, modules in CATEGORIES.items():
            for module in modules:
                self.category_of.setdefault(module, []).append(category)

    def all_names(self) -> List[str]:
        return sorted(self.terms_of)

    def uncategorised_modules(self) -> List[str]:
        """Modules this file has never been told about. Always exposed."""
        return sorted(m for m in self.tools_in if m not in self.category_of)


def _terms(text: str) -> Set[str]:
    """Significant words in a tool's name and description, snake_case split."""
    words = _WORD.findall((text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


_CACHE: Dict[int, _Index] = {}


def _index_for(registry: Any) -> _Index:
    key = id(registry)
    index = _CACHE.get(key)
    if index is None or len(index.terms_of) != len(registry.names()):
        index = _Index(registry)
        _CACHE[key] = index
    return index


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

def _names_in(index: _Index, modules: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for module in modules:
        out.update(index.tools_in.get(module, ()))
    return out


def route(request: str, registry: Any) -> Routing:
    """Pick the tools for one request. Never raises; the worst case is all of them."""
    try:
        return _route(request, registry)
    except Exception as e:
        logger.warning(f"Tool routing failed ({e}); sending the full tool set.")
        return Routing(tool_names=sorted(registry.names()), confident=False,
                       full_fallback=True, reason=f"routing failed: {e}")


def _route(request: str, registry: Any) -> Routing:
    index = _index_for(registry)
    everything = index.all_names()

    if not isinstance(request, str) or not request.strip():
        return Routing(tool_names=everything, confident=False, full_fallback=True,
                       reason="no request text to route on")

    if _PURE_CHAT.match(request.strip()):
        return Routing(tool_names=[], no_tools=True, reason="greeting, no tools needed")

    asked = _terms(request)
    if not asked:
        return Routing(tool_names=everything, confident=False, full_fallback=True,
                       reason="nothing to match on")

    # Score every tool, then let each module take its best tool's score: one
    # unmistakable match should carry its neighbours, not be averaged away by them.
    tool_score: Dict[str, float] = {}
    for name, terms in index.terms_of.items():
        hit = asked & terms
        if hit:
            tool_score[name] = sum(index.weight.get(t, 0.0) for t in hit)

    module_score: Dict[str, float] = {}
    for name, score in tool_score.items():
        module = index.module_of[name]
        module_score[module] = max(module_score.get(module, 0.0), score)

    if not module_score:
        return Routing(tool_names=everything, confident=False, full_fallback=True,
                       reason="nothing matched; sending everything")

    best = max(module_score.values())
    confident = best >= ABSOLUTE_THRESHOLD
    cutoff = best * RELATIVE_THRESHOLD
    chosen = {m for m, s in module_score.items() if s >= cutoff}

    # A weak best match means the request was not really recognised. Broaden by
    # taking whole categories rather than narrowing onto a guess.
    categories: Set[str] = set()
    for module in list(chosen):
        categories.update(index.category_of.get(module, ()))
    if not confident:
        for category in list(categories):
            chosen.update(CATEGORIES.get(category, ()))
    else:
        # Confident still pulls in the rest of the top module's categories: the
        # tools people need together live together.
        for module in [m for m, s in module_score.items() if s >= best * 0.7]:
            for category in index.category_of.get(module, ()):
                chosen.update(CATEGORIES.get(category, ()))

    chosen.update(ALWAYS_ON_MODULES)
    chosen.update(index.uncategorised_modules())

    selected = _names_in(index, chosen)

    # A floor, not a target. Below this the saving is noise and the risk of having
    # left out the one tool that mattered is not, so the next best modules are
    # added until the set is worth sending.
    if len(selected) < MIN_TOOLS:
        for module, _ in sorted(module_score.items(), key=lambda kv: -kv[1]):
            if len(selected) >= MIN_TOOLS:
                break
            chosen.add(module)
            selected = _names_in(index, chosen)

    if len(selected) >= len(everything):
        return Routing(tool_names=everything, categories=sorted(categories),
                       confident=confident, full_fallback=True,
                       reason="selection covered the whole registry")

    return Routing(
        tool_names=sorted(selected),
        categories=sorted(categories),
        confident=confident,
        reason=("matched " + ", ".join(sorted(m.split(".")[-1] for m in chosen))
                if confident else
                "weak match, broadened to " + ", ".join(sorted(categories)) or "everything"),
    )


def select_tools_for(request: str, registry: Any, enabled: Optional[bool] = None) -> Routing:
    """The entry point. `enabled` false restores the previous behaviour exactly."""
    if enabled is None:
        enabled = _routing_enabled()
    if not enabled:
        return Routing(tool_names=sorted(registry.names()), full_fallback=True,
                       reason="tool routing is switched off")
    return route(request, registry)


def _routing_enabled() -> bool:
    try:
        from core.config_loader import get_settings

        return bool(get_settings().get("tool_routing", {}).get("enabled", True))
    except Exception:
        return True


def last_user_message(messages: Sequence[Dict[str, Any]]) -> str:
    """The text routing should read: the most recent thing the user actually said."""
    for message in reversed(list(messages or ())):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""
