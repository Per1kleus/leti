"""Two ways to be Leti, sharing one of everything underneath.

Default Leti is a general assistant: voice-first, fast, and shown a small set of
tools chosen per request. Coding Leti is a software-development workspace: it
reads repositories, plans before it edits, checkpoints before it changes
anything, tests what it changed, and knows about git and GitHub.

They are not two applications. There is one orchestrator, one task manager, one
SafetyGuard, one Permission Center, one scheduler, one memory, one Project
Memory, one File Intelligence and one model. A mode changes three things and
nothing else:

    which tools exist for this turn      (VISIBLE below)
    what Leti is told it is doing        (system_note)
    how deliberate the workflow is       (core/coding.py)

The tool visibility is the load-bearing part, and it is why the coding tools do
not cost Default Mode anything at all. Leti's whole tool list already sits close
to the context window - the router picks a subset per request, but the FALLBACK
has to fit, and the fallback is the whole registry. Registering four more tools
for everyone would have spent the remaining headroom and forced num_ctx up.

So a mode owns a set of modules, the registry is filtered through it before
routing, and a turn in Default Mode is byte-identical to what it was before this
file existed: the coding tools are not in its fallback, not in its schemas, and
not in its prompt. A turn in Coding Mode sees a smaller, coding-shaped registry
that INCLUDES them - and is smaller than Default's, not larger.

Switching modes does not restart Ollama, reload the model, clear memory, cancel
tasks, change permissions or touch settings. It sets a string.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("leti.modes")

DEFAULT = "default"
CODING = "coding"
BUSINESS = "business"
MODES = (DEFAULT, CODING, BUSINESS)

# The tools that exist only in a specialised mode. Default Mode never sees these:
# not in its routing, not in its fallback, not in its context.
#
# switch_mode is in here too, and that is the important part. Entering a mode is
# an explicit COMMAND, matched deterministically in core/intent.py before any
# model call - so Default Mode does not need a tool for it, and not having one
# means the model CANNOT change mode from there however the request is phrased.
# "Check my customer emails" gets a Default Mode turn because there is no other
# turn available, not because the model decided well. The tool stays visible
# inside a specialised mode, where saying "what am I in" and "leave" are ordinary
# things to want, and the GUI selector works from anywhere.
CODING_ONLY_TOOLS = frozenset({"code_map", "git_workspace", "github"})
BUSINESS_ONLY_TOOLS = frozenset({"business_briefing", "business_calendar",
                                 "business_goals", "business_batch"})
SWITCH_TOOL = "switch_mode"
SPECIALISED_TOOLS = CODING_ONLY_TOOLS | BUSINESS_ONLY_TOOLS | {SWITCH_TOOL}

# What Coding Mode is shown. Everything a software task reaches for, and nothing
# else - a coding turn has no business being offered the weather, a paper trade
# or a social watch, and every schema left out is room for the code itself.
CODING_MODULES = (
    "tools.coding_agent",       # the coding tools: code map, git, GitHub
    "tools.coding",             # run_code, run_tests, install_dependency, inspect_project
    "tools.file_manager",
    "tools.documents",
    "tools.shell_runner",
    "tools.projects",
    "tools.web_search",
    "tools.browser",
    "tools.engineering",
    "tools.data_analysis",
    "tools.autonomous",         # long tasks, with the verification that already exists
    "tools.control_center",
    "tools.backup_restore",
    "tools.computer_use",       # still the last resort, still through the guard
    "tools.os_control",
    "tools.vision",
)


# What Business Mode is shown. The business layer Leti already has - the CRM, the
# contact book, the calendar writer - plus the things business work is actually
# made of: documents, spreadsheets, mail, research, tasks and reports. No coding
# tools, no screen control, no market monitoring: a business turn that is offered
# git and a paper-trading account is a turn whose tool list is not helping.
BUSINESS_MODULES = (
    "tools.business_agent",     # the briefing that ties the rest together
    "tools.business",           # the CRM that already exists: leads, income, expenses
    "tools.contacts",
    "tools.meeting_scheduler",
    "tools.email_client",
    "tools.documents",          # File Intelligence: proposals, invoices, contracts
    "tools.data_analysis",      # spreadsheets and CSVs, the existing way
    "tools.file_manager",
    "tools.projects",
    "tools.todo_list",
    "tools.autonomous",         # long tasks, with the verification that already exists
    "tools.workflow_tools",     # recurring business work, on the existing scheduler
    "tools.scheduler",
    "tools.web_search",
    "tools.browser",
    "tools.venture_scout",
    "tools.control_center",
    "tools.user_profile",
)


@dataclass
class Mode:
    name: str
    label: str
    description: str
    modules: Optional[List[str]] = None      # None means "everything except coding-only"
    note: str = ""
    entered_at: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


DEFAULT_MODE = Mode(
    name=DEFAULT,
    label="Leti",
    description="The general assistant: everything Leti does, routed per request.",
)

CODING_MODE = Mode(
    name=CODING,
    label="Coding Mode",
    description="A software-development workspace: repositories, tests, git and GitHub.",
    modules=list(CODING_MODULES),
    note=(
        "You are in CODING MODE: a dedicated software-development workspace, not the "
        "general assistant. Work like an engineer rather than a chat bot.\n"
        "For a question, an explanation or a one-line change, answer it directly - read "
        "what you need with the file tools and say what you found. Do not start a plan, "
        "a checkpoint or a test run for something that does not need one.\n"
        "For a substantial change - a feature, a bug with more than one possible cause, "
        "anything touching several files - work in this order: understand what was asked; "
        "explore the code with code_map before reading files, so you read the RIGHT files; "
        "plan; take a git checkpoint with git_workspace before you modify anything; "
        "implement; run the tests that relate to what you changed rather than the whole "
        "suite; diagnose real failures instead of editing until they pass; fix; re-test; "
        "review the diff with git_workspace; then report.\n"
        "Rules that do not bend: never claim a test passed unless you ran it and saw it "
        "pass. Never fabricate a diff, a file or an error. Never edit code you have not "
        "read. A test failure that existed before you started is not yours to hide - say "
        "so. If something needs permission you do not have, stop and say what and why."
    ),
)

BUSINESS_MODE = Mode(
    name=BUSINESS,
    label="Business Mode",
    description="A business operations workspace: pipeline, follow-ups, documents, reporting.",
    modules=list(BUSINESS_MODULES),
    note=(
        "You are in BUSINESS MODE: a business operations workspace, not the general "
        "assistant. Work like somebody's operations manager.\n"
        "For a simple question - today's priorities, one lead's stage, what a document "
        "says - answer it directly. Do not build a plan for something that needs a "
        "lookup.\n"
        "For real work - 'review my leads, prepare follow-ups, send the approved ones' - "
        "work in this order: understand what is being asked; gather what is actually "
        "there with business_briefing, business_next_actions, list_business_data and the "
        "document tools; plan before doing; execute; verify what you did by looking at "
        "the result rather than remembering it; leave the follow-ups somewhere they will "
        "be picked up; then report.\n"
        "Rules that do not bend. Never invent a customer, a number, a deal or a date - if "
        "the data is not there, say what is missing and where it would come from. Keep "
        "observed figures, calculated figures, assumptions and your own interpretation "
        "visibly apart, and say when a conclusion rests on too little data. Never say an "
        "email was sent, a meeting was booked or a record was updated unless a tool "
        "actually did it and said so. You are not a financial, legal, tax or employment "
        "adviser: analyse, organise, prepare and explain, and leave decisions with real "
        "regulated consequences to the user and their professionals."
    ),
)

_MODES = {DEFAULT: DEFAULT_MODE, CODING: CODING_MODE, BUSINESS: BUSINESS_MODE}

# The active mode, for this process. Deliberately not persisted to disk: Leti
# starts as the general assistant every time, and entering a specialised
# workspace is something the user does on purpose. See enter()/leave().
_current: str = DEFAULT
_entered_at: Optional[float] = None
_listeners: List[Any] = []


def current() -> str:
    return _current


def mode(name: Optional[str] = None) -> Mode:
    return _MODES.get(name or _current, DEFAULT_MODE)


def is_coding() -> bool:
    return _current == CODING


def on_change(callback) -> None:
    """The interface asks to be told, so the badge and the panel follow the mode."""
    _listeners.append(callback)


def _announce(name: str, note: str) -> None:
    for callback in list(_listeners):
        try:
            callback(name)
        except Exception:
            logger.debug("A mode listener failed; the mode change stands.")
    try:
        from core import diagnostics

        diagnostics.record_activity("mode", note)
    except Exception:
        logger.debug("Couldn't mirror a mode change to the activity log.")


def enter(name: str) -> Dict[str, Any]:
    """Switch mode. Changes a string, a tool list and a system note - nothing else.

    Nothing is restarted, reloaded, cleared or cancelled: the model stays loaded,
    Ollama is not touched, session memory and Project Memory carry across, running
    tasks keep running, and permissions are exactly what they were.
    """
    global _current, _entered_at

    wanted = str(name or "").strip().lower()
    if wanted not in _MODES:
        return {"ok": False, "error": f"'{name}' is not a mode. Use: {', '.join(MODES)}.",
                "mode": _current}
    if wanted == _current:
        return {"ok": True, "mode": _current, "changed": False,
                "note": f"Already in {_MODES[_current].label}."}

    if _current != DEFAULT and wanted != _current:
        _release_workspaces()
    previous, _current = _current, wanted
    _entered_at = time.time() if wanted != DEFAULT else None
    _MODES[wanted].entered_at = _entered_at
    _announce(wanted, f"Switched from {_MODES[previous].label} to {_MODES[wanted].label}")
    logger.info(f"Mode: {previous} -> {wanted}")
    return {"ok": True, "mode": _current, "previous": previous, "changed": True,
            "note": f"{_MODES[wanted].label} is on. {_MODES[wanted].description}"}


def leave() -> Dict[str, Any]:
    """Back to the general assistant, and stop paying for the workspace."""
    _release_workspaces()
    return enter(DEFAULT)


def _release_workspaces() -> None:
    """Whatever a specialised mode was holding, let go of it.

    Both workspaces are in-process and hold nothing durable - what deserves to
    outlive a mode is already in the stores it came from (Project Memory, the
    business records, the task manager). Releasing here is what makes "the mode
    stops costing anything when you leave it" true rather than aspirational.
    """
    from core import business, coding

    coding.release()
    business.release()


def visible_tools(registry: Any, name: Optional[str] = None) -> Set[str]:
    """The tools this mode may use. The registry still holds every one of them.

    A mode narrows what a turn can SEE, never what exists: every tool stays
    registered, stays permissioned and stays runnable the moment its mode is on.
    """
    active = mode(name)
    everything = set(registry.names())
    if active.modules is None:
        return everything - SPECIALISED_TOOLS
    allowed = set(active.modules)
    visible = {n for n in everything if _module_of(registry, n) in allowed}
    # Every specialised mode can always be left from inside it.
    return visible | ({SWITCH_TOOL} if SWITCH_TOOL in everything else set())


def _module_of(registry: Any, name: str) -> str:
    tool = registry.get(name)
    return type(tool).__module__ if tool is not None else ""


def system_note(name: Optional[str] = None) -> str:
    return mode(name).note


def describe(registry: Any = None) -> Dict[str, Any]:
    """What the interface shows: which mode, since when, and how big it is."""
    active = mode()
    out = {
        "mode": active.name,
        "label": active.label,
        "description": active.description,
        "since": (time.strftime("%H:%M", time.localtime(_entered_at)) if _entered_at else None),
        "modes": [{"name": m.name, "label": m.label, "description": m.description}
                  for m in _MODES.values()],
    }
    if registry is not None:
        out["tools_available"] = len(visible_tools(registry))
        out["tools_registered"] = len(registry.names())
    if active.name == CODING:
        from core import coding

        out["coding"] = coding.status()
    elif active.name == BUSINESS:
        from core import business

        out["business"] = business.status()
    return out


def reset_for_tests() -> None:
    """Back to a launched-just-now state. Only the tests call this."""
    global _current, _entered_at
    _current, _entered_at = DEFAULT, None
    _listeners.clear()
