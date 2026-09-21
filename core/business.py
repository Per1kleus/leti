"""What Business Mode knows that Default Mode does not.

Almost nothing here is new data. Leti already has a CRM (tools/business.py: leads
with stages and values, income, expenses, and the metrics computed from them), a
contact book, a task manager, a scheduler, Project Memory and File Intelligence.
A business operations assistant is not a new database - it is the thing that reads
all of those at once and says "these four are what today is about".

So this module owns two small things:

  A workspace. Which business is being worked on, what the current objectives
  are, and what has been promised to whom. It lives in process, it is released
  when the mode is left, and it deliberately does NOT persist on its own: what
  deserves to outlive a session is a task, a lead, a project note or a scheduled
  workflow, and all four of those already have somewhere to live. A workspace
  that quietly became permanent memory would be a second memory system.

  A briefing. Overdue and blocked tasks from the task manager, leads that have
  gone quiet from the CRM, follow-ups that are due, work waiting on the user's
  approval, and scheduled business work about to run - assembled on demand, from
  the stores that already hold them, when somebody asks.

Two things it refuses to do. It never invents a customer, a figure or a date: if
a source is not connected it says which one and where to connect it. And it never
implies Leti can read a calendar - Leti can CREATE events over CalDAV but has no
tool that reads one back, so "today's meetings" is a thing it says it cannot see
rather than a thing it guesses at.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import entities

logger = logging.getLogger("leti.business")

MAX_ITEMS = 8               # a briefing is a page, not a report
MAX_OBJECTIVES = 6
MAX_TEXT = 200
STALE_LEAD_DAYS = 14        # the CRM's own default for "gone quiet"


# --------------------------------------------------------------------------- #
# The workspace
# --------------------------------------------------------------------------- #

@dataclass
class Workspace:
    """What this business session is about. In process, released on leaving.

    Nothing here is written to disk. Anything that should outlive the session is
    put where that kind of thing already lives - a task in the task manager, a
    stage change on the lead itself, a note in Project Memory - and this holds
    only the framing that makes a conversation coherent while it is happening.
    """
    business_name: str = ""
    project: str = ""
    objectives: List[str] = field(default_factory=list)
    opened_at: float = field(default_factory=time.time)

    def summary(self) -> Dict[str, Any]:
        return {"business": self.business_name or None, "project": self.project or None,
                "objectives": list(self.objectives),
                "open_since": time.strftime("%H:%M", time.localtime(self.opened_at))}


_workspace: Optional[Workspace] = None


def workspace() -> Optional[Workspace]:
    return _workspace


def open_workspace(business_name: str = "", project: str = "",
                   objectives: Optional[List[str]] = None) -> Workspace:
    global _workspace

    if _workspace is None:
        _workspace = Workspace()
    if business_name:
        _workspace.business_name = str(business_name)[:MAX_TEXT]
    if project:
        _workspace.project = str(project)[:MAX_TEXT]
    if objectives:
        cleaned = [str(o).strip()[:MAX_TEXT] for o in objectives if str(o).strip()]
        _workspace.objectives = cleaned[:MAX_OBJECTIVES]
    return _workspace


def release() -> None:
    """Leaving Business Mode. Nothing business-specific stays in memory.

    The stores are untouched: leads, goals, tasks, contacts, projects and
    scheduled work are exactly where they were, because none of them belonged to
    the workspace. What goes is the session framing and any batch nobody acted on.
    """
    global _workspace

    _workspace = None
    from core import business_approvals

    business_approvals.release()


def status() -> Dict[str, Any]:
    space = _workspace
    out: Dict[str, Any] = {
        "workspace": space.summary() if space else None,
        "sources": connected_sources(),
    }
    try:
        from tools.business import load_records

        records = load_records()
        out["leads"] = len(records.get("lead", []))
    except Exception:
        out["leads"] = None
    return out


def connected_sources() -> Dict[str, str]:
    """Which business sources are actually available, and which are not.

    Said plainly and per source, because "I couldn't find any meetings" and "I
    cannot see your calendar at all" are different sentences and only one of them
    is honest here.
    """
    from core.config_loader import get_settings

    settings = get_settings()
    sources = {
        "crm": "built in (leads, income and expenses in Leti's own records)",
        "contacts": "built in",
        "documents": "built in (File Intelligence)",
        "tasks": "built in",
    }
    sources["email"] = ("connected" if settings.get("email", {}).get("username")
                        else "not connected - add it under Connections to read or send mail")
    sources["calendar"] = (
        "connected (read and write over CalDAV)"
        if all(settings.get("calendar", {}).get(k)
               for k in ("caldav_url", "username", "app_password"))
        else "not connected - add it under Connections to read or create events")
    sources["goals"] = "built in"
    return sources


# --------------------------------------------------------------------------- #
# The briefing
# --------------------------------------------------------------------------- #

def briefing(now: Optional[float] = None) -> Dict[str, Any]:
    """What today is about, from the stores that already know.

    Never raises: a source that cannot be read becomes a named gap in the
    briefing, because a summary that silently omits a section reads as "nothing
    to report there".
    """
    now = now if now is not None else time.time()
    sections: Dict[str, Any] = {}
    gaps: List[str] = []

    for name, gather in (("waiting_on_you", _waiting_on_you),
                         ("work_in_flight", _work_in_flight),
                         ("leads_needing_attention", _stale_leads),
                         ("pipeline", _pipeline),
                         ("goals", _goals),
                         ("todays_meetings", _todays_meetings),
                         ("scheduled", _scheduled_soon)):
        try:
            sections[name] = gather(now)
        except Exception as e:
            sections[name] = [] if name != "pipeline" else {}
            gaps.append(f"{name}: {e}")
            logger.debug(f"Briefing section {name} unavailable: {e}")

    space = _workspace
    return {
        "as_of": time.strftime("%Y-%m-%d %H:%M", time.localtime(now)),
        "workspace": space.summary() if space else None,
        # What a person should look at first, gathered from the sections rather
        # than computed separately - one source of truth per fact.
        "needs_your_attention": _attention(sections),
        **sections,
        "blocked": _blocked(sections),
        "cannot_see": _cannot_see(),
        "gaps": gaps or None,
        "how_to_report": (
            "Lead with needs_your_attention, then today's meetings, then what has gone "
            "quiet. Keep observed figures separate from anything you worked out, and say "
            "when a section is empty because there is nothing there rather than because "
            "Leti cannot see it - 'cannot_see' lists that difference and 'gaps' lists "
            "sources that failed to read."),
    }


def _goals(now: float) -> List[Dict[str, Any]]:
    """Active goals, with progress only where progress is actually knowable."""
    from core import business_goals

    out = []
    for goal in business_goals.load_goals():
        if goal.get("status") in (business_goals.COMPLETED, business_goals.PAUSED):
            continue
        out.append(business_goals.describe(goal))
    return out[:MAX_ITEMS]


def _todays_meetings(now: float) -> Dict[str, Any]:
    """Today's calendar, or why Leti cannot see it.

    The two answers this must never confuse are "you have no meetings" and "Leti
    could not reach your calendar". An unreachable calendar raises inside
    read_events and lands in `problem` here; an empty day lands in `events` as an
    empty list with available=True beside it.
    """
    from datetime import datetime, timedelta

    from tools.meeting_scheduler import calendar_is_configured, overlapping

    if not calendar_is_configured():
        return {"available": False, "events": [],
                "problem": ("No calendar is connected, so Leti cannot see your meetings. "
                            "This is not the same as having none."),
                "fix": "Add a CalDAV calendar under Connections."}

    start = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        from core.signals import run_blocking
        from tools.meeting_scheduler import read_events

        events = run_blocking(read_events(start, start + timedelta(days=1)), timeout=30)
    except Exception as e:
        return {"available": False, "events": [],
                "problem": f"Leti could not read the calendar: {e}. That is not an empty day."}
    return {"available": True, "events": events, "count": len(events),
            "conflicts": overlapping(events)}


def _attention(sections: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The short list, assembled from sections that were already gathered."""
    from core import business_goals

    attention: List[Dict[str, Any]] = []
    for waiting in sections.get("waiting_on_you") or []:
        attention.append({"what": waiting["what"], "why": waiting["why"],
                          "kind": "waiting for your approval"})
    quiet = sections.get("leads_needing_attention") or []
    if quiet:
        attention.append({
            "what": f"{len(quiet)} lead(s) have gone quiet",
            "why": "; ".join(f"{l['lead']} ({l['quiet_for_days']}d)" for l in quiet[:3]),
            "kind": "follow-up"})
    try:
        for flagged in business_goals.needs_attention():
            attention.append({"what": flagged["goal"], "why": flagged["why"],
                              "kind": "goal"})
    except Exception as e:
        logger.debug(f"Goal attention unavailable: {e}")
    meetings = sections.get("todays_meetings") or {}
    if meetings.get("conflicts"):
        attention.append({"what": f"{len(meetings['conflicts'])} overlapping meeting(s)",
                          "why": "; ".join(" vs ".join(c["between"])
                                           for c in meetings["conflicts"][:2]),
                          "kind": "calendar"})
    return attention[:MAX_ITEMS]


def _blocked(sections: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Things that cannot move until somebody does something."""
    blocked = []
    for goal in sections.get("goals") or []:
        tasks = goal.get("tasks") or {}
        if goal.get("status") == "blocked" or tasks.get("blocked"):
            blocked.append({"what": goal.get("name"), "kind": "goal",
                            "why": (f"{tasks.get('blocked')} linked task(s) blocked"
                                    if tasks.get("blocked") else "marked blocked")})
    for waiting in sections.get("waiting_on_you") or []:
        blocked.append({"what": waiting["what"], "kind": "task", "why": waiting["why"]})
    return blocked[:MAX_ITEMS]


def _waiting_on_you(now: float) -> List[Dict[str, Any]]:
    """Tasks stopped for the user's approval. Nothing moves until they answer."""
    from core import task_manager

    out = []
    for task in task_manager.load_tasks():
        if task.get("status") != task_manager.WAITING_FOR_USER:
            continue
        out.append({"what": task.get("name"), "task_id": task.get("id"),
                    "why": task.get("blocked_reason") or "a step needs permission",
                    "waiting_since": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(task.get("updated_at", now)))})
    return out[:MAX_ITEMS]


def _work_in_flight(now: float) -> List[Dict[str, Any]]:
    from core import task_manager

    out = []
    for task in task_manager.load_tasks():
        if task.get("status") not in (task_manager.RUNNING, task_manager.QUEUED,
                                      task_manager.PAUSED):
            continue
        progress = task_manager.progress(task)
        out.append({"what": task.get("name"), "status": task.get("status"),
                    "step": f"{progress['current_step']}/{progress['steps_total']}"
                            if progress["steps_total"] else "working"})
    return out[:MAX_ITEMS]


def _stale_leads(now: float) -> List[Dict[str, Any]]:
    """Open leads that have gone quiet, from the CRM's own ranking.

    Deliberately the same weighting business_next_actions uses rather than a
    second opinion about what matters - two rankings that disagree is worse than
    one that can be argued with.
    """
    from tools.business import OPEN_STAGES, load_records, stage_weights

    weights = stage_weights()
    stale_after = STALE_LEAD_DAYS * 86_400
    out = []
    for lead in load_records().get("lead", []):
        if lead.get("stage") not in OPEN_STAGES:
            continue
        touched = lead.get("stage_changed_at") or lead.get("created_at", now)
        idle = now - touched
        if idle < stale_after:
            continue
        value = float(lead.get("value", 0) or 0)
        out.append({
            "lead": lead.get("name"), "lead_id": lead.get("id"),
            "stage": lead.get("stage"),
            "value": value,
            "weighted_value": round(value * weights.get(lead.get("stage", "new"), 0), 2),
            "quiet_for_days": int(idle / 86_400),
            "contact_id": lead.get("contact_id"),
        })
    out.sort(key=lambda item: -item["weighted_value"])
    return out[:MAX_ITEMS]


def _pipeline(now: float) -> Dict[str, Any]:
    """The open pipeline, weighted the way the CRM already weights it."""
    from tools.business import OPEN_STAGES, load_records, stage_weights

    weights = stage_weights()
    leads = [l for l in load_records().get("lead", []) if l.get("stage") in OPEN_STAGES]
    by_stage: Dict[str, Dict[str, Any]] = {}
    total = weighted = 0.0
    for lead in leads:
        stage = lead.get("stage", "new")
        value = float(lead.get("value", 0) or 0)
        total += value
        weighted += value * weights.get(stage, 0)
        entry = by_stage.setdefault(stage, {"count": 0, "value": 0.0})
        entry["count"] += 1
        entry["value"] += value
    return {
        "open_leads": len(leads),
        "total_value": round(total, 2),
        "weighted_value": round(weighted, 2),
        "by_stage": {k: {"count": v["count"], "value": round(v["value"], 2)}
                     for k, v in sorted(by_stage.items())},
        "what_weighted_means": ("Each lead's value multiplied by the probability its "
                                "stage is worth. A calculated figure, not money in."),
    }


def _scheduled_soon(now: float) -> List[Dict[str, Any]]:
    """Recurring business work about to run, from the scheduler that already runs it."""
    from core.watches import SCHEDULER_TASK_NAME
    from tools import scheduler

    out = []
    for task in scheduler.load_tasks():
        due = task.get("next_run")
        if not task.get("enabled") or not due or task.get("name") == SCHEDULER_TASK_NAME:
            continue
        if due > now + 86_400:
            continue
        out.append({"what": task.get("name"),
                    "runs_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(due))})
    return sorted(out, key=lambda item: item["runs_at"])[:MAX_ITEMS]


def _cannot_see() -> List[str]:
    """The sources a briefing would want and does not have. Named, not omitted."""
    missing = []
    for name, state in connected_sources().items():
        if state.startswith("not connected") or state.startswith("write only"):
            missing.append(f"{name}: {state}")
    return missing


# --------------------------------------------------------------------------- #
# Follow-ups
# --------------------------------------------------------------------------- #

def follow_ups(now: Optional[float] = None, stale_after_days: float = STALE_LEAD_DAYS
               ) -> Dict[str, Any]:
    """Who is owed contact, with what Leti actually knows about each one.

    This prepares; it does not send. Drafting an email is a tool call and sending
    one is an external action that asks first - being in Business Mode changes
    neither of those things.
    """
    now = now if now is not None else time.time()
    contacts_by_id = {}
    try:
        from tools.contacts import _load as load_contacts

        contacts_by_id = {c.get("id"): c for c in load_contacts()}
    except Exception as e:
        logger.debug(f"Contacts unavailable for follow-ups: {e}")

    due = []
    for lead in _stale_leads(now):
        contact = contacts_by_id.get(lead.get("contact_id")) or {}
        # Whether there is an address, not what it is: a follow-up list does not
        # need everyone's email in it to say who can be reached.
        due.append({
            **lead,
            "contact_name": contact.get("name"),
            "reachable": bool(contact.get("email")),
        })

    unreachable = [d["lead"] for d in due if not d["reachable"]]
    return {
        "due": due,
        "count": len(due),
        "stale_after_days": stale_after_days,
        "no_way_to_reach": unreachable or None,
        "next": ("Draft each one, show the user, and send only the ones they approve - "
                 "sending is an external action and asks first."),
    }


# --------------------------------------------------------------------------- #
# "the customer", "that lead", "the project"
# --------------------------------------------------------------------------- #
#
# Resolution over the data that already exists - leads and goals from the business
# records, people from the contact book, projects from Project Memory, tasks from
# the task manager, meetings from the calendar. No index and no new store: these
# are small collections and this is a search, run when a sentence needs it.
#
# The rule is the one that makes it safe: several matches is an ANSWER, not a
# problem to be resolved by picking the first. Acting on the wrong customer is
# worse than asking which one.
#
# What narrows it down is core/entities.py: the name, yes, but also whether it
# came up earlier in this conversation, whether it is the project that is open,
# whether an active task names it, whether the date in the reference lines up.
# That module owns the signals AND the choice between the three outcomes, so
# "which one did they mean" is decided in one place whoever is asking.

ENTITY_KINDS = ("lead", "contact", "goal", "project", "task", "meeting")
MAX_CANDIDATES = 6

def _significant(reference: str) -> List[str]:
    """The words in a reference that could identify something.

    core/entities.py owns the list of what cannot - "the customer" has to read as
    bare in both halves of resolution, and two copies of that list would drift.
    """
    return entities.significant_words(reference)


def resolve(reference: str, kind: str = "", now: Optional[float] = None) -> Dict[str, Any]:
    """What "the customer" refers to, or the candidates, or nothing.

    Three honest outcomes and no fourth. One match resolves it. Several is
    reported as several, with what distinguishes them, so the next question is
    "which one". None says so rather than returning the closest thing to hand.
    """
    now = now if now is not None else time.time()
    words = _significant(reference)
    wanted = [k for k in ENTITY_KINDS if not kind or k == kind]
    if kind and kind not in ENTITY_KINDS:
        return {"resolved": None, "candidates": [], "problem":
                f"'{kind}' is not a business entity. Known: {', '.join(ENTITY_KINDS)}."}

    candidates: List[Dict[str, Any]] = []
    unavailable: List[str] = []
    for entity_kind in wanted:
        try:
            candidates.extend(_candidates_of(entity_kind, words, now))
        except Exception as e:
            unavailable.append(f"{entity_kind}: {e}")

    # A bare reference ("the customer") has nothing distinctive in it, so every
    # entity of that kind stays a candidate and the context below is what narrows
    # it - the right answer when one thing is clearly in play, and a question when
    # five are. With distinctive words, anything matching none of the context
    # either is dropped by choose(), which is what keeps "I cannot find Wakanda
    # Industries" from becoming "here are four things it might be".
    context = entities.gather(now=now)
    outcome = entities.choose(candidates, reference, context, kind=kind)

    if unavailable:
        outcome["unavailable"] = unavailable
        if outcome.get("problem"):
            outcome["problem"] += (" Sources that could not be read: "
                                   + "; ".join(unavailable))
    else:
        outcome["unavailable"] = None
    return outcome


def _candidates_of(kind: str, words: List[str], now: float) -> List[Dict[str, Any]]:
    """Everything of one kind that a reference could mean, scored by name overlap."""
    def scored(name: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        lowered = str(name or "").lower()
        score = sum(2 for w in words if w in lowered)
        return {"kind": kind, "name": name, "score": score, **extra}

    if kind == "lead":
        from tools.business import load_records

        return [scored(l.get("name"), {"id": l.get("id"), "stage": l.get("stage"),
                                       "value": l.get("value")})
                for l in load_records().get("lead", [])]
    if kind == "goal":
        from core import business_goals

        return [scored(g.get("name"), {"id": g.get("id"), "status": g.get("status")})
                for g in business_goals.load_goals()]
    if kind == "contact":
        from tools.contacts import _load

        return [scored(c.get("name"), {"id": c.get("id"), "company": c.get("company"),
                                       "reachable": bool(c.get("email"))})
                for c in _load()]
    if kind == "project":
        from tools.projects import list_projects

        return [scored(p if isinstance(p, str) else p.get("name"), {})
                for p in list_projects()]
    if kind == "task":
        from core import task_manager

        return [scored(t.get("name"), {"id": t.get("id"), "status": t.get("status")})
                for t in task_manager.load_tasks()
                if t.get("status") in task_manager.ACTIVE_STATUSES]
    if kind == "meeting":
        from datetime import datetime, timedelta

        from tools.meeting_scheduler import calendar_is_configured, read_events

        if not calendar_is_configured():
            raise RuntimeError("no calendar is connected")
        from core.signals import run_blocking

        start = datetime.fromtimestamp(now)
        events = run_blocking(read_events(start, start + timedelta(days=14)), timeout=30)
        return [scored(e.get("title"), {"starts": e.get("starts"),
                                        "attendees": e.get("attendees")})
                for e in events]
    return []


# --------------------------------------------------------------------------- #
# Numbers, and where each one came from
# --------------------------------------------------------------------------- #

OBSERVED = "observed"          # counted directly from a store
CALCULATED = "calculated"      # arithmetic on observed values
ASSUMED = "assumed"            # the user or the config said so
INTERPRETED = "interpreted"    # a judgement, and labelled as one


def metrics(now: Optional[float] = None, stale_after_days: float = STALE_LEAD_DAYS
            ) -> Dict[str, Any]:
    """Business numbers, each one labelled with how it came to exist.

    The labelling is the feature. "You have 14 active leads" and "your conversion
    rate is 23%" and "the pipeline looks healthy" are three different kinds of
    claim, and a business tool that presents them identically is teaching its user
    to trust the third as much as the first.
    """
    now = now if now is not None else time.time()
    from tools.business import OPEN_STAGES, load_records, stage_weights

    try:
        records = load_records()
    except Exception as e:
        return {"problem": f"The business records could not be read: {e}",
                "note": "No figures are given rather than figures from nothing."}

    leads = records.get("lead", [])
    weights = stage_weights()
    stale_seconds = float(stale_after_days) * 86_400

    open_leads = [l for l in leads if l.get("stage") in OPEN_STAGES]
    won = [l for l in leads if l.get("stage") == "won"]
    lost = [l for l in leads if l.get("stage") == "lost"]
    quiet = [l for l in open_leads
             if now - float(l.get("stage_changed_at") or l.get("created_at") or now)
             > stale_seconds]

    from core import task_manager

    try:
        tasks = task_manager.load_tasks()
    except Exception:
        tasks = []
    completed_tasks = [t for t in tasks if t.get("status") == task_manager.COMPLETED]
    blocked_tasks = [t for t in tasks if t.get("status") == task_manager.WAITING_FOR_USER]

    observed = {
        "leads_total": len(leads),
        "leads_open": len(open_leads),
        "leads_won": len(won),
        "leads_lost": len(lost),
        "leads_quiet": len(quiet),
        "income_records": len(records.get("income", [])),
        "expense_records": len(records.get("expense", [])),
        "tasks_completed": len(completed_tasks),
        "tasks_waiting_on_you": len(blocked_tasks),
    }

    decided = len(won) + len(lost)
    calculated: Dict[str, Any] = {
        "pipeline_value": round(sum(float(l.get("value", 0) or 0) for l in open_leads), 2),
        "pipeline_weighted": round(
            sum(float(l.get("value", 0) or 0) * weights.get(l.get("stage", "new"), 0)
                for l in open_leads), 2),
    }
    if decided:
        calculated["conversion_rate_percent"] = round(len(won) / decided * 100, 1)
        calculated["conversion_counted_from"] = f"{len(won)} won of {decided} decided"
    else:
        calculated["conversion_rate_percent"] = None
        calculated["conversion_counted_from"] = (
            "no lead has been marked won or lost yet, so there is no rate to calculate")

    return {
        "as_of": time.strftime("%Y-%m-%d %H:%M", time.localtime(now)),
        OBSERVED: observed,
        CALCULATED: calculated,
        ASSUMED: {"stale_after_days": stale_after_days,
                  "stage_weights": weights,
                  "where_from": "Leti's defaults unless business.stage_weights is set"},
        "how_to_report": (
            "Say which is which. 'observed' was counted from the records; 'calculated' is "
            "arithmetic on them; 'assumed' is configuration, not fact; anything you "
            "conclude on top is interpretation and should be offered as that. A figure "
            "that is None is a figure Leti does not have - never fill one in."),
    }
