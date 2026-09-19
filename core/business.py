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

    The stores are untouched: leads, tasks, contacts, projects and scheduled work
    are exactly where they were, because none of them belonged to the workspace.
    """
    global _workspace

    _workspace = None


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
    # Leti can CREATE calendar events over CalDAV. It has no tool that reads one
    # back, so a briefing cannot show today's meetings and says so rather than
    # implying an empty calendar.
    sources["calendar"] = (
        "write only - Leti can create events but has no tool that reads a calendar back, "
        "so it cannot list your meetings"
        if settings.get("calendar", {}).get("caldav_url")
        else "not connected - add it under Connections to create events")
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
                         ("scheduled", _scheduled_soon)):
        try:
            sections[name] = gather(now)
        except Exception as e:
            sections[name] = []
            gaps.append(f"{name}: {e}")
            logger.debug(f"Briefing section {name} unavailable: {e}")

    space = _workspace
    return {
        "as_of": time.strftime("%Y-%m-%d %H:%M", time.localtime(now)),
        "workspace": space.summary() if space else None,
        **sections,
        "cannot_see": _cannot_see(),
        "gaps": gaps or None,
        "how_to_report": (
            "Lead with what is waiting on the user, then what has gone quiet. Keep "
            "observed figures separate from anything you worked out, and say when a "
            "section is empty because there is nothing there rather than because Leti "
            "cannot see it - 'cannot_see' lists the difference."),
    }


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
