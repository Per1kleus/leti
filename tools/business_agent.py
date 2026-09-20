"""Business Mode's one new tool.

Leti already has the business layer: leads with stages and weighted values,
income, expenses, the metrics computed from them (tools/business.py), a contact
book, a calendar writer, mail, File Intelligence for proposals and invoices, the
data tools for spreadsheets, the task manager and the scheduler. Business Mode
mostly reaches for those.

What was missing is the thing that reads them together. "What should I be doing
today" is not answerable from any one of those stores - it is what is waiting on
you, what has gone quiet, what is running, what is about to run, and what Leti
cannot see at all. That is this tool, and it is the only new one: everything else
Business Mode does, it does with the tools that already exist.

It reads. It does not send mail, change a lead, book a meeting or create a task -
those are the existing tools, with the existing permissions, and being in
Business Mode does not change what any of them ask before doing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core import business, business_approvals, business_goals
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.business_agent")


class BusinessBriefingTool(BaseTool):
    name = "business_briefing"
    description = (
        "What the business needs from you, read across everything Leti already holds. "
        "Actions: brief (what needs attention, today's meetings, follow-ups, goals, what "
        "is running and what runs next), follow_ups (which leads are owed contact), "
        "pipeline (open leads by stage with the weighted figure), metrics (the numbers, "
        "each labelled observed / calculated / assumed), resolve (work out which customer, "
        "lead, goal, project, task or meeting a phrase like 'the customer' means - asks "
        "when several could match), workspace (set the business and objectives this "
        "session is about). Use brief for 'what should I be doing', 'catch me up'. Reads "
        "only: drafting, sending, updating a lead and booking anything are the ordinary "
        "tools and still ask before they act. Says which sources it cannot see instead of "
        "reporting them empty."
    )
    parameters = [
        ToolParameter(name="action", type="string",
                      description="brief, follow_ups, pipeline, metrics, resolve, workspace.",
                      enum=["brief", "follow_ups", "pipeline", "metrics", "resolve",
                            "workspace"]),
        ToolParameter(name="reference", type="string", required=False,
                      description="For resolve: the phrase to pin down, e.g. 'the customer'."),
        ToolParameter(name="kind", type="string", required=False,
                      description="For resolve: narrow to one kind of thing.",
                      enum=list(business.ENTITY_KINDS)),
        ToolParameter(name="business_name", type="string", required=False,
                      description="For workspace: which business this session is about."),
        ToolParameter(name="project", type="string", required=False,
                      description="For workspace: the project it belongs to."),
        ToolParameter(name="objectives", type="array", items_type="string", required=False,
                      description="For workspace: what the user is trying to achieve."),
        ToolParameter(name="stale_after_days", type="number", required=False,
                      description="For follow_ups: days of silence before a lead counts "
                                  "as owed contact (default 14)."),
    ]

    async def run(self, action: str, business_name: str = "", project: str = "",
                  objectives: Optional[List[str]] = None, stale_after_days: Any = None,
                  reference: str = "", kind: str = "", **kwargs) -> ToolResult:
        action = str(action or "brief").lower()

        if action == "metrics":
            return ToolResult(success=True, output=business.metrics())

        if action == "resolve":
            if not str(reference).strip():
                return ToolResult(success=False, error="Which phrase should Leti pin down?")
            found = business.resolve(reference, kind)
            if found.get("resolved"):
                return ToolResult(success=True, output=found)
            # Several matches, or none. Both are answers; neither is a licence to
            # pick one - acting on the wrong customer is worse than asking.
            return ToolResult(success=True, output={
                **found,
                "next": ("Ask the user which one before doing anything with it."
                         if found.get("candidates") else
                         "Say that nothing matches rather than choosing something close.")})

        if action == "workspace":
            space = business.open_workspace(business_name, project, objectives)
            return ToolResult(success=True, output={
                **space.summary(),
                "note": ("Held for this session only. Anything that should outlive it - a "
                         "task, a lead, a note on a project - goes in the place that "
                         "already keeps that kind of thing."),
            })

        if action == "brief":
            return ToolResult(success=True, output=business.briefing())

        if action == "follow_ups":
            try:
                days = float(stale_after_days) if stale_after_days else business.STALE_LEAD_DAYS
            except (TypeError, ValueError):
                days = business.STALE_LEAD_DAYS
            found = business.follow_ups(stale_after_days=days)
            if not found["due"]:
                found["note"] = ("No open lead has been quiet that long. That is the CRM's "
                                 "answer, not a guess - if leads are tracked somewhere "
                                 "else, that source is not connected.")
            return ToolResult(success=True, output=found)

        if action == "pipeline":
            brief = business.briefing()
            return ToolResult(success=True, output={
                "pipeline": brief["pipeline"],
                "leads_needing_attention": brief["leads_needing_attention"],
                "cannot_see": brief["cannot_see"],
            })

        return ToolResult(success=False,
                          error=f"'{action}' is not brief, follow_ups, pipeline or workspace.")


class BusinessCalendarTool(BaseTool):
    name = "business_calendar"
    description = (
        "Read the calendar: what is on today, tomorrow, or in a date range, and whether "
        "anything overlaps. Uses the CalDAV account already configured under Connections - "
        "the same one schedule_meeting writes to. Returns titles, times, durations, "
        "attendees and locations. If no calendar is connected, or it cannot be reached, it "
        "says so - that is never the same as having no meetings, and must not be reported "
        "as an empty day."
    )
    parameters = [
        ToolParameter(name="when", type="string", required=False,
                      description="today (default), tomorrow, week, or range.",
                      enum=["today", "tomorrow", "week", "range"]),
        ToolParameter(name="start", type="string", required=False,
                      description="For range: first day, YYYY-MM-DD."),
        ToolParameter(name="end", type="string", required=False,
                      description="For range: last day, YYYY-MM-DD."),
    ]

    async def run(self, when: str = "today", start: str = "", end: str = "",
                  **kwargs) -> ToolResult:
        from datetime import datetime, timedelta

        from tools.meeting_scheduler import (MAX_RANGE_DAYS, calendar_is_configured,
                                             overlapping, read_events)

        if not calendar_is_configured():
            return ToolResult(success=False, error=(
                "No calendar is connected, so Leti cannot see any meetings. Add a CalDAV "
                "calendar under Connections. Do not report this as an empty calendar - "
                "Leti does not know what is in it."))

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        when = str(when or "today").lower()
        if when == "range":
            try:
                first = datetime.fromisoformat(start) if start else today
                last = (datetime.fromisoformat(end) if end else first) + timedelta(days=1)
            except ValueError:
                return ToolResult(success=False,
                                  error="Dates should look like 2026-09-21.")
            if (last - first).days > MAX_RANGE_DAYS:
                return ToolResult(success=False, error=(
                    f"That is more than {MAX_RANGE_DAYS} days. Ask for a shorter window."))
        elif when == "tomorrow":
            first, last = today + timedelta(days=1), today + timedelta(days=2)
        elif when == "week":
            first, last = today, today + timedelta(days=7)
        else:
            first, last = today, today + timedelta(days=1)

        try:
            events = await read_events(first, last)
        except Exception as e:
            return ToolResult(success=False, error=(
                f"Leti could not read the calendar: {e}. That is a failure to look, not "
                "an empty calendar - say so rather than reporting no meetings."))

        return ToolResult(success=True, output={
            "from": first.strftime("%Y-%m-%d"), "to": last.strftime("%Y-%m-%d"),
            "count": len(events), "events": events,
            "conflicts": overlapping(events),
            "note": ("Read from the connected CalDAV calendar."
                     if events else
                     "The calendar was read and there is nothing in that window."),
        })


class BusinessGoalsTool(BaseTool):
    name = "business_goals"
    description = (
        "Business goals, and the projects and tasks pursued for them. Goal -> project -> "
        "task -> result. Actions: create (name, and ideally a measure and target_value so "
        "progress can be checked), list, show, link (attach an existing project name or "
        "task id), result (record something measured), status (active, paused, completed, "
        "blocked), remove. Progress is only ever reported from a measured result against a "
        "target, or from how many linked tasks are done - a goal with neither says its "
        "progress cannot be determined rather than inventing a percentage."
    )
    parameters = [
        ToolParameter(name="action", type="string",
                      description="create, list, show, link, result, status, or remove.",
                      enum=["create", "list", "show", "link", "result", "status", "remove"]),
        ToolParameter(name="goal_id", type="string", required=False,
                      description="Which goal, for everything but create and list."),
        ToolParameter(name="name", type="string", required=False,
                      description="For create: what is being achieved."),
        ToolParameter(name="measure", type="string", required=False,
                      description="For create: what is counted, e.g. 'qualified leads'."),
        ToolParameter(name="target_value", type="number", required=False,
                      description="For create: the number that counts as done."),
        ToolParameter(name="due", type="string", required=False,
                      description="For create: when it should be achieved by."),
        ToolParameter(name="project", type="string", required=False,
                      description="For link: an existing project's name."),
        ToolParameter(name="task_id", type="string", required=False,
                      description="For link: an existing task's id."),
        ToolParameter(name="value", type="number", required=False,
                      description="For result: the measured number."),
        ToolParameter(name="note", type="string", required=False,
                      description="For result or status: what happened, in a sentence."),
        ToolParameter(name="status", type="string", required=False,
                      description="For status: the new state.",
                      enum=list(business_goals.STATUSES)),
    ]

    def action_case(self, arguments: Dict[str, Any]) -> str:
        action = str((arguments or {}).get("action", "")).lower()
        return "read" if action in ("list", "show") else "write"

    async def run(self, action: str, goal_id: str = "", name: str = "", measure: str = "",
                  target_value: Any = None, due: str = "", project: str = "",
                  task_id: str = "", value: Any = None, note: str = "", status: str = "",
                  **kwargs) -> ToolResult:
        action = str(action or "list").lower()
        try:
            if action == "create":
                goal = business_goals.create(name, target=note, measure=measure,
                                             target_value=target_value, due=due)
                return ToolResult(success=True, output={
                    "goal": business_goals.describe(goal),
                    "next": ("Link the projects and tasks that pursue it, and record "
                             "results as they happen - that is what makes progress real."),
                })
            if action == "list":
                goals = [business_goals.describe(g) for g in business_goals.load_goals()]
                return ToolResult(success=True, output={
                    "goals": goals, "count": len(goals),
                    "needs_attention": business_goals.needs_attention(),
                })

            if not goal_id:
                return ToolResult(success=False, error="Which goal? Use list to find its id.")
            goal = business_goals.get(goal_id)
            if goal is None:
                return ToolResult(success=False, error=f"No goal with id '{goal_id}'.")

            if action == "show":
                return ToolResult(success=True, output=business_goals.describe(goal))
            if action == "link":
                if not (project or task_id):
                    return ToolResult(success=False,
                                      error="Link what - a project name or a task id?")
                updated = business_goals.link(goal_id, project=project, task_id=task_id)
                return ToolResult(success=True, output=business_goals.describe(updated))
            if action == "result":
                updated = business_goals.record_result(goal_id, value=value, note=note)
                return ToolResult(success=True, output=business_goals.describe(updated))
            if action == "status":
                updated = business_goals.set_status(goal_id, status, reason=note)
                return ToolResult(success=True, output=business_goals.describe(updated))
            if action == "remove":
                return ToolResult(success=True, output={"removed": business_goals.delete(goal_id)})
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        return ToolResult(success=False, error=f"'{action}' is not a business_goals action.")


class BusinessBatchTool(BaseTool):
    name = "business_batch"
    description = (
        "Prepare several business actions, show them, and do only the ones approved. For "
        "'follow up with every lead nobody has contacted for a month' - propose the list "
        "first so the user can see eight emails before eight emails happen. Actions: "
        "propose (build the list; NOTHING is done), decide (approve or reject by number), "
        "pending (what is approved and still to do), outcome (record what a tool actually "
        "did), report, cancel. This does not send, write or authorise anything: each "
        "approved action is then performed by its own ordinary tool and still asks for "
        "permission exactly where it always did."
    )
    parameters = [
        ToolParameter(name="action", type="string",
                      description="propose, decide, pending, outcome, report, or cancel.",
                      enum=["propose", "decide", "pending", "outcome", "report", "cancel"]),
        ToolParameter(name="what", type="string", required=False,
                      description="For propose: what this batch is for, in one line."),
        ToolParameter(name="items", type="array", required=False,
                      description="For propose: the actions, each {kind, about, ...}. "
                                  "kind is send_email, update_lead, schedule_meeting, "
                                  "create_task or record_result. Set blocked_because on "
                                  "one that cannot be done, e.g. no email address."),
        ToolParameter(name="batch", type="string", required=False,
                      description="Which batch, for everything but propose."),
        ToolParameter(name="approve", type="array", items_type="number", required=False,
                      description="For decide: the item numbers the user agreed to."),
        ToolParameter(name="reject", type="array", items_type="number", required=False,
                      description="For decide: the numbers they turned down."),
        ToolParameter(name="number", type="number", required=False,
                      description="For outcome: which item."),
        ToolParameter(name="succeeded", type="boolean", required=False,
                      description="For outcome: whether the tool actually did it."),
        ToolParameter(name="detail", type="string", required=False,
                      description="For outcome: what the tool said."),
    ]

    async def run(self, action: str, what: str = "", items: Any = None, batch: str = "",
                  approve: Any = None, reject: Any = None, number: Any = None,
                  succeeded: bool = False, detail: str = "", **kwargs) -> ToolResult:
        action = str(action or "").lower()

        if action == "propose":
            if not items:
                return ToolResult(success=False, error="Propose what? Give the items.")
            found = business_approvals.propose(what, list(items))
            return ToolResult(success=True, output=found)

        if not batch:
            return ToolResult(success=False, error="Which batch? propose one first.")

        if action == "decide":
            result = business_approvals.decide(
                batch, approve=[int(n) for n in (approve or [])],
                reject=[int(n) for n in (reject or [])])
        elif action == "pending":
            result = {"batch": batch, "to_do": business_approvals.approved_items(batch),
                      "note": ("Perform these with the ordinary tools, one at a time, then "
                               "record each outcome. Approval is not permission - each one "
                               "still asks where it always did.")}
        elif action == "outcome":
            if number is None:
                return ToolResult(success=False, error="Which item number?")
            result = business_approvals.record_outcome(batch, int(number), bool(succeeded),
                                                       detail)
        elif action == "report":
            result = business_approvals.report(batch)
        elif action == "cancel":
            result = business_approvals.cancel(batch)
        else:
            return ToolResult(success=False, error=f"'{action}' is not a business_batch action.")

        if isinstance(result, dict) and result.get("error"):
            return ToolResult(success=False, error=result["error"])
        return ToolResult(success=True, output=result)
