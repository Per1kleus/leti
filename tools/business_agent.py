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

from core import business, modes
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.business_agent")


class BusinessBriefingTool(BaseTool):
    name = "business_briefing"
    description = (
        "What the business needs from you, read across everything Leti already holds. "
        "Actions: brief (what is waiting on you, what has gone quiet, what is running and "
        "what runs next), follow_ups (which leads are owed contact and who can be "
        "reached), pipeline (open leads by stage, with the weighted figure), workspace "
        "(set the business, project and objectives this session is about). Use brief for "
        "'what should I be doing', 'how are we doing', 'catch me up'. It reads only - "
        "drafting, sending, updating a lead and booking anything are the ordinary tools "
        "and still ask before they act. Says which sources it cannot see instead of "
        "reporting them as empty."
    )
    parameters = [
        ToolParameter(name="action", type="string",
                      description="brief, follow_ups, pipeline, or workspace.",
                      enum=["brief", "follow_ups", "pipeline", "workspace"]),
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
                  **kwargs) -> ToolResult:
        action = str(action or "brief").lower()

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
