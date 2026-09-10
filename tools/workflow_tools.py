"""Tools for describing a workflow in words and then running it.

A workflow is a standing instruction, so these deliberately do not create one and
switch it on in a single move: create_workflow validates and shows the user what
Leti understood, and activate_workflow is the separate, explicit yes. Between
those two the workflow exists and does nothing.

Running one goes back through the task manager and therefore back through the
orchestrator, the router and the safety guard. Nothing here executes a tool.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core import task_manager, workflows
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.workflows")

_REGISTRY: Optional[Any] = None
_RUNNER: Optional[Any] = None


def set_context(registry=None, runner=None) -> None:
    """main.py wires the live registry (for validation) and task runner."""
    global _REGISTRY, _RUNNER
    if registry is not None:
        _REGISTRY = registry
    if runner is not None:
        _RUNNER = runner


class CreateWorkflowTool(BaseTool):
    name = "create_workflow"
    description = (
        "Turn a described routine into a saved workflow: a trigger and an ordered list "
        "of steps. Use this when the user describes something that should happen "
        "repeatedly or on a schedule - 'every weekday at 9 check my calendar and email "
        "and tell me what needs attention', 'every Sunday summarise my week'. This only "
        "PROPOSES the workflow: it is validated and shown back to the user, and does "
        "nothing at all until they activate it. Give plain-language steps, one "
        "instruction each. For a schedule use schedule_type daily/weekly/monthly/once/"
        "interval with a time like '09:00' and, for weekly, the weekdays."
    )
    parameters = [
        ToolParameter(name="name", type="string",
                      description="Short name, e.g. 'Morning briefing'."),
        ToolParameter(name="steps", type="array", items_type="string",
                      description="Ordered plain-language steps, one instruction each."),
        ToolParameter(name="schedule_type", type="string", required=False,
                      description="daily, weekly, monthly, once, interval, or blank for manual.",
                      enum=["daily", "weekly", "monthly", "once", "interval"]),
        ToolParameter(name="at", type="string", required=False,
                      description="Time of day as HH:MM, e.g. '09:00'."),
        ToolParameter(name="weekdays", type="array", items_type="string", required=False,
                      description="For weekly: monday, tuesday, ... "),
        ToolParameter(name="every_minutes", type="number", required=False,
                      description="For interval schedules."),
        ToolParameter(name="conditions", type="array", items_type="string", required=False,
                      description="Plain-language conditions that must hold, if any."),
        ToolParameter(name="description", type="string", required=False,
                      description="One line on what this workflow is for."),
    ]

    async def run(self, name: str, steps: List[str], schedule_type: str = "",
                  at: str = "09:00", weekdays: Optional[List[str]] = None,
                  every_minutes: Optional[float] = None,
                  conditions: Optional[List[str]] = None,
                  description: str = "", **kwargs) -> ToolResult:
        schedule: Dict[str, Any] = {}
        trigger = workflows.TRIGGER_MANUAL
        if schedule_type:
            trigger = workflows.TRIGGER_SCHEDULE
            schedule = {"schedule_type": schedule_type, "at": at or "09:00"}
            if weekdays:
                schedule["weekdays"] = [str(d).lower() for d in weekdays]
            if every_minutes:
                schedule["every_minutes"] = float(every_minutes)

        try:
            workflow = workflows.build(name, steps, trigger, schedule, conditions, description)
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

        errors, questions = workflows.validate(workflow, _REGISTRY)
        if errors:
            # Never saved in a broken state; the model is told exactly what to fix.
            return ToolResult(success=False, error="That workflow isn't valid yet.",
                              output={"problems": errors, "questions": questions})

        workflows.save_draft(workflow)
        return ToolResult(success=True, output={
            "workflow_id": workflow["id"],
            "preview": workflows.preview(workflow),
            "questions": questions,
            "activated": False,
            "note": ("Ask the user these questions before activating."
                     if questions else
                     "Show the user this preview and ask whether to activate it. "
                     "Nothing runs until you call activate_workflow."),
        })


class ActivateWorkflowTool(BaseTool):
    name = "activate_workflow"
    description = (
        "Switch on a workflow the user has agreed to, after they have seen its preview. "
        "A scheduled workflow is handed to Leti's scheduler and will start running at "
        "its next due time. Only call this when the user has actually said yes."
    )
    parameters = [
        ToolParameter(name="workflow_id", type="string",
                      description="The id create_workflow returned."),
    ]

    async def run(self, workflow_id: str, **kwargs) -> ToolResult:
        result = workflows.activate(workflow_id, _REGISTRY)
        if not result.get("ok"):
            return ToolResult(success=False, error=result.get("error"),
                              output={k: v for k, v in result.items() if k != "error"})
        return ToolResult(success=True, output=result["workflow"])


class ListWorkflowsTool(BaseTool):
    name = "list_workflows"
    description = (
        "Show the user's workflows: what each one does, when it runs next, and whether "
        "it is active or switched off. Use for 'show my workflows', 'what's running "
        "automatically', or before changing or deleting one."
    )
    parameters = []

    async def run(self, **kwargs) -> ToolResult:
        all_workflows = workflows.load_workflows()
        described = [workflows.describe(w) for w in all_workflows]
        return ToolResult(success=True, output={
            "count": len(described),
            "active": [w for w in described if w["enabled"]],
            "inactive": [w for w in described if not w["enabled"]],
        })


class ManageWorkflowTool(BaseTool):
    name = "manage_workflow"
    description = (
        "Change a workflow: switch it off or back on, move it to a different time, or "
        "delete it. Use for 'disable the morning briefing', 'change it to 8 AM', "
        "'delete that workflow'. Find the id with list_workflows first."
    )
    parameters = [
        ToolParameter(name="workflow_id", type="string", description="Which workflow."),
        ToolParameter(name="action", type="string",
                      description="enable, disable, delete, or reschedule.",
                      enum=["enable", "disable", "delete", "reschedule"]),
        ToolParameter(name="schedule_type", type="string", required=False,
                      description="For reschedule: daily, weekly, monthly, once, interval.",
                      enum=["daily", "weekly", "monthly", "once", "interval"]),
        ToolParameter(name="at", type="string", required=False,
                      description="For reschedule: time of day as HH:MM."),
        ToolParameter(name="weekdays", type="array", items_type="string", required=False,
                      description="For reschedule, weekly: monday, tuesday, ..."),
    ]

    async def run(self, workflow_id: str, action: str, schedule_type: str = "",
                  at: str = "09:00", weekdays: Optional[List[str]] = None,
                  **kwargs) -> ToolResult:
        action = str(action or "").strip().lower()

        if action in ("enable", "disable"):
            result = workflows.set_enabled(workflow_id, action == "enable")
        elif action == "delete":
            result = workflows.delete(workflow_id)
        elif action == "reschedule":
            schedule = {"schedule_type": schedule_type or "daily", "at": at or "09:00"}
            if weekdays:
                schedule["weekdays"] = [str(d).lower() for d in weekdays]
            result = workflows.update_schedule(workflow_id, schedule, _REGISTRY)
        else:
            return ToolResult(success=False,
                              error=f"'{action}' is not enable, disable, delete or reschedule.")

        if not result.get("ok"):
            return ToolResult(success=False, error=result.get("error"),
                              output={k: v for k, v in result.items() if k != "error"})
        return ToolResult(success=True, output=result)


class RunWorkflowTool(BaseTool):
    name = "run_workflow"
    description = (
        "Run a workflow now, rather than waiting for its schedule. Its steps become a "
        "long-running task, so it reports progress and can be paused like any other. "
        "This is also what Leti's scheduler calls when a scheduled workflow comes due."
    )
    parameters = [
        ToolParameter(name="workflow_id", type="string", description="Which workflow to run."),
    ]

    async def run(self, workflow_id: str, **kwargs) -> ToolResult:
        workflow = workflows.get_workflow(workflow_id)
        if workflow is None:
            return ToolResult(success=False, error=f"No workflow with id '{workflow_id}'.")

        errors, _ = workflows.validate(workflow, _REGISTRY)
        if errors:
            return ToolResult(success=False, error="That workflow isn't valid.",
                              output={"problems": errors})

        steps = workflows.to_task_steps(workflow)
        if not steps:
            return ToolResult(success=False, error="That workflow has no steps to run.")

        # Straight into the task manager, which runs each step through the
        # orchestrator - the same route a typed request takes.
        task = task_manager.create_task(
            objective=f"Run the workflow '{workflow['name']}'",
            steps=steps, name=workflow["name"])
        started = _RUNNER.start_in_background(task["id"]) if _RUNNER is not None else False
        return ToolResult(success=True, output={
            "workflow": workflow["name"],
            "task": task_manager.describe(task),
            "started": started,
        })
