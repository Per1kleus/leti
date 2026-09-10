"""Workflows: a trigger, some conditions, and an ordered list of things to do.

"Every Monday morning, check my calendar and summarise the week" is a shape, not a
sentence, and once it has been written down as a shape it should never need the
model again to run. The model may build it; running it is deterministic from the
structure.

    trigger -> conditions -> steps -> task manager -> orchestrator -> tools

Two triggers exist, and neither is a new mechanism:

  schedule  registered with the scheduler Leti already has (tools/scheduler.py),
            which already persists, computes next runs, retries, disables a task
            that keeps failing and notifies. There is no second scheduler here.
  manual    run when the user asks.

A workflow is inert until it is activated, and validation runs before that: every
tool it names has to exist in the registry, its parameters have to be present, its
schedule has to be one the scheduler understands, and its steps have to be in an
order that can actually happen. An invalid workflow is never activated, and an
ambiguous one is reported so the user can be asked rather than guessed at.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.atomic_write import atomic_write_text
from core.config_loader import resolve_path

logger = logging.getLogger("leti.workflows")

STORE_PATH = "./data/workflows.json"
MAX_STEPS = 20

TRIGGER_SCHEDULE = "schedule"
TRIGGER_MANUAL = "manual"
TRIGGERS = (TRIGGER_SCHEDULE, TRIGGER_MANUAL)

# The scheduler's own vocabulary, reused rather than reinvented so that a workflow
# schedule and a scheduled task mean exactly the same thing.
SCHEDULE_TYPES = ("once", "interval", "daily", "weekly", "monthly")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def store_path():
    return resolve_path(STORE_PATH)


def load_workflows() -> List[Dict[str, Any]]:
    path = store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Couldn't read {path} ({e}); starting with no workflows.")
        return []
    return data if isinstance(data, list) else []


def save_workflows(workflows: List[Dict[str, Any]]) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(workflows, indent=2))


def get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
    return next((w for w in load_workflows() if w.get("id") == workflow_id), None)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def build(name: str, steps: List[Any], trigger: str = TRIGGER_MANUAL,
          schedule: Optional[Dict[str, Any]] = None,
          conditions: Optional[List[str]] = None,
          description: str = "", project: str = "") -> Dict[str, Any]:
    """A workflow record. Draft until it is validated and activated."""
    normalised = []
    for i, step in enumerate(steps or []):
        if isinstance(step, str):
            step = {"instruction": step}
        if not isinstance(step, dict):
            raise ValueError(f"Step {i + 1} is not a step.")
        normalised.append({
            "n": i + 1,
            "instruction": str(step.get("instruction", "")).strip(),
            # Naming a tool is optional. When it is named it is checked against the
            # registry, which is what makes "use a tool that doesn't exist" a
            # validation error rather than a puzzling failure at run time.
            "tool": step.get("tool") or None,
            "parameters": step.get("parameters") or {},
            "depends_on": step.get("depends_on") or [],
            "on_error": step.get("on_error") or "stop",
        })

    now = time.time()
    return {
        "id": uuid.uuid4().hex[:12],
        "name": (name or "workflow")[:80],
        "description": description,
        "project": project or None,
        "trigger": trigger,
        "schedule": schedule or {},
        "conditions": [str(c) for c in (conditions or [])],
        "steps": normalised,
        "enabled": False,           # inert until explicitly activated
        "activated": False,
        "scheduled_task_id": None,
        "created_at": now,
        "updated_at": now,
        "last_run_at": None,
        "last_status": None,
    }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate(workflow: Dict[str, Any], registry: Any = None) -> Tuple[List[str], List[str]]:
    """(errors, questions). Errors block activation; questions need the user.

    Questions are deliberately separate from errors: "notify me" is not wrong, it
    is under-specified, and the answer is to ask rather than to pick one.
    """
    errors: List[str] = []
    questions: List[str] = []

    if not str(workflow.get("name", "")).strip():
        errors.append("The workflow has no name.")

    steps = workflow.get("steps") or []
    if not steps:
        errors.append("A workflow needs at least one step.")
    if len(steps) > MAX_STEPS:
        errors.append(f"That is {len(steps)} steps; the limit is {MAX_STEPS}.")

    trigger = workflow.get("trigger")
    if trigger not in TRIGGERS:
        errors.append(f"'{trigger}' is not a trigger. Use one of: {', '.join(TRIGGERS)}.")

    if trigger == TRIGGER_SCHEDULE:
        errors.extend(_schedule_errors(workflow.get("schedule") or {}))

    seen: set = set()
    for step in steps:
        n = step.get("n")
        if not str(step.get("instruction", "")).strip() and not step.get("tool"):
            errors.append(f"Step {n} says nothing to do.")

        tool_name = step.get("tool")
        if tool_name:
            tool = registry.get(tool_name) if registry is not None else None
            if registry is not None and tool is None:
                errors.append(f"Step {n} uses '{tool_name}', which is not a registered tool.")
            elif tool is not None:
                missing = [p.name for p in getattr(tool, "parameters", [])
                           if getattr(p, "required", False)
                           and p.name not in (step.get("parameters") or {})]
                if missing:
                    errors.append(
                        f"Step {n} calls '{tool_name}' without required "
                        f"{'parameters' if len(missing) > 1 else 'parameter'}: "
                        f"{', '.join(missing)}.")

        for dep in step.get("depends_on") or []:
            if dep not in seen:
                errors.append(
                    f"Step {n} depends on step {dep}, which does not come before it.")
        seen.add(n)

        text = str(step.get("instruction", "")).lower()
        if "notify" in text and not any(
                word in text for word in ("email", "desktop", "notification", "message", "leti")):
            questions.append(
                f"Step {n} says 'notify me' - do you mean a Leti desktop notification "
                "or an email?")

    return errors, questions


def _schedule_errors(schedule: Dict[str, Any]) -> List[str]:
    errors = []
    kind = schedule.get("schedule_type")
    if kind not in SCHEDULE_TYPES:
        return [f"'{kind}' is not a schedule. Use one of: {', '.join(SCHEDULE_TYPES)}."]

    at = str(schedule.get("at", "09:00"))
    if kind in ("daily", "weekly", "monthly"):
        try:
            hour, _, minute = at.partition(":")
            if not (0 <= int(hour) <= 23 and 0 <= int(minute or 0) <= 59):
                raise ValueError
        except ValueError:
            errors.append(f"'{at}' is not a time of day. Use HH:MM.")
    if kind == "weekly":
        days = [str(d).lower() for d in (schedule.get("weekdays") or [])]
        unknown = [d for d in days if d not in WEEKDAYS]
        if not days:
            errors.append("A weekly schedule needs at least one weekday.")
        if unknown:
            errors.append(f"Not weekdays: {', '.join(unknown)}.")
    if kind == "once" and not schedule.get("run_at"):
        errors.append("A one-off schedule needs a time to run at.")
    if kind == "interval" and float(schedule.get("every_minutes", 0) or 0) < 1:
        errors.append("An interval schedule needs every_minutes of at least 1.")
    return errors


# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #

def preview(workflow: Dict[str, Any]) -> str:
    """What Leti understood, in the user's words rather than the structure's.

    Shown before activation because a workflow that sends mail every morning is a
    standing instruction, and a misread one is a standing mistake.
    """
    lines = [f"Workflow: {workflow.get('name')}"]
    if workflow.get("description"):
        lines.append(workflow["description"])
    lines.append("")
    lines.append(f"When: {describe_trigger(workflow)}")
    for condition in workflow.get("conditions") or []:
        lines.append(f"Only if: {condition}")
    lines.append("")
    lines.append("Then:")
    for step in workflow.get("steps") or []:
        label = step.get("instruction") or f"run {step.get('tool')}"
        suffix = f"  (tool: {step['tool']})" if step.get("tool") else ""
        lines.append(f"  {step['n']}. {label}{suffix}")
    lines.append("")
    lines.append("Nothing runs until you activate it.")
    return "\n".join(lines)


def describe_trigger(workflow: Dict[str, Any]) -> str:
    if workflow.get("trigger") != TRIGGER_SCHEDULE:
        return "when you ask for it"
    schedule = workflow.get("schedule") or {}
    kind = schedule.get("schedule_type")
    at = schedule.get("at", "09:00")
    if kind == "daily":
        return f"every day at {at}"
    if kind == "weekly":
        days = ", ".join(d.capitalize() for d in (schedule.get("weekdays") or []))
        return f"every {days} at {at}"
    if kind == "monthly":
        return f"on day {schedule.get('day_of_month', 1)} of each month at {at}"
    if kind == "interval":
        return f"every {schedule.get('every_minutes')} minutes"
    if kind == "once":
        return f"once at {time.strftime('%Y-%m-%d %H:%M', time.localtime(schedule.get('run_at', 0)))}"
    return "on a schedule"


# --------------------------------------------------------------------------- #
# Activation - where a scheduled workflow becomes a scheduled task
# --------------------------------------------------------------------------- #

def save_draft(workflow: Dict[str, Any]) -> Dict[str, Any]:
    workflows = load_workflows()
    workflows = [w for w in workflows if w["id"] != workflow["id"]]
    workflows.append(workflow)
    save_workflows(workflows)
    return workflow


def activate(workflow_id: str, registry: Any = None) -> Dict[str, Any]:
    """Validate, then turn it on. A scheduled one is handed to the scheduler."""
    workflow = get_workflow(workflow_id)
    if workflow is None:
        return {"ok": False, "error": f"No workflow with id '{workflow_id}'."}

    errors, questions = validate(workflow, registry)
    if errors:
        return {"ok": False, "error": "This workflow isn't valid yet.",
                "problems": errors, "questions": questions}
    if questions:
        return {"ok": False, "error": "This workflow is ambiguous.",
                "problems": [], "questions": questions}

    if workflow["trigger"] == TRIGGER_SCHEDULE:
        task_id = _register_with_scheduler(workflow)
        if not task_id:
            return {"ok": False, "error": "Couldn't register the schedule."}
        workflow["scheduled_task_id"] = task_id

    workflow["enabled"] = True
    workflow["activated"] = True
    workflow["updated_at"] = time.time()
    save_draft(workflow)
    return {"ok": True, "workflow": describe(workflow)}


def _register_with_scheduler(workflow: Dict[str, Any]) -> Optional[str]:
    """One scheduled task that runs this workflow. The scheduler is the existing
    one; this only adds a row to its store."""
    from tools import scheduler

    schedule = dict(workflow.get("schedule") or {})
    task = {
        "id": uuid.uuid4().hex[:12],
        "name": workflow["name"],
        # run_workflow is a registered tool like any other, so the run goes through
        # tool routing and the safety guard exactly as a typed request would.
        "instruction": f"Run the workflow '{workflow['name']}' (id {workflow['id']}).",
        "workflow_id": workflow["id"],
        "schedule_type": schedule.get("schedule_type", "daily"),
        "at": schedule.get("at", "09:00"),
        "weekdays": schedule.get("weekdays"),
        "day_of_month": schedule.get("day_of_month"),
        "every_minutes": schedule.get("every_minutes"),
        "run_at": schedule.get("run_at"),
        "enabled": True,
        "created_at": time.time(),
        "history": [],
    }
    task["next_run"] = scheduler.compute_next_run(task)
    tasks = scheduler.load_tasks()
    tasks.append(task)
    scheduler.save_tasks(tasks)
    return task["id"]


def set_enabled(workflow_id: str, enabled: bool) -> Dict[str, Any]:
    """Turn a workflow on or off, taking its scheduled task with it."""
    workflow = get_workflow(workflow_id)
    if workflow is None:
        return {"ok": False, "error": f"No workflow with id '{workflow_id}'."}
    workflow["enabled"] = bool(enabled)
    workflow["updated_at"] = time.time()
    save_draft(workflow)
    _mirror_to_scheduler(workflow, enabled=bool(enabled))
    return {"ok": True, "workflow": describe(workflow)}


def delete(workflow_id: str) -> Dict[str, Any]:
    workflow = get_workflow(workflow_id)
    if workflow is None:
        return {"ok": False, "error": f"No workflow with id '{workflow_id}'."}
    _mirror_to_scheduler(workflow, remove=True)
    save_workflows([w for w in load_workflows() if w["id"] != workflow_id])
    return {"ok": True, "deleted": workflow["name"]}


def update_schedule(workflow_id: str, schedule: Dict[str, Any],
                    registry: Any = None) -> Dict[str, Any]:
    """"Change it to 8 AM" - revalidated, and the scheduler row moved with it."""
    workflow = get_workflow(workflow_id)
    if workflow is None:
        return {"ok": False, "error": f"No workflow with id '{workflow_id}'."}

    candidate = dict(workflow)
    candidate["trigger"] = TRIGGER_SCHEDULE
    candidate["schedule"] = schedule
    errors, _ = validate(candidate, registry)
    if errors:
        return {"ok": False, "error": "That schedule isn't valid.", "problems": errors}

    _mirror_to_scheduler(workflow, remove=True)
    workflow["trigger"] = TRIGGER_SCHEDULE
    workflow["schedule"] = schedule
    workflow["updated_at"] = time.time()
    if workflow.get("activated"):
        workflow["scheduled_task_id"] = _register_with_scheduler(workflow)
    save_draft(workflow)
    return {"ok": True, "workflow": describe(workflow)}


def _mirror_to_scheduler(workflow: Dict[str, Any], enabled: Optional[bool] = None,
                         remove: bool = False) -> None:
    task_id = workflow.get("scheduled_task_id")
    if not task_id:
        return
    try:
        from tools import scheduler

        tasks = scheduler.load_tasks()
        if remove:
            scheduler.save_tasks([t for t in tasks if t.get("id") != task_id])
            return
        for task in tasks:
            if task.get("id") == task_id and enabled is not None:
                task["enabled"] = enabled
                if enabled and not task.get("next_run"):
                    task["next_run"] = scheduler.compute_next_run(task)
        scheduler.save_tasks(tasks)
    except Exception:
        logger.exception("Couldn't update the workflow's scheduled task")


def describe(workflow: Dict[str, Any]) -> Dict[str, Any]:
    next_run = None
    task_id = workflow.get("scheduled_task_id")
    if task_id:
        try:
            from tools import scheduler

            task = next((t for t in scheduler.load_tasks() if t.get("id") == task_id), None)
            if task and task.get("next_run"):
                next_run = time.strftime("%Y-%m-%d %H:%M", time.localtime(task["next_run"]))
        except Exception:
            pass
    return {
        "id": workflow["id"],
        "name": workflow.get("name"),
        "when": describe_trigger(workflow),
        "steps": len(workflow.get("steps") or []),
        "project": workflow.get("project"),
        "enabled": bool(workflow.get("enabled")),
        "activated": bool(workflow.get("activated")),
        "next_run": next_run,
        "last_run": (time.strftime("%Y-%m-%d %H:%M", time.localtime(workflow["last_run_at"]))
                     if workflow.get("last_run_at") else None),
        "last_status": workflow.get("last_status"),
    }


def to_task_steps(workflow: Dict[str, Any]) -> List[str]:
    """The workflow's steps as plain instructions for the task manager.

    This is the join between the two systems, and it is deliberately dumb: a step
    that names a tool says so in words, so the request still travels the ordinary
    route through the router and the guard rather than around it.
    """
    steps = []
    for step in workflow.get("steps") or []:
        text = step.get("instruction") or ""
        if step.get("tool"):
            params = step.get("parameters") or {}
            text = (f"{text} (use the {step['tool']} tool"
                    + (f" with {json.dumps(params)}" if params else "") + ")").strip()
        steps.append(text)
    return steps
