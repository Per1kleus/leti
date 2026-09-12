"""The three systems as one path, and the safety properties that hold across it.

    workflow -> task manager -> orchestrator -> tool router -> SafetyGuard -> tool

Every test here is about that chain not having a shortcut in it. The new tools
manage state; none of them runs a tool, and none of them decides whether a call is
allowed.
"""
from __future__ import annotations

import ast
import inspect
import sys
import types
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

import main  # noqa: E402
from core import task_manager, tool_router, workflows  # noqa: E402
from tools import autonomous, scheduler, workflow_tools  # noqa: E402

NEW_TOOLS = ("start_autonomous_task", "list_autonomous_tasks", "control_autonomous_task",
             "create_workflow", "activate_workflow", "list_workflows",
             "manage_workflow", "run_workflow")


@pytest.fixture(scope="module")
def registry():
    sys.argv = ["main.py"]
    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(workflows, "store_path", lambda: tmp_path / "workflows.json")
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "scheduled.json")
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")


class FakeOrchestrator:
    def __init__(self):
        self.seen = []

    # **kwargs because the real orchestrator takes voice_mode and preapproved;
    # a double that refuses them stops doubling the thing it stands in for.
    async def handle_user_input(self, text, **kwargs):
        self.seen.append(text)
        return "done"


# --- Registration and routing --------------------------------------------------------

def test_the_new_tools_are_registered_like_any_other(registry):
    for name in NEW_TOOLS:
        assert registry.get(name) is not None, f"{name} is not registered"


def test_nothing_that_existed_before_was_removed(registry):
    """Nothing is traded away when something is added. The count only goes up:
    103 originally, 111 with the task and workflow tools, 119 with watches and the
    controlled GUI mode."""
    assert len(registry.names()) >= 119, f"the registry lost tools: {len(registry.names())}"
    for existing in ("read_file", "send_email", "run_shell_command", "web_search",
                     "create_scheduled_task", "read_screen", "kill_process"):
        assert registry.get(existing) is not None, f"{existing} disappeared"


@pytest.mark.parametrize("request_text,expected", [
    ("research 20 Greek engineering companies and put them in an excel file",
     "start_autonomous_task"),
    ("pause that task", "control_autonomous_task"),
    ("how is that research going", "list_autonomous_tasks"),
    ("every weekday at 9am check my calendar and email and tell me what needs attention",
     "create_workflow"),
    ("show my workflows", "list_workflows"),
    ("disable the morning briefing", "manage_workflow"),
])
def test_the_router_reaches_the_new_tools(registry, request_text, expected):
    """They go through the existing router like everything else - no bypass, and
    no need to expose all 111 tools to find them."""
    routing = tool_router.route(request_text, registry)

    assert expected in set(routing.tool_names)
    assert not routing.full_fallback, "had to fall back to every tool to find it"
    assert routing.count < len(registry.names()) * 0.5


# --- Workflow -> task -> orchestrator ------------------------------------------------

@pytest.mark.asyncio
async def test_running_a_workflow_creates_a_task_whose_steps_reach_the_orchestrator(registry):
    workflow = workflows.build("Morning briefing",
                               ["check my calendar", "summarise what needs attention"])
    workflows.save_draft(workflow)
    workflows.activate(workflow["id"], registry)

    orchestrator = FakeOrchestrator()
    runner = task_manager.TaskRunner(orchestrator)
    workflow_tools.set_context(registry=registry, runner=None)

    result = await workflow_tools.RunWorkflowTool().run(workflow_id=workflow["id"])
    assert result.success
    task_id = result.output["task"]["id"]

    outcome = await runner.run(task_id)

    assert outcome["status"] == task_manager.COMPLETED
    assert len(orchestrator.seen) == 2
    assert "check my calendar" in orchestrator.seen[0]


@pytest.mark.asyncio
async def test_a_scheduled_workflow_runs_through_the_existing_scheduler(registry):
    """The scheduler's instruction names run_workflow, which is a registered tool -
    so a scheduled run travels the same route a typed one does."""
    workflow = workflows.build("Weekly planning", ["summarise the week"],
                               workflows.TRIGGER_SCHEDULE,
                               {"schedule_type": "weekly", "at": "18:00",
                                "weekdays": ["sunday"]})
    workflows.save_draft(workflow)
    workflows.activate(workflow["id"], registry)

    task = scheduler.load_tasks()[0]
    assert "workflow" in task["instruction"].lower()
    assert workflow["id"] in task["instruction"]
    assert task["next_run"] is not None


@pytest.mark.asyncio
async def test_an_invalid_workflow_cannot_be_run(registry):
    workflow = workflows.build("broken", [{"instruction": "x", "tool": "no_such_tool"}])
    workflows.save_draft(workflow)
    workflow_tools.set_context(registry=registry, runner=None)

    result = await workflow_tools.RunWorkflowTool().run(workflow_id=workflow["id"])

    assert result.success is False
    assert task_manager.load_tasks() == [], "a task was created from an invalid workflow"


# --- Safety: no shortcuts ------------------------------------------------------------

@pytest.mark.parametrize("module", [task_manager, workflows, autonomous, workflow_tools])
def test_no_new_module_executes_a_tool_or_decides_authorization(module):
    """Structural, across all four: none of them calls a tool's run(), asks the
    guard, or reaches into the registry to execute something."""
    tree = ast.parse(inspect.getsource(module))

    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("authorize", "audit_result", "check_hard_block", "_execute_tool_call"):
        assert forbidden not in called, f"{module.__name__} calls {forbidden}()"


def test_the_only_way_a_task_step_runs_is_through_the_orchestrator():
    source = inspect.getsource(task_manager.TaskRunner)

    assert "handle_user_input" in source
    assert "tool.run" not in source and ".run(**" not in source


@pytest.mark.asyncio
async def test_a_workflow_cannot_activate_itself_by_being_created(registry):
    """Creating and switching on are deliberately two moves: a workflow that sends
    mail every morning is a standing instruction, and an ambiguous request must not
    produce one silently."""
    workflow_tools.set_context(registry=registry, runner=None)

    result = await workflow_tools.CreateWorkflowTool().run(
        name="Morning mail", steps=["check the mail", "send the summary"],
        schedule_type="daily", at="09:00")

    assert result.success
    assert result.output["activated"] is False
    stored = workflows.get_workflow(result.output["workflow_id"])
    assert stored["enabled"] is False
    assert scheduler.load_tasks() == [], "creating a workflow scheduled something"


@pytest.mark.asyncio
async def test_the_preview_is_returned_before_anything_is_switched_on(registry):
    workflow_tools.set_context(registry=registry, runner=None)

    result = await workflow_tools.CreateWorkflowTool().run(
        name="Morning briefing", steps=["check my calendar"],
        schedule_type="daily", at="09:00")

    assert "Nothing runs until you activate it." in result.output["preview"]
    assert "every day at 09:00" in result.output["preview"]


@pytest.mark.asyncio
async def test_controlling_a_task_refuses_to_choose_between_two(registry):
    task_manager.create_task("research companies", ["a"], "company research")
    task_manager.create_task("write the report", ["a"], "report writing")

    result = await autonomous.ControlAutonomousTaskTool().run(action="pause", which="")

    assert result.success is False
    assert result.output["ambiguous"] is True
    assert len(result.output["tasks"]) == 2


@pytest.mark.asyncio
async def test_starting_a_task_reports_its_real_plan(registry):
    autonomous.set_runner(None)

    result = await autonomous.StartAutonomousTaskTool().run(
        objective="research 20 companies",
        steps=["find candidates", "check each one", "write the file"],
        name="Greek engineering research")

    assert result.success
    assert result.output["task"]["steps_total"] == 3
    assert result.output["task"]["status"] == task_manager.QUEUED
    assert result.output["started"] is False, "started without a runner"


@pytest.mark.asyncio
async def test_a_task_with_no_steps_is_refused_rather_than_invented(registry):
    result = await autonomous.StartAutonomousTaskTool().run(objective="do something", steps=[])

    assert result.success is False
    assert task_manager.load_tasks() == []


# --- Idle cost -----------------------------------------------------------------------

def test_the_task_runner_never_waits_or_polls():
    """The scheduler Leti already has is the only thing that wakes on a timer; a
    task runner with its own poll would be a second one.

    Its loop is over the task's STEPS, not over time - it advances only by finishing
    a step and stops the moment there are none left or the status changes. What
    would make it a poller is waiting, so that is what this forbids.
    """
    source = inspect.getsource(task_manager.TaskRunner)

    assert "sleep" not in source, "the task runner waits"
    assert "check_interval" not in source and "call_later" not in source

    tree = ast.parse(inspect.getsource(task_manager))
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.While, ast.AsyncFor))]
    for loop in loops:
        breaks = [n for n in ast.walk(loop) if isinstance(n, ast.Break)]
        assert breaks, "a loop with no way out"


def test_creating_a_workflow_starts_nothing_running():
    """Idle means idle: describing a routine must not spin anything up."""
    source = inspect.getsource(workflows)

    assert "while True" not in source
    assert "create_task(" not in source, "the workflow layer starts work by itself"
