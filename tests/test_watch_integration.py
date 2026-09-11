"""The three features meeting the systems that were already here."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

import main  # noqa: E402
from core import task_manager, tool_router, watches, workflows  # noqa: E402
from tools import scheduler, watch_tools  # noqa: E402

NEW_TOOLS = ("create_watch", "list_watches", "manage_watch", "check_watches",
             "choose_computer_approach", "verify_screen", "end_computer_session",
             "archive_project")


@pytest.fixture(scope="module")
def registry():
    sys.argv = ["main.py"]
    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(watches, "store_path", lambda: tmp_path / "watches.json")
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "scheduled.json")
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(workflows, "store_path", lambda: tmp_path / "workflows.json")


def test_the_new_tools_are_registered_and_nothing_was_lost(registry):
    for name in NEW_TOOLS:
        assert registry.get(name) is not None, f"{name} is missing"
    # Only ever goes up: 119 with watches and GUI control, 123 with goals,
    # the Permission Center and Diagnostics.
    assert len(registry.names()) >= 123, f"the registry lost tools: {len(registry.names())}"
    for existing in ("mouse_click", "read_screen", "browser_read_page", "create_project",
                     "start_autonomous_task", "create_workflow", "send_email"):
        assert registry.get(existing) is not None, f"{existing} disappeared"


@pytest.mark.parametrize("request_text,expected", [
    ("tell me if my cpu stays above 90 percent", "create_watch"),
    ("what are you watching", "list_watches"),
    ("open this application and click the settings button", "choose_computer_approach"),
    ("archive the parot project", "archive_project"),
])
def test_the_router_reaches_them_without_falling_back(registry, request_text, expected):
    routing = tool_router.route(request_text, registry)

    assert expected in set(routing.tool_names)
    assert not routing.full_fallback
    assert routing.count < len(registry.names()) * 0.5


@pytest.mark.asyncio
async def test_a_triggered_watch_starts_a_task_rather_than_acting_itself(registry, monkeypatch):
    """Seeing a condition is not permission to act on it. The action becomes a
    task, which the orchestrator runs and SafetyGuard authorises call by call."""
    watch = watches.save_new(watches.create(
        "cpu", "cpu_above", {"percent": 90}, action="start_task",
        action_target="find out what is using the CPU", cooldown_minutes=0))
    monkeypatch.setattr(watches, "evaluate_condition", lambda w: (True, {}))
    watch_tools.set_runner(None)

    result = await watch_tools.CheckWatchesTool().run()

    assert result.success
    assert len(result.output["triggered"]) == 1
    started = result.output["tasks_started"]
    assert len(started) == 1
    task = task_manager.get_task(started[0]["id"])
    assert task["objective"] == "find out what is using the CPU"
    assert task["status"] == task_manager.QUEUED, "a watch executed something directly"


@pytest.mark.asyncio
async def test_a_watch_that_only_notifies_starts_nothing(registry, monkeypatch):
    watches.save_new(watches.create("cpu", "cpu_above", {"percent": 90}, cooldown_minutes=0))
    monkeypatch.setattr(watches, "evaluate_condition", lambda w: (True, {}))

    result = await watch_tools.CheckWatchesTool().run()

    assert result.output["triggered"] and result.output["tasks_started"] == []
    assert task_manager.load_tasks() == []


@pytest.mark.asyncio
async def test_checking_reports_an_error_without_calling_it_a_trigger(registry, monkeypatch):
    watches.save_new(watches.create("cpu", "cpu_above", {"percent": 90}))
    monkeypatch.setattr(watches, "evaluate_condition",
                        lambda w: (_ for _ in ()).throw(watches.ConditionError("gone")))

    result = await watch_tools.CheckWatchesTool().run()

    assert result.output["triggered"] == []
    assert result.output["errors"] and "gone" in result.output["errors"][0]["error"]


@pytest.mark.asyncio
async def test_a_watch_belongs_to_a_project(registry):
    watch_tools.set_runner(None)

    result = await watch_tools.CreateWatchTool().run(
        name="client site", condition_type="url_changed",
        url="https://example.com", project="Parot Automations")

    assert result.success
    assert result.output["watch"]["project"] == "Parot Automations"
    assert watches.get_watch(result.output["watch"]["id"])["project"] == "Parot Automations"


@pytest.mark.asyncio
async def test_an_unsupported_condition_is_refused_rather_than_faked(registry):
    result = await watch_tools.CreateWatchTool().run(
        name="mail", condition_type="email_from")

    assert result.success is False
    assert watches.load_watches() == []
    assert scheduler.load_tasks() == [], "an invalid watch scheduled something"


@pytest.mark.asyncio
async def test_creating_a_watch_registers_exactly_one_scheduler_row(registry):
    watch_tools.set_runner(None)
    for i in range(3):
        await watch_tools.CreateWatchTool().run(
            name=f"w{i}", condition_type="cpu_above", percent=90)

    assert len(scheduler.load_tasks()) == 1


@pytest.mark.asyncio
async def test_the_gui_session_flow_stops_when_the_screen_is_wrong(registry):
    from tools import computer_use as gui_tools

    decision = await gui_tools.ChooseComputerApproachTool().run(
        request="open Blender and click the Settings button")
    assert decision.output["layer"] == "gui"
    session_id = decision.output["session_id"]

    wrong = await gui_tools.VerifyScreenTool().run(
        session_id=session_id, expected="Settings window General tab",
        observed="An unexpected crash dialog is showing", next_action="click Save")

    assert wrong.success is False
    assert "Stopping" in wrong.error


@pytest.mark.asyncio
async def test_the_gui_flow_sends_a_normal_request_to_a_normal_tool(registry):
    from tools import computer_use as gui_tools

    decision = await gui_tools.ChooseComputerApproachTool().run(
        request="read the file notes.txt")

    assert decision.output["layer"] == "tool"
    assert "session_id" not in decision.output, "a GUI session opened for a file read"
