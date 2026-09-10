"""Workflows: described in words, checked before they are switched on, and run
through the same path everything else runs through.

A workflow is a standing instruction. A misread one is a standing mistake, which
is why validation and the explicit activation step exist and why these tests spend
most of their time on what must not activate.
"""
from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import task_manager, workflows  # noqa: E402
from tools import scheduler  # noqa: E402
from tools.base import BaseTool, ToolParameter, ToolRegistry, ToolResult  # noqa: E402


class NeedsAPath(BaseTool):
    name = "write_file"
    description = "test double"
    parameters = [ToolParameter(name="path", type="string", description="where"),
                  ToolParameter(name="content", type="string", description="what",
                                required=False)]

    async def run(self, **kwargs):
        return ToolResult(success=True)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(workflows, "store_path", lambda: tmp_path / "workflows.json")
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "scheduled.json")
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(NeedsAPath())
    return reg


def _daily(name="Morning briefing", at="09:00", steps=None):
    return workflows.build(name, steps or ["check my calendar", "summarise what needs attention"],
                           workflows.TRIGGER_SCHEDULE, {"schedule_type": "daily", "at": at})


# --- Building ---------------------------------------------------------------------

def test_a_described_routine_becomes_a_structure():
    workflow = _daily()

    assert workflow["trigger"] == "schedule"
    assert workflow["schedule"] == {"schedule_type": "daily", "at": "09:00"}
    assert [s["n"] for s in workflow["steps"]] == [1, 2]
    assert workflow["enabled"] is False and workflow["activated"] is False, (
        "a workflow was live the moment it was described")


def test_steps_can_be_plain_sentences_or_structured():
    workflow = workflows.build("w", ["just words",
                                     {"instruction": "write it", "tool": "write_file",
                                      "parameters": {"path": "/tmp/x"}}])

    assert workflow["steps"][0]["tool"] is None
    assert workflow["steps"][1]["tool"] == "write_file"


# --- Validation --------------------------------------------------------------------

def test_a_valid_workflow_has_nothing_to_report(registry):
    errors, questions = workflows.validate(_daily(), registry)
    assert errors == [] and questions == []


def test_a_step_naming_a_tool_that_does_not_exist_is_an_error(registry):
    workflow = workflows.build("w", [{"instruction": "do it", "tool": "no_such_tool"}])

    errors, _ = workflows.validate(workflow, registry)
    assert any("not a registered tool" in e for e in errors)


def test_a_missing_required_parameter_is_an_error(registry):
    workflow = workflows.build("w", [{"instruction": "write", "tool": "write_file"}])

    errors, _ = workflows.validate(workflow, registry)
    assert any("required parameter" in e and "path" in e for e in errors)


@pytest.mark.parametrize("schedule,fragment", [
    ({"schedule_type": "hourly"}, "is not a schedule"),
    ({"schedule_type": "daily", "at": "25:00"}, "not a time of day"),
    ({"schedule_type": "weekly", "at": "09:00", "weekdays": []}, "needs at least one weekday"),
    ({"schedule_type": "weekly", "at": "09:00", "weekdays": ["someday"]}, "Not weekdays"),
    ({"schedule_type": "once"}, "needs a time to run at"),
    ({"schedule_type": "interval", "every_minutes": 0}, "at least 1"),
])
def test_an_impossible_schedule_is_an_error(registry, schedule, fragment):
    workflow = workflows.build("w", ["do a thing"], workflows.TRIGGER_SCHEDULE, schedule)

    errors, _ = workflows.validate(workflow, registry)
    assert any(fragment in e for e in errors), errors


def test_a_step_depending_on_one_that_comes_later_is_an_error(registry):
    workflow = workflows.build("w", [{"instruction": "first", "depends_on": [2]},
                                     {"instruction": "second"}])

    errors, _ = workflows.validate(workflow, registry)
    assert any("does not come before" in e for e in errors)


def test_a_workflow_with_no_steps_is_an_error(registry):
    errors, _ = workflows.validate(workflows.build("w", []), registry)
    assert any("at least one step" in e for e in errors)


def test_an_ambiguous_step_becomes_a_question_not_a_guess(registry):
    """"Notify me" is not wrong, it is under-specified. The answer is to ask."""
    workflow = workflows.build("w", ["check the mail", "notify me"])

    errors, questions = workflows.validate(workflow, registry)
    assert errors == []
    assert any("desktop notification or an email" in q for q in questions)


def test_a_specific_notification_is_not_ambiguous(registry):
    workflow = workflows.build("w", ["notify me with a desktop notification"])

    _, questions = workflows.validate(workflow, registry)
    assert questions == []


# --- Preview and activation ---------------------------------------------------------

def test_the_preview_says_what_leti_understood():
    text = workflows.preview(_daily())

    assert "Morning briefing" in text
    assert "every day at 09:00" in text
    assert "check my calendar" in text
    assert "Nothing runs until you activate it." in text


def test_an_invalid_workflow_is_never_activated(registry):
    workflow = workflows.build("w", [{"instruction": "x", "tool": "no_such_tool"}])
    workflows.save_draft(workflow)

    result = workflows.activate(workflow["id"], registry)

    assert result["ok"] is False
    assert workflows.get_workflow(workflow["id"])["enabled"] is False


def test_an_ambiguous_workflow_is_not_activated_either(registry):
    workflow = workflows.build("w", ["notify me"])
    workflows.save_draft(workflow)

    result = workflows.activate(workflow["id"], registry)

    assert result["ok"] is False and result["questions"]
    assert workflows.get_workflow(workflow["id"])["enabled"] is False


def test_activating_a_scheduled_workflow_registers_it_with_the_existing_scheduler(registry):
    """No second scheduler: activation adds a row to the one Leti already has."""
    workflow = _daily()
    workflows.save_draft(workflow)

    result = workflows.activate(workflow["id"], registry)

    assert result["ok"] is True
    stored = workflows.get_workflow(workflow["id"])
    assert stored["enabled"] and stored["activated"]
    tasks = scheduler.load_tasks()
    assert len(tasks) == 1
    assert tasks[0]["id"] == stored["scheduled_task_id"]
    assert tasks[0]["workflow_id"] == workflow["id"]
    assert tasks[0]["next_run"] > time.time(), "no next run was computed"


def test_a_manual_workflow_activates_without_touching_the_scheduler(registry):
    workflow = workflows.build("on demand", ["do a thing"])
    workflows.save_draft(workflow)

    assert workflows.activate(workflow["id"], registry)["ok"] is True
    assert scheduler.load_tasks() == []


# --- Management ---------------------------------------------------------------------

def test_disabling_a_workflow_also_stops_its_scheduled_task(registry):
    workflow = _daily()
    workflows.save_draft(workflow)
    workflows.activate(workflow["id"], registry)

    workflows.set_enabled(workflow["id"], False)

    assert workflows.get_workflow(workflow["id"])["enabled"] is False
    assert scheduler.load_tasks()[0]["enabled"] is False


def test_re_enabling_brings_the_schedule_back(registry):
    workflow = _daily()
    workflows.save_draft(workflow)
    workflows.activate(workflow["id"], registry)
    workflows.set_enabled(workflow["id"], False)

    workflows.set_enabled(workflow["id"], True)

    assert scheduler.load_tasks()[0]["enabled"] is True


def test_changing_the_time_moves_the_scheduled_task(registry):
    """"Change it to 8 AM"."""
    workflow = _daily(at="09:00")
    workflows.save_draft(workflow)
    workflows.activate(workflow["id"], registry)

    result = workflows.update_schedule(
        workflow["id"], {"schedule_type": "daily", "at": "08:00"}, registry)

    assert result["ok"] is True
    tasks = scheduler.load_tasks()
    assert len(tasks) == 1, "rescheduling left a duplicate behind"
    assert tasks[0]["at"] == "08:00"
    assert workflows.get_workflow(workflow["id"])["schedule"]["at"] == "08:00"


def test_an_invalid_new_time_is_refused_and_changes_nothing(registry):
    workflow = _daily(at="09:00")
    workflows.save_draft(workflow)
    workflows.activate(workflow["id"], registry)

    result = workflows.update_schedule(
        workflow["id"], {"schedule_type": "daily", "at": "99:99"}, registry)

    assert result["ok"] is False
    assert workflows.get_workflow(workflow["id"])["schedule"]["at"] == "09:00"
    assert scheduler.load_tasks()[0]["at"] == "09:00"


def test_deleting_a_workflow_takes_its_schedule_with_it(registry):
    workflow = _daily()
    workflows.save_draft(workflow)
    workflows.activate(workflow["id"], registry)

    assert workflows.delete(workflow["id"])["ok"] is True
    assert workflows.load_workflows() == []
    assert scheduler.load_tasks() == [], "a schedule outlived its workflow"


def test_describing_a_workflow_reports_its_next_run(registry):
    workflow = _daily()
    workflows.save_draft(workflow)
    workflows.activate(workflow["id"], registry)

    described = workflows.describe(workflows.get_workflow(workflow["id"]))
    assert described["enabled"] is True
    assert described["when"] == "every day at 09:00"
    assert described["next_run"], "no next run was reported"


# --- Workflow -> task ---------------------------------------------------------------

def test_a_workflow_becomes_task_steps_in_words():
    """The join between the two systems keeps the request in plain language, so it
    still travels through the router and the guard rather than around them."""
    workflow = workflows.build("w", [
        "check the calendar",
        {"instruction": "write it up", "tool": "write_file", "parameters": {"path": "/tmp/x"}},
    ])

    steps = workflows.to_task_steps(workflow)

    assert steps[0] == "check the calendar"
    assert "write it up" in steps[1] and "write_file" in steps[1] and "/tmp/x" in steps[1]


def test_an_unreadable_workflow_store_does_not_take_leti_down(tmp_path, monkeypatch):
    monkeypatch.setattr(workflows, "store_path", lambda: tmp_path / "broken.json")
    (tmp_path / "broken.json").write_text("{ not json")

    assert workflows.load_workflows() == []
