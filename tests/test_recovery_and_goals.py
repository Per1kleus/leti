"""Self-recovery, and goals that are judged by their outcome.

Recovery's danger is a machine that will not admit defeat, so most of these are
about its limits: what it will not retry, how many times, and that it never
becomes a way to have another go at something the guard refused.
"""
from __future__ import annotations

import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import task_manager  # noqa: E402
from core.safety_guard import ConfirmationDenied, PermissionDenied  # noqa: E402
from core.task_manager import COMPLETED, FAILED, WAITING_FOR_USER, TaskRunner  # noqa: E402
from tools import autonomous  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")


class FakeOrchestrator:
    def __init__(self, behaviour):
        self.seen = []
        self.behaviour = behaviour

    async def handle_user_input(self, text):
        self.seen.append(text)
        result = self.behaviour(text, len(self.seen))
        if isinstance(result, Exception):
            raise result
        return result


def _task(steps=("do the thing",)):
    return task_manager.create_task("an objective", list(steps), "test")


# --- What counts as recoverable ------------------------------------------------------

@pytest.mark.parametrize("message", [
    "connection reset by peer", "the request timed out", "503 Service Unavailable",
    "rate limit exceeded", "the site is temporarily unavailable", "network unreachable",
])
def test_a_transient_failure_is_worth_another_approach(message):
    assert task_manager.is_recoverable(RuntimeError(message)) is True


@pytest.mark.parametrize("message", [
    "no such file or directory", "invalid argument", "syntax error",
])
def test_an_ordinary_failure_is_not_recovered_from(message):
    assert task_manager.is_recoverable(RuntimeError(message)) is False


@pytest.mark.parametrize("error", [
    ConfirmationDenied("'send_email' is an external action"),
    PermissionDenied("forbidden path"),
])
def test_an_authorization_answer_is_never_treated_as_recoverable(error):
    """Retrying a refusal differently is trying to get around it."""
    assert task_manager.is_recoverable(error) is False


# --- Recovery in a running task -------------------------------------------------------

@pytest.mark.asyncio
async def test_a_recoverable_step_is_retried_a_different_way_and_carries_on():
    task = _task(["fetch the client website", "write the summary"])
    attempts = {"n": 0}

    def website_down_at_first(text, n):
        if "fetch" in text:
            attempts["n"] += 1
            # Fails the plain retries, then succeeds once told to try another way.
            if "different way" not in text:
                return RuntimeError("the website is temporarily unavailable")
        return "done"

    result = await TaskRunner(FakeOrchestrator(website_down_at_first)).run(task["id"])

    assert result["status"] == COMPLETED
    assert result["recoveries"] == 1
    assert result["steps_done"] == 2


@pytest.mark.asyncio
async def test_the_recovery_instruction_tells_it_not_to_repeat_itself():
    task = _task(["fetch the page"])
    seen = []

    def always_down(text, n):
        seen.append(text)
        return RuntimeError("503 unavailable")

    await TaskRunner(FakeOrchestrator(always_down)).run(task["id"])

    recovery_prompts = [t for t in seen if "different way" in t]
    assert recovery_prompts, "no recovery was attempted"
    assert "Do not repeat the approach that just failed" in recovery_prompts[0]
    assert "do not skip the step" in recovery_prompts[0]


@pytest.mark.asyncio
async def test_recovery_is_bounded_and_the_task_eventually_fails():
    """The failure mode of automatic recovery is a machine that never gives up."""
    task = _task(["fetch the page"])
    calls = []

    def always_down(text, n):
        calls.append(n)
        return RuntimeError("connection reset")

    result = await TaskRunner(FakeOrchestrator(always_down)).run(task["id"])

    assert result["status"] == FAILED
    expected = task_manager.DEFAULT_MAX_ATTEMPTS * (task_manager.MAX_RECOVERIES_PER_STEP + 1)
    assert len(calls) == expected, f"unbounded: {len(calls)} attempts"
    assert result["recoveries"] == task_manager.MAX_RECOVERIES_PER_STEP


@pytest.mark.asyncio
async def test_a_step_needing_approval_still_stops_the_task_rather_than_recovering():
    task = _task(["send the emails"])
    calls = []

    def needs_approval(text, n):
        calls.append(n)
        return ConfirmationDenied("'send_email' is an external action")

    result = await TaskRunner(FakeOrchestrator(needs_approval)).run(task["id"])

    assert result["status"] == WAITING_FOR_USER
    assert len(calls) == 1, "an approval request was retried as a recovery"
    assert result["recoveries"] == 0


@pytest.mark.asyncio
async def test_recovery_is_written_into_the_task_history():
    """Do not hide repeated failures: what happened has to be visible."""
    task = _task(["fetch the page"])

    def down_then_up(text, n):
        return "done" if "different way" in text else RuntimeError("timed out")

    result = await TaskRunner(FakeOrchestrator(down_then_up)).run(task["id"])

    notes = " ".join(h["note"] for h in result["step_history"])
    assert "attempt" in notes and "failed" in notes
    assert "recovery 1 attempted" in notes
    assert "recovery 1 succeeded" in notes


@pytest.mark.asyncio
async def test_work_already_done_survives_a_recovery():
    task = _task(["first", "second", "third"])

    def second_is_flaky(text, n):
        if "second" in text and "different way" not in text:
            return RuntimeError("temporarily unavailable")
        return "done"

    result = await TaskRunner(FakeOrchestrator(second_is_flaky)).run(task["id"])

    assert result["status"] == COMPLETED
    stored = task_manager.get_task(task["id"])
    assert stored["steps"][0]["status"] == "done", "completed work was redone or lost"


@pytest.mark.asyncio
async def test_recovery_runs_through_the_orchestrator_like_everything_else():
    """A recovery attempt is a normal action, not a privileged one."""
    task = _task(["fetch the page"])
    orchestrator = FakeOrchestrator(
        lambda t, n: "done" if "different way" in t else RuntimeError("timed out"))

    await TaskRunner(orchestrator).run(task["id"])

    assert all("Autonomous task" in text for text in orchestrator.seen)


# --- Goals ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_goal_becomes_a_task_with_a_verification_step():
    """"Create a report" is not done because a file was created."""
    autonomous.set_runner(None)

    result = await autonomous.PursueGoalTool().run(
        goal="research 20 Greek accounting firms and create a report",
        steps=["find candidate firms", "check each one", "write the report"],
        success_looks_like="a file at reports/firms.md listing 20 firms with contacts")

    assert result.success
    plan = result.output["plan"]
    assert len(plan) == 4, "no verification step was added"
    assert "Check whether the goal has actually been met" in plan[-1]
    assert "reports/firms.md" in plan[-1]
    assert "Verify it rather than assuming" in plan[-1]


@pytest.mark.asyncio
async def test_a_goal_without_a_definition_of_done_is_refused():
    result = await autonomous.PursueGoalTool().run(
        goal="make it better", steps=["do something"], success_looks_like="")

    assert result.success is False
    assert "what done looks like" in result.error
    assert task_manager.load_tasks() == []


@pytest.mark.asyncio
async def test_a_goal_without_a_plan_is_refused():
    result = await autonomous.PursueGoalTool().run(
        goal="do it", steps=[], success_looks_like="it is done")

    assert result.success is False
    assert task_manager.load_tasks() == []


@pytest.mark.asyncio
async def test_a_goal_can_belong_to_a_project():
    autonomous.set_runner(None)

    result = await autonomous.PursueGoalTool().run(
        goal="research clients", steps=["find them"],
        success_looks_like="a list of 20", project="Parot Automations")

    task = task_manager.get_task(result.output["task"]["id"])
    assert task["project"] == "Parot Automations"
    assert task["goal"] == "research clients"
    assert task["success_looks_like"] == "a list of 20"


@pytest.mark.asyncio
async def test_a_goal_grants_no_extra_permissions():
    """Structurally the same object as any other task: the runner cannot tell them
    apart, so there is nothing for a goal to be privileged by."""
    autonomous.set_runner(None)
    await autonomous.PursueGoalTool().run(
        goal="g", steps=["s"], success_looks_like="done")

    task = task_manager.load_tasks()[0]
    assert task["status"] == task_manager.QUEUED
    assert "steps" in task and "current_step" in task

    import ast
    import inspect

    tree = ast.parse(inspect.getsource(autonomous))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("authorize", "set_unattended", "audit_result"):
        assert forbidden not in called, f"the goal tool calls {forbidden}()"
