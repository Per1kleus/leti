"""Objectives that outlive a turn: state, control, failure, and the approval stop.

The tests that matter most are the ones about what this must NOT do - run a tool
itself, proceed past an action that needs permission, retry forever, or resume a
step whose side effects nobody can account for.
"""
from __future__ import annotations

import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import task_manager  # noqa: E402
from core.safety_guard import ConfirmationDenied, PermissionDenied  # noqa: E402
from core.task_manager import (  # noqa: E402
    CANCELLED, COMPLETED, FAILED, PAUSED, QUEUED, RUNNING, WAITING_FOR_USER, TaskRunner,
)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")


class FakeOrchestrator:
    """Stands in for the real one. Records the instructions it was handed."""

    def __init__(self, behaviour=None):
        self.seen = []
        self.behaviour = behaviour or (lambda text, n: f"did: {text[-40:]}")

    async def handle_user_input(self, text):
        self.seen.append(text)
        result = self.behaviour(text, len(self.seen))
        if isinstance(result, Exception):
            raise result
        return result


class FakeGuard:
    def __init__(self):
        self._unattended = False
        self.history = []

    def set_unattended(self, value=True):
        self._unattended = value
        self.history.append(value)


def _task(steps=("step one", "step two")):
    return task_manager.create_task("an objective", list(steps), "test task")


# --- Creating and persisting -------------------------------------------------------

def test_a_task_is_created_queued_with_its_steps():
    task = _task(["find the companies", "write the file"])

    assert task["status"] == QUEUED
    assert [s["instruction"] for s in task["steps"]] == ["find the companies", "write the file"]
    assert task["current_step"] == 0


def test_a_task_survives_a_restart():
    """Persistence is the point: the store is all the state there is."""
    task = _task()
    task_manager._set_status(task["id"], PAUSED)

    reloaded = task_manager.get_task(task["id"])
    assert reloaded["status"] == PAUSED
    assert reloaded["objective"] == "an objective"


@pytest.mark.parametrize("bad", [("", ["a"]), ("objective", []), ("objective", ["  "])])
def test_a_task_without_an_objective_or_steps_is_refused(bad):
    with pytest.raises(ValueError):
        task_manager.create_task(bad[0], bad[1])


def test_an_absurd_number_of_steps_is_refused():
    with pytest.raises(ValueError):
        task_manager.create_task("o", [f"step {i}" for i in range(task_manager.MAX_STEPS + 1)])


def test_an_unreadable_store_does_not_take_leti_down(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "broken.json")
    (tmp_path / "broken.json").write_text("{ not json")

    assert task_manager.load_tasks() == []


# --- Progress ----------------------------------------------------------------------

def test_progress_is_counted_from_real_steps_not_invented():
    task = _task(["a", "b", "c", "d"])
    task["steps"][0]["status"] = "done"
    task["steps"][1]["status"] = "done"
    task["current_step"] = 2
    task_manager._replace(task)

    p = task_manager.progress(task_manager.get_task(task["id"]))
    assert p == {"steps_total": 4, "steps_done": 2, "current_step": 3,
                 "current_instruction": "c"}
    assert task_manager.describe(task_manager.get_task(task["id"]))["step"] == "3/4"


# --- Control -----------------------------------------------------------------------

def test_pause_resume_and_cancel_move_the_status():
    task = _task()
    assert task_manager.pause(task["id"])["status"] == PAUSED
    assert task_manager.resume(task["id"])["status"] == QUEUED
    assert task_manager.cancel(task["id"])["status"] == CANCELLED


def test_a_finished_task_cannot_be_cancelled_or_paused():
    task = _task()
    task_manager._set_status(task["id"], COMPLETED)

    assert task_manager.cancel(task["id"]) is None
    assert task_manager.pause(task["id"]) is None


def test_resuming_keeps_the_work_already_done():
    task = _task(["a", "b", "c"])
    task["current_step"] = 2
    task["steps"][0]["status"] = task["steps"][1]["status"] = "done"
    task_manager._replace(task)
    task_manager._set_status(task["id"], PAUSED)

    resumed = task_manager.resume(task["id"])
    assert resumed["current_step"] == 2, "resuming restarted the task"
    assert resumed["steps"][0]["status"] == "done"


def test_matching_a_task_by_words_and_refusing_to_guess_between_two():
    first = task_manager.create_task("research companies", ["a"], "company research")
    second = task_manager.create_task("write the report", ["a"], "report writing")

    assert [t["id"] for t in task_manager.find_active("research")] == [first["id"]]
    assert len(task_manager.find_active("")) == 2
    assert len(task_manager.find_active("nothing matches this")) == 2, (
        "a hint that matches nothing should offer everything, not pick one")
    assert second["id"] in {t["id"] for t in task_manager.find_active("")}


# --- Running -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_task_runs_every_step_and_completes():
    task = _task(["step one", "step two", "step three"])
    orchestrator = FakeOrchestrator()

    result = await TaskRunner(orchestrator).run(task["id"])

    assert result["status"] == COMPLETED
    assert len(orchestrator.seen) == 3
    assert result["steps_done"] == 3
    assert task_manager.get_task(task["id"])["completed_at"] is not None


@pytest.mark.asyncio
async def test_each_step_goes_through_the_orchestrator_not_around_it():
    """The whole safety story rests on this: steps are ordinary requests."""
    task = _task(["find the companies"])
    orchestrator = FakeOrchestrator()

    await TaskRunner(orchestrator).run(task["id"])

    assert len(orchestrator.seen) == 1
    assert "find the companies" in orchestrator.seen[0]
    assert "Autonomous task" in orchestrator.seen[0], "the step lost its context"


@pytest.mark.asyncio
async def test_a_task_paused_mid_run_stops_between_steps():
    task = _task(["a", "b", "c"])

    def pause_after_first(text, n):
        if n == 1:
            task_manager.pause(task["id"])
        return "ok"

    orchestrator = FakeOrchestrator(pause_after_first)
    await TaskRunner(orchestrator).run(task["id"])

    assert len(orchestrator.seen) == 1, "a paused task kept going"
    assert task_manager.get_task(task["id"])["status"] == PAUSED


@pytest.mark.asyncio
async def test_a_cancelled_task_stops_and_stays_cancelled():
    task = _task(["a", "b", "c"])

    def cancel_after_first(text, n):
        if n == 1:
            task_manager.cancel(task["id"])
        return "ok"

    await TaskRunner(FakeOrchestrator(cancel_after_first)).run(task["id"])

    assert task_manager.get_task(task["id"])["status"] == CANCELLED


# --- Failure, retries, and their limit ---------------------------------------------

@pytest.mark.asyncio
async def test_a_failing_step_is_retried_once_then_the_task_fails():
    task = _task(["flaky"])
    calls = []

    def always_fails(text, n):
        calls.append(n)
        return RuntimeError("the service is down")

    result = await TaskRunner(FakeOrchestrator(always_fails)).run(task["id"])

    assert result["status"] == FAILED
    assert len(calls) == task_manager.DEFAULT_MAX_ATTEMPTS, "retries are not bounded"
    assert "the service is down" in result["error"]


@pytest.mark.asyncio
async def test_a_step_that_fails_once_and_then_works_carries_on():
    task = _task(["flaky", "second"])

    def fail_first(text, n):
        return RuntimeError("transient") if n == 1 else "fine"

    result = await TaskRunner(FakeOrchestrator(fail_first)).run(task["id"])

    assert result["status"] == COMPLETED
    assert result["steps_done"] == 2


@pytest.mark.asyncio
async def test_a_failed_task_keeps_its_state_and_can_be_resumed():
    """Preserving the failure is what makes it recoverable rather than lost."""
    task = _task(["a", "b"])

    def fail_second(text, n):
        return RuntimeError("nope") if n >= 2 else "fine"

    await TaskRunner(FakeOrchestrator(fail_second)).run(task["id"])
    failed = task_manager.get_task(task["id"])
    assert failed["status"] == FAILED
    assert failed["steps"][0]["status"] == "done"
    assert failed["current_step"] == 1

    resumed = task_manager.resume(task["id"])
    assert resumed["status"] == QUEUED and resumed["current_step"] == 1


@pytest.mark.asyncio
async def test_a_failure_never_reports_success():
    task = _task(["a"])
    result = await TaskRunner(FakeOrchestrator(lambda t, n: RuntimeError("x"))).run(task["id"])

    assert result["status"] == FAILED
    assert result["result"] is None


# --- Approval ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_action_needing_approval_stops_the_task_and_waits():
    """The guard refused because nobody was there to say yes. That is not a
    failure and it is certainly not permission - the task waits."""
    task = _task(["gather the addresses", "send the emails"])

    def blocked_on_second(text, n):
        return ConfirmationDenied("'send_email' is an 'external' action") if n == 2 else "done"

    result = await TaskRunner(FakeOrchestrator(blocked_on_second)).run(task["id"])

    assert result["status"] == WAITING_FOR_USER
    assert "approval" in (result["blocked_reason"] or "").lower()
    assert result["steps_done"] == 1, "the completed work was lost"


@pytest.mark.asyncio
async def test_an_approval_stop_is_not_retried_as_if_it_were_a_glitch():
    task = _task(["send it"])
    calls = []

    def blocked(text, n):
        calls.append(n)
        return ConfirmationDenied("needs approval")

    await TaskRunner(FakeOrchestrator(blocked)).run(task["id"])

    assert len(calls) == 1, "an approval request was retried"


@pytest.mark.asyncio
async def test_a_hard_block_is_a_failure_not_something_to_wait_on():
    """PermissionDenied is a forbidden path or pattern. No approval unblocks it."""
    task = _task(["do the forbidden thing"])

    result = await TaskRunner(
        FakeOrchestrator(lambda t, n: PermissionDenied("forbidden path"))).run(task["id"])

    assert result["status"] == FAILED


@pytest.mark.asyncio
async def test_a_waiting_task_resumes_from_the_step_that_was_blocked():
    task = _task(["gather", "send"])
    state = {"blocked": True}

    def blocked_until_approved(text, n):
        if "send" in text and state["blocked"]:
            return ConfirmationDenied("needs approval")
        return "done"

    runner = TaskRunner(FakeOrchestrator(blocked_until_approved))
    await runner.run(task["id"])
    assert task_manager.get_task(task["id"])["status"] == WAITING_FOR_USER

    state["blocked"] = False
    task_manager.resume(task["id"])
    result = await runner.run(task["id"])

    assert result["status"] == COMPLETED
    assert result["steps_done"] == 2


# --- Authorization is not bypassed --------------------------------------------------

@pytest.mark.asyncio
async def test_the_runner_puts_the_guard_in_unattended_mode_and_restores_it():
    """Unattended is the guard's own idea, and the runner only turns it on for the
    duration - it never widens what is permitted."""
    task = _task(["a"])
    guard = FakeGuard()

    await TaskRunner(FakeOrchestrator(), safety_guard=guard).run(task["id"])

    assert guard.history == [True, False]
    assert guard._unattended is False, "unattended mode leaked past the task"


def test_the_task_manager_never_executes_a_tool_itself():
    """Structural: it has no registry, no tool call and no guard decision in it."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(task_manager))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)

    assert not any("registry" in m or m.startswith("tools.") for m in imported), imported
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("authorize", "audit_result", "to_ollama_schema"):
        assert forbidden not in called, f"the task manager calls {forbidden}()"


# --- Restart recovery ---------------------------------------------------------------

def test_a_task_interrupted_by_a_restart_is_paused_not_resumed():
    """Nothing can know whether the step that was running had already had its
    effect, so it is never repeated automatically."""
    task = _task()
    task_manager._set_status(task["id"], RUNNING)

    recovered = task_manager.recover_interrupted()

    assert [t["id"] for t in recovered] == [task["id"]]
    after = task_manager.get_task(task["id"])
    assert after["status"] == PAUSED
    assert "restarted" in after["blocked_reason"].lower()


def test_recovery_leaves_everything_else_alone():
    done = _task()
    task_manager._set_status(done["id"], COMPLETED)

    assert task_manager.recover_interrupted() == []
    assert task_manager.get_task(done["id"])["status"] == COMPLETED
