"""Watching a long task, and being able to stop it.

The existing tests cover what a task must not do. These cover what a person needs
to be able to do WITH one: see where it is, pause it, take it back, retry the step
that failed, and answer the question it stopped to ask - without any of that
becoming a second way to run a tool.

Two things are load-bearing here and are asserted from several directions:

  Progress is counted, never estimated. A task with seven steps and three done
  says 3/7. A task whose length nobody knows says "working", not 43%.

  Approving is not authorising. The panel's Approve marks one step as carrying
  the user's answer and hands the task back to the runner; SafetyGuard still
  classifies the action, still refuses to let an irreversible one ride on
  approval given in advance, and still writes the audit line.
"""
from __future__ import annotations

import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import task_manager  # noqa: E402
from core.safety_guard import ConfirmationDenied  # noqa: E402
from core.task_manager import (  # noqa: E402
    CANCELLED, COMPLETED, FAILED, PAUSED, QUEUED, RUNNING, WAITING_FOR_USER, TaskRunner,
)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")


class FakeOrchestrator:
    def __init__(self, behaviour=None):
        self.seen = []
        self.approvals = []
        self.behaviour = behaviour or (lambda text, n: f"did: {text[-30:]}")

    async def handle_user_input(self, text, session_id="default", voice_mode=False,
                                preapproved=False):
        self.seen.append(text)
        self.approvals.append(bool(preapproved))
        result = self.behaviour(text, len(self.seen))
        if isinstance(result, Exception):
            raise result
        return result


def _task(steps=("one", "two", "three")):
    return task_manager.create_task("an objective", list(steps), "research")


# --- Progress is counted ------------------------------------------------------------

def test_progress_is_the_steps_that_are_actually_done():
    task = _task(["a", "b", "c", "d"])
    view = task_manager.detail(task)
    assert view["steps_done"] == 0 and view["steps_total"] == 4
    assert view["step"] == "1/4"

    task["steps"][0]["status"] = "done"
    task["steps"][1]["status"] = "done"
    task["current_step"] = 2
    task_manager._replace(task)

    view = task_manager.detail(task_manager.get_task(task["id"]))
    assert view["steps_done"] == 2
    assert view["step"] == "3/4"
    assert view["current"] == "c"


def test_the_panel_sees_every_step_with_its_own_state():
    task = _task(["find them", "read them", "write it up"])
    task["steps"][0]["status"] = "done"
    task["steps"][0]["result"] = "found four"
    task["steps"][1]["status"] = "failed"
    task["steps"][1]["error"] = "the site was down"
    task_manager._replace(task)

    steps = task_manager.detail(task_manager.get_task(task["id"]))["steps"]
    assert [s["status"] for s in steps] == ["done", "failed", "pending"]
    assert steps[0]["result"] == "found four"
    assert steps[1]["error"] == "the site was down"


def test_a_step_result_is_previewed_not_reproduced():
    """A step that wrote a report must not put the report in a panel row."""
    task = _task(["write it"])
    task["steps"][0]["result"] = "x" * 5_000
    task_manager._replace(task)
    assert len(task_manager.detail(task_manager.get_task(task["id"]))["steps"][0]["result"]) <= 400


def test_describe_is_unchanged_by_the_panel_view():
    """detail() is for a panel; describe() goes into the model's context and must
    not start carrying a full step list."""
    task = _task(["a", "b"])
    assert "steps" not in task_manager.describe(task)
    assert "steps" in task_manager.detail(task)


# --- Pause, resume, cancel ------------------------------------------------------------

@pytest.mark.asyncio
async def test_pausing_stops_further_steps_and_keeps_the_done_ones():
    task = _task(["one", "two", "three"])
    orchestrator = FakeOrchestrator()
    runner = TaskRunner(orchestrator)

    def pause_after_first(text, n):
        if n == 1:
            task_manager.pause(task["id"])
        return "ok"

    orchestrator.behaviour = pause_after_first
    await runner.run(task["id"])

    stored = task_manager.get_task(task["id"])
    assert stored["status"] == PAUSED
    assert stored["current_step"] == 1
    assert [s["status"] for s in stored["steps"]] == ["done", "pending", "pending"]
    assert len(orchestrator.seen) == 1, "a paused task ran another step"


@pytest.mark.asyncio
async def test_resuming_carries_on_and_does_not_redo_what_was_done():
    task = _task(["one", "two", "three"])
    orchestrator = FakeOrchestrator()
    runner = TaskRunner(orchestrator)
    def pause_after_first(text, n):
        if n == 1:
            task_manager.pause(task["id"])
        return "ok"

    orchestrator.behaviour = pause_after_first
    await runner.run(task["id"])
    first_pass = list(orchestrator.seen)

    orchestrator.behaviour = lambda text, n: "ok"
    assert task_manager.resume(task["id"])["status"] == QUEUED
    await runner.run(task["id"])

    stored = task_manager.get_task(task["id"])
    assert stored["status"] == COMPLETED
    assert all(s["status"] == "done" for s in stored["steps"])
    # Step one ran once, in the first pass, and was not run again.
    assert sum(1 for t in orchestrator.seen if "\none" in t) == 1
    assert first_pass[0] in orchestrator.seen


@pytest.mark.asyncio
async def test_cancelling_stops_the_task_and_keeps_its_history():
    task = _task(["one", "two", "three"])
    orchestrator = FakeOrchestrator()
    runner = TaskRunner(orchestrator)
    def cancel_after_first(text, n):
        if n == 1:
            task_manager.cancel(task["id"])
        return "ok"

    orchestrator.behaviour = cancel_after_first
    await runner.run(task["id"])

    stored = task_manager.get_task(task["id"])
    assert stored["status"] == CANCELLED
    assert stored["steps"][0]["status"] == "done", "cancelling erased what was done"
    assert stored["steps"][0]["result"]
    assert len(orchestrator.seen) == 1


def test_a_cancelled_task_is_not_a_completed_one():
    task = _task()
    task_manager.cancel(task["id"], "you stopped it")
    stored = task_manager.get_task(task["id"])
    assert stored["status"] == CANCELLED
    assert stored["result"] is None
    assert stored["blocked_reason"] == "you stopped it"


def test_what_a_task_can_be_told_depends_on_where_it_is():
    task = _task()
    assert task_manager.detail(task)["can"] == {
        "pause": True, "resume": False, "cancel": True, "retry": False, "approve": False}

    task_manager._set_status(task["id"], WAITING_FOR_USER)
    can = task_manager.detail(task_manager.get_task(task["id"]))["can"]
    assert can["approve"] is True and can["resume"] is True and can["pause"] is False

    task_manager._set_status(task["id"], COMPLETED)
    can = task_manager.detail(task_manager.get_task(task["id"]))["can"]
    assert not any([can["pause"], can["cancel"], can["approve"]])


# --- Retry ------------------------------------------------------------------------------

def test_retrying_resets_only_the_step_that_stopped():
    task = _task(["one", "two", "three"])
    task["steps"][0].update(status="done", result="the first result")
    task["steps"][1].update(status="failed", error="timed out", attempts=2, recoveries=2)
    task["current_step"] = 1
    task_manager._replace(task)
    task_manager._set_status(task["id"], FAILED, error="timed out")

    updated = task_manager.retry(task["id"])
    assert updated["status"] == QUEUED
    assert updated["current_step"] == 1
    assert updated["steps"][0]["status"] == "done"
    assert updated["steps"][0]["result"] == "the first result", "a retry lost finished work"
    assert updated["steps"][1]["status"] == "pending"
    assert updated["steps"][1]["attempts"] == 0 and updated["steps"][1]["recoveries"] == 0
    assert updated["steps"][1]["error"] is None
    assert any("retried by the user" in n["note"] for n in updated["steps"][1]["history"])


def test_a_running_task_cannot_be_retried_out_from_under_itself():
    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    assert task_manager.retry(task["id"]) is None


@pytest.mark.asyncio
async def test_a_retried_step_runs_again_and_the_earlier_ones_do_not():
    task = _task(["one", "two"])
    orchestrator = FakeOrchestrator()
    runner = TaskRunner(orchestrator)
    orchestrator.behaviour = lambda text, n: (RuntimeError("no such file") if "\ntwo" in text
                                              else "ok")
    await runner.run(task["id"])
    assert task_manager.get_task(task["id"])["status"] == FAILED
    before = len(orchestrator.seen)

    orchestrator.behaviour = lambda text, n: "ok"
    task_manager.retry(task["id"])
    await runner.run(task["id"])

    stored = task_manager.get_task(task["id"])
    assert stored["status"] == COMPLETED
    ran_after = orchestrator.seen[before:]
    assert ran_after and all("\ntwo" in t for t in ran_after), "the retry redid step one"


# --- Approval ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_step_that_needs_permission_stops_and_says_what_for():
    task = _task(["gather the addresses", "send four emails"])
    orchestrator = FakeOrchestrator()
    runner = TaskRunner(orchestrator)
    orchestrator.behaviour = lambda text, n: (
        ConfirmationDenied("'send_email' is an 'external' action, and this task ran "
                           "unattended with nobody to confirm it.")
        if "\nsend four emails" in text else "ok")
    await runner.run(task["id"])

    view = task_manager.detail(task_manager.get_task(task["id"]))
    assert view["status"] == WAITING_FOR_USER
    assert view["awaiting_approval"] is True
    assert "send_email" in view["approval_request"]
    assert view["steps"][0]["status"] == "done"
    assert view["steps"][1]["status"] == "blocked"
    assert view["can"]["approve"] is True


@pytest.mark.asyncio
async def test_approving_hands_the_users_answer_to_the_guard_for_one_step_only():
    """The approval reaches SafetyGuard the way a spoken yes does, and covers the
    step it was given for - not the rest of the task."""
    task = _task(["first", "the one that asks", "after it"])
    orchestrator = FakeOrchestrator()
    runner = TaskRunner(orchestrator)
    orchestrator.behaviour = lambda text, n: (
        ConfirmationDenied("'send_email' is an 'external' action") if n == 2 else "ok")
    await runner.run(task["id"])
    assert task_manager.get_task(task["id"])["status"] == WAITING_FOR_USER
    assert orchestrator.approvals == [False, False]

    orchestrator.behaviour = lambda text, n: "ok"
    assert task_manager.approve(task["id"])["status"] == QUEUED
    await runner.run(task["id"])

    assert task_manager.get_task(task["id"])["status"] == COMPLETED
    # The approved step carried it; the step after it did not.
    assert orchestrator.approvals[2] is True
    assert orchestrator.approvals[3] is False


def test_approving_writes_nothing_that_could_run_anything():
    """Approve marks a step. It does not touch a tool, a guard or a registry.

    Read from the parsed function rather than its text: the docstring explains
    that it runs no tool, and a grep for "tool" finds the explanation.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(task_manager.approve)))
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else
              getattr(node.func, "id", "")
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for forbidden in ("authorize", "run", "execute", "handle_user_input", "start_in_background"):
        assert forbidden not in called, f"approve() calls {forbidden}()"
    assert called <= {"get_task", "_replace", "_set_status", "get", "time"}, called


@pytest.mark.asyncio
async def test_an_irreversible_action_is_not_covered_by_approval_given_in_advance():
    """SafetyGuard downgrades pre-approval for a critical action. The task stops
    again, and says why rather than offering the same button forever."""
    task = _task(["delete the archive"])
    orchestrator = FakeOrchestrator()
    runner = TaskRunner(orchestrator)
    orchestrator.behaviour = lambda text, n: ConfirmationDenied(
        "'delete_file' is a 'critical' action, and this task ran unattended")
    await runner.run(task["id"])
    task_manager.approve(task["id"])
    await runner.run(task["id"])

    view = task_manager.detail(task_manager.get_task(task["id"]))
    assert view["status"] == WAITING_FOR_USER
    assert "already approved this once" in view["approval_request"]
    assert "irreversible" in view["approval_request"]


def test_rejecting_stops_the_task_and_records_the_refusal():
    task = _task()
    task_manager._set_status(task["id"], WAITING_FOR_USER,
                             blocked_reason="Step 1 needs your approval: send 4 emails")
    updated = task_manager.reject(task["id"])
    assert updated["status"] == CANCELLED
    assert "You declined" in updated["blocked_reason"]
    assert "send 4 emails" in updated["blocked_reason"]


def test_approve_and_reject_only_apply_to_a_task_that_is_asking():
    task = _task()
    assert task_manager.approve(task["id"]) is None
    assert task_manager.reject(task["id"]) is None


# --- It survives a restart -----------------------------------------------------------------

def test_state_is_on_disk_and_a_restart_pauses_what_was_mid_step():
    task = _task(["one", "two"])
    task["steps"][0].update(status="done", result="kept")
    task["current_step"] = 1
    task_manager._replace(task)
    task_manager._set_status(task["id"], RUNNING)

    # A restart: the store is re-read from disk, and anything that was running is
    # paused because nobody can know whether that step's side effects happened.
    recovered = task_manager.recover_interrupted()
    assert [t["id"] for t in recovered] == [task["id"]]

    stored = task_manager.get_task(task["id"])
    assert stored["status"] == PAUSED
    assert stored["steps"][0]["result"] == "kept"
    assert stored["current_step"] == 1


# --- The interface's controls --------------------------------------------------------------

class _Api:
    """gui/api.py's task methods, with no orchestrator behind them."""

    def __init__(self):
        from gui.api import LetiAPI

        self.api = LetiAPI.__new__(LetiAPI)

    def tasks(self):
        from gui.api import LetiAPI

        return LetiAPI.get_tasks(self.api)

    def control(self, task_id, action):
        from gui.api import LetiAPI

        return LetiAPI.control_task(self.api, task_id, action)


def test_the_panel_reads_the_task_store_and_nothing_else():
    task = _task(["one", "two"])
    view = _Api().tasks()
    assert [t["id"] for t in view["active"]] == [task["id"]]
    assert view["recent"] == []

    task_manager._set_status(task["id"], COMPLETED)
    view = _Api().tasks()
    assert view["active"] == []
    assert [t["id"] for t in view["recent"]] == [task["id"]]


@pytest.mark.parametrize("action", ["pause", "resume", "cancel", "retry", "approve", "reject"])
def test_every_control_goes_through_the_task_manager(action, monkeypatch):
    called = {}

    def spy(name):
        def _call(task_id, *a, **k):
            called["name"] = name
            return task_manager.get_task(task_id)
        return _call

    for name in ("pause", "resume", "cancel", "retry", "approve", "reject"):
        monkeypatch.setattr(task_manager, name, spy(name))

    task = _task()
    result = _Api().control(task["id"], action)
    assert result["ok"] is True
    assert called["name"] == action


def test_an_unknown_control_is_refused():
    task = _task()
    assert _Api().control(task["id"], "delete_everything")["ok"] is False


def test_a_control_that_does_not_apply_says_so_rather_than_pretending():
    task = _task()
    task_manager._set_status(task["id"], COMPLETED)
    result = _Api().control(task["id"], "pause")
    assert result["ok"] is False
    assert "can't be paused" in result["error"]
