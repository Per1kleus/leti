"""Trying again, and knowing when not to.

The shape these all guard: recovery is BOUNDED and DIAGNOSED. It never retries a
refusal, never spins on an identical failure, never exceeds its budget, and when
it stops it says the five things a person needs rather than "it failed".
"""
from __future__ import annotations

import pytest

from core import recovery, task_history, task_manager, world_state
from core.safety_guard import ConfirmationDenied, PermissionDenied
from core.task_manager import FAILED, RETRYING, TaskRunner


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "history.json")
    yield tmp_path


class FakeOrchestrator:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.seen = []

    async def handle_user_input(self, text, session_id="default", voice_mode=False,
                               preapproved=False):
        self.seen.append(text)
        result = self.behaviour(text, len(self.seen))
        if isinstance(result, Exception):
            raise result
        return result


# --- The decision ------------------------------------------------------------------

@pytest.mark.parametrize("problem", [
    ConfirmationDenied("declined"), PermissionDenied("forbidden path"),
    PermissionError("Permission denied"),
])
def test_a_refusal_is_never_recovered_from(problem):
    decision = recovery.plan(problem)
    assert decision["recover"] is False
    assert decision["decision"] == recovery.ESCALATE
    assert "around a refusal" in decision["say"] or "refused" in decision["diagnosis"]
    # And the specific instruction, not just the refusal to retry: a different
    # route to something that was refused is the thing that was refused.
    assert "a way around a refusal" in decision["say"].lower()


def test_a_refusal_stays_unrecoverable_even_without_its_own_entry(monkeypatch):
    """Defence in depth. _NO_RETRY names refusals explicitly; plan() also fails
    closed on any kind with no recovery option, so removing one does not remove
    the guarantee. This pins the second mechanism, which a mutation of the
    first would otherwise walk straight past."""
    monkeypatch.setattr(recovery, "_NO_RETRY", {})
    decision = recovery.plan(PermissionDenied("forbidden path"))
    assert decision["recover"] is False and decision["decision"] == recovery.ESCALATE


def test_an_unrecognised_kind_fails_closed(monkeypatch):
    monkeypatch.setattr(recovery, "_NO_RETRY", {})
    monkeypatch.setattr(recovery, "_OPTIONS", {})
    for problem in (TimeoutError("timed out"), RuntimeError("anything at all")):
        assert recovery.plan(problem)["recover"] is False


@pytest.mark.parametrize("problem,expected", [
    (TimeoutError("timed out"), True),
    (RuntimeError("503 Service Unavailable"), True),
    (RuntimeError("element not found on screen"), True),
    (RuntimeError("the folder no longer exists"), True),
    (ModuleNotFoundError("No module named 'x'"), False),
    (RuntimeError("something nobody has ever seen"), False),
    (ValueError("missing required argument"), False),
])
def test_only_the_kinds_worth_another_attempt_recover(problem, expected):
    assert recovery.plan(problem)["recover"] is expected


def test_an_unknown_cause_escalates_rather_than_guessing():
    decision = recovery.plan(RuntimeError("the flange is purple"))
    assert decision["recover"] is False
    assert decision["kind"] == world_state.UNKNOWN
    assert decision["certain"] is False


def test_an_ambiguous_target_asks_rather_than_picking():
    decision = recovery.plan(RuntimeError("which one did you mean"))
    assert decision["decision"] in (recovery.ASK, recovery.ESCALATE)
    assert decision["recover"] is False


def test_the_budget_stops_it():
    decision = recovery.plan(TimeoutError("timed out"), attempts_used=3, max_attempts=3)
    assert decision["recover"] is False
    assert "budget" in decision["diagnosis"]


def test_nothing_may_exceed_the_absolute_ceiling():
    decision = recovery.plan(TimeoutError("timed out"),
                             attempts_used=recovery.MAX_ATTEMPTS,
                             max_attempts=999)
    assert decision["recover"] is False


def test_an_identical_failure_ends_recovery_before_the_budget_does():
    problem = TimeoutError("timed out")
    fingerprint = recovery.plan(problem)["fingerprint"]
    decision = recovery.plan(problem, attempts_used=0,
                             previous_failures=[fingerprint])
    assert decision["recover"] is False
    assert "unchanged" in decision["diagnosis"]


def test_a_cancelled_task_recovers_from_nothing():
    decision = recovery.plan(TimeoutError("timed out"), cancelled=True)
    assert decision["recover"] is False and "cancelled" in decision["diagnosis"]


def test_escalation_names_the_five_things():
    escalation = recovery.plan(PermissionDenied("nope"))["escalation"]
    for field in ("what_failed", "classified_as", "what_was_attempted",
                  "why_it_stopped", "user_intervention_required"):
        assert field in escalation


def test_the_recovery_instruction_says_what_to_do_differently():
    decision = recovery.plan(TimeoutError("timed out"))
    instruction = recovery.instruction(decision)
    assert "different way" in instruction
    assert "Do not repeat the approach that just failed" in instruction


def test_no_instruction_is_produced_when_there_is_nothing_to_try():
    assert recovery.instruction(recovery.plan(PermissionDenied("nope"))) == ""


def test_the_planner_executes_nothing():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/recovery.py").read_text())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("run", "start", "create_task", "authorize", "Thread",
                      "create_task", "sleep"):
        assert forbidden not in called, f"core/recovery.py calls {forbidden}"


def test_planning_never_raises_on_anything():
    for problem in (None, "", 0, [], object(), RuntimeError(), Exception("x" * 5000)):
        decision = recovery.plan(problem)
        assert decision["decision"] in (recovery.CONTINUE, recovery.ESCALATE,
                                        recovery.ASK)


# --- In a running task ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_recovered_step_is_retried_differently_and_carries_on(store):
    task = task_manager.create_task("o", ["fetch the page"], "t")

    def down_then_up(text, n):
        if "different way" not in text:
            return RuntimeError("the site is temporarily unavailable")
        return "done"

    result = await TaskRunner(FakeOrchestrator(down_then_up)).run(task["id"])
    assert result["status"] == "completed"
    assert result["recoveries"] >= 1


@pytest.mark.asyncio
async def test_a_task_shows_retrying_while_it_recovers(store):
    task = task_manager.create_task("o", ["fetch"], "t")
    seen_states = []

    def watch_state(text, n):
        seen_states.append(task_manager.get_task(task["id"])["status"])
        if "different way" not in text:
            return RuntimeError("timed out")
        return "done"

    await TaskRunner(FakeOrchestrator(watch_state)).run(task["id"])
    stored = task_manager.get_task(task["id"])
    log = stored["steps"][0].get("recovery_log") or []
    assert log, "no recovery was recorded"
    # RETRYING is what the store held between the failure and the next attempt,
    # which is what the interface reads. The attempt itself runs as RUNNING.
    assert RETRYING in task_manager.ALLOWED_FROM["cancel"]
    assert stored["steps"][0]["recoveries"] >= 1


@pytest.mark.asyncio
async def test_every_recovery_attempt_is_on_the_record(store):
    task = task_manager.create_task("o", ["fetch"], "t")

    def always_down(text, n):
        return RuntimeError("503 unavailable")

    await TaskRunner(FakeOrchestrator(always_down)).run(task["id"])
    step = task_manager.get_task(task["id"])["steps"][0]
    assert step["recovery_log"], "recovery attempts were not recorded"
    assert all("kind" in entry and "decision" in entry for entry in step["recovery_log"])


@pytest.mark.asyncio
async def test_the_recovery_shows_up_in_the_task_history(store):
    task = task_manager.create_task("o", ["fetch"], "t")
    await TaskRunner(FakeOrchestrator(
        lambda text, n: RuntimeError("connection reset"))).run(task["id"])
    record = task_history.get(task["id"])
    assert record["recoveries"], "the history hides the recovery attempts"


@pytest.mark.asyncio
async def test_a_permission_failure_is_never_retried_in_a_running_task(store):
    task = task_manager.create_task("o", ["do the forbidden thing"], "t")
    seen = []

    def forbidden(text, n):
        seen.append(text)
        return PermissionDenied("forbidden path")

    result = await TaskRunner(FakeOrchestrator(forbidden)).run(task["id"])
    assert result["status"] == FAILED
    assert not any("different way" in t for t in seen), "a refusal was retried"


@pytest.mark.asyncio
async def test_a_failed_task_says_what_remains_incomplete(store):
    task = task_manager.create_task("o", ["one", "two", "three"], "t")
    await TaskRunner(FakeOrchestrator(
        lambda text, n: RuntimeError("no idea"))).run(task["id"])
    stored = task_manager.get_task(task["id"])
    assert stored["escalation"]["still_incomplete"], "nothing said what was left undone"
    assert stored["escalation"]["classified_as"] == world_state.UNKNOWN


@pytest.mark.asyncio
async def test_recovery_stops_when_the_task_is_cancelled_mid_recovery(store):
    task = task_manager.create_task("o", ["fetch"], "t")
    seen = []

    def fail_then_cancel(text, n):
        seen.append(text)
        if n == 1:
            return RuntimeError("timed out")
        task_manager.cancel(task["id"])
        return RuntimeError("timed out")

    await TaskRunner(FakeOrchestrator(fail_then_cancel)).run(task["id"])
    assert task_manager.get_task(task["id"])["status"] == "cancelled"
    assert len(seen) <= 4, f"recovery kept going after a cancel: {len(seen)} attempts"
