"""Tasks that are remembered after they stop, and controls that really control.

Two properties run through all of these. A record never claims more than it
holds - "continue yesterday's task" against a record with no plan is answered
with what IS known and a question. And a control does what it says: a stop
stops future actions, and a cancelled task is never reported as completed.
"""
from __future__ import annotations

import json
import time

import pytest

from core import task_control, task_history, task_manager
from core.task_manager import (CANCELLED, CANCELLING, COMPLETED, FAILED, PAUSED,
                               PAUSING, QUEUED, RESUMING, RETRYING, RUNNING,
                               WAITING_FOR_USER, TaskRunner)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "history.json")
    yield tmp_path


def _task(steps=("do the thing",), name="test", **kwargs):
    return task_manager.create_task("an objective", list(steps), name, **kwargs)


class FakeOrchestrator:
    def __init__(self, behaviour=None):
        self.behaviour = behaviour or (lambda text, n: "ok")
        self.seen = []

    async def handle_user_input(self, text, session_id="default", voice_mode=False,
                                preapproved=False, owner=""):
        self.seen.append(text)
        result = self.behaviour(text, len(self.seen))
        if isinstance(result, Exception):
            raise result
        return result


# --- Persistence -------------------------------------------------------------------

def test_a_task_is_recorded_the_moment_its_status_changes(store):
    task = _task()
    task_manager.cancel(task["id"], "no longer needed")
    assert task_history.get(task["id"]) is not None


def test_a_record_keeps_what_the_spec_asks_for(store):
    task = _task(["gather", "write", "send"], success_criteria="the report is sent")
    task["steps"][0]["status"] = "done"
    task["steps"][1]["status"] = "failed"
    task["steps"][1]["recoveries"] = 2
    task["steps"][1]["error"] = "the server refused"
    task["current_step"] = 1
    task["verification"] = {"passed": False, "said": "NOT VERIFIED - nothing was sent"}
    task_manager._replace(task)
    task_manager.cancel(task["id"], "you stopped it")

    record = task_history.get(task["id"])
    assert record["task_id"] == task["id"]
    assert record["request"] and record["goal"] == "the report is sent"
    assert record["created_at"] and record["finished_at"]
    assert record["status"] == CANCELLED
    assert record["plan"] == ["gather", "write", "send"]
    assert record["completed_steps"] == [1] and record["failed_steps"] == [2]
    assert record["recoveries"] and record["recoveries"][0]["attempts"] == 2
    assert record["verification"]["passed"] is False
    assert "you stopped it" in record["reason"]


def test_a_record_is_compact(store):
    task = _task(["x" * 5000], name="huge")
    task["result"] = "y" * 20_000
    task_manager._replace(task)
    task_manager.cancel(task["id"])
    blob = json.dumps(task_history.get(task["id"]))
    assert len(blob) < 4000, f"a history record grew to {len(blob)} characters"


def test_no_conversation_is_copied_into_the_history(store):
    task = _task()
    task_manager.cancel(task["id"])
    record = task_history.get(task["id"])
    assert "messages" not in record and "conversation" not in record


def test_one_record_per_task_however_many_transitions(store):
    task = _task()
    task_manager.pause(task["id"])
    task_manager.resume(task["id"])
    task_manager.cancel(task["id"])
    assert len([r for r in task_history.load() if r["task_id"] == task["id"]]) == 1


def test_the_history_survives_the_task_store_being_trimmed(store, monkeypatch):
    monkeypatch.setattr(task_manager, "MAX_TASKS_KEPT", 2)
    ids = []
    for i in range(5):
        task = _task(name=f"task {i}")
        task_manager.cancel(task["id"])
        ids.append(task["id"])
    assert len(task_manager.load_tasks()) <= 2
    assert all(task_history.get(i) for i in ids)


def test_the_history_is_bounded(store, monkeypatch):
    monkeypatch.setattr(task_history, "MAX_RECORDS", 5)
    for i in range(12):
        task_manager.cancel(_task(name=f"t{i}")["id"])
    assert len(task_history.load()) <= 5


# --- Corrupt and missing records -----------------------------------------------------

def test_a_corrupt_history_file_is_empty_not_an_exception(store):
    (store / "history.json").write_text("{not json at all")
    assert task_history.load() == []


def test_a_history_file_holding_the_wrong_shape_is_ignored(store):
    (store / "history.json").write_text('{"a": 1}')
    assert task_history.load() == []


def test_one_malformed_record_does_not_cost_the_others(store):
    (store / "history.json").write_text(json.dumps(
        [{"task_id": "good", "name": "kept"}, {"no_id": True}, "not even a dict"]))
    loaded = task_history.load()
    assert len(loaded) == 1 and loaded[0]["task_id"] == "good"


def test_a_missing_record_says_so_rather_than_inventing_one(store):
    described = task_history.describe(task_history.get("nope"))
    assert described["found"] is False and described["resumable"] is False


def test_a_record_with_no_plan_is_not_offered_for_continuation(store):
    """A record from an older version, or one whose plan never made it. Leti must
    not offer to continue something it cannot describe."""
    (store / "history.json").write_text(json.dumps(
        [{"task_id": "old", "name": "something from before", "status": "failed"}]))
    described = task_history.describe(task_history.get("old"))
    assert described["resumable"] is False
    assert "the plan was not recorded" in described["cannot_say"]


def test_a_record_with_steps_left_is_resumable_and_says_which(store):
    task = _task(["gather", "write", "send"])
    task["steps"][0]["status"] = "done"
    task_manager._replace(task)
    task_manager.cancel(task["id"])
    described = task_history.describe(task_history.get(task["id"]))
    assert described["resumable"] is True
    assert described["remaining_steps"] == ["write", "send"]


def test_a_finished_task_is_not_resumable(store):
    task = _task(["only step"])
    task["steps"][0]["status"] = "done"
    task_manager._replace(task)
    task_manager._set_status(task["id"], COMPLETED)
    assert task_history.describe(task_history.get(task["id"]))["resumable"] is False


def test_finding_a_task_by_what_the_user_calls_it(store):
    task_manager.cancel(_task(name="website generator")["id"])
    task_manager.cancel(_task(name="quarterly report")["id"])
    found = task_history.find("website")
    assert len(found) == 1 and found[0]["name"] == "website generator"


def test_finding_nothing_returns_nothing_rather_than_the_nearest(store):
    task_manager.cancel(_task(name="website generator")["id"])
    assert task_history.find("wakanda industries") == []


# --- Lifecycle states ----------------------------------------------------------------

def test_pausing_a_running_task_goes_through_pausing(store):
    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    assert task_manager.pause(task["id"])["status"] == PAUSING
    assert task_manager.stop_requested(task["id"]) is True


def test_pausing_a_queued_task_is_immediate(store):
    assert task_manager.pause(_task()["id"])["status"] == PAUSED


def test_cancelling_a_running_task_goes_through_cancelling(store):
    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    assert task_manager.cancel(task["id"])["status"] == CANCELLING


def test_cancelling_an_idle_task_is_immediate(store):
    assert task_manager.cancel(_task()["id"])["status"] == CANCELLED


def test_resuming_goes_through_resuming(store):
    task = _task()
    task_manager.pause(task["id"])
    assert task_manager.resume(task["id"])["status"] == RESUMING


def test_a_cancelled_task_records_what_had_already_run(store):
    task = _task(["a", "b", "c"])
    task["steps"][0]["status"] = "done"
    task["steps"][1]["status"] = "done"
    task_manager._replace(task)
    cancelled = task_manager.cancel(task["id"], "you stopped it")
    assert cancelled["cancelled_after_steps"] == [1, 2]
    assert "1, 2" in cancelled["blocked_reason"]


def test_a_cancelled_task_that_did_nothing_says_so(store):
    cancelled = task_manager.cancel(_task()["id"])
    assert "Nothing had been carried out yet" in cancelled["blocked_reason"]


def test_a_cancelled_task_is_never_reported_as_completed(store):
    task = _task()
    task_manager.cancel(task["id"])
    stored = task_manager.get_task(task["id"])
    assert stored["status"] == CANCELLED
    assert stored.get("result") is None
    assert task_history.get(task["id"])["status"] == CANCELLED


def test_skipping_marks_the_step_skipped_not_done(store):
    task = _task(["a", "b"])
    task_manager.pause(task["id"])
    task_manager.skip_step(task["id"], "not needed")
    stored = task_manager.get_task(task["id"])
    assert stored["steps"][0]["status"] == "skipped"
    assert stored["current_step"] == 1
    assert task_history.get(task["id"])["skipped_steps"] == [1]


@pytest.mark.parametrize("control,from_status", [
    ("pause", COMPLETED), ("pause", CANCELLED), ("resume", RUNNING),
    ("approve", RUNNING), ("reject", PAUSED),
])
def test_a_control_refuses_from_a_state_it_does_not_apply_to(store, control, from_status):
    task = _task()
    task_manager._set_status(task["id"], from_status)
    move = {"pause": task_manager.pause, "resume": task_manager.resume,
            "approve": task_manager.approve, "reject": task_manager.reject}[control]
    assert move(task["id"]) is None


# --- Stop actually stops -------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancelling_mid_run_prevents_every_later_step(store):
    task = _task(["one", "two", "three", "four"])
    orchestrator = FakeOrchestrator()

    def cancel_during_the_first(text, n):
        if n == 1:
            task_manager.cancel(task["id"], "stop now")
        return "ok"

    orchestrator.behaviour = cancel_during_the_first
    await TaskRunner(orchestrator).run(task["id"])

    stored = task_manager.get_task(task["id"])
    assert stored["status"] == CANCELLED
    assert len(orchestrator.seen) == 1, "a cancelled task kept running steps"
    assert stored["steps"][1]["status"] == "pending"


@pytest.mark.asyncio
async def test_the_step_in_flight_is_recorded_honestly_not_erased(store):
    task = _task(["one", "two"])
    orchestrator = FakeOrchestrator()

    def cancel_during_the_first(text, n):
        task_manager.cancel(task["id"])
        return "the first step really did happen"

    orchestrator.behaviour = cancel_during_the_first
    await TaskRunner(orchestrator).run(task["id"])
    stored = task_manager.get_task(task["id"])
    assert stored["steps"][0]["status"] == "done"
    assert "really did happen" in stored["steps"][0]["result"]
    assert stored["current_step"] == 1, "a resumed task would repeat a finished step"


@pytest.mark.asyncio
async def test_pausing_mid_run_settles_to_paused_not_pausing(store):
    task = _task(["one", "two", "three"])
    orchestrator = FakeOrchestrator()

    def pause_during_the_first(text, n):
        if n == 1:
            task_manager.pause(task["id"])
        return "ok"

    orchestrator.behaviour = pause_during_the_first
    await TaskRunner(orchestrator).run(task["id"])
    assert task_manager.get_task(task["id"])["status"] == PAUSED
    assert len(orchestrator.seen) == 1


@pytest.mark.asyncio
async def test_a_cancelled_task_is_not_verified_afterwards(store):
    task = _task(["one"], success_criteria="it is done")
    orchestrator = FakeOrchestrator()

    def cancel_immediately(text, n):
        task_manager.cancel(task["id"])
        return "ok"

    orchestrator.behaviour = cancel_immediately
    await TaskRunner(orchestrator).run(task["id"])
    assert task_manager.get_task(task["id"])["status"] == CANCELLED
    assert len(orchestrator.seen) == 1, "a cancelled task was still verified"


def test_settle_stop_is_the_one_exit_from_a_transitional_state(store):
    """The in-loop check and the finally block both call it, so a mutation of
    either alone is invisible. This pins the function itself."""
    runner = TaskRunner(FakeOrchestrator())

    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    task_manager.pause(task["id"])
    assert runner._settle_stop(task["id"]) is True
    assert task_manager.get_task(task["id"])["status"] == PAUSED

    other = _task()
    task_manager._set_status(other["id"], RUNNING)
    task_manager.cancel(other["id"])
    assert runner._settle_stop(other["id"]) is True
    assert task_manager.get_task(other["id"])["status"] == CANCELLED

    ordinary = _task()
    task_manager._set_status(ordinary["id"], RUNNING)
    assert runner._settle_stop(ordinary["id"]) is False


def test_a_stop_requested_without_a_status_change_still_settles(store):
    """The in-memory set is the fast path; it must be honoured on its own."""
    runner = TaskRunner(FakeOrchestrator())
    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    task_manager._stop_requested.add(task["id"])
    try:
        assert runner._settle_stop(task["id"]) is True
        assert task_manager.get_task(task["id"])["status"] == CANCELLED
    finally:
        task_manager.clear_stop(task["id"])


# --- Restart -------------------------------------------------------------------------

def _as_if_restarted():
    """Become a different process, which is what a restart is.

    Checkpoints carry the id of the process that wrote them, so "written by a
    process that is gone" is a comparison rather than a guess - and simulating
    a restart honestly means changing the id, not just the status.
    """
    from core import checkpoints

    checkpoints.new_generation_for_tests()


@pytest.mark.parametrize("left_in,becomes", [
    (PAUSING, PAUSED), (RESUMING, PAUSED), (CANCELLING, CANCELLED),
])
def test_a_restart_resolves_every_transitional_state(store, left_in, becomes):
    task = _task()
    task_manager._set_status(task["id"], left_in)
    _as_if_restarted()
    task_manager.recover_interrupted()
    assert task_manager.get_task(task["id"])["status"] == becomes


def test_a_task_interrupted_before_any_step_is_ready_to_resume(store):
    """Nothing was in flight, so starting it again repeats nothing."""
    from core import checkpoints

    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    _as_if_restarted()
    recovered = task_manager.recover_interrupted()
    assert recovered[0]["recovery"]["state"] == checkpoints.READY_TO_RESUME
    assert task_manager.get_task(task["id"])["status"] == PAUSED


def test_a_restart_never_silently_resumes(store):
    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    _as_if_restarted()
    recovered = task_manager.recover_interrupted()
    assert recovered and "interrupted" in recovered[0]["blocked_reason"].lower()
    # Stopped, whatever the verdict. Recovery decides what a task IS, never that
    # it should run.
    assert task_manager.get_task(task["id"])["status"] in (PAUSED, WAITING_FOR_USER)


def test_a_restart_writes_what_it_found_to_the_history(store):
    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    _as_if_restarted()
    task_manager.recover_interrupted()
    assert task_history.get(task["id"])["status"] in (PAUSED, WAITING_FOR_USER)


def test_a_task_of_this_process_is_not_treated_as_interrupted(store):
    """A running task belongs to the process running it. Only a checkpoint from
    a process that is gone describes an interruption."""
    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    assert task_manager.recover_interrupted() == []
    assert task_manager.get_task(task["id"])["status"] == RUNNING


# --- Reaching the right task ----------------------------------------------------------

def test_a_command_with_nothing_running_says_so(store):
    result = task_control.apply({"action": "pause", "hint": ""})
    assert result["ok"] is False and "nothing" in result["answer"].lower()


def test_a_command_naming_one_task_moves_that_task(store):
    task = _task(name="the download")
    _task(name="the report")
    result = task_control.apply({"action": "pause", "hint": "download"})
    assert result["ok"] is True
    assert task_manager.get_task(task["id"])["status"] == PAUSED


def test_an_ambiguous_command_asks_rather_than_guessing(store):
    _task(name="the first report")
    _task(name="the second report")
    result = task_control.apply({"action": "pause", "hint": "report"})
    assert result["ok"] is False and result.get("needs_choice") is True
    assert "which one" in result["answer"].lower()


def test_stop_with_nothing_named_stops_everything(store):
    a, b = _task(name="one"), _task(name="two")
    result = task_control.apply({"action": "stop", "hint": "", "everything": False})
    assert result["ok"] is True
    for task in (a, b):
        assert task_manager.get_task(task["id"])["status"] == CANCELLED


def test_stop_says_what_it_can_and_cannot_undo(store):
    _task(name="one")
    answer = task_control.apply({"action": "stop", "hint": ""})["answer"]
    assert "Nothing further will start" in answer
    assert "has still happened" in answer


def test_resume_says_when_nothing_will_pick_it_up(store):
    task = _task(name="one")
    task_manager.pause(task["id"])
    answer = task_control.apply({"action": "resume", "hint": "one"}, runner=None)["answer"]
    assert "nothing is running it" in answer


def test_a_broken_control_never_takes_the_turn_with_it(store, monkeypatch):
    monkeypatch.setattr(task_control, "_candidates",
                        lambda action, hint: (_ for _ in ()).throw(RuntimeError("boom")))
    result = task_control.apply({"action": "pause", "hint": "x"})
    assert result["handled"] is True and result["ok"] is False


def test_an_unknown_action_falls_through_to_an_ordinary_turn(store):
    assert task_control.apply({"action": "juggle", "hint": ""}) is None


def test_the_status_report_counts_what_is_there(store):
    _task(name="running one")
    blocked = _task(name="blocked one")
    task_manager._set_status(blocked["id"], WAITING_FOR_USER,
                             blocked_reason="needs your approval")
    report = task_control.status_report()
    assert "2 active tasks" in report and "1 needs you" in report


def test_the_waiting_report_only_lists_what_a_person_can_act_on(store):
    _task(name="just running")
    blocked = _task(name="blocked one")
    task_manager._set_status(blocked["id"], WAITING_FOR_USER,
                             blocked_reason="needs your approval")
    report = task_control.waiting_report()
    assert "blocked one" in report and "just running" not in report


def test_nothing_waiting_says_nothing_is_waiting(store):
    _task(name="running")
    assert "Nothing is waiting" in task_control.waiting_report()
