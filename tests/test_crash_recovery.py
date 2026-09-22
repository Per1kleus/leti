"""Coming back after the process died, without repeating what may already have happened.

The guarantee: a step that was in flight when Leti stopped is UNKNOWN, not
failed and not done. Nothing irreversible is repeated on the strength of a
guess, and no task resumes itself.
"""
from __future__ import annotations

import json
import time

import pytest

from core import checkpoints as cp
from core import resources, task_history, task_manager
from core.task_manager import (CANCELLED, PAUSED, RUNNING, WAITING_FOR_USER,
                               TaskRunner)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "history.json")
    resources.clear()
    yield tmp_path
    resources.clear()


def _task(steps=("do the thing",), name="test", **kwargs):
    return task_manager.create_task("an objective", list(steps), name, **kwargs)


def _crash():
    """What a restart is: a different process reading the same store."""
    cp.new_generation_for_tests()


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


# --- The checkpoint itself -------------------------------------------------------------

def test_a_checkpoint_holds_what_recovery_needs(store):
    task = _task(["gather", "write", "send"], success_criteria="it is sent")
    task["steps"][0]["status"] = "done"
    task["current_step"] = 1
    written = cp.write(task, cp.STARTED, note="step 2")
    for field in ("task_id", "goal", "status", "step_index", "step", "execution",
                  "verified_steps", "failed_steps", "recoveries",
                  "last_verification", "resources", "irreversible",
                  "generation", "at"):
        assert field in written, f"a checkpoint has no {field}"
    assert written["verified_steps"] == [1]
    assert written["execution"] == cp.STARTED


def test_a_checkpoint_is_compact(store):
    task = _task(["x" * 5000])
    task["result"] = "y" * 20_000
    written = cp.write(task, cp.STARTED)
    assert len(json.dumps(written)) < 2000


def test_a_checkpoint_holds_no_screenshots_or_context(store):
    task = _task()
    written = json.dumps(cp.write(task, cp.STARTED))
    for forbidden in ("screenshot", "messages", "conversation", "base64"):
        assert forbidden not in written


def test_an_irreversible_step_is_marked(store):
    reversible = _task(["read the report"])
    assert cp.write(reversible, cp.STARTED)["irreversible"] is False
    for text in ("email it to chris", "send the invoice", "delete the archive",
                 "pay the supplier", "commit and push"):
        task = _task([text])
        assert cp.write(task, cp.STARTED)["irreversible"] is True, text


def test_an_unreadable_step_is_treated_as_irreversible(monkeypatch, store):
    """Fail closed: unable to tell means assume repeating it would cost something."""
    monkeypatch.setattr("core.computer_use.is_consequential",
                        lambda text: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cp.write(_task(["anything"]), cp.STARTED)["irreversible"] is True


def test_an_unknown_execution_state_is_refused(store):
    with pytest.raises(ValueError):
        cp.write(_task(), "SOMETHING_ELSE")


def test_verified_is_the_shared_word():
    from core import verification

    assert cp.VERIFIED is verification.VERIFIED


# --- Generations ------------------------------------------------------------------------

def test_a_checkpoint_from_this_process_is_not_an_interruption(store):
    task = _task()
    written = cp.write(task, cp.STARTED)
    assert cp.from_another_process(written) is False


def test_a_checkpoint_from_a_dead_process_is_an_interruption(store):
    written = cp.write(_task(), cp.STARTED)
    _crash()
    assert cp.from_another_process(written) is True


def test_a_checkpoint_with_no_generation_is_treated_as_foreign():
    assert cp.from_another_process({"task_id": "x", "execution": cp.STARTED}) is True


# --- Assessment ---------------------------------------------------------------------------

def test_a_crash_before_any_step_is_ready_to_resume(store):
    task = _task(["one", "two"])
    task["checkpoint"] = cp.write(task, cp.NOT_STARTED)
    verdict = cp.assess(task)
    assert verdict["state"] == cp.READY_TO_RESUME and verdict["resume"] is True


def test_a_crash_after_a_verified_step_is_ready_to_resume(store):
    task = _task(["one", "two"])
    task["steps"][0]["status"] = "done"
    task["current_step"] = 1
    task["checkpoint"] = cp.write(task, cp.VERIFIED)
    verdict = cp.assess(task)
    assert verdict["state"] == cp.READY_TO_RESUME
    assert verdict["last_verified_step"] == 1
    assert verdict["unknown_external_effect"] is False


def test_a_crash_during_a_step_needs_a_person(store):
    task = _task(["one", "two"])
    task["checkpoint"] = cp.write(task, cp.STARTED)
    verdict = cp.assess(task)
    assert verdict["state"] == cp.NEEDS_YOU and verdict["resume"] is False
    assert verdict["unknown_external_effect"] is True


def test_a_crash_during_an_irreversible_step_says_what_is_at_stake(store):
    task = _task(["email the invoice to chris"])
    task["checkpoint"] = cp.write(task, cp.STARTED)
    verdict = cp.assess(task)
    assert verdict["state"] == cp.NEEDS_YOU
    assert any("twice" in reason for reason in verdict["why"])


def test_unknown_after_crash_is_neither_failed_nor_done(store):
    task = _task(["one"])
    task["checkpoint"] = {**cp.write(task, cp.STARTED),
                          "execution": cp.UNKNOWN_AFTER_CRASH}
    verdict = cp.assess(task)
    assert verdict["state"] == cp.NEEDS_YOU
    assert verdict["unknown_external_effect"] is True


def test_no_checkpoint_at_all_needs_a_person(store):
    verdict = cp.assess(_task())
    assert verdict["state"] == cp.NEEDS_YOU
    assert "no checkpoint" in verdict["why"][0]


def test_a_changed_permission_blocks_resuming(store):
    task = _task(["one"])
    task["checkpoint"] = cp.write(task, cp.VERIFIED)
    verdict = cp.assess(task, permitted=False)
    assert verdict["state"] == cp.NEEDS_YOU
    assert any("permission" in reason for reason in verdict["why"])


def test_an_unavailable_connection_blocks_resuming(store):
    task = _task(["one"])
    task["checkpoint"] = cp.write(task, cp.VERIFIED)
    verdict = cp.assess(task, connections_ready=False)
    assert verdict["state"] == cp.NEEDS_YOU
    assert any("connection" in reason for reason in verdict["why"])


def test_unknown_conditions_do_not_block(store):
    """None means "cannot tell", which is not an objection - only a definite
    False stops a resume, because guessing would strand every task."""
    task = _task(["one"])
    task["checkpoint"] = cp.write(task, cp.VERIFIED)
    assert cp.assess(task, permitted=None, connections_ready=None)["resume"] is True


# --- Corrupt and partial checkpoints ---------------------------------------------------------

@pytest.mark.parametrize("bad", [
    None, "", 5, [], {}, {"execution": cp.STARTED},
    {"task_id": "x"}, {"task_id": "x", "execution": "nonsense"},
])
def test_a_corrupt_checkpoint_reads_as_absent(bad, store):
    assert cp.of({"id": "x", "checkpoint": bad}) is None


def test_a_task_with_a_corrupt_checkpoint_needs_a_person(store):
    task = _task()
    task["checkpoint"] = {"half": "written"}
    assert cp.assess(task)["state"] == cp.NEEDS_YOU


def test_a_partially_written_store_is_never_read(store):
    """Checkpoints ride the existing atomic write, so a reader sees the old file
    or the new one - never half of one."""
    import inspect

    source = inspect.getsource(task_manager.save_tasks)
    assert "atomic_write_text" in source


# --- Restart ------------------------------------------------------------------------------

def test_a_restart_assesses_each_task_individually(store):
    safe = _task(["one", "two"], name="safe")
    safe["steps"][0]["status"] = "done"
    safe["current_step"] = 1
    task_manager._replace(safe)
    task_manager._set_status(safe["id"], RUNNING, execution=cp.VERIFIED)

    risky = _task(["email the invoice"], name="risky")
    task_manager._set_status(risky["id"], RUNNING, execution=cp.STARTED)

    _crash()
    recovered = task_manager.recover_interrupted()
    states = {t["name"]: t["recovery"]["state"] for t in recovered}
    assert states["safe"] == cp.READY_TO_RESUME
    assert states["risky"] == cp.NEEDS_YOU


def test_a_restart_never_resumes_anything_by_itself(store):
    task = _task(["one", "two"])
    task_manager._set_status(task["id"], RUNNING, execution=cp.VERIFIED)
    _crash()
    task_manager.recover_interrupted()
    stored = task_manager.get_task(task["id"])
    assert stored["status"] in (PAUSED, WAITING_FOR_USER)
    assert stored["status"] != RUNNING


def test_a_task_mid_irreversible_step_ends_needing_a_person(store):
    task = _task(["send the payment"])
    task_manager._set_status(task["id"], RUNNING, execution=cp.STARTED)
    _crash()
    task_manager.recover_interrupted()
    stored = task_manager.get_task(task["id"])
    assert stored["status"] == WAITING_FOR_USER
    assert stored["recovery"]["unknown_external_effect"] is True


def test_a_cancelling_task_stays_cancelled_after_a_crash(store):
    task = _task()
    task_manager._set_status(task["id"], RUNNING)
    task_manager.cancel(task["id"])
    _crash()
    task_manager.recover_interrupted()
    assert task_manager.get_task(task["id"])["status"] == CANCELLED


def test_a_crash_releases_every_lock(store):
    task = _task()
    resources.acquire(resources.identity(resources.FILE, "/tmp/x"),
                      resources.WRITE, task["id"])
    task_manager._set_status(task["id"], RUNNING, execution=cp.STARTED)
    _crash()
    task_manager.recover_interrupted()
    assert resources.snapshot()["held"] == []
    assert resources.acquire(resources.identity(resources.FILE, "/tmp/x"),
                             resources.WRITE, "somebody-else")["acquired"] is True


def test_what_a_crashed_task_held_is_in_its_checkpoint(store):
    task = _task()
    resources.acquire(resources.identity(resources.FILE, "/tmp/x"),
                      resources.WRITE, task["id"])
    written = cp.write(task, cp.STARTED)
    assert written["resources"] and written["resources"][0]["mode"] == resources.WRITE


def test_finished_tasks_are_left_alone_by_a_restart(store):
    done = _task(name="done")
    task_manager._set_status(done["id"], "completed")
    _crash()
    assert task_manager.recover_interrupted() == []


def test_three_interrupted_tasks_are_restored_individually(store):
    a = _task(["one", "two"], name="A")
    a["steps"][0]["status"] = "done"
    a["current_step"] = 1
    task_manager._replace(a)
    task_manager._set_status(a["id"], RUNNING, execution=cp.VERIFIED)

    b = _task(["send the report"], name="B")
    task_manager._set_status(b["id"], RUNNING, execution=cp.STARTED)

    c = _task(["one"], name="C")
    task_manager._set_status(c["id"], WAITING_FOR_USER,
                             blocked_reason="needs your approval")

    _crash()
    recovered = task_manager.recover_interrupted()
    states = {t["name"]: t["recovery"]["state"] for t in recovered}
    assert states["A"] == cp.READY_TO_RESUME
    assert states["B"] == cp.NEEDS_YOU
    assert states["C"] in (cp.READY_TO_RESUME, cp.NEEDS_YOU)


def test_the_user_is_told_how_many_were_interrupted(store):
    for i in range(2):
        task = _task(name=f"task {i}")
        task_manager._set_status(task["id"], RUNNING, execution=cp.VERIFIED)
    _crash()
    summary = task_manager.interrupted_summary(task_manager.recover_interrupted())
    assert "2 tasks interrupted by the previous session" in summary
    assert "ready to resume" in summary


def test_nothing_interrupted_says_nothing(store):
    assert task_manager.interrupted_summary([]) == ""


# --- Checkpoints in a running task ---------------------------------------------------------

@pytest.mark.asyncio
async def test_a_step_is_marked_started_before_it_runs(store):
    task = _task(["one", "two"])
    seen = []

    def look(text, n):
        stored = task_manager.get_task(task["id"])
        seen.append(stored["checkpoint"]["execution"])
        return "ok"

    await TaskRunner(FakeOrchestrator(look)).run(task["id"])
    assert seen and seen[0] == cp.STARTED, "a step ran with no checkpoint before it"


@pytest.mark.asyncio
async def test_a_finished_step_is_marked_verified(store):
    task = _task(["one", "two"])
    await TaskRunner(FakeOrchestrator()).run(task["id"])
    stored = task_manager.get_task(task["id"])
    assert stored["checkpoint"]["verified_steps"]


@pytest.mark.asyncio
async def test_a_crash_between_a_step_and_its_checkpoint_is_unknown(store):
    """The dangerous window. The step ran; the process died before the result
    was recorded. What it did is unknown, and that is what gets said."""
    task = _task(["send the report", "tidy up"])
    orchestrator = FakeOrchestrator()

    def die_after_sending(text, n):
        # The step "happened"; the process never got to write VERIFIED.
        raise KeyboardInterrupt("force-closed")

    orchestrator.behaviour = die_after_sending
    try:
        await TaskRunner(orchestrator).run(task["id"])
    except (KeyboardInterrupt, BaseException):
        pass

    stored = task_manager.get_task(task["id"])
    assert stored["checkpoint"]["execution"] == cp.STARTED
    _crash()
    task_manager.recover_interrupted()
    after = task_manager.get_task(task["id"])
    assert after["recovery"]["state"] == cp.NEEDS_YOU
    assert after["recovery"]["unknown_external_effect"] is True


def test_checkpoint_writing_is_cheap(store):
    task = _task(["one", "two", "three"])
    cp.write(task, cp.STARTED)
    started = time.perf_counter()
    for _ in range(500):
        cp.write(task, cp.STARTED)
    assert (time.perf_counter() - started) / 500 < 0.002


def test_a_broken_checkpoint_never_stops_a_status_change(store, monkeypatch):
    monkeypatch.setattr(cp, "write",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    task = _task()
    assert task_manager._set_status(task["id"], PAUSED)["status"] == PAUSED
