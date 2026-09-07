"""Tests for scheduled tasks.

A scheduler that fails quietly is worse than none, so most of this is about what
happens when a task goes wrong: retries, disabling, notification, and the run
history that lets the user see any of it.
"""
from __future__ import annotations

import time
from datetime import datetime

import pytest

import tools.scheduler as scheduler
from tools.scheduler import (
    CreateScheduledTaskTool, DeleteScheduledTaskTool, ListScheduledTasksTool,
    SchedulerRunner, UpdateScheduledTaskTool, compute_next_run, load_tasks, save_tasks,
)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "tasks.json")


class FakeOrchestrator:
    """Stands in for the real one. Scheduled work reaches capabilities through
    handle_user_input, so that's the whole contract."""

    def __init__(self, fail_times=0, delay=0.0):
        self.seen = []
        self.fail_times = fail_times
        self.delay = delay

    async def handle_user_input(self, text, **kwargs):
        import asyncio

        self.seen.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("service unavailable")
        return f"done: {text[:30]}"


def _skip_backoff(monkeypatch):
    """Run retry backoff instantly. The real sleep has to be captured first -
    patching asyncio.sleep with a lambda that calls asyncio.sleep recurses."""
    import asyncio

    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_args, **_kw: real_sleep(0))


async def _make(**kwargs):
    defaults = {"name": "T", "instruction": "Do the thing.", "schedule_type": "interval",
                "every_minutes": 60}
    return await CreateScheduledTaskTool().run(**{**defaults, **kwargs})


async def _make_due(**kwargs):
    result = await _make(**kwargs)
    tasks = load_tasks()
    for task in tasks:
        if task["id"] == result.output["task_id"]:
            task["next_run"] = time.time() - 1
    save_tasks(tasks)
    return result.output["task_id"]


# --- Schedules -------------------------------------------------------------------

def test_weekly_lands_on_the_named_day():
    when = compute_next_run({"schedule_type": "weekly", "weekdays": ["monday"], "at": "09:00"})
    assert datetime.fromtimestamp(when).weekday() == 0
    assert datetime.fromtimestamp(when).hour == 9


def test_daily_never_returns_a_time_in_the_past():
    when = compute_next_run({"schedule_type": "daily", "at": "00:01"})
    assert when > time.time()


def test_monthly_day_is_capped_at_28():
    """'The 31st' would silently skip February and the short months."""
    when = compute_next_run({"schedule_type": "monthly", "day_of_month": 31, "at": "08:00"})
    assert datetime.fromtimestamp(when).day == 28


def test_a_one_off_that_has_run_does_not_run_again():
    task = {"schedule_type": "once", "run_at": time.time() + 60, "last_run_at": time.time()}
    assert compute_next_run(task) is None


# --- Creation --------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [
    {"schedule_type": "weekly", "weekdays": ["funday"]},
    {"schedule_type": "once"},
    {"schedule_type": "once", "run_at": "2020-01-01T00:00"},
    {"schedule_type": "hourly"},
])
async def test_bad_schedules_are_refused(kwargs):
    result = await _make(**kwargs)
    assert result.success is False


@pytest.mark.asyncio
async def test_an_empty_instruction_is_refused():
    assert (await _make(instruction="   ")).success is False


# --- Running ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_due_task_runs_through_the_orchestrator():
    """This is what lets a task use every capability without the scheduler
    knowing any of them exist."""
    await _make_due(instruction="Research 20 clients and write a report.")
    orchestrator = FakeOrchestrator()

    outcomes = await SchedulerRunner(orchestrator).run_due_tasks()

    assert outcomes[0]["status"] == "succeeded"
    assert "Research 20 clients" in orchestrator.seen[0]


@pytest.mark.asyncio
async def test_a_task_that_is_not_due_does_not_run():
    await _make()
    orchestrator = FakeOrchestrator()
    assert await SchedulerRunner(orchestrator).run_due_tasks() == []
    assert orchestrator.seen == []


@pytest.mark.asyncio
async def test_a_disabled_task_does_not_run():
    task_id = await _make_due()
    await UpdateScheduledTaskTool().run(task_id=task_id, enabled=False)
    orchestrator = FakeOrchestrator()
    assert await SchedulerRunner(orchestrator).run_due_tasks() == []


@pytest.mark.asyncio
async def test_a_task_names_its_project_so_output_lands_there():
    await _make_due(project="Business/Clients")
    orchestrator = FakeOrchestrator()
    await SchedulerRunner(orchestrator).run_due_tasks()
    assert "Business/Clients" in orchestrator.seen[0]


# --- Failure handling -------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_transient_failure_is_retried(monkeypatch):
    _skip_backoff(monkeypatch)
    task_id = await _make_due(max_retries=2)
    orchestrator = FakeOrchestrator(fail_times=2)

    outcome = await SchedulerRunner(orchestrator).execute(task_id)

    assert outcome["status"] == "succeeded"
    assert len(orchestrator.seen) == 3


@pytest.mark.asyncio
async def test_a_failure_is_recorded_with_its_error(monkeypatch):
    _skip_backoff(monkeypatch)
    task_id = await _make_due(max_retries=0)

    outcome = await SchedulerRunner(FakeOrchestrator(fail_times=5)).execute(task_id)

    assert outcome["status"] == "failed"
    history = load_tasks()[0]["history"][-1]
    assert history["status"] == "failed"
    assert "service unavailable" in history["error"]


@pytest.mark.asyncio
async def test_the_user_is_told_when_a_task_fails(monkeypatch):
    _skip_backoff(monkeypatch)
    task_id = await _make_due(max_retries=0)
    messages = []

    await SchedulerRunner(FakeOrchestrator(fail_times=5), notify=messages.append).execute(task_id)

    assert messages and "failed" in messages[0]


@pytest.mark.asyncio
async def test_a_task_that_keeps_failing_is_disabled(monkeypatch):
    """Failing forever in silence is how a scheduler becomes noise."""
    _skip_backoff(monkeypatch)
    task_id = await _make_due(max_retries=0)
    runner = SchedulerRunner(FakeOrchestrator(fail_times=99))

    for _ in range(5):
        tasks = load_tasks()
        tasks[0]["next_run"] = time.time() - 1
        save_tasks(tasks)
        await runner.execute(task_id)

    task = load_tasks()[0]
    assert task["enabled"] is False
    assert "consecutive failures" in task["disabled_reason"]


@pytest.mark.asyncio
async def test_re_enabling_clears_the_failure_count(monkeypatch):
    """Otherwise it is disabled again on the next run."""
    _skip_backoff(monkeypatch)
    task_id = await _make_due(max_retries=0)
    runner = SchedulerRunner(FakeOrchestrator(fail_times=99))
    for _ in range(5):
        tasks = load_tasks(); tasks[0]["next_run"] = time.time() - 1; save_tasks(tasks)
        await runner.execute(task_id)

    result = await UpdateScheduledTaskTool().run(task_id=task_id, enabled=True)

    assert result.output["enabled"] is True
    assert result.output["consecutive_failures"] == 0
    assert result.output["disabled_reason"] is None


@pytest.mark.asyncio
async def test_a_hanging_task_times_out_rather_than_blocking_the_scheduler():
    # No backoff patch here: the fake orchestrator's delay is a real sleep, and
    # shortening it would remove the thing being timed out.
    task_id = await _make_due(max_retries=0)
    tasks = load_tasks(); tasks[0]["timeout_seconds"] = 0.05; save_tasks(tasks)

    outcome = await SchedulerRunner(FakeOrchestrator(delay=5)).execute(task_id)

    assert outcome["status"] == "failed"
    assert "timed out" in outcome["error"]


# --- Visibility --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_listing_shows_schedule_status_and_history():
    task_id = await _make_due(name="Monday research")
    await SchedulerRunner(FakeOrchestrator()).execute(task_id)

    output = (await ListScheduledTasksTool().run()).output
    task = output["tasks"][0]
    assert task["name"] == "Monday research"
    assert task["last_status"] == "succeeded"
    assert task["runs"] == 1
    assert task["recent_history"][0]["output"].startswith("done:")


@pytest.mark.asyncio
async def test_a_single_task_returns_its_full_history():
    task_id = await _make_due()
    runner = SchedulerRunner(FakeOrchestrator())
    for _ in range(3):
        await runner.execute(task_id, manual=True)

    output = (await ListScheduledTasksTool().run(task_id=task_id)).output
    assert len(output["full_history"]) == 3


@pytest.mark.asyncio
async def test_deleting_a_task():
    task_id = (await _make()).output["task_id"]
    assert (await DeleteScheduledTaskTool().run(task_id=task_id)).success
    assert load_tasks() == []
    assert (await DeleteScheduledTaskTool().run(task_id=task_id)).success is False


@pytest.mark.asyncio
async def test_a_manual_run_does_not_move_the_schedule():
    task_id = await _make_due()
    before = load_tasks()[0]["next_run"]
    await SchedulerRunner(FakeOrchestrator()).execute(task_id, manual=True)
    assert load_tasks()[0]["next_run"] == before
