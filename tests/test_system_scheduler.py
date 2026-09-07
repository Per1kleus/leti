"""Tests for running scheduled tasks when Leti isn't open.

Two things have to hold once the OS scheduler is involved: the same task must
not run twice when both the in-app loop and a cron process are alive, and a run
with nobody present must not be able to do things the user never agreed to.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import time

import pytest

import core.system_scheduler as system_scheduler
import tools.scheduler as scheduler
from core.safety_guard import ConfirmationDenied
from tools.scheduler import (
    CreateScheduledTaskTool, SchedulerRunner, load_tasks, save_tasks, scheduler_lock,
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(scheduler, "_lock_path", lambda: tmp_path / "scheduler.lock")


class FakeOrchestrator:
    def __init__(self):
        self.seen = []

    async def handle_user_input(self, text, **kwargs):
        self.seen.append(text)
        return "done"


# --- Not running the same task twice ---------------------------------------------

def test_the_lock_excludes_a_second_process(tmp_path):
    """The in-app loop and the cron process both want to run due tasks. Without
    this they'd both pick up the same one - two emails, two reports."""
    lock_file = tmp_path / "s.lock"
    script = textwrap.dedent(f"""
        import sys, time, pathlib
        sys.path.insert(0, {str(tmp_path.parent.parent)!r})
        sys.path.insert(0, {str(__import__('pathlib').Path(__file__).resolve().parent.parent)!r})
        import tools.scheduler as sch
        sch._lock_path = lambda: pathlib.Path({str(lock_file)!r})
        with sch.scheduler_lock() as got:
            print("acquired" if got else "blocked", flush=True)
            if got: time.sleep(1.5)
    """)
    holder = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "acquired"
        second = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert second.stdout.strip() == "blocked"
    finally:
        holder.wait(timeout=10)

    # And the lock is free again once that process is gone.
    after = subprocess.run([sys.executable, "-c", script.replace("time.sleep(1.5)", "pass")],
                           capture_output=True, text=True)
    assert after.stdout.strip() == "acquired"


@pytest.mark.asyncio
async def test_due_tasks_are_skipped_while_another_process_holds_the_lock(monkeypatch):
    from contextlib import contextmanager

    result = await CreateScheduledTaskTool().run(
        name="T", instruction="Do it.", schedule_type="interval", every_minutes=60)
    tasks = load_tasks()
    tasks[0]["next_run"] = time.time() - 1
    save_tasks(tasks)

    @contextmanager
    def busy():
        yield False

    monkeypatch.setattr(scheduler, "scheduler_lock", busy)
    orchestrator = FakeOrchestrator()
    assert await SchedulerRunner(orchestrator).run_due_tasks() == []
    assert orchestrator.seen == []


@pytest.mark.asyncio
async def test_downtime_produces_one_catch_up_run_not_one_per_missed_occurrence():
    """A week with the machine off should not yield seven reports."""
    await CreateScheduledTaskTool().run(
        name="Daily", instruction="Report.", schedule_type="daily", at="09:00")
    tasks = load_tasks()
    tasks[0]["next_run"] = time.time() - 7 * 86400      # a week overdue
    save_tasks(tasks)

    orchestrator = FakeOrchestrator()
    await SchedulerRunner(orchestrator).run_due_tasks()

    assert len(orchestrator.seen) == 1
    assert load_tasks()[0]["next_run"] > time.time()


# --- What an unattended run may do -------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("tool,allowed", [
    ("web_search", True),          # execute
    ("inspect_dataset", True),     # read
    ("write_file", True),          # modify - reports have to be written somewhere
    ("send_email", False),         # external
    ("delete_file", False),        # critical
    ("run_shell_command", False),  # critical
])
async def test_unattended_permissions(tool, allowed, guard_factory):
    guard, prompts = guard_factory()
    guard.set_unattended(True)

    if allowed:
        auth = await guard.authorize(tool, {"path": "/tmp/x", "to": "a@b.c", "command": "ls"})
        assert auth.execute is True
    else:
        with pytest.raises(ConfirmationDenied):
            await guard.authorize(tool, {"path": "/tmp/x", "to": "a@b.c", "command": "ls"})
    assert prompts == [], "an unattended run must never wait on a prompt"


@pytest.mark.asyncio
async def test_the_refusal_says_how_to_allow_it(guard_factory):
    guard, _ = guard_factory()
    guard.set_unattended(True)
    with pytest.raises(ConfirmationDenied) as raised:
        await guard.authorize("send_email", {"to": "a@b.c"})
    message = str(raised.value)
    assert "unattended" in message
    assert "unattended_allows" in message


@pytest.mark.asyncio
async def test_critical_is_refused_even_if_the_user_permits_it(guard_factory):
    """Same reasoning as require_confirmation_for: a setting that could authorise
    irreversible actions with nobody present defeats the point."""
    guard, _ = guard_factory()
    guard.settings["scheduler"] = {"unattended_allows": ["read", "execute", "modify",
                                                         "external", "critical"]}
    guard.set_unattended(True)
    assert "critical" not in guard.unattended_classes
    with pytest.raises(ConfirmationDenied):
        await guard.authorize("delete_file", {"path": "/tmp/x"})


@pytest.mark.asyncio
async def test_an_attended_session_is_unaffected(guard_factory):
    """set_unattended is opt-in; a normal session still prompts."""
    guard, prompts = guard_factory(confirm=True)
    await guard.authorize("send_email", {"to": "a@b.c"})
    assert len(prompts) == 1


# --- OS registration -----------------------------------------------------------------

def test_status_answers_on_a_machine_with_no_scheduler(monkeypatch):
    """Asking about something a container can't do should answer, not crash."""
    monkeypatch.setattr(system_scheduler.shutil, "which", lambda name: None)
    result = system_scheduler.status()
    assert result["installed"] is False
    assert "platform" in result


def test_a_missing_command_is_a_failed_result_not_an_exception():
    result = system_scheduler._run(["definitely-not-a-command-xyz"])
    assert result.returncode == 127
    assert "not installed" in result.stderr


def test_the_cron_entry_runs_the_headless_mode():
    line = system_scheduler._cron_line(5)
    assert "--mode run-scheduled" in line
    assert line.startswith("*/5 * * * *")
    assert system_scheduler.MARKER in line, "it must be findable again to remove"


def test_the_installed_command_uses_a_python_that_has_the_dependencies():
    """cron runs with a bare environment, where `python3` is the system one."""
    assert system_scheduler.python_executable().endswith(("python", "python3", "python.exe"))


def _fake_crontab(monkeypatch, initial):
    """A crontab that records what gets written, so install/uninstall can be
    checked against a user who already had entries of their own."""
    state = {"lines": list(initial)}

    def write(lines):
        state["lines"] = list(lines)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(system_scheduler.shutil, "which", lambda name: "/usr/bin/crontab")
    monkeypatch.setattr(system_scheduler, "_read_crontab", lambda: state["lines"])
    monkeypatch.setattr(system_scheduler, "_write_crontab", write)
    return state


def test_uninstall_leaves_other_crontab_entries_alone(monkeypatch):
    state = _fake_crontab(monkeypatch, ["0 3 * * * /usr/bin/backup.sh",
                                        system_scheduler._cron_line(5)])

    result = system_scheduler._cron_uninstall()

    assert result["removed"] is True
    assert state["lines"] == ["0 3 * * * /usr/bin/backup.sh"]


def test_installing_twice_replaces_rather_than_duplicates(monkeypatch):
    state = _fake_crontab(monkeypatch, ["0 3 * * * /usr/bin/backup.sh",
                                        system_scheduler._cron_line(5)])

    system_scheduler._cron_install(15)
    leti_lines = [l for l in state["lines"] if system_scheduler.MARKER in l]
    assert len(leti_lines) == 1
    assert leti_lines[0].startswith("*/15")
    assert "0 3 * * * /usr/bin/backup.sh" in state["lines"]
