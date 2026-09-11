"""The RT-LOG's source: a mirror, never a second status system.

Every line the interface shows comes from a transition something else already
made. What that has to keep being true of: the ring is bounded, a listener that
raises cannot break the work that emitted the entry, and the places that emit are
the places that already changed state - not a parallel set of bookkeeping.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from core import diagnostics

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clean():
    diagnostics.reset()
    listeners = list(diagnostics._activity_listeners)
    yield
    diagnostics._activity_listeners[:] = listeners
    diagnostics.reset()


def test_the_ring_is_bounded():
    for i in range(diagnostics._ACTIVITY_KEEP * 3):
        diagnostics.record_activity("test", f"line {i}")
    entries = diagnostics.recent_activity()
    assert len(entries) == diagnostics._ACTIVITY_KEEP
    assert entries[-1]["message"] == f"line {diagnostics._ACTIVITY_KEEP * 3 - 1}"


def test_listeners_hear_entries_as_they_happen():
    seen = []
    diagnostics.on_activity(seen.append)
    diagnostics.record_activity("tool", "Web search - done (1.0s)")
    assert seen and seen[-1]["message"] == "Web search - done (1.0s)"


def test_a_broken_listener_cannot_break_the_work_that_emitted_the_entry():
    """A log line is never worth failing a tool call for."""
    def explode(entry):
        raise RuntimeError("boom")

    diagnostics.on_activity(explode)
    diagnostics.record_tool("web_search", 1.0, True)      # must not raise
    assert diagnostics.recent_activity()[-1]["tool"] == "web_search"


def test_the_timings_already_recorded_are_what_produce_the_lines():
    """No new call sites: routing, tools and turns were already reported here."""
    seen = []
    diagnostics.on_activity(seen.append)
    diagnostics.record_routing(17, 123, 0.0002, ["web_search"])
    diagnostics.record_tool("plot_chart", 0.8, True)
    diagnostics.record_tool("send_email", 0.2, False)
    diagnostics.record_turn(6.2, 400)
    kinds = [e["kind"] for e in seen]
    assert kinds == ["routing", "tool", "tool", "turn"]
    assert "17 of 123" in seen[0]["message"]
    assert seen[1]["message"].startswith("Plot chart - done")
    assert seen[2]["message"].startswith("Send email - failed")


def test_reset_clears_the_activity_ring_too():
    diagnostics.record_activity("test", "x")
    diagnostics.reset()
    assert diagnostics.recent_activity() == []


# --- Where the mirrors are -------------------------------------------------------

def _calls_record_activity(source: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_activity"
        for node in ast.walk(tree)
    )


def test_task_status_is_mirrored_from_the_one_place_it_changes():
    """_set_status is the single choke point for every task transition, so one
    mirror there covers queued/running/paused/waiting/failed/completed/cancelled
    without a second place that knows what a task's states are."""
    from core import task_manager

    source = inspect.getsource(task_manager)
    assert _calls_record_activity(source)
    set_status = inspect.getsource(task_manager._set_status)
    assert "_mirror_to_activity" in set_status
    # Every status the module defines has wording, or a transition would be mute.
    assert set(task_manager._STATUS_WORDS) == set(
        task_manager.ACTIVE_STATUSES + task_manager.FINISHED_STATUSES)


def test_step_progress_is_mirrored_too():
    """A five-step task that says only "started" and "completed" is not progress."""
    from core import task_manager

    seen = []
    diagnostics.on_activity(seen.append)
    task_manager._mirror_step(
        {"id": "t1", "name": "Weekly report",
         "steps": [{"instruction": "gather the numbers"}, {"instruction": "write it up"}]}, 0)
    assert seen[-1]["kind"] == "task"
    assert "step 1 of 2" in seen[-1]["message"]
    assert "gather the numbers" in seen[-1]["message"]
    assert "_mirror_step(task, index)" in inspect.getsource(task_manager.TaskRunner._run_step)


def test_a_watch_trigger_is_mirrored_and_cannot_stop_the_watch():
    from core import watches

    check = inspect.getsource(watches.check)
    assert "record_activity" in check
    assert "except Exception" in check, "a log failure must not stop a watch firing"


def test_the_interface_subscribes_rather_than_polling():
    """A log the interface asks for on a timer is a timer that runs whether or not
    anything happened. gui/api.py pushes instead."""
    api = (PROJECT_ROOT / "gui" / "api.py").read_text()
    assert "diagnostics.on_activity(" in api
    assert 'push("letiActivity"' in api


def test_waiting_for_approval_is_logged_without_touching_safetyguard():
    """Being asked is a fact about the turn the interface wants to show; it is not
    SafetyGuard's business that an interface exists."""
    api = (PROJECT_ROOT / "gui" / "api.py").read_text()
    guard = (PROJECT_ROOT / "core" / "safety_guard.py").read_text()
    confirmation = (PROJECT_ROOT / "core" / "confirmation.py").read_text()
    assert "_logged(make_voice_confirmation_callback" in api
    assert "_logged(make_text_confirmation_callback" in api
    assert "record_activity" not in guard and "record_activity" not in confirmation
