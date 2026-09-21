"""What a long task shows, and what it says out loud.

The constraint running through all of these: there is one task state, in
core/task_manager.py, and both the interface and the voice read it. Neither
keeps its own copy, and nothing polls for it.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from core import task_manager

HUD = Path("gui/hud.html").read_text()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    return tmp_path


# --- What the panel is given ------------------------------------------------------------

def test_a_task_reports_everything_the_panel_needs(store):
    task = task_manager.create_task("get the report out", ["gather", "write", "send"],
                                    "the report")
    detail = task_manager.detail(task)
    for field in ("name", "status", "current", "steps_done", "steps_total", "percent",
                  "completed_steps", "next_step", "error", "awaiting_approval",
                  "approval_request", "waiting_for_you", "recovering", "can"):
        assert field in detail, f"the panel has no '{field}' to show"


def test_progress_is_counted_from_the_steps_that_are_actually_done(store):
    task = task_manager.create_task("o", ["a", "b", "c", "d"], "t")
    assert task_manager.detail(task)["percent"] == 0.0
    task["steps"][0]["status"] = "done"
    task["steps"][1]["status"] = "done"
    assert task_manager.detail(task)["percent"] == 50.0


def test_the_next_step_is_shown_so_the_card_answers_and_then_what(store):
    task = task_manager.create_task("o", ["gather", "write", "send"], "t")
    assert task_manager.detail(task)["next_step"] == "write"


def test_a_task_waiting_on_the_user_says_so_in_words(store):
    task = task_manager.create_task("o", ["a"], "t")
    task["status"] = task_manager.WAITING_FOR_USER
    task["blocked_reason"] = "it needs approval to send the email"
    detail = task_manager.detail(task)
    assert detail["awaiting_approval"] is True
    assert "needs your answer" in detail["waiting_for_you"]
    assert "send the email" in detail["waiting_for_you"]


def test_a_task_that_is_not_waiting_says_nothing_about_waiting(store):
    task = task_manager.create_task("o", ["a"], "t")
    assert task_manager.detail(task)["waiting_for_you"] is None


def test_a_recovering_step_is_visible_as_recovering(store):
    task = task_manager.create_task("o", ["a", "b"], "t")
    task["steps"][0]["status"] = "recovering"
    assert task_manager.detail(task)["recovering"] is True


def test_every_active_task_can_be_cancelled(store):
    task = task_manager.create_task("o", ["a"], "t")
    for status in task_manager.ACTIVE_STATUSES:
        task["status"] = status
        assert task_manager.detail(task)["can"]["cancel"] is True


def test_a_finished_task_cannot_be_cancelled(store):
    task = task_manager.create_task("o", ["a"], "t")
    for status in task_manager.FINISHED_STATUSES:
        task["status"] = status
        assert task_manager.detail(task)["can"]["cancel"] is False


# --- What it says out loud ----------------------------------------------------------------

def test_spoken_progress_names_the_task_and_where_it_is(store):
    task = task_manager.create_task("o", ["find the data", "clean it", "write it up"],
                                    "the analysis")
    line = task_manager.spoken_progress(task, 1)
    assert "the analysis" in line and "step 2 of 3" in line


def test_spoken_progress_is_short_enough_to_hear(store):
    task = task_manager.create_task(
        "o", ["a" * 300, "b" * 300], "a task with a reasonably long name")
    assert len(task_manager.spoken_progress(task, 1)) < 160


def test_a_recovering_step_is_described_as_trying_again_not_as_progress(store):
    task = task_manager.create_task("o", ["a", "b", "c"], "the task")
    task["steps"][1]["status"] = "recovering"
    line = task_manager.spoken_progress(task, 1)
    assert "did not work" in line and "trying another way" in line


def test_a_blocked_task_says_it_is_waiting_for_you(store):
    task = task_manager.create_task("o", ["a", "b", "c"], "the task")
    task["status"] = task_manager.WAITING_FOR_USER
    task["blocked_reason"] = "it needs your approval"
    assert "waiting for you" in task_manager.spoken_progress(task, 1)


def test_a_short_task_says_nothing_until_it_is_done(store):
    """A four-step task that finishes in ten seconds should not narrate itself."""
    said = []
    runner = task_manager.TaskRunner(orchestrator=None, notify=said.append)
    task = task_manager.create_task("o", ["a", "b"], "short")
    runner._announce_progress(task, 1)
    assert said == []


def test_a_long_task_does_not_narrate_every_step(store):
    said = []
    runner = task_manager.TaskRunner(orchestrator=None, notify=said.append)
    task = task_manager.create_task("o", ["a", "b", "c", "d", "e"], "long")
    runner._announce_progress(task, 1)
    runner._announce_progress(task, 2)
    runner._announce_progress(task, 3)
    assert len(said) == 1, "a long task narrated more than once inside the quiet window"


def test_the_first_step_never_announces_progress(store):
    said = []
    runner = task_manager.TaskRunner(orchestrator=None, notify=said.append)
    task = task_manager.create_task("o", ["a", "b", "c", "d"], "long")
    runner._announce_progress(task, 0)
    assert said == []


def test_a_broken_announcement_never_breaks_the_task(store):
    def explode(message):
        raise RuntimeError("the speaker is on fire")

    runner = task_manager.TaskRunner(orchestrator=None, notify=explode)
    task = task_manager.create_task("o", ["a", "b", "c", "d"], "long")
    runner._announce_progress(task, 1)          # must not raise


def test_voice_and_the_panel_read_the_same_state(store):
    """One task state. The spoken line is derived from the same steps the panel
    counts, so they cannot disagree about where a task is."""
    task = task_manager.create_task("o", ["a", "b", "c", "d"], "t")
    task["steps"][0]["status"] = "done"
    detail = task_manager.detail(task)
    spoken = task_manager.spoken_progress(task, task["current_step"])
    assert f"of {detail['steps_total']}" in spoken


def test_there_is_no_second_voice_pipeline():
    """Progress goes down the notify path that already existed."""
    import ast

    tree = ast.parse(Path("core/task_manager.py").read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imports |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    for forbidden in ("pyttsx3", "voice", "tts", "core.tts"):
        assert forbidden not in imports, f"core/task_manager.py imports {forbidden}"


# --- The interface does not poll ------------------------------------------------------------

def test_the_task_panel_is_pushed_not_polled():
    """A timer that asks "any news?" runs whether or not anything happened."""
    assert "refreshTasksSoon" in HUD
    assert not re.search(r"setInterval\s*\(\s*refreshTasks", HUD)


def test_the_readout_covers_every_state_a_person_has_to_act_on():
    states = re.search(r"const STATES = \{(.*?)\n  \};", HUD, re.S).group(1)
    for state in ("idle", "listening", "thinking", "executing", "speaking",
                  "waiting", "approval", "recovering", "completed", "failed"):
        assert re.search(rf"\n\s+{state}:", states), f"the readout has no '{state}' state"


def test_a_busy_agent_is_not_hidden_by_a_waiting_task():
    applied = re.search(r"function applyState\(\)\{(.*?)\n  \}", HUD, re.S).group(1)
    assert "BUSY_STATES" in applied and "agentState" in applied


def test_an_idle_agent_yields_to_a_task_that_needs_the_user():
    applied = re.search(r"function applyState\(\)\{(.*?)\n  \}", HUD, re.S).group(1)
    assert "taskAttention" in applied


def test_the_attention_state_is_read_off_the_same_task_rows_the_card_renders():
    refresh = re.search(r"function refreshTasks\(\)\{(.*?)\n  \}", HUD, re.S).group(1)
    assert "taskState.active.find(t => t.awaiting_approval)" in refresh
    assert "renderTaskCard();" in refresh
