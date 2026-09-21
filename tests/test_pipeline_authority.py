"""The order of the pipeline, and who is still in charge of it.

    USER -> INTENT -> CONTEXT -> ENTITY -> GOAL -> TASK -> PLAN
         -> PERMISSION / SAFETY -> ACTION -> WORLD STATE -> VERIFICATION
         -> RESULT -> RECOVERY -> HISTORY -> USER

Two things have to stay true however much is added above them. SafetyGuard
decides before anything runs, and nothing new can decide instead of it. And a
failure is never reported as a success.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest


ORCHESTRATOR = pathlib.Path("core/orchestrator.py").read_text()


def _turn_body():
    return re.search(r"async def _handle_one_turn.*?(?=\n    (?:async )?def )",
                     ORCHESTRATOR, re.S).group(0)


# --- The order ---------------------------------------------------------------------

def test_intent_is_read_before_anything_else_happens():
    body = _turn_body()
    assert body.index("intent_reader.read") < body.index("_build_messages")


def test_the_deterministic_commands_come_before_the_model():
    body = _turn_body()
    for command in ("asks_which_mode", "mode_command", "lifecycle_command",
                    "asks_for_diagnostics"):
        assert body.index(command) < body.index("_tool_calling_loop"), command


def test_context_is_assembled_before_the_tools_are_chosen():
    body = _turn_body()
    assert body.index("_build_messages") < body.index("_tool_calling_loop")


def test_the_intent_veto_is_applied_where_the_tools_are_chosen():
    loop = re.search(r"async def _tool_calling_loop.*?(?=\n    (?:async )?def )",
                     ORCHESTRATOR, re.S).group(0)
    assert loop.index("wants_action") < loop.index("schemas_for")


# --- SafetyGuard is still in charge --------------------------------------------------

def test_every_tool_call_still_goes_through_authorize():
    execute = re.search(r"async def _execute_tool_call.*?(?=\n    (?:async )?def )",
                        ORCHESTRATOR, re.S).group(0)
    assert execute.index("safety_guard.authorize") < execute.index("await tool.run")


def test_nothing_new_can_authorize_anything():
    """The new layers advise and record. None of them decides."""
    for path in ("core/intent.py", "core/task_history.py", "core/recovery.py",
                 "core/task_conflicts.py", "core/task_control.py",
                 "core/entities.py", "core/context_engine.py"):
        tree = ast.parse(pathlib.Path(path).read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for forbidden in ("authorize", "set_unattended", "requires_confirmation",
                          "_unattended", "preapproved"):
            assert forbidden not in names, f"{path} touches {forbidden}"


def test_nothing_new_runs_a_tool():
    for path in ("core/task_history.py", "core/recovery.py", "core/task_conflicts.py",
                 "core/task_control.py"):
        tree = ast.parse(pathlib.Path(path).read_text())
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for forbidden in ("run", "handle_user_input", "execute"):
            assert forbidden not in calls, f"{path} calls {forbidden}"


def test_a_task_step_is_an_ordinary_request_through_the_orchestrator():
    """Which is what puts every tool call a task makes in front of the guard."""
    task_manager = pathlib.Path("core/task_manager.py").read_text()
    step = re.search(r"async def _run_step.*?(?=\n    def |\n    async def )",
                     task_manager, re.S).group(0)
    assert "self.orchestrator.handle_user_input" in step
    assert "tool.run" not in step and "authorize" not in step


def test_an_unattended_run_still_narrows_what_may_happen():
    task_manager = pathlib.Path("core/task_manager.py").read_text()
    assert "set_unattended(True)" in task_manager
    assert "restore_unattended()" in task_manager


# --- A failure is never a success ------------------------------------------------------

@pytest.mark.parametrize("path", [
    "core/task_manager.py", "core/task_history.py", "core/task_control.py",
    "core/watches.py",
])
def test_no_module_reports_success_from_an_attempt(path):
    """Nothing sets a completed/verified flag from the fact that something ran."""
    source = pathlib.Path(path).read_text()
    for pattern in (r'"status"\]\s*=\s*COMPLETED\s*#\s*attempted',
                    r'success=True\s*#\s*assumed'):
        assert not re.search(pattern, source)


def test_verified_and_not_verified_stay_distinct():
    from core import verification

    assert verification.VERIFIED != verification.NOT_VERIFIED
    assert verification.worst([verification.VERIFIED,
                               verification.NOT_VERIFIED]) == verification.NOT_VERIFIED
    assert verification.summarise(
        [{"result": verification.VERIFIED},
         {"result": verification.NOT_VERIFIED}])["overall"] == verification.NOT_VERIFIED


def test_a_watch_action_that_did_not_start_is_not_verified():
    from core import watches

    assert watches.verify_action({"action": "start_task"}, {})["result"] == "NOT VERIFIED"


def test_a_task_that_stopped_is_never_described_as_completed(tmp_path, monkeypatch):
    from core import task_history, task_manager

    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "t.json")
    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "h.json")
    for stop, expected in (("cancel", "cancelled"),):
        task = task_manager.create_task("o", ["a"], "t")
        getattr(task_manager, stop)(task["id"])
        assert task_manager.get_task(task["id"])["status"] == expected
        assert task_history.get(task["id"])["status"] == expected


# --- Fail closed ------------------------------------------------------------------------

def test_an_unknown_failure_does_not_recover():
    from core import recovery

    assert recovery.plan(RuntimeError("something nobody has seen"))["recover"] is False


def test_an_unknown_watch_action_is_rejected_by_validation():
    from core import watches

    watch = watches.create(name="x", condition_type="cpu_above",
                           condition={"percent": 90}, action="anything_else")
    assert watches.validate(watch)


def test_an_unknown_lifecycle_action_falls_through_rather_than_guessing():
    from core import task_control

    assert task_control.apply({"action": "levitate", "hint": ""}) is None


def test_an_unreadable_history_is_empty_rather_than_fatal(tmp_path, monkeypatch):
    from core import task_history

    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "h.json")
    (tmp_path / "h.json").write_bytes(b"\xff\xfe not text at all")
    assert task_history.load() == []
