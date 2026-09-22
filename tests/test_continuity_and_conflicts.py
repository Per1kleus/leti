"""References across turns, and tasks that must not collide.

Phases 5 and 8. The rule under both is the same one the rest of Leti already
follows: one supported candidate resolves, several is a question, none is "I
cannot find it" - and where two pieces of work want the same thing, one waits
and says why rather than both going ahead.
"""
from __future__ import annotations

import pytest

from core import entities, task_conflicts, task_history, task_manager


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "history.json")
    entities.use_conversation_source(None)
    yield tmp_path
    entities.use_conversation_source(None)


def _task(name, objective="an objective", steps=("step",)):
    return task_manager.create_task(objective, list(steps), name)


# --- Continuity -----------------------------------------------------------------------

def test_a_named_task_resolves(store):
    _task("website generator")
    _task("quarterly report")
    outcome = entities.resolve("the website task", kind="task")
    assert outcome["resolved"] and "website" in outcome["resolved"]["name"]


def test_two_comparable_tasks_ask_rather_than_pick(store):
    _task("the first report")
    _task("the second report")
    outcome = entities.resolve("report", kind="task")
    assert outcome["resolved"] is None and outcome["ask"]


def test_nothing_matching_says_so(store):
    _task("website generator")
    outcome = entities.resolve("wakanda industries", kind="task")
    assert outcome["resolved"] is None and "Nothing matches" in outcome["problem"]


def test_a_finished_task_is_resolvable_from_the_history(store):
    task = _task("website generator")
    task_manager.cancel(task["id"])
    outcome = entities.resolve("the website generator", kind="past_task")
    assert outcome["resolved"] and outcome["resolved"]["id"] == task["id"]


def test_what_was_said_earlier_settles_which_task(store):
    entities.use_conversation_source(
        lambda: [{"role": "user", "content": "how is the quarterly report going?"}])
    _task("quarterly report")
    _task("monthly report")
    outcome = entities.resolve("the report", kind="task")
    assert outcome["resolved"]["name"].startswith("quarterly")


def test_an_unreadable_source_is_reported_rather_than_read_as_empty(monkeypatch, store):
    monkeypatch.setitem(entities._SOURCES, "project",
                        lambda: (_ for _ in ()).throw(RuntimeError("no projects")))
    outcome = entities.resolve("something")
    assert outcome["unavailable"] and "project" in outcome["unavailable"][0]


def test_an_unknown_kind_is_named_rather_than_searched(store):
    outcome = entities.resolve("x", kind="unicorn")
    assert outcome["resolved"] is None and "not something Leti resolves" in outcome["problem"]


def test_resolution_still_goes_through_the_one_decision(store):
    """The general resolver must not grow its own rule."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/entities.py").read_text())
    functions = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "choose" in functions
    # resolve() must delegate, not decide: its body calls choose and returns it.
    resolver = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "resolve")
    calls = {c.func.id for c in ast.walk(resolver)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "choose" in calls, "the general resolver does not go through choose()"


# --- Conflicts -------------------------------------------------------------------------

def test_a_task_declares_the_file_it_writes():
    task = {"id": "a", "objective": "edit report.md",
            "steps": [{"instruction": "write the summary into report.md"}]}
    declared = task_conflicts.declare(task)
    assert {"name": "file:report.md", "mode": task_conflicts.WRITE} in declared


def test_reading_the_same_file_is_not_a_conflict():
    a = {"id": "a", "objective": "read report.md", "steps": [{"instruction": "read report.md"}]}
    b = {"id": "b", "objective": "read report.md", "steps": [{"instruction": "read report.md"}]}
    assert task_conflicts.may_start(b, [a])["ok"] is True


def test_two_tasks_writing_the_same_file_conflict():
    a = {"id": "a", "objective": "write report.md", "steps": [{"instruction": "save report.md"}]}
    b = {"id": "b", "objective": "edit report.md", "steps": [{"instruction": "update report.md"}]}
    verdict = task_conflicts.may_start(b, [a])
    assert verdict["ok"] is False and verdict["wait_for"] == ["a"]


def test_two_tasks_driving_the_screen_conflict():
    a = {"id": "a", "objective": "click save", "steps": [{"instruction": "click the save button"}]}
    b = {"id": "b", "objective": "type a note", "steps": [{"instruction": "type into the box"}]}
    verdict = task_conflicts.may_start(b, [a])
    assert verdict["ok"] is False
    assert any(c["resource"] == task_conflicts.SCREEN for c in verdict["conflicts"])


def test_two_tasks_driving_the_same_application_conflict():
    a = {"id": "a", "objective": "open blender", "steps": [{"instruction": "open blender"}]}
    b = {"id": "b", "objective": "close blender", "steps": [{"instruction": "close blender"}]}
    assert task_conflicts.may_start(b, [a])["ok"] is False


def test_unrelated_tasks_run_together():
    a = {"id": "a", "objective": "download the dataset",
         "steps": [{"instruction": "download data.csv"}]}
    b = {"id": "b", "objective": "answer a question",
         "steps": [{"instruction": "explain what a turbine is"}]}
    assert task_conflicts.may_start(b, [a])["ok"] is True


def test_a_conflict_names_who_is_holding_it():
    a = {"id": "downloader", "objective": "write data.csv",
         "steps": [{"instruction": "save data.csv"}]}
    b = {"id": "editor", "objective": "edit data.csv",
         "steps": [{"instruction": "update data.csv"}]}
    verdict = task_conflicts.may_start(b, [a])
    assert "downloader" in verdict["explain"]
    assert verdict["conflicts"][0]["why"]


def test_an_explicit_declaration_is_taken_as_given():
    task = {"id": "a", "objective": "do a thing", "steps": [],
            "resources": [{"name": "printer", "mode": task_conflicts.CONTROL}]}
    assert {"name": "printer", "mode": "control"} in task_conflicts.declare(task)


def test_every_pair_is_reported_once():
    tasks = [
        {"id": "a", "objective": "write x.md", "steps": [{"instruction": "save x.md"}]},
        {"id": "b", "objective": "write x.md", "steps": [{"instruction": "save x.md"}]},
    ]
    assert len(task_conflicts.all_conflicts(tasks)) == 1


def test_conflict_detection_schedules_nothing():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/task_conflicts.py").read_text())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("run", "start", "create_task", "Thread", "sleep", "pause"):
        assert forbidden not in called, f"core/task_conflicts.py calls {forbidden}"


# --- Concurrency in the runner ----------------------------------------------------------

class SlowOrchestrator:
    def __init__(self):
        self.seen = []

    async def handle_user_input(self, text, session_id="default", voice_mode=False,
                                preapproved=False, owner=""):
        import asyncio

        self.seen.append(text)
        await asyncio.sleep(0.02)
        return "ok"


@pytest.mark.asyncio
async def test_two_unrelated_tasks_run_at_the_same_time(store):
    import asyncio

    a = _task("download", "download the dataset", ["download data.csv"])
    b = _task("explain", "explain a turbine", ["explain what a turbine is"])
    runner = task_manager.TaskRunner(SlowOrchestrator())
    assert runner.start_in_background(a["id"]) is True
    assert runner.start_in_background(b["id"]) is True
    assert len(runner.running_tasks()) == 2
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_a_conflicting_task_waits_and_says_why(store):
    import asyncio

    a = _task("writer a", "write report.md", ["save report.md"])
    b = _task("writer b", "edit report.md", ["update report.md"])
    runner = task_manager.TaskRunner(SlowOrchestrator())
    assert runner.start_in_background(a["id"]) is True
    assert runner.start_in_background(b["id"]) is False
    stored = task_manager.get_task(b["id"])
    assert stored["waiting_for_tasks"] == [a["id"]]
    assert "report.md" in stored["blocked_reason"]
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_a_waiting_task_starts_once_the_conflict_clears(store):
    import asyncio

    a = _task("writer a", "write report.md", ["save report.md"])
    b = _task("writer b", "edit report.md", ["update report.md"])
    runner = task_manager.TaskRunner(SlowOrchestrator())
    runner.start_in_background(a["id"])
    runner.start_in_background(b["id"])
    for _ in range(60):
        await asyncio.sleep(0.02)
        if task_manager.get_task(b["id"])["status"] in ("running", "completed"):
            break
    assert task_manager.get_task(b["id"])["status"] in ("running", "completed")


@pytest.mark.asyncio
async def test_concurrency_is_capped(store, monkeypatch):
    import asyncio

    monkeypatch.setattr(task_manager, "MAX_CONCURRENT_TASKS", 2)
    runner = task_manager.TaskRunner(SlowOrchestrator())
    tasks = [_task(f"t{i}", f"think about topic {i}", [f"consider topic {i}"])
             for i in range(4)]
    started = [runner.start_in_background(t["id"]) for t in tasks]
    assert sum(started) <= 2
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_one_task_failing_does_not_stop_another(store):
    import asyncio

    class Mixed:
        def __init__(self):
            self.seen = []

        async def handle_user_input(self, text, session_id="default",
                                    voice_mode=False, preapproved=False, owner=""):
            self.seen.append(text)
            await asyncio.sleep(0.01)
            if "explode" in text:
                raise RuntimeError("no idea what happened")
            return "ok"

    a = _task("bad", "explode please", ["explode now"])
    b = _task("good", "think about turbines", ["consider a turbine"])
    runner = task_manager.TaskRunner(Mixed())
    runner.start_in_background(a["id"])
    runner.start_in_background(b["id"])
    for _ in range(100):
        await asyncio.sleep(0.02)
        if (task_manager.get_task(a["id"])["status"] == "failed"
                and task_manager.get_task(b["id"])["status"] == "completed"):
            break
    assert task_manager.get_task(a["id"])["status"] == "failed"
    assert task_manager.get_task(b["id"])["status"] == "completed"


@pytest.mark.asyncio
async def test_a_conflict_check_that_breaks_never_stops_a_task(store, monkeypatch):
    import asyncio

    monkeypatch.setattr(task_conflicts, "may_start",
                        lambda t, r: (_ for _ in ()).throw(RuntimeError("boom")))
    a = _task("a", "do a thing", ["a step"])
    b = _task("b", "do another", ["another step"])
    runner = task_manager.TaskRunner(SlowOrchestrator())
    assert runner.start_in_background(a["id"]) is True
    assert runner.start_in_background(b["id"]) is True
    await asyncio.sleep(0.2)
