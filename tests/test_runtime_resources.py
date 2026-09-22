"""What a task is actually holding, and who has to wait for it.

The guarantees under test: two tasks never both hold something exclusive, a
lock is released however the holder ends, a conflict makes a task WAIT rather
than fail, and holding a lock never grants permission to do anything.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core import resources as rs


@pytest.fixture(autouse=True)
def _empty_ledger():
    rs.clear()
    yield
    rs.clear()


def file_of(name):
    return rs.identity(rs.FILE, name)


# --- Mode compatibility -------------------------------------------------------------

def test_two_readers_coexist():
    assert rs.acquire(file_of("/tmp/a"), rs.READ, "a")["acquired"] is True
    assert rs.acquire(file_of("/tmp/a"), rs.READ, "b")["acquired"] is True


@pytest.mark.parametrize("held,wanted", [
    (rs.READ, rs.WRITE), (rs.WRITE, rs.READ), (rs.WRITE, rs.WRITE),
    (rs.EXCLUSIVE, rs.EXCLUSIVE), (rs.EXCLUSIVE, rs.READ),
    (rs.CONTROL, rs.CONTROL), (rs.CONTROL, rs.READ), (rs.READ, rs.CONTROL),
])
def test_everything_else_conflicts(held, wanted):
    assert rs.acquire(file_of("/tmp/a"), held, "a")["acquired"] is True
    assert rs.acquire(file_of("/tmp/a"), wanted, "b")["acquired"] is False


def test_a_task_never_conflicts_with_itself():
    assert rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")["acquired"] is True
    assert rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")["acquired"] is True
    assert len(rs.holders(file_of("/tmp/a"))) == 1, "a task stacked two holds"


def test_an_upgrade_replaces_rather_than_stacks():
    rs.acquire(file_of("/tmp/a"), rs.READ, "a")
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")
    holds = rs.holders(file_of("/tmp/a"))
    assert len(holds) == 1 and holds[0].mode == rs.WRITE
    rs.release(file_of("/tmp/a"), "a")
    assert rs.holders(file_of("/tmp/a")) == []


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        rs.acquire(file_of("/tmp/a"), "sideways", "a")


# --- Identity -----------------------------------------------------------------------

def test_the_same_file_by_different_names_is_one_resource(tmp_path):
    real = tmp_path / "report.md"
    real.write_text("x")
    link = tmp_path / "link.md"
    link.symlink_to(real)
    assert rs.identity(rs.FILE, str(real)) == rs.identity(rs.FILE, str(link))


def test_a_relative_path_resolves_to_the_same_resource(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.md").write_text("x")
    assert rs.identity(rs.FILE, "./x.md") == rs.identity(rs.FILE, str(tmp_path / "x.md"))


def test_the_screen_is_one_resource_however_it_is_named():
    assert rs.identity(rs.SCREEN, "") == rs.identity(rs.SCREEN, "")


# --- Runtime discovery from the tool boundary ------------------------------------------

@pytest.mark.parametrize("tool,arguments,kind,mode", [
    ("read_file", {"path": "/tmp/x"}, rs.FILE, rs.READ),
    ("write_file", {"path": "/tmp/x"}, rs.FILE, rs.WRITE),
    ("delete_file", {"path": "/tmp/x"}, rs.FILE, rs.WRITE),
    ("mouse_click", {"x": 1, "y": 2}, rs.SCREEN, rs.CONTROL),
    ("keyboard_type", {"text": "hi"}, rs.SCREEN, rs.CONTROL),
    ("read_screen", {"question": "?"}, rs.SCREEN, rs.READ),
    ("launch_app", {"app_name": "blender"}, rs.APPLICATION, rs.CONTROL),
    ("send_email", {"to": "a@b.c"}, rs.CONNECTION, rs.CONTROL),
    ("list_new_emails", {}, rs.CONNECTION, rs.READ),
])
def test_a_tool_call_says_what_it_will_touch(tool, arguments, kind, mode):
    found = rs.for_tool(tool, arguments)
    assert found, f"{tool} declared nothing"
    assert found[0]["resource"].startswith(kind)
    assert found[0]["mode"] == mode


def test_a_move_claims_both_ends():
    found = rs.for_tool("move_file", {"source_path": "/tmp/a",
                                      "destination_path": "/tmp/b"})
    assert len(found) == 2
    assert all(entry["mode"] == rs.WRITE for entry in found)


def test_a_tool_that_touches_nothing_claims_nothing():
    assert rs.for_tool("get_weather", {}) == []
    assert rs.for_tool("engineering_calculate", {"expression": "2+2"}) == []


def test_mail_and_calendar_are_different_connections():
    mail = rs.for_tool("send_email", {"to": "a@b.c"})[0]["resource"]
    calendar = rs.for_tool("schedule_meeting", {"title": "x"})[0]["resource"]
    assert mail != calendar


def test_reading_resources_never_raises():
    for arguments in (None, {"path": None}, {"path": 5}, {"path": ""},
                      {"path": "\x00bad"}):
        assert isinstance(rs.for_tool("read_file", arguments), list)


def test_an_undeclared_resource_is_still_caught():
    """The whole point: task B's plan never mentioned this file."""
    rs.acquire(file_of("/tmp/report.md"), rs.WRITE, "a")
    wanted = rs.for_tool("write_file", {"path": "/tmp/report.md"})[0]
    outcome = rs.acquire(wanted["resource"], wanted["mode"], "b")
    assert outcome["acquired"] is False and outcome["held_by"] == "a"


# --- Release -------------------------------------------------------------------------

def test_release_frees_it_for_the_next_task():
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")
    assert rs.release(file_of("/tmp/a"), "a") is True
    assert rs.acquire(file_of("/tmp/a"), rs.WRITE, "b")["acquired"] is True


def test_releasing_something_not_held_is_harmless():
    assert rs.release(file_of("/tmp/a"), "a") is False


def test_release_all_frees_everything_one_task_held():
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")
    rs.acquire(file_of("/tmp/b"), rs.READ, "a")
    rs.acquire(rs.identity(rs.SCREEN, ""), rs.CONTROL, "a")
    assert len(rs.release_all("a")) == 3
    assert rs.snapshot()["held"] == []


def test_one_task_releasing_does_not_free_anothers():
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")
    rs.acquire(file_of("/tmp/b"), rs.WRITE, "b")
    rs.release_all("a")
    assert len(rs.holders(file_of("/tmp/b"))) == 1


def test_a_restart_inherits_no_locks():
    """A lock held by a process that no longer exists is not a lock."""
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")
    rs.clear()
    assert rs.acquire(file_of("/tmp/a"), rs.WRITE, "b")["acquired"] is True


def test_the_ledger_is_never_written_to_disk():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/resources.py").read_text())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("write_text", "atomic_write_text", "dump", "connect", "mkdir"):
        assert forbidden not in called, f"core/resources.py persists via {forbidden}"


# --- Atomicity -------------------------------------------------------------------------

def test_acquisition_has_no_await_in_it():
    """That is what makes it atomic: the event loop cannot interleave two
    coroutines inside a function that never yields."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(rs.acquire))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Await)]


@pytest.mark.asyncio
async def test_two_tasks_starting_in_the_same_tick_cannot_both_win():
    results = []

    async def claim(task_id):
        results.append(rs.acquire(file_of("/tmp/contested"), rs.WRITE, task_id))

    await asyncio.gather(*(claim(f"t{i}") for i in range(8)))
    assert sum(1 for r in results if r["acquired"]) == 1


@pytest.mark.asyncio
async def test_many_readers_and_one_writer_race_correctly():
    async def claim(task_id, mode):
        return rs.acquire(file_of("/tmp/mixed"), mode, task_id)

    outcomes = await asyncio.gather(
        claim("w", rs.WRITE), *(claim(f"r{i}", rs.READ) for i in range(5)))
    granted = [o for o in outcomes if o["acquired"]]
    # Either the writer alone, or the readers - never a writer beside a reader.
    modes = {rs.holders(file_of("/tmp/mixed"))[0].mode} if granted else set()
    assert len(granted) >= 1
    if rs.WRITE in modes:
        assert len(granted) == 1


# --- What the user is told ----------------------------------------------------------------

def test_a_conflict_names_the_resource_the_owner_and_the_wait():
    rs.acquire(file_of("/tmp/project.py"), rs.WRITE, "task-a")
    outcome = rs.acquire(file_of("/tmp/project.py"), rs.WRITE, "task-b")
    sentence = rs.describe_conflict(outcome, "Task A")
    assert "Task A" in sentence
    assert "project.py" in sentence
    assert "waiting" in sentence


def test_waiting_tasks_are_visible():
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "b")
    waiting = rs.waiting_for(file_of("/tmp/a"))
    assert [entry["task"] for entry in waiting] == ["b"]


def test_a_task_stops_waiting_once_it_acquires():
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "b")
    rs.release_all("a")
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "b")
    assert rs.waiting_for(file_of("/tmp/a")) == []


def test_a_long_held_lock_is_visible_without_being_broken():
    rs.acquire(file_of("/tmp/a"), rs.WRITE, "a")
    rs.holders(file_of("/tmp/a"))[0].since = time.time() - rs.LONG_HELD_SECONDS - 1
    found = rs.snapshot()
    assert found["long_held"], "a stuck lock is invisible"
    # And still held: nothing is force-released on a timer.
    assert rs.acquire(file_of("/tmp/a"), rs.WRITE, "b")["acquired"] is False


def test_the_snapshot_holds_no_credentials():
    import json

    rs.acquire(rs.identity(rs.CONNECTION, "email"), rs.CONTROL, "a")
    blob = json.dumps(rs.snapshot())
    for word in ("password", "token", "secret", "api_key"):
        assert word not in blob


# --- A lock is not a permission -----------------------------------------------------------

def test_the_module_cannot_authorise_anything():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/resources.py").read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for forbidden in ("authorize", "SafetyGuard", "requires_confirmation",
                      "preapproved", "get_permissions"):
        assert forbidden not in names, f"core/resources.py touches {forbidden}"
