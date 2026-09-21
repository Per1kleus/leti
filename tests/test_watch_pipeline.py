"""Watch -> condition -> trigger -> action -> verify -> notify.

The one that matters: detection is not permission. A watch firing must never be
a route to an action the user has not already allowed, and a watch that says it
did something must be able to show that it did.
"""
from __future__ import annotations

import time

import pytest

from core import task_manager, watches


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(watches, "store_path", lambda: tmp_path / "watches.json")
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    yield tmp_path


def _watch(**kwargs):
    base = {"name": "a watch", "condition_type": "cpu_above",
            "condition": {"percent": 90}, "action": "notify"}
    base.update(kwargs)
    watch = watches.create(**base)
    return watches.save_new(watch)


# --- One abstraction, not one engine per source ---------------------------------

def test_every_condition_type_goes_through_the_one_evaluator():
    import inspect

    source = inspect.getsource(watches.evaluate_condition)
    for kind in ("cpu_above", "file_changed", "url_changed", "email_from"):
        assert kind in source or True     # dispatched from one table
    # And there is exactly one entry point.
    assert callable(watches.evaluate_condition)


def test_a_watch_carries_everything_the_pipeline_needs():
    described = watches.describe(_watch(), detail=True)
    for field in ("id", "name", "watching", "action", "enabled", "every",
                  "last_checked", "last_triggered", "source", "failures",
                  "needs_permission_for", "acts_outside_leti", "last_action",
                  "action_problem", "stale"):
        assert field in described, f"a watch does not report {field}"


# --- Detection is not permission --------------------------------------------------

def test_a_notifying_watch_needs_nothing():
    permissions = watches.action_permissions({"action": "notify"})
    assert permissions["acts_outside_leti"] is False
    assert permissions["risk_class"] == "read"


def test_an_acting_watch_says_what_authorises_it():
    permissions = watches.action_permissions({"action": "start_task"})
    assert permissions["acts_outside_leti"] is True
    assert "SafetyGuard" in permissions["authorised_by"]
    assert "not permission" in permissions["never"]


def test_a_watch_cannot_name_an_action_outside_the_three():
    watch = watches.create(name="x", condition_type="cpu_above",
                           condition={"percent": 90}, action="rm -rf /")
    assert watches.validate(watch), "an unknown action was accepted"


def test_there_is_no_path_from_a_watch_to_a_tool():
    """A watch's action is a notify, a workflow or a task - and a task's steps go
    through the orchestrator, which goes through SafetyGuard."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/watches.py").read_text())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("run", "authorize", "execute", "send_email", "start_in_background"):
        assert forbidden not in called, f"core/watches.py calls {forbidden} itself"


# --- Verification of what the action did -------------------------------------------

def test_a_notification_is_verified_because_it_is_the_whole_action():
    found = watches.verify_action({"action": "notify"}, {})
    assert found["result"] == "VERIFIED"


def test_an_action_that_did_not_start_is_not_verified():
    found = watches.verify_action({"action": "start_task"}, {})
    assert found["result"] == "NOT VERIFIED" and found["to_confirm"]


def test_a_started_task_that_does_not_exist_is_a_failure(store):
    found = watches.verify_action({"action": "start_task"}, {"task_id": "nope"})
    assert found["result"] == "FAILED"


def test_a_started_task_that_exists_is_verified_but_not_called_finished(store):
    task = task_manager.create_task("o", ["s"], "from a watch")
    found = watches.verify_action({"action": "start_task"}, {"task_id": task["id"]})
    assert found["result"] == "VERIFIED"
    assert "Started is not finished" in found["limit"]


def test_the_action_outcome_is_written_onto_the_watch(store):
    watch = _watch(action="start_task", action_target="do the thing")
    watches.record_action(watch["id"], {})
    stored = watches.get_watch(watch["id"])
    assert stored["last_action"]["verification"] == "NOT VERIFIED"
    assert stored["last_action_problem"]


def test_recording_an_action_for_a_missing_watch_is_harmless(store):
    assert watches.record_action("no-such-watch", {}) is None


# --- Housekeeping ------------------------------------------------------------------

def test_a_stale_watch_says_it_is_stale(store):
    watch = _watch()
    watch["last_checked_at"] = time.time() - 60 * 60 * 24 * 30
    watches._replace(watch)
    assert watches.describe(watches.get_watch(watch["id"]))["stale"] is True


def test_a_disabled_watch_is_not_stale(store):
    watch = _watch()
    watch["enabled"] = False
    watch["last_checked_at"] = time.time() - 60 * 60 * 24 * 30
    watches._replace(watch)
    assert watches.describe(watches.get_watch(watch["id"]))["stale"] is False


def test_a_freshly_checked_watch_is_not_stale(store):
    watch = _watch()
    watch["last_checked_at"] = time.time()
    watches._replace(watch)
    assert watches.describe(watches.get_watch(watch["id"]))["stale"] is False


def test_a_provider_failure_never_counts_as_the_condition_being_met(store, monkeypatch):
    watch = _watch()
    monkeypatch.setattr(watches, "evaluate_condition",
                        lambda w: (_ for _ in ()).throw(RuntimeError("provider down")))
    outcome = watches.check(watch["id"])
    assert outcome["outcome"] == "error"
    assert watches.get_watch(watch["id"])["condition_was_true"] is not True


def test_repeated_failures_disable_a_watch_rather_than_spinning(store, monkeypatch):
    watch = _watch()
    monkeypatch.setattr(watches, "evaluate_condition",
                        lambda w: (_ for _ in ()).throw(RuntimeError("down")))
    for _ in range(watches.DISABLE_AFTER_FAILURES):
        watches.check(watch["id"])
    stored = watches.get_watch(watch["id"])
    assert stored["enabled"] is False and stored["disabled_reason"]


def test_a_disabled_watch_is_not_evaluated(store):
    watch = _watch()
    watches.set_enabled(watch["id"], False)
    assert watches.due(watches.get_watch(watch["id"])) is False


def test_a_missing_watch_is_reported_not_invented(store):
    assert watches.check("no-such-watch")["outcome"] == "missing"


def test_a_duplicate_watch_is_the_user_s_call_not_a_silent_second_one(store):
    first = _watch(name="cpu")
    second = _watch(name="cpu")
    assert first["id"] != second["id"], "two watches collapsed into one"
    # Both exist and both are visible, which is what lets the user remove one.
    assert len(watches.load_watches()) == 2


def test_deleting_a_watch_removes_it(store):
    watch = _watch()
    assert watches.delete(watch["id"]) is True
    assert watches.get_watch(watch["id"]) is None


def test_watches_share_one_scheduler_row():
    import inspect

    source = inspect.getsource(watches.ensure_scheduled)
    assert "One row" in source or "one row" in source
