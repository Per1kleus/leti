"""Watch & Act: the transition, the error that is not a condition, and the fact
that seeing something is not permission to act on it."""
from __future__ import annotations

import sys
import time
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import task_manager, watches  # noqa: E402
from tools import scheduler  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(watches, "store_path", lambda: tmp_path / "watches.json")
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "scheduled.json")
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")


def _cpu_watch(**over):
    w = watches.create("cpu", "cpu_above", {"percent": 90, "for_minutes": 0}, **over)
    return watches.save_new(w)


# --- Creating and persisting --------------------------------------------------------

def test_a_watch_is_created_enabled_and_persisted():
    watch = _cpu_watch()

    assert watch["enabled"] is True
    stored = watches.get_watch(watch["id"])
    assert stored["condition_type"] == "cpu_above"
    assert stored["condition_was_true"] is False


@pytest.mark.parametrize("kind,condition,fragment", [
    ("cpu_above", {"percent": 500}, "between 1 and 100"),
    ("file_changed", {}, "needs a path"),
    ("url_changed", {"url": "not-a-url"}, "http"),
    ("email_from", {}, "not something Leti can watch"),
])
def test_an_unsupported_or_invalid_watch_is_refused(kind, condition, fragment):
    problems = watches.validate(watches.create("w", kind, condition))
    assert any(fragment in p for p in problems), problems


def test_an_action_that_needs_a_target_must_have_one():
    problems = watches.validate(
        watches.create("w", "cpu_above", {"percent": 90}, action="run_workflow"))
    assert any("needs a target" in p for p in problems)


def test_an_unreadable_store_does_not_take_leti_down(tmp_path, monkeypatch):
    monkeypatch.setattr(watches, "store_path", lambda: tmp_path / "broken.json")
    (tmp_path / "broken.json").write_text("{ nope")
    assert watches.load_watches() == []


# --- Transition, not repetition ------------------------------------------------------

def test_a_watch_fires_when_the_condition_becomes_true_and_not_again(monkeypatch):
    """Ten minutes above the threshold is one event, not one per check."""
    watch = _cpu_watch(cooldown_minutes=0)
    load = {"percent": 10.0}
    monkeypatch.setattr(watches, "evaluate_condition",
                        lambda w: (load["percent"] > 90, {}))

    assert watches.check(watch["id"])["outcome"] == "false"
    load["percent"] = 95.0
    assert watches.check(watch["id"])["outcome"] == "triggered"
    assert watches.check(watch["id"])["outcome"] == "still_true", "it fired twice"
    assert watches.check(watch["id"])["outcome"] == "still_true"


def test_it_can_fire_again_after_the_condition_clears(monkeypatch):
    watch = _cpu_watch(cooldown_minutes=0)
    load = {"high": True}
    monkeypatch.setattr(watches, "evaluate_condition", lambda w: (load["high"], {}))

    assert watches.check(watch["id"])["outcome"] == "triggered"
    load["high"] = False
    assert watches.check(watch["id"])["outcome"] == "cleared"
    load["high"] = True
    assert watches.check(watch["id"])["outcome"] == "triggered"


def test_a_cooldown_holds_a_second_trigger_back(monkeypatch):
    watch = _cpu_watch(cooldown_minutes=60)
    load = {"high": True}
    monkeypatch.setattr(watches, "evaluate_condition", lambda w: (load["high"], {}))

    assert watches.check(watch["id"])["outcome"] == "triggered"
    load["high"] = False
    watches.check(watch["id"])
    load["high"] = True
    assert watches.check(watch["id"])["outcome"] == "cooling_down"


# --- An error is never a condition ---------------------------------------------------

def test_a_failing_monitor_never_reads_as_the_condition_being_met(monkeypatch):
    watch = _cpu_watch()

    def broken(_w):
        raise watches.ConditionError("the service is unreachable")

    monkeypatch.setattr(watches, "evaluate_condition", broken)
    outcome = watches.check(watch["id"])

    assert outcome["outcome"] == "error"
    assert watches.get_watch(watch["id"])["condition_was_true"] is False
    assert watches.get_watch(watch["id"])["last_triggered_at"] is None


def test_a_watch_that_keeps_failing_is_disabled_rather_than_left_to_fail(monkeypatch):
    watch = _cpu_watch()
    monkeypatch.setattr(watches, "evaluate_condition",
                        lambda w: (_ for _ in ()).throw(watches.ConditionError("gone")))

    for _ in range(watches.DISABLE_AFTER_FAILURES):
        watches.check(watch["id"])

    stored = watches.get_watch(watch["id"])
    assert stored["enabled"] is False
    assert "Disabled after" in stored["disabled_reason"]


def test_a_successful_check_clears_the_failure_count(monkeypatch):
    watch = _cpu_watch()
    calls = {"n": 0}

    def sometimes(_w):
        calls["n"] += 1
        if calls["n"] == 1:
            raise watches.ConditionError("blip")
        return False, {}

    monkeypatch.setattr(watches, "evaluate_condition", sometimes)
    watches.check(watch["id"])
    assert watches.get_watch(watch["id"])["failures"] == 1
    watches.check(watch["id"])
    assert watches.get_watch(watch["id"])["failures"] == 0


# --- Real evaluators ------------------------------------------------------------------

def test_a_file_watch_treats_the_first_look_as_a_baseline(tmp_path):
    target = tmp_path / "watched.txt"
    target.write_text("one")
    watch = watches.create("f", "file_changed", {"path": str(target)})

    first, state = watches.evaluate_condition(watch)
    assert first is False, "the first look reported a change"

    watch["condition_state"] = state
    time.sleep(0.01)
    target.write_text("two")
    changed, _ = watches.evaluate_condition(watch)
    assert changed is True


def test_a_file_watch_on_a_missing_file_is_an_error_not_a_change(tmp_path):
    watch = watches.create("f", "file_changed", {"path": str(tmp_path / "nope.txt")})
    with pytest.raises(watches.ConditionError):
        watches.evaluate_condition(watch)


def test_cpu_must_stay_above_the_threshold_for_the_whole_window(monkeypatch):
    watch = watches.create("c", "cpu_above", {"percent": 90, "for_minutes": 5})
    monkeypatch.setattr("psutil.cpu_percent", lambda interval=None: 95.0)

    now_true, state = watches.evaluate_condition(watch)
    assert now_true is False, "fired immediately despite a 5 minute window"

    watch["condition_state"] = {"above_since": time.time() - 400}
    later_true, _ = watches.evaluate_condition(watch)
    assert later_true is True


def test_cpu_dropping_below_resets_the_window(monkeypatch):
    watch = watches.create("c", "cpu_above", {"percent": 90, "for_minutes": 5})
    watch["condition_state"] = {"above_since": time.time() - 400}
    monkeypatch.setattr("psutil.cpu_percent", lambda interval=None: 10.0)

    is_true, state = watches.evaluate_condition(watch)
    assert is_true is False and state["above_since"] is None


# --- Scheduling: one row, and none at all when idle ------------------------------------

def test_no_watches_means_nothing_is_scheduled():
    watches.ensure_scheduled()
    assert scheduler.load_tasks() == [], "an idle Leti scheduled something"


def test_all_watches_share_one_scheduled_row():
    _cpu_watch()
    _cpu_watch()
    watches.save_new(watches.create("f", "file_changed", {"path": "/tmp/x"}))

    tasks = scheduler.load_tasks()
    assert len(tasks) == 1, f"one row per watch: {len(tasks)}"
    assert tasks[0]["name"] == watches.SCHEDULER_TASK_NAME
    assert tasks[0]["schedule_type"] == "interval"


def test_removing_the_last_watch_removes_the_scheduled_row():
    watch = _cpu_watch()
    assert len(scheduler.load_tasks()) == 1

    watches.delete(watch["id"])
    watches.ensure_scheduled()

    assert scheduler.load_tasks() == [], "Leti kept waking up for nothing"


def test_the_row_ticks_as_often_as_the_most_frequent_watch_needs():
    watches.save_new(watches.create("slow", "cpu_above", {"percent": 90},
                                    interval_minutes=60))
    watches.save_new(watches.create("fast", "cpu_above", {"percent": 90},
                                    interval_minutes=5))

    assert scheduler.load_tasks()[0]["every_minutes"] == 5


def test_a_watch_is_not_due_before_its_interval():
    watch = _cpu_watch(interval_minutes=10)
    assert watches.due(watch) is True

    watch["last_checked_at"] = time.time()
    assert watches.due(watch) is False
    watch["last_checked_at"] = time.time() - 700
    assert watches.due(watch) is True


def test_a_disabled_or_expired_watch_is_never_due():
    watch = _cpu_watch()
    watch["enabled"] = False
    assert watches.due(watch) is False

    watch["enabled"] = True
    watch["expires_at"] = time.time() - 10
    assert watches.due(watch) is False


def test_enable_disable_and_delete():
    watch = _cpu_watch()
    assert watches.set_enabled(watch["id"], False)["enabled"] is False
    assert watches.set_enabled(watch["id"], True)["enabled"] is True
    assert watches.delete(watch["id"]) is True
    assert watches.get_watch(watch["id"]) is None


def test_re_enabling_forgives_past_failures():
    watch = _cpu_watch()
    watch["failures"] = 4
    watch["disabled_reason"] = "x"
    watches._replace(watch)

    restored = watches.set_enabled(watch["id"], True)
    assert restored["failures"] == 0 and restored["disabled_reason"] is None


# --- No shortcuts ---------------------------------------------------------------------

def test_the_watch_module_never_executes_a_tool_or_authorises_anything():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(watches))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("authorize", "audit_result", "handle_user_input"):
        assert forbidden not in called, f"watches calls {forbidden}()"


def test_watching_has_no_loop_of_its_own():
    import inspect

    source = inspect.getsource(watches)
    assert "while True" not in source
    assert "sleep(" not in source, "the watch system waits on its own"
