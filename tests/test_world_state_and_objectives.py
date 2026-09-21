"""Noticing that the world moved, and holding on to what is being pursued.

Two modules under test. core/world_state.py is what lets a failure be answered
with the right response instead of the same response again; core/objectives.py
is what lets a goal survive a turn without every request becoming one.
"""
from __future__ import annotations

import pytest

from core import intent as intent_reader
from core import objectives, verification, world_state
from core.safety_guard import ConfirmationDenied, PermissionDenied


@pytest.fixture(autouse=True)
def _clean():
    world_state.forget()
    yield
    world_state.forget()


# --- Classifying why something did not work ------------------------------------------

@pytest.mark.parametrize("problem,expected", [
    (PermissionError("Permission denied"), world_state.PERMISSION_PROBLEM),
    (ConfirmationDenied("'send_email' is an external action"), world_state.PERMISSION_PROBLEM),
    (PermissionDenied("forbidden path"), world_state.PERMISSION_PROBLEM),
    (ModuleNotFoundError("No module named 'pandas'"), world_state.DEPENDENCY_PROBLEM),
    (TimeoutError("the request timed out"), world_state.EXTERNAL_SERVICE),
    (RuntimeError("503 Service Unavailable"), world_state.EXTERNAL_SERVICE),
    (RuntimeError("rate limit exceeded"), world_state.EXTERNAL_SERVICE),
    (RuntimeError("element not found on screen"), world_state.STALE_OBSERVATION),
    (ValueError("missing required argument 'to'"), world_state.MISSING_INFORMATION),
    (RuntimeError("the folder no longer exists"), world_state.ENVIRONMENT_CHANGED),
])
def test_a_failure_is_classified_by_what_it_says(problem, expected):
    assert world_state.classify(problem)["kind"] == expected


def test_an_unrecognised_failure_says_unknown_rather_than_guessing():
    found = world_state.classify(RuntimeError("the flange is wrong colour"))
    assert found["kind"] == world_state.UNKNOWN
    assert found["certain"] is False
    assert "not known" in found["do"]


def test_every_kind_has_a_response_and_a_retry_decision():
    for kind in world_state.KINDS:
        response = world_state.RESPONSES[kind]
        assert response["do"] and isinstance(response["retry_differently"], bool)


def test_a_refusal_is_never_retried_differently():
    """Retrying a refusal a different way is trying to get around it."""
    for problem in (ConfirmationDenied("no"), PermissionDenied("no"),
                    PermissionError("Permission denied")):
        assert world_state.unlocks_recovery(problem) is False
        assert world_state.classify(problem)["retry_differently"] is False


def test_recovery_is_unlocked_only_for_the_kinds_worth_another_attempt():
    assert world_state.unlocks_recovery(TimeoutError("timed out")) is True
    assert world_state.unlocks_recovery(RuntimeError("syntax error")) is False
    assert world_state.unlocks_recovery(RuntimeError("no such file or directory")) is False


def test_the_task_manager_asks_this_module_rather_than_keeping_its_own_list():
    import pathlib

    from core import task_manager

    source = pathlib.Path("core/task_manager.py").read_text()
    assert "_RECOVERABLE" not in source
    assert task_manager.is_recoverable(TimeoutError("timed out")) is True


# --- Observe, act, observe again, compare ----------------------------------------------

def test_a_result_matching_the_expectation_is_verified():
    world_state.begin("t", "book the meeting")
    world_state.expect("t", "the event appears on Thursday")
    world_state.observe("t", "the calendar now shows the event on Thursday at 3pm")
    outcome = world_state.compare("t")
    assert outcome["verdict"] == verification.VERIFIED and outcome["continue"] is True


def test_a_result_that_does_not_match_fails_and_is_classified():
    world_state.begin("t", "book the meeting")
    world_state.expect("t", "the event appears on Thursday")
    world_state.observe("t", "an error dialog saying the server refused the request")
    outcome = world_state.compare("t")
    assert outcome["verdict"] == verification.FAILED
    assert outcome["continue"] is False and outcome["classification"]


def test_acting_with_nothing_observed_is_not_verified():
    world_state.begin("t")
    world_state.expect("t", "the file is saved")
    outcome = world_state.compare("t")
    assert outcome["verdict"] == verification.NOT_VERIFIED
    assert outcome["continue"] is False


def test_an_old_observation_is_not_evidence_about_now():
    world_state.begin("t")
    world_state.expect("t", "the settings window is open")
    world_state.observe("t", "the settings window is open", now=1000.0)
    outcome = world_state.compare("t", now=1000.0 + world_state.STALE_AFTER_SECONDS + 1)
    assert outcome["verdict"] == verification.NOT_VERIFIED
    assert outcome["next"] == "observe again"


def test_expecting_nothing_confirms_nothing():
    holds, why = world_state.expectation_holds("", "anything at all")
    assert holds is False and "nothing specific" in why


def test_nothing_tracked_is_not_applicable_rather_than_fine():
    assert world_state.compare("never-started")["verdict"] == verification.NOT_APPLICABLE


# --- The record ---------------------------------------------------------------------

def test_the_state_carries_what_a_recovery_needs_to_know():
    world_state.begin("t", "email the report", project="Turbine", goal="send it")
    world_state.succeeded("t", "found the report")
    world_state.expect("t", "the mail client reports it sent")
    world_state.observe("t", "an authentication error", location="Mail")
    described = world_state.get("t").describe()
    assert described["project"] == "Turbine"
    assert described["last_successful_action"] == "found the report"
    assert described["where"] == "Mail"
    assert described["expected_next"] and described["last_observed"]


def test_a_recovery_note_says_what_worked_what_was_expected_and_what_to_do():
    world_state.begin("t", "email the report")
    world_state.succeeded("t", "found the report")
    world_state.expect("t", "the mail client reports it sent")
    world_state.observe("t", "503 from the mail server")
    note = world_state.recovery_note("t", RuntimeError("503 Service Unavailable"))
    assert "found the report" in note and "external service failure" in note


def test_an_unknown_cause_is_said_to_be_unknown_in_the_recovery_note():
    world_state.begin("t")
    note = world_state.recovery_note("t", RuntimeError("the flange is wrong colour"))
    assert "do not invent one" in note


def test_a_resolved_entity_is_remembered_so_the_next_step_does_not_resolve_it_again():
    world_state.begin("t")
    world_state.remember_entity("t", "customer", {"id": "l1", "name": "Acme"})
    assert world_state.get("t").entities["customer"]["name"] == "Acme"


def test_nothing_is_written_to_disk():
    import pathlib

    source = pathlib.Path("core/world_state.py").read_text()
    for forbidden in ("write_text", "open(", "atomic_write", "json.dump"):
        assert forbidden not in source, f"core/world_state.py does {forbidden}"


def test_forgetting_drops_it():
    world_state.begin("t")
    world_state.forget("t")
    assert world_state.get("t") is None


# --- Objectives -----------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "what's the weather", "write me an email to Chris", "what is 12 times 40",
    "read main.py", "book a meeting with Sam on Tuesday",
])
def test_an_ordinary_request_does_not_become_an_objective(text):
    decision = objectives.worth_tracking(intent_reader.read(text), text)
    assert decision["track"] is False and decision["why"]


@pytest.mark.parametrize("text", [
    "my goal is to reach 200k in pipeline this quarter",
    "keep going until every lead has been contacted",
    "work towards getting the test suite green",
])
def test_a_request_that_says_it_is_ongoing_is_tracked(text):
    decision = objectives.worth_tracking(intent_reader.read(text), text)
    assert decision["track"] is True and decision["as"] == "goal"


def test_a_genuinely_multi_stage_request_is_tracked():
    text = "research the top five CRM tools, compare their pricing and write a summary document"
    decision = objectives.worth_tracking(intent_reader.read(text), text)
    assert decision["track"] is True and decision["as"] == "task"


def test_a_task_reads_as_goal_plan_actions_results_progress_next(tmp_path, monkeypatch):
    from core import task_manager

    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    task = task_manager.create_task("get the report out", ["gather", "write", "send"],
                                    "the report", success_criteria="the report is sent")
    view = objectives.from_task(task)
    assert view["goal"] == "get the report out"
    assert view["success_criteria"] == "the report is sent"
    assert view["plan"] == ["gather", "write", "send"]
    assert view["progress"]["basis"] == objectives.BY_STEPS
    assert view["next_action"] == "gather"


def test_steps_finished_is_not_claimed_to_be_the_objective_achieved(tmp_path, monkeypatch):
    from core import task_manager

    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    task = task_manager.create_task("o", ["a", "b"], "t")
    view = objectives.from_task(task)
    assert "not the same as the objective achieved" in view["progress"]["how"]


def test_a_task_waiting_on_the_user_says_so_as_its_next_action(tmp_path, monkeypatch):
    from core import task_manager

    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    task = task_manager.create_task("o", ["a", "b"], "t")
    task["status"] = task_manager.WAITING_FOR_USER
    task["blocked_reason"] = "it needs approval to send the email"
    view = objectives.from_task(task)
    assert view["next_action"].startswith("WAITING FOR YOU")


def test_nothing_in_flight_produces_no_turn_note(tmp_path, monkeypatch):
    from core import business_goals, task_manager

    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(business_goals, "load_goals", lambda: [])
    assert objectives.turn_note() == ""


def test_an_unmeasurable_goal_is_not_given_a_percentage(monkeypatch):
    from core import business_goals, task_manager

    monkeypatch.setattr(task_manager, "load_tasks", lambda: [])
    goal = {"id": "g1", "name": "grow the business", "status": business_goals.ACTIVE,
            "tasks": [], "results": []}
    view = objectives.from_goal(goal)
    assert view["progress"]["basis"] == objectives.UNMEASURABLE
    assert view["progress"]["percent"] is None
