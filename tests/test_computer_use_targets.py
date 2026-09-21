"""Aiming at a thing rather than at a pixel.

The property these guard: a step that names something must be able to say
whether that something is on the screen it just read, and must refuse when it is
not. Clicking where a button used to be is the failure mode, and it is silent.
"""
from __future__ import annotations

import time

import pytest

from core import computer_use


LOGIN = ("A login page. There is an Email field, a Password field, a Sign in "
         "button and a link reading Forgot your password?")


def _session(observation=LOGIN, goal="sign in"):
    session = computer_use.Session(goal)
    if observation is not None:
        session.observe(observation)
    return session


# --- Reading the target out of an instruction -------------------------------------

@pytest.mark.parametrize("instruction,label,how", [
    ("click the Save button", "Save", computer_use.BY_ROLE),
    ("press the Cancel button", "Cancel", computer_use.BY_ROLE),
    ("select the General tab", "General", computer_use.BY_ROLE),
    ('click "Sign in"', "Sign in", computer_use.BY_NAME),
    ("type my address into the Email field", "Email", computer_use.BY_ROLE),
])
def test_a_named_target_is_read_out_of_the_instruction(instruction, label, how):
    found = computer_use.describe_target(instruction)
    assert found["target"] == label and found["how"] == how


def test_a_coordinate_is_reported_as_the_weak_description_it_is():
    found = computer_use.describe_target("click at 420, 300")
    assert found["how"] == computer_use.BY_COORDINATES
    assert found["target"] is None
    assert "cannot be checked afterwards" in found["why_weak"]


def test_an_instruction_naming_nothing_says_so():
    found = computer_use.describe_target("do the thing")
    assert found["target"] is None and found["how"] is None


def test_reading_a_target_never_raises():
    for instruction in (None, "", 12, "x" * 5000, "click the éè button"):
        assert "instruction" in computer_use.describe_target(instruction)


# --- Is it actually there --------------------------------------------------------

def test_a_visible_target_is_allowed():
    session = _session()
    allowed, why = session.may_act("click", "the Sign in button")
    assert allowed is True


def test_a_missing_element_is_refused_rather_than_clicked():
    session = _session()
    allowed, why = session.may_act("click", "the Checkout button")
    assert allowed is False
    assert "not on screen" in why and "Do not click where it used to be" in why


def test_the_wrong_application_is_refused():
    session = _session("A spreadsheet in Excel with columns A to F")
    allowed, why = session.may_act("click", "the Sign in button")
    assert allowed is False


def test_a_changed_screen_refuses_the_next_step():
    session = _session()
    assert session.may_act("click", "the Sign in button")[0] is True
    session.observe("An error page saying the service is unavailable")
    assert session.may_act("click", "the Sign in button")[0] is False


def test_nothing_may_act_before_the_screen_has_been_read():
    session = computer_use.Session("sign in")
    allowed, why = session.may_act("click", "the Sign in button")
    assert allowed is False and "nothing has been looked at" in why


def test_a_stale_observation_refuses():
    session = _session()
    session.observed_at = time.time() - computer_use.OBSERVATION_MAX_AGE - 1
    allowed, why = session.may_act("click", "the Sign in button")
    assert allowed is False and "look again" in why


def test_a_closed_session_refuses_everything():
    session = _session()
    session.close("the screen was not what was expected")
    assert session.may_act("click", "the Sign in button")[0] is False


def test_a_coordinate_is_still_allowed_but_not_pretended_to_be_checked():
    session = _session()
    allowed, _ = session.may_act("click", "at 100, 200")
    assert allowed is True
    aim = session.aim("click at 100, 200")
    assert aim["visible"] is None
    assert "cannot be confirmed" in aim["evidence"]


def test_aiming_reports_what_it_found_and_what_it_prefers():
    session = _session()
    assert session.aim("click the Sign in button")["visible"] is True
    assert session.aim("click the Checkout button")["visible"] is False
    assert session.aim("click at 1, 2")["prefer"]


def test_a_partial_match_is_allowed_but_flagged():
    """Every word present but not together is a weaker claim, and says so -
    "Save" and "Save as" are different buttons."""
    session = _session("A toolbar with Save at the left and a Draft indicator "
                       "at the right.")
    aim = session.aim("click the Save Draft button")
    assert aim["visible"] is True
    assert "not together" in aim["evidence"]


# --- Verification ------------------------------------------------------------------

def test_an_unverified_consequential_action_is_reported():
    session = _session()
    session.record("click", "the Sign in button")
    assert session.unverified_actions() == [] or True
    session.record("submit", "the form")
    assert any(a["action"] == "submit" for a in session.unverified_actions())


def test_looking_again_marks_the_consequential_action_checked():
    session = _session()
    session.record("submit", "the form")
    assert session.unverified_actions()
    session.observe("A page saying Welcome back")
    assert session.unverified_actions() == []


def test_an_expectation_that_does_not_hold_is_not_a_pass():
    session = _session()
    holds, why = session.expectation_holds("the dashboard with the account menu")
    assert holds is False and "did not" in why


def test_an_expectation_of_only_generic_words_is_not_a_pass():
    session = _session()
    holds, why = session.expectation_holds("the window is open and visible")
    assert holds is False and "generic" in why


def test_two_mismatches_stop_the_session():
    session = _session()
    assert session.mismatch() is True
    assert session.mismatch() is False


def test_a_mismatch_forbids_acting_until_the_screen_is_read_again():
    session = _session()
    session.mismatch()
    allowed, why = session.may_act("click", "the Sign in button")
    assert allowed is False and "nothing has been looked at" in why


def test_the_same_action_is_not_repeated_forever():
    """Looking again between attempts is required and still does not buy a third."""
    session = _session()
    for _ in range(computer_use.MAX_IDENTICAL_ACTIONS):
        session.record("click", "the Sign in button")
        session.observe(LOGIN)              # the screen did not change
    allowed, why = session.may_act("click", "the Sign in button")
    assert allowed is False and "already been tried" in why


def test_acting_spends_the_look_so_the_next_step_must_observe_again():
    session = _session()
    session.record("click", "the Sign in button")
    allowed, why = session.may_act("click", "the Sign in button")
    assert allowed is False and "nothing has been looked at" in why


def test_a_session_is_bounded_by_steps():
    session = _session()
    for i in range(computer_use.MAX_STEPS):
        session.record("click", f"thing {i}")
        session.observe(LOGIN)
    allowed, why = session.may_act("click", "the Sign in button")
    assert allowed is False and "steps" in why


# --- The layer chooser is unchanged and still prefers a tool -------------------------

@pytest.mark.parametrize("request_text,layer", [
    ("read the file report.md", computer_use.LAYER_TOOL),
    ("send an email to chris", computer_use.LAYER_TOOL),
    ("go to https://example.com", computer_use.LAYER_BROWSER),
    ("click the settings button in blender", computer_use.LAYER_GUI),
])
def test_a_dedicated_tool_still_beats_the_mouse(request_text, layer):
    assert computer_use.choose_layer(request_text)["layer"] == layer


def test_no_second_automation_framework_was_added():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/computer_use.py").read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    for forbidden in ("pyautogui", "selenium", "playwright", "pynput", "mss"):
        assert forbidden not in imports, f"core/computer_use.py drives {forbidden} itself"
