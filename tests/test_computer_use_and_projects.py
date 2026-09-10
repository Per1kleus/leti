"""The controlled GUI mode, and project memory that does not become the prompt.

Computer-use is a decision and a check, not a second way to control the machine.
The tests below are mostly about it refusing: refusing the GUI when a tool exists,
refusing to act on a screen it has not looked at, and refusing to keep clicking.
"""
from __future__ import annotations

import ast
import inspect
import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import computer_use, task_manager, workflows  # noqa: E402
from tools import projects  # noqa: E402


# --- Choosing the layer ---------------------------------------------------------------

@pytest.mark.parametrize("request_text,expected", [
    ("read the file notes.txt", computer_use.LAYER_TOOL),
    ("delete that file", computer_use.LAYER_TOOL),
    ("search the web for train times", computer_use.LAYER_TOOL),
    ("send an email to John", computer_use.LAYER_TOOL),
    ("what's on my screen", computer_use.LAYER_TOOL),
    ("go to https://example.com", computer_use.LAYER_BROWSER),
    ("read that article on the page", computer_use.LAYER_BROWSER),
    ("fill in the form", computer_use.LAYER_BROWSER),
    ("open Blender and click the Settings button", computer_use.LAYER_GUI),
    ("change the theme in this application's preferences", computer_use.LAYER_GUI),
])
def test_the_cheapest_capable_layer_is_chosen(request_text, expected):
    """A dedicated tool beats the browser, and the browser beats the mouse."""
    assert computer_use.choose_layer(request_text)["layer"] == expected


def test_choosing_a_layer_costs_nothing_and_calls_nothing():
    """No model call to decide whether to use the model's other tools."""
    tree = ast.parse(inspect.getsource(computer_use))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("chat", "handle_user_input", "authorize", "run"):
        assert forbidden not in called, f"computer_use calls {forbidden}()"


def test_nothing_captures_the_screen_on_its_own():
    """The session records what it is TOLD it saw; it never grabs a screenshot,
    so there is no capture loop to run away with.

    Checked structurally rather than by grepping: this module legitimately contains
    the word "screenshot" inside the pattern that spots a request for one and sends
    it to read_screen instead.
    """
    tree = ast.parse(inspect.getsource(computer_use))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
    for capture in ("pyautogui", "PIL", "mss", "pyscreenshot", "cv2", "subprocess"):
        assert capture not in imported, f"computer_use imports {capture}"

    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.While, ast.For))]
    assert not any(isinstance(n, ast.While) for n in loops), "computer_use has a while loop"

    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "sleep" not in called, "computer_use waits"


# --- Verifying before acting -----------------------------------------------------------

def test_a_session_will_not_act_before_it_has_looked():
    session = computer_use.open_session("open settings")

    allowed, reason = session.may_act("click", "Settings")

    assert allowed is False
    assert "look" in reason.lower()


def test_an_expectation_that_does_not_hold_is_reported_not_ignored():
    session = computer_use.open_session("open settings")
    session.observe("A file save dialog is open")

    holds, why = session.expectation_holds("Settings window General tab")

    assert holds is False
    assert "did not" in why


def test_an_expectation_that_holds_lets_the_next_step_run():
    session = computer_use.open_session("open settings")
    session.observe("The Settings window is open, showing the General tab")

    assert session.expectation_holds("Settings window General tab")[0] is True
    assert session.may_act("click", "Save")[0] is True


def test_acting_invalidates_the_last_look():
    """The screen has moved on, so the next step needs a fresh look at it."""
    session = computer_use.open_session("goal")
    session.observe("Settings window")
    session.record("click", "Save")

    assert session.may_act("click", "Close")[0] is False


# --- Bounded ----------------------------------------------------------------------------

def test_the_same_action_is_not_repeated_forever():
    """The failure mode of GUI automation is clicking the same wrong pixel."""
    session = computer_use.open_session("goal")
    for _ in range(computer_use.MAX_IDENTICAL_ACTIONS):
        session.observe("Settings window")
        assert session.may_act("click", "Save")[0] is True
        session.record("click", "Save")

    session.observe("Settings window")
    allowed, reason = session.may_act("click", "Save")

    assert allowed is False
    assert "already been tried" in reason


def test_a_session_runs_out_of_steps():
    session = computer_use.open_session("goal")
    for i in range(computer_use.MAX_STEPS):
        session.observe("a window")
        session.record("click", f"target-{i}")

    session.observe("a window")
    allowed, reason = session.may_act("click", "one more")

    assert allowed is False and str(computer_use.MAX_STEPS) in reason


def test_a_closed_session_does_nothing_more():
    session = computer_use.open_session("goal")
    session.observe("a window")
    session.close("the dialog never appeared")

    assert session.may_act("click", "x")[0] is False
    assert session.summary()["closed_reason"] == "the dialog never appeared"


def test_a_stale_look_is_not_acted_on(monkeypatch):
    session = computer_use.open_session("goal")
    session.observe("Settings window")
    session.observed_at -= computer_use.SESSION_IDLE_TIMEOUT + 1

    assert session.may_act("click", "Save")[0] is False


# --- The GUI is not a way round the rules ------------------------------------------------

def test_the_gui_tools_do_not_move_the_mouse_themselves():
    """Every click is still mouse_click, called by the orchestrator and authorised
    by SafetyGuard - which is what stops clicking Send being cheaper than
    send_email."""
    from tools import computer_use as gui_tools

    source = inspect.getsource(gui_tools)
    for forbidden in ("pyautogui", "mouse_click(", "keyboard_type(", "subprocess"):
        assert forbidden not in source, f"the GUI tools contain {forbidden}"


def test_the_gui_tools_are_read_only_in_permissions():
    """They observe and decide; they change nothing, so they need no confirmation -
    and the tools that DO change things keep their own classes."""
    from core.config_loader import get_permissions

    entries = get_permissions()["tools"]
    for name in ("choose_computer_approach", "verify_screen", "end_computer_session"):
        assert entries[name]["action"] == "read", name
    # The things that actually touch the machine are unchanged.
    assert entries["mouse_click"]["action"] != "read"
    assert entries["keyboard_type"]["action"] != "read"


# --- Project memory ----------------------------------------------------------------------

@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "projects_root", lambda: tmp_path / "projects")
    monkeypatch.setattr(projects, "_state_path", lambda: tmp_path / "active.json")
    (tmp_path / "projects").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _make(name="Parot Automations", **manifest):
    directory = projects.project_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    projects.save_manifest(name, {"description": "automation work", **manifest})
    return name


def test_a_project_persists_and_can_be_reopened(project):
    _make()
    projects.set_active_project("Parot Automations")

    assert projects.get_active_project() == "Parot Automations"
    assert projects.load_manifest("Parot Automations")["description"] == "automation work"


def test_archiving_keeps_everything_and_only_hides_it(project):
    name = _make()
    (projects.project_dir(name) / "notes.md").write_text("important")
    projects.set_active_project(name)

    projects.set_archived(name, True)

    assert (projects.project_dir(name) / "notes.md").read_text() == "important"
    assert [p["name"] for p in projects.list_projects()] == []
    assert [p["name"] for p in projects.list_projects(include_archived=True)] == [name]
    assert projects.get_active_project() is None, "an archived project stayed active"


def test_restoring_brings_it_back(project):
    name = _make()
    projects.set_archived(name, True)
    projects.set_archived(name, False)

    assert [p["name"] for p in projects.list_projects()] == [name]


def test_project_context_names_the_files_this_request_is_about(project):
    """Not the whole folder every time - that is what dynamic tool routing exists
    to avoid, and it would be undone by pasting a directory listing beside it."""
    name = _make()
    directory = projects.project_dir(name)
    for i in range(30):
        (directory / f"unrelated_{i}.txt").write_text("x")
    (directory / "invoice_template.xlsx").write_text("x")

    focused = projects.project_context(name, request="update the invoice template")
    everything = projects.project_context(name)

    assert "invoice_template.xlsx" in focused
    assert focused.count("unrelated_") < everything.count("unrelated_"), (
        "the request made no difference to what was included")
    assert len(focused) < len(everything)


def test_project_context_holds_paths_not_file_contents(project):
    name = _make()
    (projects.project_dir(name) / "big.txt").write_text("SECRET-CONTENTS " * 200)

    context = projects.project_context(name, request="what is in big.txt")

    assert "big.txt" in context
    assert "SECRET-CONTENTS" not in context, "a file's contents were pasted into the prompt"


def test_a_project_with_no_files_says_so_rather_than_failing(project):
    _make("Empty")
    assert "no files yet" in projects.project_context("Empty")


# --- Tasks and workflows belong to projects -----------------------------------------------

def test_a_task_can_belong_to_a_project(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")

    task = task_manager.create_task("research 20 companies", ["find them"],
                                    "research", project="Parot Automations")

    assert task_manager.get_task(task["id"])["project"] == "Parot Automations"
    assert task_manager.describe(task)["project"] == "Parot Automations"


def test_a_task_without_a_project_is_simply_unattached(tmp_path, monkeypatch):
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")

    assert task_manager.create_task("o", ["s"])["project"] is None


def test_a_workflow_can_belong_to_a_project():
    workflow = workflows.build("Weekly report", ["summarise"], project="Parot Automations")

    assert workflow["project"] == "Parot Automations"
    assert workflows.describe(workflow)["project"] == "Parot Automations"
