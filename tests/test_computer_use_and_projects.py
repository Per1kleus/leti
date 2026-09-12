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
    # A dedicated tool exists for each of these, so the mouse is the wrong answer.
    ("create a calendar event for Friday at three", computer_use.LAYER_TOOL),
    ("book a meeting with Maria next week", computer_use.LAYER_TOOL),
    ("read the pdf and summarise it", computer_use.LAYER_TOOL),
    ("compare these spreadsheets", computer_use.LAYER_TOOL),
    ("add milk to my todo list", computer_use.LAYER_TOOL),
    # Getting something off a website is browser work before it is mouse work.
    ("download the invoice from their billing page", computer_use.LAYER_BROWSER),
    ("find the invoice on the website", computer_use.LAYER_BROWSER),
    # And a real GUI errand is still a GUI errand.
    ("rename the file in the application's own file browser", computer_use.LAYER_GUI),
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
    session.observed_at -= computer_use.OBSERVATION_MAX_AGE + 1

    assert session.may_act("click", "Save")[0] is False


def test_an_observation_goes_stale_long_before_the_session_does():
    """A session can sit for ten minutes while the model thinks. A look at the
    screen cannot: a window that was in front of you ten minutes ago is not
    evidence about where the mouse should go now."""
    assert computer_use.OBSERVATION_MAX_AGE < computer_use.SESSION_IDLE_TIMEOUT
    session = computer_use.open_session("goal")
    session.observe("a window")
    session.observed_at -= computer_use.OBSERVATION_MAX_AGE + 5
    allowed, why = session.may_act("click", "Save")
    assert allowed is False and "look again" in why


# --- Consequential actions get checked ------------------------------------------------------

@pytest.mark.parametrize("action,expected", [
    ("click Send", True), ("submit the form", True), ("delete the row", True),
    ("confirm the purchase", True), ("publish it", True),
    ("scroll down", False), ("read the list", False), ("open the menu", False),
])
def test_actions_that_change_something_are_known_from_ones_that_do_not(action, expected):
    assert computer_use.is_consequential(action) is expected


def test_a_consequential_action_nobody_looked_at_afterwards_is_reported():
    """Clicking Send and never checking is the one failure a GUI session can hide
    completely, so it is reported rather than assumed either way."""
    session = computer_use.open_session("send the reply")
    session.observe("a compose window with a Send button")
    session.record("click Send", "Send")

    assert [a["action"] for a in session.unverified_actions()] == ["click Send"]
    assert "never checked" in session.summary()["warning"]

    session.observe("the message was sent, inbox showing")
    assert session.unverified_actions() == []
    assert session.summary()["warning"] is None


def test_an_ordinary_action_is_not_flagged_as_unchecked():
    session = computer_use.open_session("look around")
    session.observe("a list of files")
    session.record("scroll down", "")
    assert session.unverified_actions() == []


# --- A plan, and where the errand has got to -------------------------------------------------

def test_a_plan_is_counted_never_estimated():
    session = computer_use.open_session("download the invoice", plan=[
        "open the billing page", "find the invoice", "download it", "move it"])
    assert session.plan_progress()["position"] == "0/4"
    assert session.plan_progress()["current"] == "open the billing page"

    session.complete_plan_step("the page is open")
    progress = session.plan_progress()
    assert progress["position"] == "1/4"
    assert progress["current"] == "find the invoice"
    assert [s["status"] for s in progress["steps"]] == ["done", "pending", "pending", "pending"]


def test_a_session_with_no_plan_says_working_rather_than_inventing_a_fraction():
    session = computer_use.open_session("click something")
    assert session.plan_progress()["position"] == "working"
    assert session.plan_progress()["planned"] == 0


def test_a_plan_is_bounded():
    session = computer_use.open_session("a lot", plan=[f"step {i}" for i in range(40)])
    assert len(session.plan) == computer_use.MAX_PLAN_STEPS


def test_finishing_every_planned_step_leaves_nothing_to_advance():
    session = computer_use.open_session("two things", plan=["one", "two"])
    session.complete_plan_step()
    session.complete_plan_step()
    assert session.complete_plan_step() is None
    assert session.plan_progress()["position"] == "2/2"


@pytest.mark.asyncio
async def test_planning_a_multi_step_errand_still_defers_to_a_dedicated_tool():
    """Writing a plan down does not make the GUI the right layer."""
    from tools.computer_use import ChooseComputerApproachTool

    result = await ChooseComputerApproachTool().run(
        "send an email to Maria and then another to John",
        steps=["open the mail client", "write the first", "send it"])
    assert result.success
    assert result.output["layer"] == computer_use.LAYER_TOOL
    assert result.output["suggestion"] == "send_email"
    assert "session_id" not in result.output, "a GUI session was opened anyway"


@pytest.mark.asyncio
async def test_a_gui_errand_gets_a_session_that_remembers_the_plan():
    from tools.computer_use import ChooseComputerApproachTool, CompleteComputerStepTool

    result = await ChooseComputerApproachTool().run(
        "open Blender, click Settings and change the theme",
        steps=["open Blender", "click Settings", "change the theme"])
    assert result.success and result.output["layer"] == computer_use.LAYER_GUI
    session_id = result.output["session_id"]
    assert result.output["progress"]["position"] == "0/3"

    done = await CompleteComputerStepTool().run(session_id, "Blender is open")
    assert done.success
    assert done.output["progress"]["position"] == "1/3"
    assert done.output["progress"]["current"] == "click Settings"


@pytest.mark.asyncio
async def test_one_tool_answers_the_question_with_or_without_a_plan():
    """choose_computer_approach and plan_computer_task were two tools answering the
    same question. They are one tool with an optional plan now - so asking without
    steps still works exactly as it did, and is not an error."""
    from tools.computer_use import ChooseComputerApproachTool

    plain = await ChooseComputerApproachTool().run("click the export button in this program")
    assert plain.success and plain.output["layer"] == computer_use.LAYER_GUI
    assert plain.output["progress"]["position"] == "working"
    assert computer_use.get_session(plain.output["session_id"]).plan == []

    planned = await ChooseComputerApproachTool().run(
        "click the export button in this program", steps=["find it", "click it"])
    assert planned.output["progress"]["position"] == "0/2"


@pytest.mark.asyncio
async def test_a_plan_is_not_started_when_a_tool_should_do_the_job():
    from tools.computer_use import ChooseComputerApproachTool

    result = await ChooseComputerApproachTool().run(
        "send an email to Maria", steps=["open mail", "write it", "send it"])
    assert result.output["layer"] == computer_use.LAYER_TOOL
    assert result.output["plan_not_started"] == ["open mail", "write it", "send it"]
    assert "session_id" not in result.output


@pytest.mark.asyncio
async def test_progress_cannot_be_advanced_for_a_session_that_has_no_plan():
    from tools.computer_use import CompleteComputerStepTool

    session = computer_use.open_session("no plan here")
    result = await CompleteComputerStepTool().run(session.id, "done")
    assert result.success is False
    assert "no plan" in result.error


# --- Stopping leaves an honest record -------------------------------------------------------
#
# The failure these are about is not a wrong click. It is a session that stopped -
# cancelled, closed, out of steps - and left a record that still reads like work in
# progress or, worse, like work that succeeded.

@pytest.mark.asyncio
async def test_a_cancelled_errand_leaves_no_step_looking_like_it_is_still_running():
    from tools.computer_use import EndComputerSessionTool

    session = computer_use.open_session(
        "open Blender and change the theme",
        plan=["open Blender", "click Settings", "change the theme"])
    session.complete_plan_step("Blender is open")

    ended = await EndComputerSessionTool().run(session.id, "the user cancelled it")
    assert ended.success
    statuses = [s["status"] for s in ended.output["progress"]["steps"]]
    assert statuses == ["done", "abandoned", "abandoned"]
    assert "pending" not in statuses, "a step is still shown as work in progress"
    # And the summary says which parts of the errand did not happen.
    assert ended.output["plan_incomplete"] == ["click Settings", "change the theme"]
    assert ended.output["progress"]["finished"] is False


def test_an_abandoned_step_is_not_reported_as_the_one_being_worked_on():
    session = computer_use.open_session("two things", plan=["one", "two"])
    session.close("the user cancelled it")
    progress = session.plan_progress()
    assert progress["current"] is None, "a closed session still claims to be doing something"
    assert progress["abandoned"] == 2 and progress["done"] == 0


@pytest.mark.asyncio
async def test_a_closed_session_cannot_report_a_successful_verification():
    """The dangerous answer is "matches: true" from a session that already stopped:
    it reads as the errand going fine."""
    from tools.computer_use import VerifyScreenTool

    session = computer_use.open_session("open Settings")
    session.close("the screen was not what was expected")

    result = await VerifyScreenTool().run(session.id, expected="Settings window",
                                          observed="Settings window", next_action="click Save")
    assert result.success is False
    assert "closed" in result.error
    assert result.output["session"]["steps_taken"] == 0, "a closed session recorded an action"


@pytest.mark.asyncio
async def test_a_verification_that_failed_is_never_a_success():
    from tools.computer_use import VerifyScreenTool

    session = computer_use.open_session("open Settings")
    result = await VerifyScreenTool().run(
        session.id, expected="the Settings window, General tab",
        observed="an unsaved-changes dialog", next_action="click Save")

    assert result.success is False and result.output["matches"] is False
    # And the session is over rather than carrying on into a window it did not predict.
    assert computer_use.get_session(session.id).closed is True
    assert "Do not carry on clicking" in result.output["note"]


@pytest.mark.asyncio
async def test_a_step_cannot_be_marked_done_in_a_session_that_no_longer_exists():
    from tools.computer_use import CompleteComputerStepTool, VerifyScreenTool

    for tool in (CompleteComputerStepTool(), VerifyScreenTool()):
        arguments = {"session_id": "gui-nonexistent"}
        if isinstance(tool, VerifyScreenTool):
            arguments.update(expected="anything", observed="anything")
        result = await tool.run(**arguments)
        assert result.success is False and "gui-nonexistent" in result.error


def test_an_observation_from_before_an_action_cannot_be_used_after_it():
    """Every action moves the screen on, so the look that authorised it is spent."""
    session = computer_use.open_session("click twice")
    session.observe("the Settings window")
    assert session.may_act("mouse_click", "Save")[0] is True
    session.record("mouse_click", "Save")
    allowed, reason = session.may_act("mouse_click", "Close")
    assert allowed is False and "look" in reason.lower()


# --- What the interface sees --------------------------------------------------------------

def test_the_interface_reads_live_sessions_and_captures_nothing_to_do_it():
    from gui.api import LetiAPI

    session = computer_use.open_session("download the invoice",
                                        plan=["open the page", "download it"])
    session.observe("the billing page")
    session.record("click Download", "Download")

    view = LetiAPI.get_computer_use(LetiAPI.__new__(LetiAPI))
    live = [s for s in view["sessions"] if s["session_id"] == session.id]
    assert live and live[0]["progress"]["position"] == "0/2"
    assert live[0]["last_actions"] == ["click Download"]

    source = inspect.getsource(LetiAPI.get_computer_use)
    for forbidden in ("capture", "screenshot", "ScreenCapture", "sleep"):
        assert forbidden not in source, f"the task view {forbidden}s"


def test_a_finished_session_leaves_the_interface_showing_nothing():
    """Idle means idle: no session, no panel, no work to keep it up to date."""
    from gui.api import LetiAPI

    for live in computer_use.active_sessions():
        live.close("test cleanup")
    assert LetiAPI.get_computer_use(LetiAPI.__new__(LetiAPI))["sessions"] == []


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
    for name in ("choose_computer_approach", "verify_screen", "end_computer_session",
                 "computer_step_done"):
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
