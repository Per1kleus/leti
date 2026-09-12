"""The three capabilities meeting each other, and the promises that span them.

Each is tested on its own elsewhere. What is here is the seams: that a GUI errand
still goes through a tool when one exists, that a multi-step errand stops when
the screen is wrong rather than clicking on, that nothing anywhere gained a
timer, and - the one that matters most for cost - that what these tools hand the
model is bounded no matter how big the file was.
"""
from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

from core import computer_use, documents
from tests.test_file_intelligence import build_docx, build_pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- Nothing reaches the model unbounded ------------------------------------------------

@pytest.mark.asyncio
async def test_reading_a_huge_file_hands_the_model_a_bounded_answer(tmp_path):
    """The whole point. A 4 MB file, and what comes back fits in a prompt."""
    from tools.documents import ReadDocumentTool

    big = tmp_path / "enormous.txt"
    big.write_text("\n".join(f"line {i}: some text about pricing and terms"
                             for i in range(100_000)))
    assert big.stat().st_size > 4_000_000

    result = await ReadDocumentTool().run(str(big), question="pricing")
    assert result.success
    returned = len(json.dumps(result.output))
    assert returned < 20_000, f"{returned} characters went to the model"
    assert result.output["more_available"] is True


@pytest.mark.asyncio
async def test_comparing_many_files_stays_bounded_too(tmp_path):
    from tools.documents import CompareDocumentsTool

    paths = []
    for n in range(6):
        path = tmp_path / f"offer_{n}.pdf"
        path.write_bytes(build_pdf([[f"Offer {n}", "filler " * 300],
                                    ["Pricing", f"Total price: {10_000 + n} EUR"]]))
        paths.append(str(path))

    result = await CompareDocumentsTool().run(paths=paths, question="total price")
    assert result.success
    size = len(json.dumps(result.output))
    assert size < 40_000, f"{size} characters went to the model"
    assert len(result.output["compared"]) == 6


def test_the_module_will_not_return_a_whole_file_however_it_is_asked():
    """Every path out of extract() goes through the same budget."""
    assert documents.MAX_EXTRACT_CHARS < 30_000
    source = inspect.getsource(documents.extract)
    assert "budget" in source
    assert "MAX_EXTRACT_CHARS" in source


# --- A project's files, then a task about them -------------------------------------------

@pytest.mark.asyncio
async def test_the_files_a_task_is_about_come_from_the_project_it_belongs_to(
        tmp_path, monkeypatch):
    from core import task_manager
    from tools import projects
    from tools.documents import FindDocumentsTool

    monkeypatch.setattr(projects, "projects_root", lambda: tmp_path / "projects")
    monkeypatch.setattr(projects, "_state_path", lambda: tmp_path / "active.json")
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")

    projects.project_dir("Parot").mkdir(parents=True, exist_ok=True)
    projects.save_manifest("Parot", {"name": "Parot", "description": "", "instructions": ""})
    build_docx(projects.project_dir("Parot") / "invoice_march.docx",
               [("Heading1", "Invoice"), ("", "Amount due: 1,200 EUR")])
    (tmp_path / "unrelated_invoice.docx").write_bytes(b"not even a docx")
    projects.set_active_project("Parot")

    found = await FindDocumentsTool().run("find the invoices and compare them")
    assert found.success
    assert [f["name"] for f in found.output["files"]] == ["invoice_march.docx"]

    task = task_manager.create_task(
        "compare the invoices", ["find the invoices", "read each one", "write the report"],
        project="Parot")
    view = task_manager.detail(task)
    assert view["project"] == "Parot"
    assert view["steps_total"] == 3 and view["steps_done"] == 0


# --- The GUI is still the last resort ------------------------------------------------------

@pytest.mark.asyncio
async def test_a_dedicated_tool_wins_even_when_the_request_sounds_like_clicking():
    from tools.computer_use import ChooseComputerApproachTool, PlanComputerTaskTool

    for request in ("send an email to the supplier",
                    "create a calendar event for the site visit",
                    "read the contract pdf and tell me the notice period"):
        single = await ChooseComputerApproachTool().run(request)
        assert single.output["layer"] == computer_use.LAYER_TOOL, request
        planned = await PlanComputerTaskTool().run(request, steps=["do it", "check it"])
        assert planned.output["layer"] == computer_use.LAYER_TOOL, request


@pytest.mark.asyncio
async def test_a_multi_step_errand_observes_acts_and_verifies_in_that_order():
    """The errand from the request: open the site, find the invoice, download it."""
    from tools.computer_use import (
        CompleteComputerStepTool, EndComputerSessionTool, PlanComputerTaskTool,
        VerifyScreenTool,
    )

    started = await PlanComputerTaskTool().run(
        "open the accounting application and export the invoice",
        steps=["open the application", "find the invoice", "export it"])
    session_id = started.output["session_id"]
    session = computer_use.get_session(session_id)

    # Acting before looking is refused, by the session itself.
    assert session.may_act("click Export", "Export")[0] is False

    step = await VerifyScreenTool().run(session_id,
                                        expected="the accounting application, invoice list",
                                        observed="the accounting application showing an invoice list",
                                        next_action="click the March invoice")
    assert step.success and step.output["matches"] and step.output["may_continue"]
    assert step.output["progress"]["position"] == "0/3"

    done = await CompleteComputerStepTool().run(session_id, "the application is open")
    assert done.output["progress"]["position"] == "1/3"

    # And the act invalidated the look, so the next one has to look again.
    assert session.may_act("click Export", "Export")[0] is False

    ended = await EndComputerSessionTool().run(session_id, "finished")
    assert ended.output["progress"]["done"] == 1
    assert ended.output["steps_taken"] == 1


@pytest.mark.asyncio
async def test_an_unexpected_screen_stops_the_errand_rather_than_guessing():
    from tools.computer_use import PlanComputerTaskTool, VerifyScreenTool

    started = await PlanComputerTaskTool().run(
        "open the settings dialog in this application",
        steps=["open the dialog", "change the setting"])
    session_id = started.output["session_id"]

    result = await VerifyScreenTool().run(
        session_id, expected="the Settings dialog, General tab",
        observed="a modal asking whether to save unsaved changes",
        next_action="click General")
    assert result.success is False
    assert "Stopping" in result.error
    assert computer_use.get_session(session_id).closed is True

    # And it will not act after that, whatever it is told.
    again = await VerifyScreenTool().run(session_id, expected="anything",
                                         observed="anything", next_action="click")
    assert again.success is False


@pytest.mark.asyncio
async def test_the_step_budget_still_ends_an_errand_that_will_not_finish():
    from tools.computer_use import VerifyScreenTool

    session = computer_use.open_session("keep going", plan=["a", "b"])
    for n in range(computer_use.MAX_STEPS):
        session.observe("the same window")
        session.record(f"click thing {n}", str(n))
    assert session.steps_left == 0

    result = await VerifyScreenTool().run(session.id, expected="the same window",
                                          observed="the same window",
                                          next_action="click one more thing")
    assert result.success
    assert result.output["may_continue"] is False
    assert str(computer_use.MAX_STEPS) in result.output["reason"]


# --- Nothing gained a loop ------------------------------------------------------------------

@pytest.mark.parametrize("module", ["core/documents.py", "tools/documents.py",
                                    "core/computer_use.py", "tools/computer_use.py"])
def test_none_of_the_new_code_runs_on_a_clock(module):
    """No scheduler, no polling, no capture. These are called and they return."""
    tree = ast.parse((PROJECT_ROOT / module).read_text())
    called = {node.func.attr if isinstance(node.func, ast.Attribute)
              else getattr(node.func, "id", "")
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for forbidden in ("sleep", "Timer", "Thread", "create_task", "capture", "screenshot"):
        assert forbidden not in called, f"{module} calls {forbidden}()"
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)], f"{module} has a loop"


def test_the_task_panel_is_refreshed_by_events_not_by_a_timer():
    hud = (PROJECT_ROOT / "gui" / "hud.html").read_text()
    code = re.sub(r"/\*.*?\*/|<!--.*?-->", " ", hud, flags=re.S)
    code = re.sub(r"(?<![:'\"])//[^\n]*", " ", code)
    intervals = sorted(set(re.findall(r"setInterval\(([^,]+),", code)))
    assert intervals == ["refreshDiagnostics", "refreshStats", "refreshWeather", "tickClock"], (
        f"something new polls: {intervals}")
    handler = re.search(r"window\.letiActivity = function\(entry\)\{(.*?)\n  \};", code, re.S)
    assert handler and "refreshTasksSoon()" in handler.group(1)


def test_the_task_panel_is_not_on_screen_when_nothing_is_running():
    hud = (PROJECT_ROOT / "gui" / "hud.html").read_text()
    assert 'id="taskCard" hidden' in hud
    assert "taskCard.hidden = true" in hud


# --- The guard is still the only authority -----------------------------------------------------

def test_nothing_new_authorises_anything():
    """Not one of the files added or changed for this may decide that an action is
    allowed. SafetyGuard does that, from the one place it always did."""
    for module in ("core/documents.py", "tools/documents.py", "core/computer_use.py",
                   "tools/computer_use.py", "gui/api.py"):
        tree = ast.parse((PROJECT_ROOT / module).read_text())
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        assert "authorize" not in called, f"{module} calls authorize()"
        assert "set_unattended" not in called, f"{module} changes unattended mode"


def test_the_new_tools_all_have_a_permission_class():
    from core.config_loader import get_permissions

    entries = get_permissions()["tools"]
    for name in ("find_documents", "inspect_document", "read_document", "compare_documents",
                 "look_at_image", "plan_computer_task", "computer_step_done"):
        assert name in entries, f"{name} has no permission entry"
        assert entries[name]["action"] in ("read", "execute"), name
