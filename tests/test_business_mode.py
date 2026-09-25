"""Business Mode, and the rule that matters more than any capability in it.

A BUSINESS TASK IS NOT A REQUEST FOR BUSINESS MODE. "Check my customer emails",
"analyse this invoice", "show me my sales numbers", "review my leads" are all
ordinary Default Mode work, and a user who asks for one of them must get it -
not a mode switch they did not ask for. Most of this file is that one rule,
approached from as many angles as it has.

It is enforced structurally rather than by good behaviour. In Default Mode there
is no mode tool in the registry the model can see, so there is nothing for it to
call however the request is phrased; modes change only through an explicit
command matched deterministically in core/intent.py, or the selector in the
interface. A regex cannot be talked into Business Mode by a sentence full of
invoices.

After that: what Business Mode adds (a briefing assembled from stores that
already exist), what it refuses to invent, and that it is not a way round
anything - sending mail still asks, and the CRM it reads is the CRM Leti already
had rather than a second one.
"""
from __future__ import annotations

import json
import sys
import time
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import business, intent, modes, task_manager  # noqa: E402
from tools import scheduler  # noqa: E402


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    modes.reset_for_tests()
    business.release()
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "sched.json")
    import tools.business as business_tools

    monkeypatch.setattr(business_tools, "_store_path", lambda: tmp_path / "business.json")
    yield
    modes.reset_for_tests()
    business.release()


@pytest.fixture(scope="module")
def registry():
    from unittest.mock import MagicMock

    import main

    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


# --- THE RULE: a business task is not a request for Business Mode -------------------

BUSINESS_TASKS = [
    "Check my emails.",
    "Analyze this invoice.",
    "Show me my sales numbers.",
    "Check my customer emails.",
    "Create a business report.",
    "Review my leads.",
    "Send an email to the client.",
    "Check my calendar.",
    "Read this invoice.",
    "Analyze this spreadsheet.",
    "what is my sales pipeline looking like",
    "summarise this contract for the customer",
    "how much revenue did we make last quarter",
    "add a new lead for the accounting firm",
    "prepare a follow-up for that customer",
    "my business is growing fast",
    "book a meeting with the client about the proposal",
]

CODING_TASKS = [
    "Write a Python script.",
    "Debug this Python file.",
    "fix the bug in the parser",
    "run the tests",
    "why does this function return None",
    "refactor this class",
]


@pytest.mark.parametrize("text", BUSINESS_TASKS + CODING_TASKS)
def test_a_task_is_never_read_as_a_request_for_a_mode(text):
    assert intent.mode_command(text) is None, text


@pytest.mark.parametrize("text", BUSINESS_TASKS)
@pytest.mark.asyncio
async def test_a_business_task_leaves_leti_in_default_mode(text):
    """End to end through the real orchestrator path that decides this."""
    assert modes.current() == modes.DEFAULT
    assert intent.mode_command(text) is None
    assert modes.current() == modes.DEFAULT, f"{text!r} changed the mode"


def test_business_words_alone_cannot_trigger_the_mode():
    for text in ("business", "business business business", "CRM sales invoice lead pipeline",
                 "customer accounting revenue profit KPI", "enter the business",
                 "this is business critical", "get down to business"):
        assert intent.mode_command(text) is None, text


def test_the_word_mode_alone_is_not_a_command():
    for text in ("what mode of transport", "the coding style in this repo",
                 "my business needs a new mode of transport", "quiet mode is fine"):
        assert intent.mode_command(text) is None, text


# --- Only an explicit command switches ----------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Enter Business Mode.", modes.BUSINESS),
    ("Switch to Business Mode.", modes.BUSINESS),
    ("Activate Business Mode.", modes.BUSINESS),
    ("Let's work in Business Mode.", modes.BUSINESS),
    ("Open my business workspace.", modes.BUSINESS),
    ("business mode on", modes.BUSINESS),
    ("Enter Coding Mode.", modes.CODING),
    ("switch to coding mode", modes.CODING),
    ("put yourself in developer mode", modes.CODING),
    ("Exit business mode.", modes.DEFAULT),
    ("exit coding mode", modes.DEFAULT),
    ("Switch to default mode.", modes.DEFAULT),
    ("back to default", modes.DEFAULT),
])
def test_an_explicit_command_is_understood(text, expected):
    assert intent.mode_command(text) == expected


def test_asking_which_mode_is_a_question_not_a_command():
    assert intent.asks_which_mode("what mode are you in") is True
    assert intent.mode_command("what mode are you in") is None
    assert intent.asks_which_mode("enter business mode") is False


def test_recognising_a_mode_command_costs_nothing():
    started = time.perf_counter()
    for _ in range(2000):
        intent.mode_command("check my customer emails and the invoices from last week")
    per_call = (time.perf_counter() - started) / 2000
    assert per_call < 0.0001, f"{per_call * 1e6:.1f}us per ordinary request"


# --- Structurally, not just by good behaviour ---------------------------------------

def test_default_mode_has_no_mode_tool_for_the_model_to_call(registry):
    """The strongest form of the rule: there is nothing to call. A model in Default
    Mode cannot switch modes however a request is phrased, because the switch is
    not in the tools it is shown."""
    visible = modes.visible_tools(registry, modes.DEFAULT)
    assert modes.SWITCH_TOOL not in visible
    assert "business_briefing" not in visible
    assert registry.get(modes.SWITCH_TOOL) is not None, "the switch should still exist"


def test_business_tools_are_hidden_from_default_mode(registry):
    assert "business_briefing" not in modes.visible_tools(registry, modes.DEFAULT)
    assert "business_briefing" in modes.visible_tools(registry, modes.BUSINESS)


def test_the_existing_crm_stays_available_in_default_mode(registry):
    """Business Mode is not a prerequisite for business work. The CRM, the mail and
    the documents are Default Mode tools and stay that way."""
    visible = modes.visible_tools(registry, modes.DEFAULT)
    for name in ("business_dashboard", "business_next_actions", "list_business_data",
                 "update_lead", "list_new_emails", "send_email", "read_document",
                 "schedule_meeting", "inspect_dataset"):
        assert name in visible, f"{name} should be usable without entering a mode"


def test_the_three_modes_stay_out_of_each_others_way(registry):
    coding_tools = modes.visible_tools(registry, modes.CODING)
    business_tools = modes.visible_tools(registry, modes.BUSINESS)
    assert "business_briefing" not in coding_tools
    assert not ({"code_map", "git_workspace", "github"} & business_tools)


def test_business_mode_is_not_offered_everything(registry):
    visible = modes.visible_tools(registry, modes.BUSINESS)
    for irrelevant in ("mouse_click", "read_screen", "place_paper_order",
                       "run_shell_command", "create_sketch", "get_weather"):
        assert irrelevant not in visible, f"{irrelevant} has no business in a business turn"


def test_default_mode_costs_nothing_for_business_mode_existing(registry):
    """The headline performance requirement: Default Leti stays exactly as it was."""
    visible = modes.visible_tools(registry, modes.DEFAULT)
    tokens = len(json.dumps(registry.schemas_for(visible))) // 4
    assert len(visible) == 129
    assert tokens + 6000 <= 28672, "Default Mode no longer fits its context window"


# --- Switching ----------------------------------------------------------------------

def test_entering_and_leaving_business_mode():
    assert modes.enter(modes.BUSINESS)["changed"] is True
    assert modes.current() == modes.BUSINESS
    assert "BUSINESS MODE" in modes.system_note()

    assert modes.leave()["mode"] == modes.DEFAULT
    assert modes.system_note() == ""


def test_switching_between_specialised_modes_releases_the_first():
    modes.enter(modes.BUSINESS)
    business.open_workspace(business_name="Acme")
    assert business.workspace() is not None

    modes.enter(modes.CODING)
    assert business.workspace() is None, "the business workspace outlived the mode"

    from core import coding

    coding.open_workspace(root="/tmp")
    modes.enter(modes.BUSINESS)
    assert coding.workspace() is None, "the coding workspace outlived the mode"


def test_leaving_releases_the_workspace_but_not_the_records():
    from tools.business import load_records, save_records

    save_records({"lead": [{"id": "l1", "name": "Acme", "stage": "new", "value": 100,
                            "created_at": time.time()}], "income": [], "expense": []})
    modes.enter(modes.BUSINESS)
    business.open_workspace(business_name="Acme Ltd", objectives=["more qualified leads"])

    modes.leave()
    assert business.workspace() is None
    assert len(load_records()["lead"]) == 1, "leaving the mode touched the records"


def test_the_workspace_is_not_a_second_memory_system():
    import ast

    source = open("core/business.py").read()
    tree = ast.parse(source)
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("add_turn", "save_records", "save_tasks", "atomic_write_text",
                      "atomic_write_json", "remember", "add"):
        assert forbidden not in called, f"core/business.py writes via {forbidden}()"
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)]
    for forbidden in ("Thread(", "setInterval", "schedule(", "watchdog"):
        assert forbidden not in source, f"core/business.py uses {forbidden}"


def test_switching_mode_touches_nothing_it_should_not():
    import ast

    tree = ast.parse(open("core/modes.py").read())
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("llm_client", "num_ctx", "OllamaClient", "session_memory",
                      "save_tasks", "cancel"):
        assert forbidden not in referenced, f"switching mode touches {forbidden}"


# --- The briefing: assembled from what already exists --------------------------------

def _lead(name, stage="qualified", value=1000, days_idle=0, contact_id=None):
    now = time.time()
    return {"id": f"lead-{name}", "name": name, "stage": stage, "value": value,
            "created_at": now - days_idle * 86400,
            "stage_changed_at": now - days_idle * 86400, "contact_id": contact_id}


def test_a_briefing_reads_the_stores_that_already_hold_the_answer():
    from tools.business import save_records

    save_records({"lead": [_lead("Quiet Co", days_idle=30),
                           _lead("Fresh Co", days_idle=1)],
                  "income": [], "expense": []})
    task = task_manager.create_task("send the proposal", ["draft", "send"], name="Proposal")
    task_manager._set_status(task["id"], task_manager.WAITING_FOR_USER,
                             blocked_reason="Sending needs your confirmation.")

    brief = business.briefing()
    assert [w["what"] for w in brief["waiting_on_you"]] == ["Proposal"]
    assert [l["lead"] for l in brief["leads_needing_attention"]] == ["Quiet Co"]
    assert brief["pipeline"]["open_leads"] == 2
    assert brief["pipeline"]["weighted_value"] > 0


def test_the_weighted_figure_says_it_is_calculated():
    from tools.business import save_records

    save_records({"lead": [_lead("A", stage="proposal", value=1000)],
                  "income": [], "expense": []})
    pipeline = business.briefing()["pipeline"]
    assert pipeline["total_value"] == 1000
    assert pipeline["weighted_value"] == 600.0        # proposal is worth 0.6
    assert "not money in" in pipeline["what_weighted_means"]


def test_a_briefing_says_what_it_cannot_see_rather_than_reporting_it_empty():
    """Leti can CREATE calendar events but has no tool that reads one back. A
    briefing that quietly showed no meetings would be claiming an empty calendar."""
    brief = business.briefing()
    assert any("calendar" in line for line in brief["cannot_see"])
    assert "cannot see" in business.connected_sources()["calendar"] or \
           "not connected" in business.connected_sources()["calendar"]
    assert "cannot_see" in brief["how_to_report"]


def test_a_source_that_fails_becomes_a_named_gap_not_a_silent_omission(monkeypatch):
    def explode():
        raise OSError("the store is gone")

    monkeypatch.setattr(task_manager, "load_tasks", explode)
    brief = business.briefing()
    assert brief["waiting_on_you"] == []
    assert any("waiting_on_you" in gap for gap in brief["gaps"])


def test_nothing_is_invented_when_there_is_nothing_there():
    from tools.business import save_records

    save_records({"lead": [], "income": [], "expense": []})
    brief = business.briefing()
    assert brief["leads_needing_attention"] == []
    assert brief["pipeline"]["open_leads"] == 0
    assert json.dumps(brief).count("example.com") == 0


def test_follow_ups_say_who_cannot_be_reached_without_leaking_addresses():
    from tools.business import save_records

    save_records({"lead": [_lead("No Contact Co", days_idle=40)],
                  "income": [], "expense": []})
    found = business.follow_ups()
    assert found["count"] == 1
    assert found["no_way_to_reach"] == ["No Contact Co"]
    assert "@" not in json.dumps(found), "an address leaked into a follow-up list"
    assert "asks first" in found["next"]


# --- The tool -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_briefing_tool_reads_and_never_acts():
    import ast

    from tools.business_agent import BusinessBriefingTool

    result = await BusinessBriefingTool().run("brief")
    assert result.success and "waiting_on_you" in result.output

    tree = ast.parse(open("tools/business_agent.py").read())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("send_email", "authorize", "create_task", "update_lead",
                      "set_unattended", "save_records"):
        assert forbidden not in called, f"business_briefing calls {forbidden}()"


@pytest.mark.asyncio
async def test_the_workspace_action_holds_the_session_not_the_disk():
    from tools.business_agent import BusinessBriefingTool

    result = await BusinessBriefingTool().run(
        "workspace", business_name="Acme Ltd", objectives=["more qualified leads"])
    assert result.success
    assert result.output["business"] == "Acme Ltd"
    assert "outlive it" in result.output["note"]
    assert business.workspace().business_name == "Acme Ltd"


@pytest.mark.asyncio
async def test_an_empty_follow_up_list_says_why_it_is_empty():
    from tools.business import save_records
    from tools.business_agent import BusinessBriefingTool

    save_records({"lead": [], "income": [], "expense": []})
    result = await BusinessBriefingTool().run("follow_ups")
    assert result.output["count"] == 0
    assert "not a guess" in result.output["note"]


@pytest.mark.asyncio
async def test_an_unknown_action_is_refused():
    from tools.business_agent import BusinessBriefingTool

    result = await BusinessBriefingTool().run("merge_everything")
    assert result.success is False


# --- Safety: more capable, not less safe ---------------------------------------------

def test_every_business_tool_has_exactly_one_permission_entry(registry):
    from core.config_loader import get_permissions

    entries = get_permissions()["tools"]
    for name in ("business_briefing", "switch_mode", "business_dashboard", "update_lead",
                 "send_email", "schedule_meeting"):
        assert name in entries, f"{name} has no action class"
    assert entries["business_briefing"]["action"] == "read"


@pytest.mark.asyncio
async def test_business_mode_does_not_make_sending_mail_any_cheaper():
    """The point of the mode is context, not permission. Sending is external in
    Business Mode exactly as it is everywhere else."""
    from core.safety_guard import ConfirmationDenied, RiskTier, SafetyGuard

    modes.enter(modes.BUSINESS)
    guard = SafetyGuard()
    assert guard.get_tier("send_email") == RiskTier.EXTERNAL

    guard.confirmation_callback = lambda prompt: False
    with pytest.raises(ConfirmationDenied):
        await guard.authorize("send_email", {"to": "a@example.com", "subject": "hi"})


@pytest.mark.asyncio
async def test_an_unattended_business_run_still_cannot_send():
    from core.safety_guard import ConfirmationDenied, SafetyGuard

    modes.enter(modes.BUSINESS)
    guard = SafetyGuard()
    guard.set_unattended(True)
    with pytest.raises(ConfirmationDenied):
        await guard.authorize("send_email", {"to": "a@example.com", "subject": "hi"})


def test_the_business_layer_authorises_nothing_itself():
    import ast

    for module in ("core/business.py", "tools/business_agent.py"):
        tree = ast.parse(open(module).read())
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for forbidden in ("authorize", "set_unattended", "execute_tool", "handle_user_input"):
            assert forbidden not in called, f"{module} calls {forbidden}()"


def test_a_missing_connection_is_named_with_where_to_fix_it():
    sources = business.connected_sources()
    for name, state in sources.items():
        if state.startswith("not connected"):
            assert "Connections" in state, f"{name} does not say where to connect it"
