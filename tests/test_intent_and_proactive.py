"""Reading the request, budgeting the turn, and what Leti may say unasked.

Four capabilities meet in this file and they share one discipline: none of them
is allowed to be expensive, and none of them is allowed to decide anything.

  Intent (core/intent.py) reads the request before anything is sent. It must cost
  no model call and no measurable time, it must not turn a one-line question into
  a complex turn, and when something is genuinely missing it must produce a
  question rather than a guess.

  The adaptive engine (core/performance.py) says what a turn may spend. The thing
  to protect here is that a saving is never a capability: a smaller budget may
  only ever drop tools the router had already ranked as marginal, and the tool
  the request actually needs must survive every budget.

  Continuity (memory/session_memory.py) remembers what Leti just listed, so "the
  first three" points at the same three. It is not a second memory and it is
  bounded.

  Proactive (core/proactive.py) reads four stores that already exist and produces
  sentences. The tests that matter most are the ones asserting what it CANNOT do:
  no execution, no bypass, nothing at all when it is switched off.
"""
from __future__ import annotations

import sys
import time
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import intent, performance, proactive, task_manager, watches  # noqa: E402
from tools import scheduler  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    from core import computer_use

    monkeypatch.setattr(watches, "store_path", lambda: tmp_path / "watches.json")
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "scheduled.json")
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    # GUI sessions live in a module-level dict, so a session another test file
    # opened would put every turn here in computer-use mode.
    monkeypatch.setattr(computer_use, "_SESSIONS", {})
    proactive._raised.clear()


# --- Intent: what kind of request is this -------------------------------------------

@pytest.mark.parametrize("text,shape", [
    ("hi", intent.CHAT),
    ("thanks leti", intent.CHAT),
    ("what's the price of AAPL", intent.RETRIEVAL),
    ("watch TTWO and tell me if it moves 5%", intent.MONITORING),
    ("email the report to Maria", intent.COMMUNICATION),
    ("open blender and click settings", intent.COMPUTER_USE),
    ("find the best laptop for matlab", intent.RESEARCH),
])
def test_a_request_is_read_as_what_it_is(text, shape):
    assert intent.read(text).shape == shape


def test_the_worked_example_becomes_an_ordered_objective():
    """The request the whole capability exists for."""
    read = intent.read("Find the five best laptops for MATLAB under EUR 1100, "
                       "compare them and put the results into a file.")

    assert read.complexity == intent.COMPLEX
    assert read.stages == ["research", "compare", "write"]
    assert read.output == "file"
    objective = read.as_objective()
    assert "under" in " ".join(objective["constraints"]).lower()
    assert any("five" in c for c in objective["constraints"])
    # And the note the model gets names the order and ends in a check.
    note = intent.system_note(read)
    assert "research -> compare -> write -> verify" in note


def test_a_simple_question_gets_no_note_at_all():
    """The cost of this feature on an ordinary turn has to be nothing."""
    for text in ("what time is it", "what's the weather", "hello"):
        read = intent.read(text)
        assert read.complexity == intent.SIMPLE
        assert intent.system_note(read) == ""
        assert intent.continuity_note(read, ["a", "b"]) == ""


def test_reading_a_request_never_calls_a_model_and_costs_nothing():
    import ast

    source = open("core/intent.py").read()
    for forbidden in ("llm_client", "ollama", "chat(", "generate(", "requests.", "httpx"):
        assert forbidden not in source, f"core/intent.py reaches for {forbidden}"
    assert not [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.While)]

    started = time.perf_counter()
    for _ in range(500):
        intent.read("find five laptops under 1100 and compare them")
    per_call = (time.perf_counter() - started) / 500
    assert per_call < 0.002, f"{per_call * 1000:.2f}ms per request is not free"


def test_an_unreadable_request_is_a_simple_one_rather_than_an_error():
    for odd in (None, "", "\x00\x01", "?" * 5000):
        assert intent.read(odd).complexity in (intent.SIMPLE, intent.NORMAL, intent.COMPLEX)


# --- Intent: ambiguity that would change the answer ----------------------------------

@pytest.mark.parametrize("text,fragment", [
    ("watch that stock and tell me if it drops", "which symbol"),
    ("read that file and summarise it", "which file"),
    ("email him about tomorrow", "who should"),
])
def test_a_missing_piece_that_changes_the_answer_becomes_a_question(text, fragment):
    assert fragment in intent.read(text).question_to_ask.lower()


def test_nothing_is_asked_when_the_conversation_already_said_it():
    """'Watch that stock' after two turns about TTWO is not ambiguous, and asking
    anyway is its own kind of unhelpful."""
    history = [{"role": "user", "content": "how is TTWO doing"},
               {"role": "assistant", "content": "TTWO is at 100."}]
    assert intent.read("watch that stock", history).question_to_ask == ""


@pytest.mark.parametrize("earlier", [
    "read the PDF and summarise it",
    "laptops under 1100 EUR",
    "OK, the USB drive is mounted",
    "the best laptop for MATLAB",
    "the CSV has 400 rows",
])
def test_an_acronym_in_the_conversation_is_not_a_ticker(earlier):
    """Found by running a real turn: the antecedent check was "any two-to-five
    letter capitalised word", so a conversation that had said PDF, EUR, OK, USB
    or MATLAB silently suppressed "which symbol do you mean?" - the question that
    makes "watch that stock" answerable at all."""
    history = [{"role": "user", "content": earlier}]
    assert intent.read("watch that stock", history).question_to_ask


@pytest.mark.parametrize("earlier", [
    "how is TTWO doing",
    "is bitcoin up today",
    "BTC/USD looks volatile",
    "what's the price of AAPL",
])
def test_a_symbol_in_the_conversation_still_answers_it(earlier):
    history = [{"role": "user", "content": earlier}]
    assert intent.read("watch that stock", history).question_to_ask == ""


def test_a_folder_is_not_a_filename():
    """"Open the thesis project" says where a file might be, not which one."""
    assert intent.read("read that file",
                       [{"role": "user", "content": "open the thesis project"}]).question_to_ask
    assert intent.read("read that file",
                       [{"role": "user", "content": "I saved report.pdf"}]).question_to_ask == ""


def test_an_unambiguous_request_is_never_questioned():
    for text in ("watch TTWO and tell me if it drops", "read report.pdf and summarise it",
                 "email maria@example.com about tomorrow"):
        assert intent.read(text).question_to_ask == ""


# --- Continuity: what "the first three" points at ------------------------------------

def test_an_answer_that_listed_things_is_remembered_in_order():
    from core.orchestrator import _enumerated_items

    items = _enumerated_items(
        "Here are the best five:\n\n"
        "1. Dell XPS 15 - 1,049 EUR\n2. Lenovo ThinkPad E16 - 989 EUR\n"
        "3. HP ProBook 450: 1,079 EUR\n4. Acer Swift Go 14\n5. ASUS Vivobook Pro\n\n"
        "Want a comparison?")
    assert items[:3] == ["Dell XPS 15", "Lenovo ThinkPad E16", "HP ProBook 450"]


def test_prose_is_not_a_list():
    from core.orchestrator import _enumerated_items

    assert _enumerated_items("The XPS costs 1049 EUR and was released in 2023.") == []
    assert _enumerated_items("1. Only one item here") == []


@pytest.mark.parametrize("text", ["compare the first three", "put the winner in a file",
                                  "use those instead", "send it to Maria"])
def test_a_request_that_points_backwards_is_recognised(text):
    assert intent.read(text).refers_back is True


def test_a_request_that_finds_what_it_then_refers_to_is_not_pointing_backwards():
    """'Find five laptops and compare them' supplies its own antecedent. Asking
    which five would be absurd."""
    assert intent.read("find five laptops and compare them").refers_back is False


def test_the_referents_are_handed_over_in_the_order_they_were_given():
    note = intent.continuity_note(intent.read("compare the first three"),
                                  ["Dell XPS 15", "Lenovo ThinkPad E16", "HP ProBook 450"])
    assert "1. Dell XPS 15" in note and "3. HP ProBook 450" in note


def test_with_nothing_to_point_at_it_says_to_ask_rather_than_to_pick():
    note = intent.continuity_note(intent.read("compare the first three"), [])
    assert "ask" in note.lower() and "rather than picking" in note.lower()


def test_the_referent_list_is_bounded_and_is_not_a_second_memory():
    from memory import session_memory

    memory = session_memory.SessionMemory.__new__(session_memory.SessionMemory)
    memory._referents = []
    memory.note_referents([f"item {n}" for n in range(60)])
    assert len(memory.recent_referents()) <= session_memory.MAX_REFERENTS
    assert all(len(r) <= session_memory.MAX_REFERENT_CHARS
               for r in memory.recent_referents())
    memory.note_referents([])
    assert memory.recent_referents(), "an empty answer wiped what was remembered"


# --- The adaptive engine -------------------------------------------------------------

def test_the_mode_matches_what_the_turn_actually_is():
    assert performance.for_turn(intent.read("hi"), measure=False).name == performance.IDLE
    assert performance.for_turn(intent.read("what's the price of AAPL"),
                                measure=False).name == performance.SIMPLE
    assert performance.for_turn(intent.read("find five laptops, compare them and write a file"),
                                measure=False).name == performance.COMPLEX


def test_a_complex_turn_is_never_given_a_smaller_budget_than_a_simple_one():
    simple = performance.for_turn(intent.read("what time is it"), measure=False)
    complex_ = performance.for_turn(
        intent.read("research five companies and write a report then email it"), measure=False)
    assert simple.tool_budget is not None
    assert complex_.tool_budget is None, "a complex task was rationed"


def test_pressure_sheds_the_optional_work_and_nothing_else(monkeypatch):
    monkeypatch.setattr(performance, "_resources", lambda: {"cpu": 99.0, "ram": 40.0})
    mode = performance.for_turn(intent.read("find five laptops and compare them"))

    assert mode.under_pressure is True
    assert mode.recall_memory is False and mode.derive_visuals is False
    assert "cpu 99%" in mode.reason
    # The budget still exists, so the turn still gets tools and still runs.
    assert mode.tool_budget is None or mode.tool_budget >= 10


def test_a_broken_resource_read_does_not_ration_anything(monkeypatch):
    def explode():
        raise OSError("no psutil here")

    monkeypatch.setattr(performance, "_resources", explode)
    mode = performance.for_turn(intent.read("anything"))
    assert mode.recall_memory is True and mode.under_pressure is False


def test_deciding_the_mode_costs_no_model_call_and_no_loop():
    import ast

    source = open("core/performance.py").read()
    for forbidden in ("llm_client", "ollama", "Thread", "create_task", "sleep("):
        assert forbidden not in source, f"core/performance.py uses {forbidden}"
    assert not [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.While)]


# --- The budget can shrink a turn but never break one --------------------------------

@pytest.fixture(scope="module")
def registry():
    from unittest.mock import MagicMock

    import main

    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


NEEDED = [
    ("what's the price of AAPL", "get_market_quote"),
    ("read the invoice and tell me the total", "read_document"),
    ("what files are in this folder", "list_files"),
    ("open blender and click settings", "launch_app"),
    ("delete the file notes.txt", "delete_file"),
    ("tell me if my cpu goes above 90", "create_watch"),
    ("what's the weather tomorrow", "get_weather"),
    ("add milk to my todo list", "add_todo"),
    ("search the web for train times", "web_search"),
    ("back up my documents folder", "create_backup"),
]


@pytest.mark.parametrize("text,needed", NEEDED)
def test_a_budget_never_removes_the_tool_the_request_needs(registry, text, needed):
    """The first version of this counted tools to a number and dropped list_files
    from "what files are in this folder". A budget that removes the matching tool
    is not a saving, it is a wrong answer arriving faster."""
    from core.tool_router import select_tools_for

    unbudgeted = select_tools_for(text, registry)
    mode = performance.for_turn(intent.read(text), measure=False)
    budgeted = select_tools_for(text, registry, budget=mode.tool_budget)

    if needed in unbudgeted.tool_names:
        assert needed in budgeted.tool_names, f"the budget dropped {needed}"
    assert len(budgeted.tool_names) <= len(unbudgeted.tool_names)


def test_simple_requests_are_measurably_lighter(registry):
    from core.tool_router import select_tools_for

    before = after = 0
    for text, _ in NEEDED:
        mode = performance.for_turn(intent.read(text), measure=False)
        before += len(select_tools_for(text, registry).tool_names)
        after += len(select_tools_for(text, registry, budget=mode.tool_budget).tool_names)
    assert after < before, "the adaptive budget saved nothing at all"


def test_a_budget_cannot_make_the_fallback_smaller(registry):
    """An unrecognised request still sends everything: a budget tightens a route
    that was confident, and never one that was not."""
    from core.tool_router import select_tools_for

    routed = select_tools_for("zxqv plorbish thwarp", registry, budget=5)
    assert routed.full_fallback is True
    assert len(routed.tool_names) == len(registry.names())


def test_a_budget_never_falls_below_the_routers_own_floor(registry):
    from core import tool_router

    routed = tool_router.select_tools_for("read the invoice", registry, budget=1)
    assert len(routed.tool_names) >= tool_router.MIN_TOOLS


# --- Proactive: what it may say, and what it may never do ----------------------------

def _waiting_task():
    task = task_manager.create_task("send the report", ["write it", "send it"],
                                    name="Report")
    task_manager._set_status(task["id"], task_manager.WAITING_FOR_USER,
                             blocked_reason="Sending email needs your confirmation.")
    return task


def test_nothing_at_all_when_it_is_switched_off():
    _waiting_task()
    assert proactive.items(current_level=proactive.OFF) == []
    assert proactive.turn_note([{"title": "x", "detail": "y"}], proactive.OFF) == ""


def test_a_task_waiting_on_the_user_is_what_gets_raised_first():
    _waiting_task()
    watch = watches.save_new(watches.create("w", "cpu_above", {"percent": 90}))
    watch["last_triggered_at"] = time.time()
    watches._replace(watch)

    found = proactive.items(current_level=proactive.ACTIVE)
    assert found[0]["kind"] == "approval"
    assert found[0]["needs_approval"] is True
    assert "waiting for your approval" in found[0]["title"]
    assert any(i["kind"] == "watch" for i in found)


def test_a_watch_that_fired_comes_with_the_evidence_it_fired_on():
    watch = watches.save_new(watches.create("TTWO 5%", "cpu_above", {"percent": 90}))
    watch["last_triggered_at"] = time.time()
    watch["history"] = [{"at": "now", "why": "TTWO moved +7.2% today"}]
    watches._replace(watch)

    item = next(i for i in proactive.items(current_level=proactive.NOTIFICATIONS)
                if i["kind"] == "watch")
    assert "7.2%" in item["detail"]
    assert item["needs_approval"] is False


def test_the_same_thing_is_not_raised_twice_in_a_row():
    _waiting_task()
    first = proactive.items(current_level=proactive.ACTIVE)
    assert first
    proactive.mark_raised(first)
    assert proactive.items(current_level=proactive.ACTIVE) == []


def test_the_level_decides_what_is_allowed_to_appear_unprompted():
    _waiting_task()
    scheduler.save_tasks([{"id": "s1", "name": "Weekly report", "enabled": True,
                           "instruction": "write the weekly report",
                           "next_run": time.time() + 300}])

    quiet = proactive.items(current_level=proactive.SUGGESTIONS)
    louder = proactive.items(current_level=proactive.NOTIFICATIONS)
    assert not any(i["kind"] == "schedule" for i in quiet)
    assert any(i["kind"] == "schedule" for i in louder)


def test_at_most_a_handful_is_ever_raised():
    for n in range(12):
        watch = watches.save_new(watches.create(f"w{n}", "cpu_above", {"percent": 90}))
        watch["last_triggered_at"] = time.time()
        watches._replace(watch)
    assert len(proactive.items(current_level=proactive.ACTIVE)) <= proactive.MAX_ITEMS


def test_an_offer_is_an_offer_even_at_the_most_forward_level():
    _waiting_task()
    note = proactive.turn_note(proactive.items(current_level=proactive.ACTIVE),
                               proactive.ACTIVE)
    assert "offer" in note.lower()
    assert "do not start" in note.lower()
    assert "still needs them to say yes" in note.lower()


def test_being_proactive_cannot_run_anything():
    """The hard requirement. This module has no tool, calls nothing, authorises
    nothing, and the most consequential thing it produces is a sentence."""
    import ast

    tree = ast.parse(open("core/proactive.py").read())
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    for forbidden in ("authorize", "set_unattended", "run", "execute_tool",
                      "start_in_background", "handle_user_input", "approve"):
        assert forbidden not in called, f"core/proactive.py calls {forbidden}()"
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)]
    source = open("core/proactive.py").read()
    assert "Thread" not in source and "create_task" not in source


def test_a_store_that_cannot_be_read_produces_nothing_rather_than_an_error(monkeypatch):
    def explode():
        raise OSError("disk gone")

    monkeypatch.setattr(task_manager, "load_tasks", explode)
    assert proactive.items(current_level=proactive.ACTIVE) == []


def test_leti_does_not_offer_to_read_a_calendar_it_cannot_read():
    """The brief's own example - "you have a meeting in 30 minutes" - needs a
    calendar READ, and Leti has no tool for that. It says so rather than
    inventing the capability."""
    source = open("core/proactive.py").read()
    assert "READ a calendar" in source
    kinds = {i["kind"] for i in proactive.items(current_level=proactive.ACTIVE)}
    assert "meeting" not in kinds
