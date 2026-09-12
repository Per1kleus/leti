"""Which tools a request is shown, and everything that must stay true regardless.

Sending all 103 tool schemas on every call costs roughly 16,000 tokens of a
24,576-token window before the conversation gets any. This picks a subset - and
the tests that matter most are the ones pinning what it must never do: lose a
tool, guess a destructive one, or leave Leti unable to act because the router was
unsure.
"""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

import main  # noqa: E402
from core import tool_router  # noqa: E402
from core.tool_router import CATEGORIES, Routing, last_user_message, route, select_tools_for  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    sys.argv = ["main.py"]
    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


# --- Nothing is lost -------------------------------------------------------------

def test_every_tool_is_still_registered_and_reachable(registry):
    """The router narrows what is SHOWN. Nothing leaves the registry."""
    names = registry.names()
    # A count, so that a tool quietly disappearing is a failing test rather than a
    # capability nobody notices is gone. It went 103 -> 111 when the autonomous
    # task and workflow tools were added; it must never go DOWN.
    assert len(names) >= 111, f"the registry lost tools: {len(names)}"
    for name in names:
        assert registry.get(name) is not None
        assert callable(getattr(registry.get(name), "run", None))


def test_every_tool_can_still_be_selected_by_some_request(registry):
    """A tool no request can reach is a tool that has been quietly disabled. Asking
    for a tool by its own name is the floor: if that cannot surface it, nothing can."""
    unreachable = []
    for name in registry.names():
        words = name.replace("_", " ")
        if name not in set(route(f"please {words}", registry).tool_names):
            unreachable.append(name)
    assert not unreachable, f"no request could reach these tools: {unreachable}"


def test_the_category_table_still_covers_the_whole_registry(registry):
    """A module missing from the table is always exposed, so drift is safe rather
    than silent - but it should still be noticed, which is what this is for."""
    known = {m for modules in CATEGORIES.values() for m in modules}
    actual = {type(registry.get(n)).__module__ for n in registry.names()}
    missing = sorted(actual - known)
    assert not missing, f"modules not in any category (they are always exposed): {missing}"


def test_schemas_for_returns_real_unmodified_schemas(registry):
    """Tool names and arguments must be byte-identical to what was sent before."""
    subset = sorted(registry.names())[:20]
    picked = registry.schemas_for(subset)
    full = {s["function"]["name"]: s for s in registry.all_schemas()}

    assert [s["function"]["name"] for s in picked] == subset
    for schema in picked:
        assert schema == full[schema["function"]["name"]], "a schema was altered"


def test_schemas_for_ignores_names_that_do_not_exist(registry):
    assert registry.schemas_for(["read_file", "not_a_tool"]) == registry.schemas_for(["read_file"])


def test_schemas_for_is_stable_so_the_prompt_prefix_can_be_cached(registry):
    a = registry.schemas_for(["write_file", "read_file", "list_files"])
    b = registry.schemas_for(["list_files", "read_file", "write_file"])
    assert json.dumps(a) == json.dumps(b)


# --- The right tools show up -----------------------------------------------------

@pytest.mark.parametrize("request_text,expected", [
    ("read the file config/settings.yaml", {"read_file"}),
    ("delete the old backup folder", {"delete_file"}),
    ("what's the weather in Athens", {"get_weather"}),
    ("open spotify", {"launch_app"}),
    ("what's on my screen right now", {"read_screen"}),
    ("run the tests for this project", {"run_tests"}),
    ("check if my firewall is on", {"check_firewall_status"}),
    ("add milk to my todo list", {"add_todo_item"}),
    ("search the web for local news", {"web_search"}),
    ("remember that I prefer tea", {"remember_about_user"}),
    ("what's the price of AAPL", {"get_market_quote"}),
    ("convert 5 inches to cm", {"convert_units"}),
    ("kill the process using port 8080", {"kill_process"}),
    ("make a chart of sales.csv", {"visualize_dataset"}),
])
def test_the_tool_the_request_needs_is_exposed(registry, request_text, expected):
    assert expected <= set(route(request_text, registry).tool_names)


def test_a_request_needing_several_capabilities_gets_all_of_them(registry):
    """"Find John's email, check tomorrow's calendar, and send him a message" is one
    request and three capabilities. Narrowing to one of them fails it."""
    routing = route("Find John's email, check tomorrow's calendar, and send him a message",
                    registry)
    names = set(routing.tool_names)

    assert {"resolve_contact", "send_email", "schedule_meeting"} <= names
    assert not routing.full_fallback, "this should have been routed, not fallen back"


def test_a_small_relevant_group_beats_guessing_one_tool(registry):
    """Contacts, mail and the calendar travel together; exposing the group is the
    conservative answer, not the lazy one."""
    names = set(route("Send an email to John about tomorrow's meeting", registry).tool_names)

    assert {"send_email", "resolve_contact"} <= names


def test_irrelevant_capabilities_are_left_out(registry):
    """The saving is real only if unrelated tools actually go."""
    names = set(route("what's the weather in Athens", registry).tool_names)

    assert "get_weather" in names
    for unrelated in ("place_paper_order", "send_email", "run_shell_command",
                      "delete_file", "kill_process", "restore_from_snapshot"):
        assert unrelated not in names, f"{unrelated} was exposed to a weather question"


DESTRUCTIVE = ("delete_file", "run_shell_command", "kill_process",
               "restore_from_snapshot", "apply_system_updates", "place_paper_order")


@pytest.mark.parametrize("request_text,expected_adjacent", [
    ("what's the weather in Athens", set()),
    ("add milk to my todo list", set()),
    ("convert 5 inches to cm", set()),
    # A question about a share price routes to the trading module, and
    # place_paper_order lives in it. That is adjacency, not a wrong guess: the unit
    # of exposure is a whole capability, and showing a tool is not running one -
    # SafetyGuard still gates the call exactly as it did before.
    ("what's the price of AAPL", {"place_paper_order"}),
])
def test_a_confident_route_does_not_drag_in_unrelated_destructive_tools(
        registry, request_text, expected_adjacent):
    """Preferring no tool over the wrong tool matters most here.

    This is about a CONFIDENT route. A request the router recognises nothing in is
    a different case, where the safe answer is the whole registry - see
    test_a_request_that_matches_nothing_falls_back_to_every_tool.
    """
    routing = route(request_text, registry)
    assert routing.confident and not routing.full_fallback, request_text

    exposed = set(routing.tool_names) & set(DESTRUCTIVE)
    assert exposed == expected_adjacent, (
        f"{request_text!r} exposed {sorted(exposed - expected_adjacent)}")


def test_most_requests_expose_far_fewer_tools(registry):
    """The whole point, asserted rather than assumed."""
    requests = ["what's the weather in Athens", "open spotify", "read the file notes.txt",
                "add milk to my todo list", "convert 5 inches to cm",
                "check if my firewall is on", "what's the price of AAPL"]
    counts = [route(r, registry).count for r in requests]
    total = len(registry.names())

    assert max(counts) < total * 0.65, f"barely narrowed anything: {counts}"
    assert sum(counts) / len(counts) < total * 0.4, f"mean too broad: {counts}"


# --- Uncertainty and failure -----------------------------------------------------

def test_a_request_that_matches_nothing_falls_back_to_every_tool(registry):
    routing = route("asdkjfh qwptu zzz", registry)

    assert routing.full_fallback is True
    assert set(routing.tool_names) == set(registry.names())
    assert routing.confident is False


@pytest.mark.parametrize("empty", ["", "   ", None, 12345])
def test_an_unusable_request_falls_back_rather_than_sending_nothing(registry, empty):
    routing = route(empty, registry)

    assert routing.full_fallback is True
    assert set(routing.tool_names) == set(registry.names())


def test_a_router_that_throws_falls_back_to_every_tool(registry, monkeypatch):
    """The safety net. However this file breaks, Leti keeps every tool it had."""
    def explode(*_a, **_k):
        raise RuntimeError("router is broken")

    monkeypatch.setattr(tool_router, "_route", explode)
    routing = route("send an email", registry)

    assert routing.full_fallback is True
    assert set(routing.tool_names) == set(registry.names())
    assert "routing failed" in routing.reason


def test_routing_can_be_switched_off_entirely(registry):
    """Point 14: the previous behaviour has to stay recoverable."""
    routing = select_tools_for("what's the weather", registry, enabled=False)

    assert set(routing.tool_names) == set(registry.names())
    assert routing.full_fallback is True


def test_an_unknown_module_is_always_exposed(registry, monkeypatch):
    """A tool added in a module nobody told the router about must not vanish."""
    monkeypatch.setitem(CATEGORIES, "files", [])       # orphan the file tools
    tool_router._CACHE.clear()
    try:
        names = set(route("what's the weather in Athens", registry).tool_names)
        assert "read_file" in names, "an uncategorised module was hidden"
    finally:
        tool_router._CACHE.clear()


# --- Conversations that need no tools ---------------------------------------------

@pytest.mark.parametrize("greeting", ["hi", "hello", "hey Leti", "thanks!", "thank you",
                                      "ok", "cool", "good morning", "bye", "how are you"])
def test_a_greeting_gets_no_tool_schemas_at_all(registry, greeting):
    routing = route(greeting, registry)

    assert routing.no_tools is True
    assert routing.tool_names == []


@pytest.mark.parametrize("not_a_greeting", [
    "hi, now open spotify",
    "thanks, can you delete that file",
    "ok send the email",
    "hello world program in python",
])
def test_a_greeting_with_a_request_in_it_is_still_a_request(registry, not_a_greeting):
    """The dangerous half of the no-tools path: anything more than a greeting has to
    go down the ordinary route, or Leti simply will not act on it."""
    routing = route(not_a_greeting, registry)

    assert routing.no_tools is False
    assert routing.tool_names, "a real request was sent with no tools"


# --- Cost -------------------------------------------------------------------------

def test_routing_is_far_cheaper_than_what_it_saves(registry):
    """A router that costs more time than the context it saves is not worth having."""
    import time

    route("warm the index", registry)
    t0 = time.perf_counter()
    for _ in range(500):
        route("send an email to john about the meeting", registry)
    per_ms = (time.perf_counter() - t0) / 500 * 1000

    assert per_ms < 5.0, f"routing took {per_ms:.2f} ms per request"


def test_the_index_is_built_once_not_per_request(registry):
    built = []
    original = tool_router._Index

    class Counting(original):
        def __init__(self, reg):
            built.append(1)
            super().__init__(reg)

    tool_router._CACHE.clear()
    tool_router._Index = Counting
    try:
        for _ in range(25):
            route("read a file", registry)
    finally:
        tool_router._Index = original
        tool_router._CACHE.clear()

    assert built == [1], f"the index was rebuilt {len(built)} times"


# --- The orchestrator's use of it --------------------------------------------------

def test_the_loop_routes_once_per_turn_not_once_per_iteration():
    """Re-routing inside the loop would change the tool list underneath a
    conversation that is still going, throwing away the server's cached prefix on
    every pass - and could withdraw a tool the model was about to call."""
    import inspect

    from core.orchestrator import Orchestrator

    source = inspect.getsource(Orchestrator._tool_calling_loop)
    body = source[source.index("async def"):]
    assert body.count("select_tools_for") == 1
    assert body.index("select_tools_for") < body.index("for iteration in range")


def test_the_loop_still_hands_the_same_schemas_to_the_model():
    import inspect

    from core.orchestrator import Orchestrator

    source = inspect.getsource(Orchestrator._tool_calling_loop)
    assert "schemas_for(routing.tool_names)" in source
    assert "tools=tool_schemas" in source


def test_last_user_message_reads_the_most_recent_user_turn():
    messages = [
        {"role": "system", "content": "you are leti"},
        {"role": "user", "content": "first thing"},
        {"role": "assistant", "content": "sure"},
        {"role": "user", "content": "the actual request"},
    ]
    assert last_user_message(messages) == "the actual request"
    assert last_user_message([]) == ""
    assert last_user_message([{"role": "system", "content": "x"}]) == ""


def test_routing_never_touches_authorization(registry):
    """It answers which tools to show, never whether a call is allowed.

    Checked structurally rather than by grepping the file: this module's own prose
    explains that authorization stays with SafetyGuard, so a text search would only
    ever find its own explanation. What matters is that it neither imports the
    guard nor calls it.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(tool_router))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("safety" in m for m in imported), f"the router imports {imported}"

    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    for forbidden in ("authorize", "audit_result"):
        assert forbidden not in called, f"the router calls {forbidden}()"


def test_a_routing_result_is_only_ever_names_the_registry_knows(registry):
    for text in ["send an email", "delete a file", "hello", "asdkjfh", ""]:
        for name in route(text, registry).tool_names:
            assert registry.get(name) is not None, f"routing invented '{name}'"

# --- The document tools are reachable ------------------------------------------------
# Routing exists to hide tools, so a capability that routing can never expose is a
# capability that does not exist. These are the phrasings the file work was for.

DOCUMENT_TOOLS = {"find_documents", "inspect_document", "read_document",
                  "compare_documents", "look_at_image"}


@pytest.mark.parametrize("request_text", [
    "read these PDFs and compare the offers",
    "find the important information in these documents",
    "compare these excel files and tell me which option is better",
    "summarise this folder",
    "create a report based on these files",
    "what do these contracts say about notice periods",
    "read the invoice and tell me the total",
])
def test_a_request_about_files_reaches_the_document_tools(registry, request_text):
    routing = route(request_text, registry)
    exposed = set(routing.tool_names) & DOCUMENT_TOOLS
    assert exposed, f"{request_text!r} exposed none of the document tools"


@pytest.mark.parametrize("request_text", [
    "what's the weather in Athens",
    "convert 5 inches to cm",
    "add milk to my todo list",
])
def test_a_request_about_nothing_of_the_kind_does_not(registry, request_text):
    routing = route(request_text, registry)
    assert not (set(routing.tool_names) & DOCUMENT_TOOLS), request_text


@pytest.mark.parametrize("request_text", [
    "open the application and click through the settings dialog",
    "click the export button in this program",
])
def test_a_desktop_errand_reaches_the_computer_use_tools(registry, request_text):
    routing = route(request_text, registry)
    exposed = set(routing.tool_names) & {"choose_computer_approach", "plan_computer_task",
                                         "computer_step_done", "verify_screen"}
    assert exposed, f"{request_text!r} exposed none of the computer-use tools"

