"""How a tool call is read off an Ollama response before it is executed.

Ollama hands back {"function": {"name": ..., "arguments": {...}}}, and that is
what the orchestrator's execution path consumes. Templates vary in how they spell
the SAME call, though: arguments sometimes arrive as a JSON string rather than an
object, and a call sometimes lands in the message text because the template failed
to lift it out. Both used to end badly - the first raised out of the turn, the
second was returned to the user as a finished answer without anything having run.

Normalising the spelling is all this layer does. The tests that matter most here
are the ones asserting what it does NOT do: it must never turn an assistant
talking about a tool into an assistant calling one.
"""
from __future__ import annotations

import sys
import types

import pytest

# Same stub as test_orchestrator_authorization: core.orchestrator reaches
# chromadb through memory.vector_store at import time.
sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core.orchestrator import (  # noqa: E402
    Orchestrator,
    _normalize_arguments,
    _recover_tool_calls_from_text,
)
from tools.base import BaseTool, ToolParameter, ToolRegistry, ToolResult  # noqa: E402


class RecordingTool(BaseTool):
    """Records exactly what arguments reached the tool body, and how often."""

    name = "get_weather"
    description = "test double"
    parameters = [ToolParameter(name="city", type="string", description="city")]

    def __init__(self):
        self.calls = []

    async def run(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(success=True, output="sunny")


class SecondTool(RecordingTool):
    name = "get_time"
    parameters = [ToolParameter(name="zone", type="string", description="zone")]


def _orchestrator(guard_factory):
    guard, prompts = guard_factory(confirm=True)
    registry = ToolRegistry()
    weather, clock = RecordingTool(), SecondTool()
    registry.register(weather)
    registry.register(clock)

    orch = Orchestrator.__new__(Orchestrator)
    orch.tool_registry = registry
    orch.safety_guard = guard
    orch._preapproved_this_turn = False
    return orch, weather, clock


# --- 1 & 2. The existing path is untouched ----------------------------------------

@pytest.mark.asyncio
async def test_a_normal_ollama_tool_call_still_works(guard_factory):
    """The shape Ollama produces when nothing is unusual. This is the path that
    must not change; everything else here is about spellings around it."""
    orch, weather, _ = _orchestrator(guard_factory)

    result = await orch._execute_tool_call(
        {"function": {"name": "get_weather", "arguments": {"city": "Athens"}}}
    )

    assert result.success
    assert weather.calls == [{"city": "Athens"}]


@pytest.mark.asyncio
async def test_a_call_with_no_arguments_still_runs(guard_factory):
    """Absent, null and empty arguments all meant "no arguments" before, and a
    tool that takes none is a legitimate call - not a malformed one."""
    orch, weather, _ = _orchestrator(guard_factory)

    for raw in ({}, None, ""):
        call = {"function": {"name": "get_weather", "arguments": raw}}
        assert (await orch._execute_tool_call(call)).success, raw
    assert weather.calls == [{}, {}, {}]


def test_a_mapping_is_passed_through_as_the_same_object():
    args = {"city": "Athens"}
    assert _normalize_arguments(args) is args


# --- 3. Arguments as a JSON string -------------------------------------------------

@pytest.mark.asyncio
async def test_json_string_arguments_reach_the_tool_as_a_mapping(guard_factory):
    """The same call, spelled differently. It used to raise AttributeError inside
    the safety guard and take the whole turn down with it."""
    orch, weather, _ = _orchestrator(guard_factory)

    result = await orch._execute_tool_call(
        {"function": {"name": "get_weather", "arguments": '{"city": "Athens"}'}}
    )

    assert result.success
    assert weather.calls == [{"city": "Athens"}], "the tool did not receive a mapping"


def test_json_string_arguments_normalise_without_touching_the_tool():
    assert _normalize_arguments('{"city": "Athens"}') == {"city": "Athens"}


# --- 4. Malformed arguments --------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ['{"city": ', "not json at all", "[1, 2, 3]", 42, ["a"]])
async def test_unreadable_arguments_fail_the_call_instead_of_the_turn(guard_factory, bad):
    """Never an uncaught exception, and never a guess: running the tool anyway
    would mean inventing what the user asked for."""
    orch, weather, _ = _orchestrator(guard_factory)

    result = await orch._execute_tool_call(
        {"function": {"name": "get_weather", "arguments": bad}}
    )

    assert result.success is False
    assert "Malformed arguments" in (result.error or "")
    assert weather.calls == [], "the tool ran with arguments that could not be read"


@pytest.mark.asyncio
@pytest.mark.parametrize("call", [{}, {"function": None}, {"function": "get_weather"},
                                  {"function": {"name": ["get_weather"]}}, "not a call"])
async def test_a_structurally_broken_call_never_raises(guard_factory, call):
    orch, weather, _ = _orchestrator(guard_factory)

    result = await orch._execute_tool_call(call)

    assert result.success is False
    assert weather.calls == []


# --- 5. Unknown tools --------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_unknown_tool_is_not_executed(guard_factory):
    orch, weather, clock = _orchestrator(guard_factory)

    result = await orch._execute_tool_call(
        {"function": {"name": "rm_rf_everything", "arguments": {}}}
    )

    assert result.success is False
    assert "Unknown tool" in (result.error or "")
    assert weather.calls == [] and clock.calls == []


# --- 6. Calls recovered from message text ------------------------------------------

def _find(name):
    return object() if name in {"get_weather", "get_time"} else None


def test_a_tagged_call_left_in_the_text_is_recovered():
    """The <tool_call> delimiters are the evidence of intent; the name still has to
    match a registered tool and the arguments still have to parse."""
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Athens"}}\n</tool_call>'

    assert _recover_tool_calls_from_text(text, _find) == [
        {"function": {"name": "get_weather", "arguments": {"city": "Athens"}}}
    ]


def test_a_message_that_is_entirely_one_call_object_is_recovered():
    text = '  {"name": "get_weather", "arguments": {"city": "Athens"}}  '

    assert _recover_tool_calls_from_text(text, _find) == [
        {"function": {"name": "get_weather", "arguments": {"city": "Athens"}}}
    ]


def test_recovered_calls_are_normalised_the_same_way():
    """A recovered call whose arguments are themselves a JSON string still arrives
    as a mapping - one normalisation, not two spellings of it."""
    text = '<tool_call>{"name": "get_weather", "arguments": "{\\"city\\": \\"Athens\\"}"}</tool_call>'

    assert _recover_tool_calls_from_text(text, _find) == [
        {"function": {"name": "get_weather", "arguments": {"city": "Athens"}}}
    ]


def test_several_tagged_calls_are_all_recovered():
    text = ('<tool_call>{"name": "get_weather", "arguments": {"city": "Athens"}}</tool_call>'
            '<tool_call>{"name": "get_time", "arguments": {"zone": "EET"}}</tool_call>')

    assert [c["function"]["name"] for c in _recover_tool_calls_from_text(text, _find)] == [
        "get_weather", "get_time"
    ]


# --- 7. Prose is not a tool call ---------------------------------------------------

@pytest.mark.parametrize("text", [
    "I can use the weather tool to find that information.",
    "get_weather is the tool for that.",
    "Call get_weather with {\"city\": \"Athens\"} and you'll see.",
    'Here is an example: {"name": "get_weather", "arguments": {"city": "Athens"}} - shall I?',
    "The weather in Athens is sunny.",
    '{"name": "get_weather"}',
    '{"name": "not_a_registered_tool", "arguments": {"city": "Athens"}}',
    '{"arguments": {"city": "Athens"}}',
    '{"name": "get_weather", "arguments": "not json"}',
    '<tool_call>not json at all</tool_call>',
    "",
    None,
])
def test_text_that_is_not_a_call_is_left_alone(text):
    """The failure that matters. An assistant mentioning a tool, quoting one, or
    offering to use one is talking - and must keep being treated as talking."""
    assert _recover_tool_calls_from_text(text, _find) == []


@pytest.mark.asyncio
async def test_ordinary_prose_is_still_returned_as_the_final_answer(guard_factory):
    """End to end through the loop: no tool_calls and nothing recoverable means the
    message is the answer, exactly as before."""
    orch, weather, _ = _orchestrator(guard_factory)
    said = "I can use the weather tool to find that information."

    orch.llm_client = _ScriptedClient([{"message": {"content": said}}])
    answer = await _run_loop(orch)

    assert answer == said
    assert weather.calls == []


@pytest.mark.asyncio
async def test_a_call_stranded_in_the_text_is_run_rather_than_reported_as_an_answer(guard_factory):
    """Without recovery the raw <tool_call> blob was handed back as a finished
    reply: the user is told their request is done and nothing ever ran."""
    orch, weather, _ = _orchestrator(guard_factory)
    blob = '<tool_call>{"name": "get_weather", "arguments": {"city": "Athens"}}</tool_call>'

    orch.llm_client = _ScriptedClient([
        {"message": {"content": blob}},
        {"message": {"content": "It is sunny in Athens."}},
    ])
    answer = await _run_loop(orch)

    assert weather.calls == [{"city": "Athens"}], "the stranded call never ran"
    assert answer == "It is sunny in Athens."
    assert blob not in answer


# --- 8. Several calls in one response ----------------------------------------------

@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_response_all_run(guard_factory):
    """Including one of each spelling in the same response."""
    orch, weather, clock = _orchestrator(guard_factory)

    orch.llm_client = _ScriptedClient([
        {"message": {"tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Athens"}}},
            {"function": {"name": "get_time", "arguments": '{"zone": "EET"}'}},
        ]}},
        {"message": {"content": "done"}},
    ])
    answer = await _run_loop(orch)

    assert weather.calls == [{"city": "Athens"}]
    assert clock.calls == [{"zone": "EET"}]
    assert answer == "done"


# --- helpers -----------------------------------------------------------------------

class _ScriptedClient:
    """Returns canned Ollama responses in order, recording what it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    async def chat(self, messages, tools=None, **kwargs):
        self.sent.append((messages, tools))
        return self._responses.pop(0)


async def _run_loop(orch):
    """Drive _tool_calling_loop with the collaborators it actually touches.

    Orchestrator.settings is already a property over the real config, so the loop
    reads max_tool_iterations from there rather than from anything set up here.
    """
    orch.visual_callback = None
    orch._set_state = lambda *_a, **_k: None
    return await orch._tool_calling_loop([{"role": "user", "content": "hi"}])
