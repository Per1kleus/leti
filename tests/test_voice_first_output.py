"""Leti speaks. Leti shows text when asked, and not before.

The behaviour these protect is one sentence long and easy to break in six
places: the answer is spoken, the words are kept, and "show me the text" hands
back the SAME words without asking the model anything. Every test here is some
form of that - the answer survives, the panel does not open by itself, showing
costs no model call, and hiding stops nothing.
"""
from __future__ import annotations

import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import intent, transcript  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from tools.base import BaseTool, ToolParameter, ToolRegistry, ToolResult  # noqa: E402


class FakeLLM:
    """Records every call, so a test can prove one did not happen."""

    def __init__(self, script=()):
        self.script = list(script)
        self.calls = 0

    async def chat(self, messages, tools=None):
        self.calls += 1
        return {"message": self.script.pop(0) if self.script
                else {"content": "done", "tool_calls": None}}


class Memory:
    def __init__(self):
        self.turns = []
        self._referents = []

    def add_turn(self, role, content, session_id="default"):
        self.turns.append({"role": role, "content": content})

    def get_recent_messages(self):
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def recent_referents(self):
        return list(self._referents)

    def note_referents(self, items):
        self._referents = list(items)


class Chart(BaseTool):
    name = "visualize_dataset"
    description = "draw a chart"
    parameters = [ToolParameter(name="path", type="string", description="data")]

    async def run(self, **kwargs):
        return ToolResult(success=True, output={
            "success": True, "message": "Charted.",
            "visual": {"type": "images", "title": "Measurements",
                       "images": [{"url": "file:///tmp/chart.png"}]}})


def _orchestrator(guard_factory, script=()):
    import asyncio

    from core import entities

    guard, prompts = guard_factory(confirm=True, confirm_classes=[])
    registry = ToolRegistry()
    registry.register(Chart())
    llm = FakeLLM(script)

    orch = Orchestrator.__new__(Orchestrator)
    orch.llm_client = llm
    orch.tool_registry = registry
    orch.safety_guard = guard
    orch.session_memory = Memory()
    orch.vector_memory = None
    orch.speak_callback = None
    orch.visual_callback = None
    orch.state = None
    orch._state_listeners = []
    orch._preapproved_this_turn = False
    orch._checked_watches_this_session = True
    orch._turn_lock = asyncio.Lock()
    entities.use_conversation_source(orch.session_memory.get_recent_messages)
    return orch, llm


class Surface:
    """Everything that reached a screen or a speaker this turn."""

    def __init__(self):
        self.spoken = []
        self.shown = []

    def attach(self, orch):
        async def speak(text):
            self.spoken.append(text)

        async def show(visual):
            self.shown.append(visual)

        orch.speak_callback = speak
        orch.visual_callback = show
        return self

    @property
    def transcripts(self):
        return [v for v in self.shown if v.get("type") == "transcript"]

    @property
    def visible_text(self):
        return [v for v in self.transcripts if v.get("visible")]


@pytest.fixture(autouse=True)
def _fresh():
    transcript.clear()
    yield
    transcript.clear()


# --- The answer is spoken, and kept ------------------------------------------------------

@pytest.mark.asyncio
async def test_an_answer_is_spoken_and_held(guard_factory):
    orch, _ = _orchestrator(guard_factory, [{"content": "It is twelve degrees."}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("what is the weather?")

    assert surface.spoken == ["It is twelve degrees."]
    assert transcript.current().text == "It is twelve degrees."


@pytest.mark.asyncio
async def test_an_answer_does_not_show_itself(guard_factory):
    orch, _ = _orchestrator(guard_factory, [{"content": "It is twelve degrees."}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("what is the weather?")

    assert surface.transcripts == [], "the answer put itself on screen"
    assert transcript.is_visible() is False


@pytest.mark.asyncio
async def test_the_conversation_is_still_remembered(guard_factory):
    """Hidden is a fact about a panel. The conversation is untouched."""
    orch, _ = _orchestrator(guard_factory, [{"content": "It is twelve degrees."}])
    Surface().attach(orch)
    await orch.handle_user_input("what is the weather?")

    assert orch.session_memory.turns[-1] == {"role": "assistant",
                                             "content": "It is twelve degrees."}


# --- Asking to see it --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_asking_for_the_text_shows_what_was_said(guard_factory):
    orch, llm = _orchestrator(guard_factory, [{"content": "The acceleration is F over m."}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("why does it speed up?")
    calls_after_answering = llm.calls

    await orch.handle_user_input("show me the text")

    assert surface.visible_text, "asking for the text showed nothing"
    assert surface.visible_text[-1]["text"] == "The acceleration is F over m."
    assert llm.calls == calls_after_answering, "showing the text asked the model again"


@pytest.mark.asyncio
async def test_the_text_shown_is_the_text_that_was_spoken(guard_factory):
    """Not a paraphrase, not a regeneration - the same string."""
    answer = "Force equals mass times acceleration, which is why it speeds up."
    orch, _ = _orchestrator(guard_factory, [{"content": answer}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("why?")
    await orch.handle_user_input("show the answer")

    assert surface.visible_text[-1]["text"] == answer
    assert surface.spoken[0] == answer


@pytest.mark.parametrize("asking", [
    "show me the text", "show the answer", "display the response",
    "let me read that", "show me what you said", "open the transcript",
    "display the explanation", "show me everything you said",
])
@pytest.mark.asyncio
async def test_every_way_of_asking_works(guard_factory, asking):
    orch, llm = _orchestrator(guard_factory, [{"content": "Here is the answer."}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("a question")
    before = llm.calls
    await orch.handle_user_input(asking)

    assert surface.visible_text, f"{asking!r} did not show the text"
    assert llm.calls == before, f"{asking!r} went to the model"


@pytest.mark.asyncio
async def test_asking_before_there_is_anything_falls_through(guard_factory):
    """Nothing to show is not an error - it is an ordinary turn, because the
    user may have meant something this could not see."""
    orch, llm = _orchestrator(guard_factory, [{"content": "There is nothing yet."}])
    Surface().attach(orch)
    await orch.handle_user_input("show me the text")
    assert llm.calls == 1, "a request with nothing to show should be answered normally"


# --- Hiding it again ----------------------------------------------------------------------

@pytest.mark.parametrize("asking", ["hide the text", "close the transcript", "hide the answer"])
@pytest.mark.asyncio
async def test_hiding_works(guard_factory, asking):
    orch, llm = _orchestrator(guard_factory, [{"content": "Here is the answer."}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("a question")
    await orch.handle_user_input("show me the text")
    before = llm.calls

    await orch.handle_user_input(asking)

    assert transcript.is_visible() is False
    assert surface.transcripts[-1]["visible"] is False
    assert llm.calls == before, "hiding went to the model"


@pytest.mark.asyncio
async def test_hiding_keeps_the_answer(guard_factory):
    orch, _ = _orchestrator(guard_factory, [{"content": "Something worth keeping."}])
    Surface().attach(orch)
    await orch.handle_user_input("a question")
    await orch.handle_user_input("show me the text")
    await orch.handle_user_input("hide the text")

    assert transcript.current().text == "Something worth keeping."
    shown = transcript.reveal()
    assert shown["text"] == "Something worth keeping."


def test_hiding_touches_nothing_but_the_panel():
    """Stated as a guarantee in core/transcript.py, checked as one here."""
    transcript.begin()
    transcript.said("An answer.")
    transcript.mark_spoken()
    transcript.add_visual({"type": "images", "images": []})
    before = transcript.current()

    transcript.reveal()
    transcript.hide()

    after = transcript.current()
    assert after is before, "hiding replaced the response"
    assert after.text == "An answer."
    assert after.spoken is True, "hiding un-said what was said"
    assert len(after.visuals) == 1, "hiding threw away a visual"


@pytest.mark.asyncio
async def test_while_the_panel_is_open_the_next_answer_goes_into_it(guard_factory):
    """They asked to read Leti and have not asked to stop."""
    orch, _ = _orchestrator(guard_factory, [{"content": "First."}, {"content": "Second."}])
    Surface().attach(orch)
    await orch.handle_user_input("one")
    await orch.handle_user_input("show me the text")
    assert transcript.is_visible() is True

    await orch.handle_user_input("two")
    # A new turn replaces the response; the panel's own state is the surface's,
    # and gui/api.py reads is_visible() to decide whether to push the reply.
    assert transcript.current().text == "Second."


# --- Visuals are independent of the transcript --------------------------------------------

@pytest.mark.asyncio
async def test_a_chart_still_appears_with_the_transcript_hidden(guard_factory):
    orch, _ = _orchestrator(guard_factory, [
        {"content": None, "tool_calls": [
            {"function": {"name": "visualize_dataset", "arguments": {"path": "/tmp/x.csv"}}}]},
        {"content": "Here is the graph."}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("plot these measurements")

    charts = [v for v in surface.shown if v.get("type") == "images"]
    assert charts, "the chart never reached the screen"
    assert charts[0]["offer"] is False, "a chart that was asked for should open itself"
    assert surface.transcripts == [], "the transcript opened alongside it"
    assert surface.spoken == ["Here is the graph."]


@pytest.mark.asyncio
async def test_a_visual_is_remembered_so_it_need_not_be_remade(guard_factory):
    orch, llm = _orchestrator(guard_factory, [
        {"content": None, "tool_calls": [
            {"function": {"name": "visualize_dataset", "arguments": {"path": "/tmp/x.csv"}}}]},
        {"content": "Here is the graph."}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("plot these measurements")
    before = llm.calls

    await orch.handle_user_input("show me the graph")

    again = [v for v in surface.shown if v.get("type") == "images"]
    assert len(again) == 2, "the graph was not shown again"
    assert llm.calls == before, "showing it again asked the model"


@pytest.mark.asyncio
async def test_asking_for_a_graph_that_was_never_made_is_an_ordinary_turn(guard_factory):
    """"Show me the graph" with no graph is a request to MAKE one."""
    orch, llm = _orchestrator(guard_factory, [{"content": "I'll plot that."}])
    Surface().attach(orch)
    await orch.handle_user_input("show me the graph")
    assert llm.calls == 1


def test_a_visual_is_forgotten_once_it_is_stale():
    transcript.begin()
    transcript.said("An answer.")
    transcript.add_visual({"type": "images", "images": []})
    assert transcript.visuals()
    transcript.current().at -= transcript.VISUAL_MAX_AGE_SECONDS + 1
    assert transcript.visuals() == [], "a very old visual is not 'that graph'"


# --- Mathematics on request ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_asking_for_the_equation_renders_what_was_said(guard_factory):
    orch, llm = _orchestrator(guard_factory, [
        {"content": r"The kinetic energy is \(E_k = \frac{1}{2}mv^2\)."}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("what is kinetic energy?")
    before = llm.calls

    await orch.handle_user_input("show me the equation")

    maths = [v for v in surface.shown if v.get("type") == "math"]
    assert maths, "the equation was not rendered"
    assert maths[0]["items"][0]["latex"] == r"E_k = \frac{1}{2}mv^2"
    assert llm.calls == before, "rendering the equation asked the model"


@pytest.mark.asyncio
async def test_simple_conversational_arithmetic_opens_no_panel(guard_factory):
    orch, _ = _orchestrator(guard_factory, [{"content": "The answer is four."}])
    surface = Surface().attach(orch)
    await orch.handle_user_input("what is 2 + 2?")

    assert surface.shown == [], "a panel opened for a spoken number"
    assert surface.spoken == ["The answer is four."]


@pytest.mark.asyncio
async def test_asking_for_an_equation_that_was_never_given_is_an_ordinary_turn(guard_factory):
    orch, llm = _orchestrator(guard_factory, [{"content": "It is about four."}])
    Surface().attach(orch)
    await orch.handle_user_input("roughly how much?")
    await orch.handle_user_input("show me the equation")
    assert llm.calls == 2, "there was no equation, so this was a request to give one"


# --- The command reader ---------------------------------------------------------------------

@pytest.mark.parametrize("ordinary", [
    "show me how to write a for loop",
    "what is the weather",
    "show me the file contents of report.md",
    "can you open the project folder",
    "display mode is broken",
    "what did you mean by that",
    "read the file and summarise it",
])
def test_an_ordinary_request_is_not_a_request_to_show_the_text(ordinary):
    assert intent.transcript_command(ordinary) is None


@pytest.mark.parametrize("text,action", [
    ("show me the text", intent.SHOW_TEXT),
    ("hide the answer", intent.HIDE_TEXT),
    ("show me the equation", intent.SHOW_MATH),
    ("show me the graph", intent.SHOW_LAST_VISUAL),
    ("show that chart again", intent.SHOW_LAST_VISUAL),
])
def test_the_right_thing_is_asked_for(text, action):
    assert intent.transcript_command(text) == action


def test_a_long_sentence_is_never_only_about_showing():
    long_one = ("show me the text of the report you wrote yesterday about the "
                "quarterly figures and then email it to chris")
    assert intent.transcript_command(long_one) is None


def test_reading_a_request_costs_nothing():
    import time

    intent.transcript_command("show me the text")
    started = time.perf_counter()
    for _ in range(2000):
        intent.transcript_command("what is the weather today please")
    each = (time.perf_counter() - started) / 2000
    assert each < 0.0005, f"{each * 1e6:.0f} us in front of every message"


# --- Bounds ------------------------------------------------------------------------------------

def test_a_very_long_answer_is_bounded():
    transcript.begin()
    transcript.said("x" * (transcript.MAX_TEXT_CHARS + 5000))
    assert len(transcript.current().text) == transcript.MAX_TEXT_CHARS


def test_only_a_handful_of_visuals_are_remembered():
    transcript.begin()
    transcript.said("An answer.")
    for i in range(20):
        transcript.add_visual({"type": "images", "n": i})
    assert len(transcript.visuals()) <= transcript.MAX_VISUALS


def test_the_diagnostics_section_says_what_is_held_not_what_it_says():
    transcript.begin()
    transcript.said("A private answer about something personal.")
    section = transcript.section()
    assert section["holding"] is True and section["characters"] > 0
    import json

    assert "private" not in json.dumps(section).lower()


def test_the_buffer_keeps_nothing_on_disk():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/transcript.py").read_text())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    called |= {n.func.id for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for forbidden in ("open", "write_text", "atomic_write_text", "store_path", "connect"):
        assert forbidden not in called, f"core/transcript.py calls {forbidden}"
