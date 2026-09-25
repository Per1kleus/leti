"""Speaking before the model has finished, and stopping without waiting for it.

Two behaviours, both about time. Leti should say the first sentence while the
model is still writing the third - so a long answer starts immediately rather
than after a pause the length of the whole answer. And "stop" should land
immediately, which it could not before: the turn that was speaking held the turn
lock, so the message asking it to stop queued behind the speech it was stopping.

The race in the middle is the part worth testing hardest. A sentence that was
ready when STOP arrived must not start playing afterwards, and a fragment that
arrives after STOP must be dropped rather than spoken.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import speech, transcript  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from tools.base import BaseTool, ToolParameter, ToolRegistry, ToolResult  # noqa: E402


class StreamingLLM:
    """A model that writes in fragments, like Ollama does.

    `script` is a list of turns; each turn is a list of fragments, or a dict for
    a turn that calls a tool. Records how far it got, so a test can prove the
    generation was actually cancelled rather than merely ignored.
    """

    def __init__(self, script):
        self.script = [list(turn) if isinstance(turn, (list, tuple)) else turn
                       for turn in script]
        self.calls = 0
        self.fragments_sent = 0
        self.finished_streams = 0
        self.on_fragment = None          # a test hook, run after each fragment

    async def chat(self, messages, tools=None):
        raise AssertionError("the streaming path should not fall back to chat()")

    async def stream_response(self, messages, tools=None, model=None, should_stop=None):
        self.calls += 1
        turn = self.script.pop(0) if self.script else ["Done."]
        if isinstance(turn, dict):
            yield {"done": True, "stopped": False, "error": "", "message": turn}
            self.finished_streams += 1
            return
        content = []
        for fragment in turn:
            if should_stop is not None and should_stop():
                yield {"done": True, "stopped": True, "error": "",
                       "message": {"role": "assistant", "content": "".join(content)}}
                return
            content.append(fragment)
            self.fragments_sent += 1
            yield {"text": fragment}
            if self.on_fragment is not None:
                await self.on_fragment(self)
            await asyncio.sleep(0)       # let the loop breathe, as a socket would
        yield {"done": True, "stopped": False, "error": "",
               "message": {"role": "assistant", "content": "".join(content)}}
        self.finished_streams += 1


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


class Voice:
    """Everything that reached a speaker or a screen, and when."""

    def __init__(self):
        self.said = []
        self.shown = []
        self.interrupted = 0
        self.before_saying = None        # a hook, run before each utterance

    def attach(self, orch):
        async def speak(text):
            if self.before_saying is not None:
                await self.before_saying(self)
            # The real GUI callback checks this between utterances; a test double
            # that skipped the check would prove nothing about stopping.
            if transcript.should_stop_speaking():
                return
            self.said.append(text)

        async def show(visual):
            self.shown.append(visual)

        orch.speak_callback = speak
        orch.visual_callback = show
        orch.interrupt_callback = self._interrupt
        return self

    def _interrupt(self):
        self.interrupted += 1


def _orchestrator(guard_factory, script=()):
    from core import entities

    guard, _ = guard_factory(confirm=True, confirm_classes=[])
    registry = ToolRegistry()
    registry.register(Chart())
    llm = StreamingLLM(script)

    orch = Orchestrator.__new__(Orchestrator)
    orch.llm_client = llm
    orch.tool_registry = registry
    orch.safety_guard = guard
    orch.session_memory = Memory()
    orch.vector_memory = None
    orch.speak_callback = None
    orch.visual_callback = None
    orch.interrupt_callback = None
    orch.state = None
    orch._state_listeners = []
    orch._preapproved_this_turn = False
    orch._checked_watches_this_session = True
    orch._turn_lock = asyncio.Lock()
    entities.use_conversation_source(orch.session_memory.get_recent_messages)
    return orch, llm


THREE = ["The first measurement ", "finished cleanly. ",
         "The second one is ", "still running now. ",
         "The third has not started yet."]


@pytest.fixture(autouse=True)
def _fresh():
    transcript.clear()
    yield
    transcript.clear()


# --- Speaking before the model has finished -------------------------------------------------

@pytest.mark.asyncio
async def test_the_first_sentence_is_said_before_the_last_one_is_written(guard_factory):
    """The whole point. Recorded as the model goes, so this is about ORDER in
    time and not just about what was eventually said."""
    orch, llm = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)
    timeline = []

    async def note(model):
        timeline.append(("sent", model.fragments_sent, len(voice.said)))

    llm.on_fragment = note
    await orch.handle_user_input("tell me about the measurements")

    spoke_early = any(said > 0 and sent < len(THREE) for _, sent, said in timeline)
    assert spoke_early, f"nothing was said until the model finished: {timeline}"


@pytest.mark.asyncio
async def test_the_answer_is_said_in_sentences_not_fragments(guard_factory):
    orch, _ = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)
    await orch.handle_user_input("tell me")

    assert len(voice.said) > 1, "the answer was said as one utterance"
    assert len(voice.said) < len(THREE), \
        f"the fragments were spoken as they arrived: {voice.said}"
    for utterance in voice.said:
        assert len(utterance.split()) > 2, f"said in pieces too small: {utterance!r}"


@pytest.mark.asyncio
async def test_every_word_is_said_exactly_once(guard_factory):
    orch, _ = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)
    await orch.handle_user_input("tell me")

    whole = "".join(THREE)
    said = " ".join(voice.said)
    assert said.split() == whole.split(), \
        f"the answer was not reconstructed:\n  said: {said!r}\n  wrote: {whole!r}"


@pytest.mark.asyncio
async def test_the_answer_is_not_said_again_at_the_end_of_the_turn(guard_factory):
    """It was said while it was being written. Saying it again is the answer
    twice, which is the obvious way to get this wrong."""
    orch, _ = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)
    answer = await orch.handle_user_input("tell me")

    assert answer not in voice.said, "the whole answer was spoken a second time"
    assert " ".join(voice.said).count("The first measurement") == 1


@pytest.mark.asyncio
async def test_the_turn_still_returns_the_whole_answer(guard_factory):
    orch, _ = _orchestrator(guard_factory, [THREE])
    Voice().attach(orch)
    answer = await orch.handle_user_input("tell me")
    assert answer == "".join(THREE).strip()


@pytest.mark.asyncio
async def test_the_buffer_holds_the_whole_answer_for_showing(guard_factory):
    orch, _ = _orchestrator(guard_factory, [THREE])
    Voice().attach(orch)
    await orch.handle_user_input("tell me")
    assert transcript.current().text == "".join(THREE).strip()


@pytest.mark.asyncio
async def test_greek_streamed_in_fragments_is_said_correctly(guard_factory):
    pieces = ["Το αποτέλεσμα ", "είναι σωστό εδώ. ", "Και το δεύτερο επίσης."]
    orch, _ = _orchestrator(guard_factory, [pieces])
    voice = Voice().attach(orch)
    await orch.handle_user_input("πες μου")

    assert " ".join(voice.said).split() == "".join(pieces).split()


@pytest.mark.asyncio
async def test_a_client_with_no_streaming_still_works(guard_factory):
    """Backward compatibility, and what every existing test double relies on."""
    class WholeResponseOnly:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tools=None):
            self.calls += 1
            return {"message": {"content": "All at once.", "tool_calls": None}}

    orch, _ = _orchestrator(guard_factory, [])
    orch.llm_client = WholeResponseOnly()
    voice = Voice().attach(orch)
    answer = await orch.handle_user_input("tell me")

    assert answer == "All at once."
    assert voice.said == ["All at once."]


@pytest.mark.asyncio
async def test_exactly_one_model_call_per_answer(guard_factory):
    orch, llm = _orchestrator(guard_factory, [THREE])
    Voice().attach(orch)
    await orch.handle_user_input("tell me")
    assert llm.calls == 1


# --- Mathematics ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_expression_arriving_in_pieces_is_never_said_as_markup(guard_factory):
    """The guarantee that must survive streaming. A fragment ending mid-expression
    is held back, because `say` only rewrites a span it can see the end of."""
    pieces = ["The kinetic energy is ", r"\(E_k = ", r"\frac{1}{2}", r"mv^2\)",
              " for a moving body."]
    orch, _ = _orchestrator(guard_factory, [pieces])
    voice = Voice().attach(orch)
    await orch.handle_user_input("what is kinetic energy?")

    assert voice.said, "nothing was said at all"
    for utterance in voice.said:
        assert not speech.contains_markup(utterance), f"markup was spoken: {utterance!r}"
        for banned in ("\\frac", "\\(", "$", "^", "{", "}", "_"):
            assert banned not in utterance, f"{banned!r} reached the voice: {utterance!r}"
    assert "one half" in " ".join(voice.said)


@pytest.mark.asyncio
async def test_an_incomplete_expression_is_not_spoken_while_it_is_incomplete(guard_factory):
    """Checked as the stream runs, not only at the end: the utterance must not
    exist while the expression is still open."""
    pieces = ["The result is ", r"\( \frac{a", "}{b} ", r"\)", " exactly."]
    orch, llm = _orchestrator(guard_factory, [pieces])
    voice = Voice().attach(orch)

    async def check(model):
        for utterance in voice.said:
            assert r"\frac" not in utterance and r"\(" not in utterance, \
                f"half an expression was said: {utterance!r}"

    llm.on_fragment = check
    await orch.handle_user_input("what is it?")
    assert "a over b" in " ".join(voice.said)


@pytest.mark.asyncio
async def test_a_completed_expression_still_renders_when_it_was_asked_for(guard_factory):
    pieces = [r"The energy is \(E_k = ", r"\frac{1}{2}mv^2\)."]
    orch, _ = _orchestrator(guard_factory, [pieces])
    voice = Voice().attach(orch)
    await orch.handle_user_input("show me the formula for kinetic energy")

    maths = [v for v in voice.shown if v.get("type") == "math"]
    assert maths, "a completed expression did not render"
    assert maths[0]["items"][0]["latex"] == r"E_k = \frac{1}{2}mv^2"


@pytest.mark.asyncio
async def test_no_visual_is_produced_from_a_fragment(guard_factory):
    """Visuals are evaluated on completed content, not on every piece."""
    pieces = [r"It is \(\frac{a", "}{b}", r"\) exactly."]
    orch, llm = _orchestrator(guard_factory, [pieces])
    voice = Voice().attach(orch)

    async def check(model):
        assert not [v for v in voice.shown if v.get("type") == "math"], \
            "a panel opened while the expression was still arriving"

    llm.on_fragment = check
    await orch.handle_user_input("what is the formula?")


# --- When the stream dies part-way ---------------------------------------------------------------

class DyingLLM:
    """Writes a few fragments, then reports the connection gone - the shape
    core/llm_client.py produces when a read fails mid-answer."""

    def __init__(self, fragments, error="ConnectionError: the connection went away"):
        self.fragments = list(fragments)
        self.error = error
        self.calls = 0

    async def chat(self, messages, tools=None):
        raise AssertionError("the streaming path should not fall back to chat()")

    async def stream_response(self, messages, tools=None, model=None, should_stop=None):
        self.calls += 1
        content = []
        for fragment in self.fragments:
            content.append(fragment)
            yield {"text": fragment}
        yield {"done": True, "stopped": False, "error": self.error,
               "message": {"role": "assistant", "content": "".join(content)}}


@pytest.mark.asyncio
async def test_a_half_answer_is_not_presented_as_a_whole_one(guard_factory):
    """The rule from the brief: do not silently pretend the complete answer was
    generated. What arrived is kept and said; that it stopped is said too."""
    orch, _ = _orchestrator(guard_factory, [])
    orch.llm_client = DyingLLM(["Το αποτέλεσμα ", "είναι"])
    voice = Voice().attach(orch)

    answer = await orch.handle_user_input("πες μου")

    assert "Το αποτέλεσμα είναι" in answer, "what arrived was thrown away"
    assert "could not finish" in answer, "a cut-off answer looked complete"
    assert any("could not finish" in u for u in voice.said), \
        "the user heard a sentence stop mid-thought with no explanation"


@pytest.mark.asyncio
async def test_what_arrived_before_the_failure_is_still_spoken(guard_factory):
    orch, _ = _orchestrator(guard_factory, [])
    orch.llm_client = DyingLLM(["The first part is fine. ", "The second is cut"])
    voice = Voice().attach(orch)
    await orch.handle_user_input("tell me")

    assert any("first part is fine" in u for u in voice.said)


@pytest.mark.asyncio
async def test_a_failed_stream_does_not_ask_the_model_again(guard_factory):
    """No retry is started here. The fallback model in core/llm_client.py is the
    one retry policy and it covers the request, not a dead connection."""
    orch, _ = _orchestrator(guard_factory, [])
    orch.llm_client = DyingLLM(["partial"])
    Voice().attach(orch)
    await orch.handle_user_input("tell me")
    assert orch.llm_client.calls == 1, "a second model call was made on its own"


@pytest.mark.asyncio
async def test_the_system_is_usable_after_a_failed_stream(guard_factory):
    orch, _ = _orchestrator(guard_factory, [["A clean answer this time."]])
    orch.llm_client = DyingLLM(["cut off"])
    Voice().attach(orch)
    await orch.handle_user_input("tell me")

    orch.llm_client = StreamingLLM([["A clean answer this time."]])
    # Deliberately not "try again": that is a retry command about a task, which
    # core/intent.py reads before the model, and rightly so.
    answer = await orch.handle_user_input("what about the second measurement?")
    assert answer == "A clean answer this time."
    assert orch._turn_lock.locked() is False


# --- The transcript stays hidden ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_does_not_put_the_answer_on_screen(guard_factory):
    orch, _ = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)
    await orch.handle_user_input("tell me")

    assert [v for v in voice.shown if v.get("type") == "transcript"] == [], \
        "a streamed answer showed itself"
    assert transcript.is_visible() is False


@pytest.mark.asyncio
async def test_nothing_is_pushed_to_a_screen_per_fragment(guard_factory):
    orch, llm = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)

    async def check(model):
        assert voice.shown == [], f"a fragment reached the screen: {voice.shown}"

    llm.on_fragment = check
    await orch.handle_user_input("tell me")


@pytest.mark.asyncio
async def test_asking_to_see_a_streamed_answer_still_works(guard_factory):
    orch, llm = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)
    await orch.handle_user_input("tell me")
    before = llm.calls

    await orch.handle_user_input("show me what you said")

    shown = [v for v in voice.shown if v.get("type") == "transcript" and v.get("visible")]
    assert shown, "the streamed answer could not be shown"
    assert shown[-1]["text"] == "".join(THREE).strip()
    assert llm.calls == before, "showing a streamed answer asked the model again"


# --- Visuals are untouched ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_chart_still_appears_during_a_streamed_turn(guard_factory):
    orch, _ = _orchestrator(guard_factory, [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "visualize_dataset", "arguments": {"path": "/tmp/x.csv"}}}]},
        ["Here is the graph."]])
    voice = Voice().attach(orch)
    await orch.handle_user_input("plot these measurements")

    charts = [v for v in voice.shown if v.get("type") == "images"]
    assert charts, "the chart never reached the screen"
    assert charts[0]["offer"] is False


@pytest.mark.asyncio
async def test_a_preamble_before_a_tool_call_is_not_spoken(guard_factory):
    """The loop throws that prose away and answers again afterwards, so speaking
    it would say something the user was never told was provisional."""
    orch, _ = _orchestrator(guard_factory, [
        {"role": "assistant", "content": "Let me look that up for you now.",
         "tool_calls": [
             {"function": {"name": "visualize_dataset", "arguments": {"path": "/tmp/x.csv"}}}]},
        ["Here is the graph."]])
    voice = Voice().attach(orch)
    await orch.handle_user_input("plot these measurements")

    assert not any("look that up" in u for u in voice.said), \
        f"a discarded preamble was spoken: {voice.said}"
    assert any("graph" in u for u in voice.said)


# --- Stopping, immediately -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_does_not_wait_for_the_turn_that_is_speaking(guard_factory):
    """The gap this closes. The turn holds the turn lock while it speaks, so a
    stop routed through the lock would arrive after the answer it was stopping."""
    orch, llm = _orchestrator(guard_factory, [THREE])
    Voice().attach(orch)
    answers = {}

    async def say_stop_midway(model):
        if model.fragments_sent == 2 and "stop" not in answers:
            # Sent from outside the turn, exactly as another surface would.
            answers["stop"] = await orch.handle_user_input("stop")

    llm.on_fragment = say_stop_midway
    await asyncio.wait_for(orch.handle_user_input("tell me about it"), timeout=5)

    assert answers.get("stop") == "Stopped.", "the stop never came back"
    assert llm.fragments_sent < len(THREE), \
        f"the model kept generating after stop: {llm.fragments_sent}/{len(THREE)}"
    assert llm.finished_streams == 0, "the stream ran to completion after stop"


@pytest.mark.asyncio
async def test_stop_cuts_the_utterance_already_playing(guard_factory):
    orch, llm = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)

    async def stop_once_speaking(model):
        if voice.said and voice.interrupted == 0:
            await orch.handle_user_input("stop")

    llm.on_fragment = stop_once_speaking
    await orch.handle_user_input("tell me")

    assert voice.interrupted == 1, "the engine's interrupt was never called"


@pytest.mark.asyncio
async def test_nothing_further_is_said_after_stop(guard_factory):
    orch, llm = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)
    said_when_stopped = {}

    async def stop_after_first_utterance(model):
        if voice.said and "at" not in said_when_stopped:
            said_when_stopped["at"] = len(voice.said)
            await orch.handle_user_input("stop")

    llm.on_fragment = stop_after_first_utterance
    await orch.handle_user_input("tell me")

    assert len(voice.said) == said_when_stopped["at"], \
        f"speech continued after stop: {voice.said}"


@pytest.mark.asyncio
async def test_a_sentence_ready_when_stop_arrives_never_starts(guard_factory):
    """The race in the brief: the utterance exists, STOP lands, and it must not
    begin. The check sits between the queue and the engine, where it has to."""
    orch, _ = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)

    async def stop_just_before_speaking(surface):
        if not surface.said:
            transcript.stop_speaking()

    voice.before_saying = stop_just_before_speaking
    await orch.handle_user_input("tell me")

    assert voice.said == [], f"an utterance started after stop: {voice.said}"


@pytest.mark.asyncio
async def test_a_fragment_arriving_after_stop_is_discarded(guard_factory):
    """The second race from the brief: the voice is cut, and a fragment the model
    had already put on the wire turns up afterwards. It must not be spoken."""
    orch, llm = _orchestrator(guard_factory, [THREE])
    voice = Voice().attach(orch)
    stopped_after = {}

    async def stop_on_the_second_fragment(model):
        if model.fragments_sent == 2 and not stopped_after:
            stopped_after["said"] = len(voice.said)
            transcript.stop_speaking()

    llm.on_fragment = stop_on_the_second_fragment
    await orch.handle_user_input("tell me")

    assert len(voice.said) == stopped_after["said"], \
        f"a fragment that arrived after the stop was spoken: {voice.said}"
    assert llm.finished_streams == 0, "the model was left to finish anyway"


@pytest.mark.asyncio
async def test_a_stop_left_by_the_previous_turn_does_not_silence_this_one(guard_factory):
    """The mirror of the test above, and the bug it is easy to write instead: a
    stop belongs to the answer it stopped. Saying "stop" once must not mute
    everything afterwards."""
    orch, _ = _orchestrator(guard_factory, [["A fresh answer, said in full."]])
    voice = Voice().attach(orch)
    transcript.stop_speaking()

    await orch.handle_user_input("tell me")

    assert voice.said, "an old stop silenced a new answer"


@pytest.mark.asyncio
async def test_a_new_request_after_stop_works_normally(guard_factory):
    orch, llm = _orchestrator(guard_factory, [THREE, ["A clean new answer here."]])
    voice = Voice().attach(orch)

    async def stop_midway(model):
        if model.fragments_sent == 2:
            await orch.handle_user_input("stop")

    llm.on_fragment = stop_midway
    await orch.handle_user_input("tell me")
    llm.on_fragment = None

    answer = await orch.handle_user_input("something else")

    assert answer == "A clean new answer here."
    assert "A clean new answer here." in " ".join(voice.said), \
        "the new turn inherited the stop from the old one"


@pytest.mark.asyncio
async def test_no_stale_state_leaks_into_the_next_turn(guard_factory):
    orch, llm = _orchestrator(guard_factory, [THREE, ["The next answer."]])
    Voice().attach(orch)

    async def stop_midway(model):
        if model.fragments_sent == 2:
            await orch.handle_user_input("stop")

    llm.on_fragment = stop_midway
    await orch.handle_user_input("tell me")
    llm.on_fragment = None
    await orch.handle_user_input("and now?")

    assert transcript.should_stop_speaking() is False, "the stop outlived its turn"
    assert transcript.current().text == "The next answer."


@pytest.mark.asyncio
async def test_the_turn_ends_in_a_clean_idle_state(guard_factory):
    from core.orchestrator import AgentState

    orch, llm = _orchestrator(guard_factory, [THREE])
    Voice().attach(orch)
    states = []
    orch._state_listeners = [states.append]

    async def stop_midway(model):
        if model.fragments_sent == 2:
            await orch.handle_user_input("stop")

    llm.on_fragment = stop_midway
    await orch.handle_user_input("tell me")

    assert states[-1] == AgentState.IDLE, f"left in {states[-1]}"
    assert orch._turn_lock.locked() is False, "the turn lock was not released"


@pytest.mark.asyncio
async def test_a_stopped_answer_is_still_there_to_read(guard_factory):
    """Stopping silences the voice; it does not throw away what was written."""
    orch, llm = _orchestrator(guard_factory, [THREE])
    Voice().attach(orch)

    async def stop_midway(model):
        if model.fragments_sent == 3:
            await orch.handle_user_input("stop")

    llm.on_fragment = stop_midway
    await orch.handle_user_input("tell me")

    assert transcript.current().text, "the stopped answer was discarded"
    assert "first measurement" in transcript.current().text


@pytest.mark.asyncio
async def test_a_stop_naming_a_task_is_not_the_shortcut(guard_factory):
    """"Stop the research task" is about work, and work is the task controls'
    business on the ordinary path. Only a bare stop jumps the queue."""
    from core.orchestrator import _is_stop

    assert _is_stop("stop") is True
    assert _is_stop("stop it") is True
    assert _is_stop("stop talking") is True
    assert _is_stop("stop the research task") is False
    assert _is_stop("what is the weather") is False
    assert _is_stop(None) is False


def test_the_shortcut_only_applies_while_a_turn_is_running():
    """With nothing in flight the ordinary path answers it properly, including
    the task controls - so the shortcut is not a way around them."""
    import inspect

    source = inspect.getsource(Orchestrator.handle_user_input)
    assert "self._turn_lock.locked() and _is_stop(user_text)" in source


def _code_names(source):
    """What a function DOES, with its prose left out.

    A docstring that explains there is no thread to kill reads, to a text scan,
    as a use of the word kill. This walks the tree instead.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    return names


def test_stopping_reuses_the_flag_rather_than_adding_a_mechanism():
    import inspect

    source = inspect.getsource(Orchestrator._stop_now)
    assert "transcript.stop_speaking()" in source
    names = _code_names(source)
    assert "interrupt_callback" in names
    for forbidden in ("Thread", "Event", "Queue", "task_manager", "kill", "terminate"):
        assert forbidden not in names, f"_stop_now reaches for {forbidden}"


def test_the_stream_is_cancelled_cooperatively():
    import inspect

    from core.llm_client import OllamaClient

    source = inspect.getsource(OllamaClient.stream_response)
    names = _code_names(source)
    assert "should_stop" in names
    for forbidden in ("Thread", "signal", "kill", "terminate", "Queue"):
        assert forbidden not in names, f"the stream uses {forbidden}"
    # Cooperative means the caller's flag decides, checked per chunk - not a
    # spin. There is exactly one loop over the body and it is bounded by it.
    assert "while True" not in source


def test_streaming_adds_no_worker_and_no_loop():
    import ast
    import pathlib

    for module in ("core/llm_client.py", "core/speech.py"):
        tree = ast.parse(pathlib.Path(module).read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for forbidden in ("Thread", "Queue", "Timer", "Process", "create_task",
                          "ensure_future", "sleep"):
            assert forbidden not in names, f"{module} uses {forbidden}"


# --- The stream buffer on its own -----------------------------------------------------------------

def test_the_buffer_hands_over_whole_sentences_only():
    buffer = speech.Stream()
    assert buffer.feed("The acce") == []
    assert buffer.feed("leration of the ") == []
    assert buffer.feed("object is F over m.") == []
    assert buffer.feed(" And then") == ["The acceleration of the object is F over m."]


def test_the_buffer_reconstructs_what_it_was_fed():
    buffer = speech.Stream()
    pieces = ["Ένα ", "δύο. ", "Τρία τέσσερα πέντε. ", "Έξι."]
    for piece in pieces:
        buffer.feed(piece)
    buffer.flush()
    assert buffer.text == "".join(pieces)


def test_the_buffer_holds_an_open_expression():
    buffer = speech.Stream()
    assert buffer.feed(r"The result is \( \frac{a") == []
    assert buffer.feed("}{b}") == []
    assert buffer.pending, "an open expression was let through"
    assert buffer.feed(r"\) exactly. And more text after it.") or buffer.flush()


@pytest.mark.parametrize("held", [
    "A number like 3.14159 does not", "The file report.md is not",
    "See example.com/a.b for", "Dr. Adams said", "At 9.8 m/s it",
])
def test_the_buffer_does_not_cut_at_a_false_boundary(held):
    buffer = speech.Stream()
    out = buffer.feed(held) + buffer.feed(" the end of a sentence at all here.")
    out += buffer.flush()
    inside = held.split()[-2] if len(held.split()) > 1 else held
    assert any(inside in u for u in out), f"cut inside {held!r}: {out}"


def test_a_very_long_sentence_is_eventually_handed_over():
    buffer = speech.Stream()
    out = []
    for i in range(80):
        out += buffer.feed(f"word{i} ")
    assert out, "a sentence with no punctuation was held forever"
    for utterance in out:
        assert not utterance.startswith("word") or " " in utterance
        assert "wor" not in utterance.split()[-1] or utterance.split()[-1].startswith("word")


def test_a_long_sentence_is_only_cut_at_whitespace():
    buffer = speech.Stream()
    out = []
    for i in range(80):
        out += buffer.feed(f"word{i} ")
    out += buffer.flush()
    words = " ".join(out).split()
    for word in words:
        assert word.startswith("word") and word[4:].isdigit(), \
            f"a word was cut in half: {word!r}"


def test_the_buffer_costs_nothing_per_fragment():
    import time

    buffer = speech.Stream()
    buffer.feed("warm up. ")
    started = time.perf_counter()
    for i in range(2000):
        buffer.feed(f"fragment {i} ")
        if i % 20 == 0:
            buffer.flush()
    each = (time.perf_counter() - started) / 2000
    assert each < 0.001, f"{each * 1e6:.0f} us per streamed fragment"


def test_nothing_is_emitted_for_nothing():
    buffer = speech.Stream()
    assert buffer.feed("") == []
    assert buffer.flush() == []
    assert buffer.text == ""
