"""The voice-first layers as they actually sit in the machine.

Three things these check that the unit tests cannot. Stopping: the voice stops,
the answer does not. Cost: nothing new runs, loops, polls or asks a model. And
singularity: there is still one TTS, one visualisation path, one orchestrator
and one place that decides whether a panel opens.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import artifacts, speech, transcript  # noqa: E402

API = pathlib.Path("gui/api.py").read_text()
HUD = pathlib.Path("gui/hud.html").read_text()
ORCHESTRATOR = pathlib.Path("core/orchestrator.py").read_text()
MAIN = pathlib.Path("main.py").read_text()


def _code_of(path):
    """Everything a module actually DOES, with its prose left out.

    A raw text scan reads a docstring explaining why matplotlib is not used as
    a use of matplotlib. This walks the tree instead: names, attributes, calls
    and imports, and no comment or docstring anywhere.
    """
    tree = ast.parse(pathlib.Path(path).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
                names.add(node.module)
            names |= {a.name for a in node.names}
    return names


@pytest.fixture(autouse=True)
def _fresh():
    transcript.clear()
    yield
    transcript.clear()


# --- Stopping ----------------------------------------------------------------------------

class FakeTTS:
    """Counts what was said and whether it was cut off."""

    def __init__(self, stop_after=None):
        self.said = []
        self.interrupted = 0
        self.stop_after = stop_after

    async def speak(self, text):
        self.said.append(text)
        if self.stop_after is not None and len(self.said) >= self.stop_after:
            transcript.stop_speaking()

    def interrupt(self):
        self.interrupted += 1


@pytest.mark.asyncio
async def test_an_answer_is_spoken_in_pieces_not_all_at_once():
    from gui.api import _make_gui_speak_callback

    tts = FakeTTS()
    speak = _make_gui_speak_callback(_SilentAPI(), tts)
    await speak("The first measurement finished cleanly. The second is still running. "
                "The third has not started at all yet.")

    assert len(tts.said) > 1, "the whole answer went to the engine as one utterance"
    assert all(len(u.split()) > 1 for u in tts.said), "something was said word by word"


@pytest.mark.asyncio
async def test_stopping_stops_the_next_utterance():
    from gui.api import _make_gui_speak_callback

    tts = FakeTTS(stop_after=1)
    speak = _make_gui_speak_callback(_SilentAPI(), tts)
    await speak("The first measurement finished cleanly. The second is still running. "
                "The third has not started at all yet.")

    assert len(tts.said) == 1, f"speaking continued after stop: {tts.said}"


@pytest.mark.asyncio
async def test_stopping_does_not_lose_the_answer():
    from gui.api import _make_gui_speak_callback

    transcript.begin()
    transcript.said("The first one. The second one. The third one, which is longer.")
    tts = FakeTTS(stop_after=1)
    speak = _make_gui_speak_callback(_SilentAPI(), tts)
    await speak(transcript.current().text)

    assert transcript.current().text.startswith("The first one.")
    shown = transcript.reveal()
    assert shown["shown"] is True
    assert "third one" in shown["text"], "the part never spoken was thrown away"


@pytest.mark.asyncio
async def test_a_new_answer_clears_a_stop_left_over_from_the_last_one():
    from gui.api import _make_gui_speak_callback

    transcript.stop_speaking()
    tts = FakeTTS()
    speak = _make_gui_speak_callback(_SilentAPI(), tts)
    await speak("A new answer, which should be said in full this time around.")
    assert tts.said, "a stop from the previous answer silenced the next one"


def test_the_stop_control_interrupts_the_engine_and_keeps_the_answer():
    from gui.api import LetiAPI

    api = LetiAPI.__new__(LetiAPI)
    api.ws_clients = set()
    api.tts = FakeTTS()
    transcript.begin()
    transcript.said("Something worth keeping.")
    transcript.start_speaking()

    result = api.stop_speaking()

    assert result["stopped"] is True
    assert result["answer_kept"] is True
    assert api.tts.interrupted == 1, "the utterance already playing was not cut"
    assert transcript.should_stop_speaking() is True
    assert transcript.current().text == "Something worth keeping."


def test_stopping_with_no_voice_at_all_is_not_an_error():
    from gui.api import LetiAPI

    api = LetiAPI.__new__(LetiAPI)
    api.ws_clients = set()
    api.tts = None
    assert api.stop_speaking()["stopped"] is True


def test_the_stop_control_does_not_wait_for_the_turn():
    """Synchronous on purpose: the turn that is speaking holds the turn lock, so
    anything routed through send_text_message would queue behind the speech."""
    assert re.search(r"^    def stop_speaking\(self\)", API, re.M), \
        "stop_speaking became async, which would make it queue behind the speech"
    assert '"stop_speaking",' in API.split("ASYNC_METHODS")[0], \
        "stop_speaking must be a sync method to reach the engine immediately"


def test_stopping_the_voice_is_not_a_second_cancellation_system():
    """core/task_control.py stays the authority on stopping WORK. The speech flag
    cancels speech and nothing else - so the buffer never reaches for a task."""
    code = _code_of("core/transcript.py")
    for forbidden in ("task_manager", "task_control", "cancel", "CancelledError",
                      "core.task_manager", "core.task_control"):
        assert forbidden not in code, f"core/transcript.py reaches for {forbidden}"


def test_stop_reaches_the_voice_through_the_command_that_already_existed():
    assert "transcript.stop_speaking()" in ORCHESTRATOR
    stop_block = ORCHESTRATOR[ORCHESTRATOR.index("control = intent_reader.lifecycle_command"):]
    stop_block = stop_block[:stop_block.index("# Asking to see the text")]
    assert "task_control.apply" in stop_block, "the task controls were bypassed"


class _SilentAPI:
    """A LetiAPI with nobody connected: push is a no-op."""

    def push(self, *args):
        pass


# --- Visuals are untouched ------------------------------------------------------------------

def test_the_image_path_was_not_replaced():
    """search_images and create_sketch still decide their own visuals, and
    core/artifacts.py still decides whether a panel opens."""
    assert "search_images" in artifacts.DELIBERATE_VISUAL_TOOLS
    assert "create_sketch" in artifacts.DELIBERATE_VISUAL_TOOLS
    assert "visualize_dataset" in artifacts.DELIBERATE_VISUAL_TOOLS
    assert artifacts.AUTO_OPEN_KINDS == ("images", "diagram")


def test_there_is_one_place_that_decides_whether_a_panel_opens():
    assert ORCHESTRATOR.count("artifacts.should_open(") == 1, \
        "a second decision about opening panels appeared"


def test_the_page_renders_every_kind_python_can_send():
    kinds = set(re.findall(r"^\s+(\w+): artifact\w+,",
                           HUD[HUD.index("const ARTIFACT_RENDERERS"):
                               HUD.index("const ARTIFACT_RENDERERS") + 400], re.M))
    assert {"images", "diagram", "table", "research", "document", "math"} <= kinds


def test_the_transcript_is_not_an_artifact():
    """It is the answer, shown because it was asked for - never offered in the
    log alongside a chart, and never auto-opened by the artifact rules."""
    assert "transcript" not in artifacts.titles()
    assert "transcript" not in artifacts.AUTO_OPEN_KINDS
    assert "if(payload.type === 'transcript')" in HUD


def test_a_visual_does_not_need_the_transcript_to_be_visible():
    """Stated as the point of the feature; checked as an absence of coupling."""
    show = ORCHESTRATOR[ORCHESTRATOR.index("async def _push_visual"):]
    show = show[:show.index("\n    async def ") if "\n    async def " in show[10:] else 600]
    assert "is_visible" not in show, "showing a visual consults the transcript's state"


# --- Nothing new runs -------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["core/speech.py", "core/math_render.py",
                                    "core/transcript.py"])
def test_no_new_module_starts_anything(module):
    code = _code_of(module)
    for forbidden in ("threading", "multiprocessing", "subprocess", "asyncio",
                      "httpx", "requests", "urllib", "socket", "sqlite3", "schedule"):
        assert forbidden not in code, f"{module} imports {forbidden}"


@pytest.mark.parametrize("module", ["core/speech.py", "core/math_render.py",
                                    "core/transcript.py"])
def test_no_new_module_asks_a_model_anything(module):
    code = _code_of(module)
    for forbidden in ("llm_client", "OllamaClient", "chat", "analyze_image", "embed"):
        assert forbidden not in code, f"{module} reaches for a model ({forbidden})"


def test_showing_the_text_is_reachable_without_a_model():
    """The path from the request to the panel, read off the source: the command
    is answered and returns before _build_messages is ever called."""
    turn = ORCHESTRATOR[ORCHESTRATOR.index("showing = intent_reader.transcript_command"):]
    to_model = turn.index("_build_messages")
    handled = turn.index("return answer")
    assert handled < to_model, "showing the text falls through to the model"


def test_no_second_speech_engine_was_added():
    for module in ("core/speech.py", "core/transcript.py"):
        code = _code_of(module)
        for engine in ("pyttsx3", "gTTS", "pyaudio", "espeak", "piper", "TTS"):
            assert engine not in code, f"{module} builds a second speech engine"
    assert len(re.findall(r"pyttsx3\.init\(\)",
                          pathlib.Path("audio/tts.py").read_text())) == 1


def test_no_second_visual_renderer_was_added():
    """Everything still goes through the one visual_callback and the one page
    dispatcher."""
    assert ORCHESTRATOR.count("self.visual_callback = ") <= 1
    assert HUD.count("window.showVisual = function") == 1


def test_the_math_renderer_draws_no_pictures():
    """An equation is markup the browser lays out, not a PNG. A picture of one
    cannot be selected, searched or scaled, and would be a file to write."""
    code = _code_of("core/math_render.py")
    for forbidden in ("matplotlib", "pyplot", "savefig", "PIL", "Image", "figure"):
        assert forbidden not in code, f"core/math_render.py rasterises via {forbidden}"


def test_the_context_window_is_unchanged():
    assert "28672" in pathlib.Path("tests/test_control_panels.py").read_text()
    for module in ("core/speech.py", "core/math_render.py", "core/transcript.py"):
        assert "num_ctx" not in pathlib.Path(module).read_text()


def test_the_tool_count_is_unchanged():
    """No tool was added for any of this - showing text is a command, not a tool,
    so Default Mode's schema budget is untouched."""
    assert "assert len(visible) == 129" in pathlib.Path("tests/test_business_mode.py").read_text()


def test_speaking_costs_no_extra_websocket_traffic():
    """The reply used to be pushed on every turn. Now it is pushed only when the
    panel is open or there is no voice - strictly fewer messages, never more."""
    speak = API[API.index("def _make_gui_speak_callback"):]
    speak = speak[:speak.index("return _speak")]
    pushes = re.findall(r'api\.push\("(\w+)"', speak)
    assert pushes.count("appendLetiReply") == 1
    guarded = speak[speak.index("appendLetiReply") - 200:speak.index("appendLetiReply")]
    assert "is_visible()" in guarded and "tts is None" in guarded


def test_the_whole_output_path_costs_microseconds():
    import time

    answer = ("The acceleration of the object is force over mass. " * 8 +
              r"We write that as \(a = \frac{F}{m}\).")
    question = "why does it speed up?"

    def one_turn():
        speech.utterances(answer)
        artifacts.asked_to_see_mathematics(question)
        transcript.begin()
        transcript.said(answer)

    one_turn()
    started = time.perf_counter()
    for _ in range(200):
        one_turn()
    each = (time.perf_counter() - started) / 200
    assert each < 0.01, f"{each * 1000:.1f} ms added to every answer"


def test_detecting_mathematics_does_not_fire_on_an_ordinary_question():
    for ordinary in ("what is the weather", "email chris the figures",
                     "what is 2 + 2", "open the project folder"):
        assert artifacts.asked_to_see_mathematics(ordinary) is False


# --- The task interface is separate from the transcript ------------------------------------

def test_task_progress_still_reaches_the_interface():
    """Hiding the conversational transcript must not hide what Leti is doing."""
    assert "get_tasks" in API and "updateTasks" in HUD or "refreshTasks" in HUD
    assert "letiActivity" in API, "the activity log stopped being pushed"


def test_the_task_panel_does_not_go_through_the_transcript():
    tasks = API[API.index("def get_tasks"):]
    tasks = tasks[:tasks.index("\n    def ", 10)]
    assert "transcript" not in tasks


def test_the_voice_mode_terminal_also_holds_the_text_back():
    speak = MAIN[MAIN.index("    async def speak(text: str)"):]
    speak = speak[:speak.index("orchestrator.speak_callback")]
    assert "transcript.is_visible()" in speak, \
        "voice mode prints the whole answer, which is the transcript by another name"
    assert "speech.utterances(" in speak


def test_every_surface_uses_the_same_speech_preparation():
    for surface, source in (("gui/api.py", API), ("main.py", MAIN)):
        assert "speech.utterances(" in source, f"{surface} speaks unprepared text"


# --- The diagnostics view -------------------------------------------------------------------

def test_the_panel_reports_the_output_state_without_the_words():
    import json

    from core import diagnostics

    transcript.begin()
    transcript.said("Something confidential about the merger with Acme.")
    section = diagnostics.output_section()
    assert section["status"] == "ok" and section["holding"] is True
    blob = json.dumps(section).lower()
    for leak in ("confidential", "merger", "acme"):
        assert leak not in blob, "the diagnostics panel printed the conversation"


def test_the_output_section_is_in_the_snapshot():
    from core import diagnostics

    assert "output" in diagnostics.snapshot()


def test_the_panel_never_carries_a_screenshot_or_a_tree():
    import json

    from core import diagnostics

    transcript.begin()
    transcript.add_visual({"type": "images",
                           "images": [{"url": "data:image/png;base64,AAAA"}]})
    blob = json.dumps(diagnostics.output_section())
    assert "base64" not in blob and "data:image" not in blob
    assert "mathml" not in blob
