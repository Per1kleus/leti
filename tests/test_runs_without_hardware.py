"""Tests for what Leti does on a machine that is missing hardware it wants.

These come from actually running the app rather than reasoning about it: on a
headless container with no display, no sound card and no browser, Leti failed in
three separate places before reaching the point where it could tell anyone why.
The rule they encode is that missing hardware costs you the feature that needs it
and nothing else - a machine with no microphone still gets a working assistant.

They matter beyond containers. `main.py --mode run-scheduled` is started by cron,
which runs with no DISPLAY however many screens the machine has, so "headless" is
a state every installation is in for part of its life.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- Importing must not need a screen -------------------------------------------

def test_os_control_imports_with_no_display():
    """pyautogui builds an X11 connection at import time, so on a machine with no
    display `import pyautogui` raises KeyError('DISPLAY'). At module scope that took
    down every mode, including the cron-started one that never touches a mouse."""
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp"}  # deliberately no DISPLAY
    result = subprocess.run(
        [sys.executable, "-c", "import tools.os_control; print('imported')"],
        cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


@pytest.mark.asyncio
async def test_mouse_and_keyboard_explain_themselves_with_no_display(monkeypatch):
    """The four tools that genuinely need a screen pay the cost instead, and say
    something a person can act on rather than raising KeyError: 'DISPLAY'."""
    import tools.os_control as os_control

    def no_display():
        raise KeyError("DISPLAY")

    monkeypatch.setattr(os_control, "_mouse_keyboard", no_display)
    monkeypatch.setattr(os_control, "_pyautogui", None)

    for tool, kwargs in (
        (os_control.MouseClickTool(), {"x": 1, "y": 1}),
        (os_control.KeyboardTypeTool(), {"text": "hello"}),
        (os_control.KeyboardHotkeyTool(), {"keys": ["ctrl", "c"]}),
    ):
        result = await tool.run(**kwargs)
        assert result.success is False
        assert "no graphical display" in result.error, tool.name


# --- The interface must survive having no voice ----------------------------------

def test_both_entry_points_load_voice_through_the_one_loader():
    """GUI mode and --mode voice both need "voice, or why not". Two copies of that
    would drift the moment one of them learned about a new failure."""
    import inspect

    import gui.api
    import audio

    for source in (inspect.getsource(gui.api.run_gui_mode),
                   (Path(__file__).resolve().parent.parent / "main.py").read_text()):
        assert "load_voice_stack" in source
    assert callable(audio.load_voice_stack)


def test_gui_mode_does_not_import_the_voice_stack_at_module_scope():
    """`import pyaudio` with no pyaudio installed is an ImportError, and it used to
    happen inside run_gui_mode's first lines - before the web server existed. The
    HUD is fully usable by typing, so that failure has to arrive as a message, not
    instead of the application."""
    import inspect

    import gui.api

    source = inspect.getsource(gui.api.run_gui_mode)
    assert "from audio.stt import" not in source
    assert "from audio.tts import" not in source


def test_voice_stack_reports_why_it_is_unavailable_instead_of_raising():
    from audio import load_voice_stack

    tts, transcriber, error = load_voice_stack()
    # This machine may or may not have a working voice stack; what must hold either
    # way is that the answer is returned, never raised, and that the two halves
    # agree - half a voice pipeline is worse than none.
    assert (tts is None) == (transcriber is None)
    if tts is None:
        assert error and ":" in error
    else:
        assert error is None


@pytest.mark.asyncio
async def test_replies_still_reach_the_screen_with_no_tts():
    """The reply bubble is pushed before anything is spoken, and with no TTS the orb
    must not be left stuck in its speaking animation."""
    from gui.api import _make_gui_speak_callback

    class FakeAPI:
        def __init__(self):
            self.pushed = []

        def push(self, fn, *args):
            self.pushed.append((fn, args))

    api = FakeAPI()
    await _make_gui_speak_callback(api, None)("the answer")

    names = [fn for fn, _ in api.pushed]
    assert ("appendLetiReply", ("the answer",)) in api.pushed
    assert "setHudState" not in names, "no TTS means the orb never enters 'speaking'"
    assert names[-1] == "focusChatInput"


# --- Confirmation has to be answerable without a microphone -----------------------

class _FakeAPI:
    """Enough of LetiAPI for the confirmation handshake."""

    def __init__(self, connected=True):
        self.ws_clients = {"a-client"} if connected else set()
        self.pushed = []
        self._pending_confirmation = None

    def push(self, fn, *args):
        self.pushed.append((fn, args))


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [
    ("yes", True), ("yeah go ahead", True), ("do it", True),
    ("no", False), ("no don't", False), ("stop", False),
])
async def test_a_typed_answer_confirms_or_declines(answer, expected):
    """Voice confirmation was GUI mode's only path, and it needs a mic. On a desktop
    without one the user sat in front of a working chat window unable to approve the
    thing they had just asked for."""
    from core.confirmation import make_text_confirmation_callback

    api = _FakeAPI()
    callback = make_text_confirmation_callback(api)
    task = asyncio.create_task(callback("Leti wants to open sleep with 1."))
    await asyncio.sleep(0)  # let it register and ask

    assert any(fn == "appendLetiReply" and "yes / no" in a[0] for fn, a in api.pushed)
    api._pending_confirmation.set_result(answer)
    assert await task is expected


@pytest.mark.asyncio
async def test_an_unclear_answer_fails_closed():
    from core.confirmation import make_text_confirmation_callback

    api = _FakeAPI()
    task = asyncio.create_task(make_text_confirmation_callback(api)("Do the thing?"))
    await asyncio.sleep(0)
    api._pending_confirmation.set_result("what does that even mean")
    assert await task is False
    assert any("couldn't tell" in a[0].lower() for fn, a in api.pushed if fn == "appendLetiReply")


@pytest.mark.asyncio
async def test_nobody_connected_means_declined():
    """There is no one to ask, so the answer is no - the same fail-closed rule every
    other confirmation path follows."""
    from core.confirmation import make_text_confirmation_callback

    api = _FakeAPI(connected=False)
    assert await make_text_confirmation_callback(api)("Do the thing?") is False


@pytest.mark.asyncio
async def test_the_pending_question_is_cleared_even_if_the_turn_is_cancelled():
    """A stale future here would swallow the user's next message as the answer to a
    question nobody is asking any more."""
    from core.confirmation import make_text_confirmation_callback

    api = _FakeAPI()
    task = asyncio.create_task(make_text_confirmation_callback(api)("Do the thing?"))
    await asyncio.sleep(0)
    assert api._pending_confirmation is not None

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert api._pending_confirmation is None


@pytest.mark.asyncio
async def test_an_answer_is_not_mistaken_for_a_new_request():
    """'yes' typed at a confirmation prompt must answer it, not start a turn asking
    Leti to do something about the word 'yes'."""
    from gui.api import LetiAPI

    class FakeOrchestrator:
        def __init__(self):
            self.turns = []

        async def handle_user_input(self, text, voice_mode=False):
            self.turns.append(text)

    orchestrator = FakeOrchestrator()
    api = LetiAPI(orchestrator, asyncio.get_running_loop())
    api.ws_clients.add("a-client")

    pending = asyncio.get_running_loop().create_future()
    api._pending_confirmation = pending

    await api.a_send_text_message("yes")
    assert pending.result() == "yes"
    assert orchestrator.turns == [], "the answer must not also start a new turn"

    # With nothing pending, the next message is an ordinary request again.
    api._pending_confirmation = None
    await api.a_send_text_message("what's the weather")
    assert orchestrator.turns == ["what's the weather"]
