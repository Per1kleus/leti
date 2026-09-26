"""The four ways Leti could answer a user with nothing at all.

Each of these was reported from a real Windows machine as a dead subsystem, and
each is really a failure that had no way of being seen: the reply handler treated
"handed to the speech engine" as "spoken", the weather panel threw away the reason
it had no weather, the voice loop returned silently when nothing was transcribed,
and the task panel turned a failed call into an empty list.

The shared property being pinned: a failure must reach the user as something,
never as silence.
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio import setup as audio_setup  # noqa: E402
from audio.tts import Pyttsx3TTS  # noqa: E402
from core import transcript  # noqa: E402


class RecordingAPI:
    def __init__(self):
        self.pushes = []

    def push(self, event, payload=None):
        self.pushes.append((event, payload))

    def sent(self, event):
        return [p for e, p in self.pushes if e == event]


def engine_that(side_effect):
    """A Pyttsx3TTS whose engine behaves as given, without a real speech engine."""
    tts = Pyttsx3TTS.__new__(Pyttsx3TTS)
    tts.engine = MagicMock()
    tts.engine.runAndWait.side_effect = side_effect
    tts._current_task = None
    return tts


# --- Speaking that does not speak ------------------------------------------------

@pytest.mark.asyncio
async def test_speak_reports_whether_it_actually_spoke():
    assert await engine_that(None).speak("hello") is True
    assert await engine_that(RuntimeError("run loop already started")).speak("hello") is False
    assert await engine_that(OSError("no voice installed")).speak("hello") is False
    assert await engine_that(None).speak("   ") is False, "empty text was not spoken"


@pytest.mark.asyncio
async def test_a_reply_that_could_not_be_spoken_is_shown_instead():
    """The failure behind "Leti never produces a response".

    pyttsx3 raises "run loop already started" when runAndWait() is re-entered, and
    streaming speaks an answer as a series of utterances back to back - so that is
    the normal case, not a rare one. It was caught and logged, speak() returned
    normally, and the turn completed. Voice-first means a spoken answer is not also
    printed, so the user got no audio, no text, and no error: an entire reply
    disappearing one sentence at a time.
    """
    from gui.api import _make_gui_speak_callback

    api = RecordingAPI()
    transcript.clear()
    transcript.begin()
    await _make_gui_speak_callback(api, engine_that(RuntimeError("run loop already started")))(
        "The answer is 4. Anything else?")

    assert api.sent("appendLetiReply") == ["The answer is 4. Anything else?"], \
        "the answer was not delivered at all"


@pytest.mark.asyncio
async def test_a_reply_that_was_spoken_is_not_also_printed():
    """Voice-first is preserved: this must not become a chat window that talks."""
    from gui.api import _make_gui_speak_callback

    api = RecordingAPI()
    transcript.clear()
    transcript.begin()
    await _make_gui_speak_callback(api, engine_that(None))("Spoken perfectly well.")
    assert api.sent("appendLetiReply") == [], "a spoken answer was printed too"


@pytest.mark.asyncio
async def test_a_stopped_reply_is_not_printed_as_a_failure():
    """Stopping is the user's decision, not a delivery failure - and printing the
    rest of what they just stopped would be the opposite of stopping."""
    from gui.api import _make_gui_speak_callback

    api = RecordingAPI()
    transcript.clear()
    transcript.begin()
    transcript.stop_speaking()
    await _make_gui_speak_callback(api, engine_that(None))("One. Two. Three.")
    assert api.sent("appendLetiReply") == []


# --- Weather that says why -------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("raised,expected", [
    (None, "LOCATION_UNAVAILABLE"),
    ("provider", "PROVIDER_ERROR"),
    ("network", "NETWORK_ERROR"),
    ("garbage", "INVALID_RESPONSE"),
])
async def test_weather_says_which_kind_of_failure_it_was(monkeypatch, raised, expected):
    """"Unavailable" gave somebody no way to tell a missing location from an
    outage - and the location case is the one they can fix in thirty seconds."""
    import httpx

    from gui.api import LetiAPI
    from tools.weather import LocationUnavailable

    def failing():
        if raised is None:
            raise LocationUnavailable("No location is set for weather. Add ...")
        if raised == "provider":
            raise httpx.HTTPStatusError(
                "503", request=httpx.Request("GET", "http://x"),
                response=httpx.Response(503, request=httpx.Request("GET", "http://x")))
        if raised == "network":
            raise httpx.ConnectError("no route")
        raise ValueError("not json")

    async def fake():
        return failing()

    monkeypatch.setattr("gui.api.get_current_weather", fake)
    result = await LetiAPI.a_get_weather(LetiAPI.__new__(LetiAPI))
    assert result["reason"] == expected
    assert result["error"], "a reason with no sentence to go with it"


@pytest.mark.asyncio
async def test_a_real_reading_is_marked_as_one(monkeypatch):
    from gui.api import LetiAPI

    async def fresh():
        return {"temperature": 14, "unit": "C", "condition": "Clear",
                "location": "Athens", "stale": False}

    async def cached():
        return {"temperature": 14, "unit": "C", "condition": "Clear",
                "location": "Athens", "stale": True, "age_minutes": 90}

    monkeypatch.setattr("gui.api.get_current_weather", fresh)
    assert (await LetiAPI.a_get_weather(LetiAPI.__new__(LetiAPI)))["reason"] == "SUCCESS"
    monkeypatch.setattr("gui.api.get_current_weather", cached)
    assert (await LetiAPI.a_get_weather(LetiAPI.__new__(LetiAPI)))["reason"] == "STALE_DATA"


def test_the_panel_renders_a_reason_rather_than_just_unavailable():
    hud = (ROOT / "gui" / "hud.html").read_text(encoding="utf-8")
    assert "WEATHER_REASONS" in hud
    for code in ("LOCATION_UNAVAILABLE", "NETWORK_ERROR", "PROVIDER_ERROR",
                 "INVALID_RESPONSE"):
        assert code in hud, f"the panel cannot render {code}"
    assert "setWeatherUnavailable(data.reason, data.error)" in hud


# --- Voice input that says where it stopped --------------------------------------

def _audio(level: int, samples: int = 16000) -> bytes:
    return struct.pack(f"{samples}h", *([level] * samples))


@pytest.mark.parametrize("audio,text,failed,expected", [
    (_audio(12000), "what is the weather", False, "TRANSCRIPTION_SUCCESS"),
    (_audio(12000), "", False, "TRANSCRIPTION_EMPTY"),
    (_audio(12000), "   ", False, "TRANSCRIPTION_EMPTY"),
    (_audio(5), "", False, "VAD_REJECTED"),
    (_audio(12000), "", True, "TRANSCRIPTION_FAILED"),
    (b"", "", False, "NO_AUDIO_SIGNAL"),
])
def test_a_captured_utterance_is_classified_by_where_it_stopped(audio, text, failed, expected):
    assert audio_setup.classify_capture(audio, text, failed=failed) == expected


@pytest.mark.parametrize("stage", ["VAD_REJECTED", "TRANSCRIPTION_EMPTY",
                                   "TRANSCRIPTION_FAILED", "TRANSCRIPTION_SUCCESS"])
def test_a_downstream_failure_never_blames_the_microphone(stage):
    """The rule this whole vocabulary exists for: the microphone worked in every
    one of these, and telling somebody it is dead sends them to fix hardware that
    is fine."""
    meaning = audio_setup.describe_stage(stage)
    assert meaning and meaning != audio_setup.describe_stage("NOT_A_STAGE")
    if stage != "TRANSCRIPTION_SUCCESS":
        assert "microphone is working" in meaning, meaning


@pytest.mark.parametrize("stage,expected", [
    ("NO_DEVICE", "No microphone"),
    ("DEVICE_OPEN_FAILED", "could not be opened"),
    ("NO_AUDIO_SIGNAL", "nothing came through"),
])
def test_the_microphone_stages_do_talk_about_the_microphone(stage, expected):
    assert expected in audio_setup.describe_stage(stage)


def test_the_microphone_check_reports_the_stage_it_reached():
    """measure_microphone answers with a peak rather than a boolean precisely so
    that a live-but-silent input is not read as success; the stage names it."""
    import inspect

    source = inspect.getsource(audio_setup.measure_microphone)
    for stage in ("NO_DEVICE", "DEVICE_OPEN_FAILED", "AUDIO_SIGNAL_DETECTED",
                  "NO_AUDIO_SIGNAL"):
        assert stage in source, f"{stage} is never reported"


def test_the_voice_loop_does_not_return_silently_on_an_empty_transcript():
    """`if not text: return` is what made "I spoke and nothing happened" look
    like a dead microphone."""
    source = (ROOT / "gui" / "api.py").read_text(encoding="utf-8")
    body = source[source.index("async def handle_utterance"):
                  source.index("if continuous:")]
    assert "classify_capture" in body, "nothing works out why there were no words"
    assert "record_activity" in body, "the user is never told"
    assert re.search(r"if not text:\s*\n\s*return", body) is None, \
        "the silent return is back"


def test_a_transcription_that_raises_is_told_apart_from_one_that_heard_nothing():
    source = (ROOT / "gui" / "api.py").read_text(encoding="utf-8")
    assert "async def _transcribe" in source
    body = source[source.index("async def _transcribe"):]
    assert "return \"\", True" in body, "a raising Whisper is folded into an empty result"


# --- The task panel says when it could not read the list -------------------------

def test_the_task_panel_distinguishes_no_tasks_from_a_failed_call():
    """An empty list and a broken transport looked identical in the one panel
    whose job is to say what is running."""
    hud = (ROOT / "gui" / "hud.html").read_text(encoding="utf-8")
    body = hud[hud.index("function refreshTasks"):hud.index("function taskSummaryLine")]
    assert "catch(() => ({active: [], recent: []}))" not in body, "still silent"
    assert "Could not read the task list" in body
    assert "tasks.error" in body, "the backend's own error is ignored"


# --- The two contracts between the page and the server ---------------------------

def test_every_method_the_page_calls_is_one_the_server_answers():
    """A name the server does not know comes back as an error the page cannot do
    anything with - and there are 36 of them to keep in step by hand."""
    from gui.api import SYNC_METHODS, LetiAPI

    hud = (ROOT / "gui" / "hud.html").read_text(encoding="utf-8")
    server = (ROOT / "gui" / "server.py").read_text(encoding="utf-8")
    wanted = set(re.findall(r"callApi\('([a-zA-Z_]+)'", hud))
    explicit = set(re.findall(r'method == "([a-zA-Z_]+)"', server))

    missing = []
    for method in sorted(wanted):
        if method in explicit:
            continue
        if method in SYNC_METHODS and hasattr(LetiAPI, method):
            continue
        missing.append(method)
    assert not missing, f"the page calls methods the server cannot answer: {missing}"


def test_every_event_the_backend_pushes_is_one_the_page_will_run():
    """The page runs pushes from an allowlist, so an unlisted event is dropped in
    silence - which is how a backend that is working looks like one that is not."""
    api = (ROOT / "gui" / "api.py").read_text(encoding="utf-8")
    server = (ROOT / "gui" / "server.py").read_text(encoding="utf-8")
    hud = (ROOT / "gui" / "hud.html").read_text(encoding="utf-8")

    pushed = set(re.findall(r'push\(\s*["\']([A-Za-z_]+)["\']', api + server))
    allowlist = hud[hud.index("PUSH_HANDLERS"):hud.index("const wsProtocol")]
    allowed = set(re.findall(r"'([A-Za-z_]+)'", allowlist))
    assert pushed <= allowed, f"pushed but never handled: {sorted(pushed - allowed)}"
