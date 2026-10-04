"""One spoken turn, end to end, with the real model listening to real audio.

Every other test of the voice path in this project reads the source or stands the
pieces up one at a time, because a microphone and a Whisper install are not things
a test machine has. Those tests are worth having and they could not have caught
what was actually wrong: that the wake word had no model, so nothing downstream of
it ever ran at all.

This one starts from audio. A committed WAV goes through the real
WakeWordListener - the real loop, the real threshold comparison, the real
openWakeWord model loaded from the file that ships - into the real
`gui.api._run_voice_loop`, and out the other side through the real intent reader,
the real Ollama client (against an HTTP server this file starts), the real
sentence chunker and into a speech engine that records what it was asked to say.

What is stubbed, and why:

    the microphone      there isn't one. A fake PyAudio hands out the fixture's
                        frames and then silence, which is exactly what a
                        microphone does.
    Whisper             not installed on a machine with no sound card, and a
                        transcriber that returns a known sentence is what makes
                        the rest of the test an assertion about Leti rather than
                        about speech recognition.
    the orchestrator    a stub, but one that runs the real intent reader, the real
                        client and the real chunker - so the handoffs are real and
                        only the tool-calling and memory it would also do are not.
    the speaker         records instead of making noise.

Nothing about the wake word is stubbed. That is the point.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest
from aiohttp import web

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

pytest.importorskip("openwakeword", reason="openWakeWord is not installed")

from audio import wake_word                                    # noqa: E402
from core import diagnostics, intent, speech                    # noqa: E402
from core.llm_client import OllamaClient                        # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
MODEL = PROJECT / "data" / "wake_words" / "hey_leti.onnx"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wake_word"
SPOKEN = "what is two plus two"

pytestmark = pytest.mark.skipif(not MODEL.is_file(),
                                reason="data/wake_words/hey_leti.onnx is not present")


def _frames(path: Path, trailing_silence_frames: int = 24):
    """The WAV as the 80 ms byte frames a microphone would deliver."""
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    size = wake_word.CHUNK_SAMPLES * 2
    out = [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]
    return out + [b"\x00" * size] * trailing_silence_frames


class _Stream:
    """A microphone that plays a file and then stops existing."""

    class Exhausted(RuntimeError):
        pass

    def __init__(self, frames):
        self._frames = list(frames)
        self.reads = 0

    def read(self, _count, exception_on_overflow=False):
        if not self._frames:
            raise _Stream.Exhausted("the fixture ran out")
        self.reads += 1
        return self._frames.pop(0)

    def stop_stream(self):
        pass

    def close(self):
        pass


class _PyAudio:
    paInt16 = 8

    def __init__(self, frames):
        self._frames = frames
        self.stream = None
        self.opened = 0
        self.terminated = 0

    def PyAudio(self):                                   # noqa: N802 - mirrors pyaudio
        return self

    def open(self, **kwargs):
        self.opened += 1
        self.kwargs = kwargs
        self.stream = _Stream(self._frames)
        return self.stream

    def terminate(self):
        self.terminated += 1


@pytest.fixture(autouse=True)
def clean_trail():
    diagnostics.reset_voice_trail()
    yield
    diagnostics.reset_voice_trail()


def _fixture(prefix: str) -> Path:
    found = sorted(FIXTURES.glob(f"{prefix}*.wav"))
    assert found, f"no {prefix}*.wav in {FIXTURES}"
    return found[0]


async def _listen_to(path: Path, monkeypatch, lead_in_frames: int = 0,
                     warm_on_self: bool = False):
    """Run the real listener over one fixture. Returns how many times it woke."""
    import audio.setup as audio_setup

    frames = _frames(path)
    if warm_on_self:
        # The listener settles on silence in its constructor; for a sound that is
        # already playing, settle it on the sound instead.
        frames = frames[:]
    fake = _PyAudio(frames)
    monkeypatch.setattr(wake_word, "pyaudio", fake)
    monkeypatch.setattr(audio_setup, "chosen_input_device", lambda: None)

    woke = []

    async def on_wake():
        woke.append(True)

    listener = wake_word.WakeWordListener(on_wake=on_wake)
    if warm_on_self:
        size = wake_word.CHUNK_SAMPLES * 2
        with wave.open(str(path), "rb") as handle:
            raw = handle.readframes(handle.getnframes())
        for i in range(0, min(len(raw) - size, listener.SETTLE_FRAMES * size), size):
            listener.model.predict(
                np.frombuffer(raw[i:i + size], dtype=np.int16))
    try:
        await listener.start()
    except _Stream.Exhausted:
        pass
    return woke, listener, fake


# --- The wake word, through its own loop ------------------------------------------

async def test_the_listener_wakes_on_the_fixture(monkeypatch):
    """The whole reason this work happened: Leti answers to its own name.

    Not the model scored in isolation - the loop, with the configured threshold
    read from the real settings, reset() after waking, the lot.
    """
    woke, listener, fake = await _listen_to(_fixture("wake_"), monkeypatch)

    assert woke, "the listener never woke"
    assert listener.resolution.kind == wake_word.CUSTOM
    assert listener.framework == "onnx"
    assert fake.opened == 1, "the microphone was opened more than once"
    assert fake.kwargs["rate"] == wake_word.SAMPLE_RATE
    assert fake.kwargs["frames_per_buffer"] == wake_word.CHUNK_SAMPLES


async def test_the_listener_stays_quiet_on_speech_that_is_not_the_wake_word(monkeypatch):
    for path in sorted(FIXTURES.glob("quiet_*.wav")):
        woke, _listener, _fake = await _listen_to(path, monkeypatch)
        assert not woke, f"woke on {path.name}"


async def test_the_listener_stays_quiet_on_background_noise(monkeypatch):
    """Noise that is simply playing, which is what a room is.

    The frames are fed without the usual silence in front, because a steady sound
    arriving after silence is the onset case - a window half silence and half sound
    has the shape of a word beginning - and that is measured on its own in
    tests/test_hey_leti_model.py and recorded in validation.json. Feeding it here
    would test the onset and call it the noise.
    """
    for path in sorted(FIXTURES.glob("noise_*.wav")):
        woke, _listener, _fake = await _listen_to(path, monkeypatch,
                                                  lead_in_frames=0, warm_on_self=True)
        assert not woke, f"woke on steady {path.name}"


# --- The rest of the turn ---------------------------------------------------------

class _Model:
    """An HTTP model server that answers the one question this test asks.

    Real HTTP, because the client's job is reading an NDJSON stream off a socket
    and a stand-in for that would not be testing it.
    """

    def __init__(self):
        self.asked = []
        self._runner = None
        self.port = 0

    async def start(self):
        app = web.Application()
        app.router.add_get("/api/tags", self._tags)
        app.router.add_post("/api/chat", self._chat)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _tags(self, _request):
        return web.json_response({"models": [{"name": "qwen2.5:7b"}]})

    async def _chat(self, request):
        body = await request.json()
        self.asked.append(body)
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(request)
        for fragment in ("Two plus two ", "is four. ", "Anything else?"):
            await response.write(
                (json.dumps({"message": {"role": "assistant", "content": fragment}})
                 + "\n").encode())
        await response.write(
            (json.dumps({"message": {"role": "assistant", "content": ""},
                         "done": True}) + "\n").encode())
        return response


@pytest.fixture
async def model_server():
    server = await _Model().start()
    yield server
    await server.stop()


class _Transcriber:
    """Whisper's place in the chain, with a known answer."""

    def __init__(self, text: str):
        self.text = text
        self.recordings = 0

    async def record_until_silence(self):
        self.recordings += 1
        diagnostics.record_voice_stage("RECORDING_STARTED")
        audio = (np.zeros(16000, dtype=np.int16)).tobytes()
        diagnostics.record_voice_stage("RECORDING_STOPPED", bytes=len(audio))
        return audio

    async def transcribe(self, audio: bytes):
        diagnostics.record_voice_stage("AUDIO_BUFFER_RECEIVED", bytes=len(audio))
        diagnostics.record_voice_stage("TRANSCRIPTION_STARTED")
        diagnostics.record_voice_stage("TRANSCRIPTION_SUCCESS", chars=len(self.text))
        return self.text


class _Speaker:
    """A speaker that writes down what it was told to say."""

    def __init__(self):
        self.said = []

    def speak(self, text: str):
        self.said.append(text)


class _Orchestrator:
    """A stub that does the real work of a turn with the real components.

    Not the production Orchestrator - that one wants a tool registry, a memory
    store and a vector database, none of which this test is about. What it does
    want is for the transcript to reach the model unchanged and the model's answer
    to reach the speaker in sentences, and both of those are real here.
    """

    def __init__(self, client: OllamaClient, speaker: _Speaker):
        self.client = client
        self.speaker = speaker
        self.seen = []
        self.intents = []

    async def handle_user_input(self, text: str, voice_mode: bool = False, **_kw):
        self.seen.append(text)
        read = intent.read(text)
        self.intents.append(read)
        diagnostics.record_voice_stage("INTENT_CLASSIFIED",
                                       kind=getattr(read, "kind", ""))

        diagnostics.record_voice_stage("MODEL_REQUEST_STARTED")
        stream = speech.Stream()
        first = True
        async for event in self.client.stream_response(
                [{"role": "user", "content": text}]):
            if event.get("done"):
                diagnostics.record_voice_stage("MODEL_RESPONSE_COMPLETE")
                break
            if first:
                diagnostics.record_voice_stage("MODEL_FIRST_FRAGMENT")
                first = False
            for sentence in stream.feed(event["text"]):
                if not self.speaker.said:
                    diagnostics.record_voice_stage("TTS_STARTED")
                self.speaker.speak(sentence)
        for sentence in stream.flush():
            self.speaker.speak(sentence)
        diagnostics.record_voice_stage("TTS_COMPLETE")


class _Api:
    """LetiAPI's push, and nothing else - which is all the voice loop uses."""

    def __init__(self):
        self.pushed = []

    def push(self, method, *args):
        self.pushed.append((method, args))


async def test_a_whole_spoken_turn_runs_from_the_wake_word_to_speech(
        model_server, monkeypatch):
    """Audio in, speech out, through gui.api's own voice loop.

    This is the test that would have failed for every version of this project
    before the model existed: the loop would have raised WakeWordUnavailable and
    fallen through to push-to-talk, and nothing would ever have heard "hey leti".
    """
    import audio.setup as audio_setup
    import core.config_loader as loader
    from gui import api as gui_api

    settings = loader.get_settings()
    settings["ollama"]["host"] = model_server.host
    monkeypatch.setattr(loader, "get_settings", lambda: settings)
    monkeypatch.setattr("core.llm_client.get_settings", lambda: settings)

    fake = _PyAudio(_frames(_fixture("wake_")))
    monkeypatch.setattr(wake_word, "pyaudio", fake)
    monkeypatch.setattr(audio_setup, "chosen_input_device", lambda: None)

    client = OllamaClient()
    speaker = _Speaker()
    orchestrator = _Orchestrator(client, speaker)
    api = _Api()
    transcriber = _Transcriber(SPOKEN)

    try:
        with pytest.raises(_Stream.Exhausted):
            await asyncio.wait_for(
                gui_api._run_voice_loop(orchestrator, api, True, transcriber),
                timeout=120)
    finally:
        await client.close()

    # The wake word was heard, and only because of the model that now ships.
    stages = [entry["stage"] for entry in diagnostics.voice_trail()]
    assert "WAKE_WORD_MODEL_LOADED" in stages
    assert "WAKE_WORD_LISTENING" in stages
    assert "WAKE_WORD_DETECTED" in stages, stages
    assert "WAKE_WORD_UNAVAILABLE" not in stages

    # Everything after it ran, in order.
    for stage in ("RECORDING_STARTED", "RECORDING_STOPPED",
                  "AUDIO_BUFFER_RECEIVED", "TRANSCRIPTION_STARTED",
                  "TRANSCRIPTION_SUCCESS", "INTENT_CLASSIFIED",
                  "MODEL_REQUEST_STARTED", "MODEL_FIRST_FRAGMENT",
                  "MODEL_RESPONSE_COMPLETE", "TTS_STARTED", "TTS_COMPLETE"):
        assert stage in stages, f"{stage} never happened: {stages}"
    assert stages.index("WAKE_WORD_DETECTED") < stages.index("RECORDING_STARTED")
    assert stages.index("TRANSCRIPTION_SUCCESS") < stages.index("MODEL_REQUEST_STARTED")
    assert stages.index("MODEL_FIRST_FRAGMENT") < stages.index("TTS_STARTED")

    # The transcript reached the model unchanged.
    assert orchestrator.seen == [SPOKEN]
    assert model_server.asked, "the model was never asked anything"
    said_to_model = [m["content"] for m in model_server.asked[0]["messages"]
                     if m.get("role") == "user"]
    assert said_to_model == [SPOKEN], said_to_model

    # And the answer came back as whole sentences, out loud. Joined with a space,
    # because the chunker hands over each sentence already trimmed - "is four." and
    # "Anything else?" rather than one string with the spacing of the stream.
    assert " ".join(s.strip() for s in speaker.said) == "Two plus two is four. Anything else?"
    assert len(speaker.said) >= 2, speaker.said
    assert all(s.strip() for s in speaker.said), "an empty sentence was spoken"

    # The interface was told what was happening, and the user's words were shown.
    assert ("appendUserMessage", (SPOKEN,)) in api.pushed
    assert ("setHudState", ("listening",)) in api.pushed


async def test_the_trail_from_a_whole_turn_still_carries_no_words(model_server,
                                                                 monkeypatch):
    """A stage trail that recorded the transcript would be a transcript log.

    Checked after a real turn rather than on hand-made entries, because the
    stages that could leak are the ones the turn actually records.
    """
    import audio.setup as audio_setup
    import core.config_loader as loader
    from gui import api as gui_api

    settings = loader.get_settings()
    settings["ollama"]["host"] = model_server.host
    monkeypatch.setattr(loader, "get_settings", lambda: settings)
    monkeypatch.setattr("core.llm_client.get_settings", lambda: settings)
    monkeypatch.setattr(wake_word, "pyaudio", _PyAudio(_frames(_fixture("wake_"))))
    monkeypatch.setattr(audio_setup, "chosen_input_device", lambda: None)

    client = OllamaClient()
    try:
        with pytest.raises(_Stream.Exhausted):
            await asyncio.wait_for(
                gui_api._run_voice_loop(_Orchestrator(client, _Speaker()), _Api(),
                                        True, _Transcriber(SPOKEN)),
                timeout=120)
    finally:
        await client.close()

    blob = json.dumps(diagnostics.voice_trail()).lower()
    # Whole words, not substrings: the stage names themselves contain "is" twice
    # over ("listening", "transcription"), and a test that reads those as a leaked
    # transcript would fail on every correct trail forever. The distinctive words
    # are what matters, plus the phrase itself and the answer's.
    for word in ("two", "plus", "four", "what"):
        assert not re.search(rf"\b{word}\b", blob), f"the trail recorded '{word}'"
    assert SPOKEN not in blob
    assert "hey leti" not in blob
