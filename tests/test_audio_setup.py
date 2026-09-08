"""First-run audio permission: ask once, verify it works, remember the answer.

The thing worth protecting here isn't the prompt, it's what the prompt is FOR.
A microphone that opens and delivers silence looks exactly like one that works
unless the level is measured; a device the user picked is worth nothing if the
recorder opens a different one; and an answer that isn't remembered is a question
asked forever. Each of those has a test below because each was a real defect
during development - the device picker in the HUD did silently reset itself to
the system default, and it took a browser driving the real card to notice.

Audio hardware is stood in for throughout. This suite has to pass on a laptop
with a sound card and on a build machine with none, so a fake PyAudio is the only
way the accept path is testable at all.
"""
from __future__ import annotations

import asyncio
import json
import math
import struct
import sys
import types

import pytest


# --- A machine with a sound card, and one without ---------------------------------

class FakeStream:
    def __init__(self, loud=True, **kwargs):
        self.kwargs = kwargs
        self.loud = loud
        self.written = 0

    def read(self, frames, exception_on_overflow=False):
        out = bytearray()
        for i in range(frames):
            value = int(0.4 * 32767 * math.sin(2 * math.pi * 300 * i / 16000)) if self.loud else 2
            out += struct.pack("<h", value)
        return bytes(out)

    def write(self, data):
        self.written += len(data)

    def stop_stream(self):
        pass

    def close(self):
        pass


class FakePyAudio:
    DEVICES = [
        {"index": 0, "name": "Built-in Microphone", "maxInputChannels": 1,
         "maxOutputChannels": 0, "defaultSampleRate": 44100.0},
        {"index": 1, "name": "USB Headset Mic", "maxInputChannels": 2,
         "maxOutputChannels": 0, "defaultSampleRate": 48000.0},
        {"index": 2, "name": "Studio Display Speakers", "maxInputChannels": 0,
         "maxOutputChannels": 2, "defaultSampleRate": 44100.0},
    ]

    loud = True
    opened: list = []

    def get_device_count(self):
        return len(self.DEVICES)

    def get_device_info_by_index(self, index):
        return self.DEVICES[index]

    def get_default_input_device_info(self):
        return self.DEVICES[0]

    def get_default_output_device_info(self):
        return self.DEVICES[2]

    def open(self, **kwargs):
        FakePyAudio.opened.append(kwargs)
        return FakeStream(loud=FakePyAudio.loud, **kwargs)

    def terminate(self):
        pass


@pytest.fixture
def sound_card(monkeypatch, tmp_path):
    """A machine with two microphones and a speaker, and a scratch setup file."""
    FakePyAudio.opened = []
    FakePyAudio.loud = True
    module = types.ModuleType("pyaudio")
    module.paInt16 = 8
    module.PyAudio = lambda *a, **k: FakePyAudio()
    monkeypatch.setitem(sys.modules, "pyaudio", module)

    import audio.setup as audio_setup

    monkeypatch.setattr(audio_setup, "setup_path", lambda: tmp_path / "audio_setup.json")
    return audio_setup


@pytest.fixture
def no_sound_card(monkeypatch, tmp_path):
    module = types.ModuleType("pyaudio")
    module.paInt16 = 8

    def explode(*a, **k):
        raise OSError("No Default Input Device Available")

    module.PyAudio = explode
    monkeypatch.setitem(sys.modules, "pyaudio", module)

    import audio.setup as audio_setup

    monkeypatch.setattr(audio_setup, "setup_path", lambda: tmp_path / "audio_setup.json")
    return audio_setup


# --- Asking once ------------------------------------------------------------------

def test_a_machine_that_has_never_been_asked_is_not_configured(sound_card):
    assert sound_card.is_configured() is False
    assert sound_card.load_setup() is None


def test_an_answer_is_remembered(sound_card):
    sound_card.record_choice(mic_allowed=True, speakers_ok=True, device_index=1,
                             device_name="USB Headset Mic")
    assert sound_card.is_configured() is True
    assert sound_card.chosen_input_device() == 1


def test_a_decline_is_remembered_rather_than_re_asked(sound_card):
    """The point of asking once: 'no' has to stick, or the prompt is a nag."""
    sound_card.record_choice(mic_allowed=False, speakers_ok=False)
    assert sound_card.is_configured() is True
    assert sound_card.microphone_allowed() is False
    assert "chose not to" in sound_card.blocked_reason()


def test_not_yet_asked_is_not_the_same_as_declined(sound_card):
    """Three states, not two - only 'never asked' prompts, and it must not read as
    a refusal in the meantime."""
    assert sound_card.microphone_allowed() is True
    assert sound_card.blocked_reason() is None


def test_a_corrupt_file_means_ask_again_not_stay_silent(sound_card, tmp_path):
    (tmp_path / "audio_setup.json").write_text("{ this is not json")
    assert sound_card.load_setup() is None
    assert sound_card.is_configured() is False
    assert sound_card.microphone_allowed() is True


def test_the_voice_stack_honours_a_saved_decline(sound_card, monkeypatch):
    """Saying no has to actually stop Leti opening the microphone - not just be
    recorded somewhere."""
    import audio

    monkeypatch.setattr("audio.setup.blocked_reason", sound_card.blocked_reason)
    sound_card.record_choice(mic_allowed=False, speakers_ok=False)
    tts, transcriber, error = audio.load_voice_stack()
    assert (tts, transcriber) == (None, None)
    assert "chose not to" in error


# --- Checking that it actually works ----------------------------------------------

def test_devices_are_split_into_inputs_and_the_real_default_output(sound_card):
    devices = sound_card.list_devices()
    assert devices["available"] is True
    assert [d["name"] for d in devices["inputs"]] == ["Built-in Microphone", "USB Headset Mic"]
    assert devices["default_input"]["index"] == 0
    assert devices["default_output"]["name"] == "Studio Display Speakers"


def test_a_microphone_delivering_sound_is_reported_as_working(sound_card):
    result = sound_card.measure_microphone(1, seconds=0.2)
    assert result["ok"] and result["heard_sound"]
    assert result["peak"] > sound_card.SIGNAL_THRESHOLD


def test_a_microphone_delivering_silence_is_not_reported_as_working(sound_card):
    """The failure people actually hit: the stream opens, permission looks granted,
    and nothing arrives. Indistinguishable from success unless the level is measured."""
    FakePyAudio.loud = False
    result = sound_card.measure_microphone(0, seconds=0.2)
    assert result["ok"] is True          # opening worked...
    assert result["heard_sound"] is False  # ...but nothing came through
    assert result["peak"] < sound_card.SIGNAL_THRESHOLD


def test_the_measurement_opens_the_device_it_was_asked_for(sound_card):
    sound_card.measure_microphone(1, seconds=0.05)
    assert FakePyAudio.opened[-1]["input_device_index"] == 1


def test_no_device_preference_is_passed_through_as_the_system_default(sound_card):
    sound_card.measure_microphone(None, seconds=0.05)
    assert FakePyAudio.opened[-1]["input_device_index"] is None


def test_a_test_tone_is_actually_written_to_the_output(sound_card):
    assert sound_card.play_test_tone(seconds=0.05)["ok"] is True
    assert FakePyAudio.opened[-1]["output"] is True


def test_missing_hardware_is_returned_as_an_error_not_raised(no_sound_card):
    """'There is no sound card' is the normal case on a server. The caller's job is
    to say so, not to crash."""
    devices = no_sound_card.list_devices()
    assert devices["available"] is False and devices["inputs"] == []
    assert no_sound_card.measure_microphone(None)["ok"] is False
    assert no_sound_card.play_test_tone()["ok"] is False


# --- The recorder uses the microphone that was chosen -----------------------------

def test_the_transcriber_records_from_the_saved_device(sound_card, monkeypatch):
    """A device picker that the recorder ignores is worse than no picker: the check
    passes on one microphone and Leti then listens to another."""
    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda *a, **k: types.SimpleNamespace(
        transcribe=lambda *a, **k: {"text": ""})
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setattr("audio.setup.chosen_input_device", lambda: 1)

    for name in [m for m in list(sys.modules) if m == "audio.stt"]:
        del sys.modules[name]
    from audio.stt import WhisperTranscriber

    transcriber = WhisperTranscriber()
    assert transcriber.input_device_index == 1

    transcriber._open_input_stream()
    assert FakePyAudio.opened[-1]["input_device_index"] == 1


# --- The console flow -------------------------------------------------------------

def _answers(monkeypatch, *replies):
    queue = iter(replies)

    async def fake_read_line(prompt=""):
        try:
            return next(queue)
        except StopIteration:
            return None

    import core.console_input

    monkeypatch.setattr(core.console_input, "read_line", fake_read_line)


@pytest.mark.asyncio
async def test_declining_at_the_prompt_saves_a_decline(sound_card, monkeypatch, capsys):
    _answers(monkeypatch, "no")
    record = await sound_card.run_console_setup()
    assert record["microphone"]["allowed"] is False
    assert "won't ask again" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_accepting_records_the_chosen_device_and_the_measured_level(sound_card, monkeypatch):
    # allow -> pick device 2 (the USB headset) -> heard the tone
    _answers(monkeypatch, "yes", "2", "yes")
    record = await sound_card.run_console_setup()
    assert record["microphone"]["allowed"] is True
    assert record["microphone"]["device_index"] == 1          # menu position 2
    assert record["microphone"]["device_name"] == "USB Headset Mic"
    assert record["microphone"]["verified"] is True
    assert record["speakers"]["allowed"] is True


@pytest.mark.asyncio
async def test_pressing_enter_takes_the_system_default_microphone(sound_card, monkeypatch):
    _answers(monkeypatch, "yes", "", "yes")
    record = await sound_card.run_console_setup()
    assert record["microphone"]["device_index"] == 0


@pytest.mark.asyncio
async def test_a_silent_microphone_offers_a_retry_and_is_saved_unverified(sound_card, monkeypatch, capsys):
    FakePyAudio.loud = False
    # allow -> default device -> retry once -> give up retrying -> heard the tone
    _answers(monkeypatch, "yes", "", "yes", "no", "yes")
    record = await sound_card.run_console_setup()
    output = capsys.readouterr().out
    assert "Nothing came through" in output
    assert record["microphone"]["allowed"] is True
    assert record["microphone"]["verified"] is False


@pytest.mark.asyncio
async def test_not_hearing_the_tone_records_the_speakers_as_not_working(sound_card, monkeypatch):
    # allow -> default device -> didn't hear it -> don't retry
    _answers(monkeypatch, "yes", "", "no", "no")
    record = await sound_card.run_console_setup()
    assert record["speakers"]["allowed"] is False
    assert "chose not to let Leti use the speakers" in sound_card.blocked_reason()


@pytest.mark.asyncio
async def test_the_prompt_does_not_run_a_second_time(sound_card, monkeypatch):
    _answers(monkeypatch, "no")
    assert await sound_card.run_console_setup() is not None
    assert await sound_card.run_console_setup() is None, "asked twice"


@pytest.mark.asyncio
async def test_but_it_can_be_re_run_on_request(sound_card, monkeypatch):
    """A 'no' must not be a trap - /audio and the HUD's Audio button re-open it."""
    _answers(monkeypatch, "no")
    await sound_card.run_console_setup()
    _answers(monkeypatch, "yes", "", "yes")
    record = await sound_card.run_console_setup(force=True)
    assert record["microphone"]["allowed"] is True


@pytest.mark.asyncio
async def test_a_launch_with_nobody_to_ask_leaves_the_question_open(sound_card, monkeypatch, capsys):
    """A service, a launcher script or a piped stdin has no one at the keyboard.
    It must not block waiting - and it must not record a decline either, which
    would turn voice off forever for someone who was never actually asked."""
    _answers(monkeypatch)  # stdin ends immediately
    assert await sound_card.run_console_setup() is None
    assert sound_card.is_configured() is False, "nothing should have been saved"
    assert "ask again next time" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_stdin_ending_midway_also_saves_nothing(sound_card, monkeypatch):
    """Ctrl-D at the device menu is the same situation as Ctrl-D at the first
    question - half an answer isn't an answer."""
    _answers(monkeypatch, "yes")   # allows, then stdin ends at the device prompt
    assert await sound_card.run_console_setup() is None
    assert sound_card.is_configured() is False


@pytest.mark.asyncio
async def test_a_machine_with_no_audio_is_recorded_as_text_only(no_sound_card, monkeypatch):
    _answers(monkeypatch, "yes")
    record = await no_sound_card.run_console_setup()
    assert record["microphone"]["allowed"] is False
    assert "No audio system" in record["note"]


# --- End of input, once, is end of input for good ---------------------------------

def _detached_reader():
    """A reader bound to this loop but not to the real stdin pump.

    _ensure_started() starts a thread reading the process's actual stdin, which
    under pytest is closed or captured. The queue is the part under test, so the
    thread is dropped and the queue driven directly.
    """
    import core.console_input as console_input

    reader = console_input._ConsoleReader()
    # Bind by hand rather than via _ensure_started(), which would also spawn the
    # thread that reads the process's real stdin - under pytest that raises
    # "reading from stdin while output is captured".
    reader._loop = asyncio.get_running_loop()
    reader._queue = asyncio.Queue()
    reader._thread = object()   # non-None so nothing tries to start one
    return reader, reader._queue


@pytest.mark.asyncio
async def test_end_of_input_is_remembered_by_the_shared_reader():
    """The stdin pump publishes its EOF sentinel exactly once, so whoever read it
    first used to consume the only notice of it - and every later prompt then waited
    forever on a queue nothing would fill again.

    Adding the audio prompt made that reachable in ordinary use: `leti < script.txt`
    ran the prompt, which swallowed the EOF, and the chat loop afterwards hung
    instead of exiting. It was already reachable through /settings.
    """
    import core.console_input as console_input

    reader, queue = _detached_reader()

    # drain_stale=False: read_line() otherwise discards type-ahead before waiting,
    # which would include anything queued up in advance here.
    queue.put_nowait("only line")
    assert await reader.read_line(drain_stale=False) == "only line"

    queue.put_nowait(console_input.EOF)
    assert await reader.read_line(drain_stale=False) is None

    # The point: still None, immediately, rather than blocking forever.
    for _ in range(3):
        assert await asyncio.wait_for(reader.read_line("You: "), timeout=2.0) is None


@pytest.mark.asyncio
async def test_draining_does_not_discard_the_end_of_input():
    """drain() throws away type-ahead before showing a prompt. Throwing away an EOF
    along with it would reintroduce the hang by another route."""
    import core.console_input as console_input

    reader, queue = _detached_reader()
    queue.put_nowait("stale type-ahead")
    queue.put_nowait(console_input.EOF)

    reader.drain()
    assert await asyncio.wait_for(reader.read_line("You: "), timeout=2.0) is None
