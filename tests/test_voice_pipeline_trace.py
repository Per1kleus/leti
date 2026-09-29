"""Where the spoken pipeline actually stops, when it stops.

"I spoke and nothing happened" has nine or ten possible meanings - no
microphone, a microphone that will not open, a wake word with no model, audio
with no words in it, speech recognition failing, a model server that is not
running, a model that is not downloaded - and until these stages existed the
only way to tell them apart was a log the Windows launchers hide the moment
Leti's window appears.

Each stage is recorded where the work happens, so the LAST one recorded is the
place it stopped. These tests check that the trail is complete, that it stops
where the failure is, and that it never carries what somebody said.
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import diagnostics  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Read as TEXT rather than imported: audio/stt.py imports whisper at module
# scope, and whisper is an optional install that a machine without a microphone
# has no reason to have. The properties being checked are in the source.
STT_SOURCE = (PROJECT_ROOT / "audio" / "stt.py").read_text()


@pytest.fixture(autouse=True)
def clean_trail():
    diagnostics.reset_voice_trail()
    yield
    diagnostics.reset_voice_trail()


def _stages():
    return [entry["stage"] for entry in diagnostics.voice_trail()]


# --- The trail itself ---------------------------------------------------------

def test_every_stage_the_brief_asks_for_exists():
    for stage in ("MICROPHONE_DEVICE_DETECTED", "MICROPHONE_INITIALIZED",
                  "WAKE_WORD_MODEL_LOADED", "WAKE_WORD_LISTENING",
                  "WAKE_WORD_DETECTED", "RECORDING_STARTED", "RECORDING_STOPPED",
                  "AUDIO_BUFFER_RECEIVED", "TRANSCRIPTION_STARTED",
                  "TRANSCRIPTION_SUCCESS", "INTENT_CLASSIFIED",
                  "MODEL_REQUEST_STARTED", "MODEL_FIRST_FRAGMENT",
                  "MODEL_RESPONSE_COMPLETE", "TTS_STARTED", "TTS_COMPLETE"):
        assert stage in diagnostics.VOICE_STAGES, f"{stage} is not a known stage"


def test_the_last_stage_recorded_is_where_it_stopped():
    diagnostics.record_voice_stage("WAKE_WORD_DETECTED")
    diagnostics.record_voice_stage("RECORDING_STARTED")
    diagnostics.record_voice_stage("RECORDING_STOPPED", bytes=32000)
    diagnostics.record_voice_stage("TRANSCRIPTION_STARTED")
    diagnostics.record_voice_stage("TRANSCRIPTION_EMPTY", seconds=0.3)

    assert diagnostics.last_voice_stage() == "TRANSCRIPTION_EMPTY"
    assert "MODEL_REQUEST_STARTED" not in _stages(), \
        "it would look as though the model had been asked"


def test_recording_a_stage_never_raises_into_the_work():
    """Every call site is real work that must not fail because a diagnostic
    could not be written."""
    diagnostics.record_voice_stage(None)            # type: ignore[arg-type]
    diagnostics.record_voice_stage("FINE", unknown_field=object())
    assert diagnostics.last_voice_stage() is not None


def test_the_trail_is_bounded():
    for i in range(500):
        diagnostics.record_voice_stage("RECORDING_STARTED", count=i)
    assert len(diagnostics.voice_trail()) <= 60


# --- What it must never carry -------------------------------------------------

def test_a_stage_never_carries_what_was_said():
    """A transcript is the content of what somebody said. The LENGTH of one is a
    diagnostic; the words are not."""
    diagnostics.record_voice_stage(
        "TRANSCRIPTION_SUCCESS", chars=17,
        text="my bank password is hunter2",      # must be dropped
        transcript="also dropped", audio=b"\x00\x01")

    entry = diagnostics.voice_trail()[-1]
    assert entry["chars"] == 17
    assert "text" not in entry and "transcript" not in entry and "audio" not in entry
    assert "hunter2" not in repr(entry)


def test_the_allowed_fields_are_an_allowlist_not_a_denylist():
    """So a new call site cannot quietly add a field that turns out to hold
    speech."""
    import inspect

    source = inspect.getsource(diagnostics.record_voice_stage)
    assert "k in _SAFE_DETAIL" in source
    for banned in ("text", "transcript", "audio", "samples", "token", "password"):
        assert banned not in diagnostics._SAFE_DETAIL


def test_nothing_is_recorded_per_audio_frame():
    """The highest-frequency entry is one per utterance. A stage per frame would
    be eighty a second and would bury the trail it exists to make readable."""
    recording = STT_SOURCE[STT_SOURCE.index("async def record_until_silence"):]
    recording = recording[:recording.index("async def transcribe")]
    # One at the start and one at the end, and none inside the frame loop.
    assert recording.count("record_voice_stage") == 2, (
        "record_until_silence records a stage somewhere other than its two ends")


# --- Where the stages are actually recorded -----------------------------------

def test_transcription_records_which_of_the_three_outcomes_it_was():
    """Empty is not the same as failed, and neither is the same as success. They
    look identical from outside and have different fixes."""
    for stage in ("TRANSCRIPTION_STARTED", "TRANSCRIPTION_SUCCESS",
                  "TRANSCRIPTION_EMPTY", "TRANSCRIPTION_FAILED"):
        assert stage in STT_SOURCE, f"{stage} is never recorded"


def test_opening_the_microphone_records_both_answers():
    """A device being LISTED is not a device that opens."""
    opener = STT_SOURCE[STT_SOURCE.index("def _open_input_stream"):]
    opener = opener[:opener.index("def _transcribe_array")]
    assert opener.count("MICROPHONE_INITIALIZED") == 2, \
        "the failing case is not recorded, so a held microphone looks like no microphone"


def test_the_model_stages_are_recorded_by_the_orchestrator():
    import inspect

    from core import orchestrator

    source = inspect.getsource(orchestrator)
    for stage in ("INTENT_CLASSIFIED", "MODEL_REQUEST_STARTED",
                  "MODEL_FIRST_FRAGMENT", "MODEL_RESPONSE_COMPLETE", "MODEL_FAILED"):
        assert stage in source, f"{stage} is never recorded"


def test_speaking_is_recorded_once_per_answer_not_once_per_sentence():
    """A long reply is a dozen utterances."""
    import inspect

    from core.orchestrator import Orchestrator

    say = inspect.getsource(Orchestrator._say)
    assert "if not self._said_anything_this_turn:" in say
    assert say.index("_said_anything_this_turn") < say.index("TTS_STARTED")


# --- Does the spoken transcript actually reach the model? -------------------------

def test_the_voice_loop_hands_the_transcript_to_the_orchestrator():
    """The reported symptom was that spoken input appeared not to reach the model.
    This is the line that decides it: whatever speech recognition produced is what
    handle_user_input is called with, unchanged."""
    import inspect

    from gui import api

    loop = inspect.getsource(api._run_voice_loop)
    assert "await orchestrator.handle_user_input(text, voice_mode=True)" in loop
    # And the text came from the transcriber, not from anywhere else.
    assert "text, failed = await _transcribe(transcriber, audio)" in loop


def test_an_empty_transcript_is_not_sent_as_if_it_were_speech():
    """Sending "" to the model would produce an answer to nothing, which is worse
    than saying the microphone heard no words."""
    import inspect

    from gui import api

    handler = inspect.getsource(api._run_voice_loop)
    assert "classify_capture" in handler, \
        "an empty transcript is not classified, so it cannot be reported as empty"


def test_there_is_one_ollama_client_and_the_voice_path_uses_it():
    """No second model client. The voice path goes through the same orchestrator
    the typed path does, which holds the one client built in main.build_app."""
    import inspect

    from gui import api

    source = inspect.getsource(api)
    assert "OllamaClient(" not in source, "gui/api.py builds its own model client"

    loop = inspect.getsource(api._run_voice_loop)
    assert "orchestrator.handle_user_input" in loop
    assert "llm_client" not in loop, "the voice loop reaches past the orchestrator"


def test_there_is_one_speech_engine():
    """The transcriber is built once in run_gui_mode and passed in - loading
    Whisper twice is seconds of startup and a second copy in memory."""
    import inspect

    from gui import api

    source = inspect.getsource(api)
    assert source.count("load_voice_stack") <= 2, \
        "the voice stack is loaded from more than one place"
    assert "run_in_executor(None, load_voice_stack)" in source

    loop = inspect.getsource(api._run_voice_loop)
    assert "WhisperTranscriber(" not in loop, "the voice loop builds a second transcriber"


# --- 16, 17. Failure does not take the interface with it --------------------------

def test_a_missing_wake_word_model_does_not_kill_voice_or_the_interface():
    """It used to raise out of the voice loop into a bare `except Exception: log`,
    which killed voice while the interface still said it was ready - so the
    microphone appeared to work and Leti simply never answered.

    Push-to-talk needs no wake-word model, so that is what runs instead."""
    import inspect

    from gui import api

    loop = inspect.getsource(api._run_voice_loop)
    assert "except WakeWordUnavailable" in loop
    assert "WAKE_WORD_UNAVAILABLE" in loop
    assert 'api.push("setVoiceState", "no_wake_word")' in loop
    # and it falls through to the loop that does not need a wake word
    assert loop.index("except WakeWordUnavailable") < loop.index(
        "# Push to talk: no wake-word model needed")


def test_the_interface_is_never_told_voice_is_ready_when_the_wake_word_failed():
    import inspect

    from gui import api

    loop = inspect.getsource(api._run_voice_loop)
    failure = loop[loop.index("except WakeWordUnavailable"):
                   loop.index("# Push to talk")]
    assert '"ready"' not in failure, "it reports voice as ready after a failure"


def test_the_wake_word_model_is_loaded_once_rather_than_per_utterance():
    """Loading it per detection would put the model load - measured at 344ms for
    the ONNX runtime on this machine - between the wake word and the recording."""
    import inspect

    from audio import wake_word

    listener = inspect.getsource(wake_word.WakeWordListener)
    constructor = listener[:listener.index("async def start")]
    assert "_load(" in constructor, "the model is not loaded in the constructor"

    running = listener[listener.index("async def start"):]
    assert "_load(" not in running, "the model is reloaded while listening"
    assert "Model(" not in running, "a model is constructed inside the listen loop"


def test_the_listener_keeps_one_model_for_its_whole_life():
    import inspect

    from audio import wake_word

    import re

    source = inspect.getsource(wake_word.WakeWordListener)
    # Assigned as `self.model, self.framework = _load(...)`, so match the binding
    # rather than one spelling of it.
    assignments = re.findall(r"self\.model\s*[,=]", source)
    assert len(assignments) == 1, f"self.model is bound {len(assignments)} times"
