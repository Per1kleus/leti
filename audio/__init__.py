"""Leti's voice pipeline: wake word, speech-to-text, text-to-speech.

Every module in here needs hardware or a system package that a given machine may
not have - a microphone, a sound card, espeak on Linux, the whisper and pyaudio
packages. They import their dependencies at module scope, so importing them is
itself the thing that can fail, which is why load_voice_stack() below exists and
why callers use it instead of importing the classes directly.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple


def load_voice_stack() -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """(tts, transcriber, error) - the voice pipeline, or why there isn't one.

    Voice fails for ordinary reasons on real machines: no espeak on Linux, no
    microphone, no audio device at all in a VM or container, or the optional voice
    dependencies simply not installed. Callers decide what that means for them -
    GUI mode carries on in text-only, `--mode voice` cannot and says so - but
    neither should be deciding it from a traceback, so this returns the reason
    rather than raising it.

    Both halves are loaded together and returned together: half a voice pipeline
    (speaking with no way to answer, or listening with no way to reply) is worse
    than text, and confirmation needs both to ask a question and hear the answer.
    """
    try:
        from audio.stt import WhisperTranscriber
        from audio.tts import Pyttsx3TTS

        return Pyttsx3TTS(), WhisperTranscriber(), None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
