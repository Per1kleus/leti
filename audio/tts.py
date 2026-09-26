"""
Text-to-speech via pyttsx3 — a lightweight wrapper over the OS's native TTS
engine (SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak on Linux).
No model downloads or external binaries required, at the cost of a more
robotic voice than Piper/Coqui.

pyttsx3's `runAndWait()` is blocking, so it's run in a thread executor.
Interruption calls `engine.stop()`, which pyttsx3 supports for cutting off
mid-utterance from another thread.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import pyttsx3

from core.config_loader import get_settings

logger = logging.getLogger("leti.tts")


class Pyttsx3TTS:
    @property
    def settings(self) -> Dict[str, Any]:
        """Read live so /settings edits apply without a restart (rate/volume/voice are applied to the engine at startup; `interruptible` is read live)."""
        return get_settings()["tts"]

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", self.settings.get("rate", 180))
        self.engine.setProperty("volume", self.settings.get("volume", 1.0))

        voice_id = self.settings.get("voice_id", "")
        if voice_id:
            self.engine.setProperty("voice", voice_id)

        self._current_task: Optional[asyncio.Task] = None

    async def speak(self, text: str) -> bool:
        """Synthesize and play `text`, cancellable via `interrupt()`.

        Returns whether the words actually reached the speakers. That answer
        matters: Leti is voice-first, so an answer that was spoken is not also
        printed, and a `speak()` that quietly did nothing meant the user got no
        audio AND no text - the turn completed, the model had answered, and
        nothing whatsoever appeared. Saying so lets the caller put the text on
        screen instead (see gui/api.py's reply handler).

        An interruption returns False too, and that is correct: the user stopped
        it on purpose, and the caller's own stop handling owns what happens next.
        """
        if not text.strip():
            return False

        loop = asyncio.get_event_loop()
        self._current_task = loop.run_in_executor(None, self._speak_blocking, text)
        try:
            return bool(await self._current_task)
        except asyncio.CancelledError:
            logger.info("Speech interrupted.")
            return False

    def _speak_blocking(self, text: str) -> bool:
        """True when the engine actually spoke. Never raises into the executor."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except RuntimeError as e:
            # pyttsx3 raises if runAndWait is re-entered while already running;
            # this can happen if speak() is called again before the previous
            # utterance's executor thread has fully unwound. Streaming makes that
            # the normal case rather than a rare one - an answer arrives as a
            # series of utterances spoken back to back - so this used to be able
            # to swallow an entire reply one sentence at a time.
            logger.warning(f"TTS engine busy or interrupted: {e}")
            return False
        except Exception as e:
            # Anything else the platform's speech engine does: a SAPI voice that
            # is not installed, a driver that has gone away. Reported rather than
            # raised, because the executor would only turn it into an exception
            # the caller cannot act on - and False is exactly what the caller
            # needs to know.
            logger.warning(f"Speech failed ({type(e).__name__}: {e}).")
            return False

    def interrupt(self) -> None:
        """Stops current playback immediately, if interruptible mode is enabled."""
        if not self.settings.get("interruptible", True):
            return
        try:
            self.engine.stop()
        except Exception as e:
            logger.warning(f"Failed to interrupt TTS: {e}")

    def list_voices(self) -> list:
        """Utility to discover available voice IDs on this machine (see README)."""
        return [{"id": v.id, "name": v.name, "languages": v.languages} for v in self.engine.getProperty("voices")]

    def close(self):
        try:
            self.engine.stop()
        except Exception:
            pass
