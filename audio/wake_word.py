"""
Continuous wake-word detection using openWakeWord. Runs a microphone stream
through the wake-word model and fires a callback when the configured
wake word's confidence crosses the threshold. Used only in "continuous" mode;
push-to-talk mode bypasses this entirely.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Awaitable, Callable

import numpy as np
import pyaudio
from openwakeword.model import Model

from core.config_loader import get_settings

logger = logging.getLogger("leti.wake_word")

CHUNK_SAMPLES = 1280  # openWakeWord expects 80ms chunks at 16kHz
SAMPLE_RATE = 16000


class WakeWordListener:
    @property
    def settings(self) -> Dict[str, Any]:
        """Read live so /settings edits apply without a restart (the wake-word model is loaded once at startup)."""
        return get_settings()["app"]

    def __init__(self, on_wake: Callable[[], Awaitable[None]]):
        self.on_wake = on_wake
        self.model = Model(wakeword_models=[self.settings["wake_word"]])
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SAMPLES,
        )
        logger.info(f"Wake word listener active for '{self.settings['wake_word']}'")

        loop = asyncio.get_event_loop()
        threshold = self.settings.get("wake_word_threshold", 0.5)

        while self._running:
            audio_chunk = await loop.run_in_executor(
                None, lambda: self._stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            )
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            predictions = await loop.run_in_executor(None, self.model.predict, audio_array)

            for wake_word_name, score in predictions.items():
                if score > threshold:
                    logger.info(f"Wake word detected: '{wake_word_name}' ({score:.2f})")
                    await self.on_wake()
                    self.model.reset()

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
