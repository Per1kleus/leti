"""
Speech-to-text handler built on OpenAI's original `whisper` package. Simpler
dependency footprint than faster-whisper (pure PyTorch, no ctranslate2 build),
at the cost of somewhat slower inference on CPU.

Supports two capture modes, both funneling into the same WhisperTranscriber:
  - push_to_talk: caller records while a hotkey is held, then calls transcribe().
  - continuous: record_until_silence() auto-stops on a silence timeout, then transcribe().
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Callable, Optional

import numpy as np
import pyaudio
import whisper

from core.config_loader import get_settings

logger = logging.getLogger("leti.stt")

SAMPLE_RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16


class WhisperTranscriber:
    @property
    def settings(self) -> Dict[str, Any]:
        """Read live so /settings edits apply without a restart (the Whisper model itself is loaded once at startup)."""
        return get_settings()["stt"]

    def __init__(self):
        model_size = self.settings.get("model_size", "base.en")
        device = self.settings.get("device", "cpu")
        logger.info(f"Loading whisper model '{model_size}' on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        self.fp16 = self.settings.get("fp16", False)
        self._pa = pyaudio.PyAudio()

        # The microphone chosen during first-run setup, or None for the system
        # default. Read here rather than per-recording: switching input device
        # mid-session isn't a thing, and PyAudio takes the index at stream open.
        from audio.setup import chosen_input_device

        self.input_device_index = chosen_input_device()
        if self.input_device_index is not None:
            logger.info(f"Recording from input device index {self.input_device_index}.")

    def _open_input_stream(self):
        """One place that opens the microphone, so the chosen device applies to
        every capture path - push-to-talk and silence-detected alike."""
        return self._pa.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True,
            frames_per_buffer=CHUNK, input_device_index=self.input_device_index,
        )

    def _transcribe_array(self, audio_np: np.ndarray) -> str:
        result = self.model.transcribe(audio_np, fp16=self.fp16)
        return result.get("text", "").strip()

    async def record_while(self, should_continue: Callable[[], bool]) -> bytes:
        """Generic recorder: keeps capturing frames while should_continue() is True.
        Use this to wire up a real push-to-talk hotkey (press -> True, release -> False)."""
        stream = self._open_input_stream()
        frames = []
        loop = asyncio.get_event_loop()
        try:
            while should_continue():
                data = await loop.run_in_executor(None, lambda: stream.read(CHUNK, exception_on_overflow=False))
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()
        return b"".join(frames)

    async def record_until_silence(self, silence_timeout: Optional[float] = None) -> bytes:
        """Records continuously until `silence_timeout` seconds of near-silence is detected."""
        timeout = silence_timeout or self.settings.get("silence_timeout_seconds", 1.2)
        stream = self._open_input_stream()
        frames = []
        silence_chunks = 0
        silence_threshold = 500  # RMS amplitude below this counts as silence
        max_silence_chunks = int(timeout * SAMPLE_RATE / CHUNK)
        loop = asyncio.get_event_loop()

        try:
            while True:
                data = await loop.run_in_executor(None, lambda: stream.read(CHUNK, exception_on_overflow=False))
                frames.append(data)
                audio_np = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)) if len(audio_np) else 0

                if rms < silence_threshold:
                    silence_chunks += 1
                    if silence_chunks > max_silence_chunks and len(frames) > max_silence_chunks:
                        break
                else:
                    silence_chunks = 0
        finally:
            stream.stop_stream()
            stream.close()

        return b"".join(frames)

    async def transcribe(self, raw_audio: bytes) -> str:
        """Converts raw int16 PCM bytes to the float32 array whisper expects, then transcribes."""
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_array, audio_np)

    def close(self):
        self._pa.terminate()
