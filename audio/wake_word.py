"""
Continuous wake-word detection using openWakeWord. Runs a microphone stream
through the wake-word model and fires a callback when the configured
wake word's confidence crosses the threshold. Used only in "continuous" mode;
push-to-talk mode bypasses this entirely.

WHICH MODEL A WAKE WORD NEEDS, AND WHETHER IT IS THERE

This file is the one authority on that, because it is the file that loads it.
It was not answering the question at all: the whole of it was

    Model(wakeword_models=[self.settings["wake_word"]])

which hands openWakeWord a bare string and hopes. Two separate things went
wrong with that, and the reported crash is only the first.

1. openWakeWord treats an entry in `wakeword_models` as a FILE PATH when
   os.path.exists() says so, and otherwise as the name of one of its six
   pretrained models - alexa, hey_mycroft, hey_jarvis, hey_rhasspy, timer,
   weather - matched as a substring after replacing spaces with underscores.
   Anything else raises

       ValueError: Could not find pretrained model for model name '...'

   "Hey Leti" is not one of those six. Neither is "hey_leti", which is what
   config/settings.yaml has shipped as the default - so this has never worked
   for anybody, and the reported error is the shipped configuration rather
   than something the user did.

2. Even a correct name fails on a clean machine. The pretrained files are NOT
   bundled with the package; openwakeword.utils.download_models() fetches them
   on first use. Asked for "hey_jarvis" before that has happened, openWakeWord
   resolves the name against a list of paths WITHOUT checking they exist and
   then fails to open the file. Verified here, all three cases:

       "Hey Leti"   -> ValueError: Could not find pretrained model
       "hey_leti"   -> ValueError: Could not find pretrained model
       "hey_jarvis" -> Could not open .../hey_jarvis_v0.1.tflite

   So fixing only the name would have left a clean Windows install just as
   broken, with a different error.

A custom model is a real, supported thing - it is the first branch above, a
path to a .tflite or .onnx file - and it still needs the melspectrogram and
embedding models that download_models() fetches, because every wake word is
scored on top of those. There is no "Hey Leti" model in this repository and
one cannot be written by hand: it has to be trained. `resolve` says so
precisely rather than falling back to something else and pretending.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np
import pyaudio
from openwakeword.model import Model

from core.config_loader import get_settings, resolve_path

logger = logging.getLogger("leti.wake_word")

CHUNK_SAMPLES = 1280  # openWakeWord expects 80ms chunks at 16kHz
SAMPLE_RATE = 16000

# Where a custom wake-word model goes. A folder rather than a setting, so
# dropping the file in is the whole of the installation and there is nothing to
# configure that can disagree with where the file actually is.
CUSTOM_MODEL_DIR = "./data/wake_words"
# openWakeWord reads both; tflite first because that is what it defaults to.
MODEL_SUFFIXES = (".tflite", ".onnx")

# How a wake word resolved. `kind` is the answer to "what is this", and every
# other field exists to be shown to a person who has to fix it.
PRETRAINED = "pretrained"          # one of openWakeWord's own, and on disk
NEEDS_DOWNLOAD = "needs_download"  # one of openWakeWord's own, not yet fetched
CUSTOM = "custom"                  # a model file in CUSTOM_MODEL_DIR
UNTRAINED = "untrained"            # no such pretrained model, and no file either


@dataclass
class Resolution:
    """What the configured wake word actually is, and where."""

    wake_word: str
    kind: str
    path: Optional[str] = None
    detail: str = ""
    repair: str = ""

    @property
    def ok(self) -> bool:
        """Usable right now, without anything being downloaded or trained."""
        return self.kind in (PRETRAINED, CUSTOM)


def custom_model_dir() -> Path:
    return resolve_path(CUSTOM_MODEL_DIR)


def pretrained_names() -> List[str]:
    """The names openWakeWord ships, read from the package rather than copied.

    A list written here would be a second source of truth that goes stale the
    moment openWakeWord adds or renames one.
    """
    try:
        import openwakeword

        return sorted(openwakeword.MODELS.keys())
    except Exception as e:                                 # pragma: no cover
        logger.warning(f"Couldn't read openWakeWord's model list ({e}).")
        return []


def _pretrained_path(name: str) -> Optional[str]:
    """The file openWakeWord would use for this name, whether or not it exists.

    Matched the same way openWakeWord matches it - substring, spaces to
    underscores - so this agrees with what the loader will actually do rather
    than with a guess about it.
    """
    try:
        import openwakeword

        wanted = name.replace(" ", "_")
        for path in openwakeword.get_pretrained_model_paths("tflite"):
            if wanted in os.path.basename(path):
                return path
    except Exception as e:                                 # pragma: no cover
        logger.warning(f"Couldn't look up pretrained model paths ({e}).")
    return None


def _custom_path(name: str) -> Optional[str]:
    """A model file the user supplied, if there is one.

    An explicit path in the setting is honoured first - somebody who wrote a
    path meant it - and otherwise the name is looked up in CUSTOM_MODEL_DIR.
    """
    direct = Path(name).expanduser()
    if direct.suffix.lower() in MODEL_SUFFIXES and direct.is_file():
        return str(direct)
    folder = custom_model_dir()
    for suffix in MODEL_SUFFIXES:
        candidate = folder / f"{name}{suffix}"
        if candidate.is_file():
            return str(candidate)
    return None


def resolve(wake_word: Optional[str] = None) -> Resolution:
    """What the configured wake word is, and whether it can be loaded now.

    Never raises and never guesses. The four answers are the four real cases,
    and each carries what a person would need to do about it.
    """
    if wake_word is None:
        try:
            wake_word = str(get_settings()["app"]["wake_word"])
        except Exception as e:
            return Resolution("", UNTRAINED,
                              detail=f"The wake word could not be read from the settings ({e}).",
                              repair="Set app.wake_word in config/settings.yaml.")
    wake_word = (wake_word or "").strip()
    if not wake_word:
        return Resolution("", UNTRAINED,
                          detail="No wake word is configured.",
                          repair="Set app.wake_word in config/settings.yaml.")

    custom = _custom_path(wake_word)
    if custom:
        return Resolution(wake_word, CUSTOM, path=custom,
                          detail=f"Using the custom model at {custom}.")

    pretrained = _pretrained_path(wake_word)
    if pretrained and os.path.exists(pretrained):
        return Resolution(wake_word, PRETRAINED, path=pretrained,
                          detail=f"Using openWakeWord's built-in '{wake_word}' model.")
    if pretrained:
        return Resolution(
            wake_word, NEEDS_DOWNLOAD, path=pretrained,
            detail=(f"'{wake_word}' is one of openWakeWord's own models but it has "
                    "not been downloaded yet."),
            repair="Leti downloads it on first launch; run the launcher again, or "
                   "run: python -m audio.wake_word --install")

    # .onnx rather than .tflite in the message, because onnx is the format that
    # runs everywhere: openWakeWord declares tflite-runtime only for Linux, so a
    # .tflite is unloadable on Windows. Both are still accepted - MODEL_SUFFIXES -
    # and this is the one to recommend.
    where = custom_model_dir() / f"{wake_word}.onnx"
    return Resolution(
        wake_word, UNTRAINED,
        detail=(f"There is no wake-word model for '{wake_word}'. openWakeWord's "
                f"built-in models are {', '.join(pretrained_names()) or 'unavailable'}, "
                "and no custom model of that name was found."),
        repair=(f"Train a '{wake_word}' model and put it at {where} - "
                "training/wake_word/README.md is the recipe, and it is how the "
                "model Leti ships with was made. Or set app.wake_word to one of "
                "the built-in names, which work immediately."))


def _load(found: "Resolution"):
    """Load the model, trying both of openWakeWord's runtimes. Returns (model, framework).

    openWakeWord ships every model in two formats and can run either, and which
    one works is a property of the machine rather than of the wake word: the
    tflite runtime is a compiled wheel, and on a machine where it does not match
    the installed numpy it fails at import of its own C extension with

        AttributeError: _ARRAY_API not found

    which is not a wake-word problem and cannot be fixed by re-downloading the
    model. Reproduced here on this container. Windows is the platform where
    tflite_runtime wheels are hardest to get, so falling back to the ONNX copy of
    the SAME model - which is already downloaded beside it - is the difference
    between voice working and voice not working, on exactly the machines this
    report came from.

    Not a second implementation: it is openWakeWord's own `inference_framework`
    argument, and the same Model class.
    """
    attempts: List[tuple] = []
    primary = found.path or ""
    if primary.endswith(".onnx"):
        candidates = [(primary, "onnx")]
    else:
        candidates = [(primary, "tflite")]
    sibling = (primary.replace(".tflite", ".onnx") if primary.endswith(".tflite")
               else primary.replace(".onnx", ".tflite"))
    if sibling != primary and os.path.exists(sibling):
        candidates.append((sibling, "onnx" if sibling.endswith(".onnx") else "tflite"))

    for path, framework in candidates:
        try:
            model = Model(wakeword_models=[path], inference_framework=framework)
            if framework != candidates[0][1]:
                logger.info(
                    f"The {candidates[0][1]} runtime could not load the wake-word "
                    f"model, so the {framework} copy was used instead.")
            return model, framework
        except Exception as e:
            attempts.append((path, framework, f"{type(e).__name__}: {e}"))

    tried = "; ".join(f"{fw} -> {why}" for _p, fw, why in attempts) or "no model file to try"
    raise WakeWordUnavailable(Resolution(
        found.wake_word, UNTRAINED, path=found.path,
        detail=f"openWakeWord could not load a model for '{found.wake_word}' ({tried}).",
        repair=("If both runtimes failed the model file may be damaged - delete "
                f"{found.path} and let Leti download it again. If only one failed, "
                "that runtime is not working on this machine and the other was "
                "already tried.")))


class WakeWordUnavailable(RuntimeError):
    """The wake word cannot be listened for, and why.

    Carries the Resolution so a caller can show the reason and the repair
    without re-deriving either. Raised instead of letting openWakeWord's own
    ValueError escape, because that one names a model file and says nothing
    about what the user should do.
    """

    def __init__(self, resolution: "Resolution") -> None:
        super().__init__(resolution.detail or "The wake word is unavailable.")
        self.resolution = resolution


def assets_present() -> bool:
    """Whether openWakeWord's shared models are on disk.

    Every wake word is scored on top of the melspectrogram and embedding models,
    including a custom one, so this is the question "can anything be detected at
    all" rather than "is this particular wake word installed".
    """
    try:
        import openwakeword

        folder = Path(openwakeword.__file__).parent / "resources" / "models"
        for feature in openwakeword.FEATURE_MODELS.values():
            name = feature["download_url"].split("/")[-1]
            if not (folder / name).is_file():
                return False
        return True
    except Exception as e:                                 # pragma: no cover
        logger.warning(f"Couldn't check openWakeWord's shared models ({e}).")
        return False


def install_assets(names: Optional[List[str]] = None,
                   download: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
    """Download openWakeWord's models and CHECK THEY ARRIVED. Never raises.

    The check is the point. openwakeword.utils.download_models() returns None
    whether it worked or not, and the failure this exists to prevent is a launch
    that reports voice as ready because a download function returned without
    complaining - and then raises "Could not open ...tflite" the first time
    somebody speaks.

    `names` limits it to particular wake words; the shared feature and VAD
    models are fetched regardless, because they are what everything else needs.
    """
    try:
        from openwakeword.utils import download_models
    except Exception as e:
        return {"ok": False, "error": f"openWakeWord is not installed ({e}).",
                "installed": []}

    try:
        (download or download_models)(**({"model_names": list(names)} if names else {}))
    except Exception as e:
        logger.warning(f"Downloading the wake-word models failed ({e}).")
        return {"ok": False, "error": str(e), "installed": []}

    if not assets_present():
        return {"ok": False,
                "error": ("The download reported success but openWakeWord's shared "
                          "models are still not on disk."),
                "installed": []}

    installed = [name for name in pretrained_names()
                 if (_pretrained_path(name) or "") and os.path.exists(_pretrained_path(name) or "")]
    return {"ok": True, "error": None, "installed": installed}


class WakeWordListener:
    @property
    def settings(self) -> Dict[str, Any]:
        """Read live so /settings edits apply without a restart (the wake-word model is loaded once at startup)."""
        return get_settings()["app"]

    def __init__(self, on_wake: Callable[[], Awaitable[None]],
                 resolution: Optional[Resolution] = None):
        """Load the wake-word model ONCE, here, and say clearly if it cannot be.

        It used to pass the configured string straight to openWakeWord, which
        raised ValueError naming a model file - an error about openWakeWord's
        internals, in a traceback, for a problem whose fix is a setting or a
        download. Resolving first means the caller is told which of the four
        real situations it is and what to do about it.

        The model is loaded from a PATH rather than a name, because by this point
        the path is known and openWakeWord treats an existing path as a model
        file directly. That is also what makes a custom model work at all.
        """
        self.on_wake = on_wake
        self.resolution = resolution or resolve()
        if not self.resolution.ok:
            raise WakeWordUnavailable(self.resolution)
        if not assets_present():
            raise WakeWordUnavailable(Resolution(
                self.resolution.wake_word, NEEDS_DOWNLOAD, path=self.resolution.path,
                detail=("openWakeWord's shared speech models are not downloaded, so "
                        "no wake word can be detected."),
                repair="Run the launcher again, or run: python -m audio.wake_word --install"))
        self.model, self.framework = _load(self.resolution)
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._running = False

    async def start(self) -> None:
        from audio.setup import chosen_input_device

        self._running = True
        # The same microphone the user picked during first-run setup. Listening for
        # the wake word on one device while transcribing from another is the kind of
        # split that looks like "Leti ignores me" on a machine with two inputs.
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SAMPLES,
            input_device_index=chosen_input_device(),
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


# --------------------------------------------------------------------------- #
# Being asked from outside the application
#
# launcher/bootstrap.py is standard library only - it is what runs when nothing
# is installed - so it cannot import this module. It runs it, with the
# interpreter it has just prepared, exactly as it does for core/model_setup.py.
# One authority on wake-word assets, asked across a process boundary.
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    """`python -m audio.wake_word --install` downloads and verifies the models.

    Also `--check`, which reports without changing anything - what the
    diagnostics panel asks, from a terminal.
    """
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if "--check" in args:
        found = resolve()
        print(f"  wake word: {found.wake_word or '(none configured)'}")
        print(f"  state:     {found.kind}")
        print(f"  {found.detail}")
        if found.repair:
            print(f"  fix:       {found.repair}")
        print(f"  shared models downloaded: {'yes' if assets_present() else 'no'}")
        return 0 if found.ok and assets_present() else 1

    if "--install" not in args:
        print("usage: python -m audio.wake_word [--install | --check]", file=sys.stderr)
        return 2

    if assets_present():
        print("  Wake-word models are already downloaded.")
    else:
        print("  Downloading the wake-word models - this is a few megabytes.")
        outcome = install_assets()
        if not outcome["ok"]:
            print(f"  Could not download the wake-word models: {outcome['error']}",
                  file=sys.stderr)
            return 1
        print("  Wake-word models are ready.")

    # Having the models is not the same as the CONFIGURED wake word working, and
    # saying so here is the difference between a launch that looks fine and one
    # that tells the truth before the user tries to talk to it. The word MISSING
    # is there on purpose: this is printed to the console a first launch scrolls
    # past, and "voice will start without a wake word" was too easy to read as
    # progress.
    found = resolve()
    if found.ok:
        where = f" ({found.path})" if found.kind == CUSTOM and found.path else ""
        print(f"  Wake-word model for '{found.wake_word}': ready{where}.")
        return 0
    print(f"  Wake-word model for '{found.wake_word}': MISSING.")
    print(f"  {found.detail}")
    if found.repair:
        print(f"  {found.repair}")
    print("  Leti will still hear you when you press the microphone button.")
    return 0        # not fatal: Leti runs, and says so in diagnostics


if __name__ == "__main__":
    raise SystemExit(main())
