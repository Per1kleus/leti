"""The 'hey leti' model itself: that it exists, that it is real, and what it does.

The previous state of this project was that `app.wake_word` was "hey_leti" and no
model for it existed anywhere - audio/wake_word.py said so precisely and voice
fell back to push-to-talk. These tests are about the model that now ships, and
the first three of them are there because the tempting way to make this problem
go away is to rename hey_jarvis and claim it. They check that nothing of the kind
happened.

Everything here runs in Leti's own interpreter against the file that ships. None
of it needs the training environment, PyTorch, or a network: a model is a few
hundred kilobytes of ONNX and onnxruntime is already a runtime dependency.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import types
import wave
from pathlib import Path

import numpy as np
import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

pytest.importorskip("openwakeword", reason="openWakeWord is not installed")

from audio import wake_word                                   # noqa: E402
from core.config_loader import get_settings, resolve_path      # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
MODEL = PROJECT / "data" / "wake_words" / "hey_leti.onnx"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wake_word"

FRAME = wake_word.CHUNK_SAMPLES
RATE = wake_word.SAMPLE_RATE


def _wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == RATE, f"{path.name} is not 16 kHz"
        assert handle.getnchannels() == 1, f"{path.name} is not mono"
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def _fixtures(prefix: str):
    found = sorted(FIXTURES.glob(f"{prefix}*.wav"))
    assert found, f"no {prefix}*.wav fixtures in {FIXTURES}"
    return found


@pytest.fixture(scope="module")
def detector():
    """openWakeWord, loaded from the shipped file exactly as the runtime loads it."""
    from openwakeword.model import Model

    if not MODEL.is_file():
        pytest.skip("data/wake_words/hey_leti.onnx is not present")
    return Model(wakeword_models=[str(MODEL)], inference_framework="onnx")


# How many frames to push through after reset() before a score means anything.
#
# openWakeWord's reset() fills its 16-frame feature window with the embeddings of
# four seconds of RANDOM NOISE - np.random.randint, unseeded - so until real audio
# has pushed that out, every score is about noise nobody played, and a different
# one each time. Without this, ten byte-identical clips of silence scored
# differently and three of ten crossed the threshold.
WARMUP_FRAMES = 20


def _peak(model, samples: np.ndarray, warm_on_self: bool = False) -> float:
    """The highest score over one clip, from a warmed state.

    `warm_on_self` warms on the clip's own opening rather than on silence, for a
    sound that is simply present. Playing silence and then a steady tone measures
    the silence-to-tone transition: a window half silence and half sound has the
    shape of a word beginning, and one of the committed noise fixtures scores 0.9981
    that way against 0.0005 once the buffer holds the noise - which is what a room
    holds.
    """
    model.reset()
    if warm_on_self:
        used = min(WARMUP_FRAMES * FRAME, max(0, samples.size - FRAME))
        for start in range(0, used, FRAME):
            model.predict(samples[start:start + FRAME])
        samples = samples[used:]
    else:
        quiet = np.zeros(FRAME, dtype=np.int16)
        for _ in range(WARMUP_FRAMES):
            model.predict(quiet)
    best = 0.0
    for start in range(0, samples.size - FRAME + 1, FRAME):
        best = max(best, max(model.predict(samples[start:start + FRAME]).values()))
    return best


def configured_threshold() -> float:
    return float(get_settings()["app"].get("wake_word_threshold", 0.5))


# --- 1-3. It is a real model, and it is not a renamed pretrained one -------------

def test_the_model_ships_with_the_project():
    """Not just present on the machine that trained it.

    data/ is ignored wholesale because it is runtime state and personal data, and
    a model dropped in there would never have reached anybody. .gitignore
    un-ignores this one directory for that reason; this test is what notices if
    that line is ever removed.
    """
    assert MODEL.is_file(), f"{MODEL} is missing"
    ignored = subprocess.run(["git", "check-ignore", str(MODEL)],
                             cwd=PROJECT, capture_output=True, text=True)
    assert ignored.returncode != 0, (
        "the model is git-ignored, so it would not ship: "
        f"{ignored.stdout.strip()}")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(MODEL)],
                             cwd=PROJECT, capture_output=True, text=True)
    assert tracked.returncode == 0, "the model is not tracked by git"


def test_the_model_is_a_real_onnx_graph_of_the_expected_shape():
    """Parsed rather than trusted.

    An openWakeWord model takes a window of speech embeddings - 16 frames of 96
    features for a two-second window - and returns one score. Anything else is
    not a wake-word model, whatever it is called.
    """
    if not MODEL.is_file():
        pytest.skip("the model is not present")
    import onnxruntime

    session = onnxruntime.InferenceSession(str(MODEL),
                                           providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    assert len(inputs) == 1 and len(outputs) == 1
    assert list(inputs[0].shape)[-2:] == [16, 96], inputs[0].shape
    assert list(outputs[0].shape)[-1] == 1, outputs[0].shape


def test_the_model_is_not_a_copy_of_any_pretrained_model():
    """The substitution this whole exercise exists to not make.

    Compared by content, so a rename, a copy or a re-save does not get past it.
    """
    if not MODEL.is_file():
        pytest.skip("the model is not present")
    import openwakeword

    ours = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    folder = Path(openwakeword.__file__).parent / "resources" / "models"
    theirs = {}
    for path in sorted(folder.glob("*")):
        if path.is_file():
            theirs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert ours not in theirs.values(), (
        "the shipped model is byte-identical to "
        f"{[n for n, d in theirs.items() if d == ours]}")
    # And it does not merely differ by a byte: the sizes of openWakeWord's own
    # wake-word models are a narrow band, and a model of a completely different
    # size would be a different kind of thing altogether.
    assert 10_000 < MODEL.stat().st_size < 10_000_000, MODEL.stat().st_size


# --- 4-6. The one architecture finds it, loads it, and says which runtime --------

def test_the_shipped_wake_word_resolves_to_the_shipped_file():
    """What config/settings.yaml asks for is what is on disk.

    No monkeypatching: this reads the real setting and the real directory, which
    is the only way to catch the two disagreeing.
    """
    if not MODEL.is_file():
        pytest.skip("the model is not present")
    found = wake_word.resolve()

    assert found.wake_word == "hey_leti"
    assert found.kind == wake_word.CUSTOM
    assert found.ok is True
    assert Path(found.path).resolve() == MODEL.resolve()


def test_the_custom_model_directory_is_the_one_the_resolver_looks_in():
    """One place, agreed on by the resolver and by where the file actually is."""
    assert wake_word.custom_model_dir().resolve() == MODEL.parent.resolve()
    assert resolve_path(wake_word.CUSTOM_MODEL_DIR).resolve() == MODEL.parent.resolve()


def test_openwakeword_loads_it_through_the_onnx_runtime(detector):
    """The call the report asked for, made for real.

    Model(wakeword_models=[...], inference_framework="onnx"). ONNX rather than
    tflite because openWakeWord declares tflite-runtime only for Linux, so on
    Windows there is no tflite runtime to load it with.
    """
    assert detector.models, "openWakeWord loaded no model"
    name = list(detector.models.keys())[0]
    assert "hey_leti" in name, name


def test_loading_the_model_needs_no_tflite_runtime():
    """Proved by taking tflite_runtime away and loading it anyway.

    This is the Windows case: the wheel does not exist there. Leti's own loader
    is used, so this also covers audio/wake_word._load choosing the ONNX copy.
    """
    if not MODEL.is_file():
        pytest.skip("the model is not present")
    script = (
        "import sys\n"
        "class Blocked:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] in ('tflite_runtime', 'tensorflow'):\n"
        "            raise ImportError('blocked for this test: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocked())\n"
        "import types\n"
        "sys.modules.setdefault('chromadb', types.ModuleType('chromadb'))\n"
        "from audio import wake_word\n"
        "found = wake_word.resolve()\n"
        "model, framework = wake_word._load(found)\n"
        "assert framework == 'onnx', framework\n"
        "assert 'tflite_runtime' not in sys.modules\n"
        "print('loaded via', framework)\n"
    )
    done = subprocess.run([sys.executable, "-c", script], cwd=PROJECT,
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "loaded via onnx" in done.stdout


# --- 7-10. What it hears ----------------------------------------------------------

def test_it_wakes_on_the_wake_word(detector):
    """Deterministic audio, committed to the repository.

    There is no microphone in the environment these tests run in, so the fixtures
    are speech from voices that were held out of training - the same held-out half
    the validation set was drawn from. A fixture that the model had been trained
    on would prove nothing.
    """
    threshold = configured_threshold()
    peaks = {path.name: _peak(detector, _wav(path)) for path in _fixtures("wake_")}
    missed = {name: round(score, 3) for name, score in peaks.items()
              if score <= threshold}
    assert not missed, f"did not wake at {threshold}: {missed}"


def test_it_stays_quiet_through_silence(detector):
    assert _peak(detector, np.zeros(5 * RATE, dtype=np.int16)) <= configured_threshold()


def test_it_stays_quiet_through_speech_that_is_not_the_wake_word(detector):
    """Half the phrase, phonetic neighbours, the name in an ordinary sentence.

    These are the fixtures that matter: a model that wakes on "hey" or on
    "I asked Leti to open the file" is worse than no wake word, because the
    person cannot talk near the machine.
    """
    threshold = configured_threshold()
    peaks = {path.name: _peak(detector, _wav(path)) for path in _fixtures("quiet_")}
    fired = {name: round(score, 3) for name, score in peaks.items()
             if score > threshold}
    assert not fired, f"woke at {threshold} on: {fired}"


def test_it_stays_quiet_through_background_noise(detector):
    threshold = configured_threshold()
    peaks = {path.name: _peak(detector, _wav(path), warm_on_self=True)
             for path in _fixtures("noise_")}
    fired = {name: round(score, 3) for name, score in peaks.items()
             if score > threshold}
    assert not fired, f"woke at {threshold} on steady noise: {fired}"


def test_a_steady_sound_starting_suddenly_is_a_known_and_bounded_fault(detector):
    """Leti can wake once when a fan or a compressor cuts in, and that is measured.

    A window that is half silence and half sound has the shape of a word beginning,
    so the onset of a sustained tone produces a transient. It is one spurious wake
    per onset rather than a continuous failure - the same clip, once it is simply
    playing, scores about 0.0005 - and training/wake_word/validation.json carries the
    rate as a condition of its own.

    This test pins the DIFFERENCE rather than either number, because that is the
    claim: the fault is in the onset and not in the sound.
    """
    quiet = np.zeros(int(1.5 * RATE), dtype=np.int16)
    for path in _fixtures("noise_"):
        noise = _wav(path)
        steady = _peak(detector, noise, warm_on_self=True)
        assert steady <= configured_threshold(), (
            f"{path.name} wakes Leti while simply playing, at {steady:.4f}")
        onset = _peak(detector, np.concatenate([quiet, noise]))
        assert onset >= steady, (
            f"{path.name}: onset {onset:.4f} below steady {steady:.4f}, so the "
            "transient this documents is not where it was measured")


def test_the_configured_threshold_is_the_one_validation_chose():
    """The number in config/settings.yaml is not a guess.

    training/wake_word/validation.json records the measurements and the threshold
    they justify. If somebody changes the setting without re-measuring, this says
    so.
    """
    import json

    report = PROJECT / "training" / "wake_word" / "validation.json"
    if not report.is_file():
        pytest.skip("no validation report")
    chosen = json.loads(report.read_text())["threshold"]["chosen"]
    assert configured_threshold() == pytest.approx(chosen), (
        f"settings say {configured_threshold()}, validation chose {chosen}")


# --- 11-13. What it costs ---------------------------------------------------------

def test_the_model_loads_quickly_enough_to_be_loaded_at_startup():
    """Loaded once, during startup, so this is a startup cost and is measured.

    Generous bound: the point is to catch a model that takes seconds, not to
    police the millisecond on a shared machine.
    """
    if not MODEL.is_file():
        pytest.skip("the model is not present")
    from openwakeword.model import Model

    started = time.perf_counter()
    Model(wakeword_models=[str(MODEL)], inference_framework="onnx")
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"took {elapsed:.2f} s to load"


def test_scoring_a_frame_is_far_faster_than_the_frame_is_long(detector):
    """80 ms of audio must cost well under 80 ms to score, or it cannot keep up.

    This is the whole performance question for a wake word: it runs on every frame
    for as long as Leti is listening.
    """
    samples = _wav(_fixtures("wake_")[0])
    detector.reset()
    frames = [samples[i:i + FRAME] for i in range(0, samples.size - FRAME + 1, FRAME)]
    for frame in frames[:5]:                       # warm the runtime up
        detector.predict(frame)
    started = time.perf_counter()
    rounds = 0
    for _ in range(3):
        for frame in frames:
            detector.predict(frame)
            rounds += 1
    per_frame = (time.perf_counter() - started) / rounds
    assert per_frame < 0.020, f"{per_frame*1000:.1f} ms per 80 ms frame"


def test_the_model_is_loaded_once_and_not_per_frame():
    """A reload inside the listening loop would be the obvious way to ruin this."""
    import inspect

    source = inspect.getsource(wake_word.WakeWordListener)
    loop = source.split("while self._running")[1]
    for forbidden in ("Model(", "_load(", "resolve(", "assets_present("):
        assert forbidden not in loop, f"{forbidden} is inside the listening loop"
    assert "self.model, self.framework = _load" in source


def test_scoring_frames_forever_does_not_grow(detector):
    """A leak here is unbounded: this runs for as long as Leti is open.

    Measured as resident memory across five thousand frames - about seven minutes
    of listening - after a warm-up, so the allocation that happens on the first
    few calls is not counted as growth.
    """
    frame = np.zeros(FRAME, dtype=np.int16)
    for _ in range(200):
        detector.predict(frame)

    def rss() -> int:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

    if not os.path.exists("/proc/self/statm"):
        pytest.skip("no /proc on this platform")
    before = rss()
    for _ in range(5000):
        detector.predict(frame)
    grew = rss() - before
    assert grew < 20_000_000, f"resident memory grew {grew} bytes over 5000 frames"


def test_listening_starts_no_thread_or_process_of_its_own():
    """"No extra background processes or threads" is a property of the source.

    The listening loop hands each frame to the event loop's DEFAULT executor -
    run_in_executor(None, ...) - which is the pool everything else already uses.
    A Thread, a Process or an executor of its own would be a permanent worker
    added for the wake word alone.
    """
    source = Path(wake_word.__file__).read_text(encoding="utf-8")
    for forbidden in ("threading.Thread", "multiprocessing",
                      "ThreadPoolExecutor", "ProcessPoolExecutor"):
        assert forbidden not in source, f"audio/wake_word.py uses {forbidden}"
    assert "run_in_executor(None," in source or "run_in_executor(\n" in source


def test_audio_is_scored_one_frame_at_a_time():
    """Incremental, not buffered.

    openWakeWord keeps its own rolling state, so a frame at a time is both the
    cheapest way to run it and the only way to detect the wake word as it is
    said rather than after a pause.
    """
    import inspect

    loop = inspect.getsource(wake_word.WakeWordListener.start)
    assert "self._stream.read(CHUNK_SAMPLES" in loop
    assert "self.model.predict" in loop
    assert wake_word.CHUNK_SAMPLES == 1280, "openWakeWord expects 80 ms at 16 kHz"
    assert wake_word.SAMPLE_RATE == 16000


# --- 14-16. One architecture, and honest reporting --------------------------------

def test_there_is_only_one_wake_word_model_directory():
    """No second place for a model to hide.

    The failure this prevents is two directories, one of which has the model and
    the other of which the resolver looks in.
    """
    out = subprocess.run(
        ["git", "grep", "-l", "-E", r"wake_words?/|wakeword_models"],
        cwd=PROJECT, capture_output=True, text=True)
    files = [line for line in out.stdout.splitlines() if line]
    offenders = [f for f in files
                 if f.endswith(".py") and not f.startswith("tests/")
                 and f not in ("audio/wake_word.py",)
                 and not f.startswith("training/")]
    assert not offenders, f"these also name a wake-word model location: {offenders}"
    assert wake_word.CUSTOM_MODEL_DIR == "./data/wake_words"


def test_there_is_only_one_detector_and_one_microphone_loop():
    """Counted in the source, because the brief for this work forbids a second."""
    # Imports, not mentions: core/diagnostics.py and launcher/bootstrap.py both
    # explain in prose why openWakeWord behaves as it does, and prose is not a
    # second detector.
    out = subprocess.run(["git", "grep", "-l", "-E",
                          r"^\s*(import openwakeword|from openwakeword)", "--", "*.py"],
                         cwd=PROJECT, capture_output=True, text=True)
    users = {line for line in out.stdout.splitlines() if line}
    runtime = {f for f in users
               if not f.startswith(("tests/", "training/", "scripts/"))}
    assert runtime == {"audio/wake_word.py"}, (
        f"openWakeWord is imported outside audio/wake_word.py: {sorted(runtime)}")

    builds = subprocess.run(["git", "grep", "-l", "wakeword_models", "--", "*.py"],
                            cwd=PROJECT, capture_output=True, text=True)
    constructors = {f for f in builds.stdout.splitlines() if f
                    and not f.startswith(("tests/", "training/", "scripts/"))}
    assert constructors == {"audio/wake_word.py"}, (
        f"a wake-word model is constructed outside audio/wake_word.py: "
        f"{sorted(constructors)}")

    # Three files open an audio stream, and they are three different jobs:
    # wake_word.py listens continuously, stt.py records one utterance, and
    # setup.py runs the microphone and speaker check during first-run setup and
    # the /audio command. None of them is a second listening loop, and a fourth
    # would be.
    opens = subprocess.run(["git", "grep", "-l", "paInt16", "--", "*.py"],
                           cwd=PROJECT, capture_output=True, text=True)
    streams = {f for f in opens.stdout.splitlines() if f
               and not f.startswith(("tests/", "training/", "scripts/"))}
    assert streams == {"audio/wake_word.py", "audio/stt.py", "audio/setup.py"}, \
        sorted(streams)

    listening = subprocess.run(["git", "grep", "-l", "-E",
                                r"while self\._running", "--", "audio/*.py"],
                               cwd=PROJECT, capture_output=True, text=True)
    loops = {f for f in listening.stdout.splitlines() if f}
    assert loops == {"audio/wake_word.py"}, sorted(loops)


def test_diagnostics_reports_the_wake_word_as_working():
    """The panel must say PASS now, and name the file it found.

    It said FAIL for as long as there was no model, which was correct then. A
    panel that still said FAIL would be the same lie in the other direction.
    """
    if not MODEL.is_file():
        pytest.skip("the model is not present")
    from core import diagnostics

    check = diagnostics._check_wake_word()
    assert check["state"] == diagnostics.PASS, check
    assert check.get("wake_word") == "hey_leti"
    assert "hey_leti.onnx" in str(check.get("model", "")), check


def test_first_launch_says_the_model_is_ready_rather_than_guessing(tmp_path, monkeypatch):
    """`python -m audio.wake_word --install` is what the launcher asks.

    It has to distinguish three things: the shared models missing, the configured
    wake word having no model, and everything being fine. This is the third, and
    it has to name the wake word rather than say "voice ready".
    """
    if not MODEL.is_file():
        pytest.skip("the model is not present")
    done = subprocess.run([sys.executable, "-m", "audio.wake_word", "--check"],
                          cwd=PROJECT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "hey_leti" in done.stdout
    assert "custom" in done.stdout
    assert "shared models downloaded: yes" in done.stdout


def test_the_untrained_message_is_still_there_for_a_wake_word_with_no_model(tmp_path,
                                                                           monkeypatch):
    """Shipping one model must not have removed the honest answer for the others.

    Somebody who sets app.wake_word to their own name still needs to be told that
    the model does not exist and what it would take.
    """
    monkeypatch.setattr(wake_word, "custom_model_dir", lambda: tmp_path)
    found = wake_word.resolve("hey_bartholomew")

    assert found.kind == wake_word.UNTRAINED
    assert found.ok is False
    assert "train" in found.repair.lower()
    assert str(tmp_path) in found.repair


def test_the_model_needs_no_network_to_load(detector):
    """Local, offline, for good. A wake word that phones home is not local-first."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("the wake-word model tried to open a socket")

    original = socket.socket
    socket.socket = refuse                                     # type: ignore[assignment]
    try:
        from openwakeword.model import Model

        model = Model(wakeword_models=[str(MODEL)], inference_framework="onnx")
        model.predict(np.zeros(FRAME, dtype=np.int16))
    finally:
        socket.socket = original                               # type: ignore[assignment]


def test_the_fixtures_are_small_enough_to_live_in_the_repository():
    """A guard on the repository rather than on the model.

    Committing test audio is the right call - there is no microphone here - but it
    is also the easy way to put a hundred megabytes in a git history.
    """
    total = sum(p.stat().st_size for p in FIXTURES.glob("*.wav"))
    assert total < 3_000_000, f"{total} bytes of fixtures"
    assert os.path.isdir(FIXTURES)
