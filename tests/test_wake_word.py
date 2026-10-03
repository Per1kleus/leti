"""The wake word: which model it needs, whether it is there, and what happens when it is not.

The reported crash was

    ValueError: Could not find pretrained model for model name "Hey Leti"

raised from inside openwakeword/model.py during voice startup. Two separate
things were wrong and the error is only the first.

openWakeWord treats an entry in `wakeword_models` as a file PATH when
os.path.exists() says so, and otherwise as one of its six pretrained NAMES,
matched as a substring with spaces turned into underscores. Anything else raises
that ValueError. Verified against the installed package:

    "Hey Leti"   -> ValueError: Could not find pretrained model
    "hey_leti"   -> ValueError: Could not find pretrained model      <- the SHIPPED default
    "hey_jarvis" -> Could not open .../hey_jarvis_v0.1.tflite        <- a valid name!

So the shipped configuration had never worked for anybody, and fixing only the
name would have left a clean install broken with a different error: openWakeWord
does not ship its models at all, and nothing called download_models().

These tests do not require the models to be downloaded. The ones that would are
skipped when they are not.
"""
from __future__ import annotations

import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

pytest.importorskip("openwakeword", reason="openWakeWord is not installed")

from audio import wake_word  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A custom-model folder of our own, so no test reads the real data/."""
    folder = tmp_path / "wake_words"
    folder.mkdir()
    monkeypatch.setattr(wake_word, "custom_model_dir", lambda: folder)
    return folder


def _configure(monkeypatch, value):
    monkeypatch.setattr(wake_word, "get_settings",
                        lambda: {"app": {"wake_word": value}})


# --- 1, 3. What the configuration resolves to ------------------------------------

def test_a_built_in_name_resolves_to_openwakewords_own_model(workspace):
    found = wake_word.resolve("hey_jarvis")

    assert found.kind in (wake_word.PRETRAINED, wake_word.NEEDS_DOWNLOAD)
    assert found.path and found.path.endswith(".tflite")


def test_the_reported_wake_word_is_reported_as_untrained(workspace):
    """The exact value from the crash. Not a pretrained model, and no file - so
    the honest answer is that the model does not exist, with what it would take
    to make one."""
    found = wake_word.resolve("Hey Leti")

    assert found.kind == wake_word.UNTRAINED
    assert found.ok is False
    assert "no wake-word model" in found.detail
    assert "train" in found.repair.lower()
    assert str(workspace) in found.repair, "it does not say where to put the model"


def test_the_shipped_default_is_the_same_untrained_case(workspace):
    """hey_leti is not one of openWakeWord's six either. This is the finding that
    makes the bug a shipped-configuration bug rather than a user error."""
    assert wake_word.resolve("hey_leti").kind == wake_word.UNTRAINED


def test_the_built_in_names_are_read_from_the_package_not_copied():
    """A list written into Leti would go stale the moment openWakeWord changed
    one, and then Leti would reject a name that works."""
    import openwakeword

    assert wake_word.pretrained_names() == sorted(openwakeword.MODELS.keys())
    assert "hey_leti" not in wake_word.pretrained_names()


def test_an_empty_or_unreadable_wake_word_is_not_a_crash(workspace, monkeypatch):
    assert wake_word.resolve("").kind == wake_word.UNTRAINED
    assert wake_word.resolve("   ").kind == wake_word.UNTRAINED

    def explode():
        raise RuntimeError("no settings")

    monkeypatch.setattr(wake_word, "get_settings", explode)
    found = wake_word.resolve()
    assert found.kind == wake_word.UNTRAINED
    assert found.ok is False


# --- A custom model is a real, supported thing -----------------------------------

@pytest.mark.parametrize("suffix", [".tflite", ".onnx"])
def test_a_custom_model_file_is_found_and_used(workspace, suffix):
    """This is how 'Hey Leti' works the moment a model for it exists: a file in
    data/wake_words, named for the wake word."""
    model = workspace / f"hey_leti{suffix}"
    model.write_bytes(b"not a real model, but a real file")

    found = wake_word.resolve("hey_leti")

    assert found.kind == wake_word.CUSTOM
    assert found.ok is True
    assert found.path == str(model)


def test_a_custom_model_beats_a_built_in_name(workspace):
    """Somebody who put a file there meant it."""
    model = workspace / "hey_jarvis.tflite"
    model.write_bytes(b"mine")

    found = wake_word.resolve("hey_jarvis")

    assert found.kind == wake_word.CUSTOM
    assert found.path == str(model)


def test_an_explicit_path_in_the_setting_is_honoured(workspace, tmp_path):
    elsewhere = tmp_path / "somewhere-else.tflite"
    elsewhere.write_bytes(b"mine")

    found = wake_word.resolve(str(elsewhere))

    assert found.kind == wake_word.CUSTOM
    assert found.path == str(elsewhere)


def test_a_file_that_is_not_a_model_is_not_mistaken_for_one(workspace):
    (workspace / "hey_leti.txt").write_text("notes")

    assert wake_word.resolve("hey_leti").kind == wake_word.UNTRAINED


# --- 2, 4. What the listener does about it ----------------------------------------

def test_the_listener_refuses_with_a_reason_rather_than_openwakewords_valueerror(workspace):
    """The whole point. openWakeWord's ValueError names a model file and says
    nothing a person can act on; this says which of the four situations it is and
    what to do."""
    with pytest.raises(wake_word.WakeWordUnavailable) as raised:
        wake_word.WakeWordListener(on_wake=None,
                                   resolution=wake_word.resolve("Hey Leti"))

    found = raised.value.resolution
    assert found.kind == wake_word.UNTRAINED
    assert found.repair, "the refusal carries no repair"


def test_the_listener_refuses_when_the_shared_models_are_missing(workspace, monkeypatch):
    """Every wake word is scored on top of the melspectrogram and embedding
    models, including a custom one. Without them nothing can be detected, and a
    custom model file on disk does not change that."""
    model = workspace / "hey_leti.tflite"
    model.write_bytes(b"a file")
    monkeypatch.setattr(wake_word, "assets_present", lambda: False)

    with pytest.raises(wake_word.WakeWordUnavailable) as raised:
        wake_word.WakeWordListener(on_wake=None)

    assert raised.value.resolution.kind == wake_word.NEEDS_DOWNLOAD
    assert "install" in raised.value.resolution.repair


def test_a_model_file_openwakeword_cannot_load_is_reported_not_raised(workspace, monkeypatch):
    """A truncated download, a damaged file, a format this machine's runtime
    cannot read. Reported as a wake-word problem with a repair, rather than as
    whatever exception the runtime chose."""
    model = workspace / "hey_leti.tflite"
    model.write_bytes(b"not actually a tflite model")
    monkeypatch.setattr(wake_word, "assets_present", lambda: True)

    with pytest.raises(wake_word.WakeWordUnavailable) as raised:
        wake_word.WakeWordListener(on_wake=None)

    assert "could not load" in raised.value.resolution.detail.lower()


def test_both_runtimes_are_tried_before_giving_up(workspace, monkeypatch):
    """openWakeWord ships every model twice and can run either. Which one works is
    a property of the machine: the tflite runtime is a compiled wheel, it is not
    installed at all on Windows, and where it IS installed it can still fail to
    run against a mismatched numpy. Reproduced on this container, where tflite
    imports and then raises _ARRAY_API not found - which openWakeWord's own
    ImportError guard does not catch."""
    import inspect

    source = inspect.getsource(wake_word._load)
    assert "inference_framework=framework" in source
    assert '"onnx"' in source and '"tflite"' in source

    tried = []

    class Boom(Exception):
        pass

    def fake_model(wakeword_models, inference_framework):
        tried.append(inference_framework)
        raise Boom("no")

    monkeypatch.setattr(wake_word, "Model", fake_model)
    model = workspace / "hey_leti.tflite"
    model.write_bytes(b"x")
    (workspace / "hey_leti.onnx").write_bytes(b"x")

    with pytest.raises(wake_word.WakeWordUnavailable):
        wake_word._load(wake_word.resolve("hey_leti"))

    assert "tflite" in tried and "onnx" in tried, f"only tried {tried}"


# --- Downloading, and verifying it worked -----------------------------------------

def test_the_installer_checks_the_download_actually_arrived(monkeypatch):
    """download_models() returns None whether it worked or not. A launch that
    trusted that would report voice as ready and then fail on the first word."""
    monkeypatch.setattr(wake_word, "assets_present", lambda: False)

    outcome = wake_word.install_assets(download=lambda **kw: None)

    assert outcome["ok"] is False
    assert "still not on disk" in outcome["error"]


def test_the_installer_reports_a_download_that_raised(monkeypatch):
    def explode(**kwargs):
        raise OSError("no network")

    outcome = wake_word.install_assets(download=explode)

    assert outcome["ok"] is False
    assert "no network" in outcome["error"]


def test_the_installer_succeeds_when_the_models_are_there(monkeypatch):
    monkeypatch.setattr(wake_word, "assets_present", lambda: True)

    outcome = wake_word.install_assets(download=lambda **kw: None)

    assert outcome["ok"] is True
    assert outcome["error"] is None


# --- The command line the launcher uses -------------------------------------------

def test_the_command_line_needs_an_explicit_verb():
    assert wake_word.main([]) == 2
    assert wake_word.main(["--nonsense"]) == 2


def test_installing_is_not_fatal_when_the_wake_word_is_untrained(monkeypatch,
                                                                 capsys, tmp_path):
    """A launch must not stop because the wake word has no model. Leti opens,
    types, and listens when the microphone button is pressed.

    It must also SAY SO, in a word a person scrolling a first-launch console will
    notice. The earlier wording - "voice will start without a wake word" - was
    true and read like progress.
    """
    monkeypatch.setattr(wake_word, "assets_present", lambda: True)
    monkeypatch.setattr(wake_word, "custom_model_dir", lambda: tmp_path)
    _configure(monkeypatch, "hey_bartholomew")

    assert wake_word.main(["--install"]) == 0
    printed = capsys.readouterr().out
    assert "MISSING" in printed
    assert "hey_bartholomew" in printed
    assert "microphone button" in printed, "it does not say what still works"


# --- What the diagnostics panel says ----------------------------------------------

def test_the_panel_says_the_wake_word_failed_rather_than_that_voice_is_ready(
        workspace, monkeypatch):
    """The failure this prevents: wake-word initialisation raised into a bare
    `except Exception: log`, voice died, and the panel went on reporting a
    working microphone - so the user saw a healthy voice system that never
    answered."""
    from core import diagnostics

    _configure(monkeypatch, "hey_leti")
    monkeypatch.setattr(wake_word, "assets_present", lambda: True)

    verdict = diagnostics._check_wake_word()

    assert verdict["state"] == diagnostics.FAIL
    assert "hey_leti" in verdict["detail"]
    assert "train" in verdict["detail"].lower()
    assert verdict["expected_location"], "it does not say where the model should go"


def test_the_panel_says_when_the_shared_models_are_missing(workspace, monkeypatch):
    from core import diagnostics

    _configure(monkeypatch, "hey_jarvis")
    monkeypatch.setattr(wake_word, "assets_present", lambda: False)

    verdict = diagnostics._check_wake_word()

    assert verdict["state"] == diagnostics.FAIL
    assert "not downloaded" in verdict["detail"]
    assert "microphone button" in verdict["detail"], \
        "it does not say that Leti still works"


def test_the_panel_passes_when_a_custom_model_is_installed(workspace, monkeypatch):
    from core import diagnostics

    (workspace / "hey_leti.tflite").write_bytes(b"a model")
    _configure(monkeypatch, "hey_leti")
    monkeypatch.setattr(wake_word, "assets_present", lambda: True)

    verdict = diagnostics._check_wake_word()

    assert verdict["state"] == diagnostics.PASS
    assert verdict["model"].endswith("hey_leti.tflite")


def test_the_wake_word_check_never_raises_into_a_diagnostics_run(monkeypatch):
    """A diagnostics run that dies is the least useful possible outcome."""
    from core import diagnostics

    def explode(*args, **kwargs):
        raise RuntimeError("openWakeWord exploded")

    monkeypatch.setattr(wake_word, "resolve", explode)

    verdict = diagnostics._check_wake_word()

    assert verdict["state"] == diagnostics.NOT_AVAILABLE
    assert "could not be checked" in verdict["detail"]
