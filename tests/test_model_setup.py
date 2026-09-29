"""The first-launch model check: hardware in, a recommendation out, nothing
written until the user says so.

The tests that matter most here are the negative ones. This is the only part of
Leti that downloads gigabytes and rewrites which model runs, so what it must not
do - not on a failed detection, not on a failed download, not without an answer,
and never twice - is the specification.
"""
from __future__ import annotations


import pytest

from core import model_setup


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the setup file and the override file at a temp dir, so no test can
    touch the real data/ or config/ directory."""
    monkeypatch.setattr(model_setup, "setup_path", lambda: tmp_path / "model_setup.json")
    written = {}

    def fake_write(model):
        written["model"] = model

    monkeypatch.setattr(model_setup, "_write_reasoning_model", fake_write)
    return tmp_path, written


GOOD_GPU = {
    "ok": True, "error": None, "os": "Windows 11",
    "cpu": {"model": "Ryzen 5", "physical_cores": 6, "logical_cores": 12},
    "ram_gib": 16.0, "free_disk_gib": 400.0,
    "gpu": {"name": "NVIDIA GeForce RTX 4060", "vram_gib": 12.0}, "gpu_detected": True,
}


# --- 1 & 11. When the screen appears, and when it stops -----------------------------

def test_first_launch_is_not_configured(isolated):
    assert model_setup.is_configured() is False


def test_answering_marks_it_done_so_it_never_asks_again(isolated):
    model_setup.save_setup({"choice": "keep_current", "model": "qwen2.5:7b"})

    assert model_setup.is_configured() is True
    assert model_setup.load_setup()["choice"] == "keep_current"


def test_an_unreadable_setup_file_asks_again_rather_than_guessing(isolated):
    tmp_path, _ = isolated
    (tmp_path / "model_setup.json").write_text("{ this is not json")

    assert model_setup.is_configured() is False


# --- 2. Hardware detection ---------------------------------------------------------

def test_hardware_detection_reports_this_machine():
    hw = model_setup.detect_hardware()

    assert hw["ok"] is True, hw.get("error")
    assert hw["ram_gib"] and hw["ram_gib"] > 0
    assert hw["cpu"]["logical_cores"] >= 1
    assert "gpu" in hw and "free_disk_gib" in hw


def test_a_gpu_is_read_from_nvidia_smi(monkeypatch):
    class Out:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 4060, 12288\n"

    monkeypatch.setattr(model_setup.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(model_setup.subprocess, "run", lambda *a, **k: Out())

    assert model_setup._nvidia_gpu() == {"name": "NVIDIA GeForce RTX 4060", "vram_gib": 12.0}


@pytest.mark.parametrize("boom", [FileNotFoundError("no nvidia-smi"), OSError("denied")])
def test_a_broken_gpu_query_is_unknown_not_an_exception(monkeypatch, boom):
    monkeypatch.setattr(model_setup.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def explode(*a, **k):
        raise boom

    monkeypatch.setattr(model_setup.subprocess, "run", explode)
    assert model_setup._nvidia_gpu() is None


# --- 3. Recommendations ------------------------------------------------------------

def test_a_12gb_card_is_recommended_the_model_that_fits_beside_the_kv_cache():
    """14B weighs 8.4 GiB and would fit on weights alone. Its KV cache at 24k does
    not, which is the whole reason this check exists."""
    advice = model_setup.recommend(GOOD_GPU, num_ctx=24576)

    assert advice["recommended"] == "qwen2.5:7b"
    assert advice["confident"] is True
    fourteen = [c for c in advice["considered"] if c["label"] == "14B"][0]
    assert fourteen["fits"] is False
    assert "CPU" in fourteen["why_not"]


def test_a_big_card_gets_a_bigger_model():
    hw = {**GOOD_GPU, "gpu": {"name": "RTX 4090", "vram_gib": 24.0}}

    assert model_setup.recommend(hw, num_ctx=24576)["recommended"] == "qwen2.5:14b"


def test_a_smaller_context_window_lets_a_bigger_model_fit():
    """Proof the KV cache is doing the work rather than the weights: the same card
    and the same 8.4 GiB of weights, and only the context window changed."""
    assert model_setup.recommend(GOOD_GPU, num_ctx=24576)["recommended"] == "qwen2.5:7b"
    assert model_setup.recommend(GOOD_GPU, num_ctx=2048)["recommended"] == "qwen2.5:14b"


def test_no_gpu_recommends_the_smallest_model_and_says_it_is_unsure():
    hw = {**GOOD_GPU, "gpu": None, "gpu_detected": False}

    advice = model_setup.recommend(hw, num_ctx=24576)

    assert advice["recommended"] == "qwen2.5:3b"
    assert advice["confident"] is False
    assert any("No NVIDIA GPU" in r for r in advice["reasons"])


def test_a_tiny_card_falls_back_to_the_smallest_rather_than_nothing():
    hw = {**GOOD_GPU, "gpu": {"name": "GTX 1050", "vram_gib": 2.0}}

    advice = model_setup.recommend(hw, num_ctx=24576)

    assert advice["recommended"] == "qwen2.5:3b"
    assert advice["confident"] is False


def test_a_full_disk_rules_a_model_out():
    hw = {**GOOD_GPU, "free_disk_gib": 3.0}

    advice = model_setup.recommend(hw, num_ctx=24576)

    assert advice["confident"] is False, "recommended a model there is no room to store"
    # The models that would otherwise have fitted are the ones the disk rules out;
    # the oversized ones are still reported as not fitting VRAM.
    small = [c for c in advice["considered"] if c["label"] in ("3B", "7B")]
    assert all(not c["fits"] and "disk" in c["why_not"] for c in small)


# --- 10. Detection failure ---------------------------------------------------------

def test_unreadable_hardware_recommends_nothing_at_all():
    advice = model_setup.recommend({"ok": False, "error": "psutil missing"}, num_ctx=24576)

    assert advice["recommended"] is None
    assert advice["confident"] is False
    assert any("could not be read" in r for r in advice["reasons"])


def test_detection_failure_does_not_crash_detect_hardware(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    hw = model_setup.detect_hardware()

    assert hw["ok"] is False
    assert hw["error"]


# --- 5. The existing-model option --------------------------------------------------

@pytest.mark.asyncio
async def test_keeping_the_current_models_writes_no_configuration(isolated, monkeypatch):
    _, written = isolated
    monkeypatch.setattr(model_setup, "current_model", lambda: "qwen2.5:14b")

    async def must_not_run(*a, **k):
        raise AssertionError("a model was downloaded despite keeping the current ones")

    monkeypatch.setattr(model_setup, "pull_model", must_not_run)

    result = await model_setup.apply_choice("keep_current")

    assert result["ok"] is True
    assert written == {}, "the configuration was written"
    assert model_setup.is_configured() is True
    assert model_setup.load_setup()["model"] == "qwen2.5:14b"


# --- 4, 6, 7, 8. Accepting the recommendation --------------------------------------

@pytest.mark.asyncio
async def test_an_approved_model_is_downloaded_then_configured(isolated, monkeypatch):
    _, written = isolated
    order = []
    tags = ["qwen2.5:14b"]

    async def fake_installed():
        return list(tags)

    async def fake_pull(model):
        order.append("pull")
        tags.append(model)
        return {"ok": True, "error": None}

    monkeypatch.setattr(model_setup, "installed_models", fake_installed)
    monkeypatch.setattr(model_setup, "pull_model", fake_pull)
    monkeypatch.setattr(model_setup, "_write_reasoning_model",
                        lambda m: (order.append("configure"), written.update(model=m)))

    result = await model_setup.apply_choice("recommended", "qwen2.5:7b")

    assert result["ok"] is True
    assert written["model"] == "qwen2.5:7b"
    assert order == ["pull", "configure"], "configured before the download finished"
    assert model_setup.load_setup()["choice"] == "recommended"


@pytest.mark.asyncio
async def test_a_model_already_installed_is_not_downloaded_again(isolated, monkeypatch):
    _, written = isolated

    async def fake_installed():
        return ["qwen2.5:7b", "nomic-embed-text:latest"]

    async def must_not_run(*a, **k):
        raise AssertionError("re-downloaded a model that was already installed")

    monkeypatch.setattr(model_setup, "installed_models", fake_installed)
    monkeypatch.setattr(model_setup, "pull_model", must_not_run)

    result = await model_setup.apply_choice("recommended", "qwen2.5:7b")

    assert result["ok"] is True
    assert result["downloaded"] is False
    assert written["model"] == "qwen2.5:7b"


def test_a_bare_model_name_matches_ollamas_latest_tag():
    assert model_setup._is_installed("qwen2.5", ["qwen2.5:latest"]) is True
    assert model_setup._is_installed("qwen2.5:7b", ["qwen2.5:latest"]) is False


@pytest.mark.asyncio
async def test_nothing_is_written_without_a_choice(isolated):
    _, written = isolated

    result = await model_setup.apply_choice("recommended", None)

    assert result["ok"] is False
    assert written == {}
    assert model_setup.is_configured() is False, "an unanswered screen was marked done"


# --- 9. Download failure -----------------------------------------------------------

@pytest.mark.asyncio
async def test_a_failed_download_leaves_the_configuration_alone(isolated, monkeypatch):
    _, written = isolated

    async def fake_installed():
        return ["qwen2.5:14b"]

    async def failing_pull(model):
        return {"ok": False, "error": "connection reset"}

    monkeypatch.setattr(model_setup, "installed_models", fake_installed)
    monkeypatch.setattr(model_setup, "pull_model", failing_pull)

    result = await model_setup.apply_choice("recommended", "qwen2.5:7b")

    assert result["ok"] is False
    assert "connection reset" in result["error"]
    assert written == {}, "configuration was changed by a failed download"
    assert model_setup.is_configured() is False, "a failed setup was marked complete"


@pytest.mark.asyncio
async def test_a_pull_that_leaves_nothing_behind_does_not_configure(isolated, monkeypatch):
    """Ollama said yes and the model still is not there. Pointing Leti at it would
    be worse than the download having failed outright."""
    _, written = isolated

    async def fake_installed():
        return ["qwen2.5:14b"]

    async def lying_pull(model):
        return {"ok": True, "error": None}

    monkeypatch.setattr(model_setup, "installed_models", fake_installed)
    monkeypatch.setattr(model_setup, "pull_model", lying_pull)

    result = await model_setup.apply_choice("recommended", "qwen2.5:7b")

    assert result["ok"] is False
    assert written == {}
    assert model_setup.is_configured() is False


@pytest.mark.asyncio
async def test_a_failed_configuration_write_is_reported_not_raised(isolated, monkeypatch):
    async def fake_installed():
        return ["qwen2.5:7b"]

    def explode(_model):
        raise OSError("read-only file system")

    monkeypatch.setattr(model_setup, "installed_models", fake_installed)
    monkeypatch.setattr(model_setup, "_write_reasoning_model", explode)

    result = await model_setup.apply_choice("recommended", "qwen2.5:7b")

    assert result["ok"] is False
    assert "read-only" in result["error"]
    assert model_setup.is_configured() is False


# --- The console screen ------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_screen_is_skipped_once_it_has_been_answered(isolated, monkeypatch):
    model_setup.save_setup({"choice": "keep_current", "model": "qwen2.5:7b"})

    async def must_not_run():
        raise AssertionError("the screen ran again after being answered")

    monkeypatch.setattr(model_setup, "gather", must_not_run)

    assert await model_setup.run_console_setup() is None


@pytest.mark.asyncio
async def test_no_one_at_the_keyboard_changes_nothing_and_stays_unanswered(isolated, monkeypatch):
    """stdin ended. Swapping models unattended is not a decision to take on a
    guess, so the current setup stands AND the question is still open."""
    _, written = isolated

    async def fake_gather():
        return {"configured": False, "hardware": GOOD_GPU,
                "recommendation": model_setup.recommend(GOOD_GPU, num_ctx=24576)}

    monkeypatch.setattr(model_setup, "gather", fake_gather)
    # Something other than the recommendation, so the prompt is actually reached.
    monkeypatch.setattr(model_setup, "current_model", lambda: "qwen2.5:14b")
    monkeypatch.setattr("core.console_input.read_line", lambda *_a, **_k: _none())

    assert await model_setup.run_console_setup() is None
    assert written == {}
    assert model_setup.is_configured() is False


async def _none():
    return None


def test_the_screen_text_names_the_hardware_and_the_reason():
    state = {"hardware": GOOD_GPU,
             "recommendation": model_setup.recommend(GOOD_GPU, num_ctx=24576)}

    text = "\n".join(model_setup.describe(state))

    assert "RTX 4060" in text and "12.0 GiB VRAM" in text
    assert "Ryzen 5" in text
    assert "qwen2.5:7b" in text
    assert "KV cache" in text, "the reason the bigger model was ruled out is not shown"


# --- The numbers in settings.yaml are this module's numbers -------------------------

def test_settings_yaml_quotes_the_arithmetic_this_module_computes():
    """config/settings.yaml tells the user which model their card can run.

    It is the first thing anyone reads when choosing a model, and it had been
    computed at a num_ctx of 24,576 - two windows ago - so it advised a 16GB card
    to use the 14b. At the configured window the 14b needs 14.35 GiB against a
    13.7 GiB budget, and Ollama answers that by moving layers onto the CPU rather
    than by refusing. Wrong advice that looks authoritative, so it is checked.
    """
    import pathlib
    import re

    import yaml

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "config" / "settings.yaml").read_text(encoding="utf-8")
    num_ctx = int(yaml.safe_load(text)["ollama"]["num_ctx"])

    # "  #   qwen2.5:7b    4.4 GiB weights +  1.53 GiB KV =  6.63 GiB"
    claims = dict()
    for model, weights, kv, total in re.findall(
            r"(qwen2\.5:\d+b)\s+([\d.]+) GiB weights \+\s+([\d.]+) GiB KV =\s+([\d.]+) GiB", text):
        claims[model] = (float(weights), float(kv), float(total))
    assert claims, "settings.yaml no longer states the per-model VRAM arithmetic"

    for candidate in model_setup.CANDIDATES:
        name = candidate["model"]
        if name not in claims:
            continue
        weights, kv, total = claims[name]
        assert weights == candidate["weights_gib"], f"{name} weights"
        assert kv == model_setup._kv_gib(candidate, num_ctx), f"{name} KV cache at {num_ctx}"
        assert total == model_setup._needs_gib(candidate, num_ctx), f"{name} total"

    # And the budgets the same comment quotes, from the same reserves.
    for vram, budget in ((8, 5.7), (12, 9.7), (16, 13.7), (24, 21.7)):
        computed = round(vram - model_setup._VRAM_RESERVED_GIB
                         - model_setup._VRAM_HEADROOM_GIB, 2)
        assert computed == budget, f"{vram}GB budget is {computed}, not the {budget} claimed"
        assert f"{vram}GB -> {budget}" in text, f"settings.yaml no longer states the {vram}GB budget"


# --- Which models are required, and keeping them downloaded -------------------------
#
# The launcher asks these rather than naming a model, because a model name in the
# launcher is a second answer to a question this module owns - wrong from the
# first time anybody edits settings.yaml, and unable to say which of the four
# model settings even matter.

@pytest.fixture
def configured(monkeypatch):
    """A model configuration, without reading config/settings.yaml."""
    settings = {"ollama": {
        "reasoning_model": "qwen2.5:7b",
        "embedding_model": "nomic-embed-text",
        "vision_model": "llama3.2-vision",
        "fallback_reasoning_model": "mistral-nemo",
    }}
    monkeypatch.setattr(model_setup, "get_settings", lambda: settings)
    return settings


def test_the_required_models_come_from_the_configuration(configured):
    names = [entry["model"] for entry in model_setup.required_models()]
    assert names == ["qwen2.5:7b", "nomic-embed-text"]


def test_changing_the_configuration_changes_what_is_required(configured):
    configured["ollama"]["reasoning_model"] = "qwen2.5:3b"
    assert model_setup.required_models()[0]["model"] == "qwen2.5:3b"


def test_the_large_optional_models_are_not_required(configured):
    """Eight gigabytes of vision model on a first launch, for a feature nobody has
    asked for yet, is not a first launch anybody would wait through. The fallback
    exists for when the primary is missing, so fetching it alongside the primary
    is fetching a spare for a car that is not broken."""
    essential = [entry["model"] for entry in model_setup.required_models()]
    assert "llama3.2-vision" not in essential
    assert "mistral-nemo" not in essential

    everything = [entry["model"] for entry in model_setup.required_models(include_optional=True)]
    assert "llama3.2-vision" in everything
    assert "mistral-nemo" in everything


def test_a_role_left_blank_is_not_reported_as_a_model_called_nothing(configured):
    configured["ollama"]["embedding_model"] = "   "
    assert [e["model"] for e in model_setup.required_models()] == ["qwen2.5:7b"]


def test_a_configuration_that_cannot_be_read_requires_nothing(monkeypatch):
    def explode():
        raise RuntimeError("no settings file")

    monkeypatch.setattr(model_setup, "get_settings", explode)
    assert model_setup.required_models() == []


def test_nothing_is_downloaded_before_the_first_launch_screen_is_answered(monkeypatch):
    """This module's promise, in the docstring at the top of the file: declining
    leaves the configuration byte for byte as it was. Pre-fetching the default
    model before the screen has run would spend several gigabytes on a model the
    user is about to replace."""
    monkeypatch.setattr(model_setup, "is_configured", lambda: False)
    assert model_setup.may_pull_unattended("reasoning_model") is False


def test_the_embedding_model_is_fetched_whatever_the_screen_says(monkeypatch):
    """The screen only ever sets reasoning_model, so no answer to it can change
    which embedding model is right - and memory does not work without one."""
    monkeypatch.setattr(model_setup, "is_configured", lambda: False)
    assert model_setup.may_pull_unattended("embedding_model") is True


def test_once_the_screen_is_answered_the_configuration_is_kept_downloaded(monkeypatch):
    monkeypatch.setattr(model_setup, "is_configured", lambda: True)
    assert model_setup.may_pull_unattended("reasoning_model") is True


# --- A state file is not evidence ---------------------------------------------------

@pytest.mark.asyncio
async def test_a_model_the_state_file_claims_but_ollama_lacks_is_missing(
        configured, monkeypatch):
    """The rule from the brief: do not blindly trust a state file if the actual
    model is missing. data/model_setup.json records what was CHOSEN, which is a
    different question from what is on the disk now - `ollama rm` and an
    interrupted download both leave the record saying it is there."""
    monkeypatch.setattr(model_setup, "installed_models",
                        _async(["nomic-embed-text:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable", _async(True))

    missing = await model_setup.missing_models()

    assert [entry["model"] for entry in missing] == ["qwen2.5:7b"]
    assert missing[0]["reason"] == "not installed"


@pytest.mark.asyncio
async def test_a_model_that_lists_but_cannot_load_counts_as_missing(configured, monkeypatch):
    """A download interrupted partway leaves a manifest with blobs missing, and a
    corrupted blob lists exactly like a good one. /api/tags cannot tell them apart
    and /api/show can, because it resolves the blobs."""
    monkeypatch.setattr(model_setup, "installed_models",
                        _async(["qwen2.5:7b", "nomic-embed-text:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable",
                        _async_by(lambda model: model != "qwen2.5:7b"))

    missing = await model_setup.missing_models()

    assert [entry["model"] for entry in missing] == ["qwen2.5:7b"]
    assert missing[0]["reason"] == "installed but unusable"


@pytest.mark.asyncio
async def test_an_unreachable_ollama_is_not_read_as_nothing_installed(configured, monkeypatch):
    """"I could not ask" is not "nothing is there". Pulling four models on the
    strength of a refused connection is the wrong way to be wrong."""
    monkeypatch.setattr(model_setup, "installed_models", _async([]))

    assert await model_setup.missing_models() == []


# --- Repairing ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_missing_model_is_downloaded_and_then_verified(configured, monkeypatch):
    monkeypatch.setattr(model_setup, "is_configured", lambda: True)
    monkeypatch.setattr(model_setup, "installed_models",
                        _async(["nomic-embed-text:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable", _async(True))
    pulled = []

    async def pull(model, progress=None):
        pulled.append(model)
        return {"ok": True, "error": None}

    monkeypatch.setattr(model_setup, "pull_model", pull)

    result = await model_setup.ensure_models()

    assert pulled == ["qwen2.5:7b"]
    assert result["ok"] is True
    assert result["pulled"] == ["qwen2.5:7b"]


@pytest.mark.asyncio
async def test_a_download_that_reports_success_but_leaves_nothing_is_a_failure(
        configured, monkeypatch):
    """Exactly the state this function exists to get out of. Reporting it as fixed
    would hide it until the first question the user asked."""
    monkeypatch.setattr(model_setup, "is_configured", lambda: True)
    monkeypatch.setattr(model_setup, "installed_models",
                        _async(["nomic-embed-text:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable", _async(False))
    monkeypatch.setattr(model_setup, "pull_model", _async({"ok": True, "error": None}, kw=True))

    result = await model_setup.ensure_models()

    assert result["ok"] is False
    assert result["failed"][0]["model"] == "qwen2.5:7b"
    assert "cannot be loaded" in result["failed"][0]["error"]


@pytest.mark.asyncio
async def test_nothing_already_present_is_downloaded_again(configured, monkeypatch):
    """Later launches must not re-download. The check is what Ollama has, so this
    holds however many times Leti is started."""
    monkeypatch.setattr(model_setup, "is_configured", lambda: True)
    monkeypatch.setattr(model_setup, "installed_models",
                        _async(["qwen2.5:7b", "nomic-embed-text:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable", _async(True))

    async def pull(model, progress=None):
        raise AssertionError(f"re-downloaded {model}")

    monkeypatch.setattr(model_setup, "pull_model", pull)

    result = await model_setup.ensure_models()

    assert result["pulled"] == []
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_an_unanswered_first_launch_defers_the_reasoning_model_only(
        configured, monkeypatch):
    monkeypatch.setattr(model_setup, "is_configured", lambda: False)
    monkeypatch.setattr(model_setup, "installed_models", _async(["something:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable", _async(True))
    pulled = []

    async def pull(model, progress=None):
        pulled.append(model)
        return {"ok": True, "error": None}

    monkeypatch.setattr(model_setup, "pull_model", pull)

    result = await model_setup.ensure_models()

    assert pulled == ["nomic-embed-text"]
    assert [entry["model"] for entry in result["skipped"]] == ["qwen2.5:7b"]


@pytest.mark.asyncio
async def test_keeping_the_current_model_downloads_it_when_it_is_not_there(
        isolated, configured, monkeypatch):
    """"Keep what I have" is not the same as "I have it". Answering this screen
    used to save the answer and never ask again, so a machine with nothing
    downloaded failed on its first question with a model error."""
    monkeypatch.setattr(model_setup, "installed_models", _async(["something:latest"]))
    pulled = []

    async def pull(model, progress=None):
        pulled.append(model)
        return {"ok": True, "error": None}

    monkeypatch.setattr(model_setup, "pull_model", pull)

    result = await model_setup.apply_choice("keep_current")

    assert pulled == ["qwen2.5:7b"]
    assert result["ok"] is True
    assert model_setup.is_configured()


@pytest.mark.asyncio
async def test_keeping_a_model_that_cannot_be_downloaded_is_not_recorded(
        isolated, configured, monkeypatch):
    """Not saved means the next launch asks again, rather than leaving Leti
    pointed at a model that is not there."""
    monkeypatch.setattr(model_setup, "installed_models", _async(["something:latest"]))
    monkeypatch.setattr(model_setup, "pull_model",
                        _async({"ok": False, "error": "no disk space"}, kw=True))

    result = await model_setup.apply_choice("keep_current")

    assert result["ok"] is False
    assert not model_setup.is_configured(), "a failed download was recorded as done"


# --- The progress bar ----------------------------------------------------------------

def test_the_bar_is_drawn_from_the_numbers_it_is_given():
    line = model_setup.progress_bar("qwen2.5:7b", 2_350_000_000, 4_700_000_000)
    assert "qwen2.5:7b" in line
    assert "50.0%" in line
    assert line.count("#") == model_setup.BAR_WIDTH // 2


def test_the_bar_does_not_divide_by_a_total_it_does_not_have():
    """Ollama sends status lines with no total on them - "pulling manifest" and
    "verifying sha256" carry no bytes at all."""
    assert "starting" in model_setup.progress_bar("qwen2.5:7b", 0, 0)
    assert "%" not in model_setup.progress_bar("qwen2.5:7b", 0, 0)


def test_the_bar_never_runs_past_its_own_width():
    """A resumed download can report more completed than total."""
    line = model_setup.progress_bar("m", 9_000, 4_000)
    assert line.count("#") == model_setup.BAR_WIDTH
    assert "100.0%" in line


def test_a_progress_display_that_throws_never_fails_a_download():
    """It is a display. Nothing it does may cost somebody a three gigabyte
    download they have already waited for."""
    import inspect

    source = inspect.getsource(model_setup.pull_model)
    assert "except Exception:" in source
    assert "# A progress display is never allowed to fail a download." in source


def test_the_command_line_only_downloads_and_never_chooses():
    """Choosing a model is a question for a person and it has a screen. The
    command line is the part that needs no answer."""
    assert model_setup.main([]) == 2
    assert model_setup.main(["--pick", "qwen2.5:32b"]) == 2


# --- helpers -------------------------------------------------------------------------

def _async(value, kw=False):
    async def f(*args, **kwargs):
        return value
    return f


def _async_by(fn):
    async def f(model, *args, **kwargs):
        return fn(model)
    return f


# --- What the diagnostics panel says about a model that is not usable ---------------
#
# "Configured" was the whole of the old answer and it is not one. A model that was
# never downloaded, one whose download was interrupted, and one too big for the
# card all read as configured - and all three look to the user like Leti simply not
# answering, which is the reported symptom.

def test_the_model_check_says_when_the_model_was_never_downloaded(configured, monkeypatch):
    from core import diagnostics

    monkeypatch.setattr(model_setup, "installed_models", _async(["something-else:latest"]))
    verdict = diagnostics._check_model(reach_out=True)

    assert verdict["state"] == diagnostics.FAIL
    assert "not downloaded" in verdict["detail"]
    assert "ollama pull qwen2.5:7b" in verdict["detail"], "no way to act on it"


def test_the_model_check_says_when_the_file_is_damaged(configured, monkeypatch):
    """A download interrupted partway leaves a manifest with blobs missing, and it
    lists exactly like a good model."""
    from core import diagnostics

    monkeypatch.setattr(model_setup, "installed_models",
                        _async(["qwen2.5:7b", "nomic-embed-text:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable", _async(False))
    verdict = diagnostics._check_model(reach_out=True)

    assert verdict["state"] == diagnostics.FAIL
    assert "cannot be loaded" in verdict["detail"]


def test_the_model_check_passes_when_the_model_is_really_there(configured, monkeypatch):
    from core import diagnostics

    monkeypatch.setattr(model_setup, "installed_models",
                        _async(["qwen2.5:7b", "nomic-embed-text:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable", _async(True))
    monkeypatch.setattr(diagnostics, "_model_fit", lambda *a: None)
    verdict = diagnostics._check_model(reach_out=True)

    assert verdict["state"] == diagnostics.PASS


def test_the_model_check_is_not_a_guess_when_the_server_is_unreachable(configured, monkeypatch):
    """No models listed means the server is what is wrong, and _check_ollama says
    so. Reporting the model as missing on top of that would be two complaints
    about one fault, and the second one wrong."""
    from core import diagnostics

    monkeypatch.setattr(model_setup, "installed_models", _async([]))
    verdict = diagnostics._check_model(reach_out=True)

    assert verdict["state"] == diagnostics.NOT_TESTED


def test_a_model_too_big_for_the_card_is_a_warning_with_the_numbers(configured, monkeypatch):
    """Ollama does not refuse a model that does not fit - it moves layers onto the
    processor, which turns an ordinary turn into minutes and looks exactly like
    Leti hanging. Retrying does not help; saying so, with the arithmetic, does."""
    from core import diagnostics

    monkeypatch.setattr(model_setup, "installed_models",
                        _async(["qwen2.5:7b", "nomic-embed-text:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable", _async(True))
    monkeypatch.setattr(model_setup, "detect_hardware", lambda: {
        "ok": True, "gpu": {"name": "GTX 1650", "vram_gib": 4.0}, "ram_gib": 16.0,
        "free_disk_gib": 200.0, "cpu": {}, "gpu_detected": True})

    verdict = diagnostics._check_model(reach_out=True)

    assert verdict["state"] == diagnostics.WARNING
    assert "does not fit" in verdict["detail"]
    assert verdict["needs_gib"] and verdict["available_gib"]
    assert str(verdict["needs_gib"]) in verdict["detail"], "the requirement is not shown"
    assert str(verdict["available_gib"]) in verdict["detail"], "what is available is not shown"


def test_the_model_check_works_whether_or_not_a_loop_is_already_running(configured, monkeypatch):
    """full_check normally runs in a worker thread with no loop of its own, where
    asyncio.run is right. It is not right on a thread that already has one -
    asyncio.run refuses, and the check reported that it could not ask a server
    that was answering perfectly well."""
    import asyncio as _asyncio

    from core import diagnostics

    monkeypatch.setattr(model_setup, "installed_models",
                        _async(["qwen2.5:7b", "nomic-embed-text:latest"]))
    monkeypatch.setattr(model_setup, "model_is_usable", _async(True))
    monkeypatch.setattr(diagnostics, "_model_fit", lambda *a: None)

    # No loop here.
    assert diagnostics._check_model(reach_out=True)["state"] == diagnostics.PASS

    # And inside one.
    async def inside():
        return diagnostics._check_model(reach_out=True)

    assert _asyncio.run(inside())["state"] == diagnostics.PASS
