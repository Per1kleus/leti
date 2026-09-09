"""The first-launch model check: hardware in, a recommendation out, nothing
written until the user says so.

The tests that matter most here are the negative ones. This is the only part of
Leti that downloads gigabytes and rewrites which model runs, so what it must not
do - not on a failed detection, not on a failed download, not without an answer,
and never twice - is the specification.
"""
from __future__ import annotations

import json

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
