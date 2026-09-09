"""First-launch model selection: what THIS machine can actually run.

Leti's defaults are sized for one machine, and the machine that runs it is a
different one. This asks, once, on the first launch: what hardware is here, which
local model fits it comfortably, and does the user want it - before anything is
downloaded and before a single configuration value is written.

Three things shape the recommendation, and the third is the one that catches
people out:

  1. The weights have to sit in VRAM.
  2. So does the KV cache, and its size is set by ollama.num_ctx, which Leti runs
     high because every one of its tools is sent on every call. On a 14B at 24k
     that cache is larger than a 3B model's entire weights. A model chosen on
     weights alone will not fit.
  3. Whatever is left has to still hold the desktop, Whisper, and Leti itself.

Nothing here changes num_ctx, the tool set, or any model setting on its own. It
reads num_ctx to size the estimate, shows its arithmetic, and waits. Declining
leaves the configuration byte-for-byte as it was, and is a first-class answer
rather than a way out of the screen.

Isolated on purpose: nothing else in Leti imports this, and it imports only the
config loader, the settings writer and the same Ollama HTTP API the rest of the
application already speaks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from core.atomic_write import atomic_write_text
from core.config_loader import get_settings, resolve_path

logger = logging.getLogger("leti.model_setup")

SETUP_PATH = "./data/model_setup.json"
SETUP_VERSION = 1

# What the machine needs for things that are not the model. Deliberately generous:
# the cost of over-reserving is a slightly smaller model, and the cost of
# under-reserving is the thing this whole module exists to prevent.
_VRAM_RESERVED_GIB = 1.5      # desktop compositor, browser, whatever else is open
_VRAM_HEADROOM_GIB = 0.8      # Whisper on the GPU, and not running at the edge
_COMPUTE_BUFFER_GIB = 0.7     # llama.cpp's own scratch buffers
_DISK_MARGIN_GIB = 5.0        # never fill the disk to download a model

# Candidate reasoning models, smallest first. Same family as the configured
# default, so this recommends a size rather than introducing a model nobody chose.
#
# weights_gib and kv_kib_per_token are APPROXIMATIONS used only for fitting. They
# are close enough to choose a size with, and every one of them is shown to the
# user before anything happens, so an estimate that is off is visible rather than
# silent. kv_kib_per_token is 2 (K and V) x layers x kv_heads x head_dim x 2 bytes.
CANDIDATES: List[Dict[str, Any]] = [
    {"model": "qwen2.5:3b",  "label": "3B",  "weights_gib": 1.9,  "kv_kib_per_token": 36},
    {"model": "qwen2.5:7b",  "label": "7B",  "weights_gib": 4.4,  "kv_kib_per_token": 56},
    {"model": "qwen2.5:14b", "label": "14B", "weights_gib": 8.4,  "kv_kib_per_token": 192},
    {"model": "qwen2.5:32b", "label": "32B", "weights_gib": 19.9, "kv_kib_per_token": 256},
]


# --------------------------------------------------------------------------- #
# Setup state - the same shape as audio/setup.py, for the same reason
# --------------------------------------------------------------------------- #

def setup_path():
    return resolve_path(SETUP_PATH)


def load_setup() -> Optional[Dict[str, Any]]:
    """The saved answer, or None if this machine has never been asked."""
    path = setup_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        # Unreadable means "we don't know", which is the same as never asked. One
        # extra prompt is cheaper than a machine stuck on a model that does not fit.
        logger.warning(f"Couldn't read {path} ({e}); treating model setup as not done.")
        return None
    return data if isinstance(data, dict) else None


def save_setup(data: Dict[str, Any]) -> Dict[str, Any]:
    data = {**data, "version": SETUP_VERSION, "completed_at": time.time()}
    path = setup_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2))
    return data


def is_configured() -> bool:
    """True once the user has answered. The screen is shown exactly until this is."""
    return load_setup() is not None


# --------------------------------------------------------------------------- #
# Hardware detection - reads only, never raises
# --------------------------------------------------------------------------- #

def _nvidia_gpu() -> Optional[Dict[str, Any]]:
    """The first NVIDIA GPU and its VRAM, or None if we cannot say."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"nvidia-smi failed ({e}); treating VRAM as unknown.")
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    first = out.stdout.strip().splitlines()[0]
    name, _, mib = first.partition(",")
    try:
        vram_gib = round(float(mib.strip()) / 1024, 1)
    except ValueError:
        return None
    return {"name": name.strip(), "vram_gib": vram_gib}


def detect_hardware() -> Dict[str, Any]:
    """What this machine is. Never raises: a failure is reported, not thrown.

    `ok` false means the recommendation cannot be trusted and the caller should
    offer to keep the current models rather than guess at new ones.
    """
    info: Dict[str, Any] = {
        "ok": False, "error": None,
        "os": None, "cpu": None, "ram_gib": None, "free_disk_gib": None,
        "gpu": None, "gpu_detected": False,
    }
    try:
        info["os"] = f"{platform.system()} {platform.release()}"
    except Exception:
        pass

    try:
        import psutil

        cpu_freq = None
        try:
            cpu_freq = psutil.cpu_freq()
        except Exception:
            pass
        info["cpu"] = {
            "model": platform.processor() or platform.machine() or "unknown",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency_mhz": cpu_freq.max if cpu_freq else None,
        }
        info["ram_gib"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        try:
            info["free_disk_gib"] = round(
                psutil.disk_usage(str(resolve_path("."))).free / (1024 ** 3), 1)
        except Exception:
            pass
        info["ok"] = True
    except Exception as e:
        # No psutil, or it could not read this machine. Everything downstream
        # treats ok=False as "do not recommend anything but the safest option".
        logger.warning(f"Hardware detection failed ({e}).")
        info["error"] = str(e)
        return info

    try:
        gpu = _nvidia_gpu()
    except Exception as e:                      # belt and braces: never raise
        logger.warning(f"GPU detection failed ({e}).")
        gpu = None
    if gpu:
        info["gpu"] = gpu
        info["gpu_detected"] = True
    return info


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #

def _kv_gib(candidate: Dict[str, Any], num_ctx: int) -> float:
    return round(candidate["kv_kib_per_token"] * num_ctx / (1024 ** 2), 2)


def _needs_gib(candidate: Dict[str, Any], num_ctx: int) -> float:
    return round(candidate["weights_gib"] + _kv_gib(candidate, num_ctx)
                 + _COMPUTE_BUFFER_GIB, 2)


def current_model() -> str:
    try:
        return str(get_settings()["ollama"]["reasoning_model"])
    except Exception:
        return ""


def recommend(hardware: Dict[str, Any], num_ctx: Optional[int] = None) -> Dict[str, Any]:
    """The largest candidate that fits comfortably, with the arithmetic shown.

    "Comfortably" is the whole point: a model that fits only by taking the VRAM
    the desktop is using, or only by spilling layers onto the CPU, is not a
    recommendation - it is the problem this exists to avoid. Where the machine
    cannot be read confidently the answer is the smallest candidate, flagged as
    uncertain, never an optimistic guess.
    """
    if num_ctx is None:
        try:
            num_ctx = int(get_settings()["ollama"]["num_ctx"])
        except Exception:
            num_ctx = 16384

    current = current_model()
    result: Dict[str, Any] = {
        "current": current,
        "recommended": None,
        "confident": False,
        "num_ctx": num_ctx,
        "reasons": [],
        "tradeoffs": [],
        "considered": [],
        "estimate": None,
    }

    vram = None
    if hardware.get("gpu"):
        vram = hardware["gpu"].get("vram_gib")
    budget = None
    if isinstance(vram, (int, float)) and vram > 0:
        budget = round(vram - _VRAM_RESERVED_GIB - _VRAM_HEADROOM_GIB, 2)
    result["vram_budget_gib"] = budget

    free_disk = hardware.get("free_disk_gib")
    ram = hardware.get("ram_gib")

    for candidate in CANDIDATES:
        needs = _needs_gib(candidate, num_ctx)
        row = {
            "model": candidate["model"], "label": candidate["label"],
            "weights_gib": candidate["weights_gib"],
            "kv_gib": _kv_gib(candidate, num_ctx),
            "total_gib": needs, "fits": False, "why_not": None,
        }
        if budget is None:
            row["why_not"] = "no GPU VRAM figure, so this cannot be checked"
        elif needs > budget:
            row["why_not"] = (f"needs ~{needs} GiB against a ~{budget} GiB budget - "
                              "Ollama would move layers onto the CPU")
        elif (isinstance(free_disk, (int, float))
              and free_disk < candidate["weights_gib"] + _DISK_MARGIN_GIB):
            row["why_not"] = f"only {free_disk} GiB of disk free"
        else:
            row["fits"] = True
        result["considered"].append(row)

    fitting = [row for row in result["considered"] if row["fits"]]

    if not hardware.get("ok"):
        result["recommended"] = None
        result["reasons"].append(
            "This machine's hardware could not be read, so nothing is being "
            "recommended. Keeping the current models is the safe answer.")
        return result

    if budget is None:
        # No GPU, or VRAM we could not read. Everything would run on the CPU, and
        # guessing a size for that is exactly the aggressive recommendation the
        # brief rules out - so: the smallest candidate, and say why.
        smallest = CANDIDATES[0]
        result["recommended"] = smallest["model"]
        result["confident"] = False
        result["estimate"] = {
            "weights_gib": smallest["weights_gib"],
            "kv_gib": _kv_gib(smallest, num_ctx),
            "total_gib": _needs_gib(smallest, num_ctx),
        }
        result["reasons"].append(
            "No NVIDIA GPU was detected, or its VRAM could not be read. Without a "
            f"VRAM figure the only safe choice is the smallest model ({smallest['label']}).")
        result["tradeoffs"].append(
            "On the CPU, expect a few tokens per second rather than tens. If this "
            "machine does have a usable GPU, keeping the current models and setting "
            "the model by hand will serve you better than this guess.")
        if isinstance(ram, (int, float)) and ram < 8:
            result["tradeoffs"].append(
                f"{ram} GiB of RAM is tight for CPU inference alongside Leti itself.")
        return result

    if not fitting:
        smallest = CANDIDATES[0]
        result["recommended"] = smallest["model"]
        result["confident"] = False
        result["estimate"] = {
            "weights_gib": smallest["weights_gib"],
            "kv_gib": _kv_gib(smallest, num_ctx),
            "total_gib": _needs_gib(smallest, num_ctx),
        }
        result["reasons"].append(
            f"Nothing fits a ~{budget} GiB VRAM budget at a {num_ctx:,}-token context "
            f"window, so the smallest model ({smallest['label']}) is the least bad option.")
        result["tradeoffs"].append(
            "Some of it will still run on the CPU. Lowering ollama.num_ctx would free "
            "VRAM, but Leti's tool list needs most of the window it has - that is a "
            "decision for you, not for this screen.")
        return result

    best = fitting[-1]
    result["recommended"] = best["model"]
    result["confident"] = True
    result["estimate"] = {
        "weights_gib": best["weights_gib"],
        "kv_gib": best["kv_gib"],
        "total_gib": best["total_gib"],
    }
    gpu_name = hardware["gpu"].get("name", "this GPU")
    result["reasons"].append(
        f"{gpu_name} reports {vram} GiB of VRAM. After ~{_VRAM_RESERVED_GIB} GiB for the "
        f"desktop and ~{_VRAM_HEADROOM_GIB} GiB of headroom, ~{budget} GiB is left for the model.")
    result["reasons"].append(
        f"{best['label']} needs ~{best['total_gib']} GiB: {best['weights_gib']} GiB of weights "
        f"plus ~{best['kv_gib']} GiB of KV cache at {num_ctx:,} tokens, which is the window "
        "Leti's tool list requires. It fits entirely in VRAM, so nothing runs on the CPU.")
    bigger = [row for row in result["considered"] if not row["fits"] and row["why_not"]]
    if bigger:
        nxt = bigger[0]
        result["tradeoffs"].append(
            f"{nxt['label']} would be the more capable model but {nxt['why_not']}.")
    result["tradeoffs"].append(
        "The vision model is loaded separately and may still swap with this one on a "
        "smaller card; that is unchanged either way and is not part of this choice.")
    if isinstance(ram, (int, float)) and ram < 16:
        result["tradeoffs"].append(
            f"{ram} GiB of system RAM is on the low side for Leti's other components.")
    return result


# --------------------------------------------------------------------------- #
# Ollama - the same HTTP API the rest of Leti already speaks
# --------------------------------------------------------------------------- #

def _host() -> str:
    try:
        return str(get_settings()["ollama"]["host"]).rstrip("/")
    except Exception:
        return "http://localhost:11434"


async def installed_models() -> List[str]:
    """Model names Ollama already has. Empty when it cannot be reached."""
    import httpx

    try:
        async with httpx.AsyncClient(base_url=_host(), timeout=15) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
    except Exception as e:
        logger.warning(f"Couldn't list installed models ({e}).")
        return []
    return [m.get("name", "") for m in models if isinstance(m, dict) and m.get("name")]


def _is_installed(model: str, installed: List[str]) -> bool:
    """Ollama reports 'qwen2.5:7b'; a bare 'qwen2.5' means the latest tag."""
    wanted = model if ":" in model else model + ":latest"
    return wanted in installed


async def pull_model(model: str) -> Dict[str, Any]:
    """Download one model. Returns {"ok": bool, "error": str|None}.

    Long timeout because this is several gigabytes over whatever connection the
    machine has; a failure here is reported, never raised.
    """
    import httpx

    try:
        async with httpx.AsyncClient(base_url=_host(), timeout=httpx.Timeout(None)) as client:
            resp = await client.post("/api/pull", json={"model": model, "stream": False})
            resp.raise_for_status()
            body = resp.json()
    except Exception as e:
        logger.warning(f"Pulling '{model}' failed ({e}).")
        return {"ok": False, "error": str(e)}
    status = body.get("status") if isinstance(body, dict) else None
    if isinstance(body, dict) and body.get("error"):
        return {"ok": False, "error": str(body["error"])}
    return {"ok": True, "error": None, "status": status}


# --------------------------------------------------------------------------- #
# Applying a choice
# --------------------------------------------------------------------------- #

def _write_reasoning_model(model: str) -> None:
    """Set ollama.reasoning_model in settings.local.yaml and nothing else.

    Goes through the same override file and writer the /settings editor uses, so
    settings.yaml is never touched and every other key - num_ctx, the vision and
    embedding models, the fallback - keeps whatever value it had.
    """
    from core.config_loader import reload_settings
    from core.settings_editor import _deep_set, _load_overrides, _save_overrides

    overrides = _load_overrides()
    _deep_set(overrides, ["ollama"], {"reasoning_model": model})
    _save_overrides(overrides)
    reload_settings()


async def apply_choice(choice: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Act on the user's answer. Returns {"ok", "choice", "model", "error", "note"}.

    The ordering is the guarantee: a model is downloaded and verified present
    BEFORE any configuration is written, so a failed download leaves the working
    configuration exactly as it was and the screen is shown again next launch.
    """
    if choice == "keep_current":
        kept = current_model()
        save_setup({"choice": "keep_current", "model": kept})
        return {"ok": True, "choice": "keep_current", "model": kept, "error": None,
                "note": "Keeping the models Leti is already configured for. Nothing was changed."}

    if choice != "recommended" or not model:
        return {"ok": False, "choice": choice, "model": model,
                "error": "No model was chosen, so nothing was changed."}

    installed = await installed_models()
    downloaded = False
    if not _is_installed(model, installed):
        pull = await pull_model(model)
        if not pull["ok"]:
            # Nothing written, nothing marked done: next launch asks again.
            return {"ok": False, "choice": choice, "model": model, "error": pull["error"],
                    "note": (f"Couldn't download '{model}'. Your existing model "
                             "configuration has not been changed.")}
        downloaded = True
        installed = await installed_models()

    # Verify before configuring: a pull that reported success but left nothing
    # behind would otherwise point Leti at a model that is not there.
    if installed and not _is_installed(model, installed):
        return {"ok": False, "choice": choice, "model": model,
                "error": f"'{model}' is still not listed by Ollama after downloading.",
                "note": "Your existing model configuration has not been changed."}

    try:
        _write_reasoning_model(model)
    except Exception as e:
        logger.exception("Writing the model setting failed")
        return {"ok": False, "choice": choice, "model": model, "error": str(e),
                "note": "Your existing model configuration has not been changed."}

    save_setup({"choice": "recommended", "model": model, "downloaded": downloaded})
    return {"ok": True, "choice": "recommended", "model": model, "error": None,
            "downloaded": downloaded,
            "note": f"Leti will use '{model}'. Other model settings were left as they were."}


async def gather() -> Dict[str, Any]:
    """Everything the first-launch screen needs, in one call.

    Detection runs in a thread: nvidia-smi can take a second, and in GUI mode this
    is called on the loop that serves every connected client.
    """
    loop = asyncio.get_running_loop()
    hardware = await loop.run_in_executor(None, detect_hardware)
    advice = recommend(hardware)
    installed = await installed_models()
    if advice.get("recommended"):
        advice["already_installed"] = _is_installed(advice["recommended"], installed)
    else:
        advice["already_installed"] = False
    advice["current_installed"] = _is_installed(advice["current"], installed) if advice["current"] else False
    return {"configured": is_configured(), "hardware": hardware, "recommendation": advice}


# --------------------------------------------------------------------------- #
# The first-launch screen, in a terminal
# --------------------------------------------------------------------------- #

def describe(state: Dict[str, Any]) -> List[str]:
    """The screen's text, as lines. Shared so the terminal and the window say the
    same things in the same order rather than drifting apart."""
    hardware, advice = state["hardware"], state["recommendation"]
    lines: List[str] = ["", "  Leti - choosing a model for this computer", ""]

    if not hardware.get("ok"):
        lines.append(f"  Couldn't read this machine's hardware: {hardware.get('error')}")
    else:
        cpu = hardware.get("cpu") or {}
        lines.append(f"  OS       {hardware.get('os') or 'unknown'}")
        lines.append(f"  CPU      {cpu.get('model', 'unknown')} "
                     f"({cpu.get('physical_cores')} cores / {cpu.get('logical_cores')} threads)")
        lines.append(f"  RAM      {hardware.get('ram_gib')} GiB")
        gpu = hardware.get("gpu")
        lines.append(f"  GPU      {gpu['name']} - {gpu['vram_gib']} GiB VRAM" if gpu
                     else "  GPU      none detected (or VRAM unreadable)")
        if hardware.get("free_disk_gib") is not None:
            lines.append(f"  Disk     {hardware['free_disk_gib']} GiB free")

    lines.append("")
    lines.append(f"  Currently configured: {advice.get('current') or 'unknown'}")
    if advice.get("recommended"):
        confident = "" if advice.get("confident") else "  (uncertain - see below)"
        lines.append(f"  Recommended:          {advice['recommended']}{confident}")
        if advice.get("already_installed"):
            lines.append("                        already downloaded")
        estimate = advice.get("estimate") or {}
        if estimate:
            lines.append(f"  Expected VRAM use:    ~{estimate.get('total_gib')} GiB "
                         f"({estimate.get('weights_gib')} weights + "
                         f"~{estimate.get('kv_gib')} KV cache at "
                         f"{advice.get('num_ctx', 0):,} tokens)")
    else:
        lines.append("  Recommended:          nothing - not enough is known about this machine")

    for reason in advice.get("reasons", []):
        lines.append("")
        lines.append(f"  Why: {reason}")
    for tradeoff in advice.get("tradeoffs", []):
        lines.append("")
        lines.append(f"  Note: {tradeoff}")
    lines.append("")
    return lines


async def run_console_setup(force: bool = False) -> Optional[Dict[str, Any]]:
    """Ask once, in the terminal. None if this machine has already answered.

    Reads through core.console_input and core.intent_signals, the same reader and
    the same yes/no understanding the audio prompt and the confirmations use.
    """
    from core.console_input import read_line
    from core.intent_signals import resolve_yes_no

    if is_configured() and not force:
        return None

    try:
        state = await gather()
    except Exception as e:
        # Detection or Ollama being unreachable must not stop Leti from starting.
        logger.warning(f"Model setup couldn't run ({e}); keeping the current models.")
        print("\n  Couldn't check this computer's hardware; keeping the current models.\n")
        return None

    for line in describe(state):
        print(line)

    advice = state["recommendation"]
    if not advice.get("recommended") or advice["recommended"] == advice.get("current"):
        if advice.get("recommended"):
            print("  That is already what Leti is configured to use - nothing to do.\n")
        save_setup({"choice": "keep_current", "model": advice.get("current", "")})
        return {"ok": True, "choice": "keep_current", "model": advice.get("current", "")}

    print("  [1] Download and use the recommended model")
    print("  [2] Leti was designed for the current models we have now (change nothing)")
    print("")

    while True:
        answer = await read_line("  Choose 1 or 2 [2]: ")
        if answer is None:
            # Nobody at the keyboard. Changing models unattended is not something
            # to do on a guess, so the safe answer stands and is NOT recorded -
            # the next interactive launch still gets to choose.
            print("  No answer; keeping the current models for now.\n")
            return None
        answer = answer.strip().lower()
        if answer in ("", "2", "no", "n", "keep", "current"):
            result = await apply_choice("keep_current")
            print(f"  {result['note']}\n")
            return result
        if answer in ("1", "yes", "y", "download", "recommended"):
            print(f"  Downloading {advice['recommended']} (this can take a while)...")
            result = await apply_choice("recommended", advice["recommended"])
            print(f"  {result.get('note') or result.get('error')}\n")
            return result
        decision = resolve_yes_no(answer)
        if decision is False:
            result = await apply_choice("keep_current")
            print(f"  {result['note']}\n")
            return result
        print("  Please answer 1 or 2.")
