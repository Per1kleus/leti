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
import sys
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


async def pull_model(model: str, progress: Optional[Any] = None) -> Dict[str, Any]:
    """Download one model. Returns {"ok": bool, "error": str|None}.

    Long timeout because this is several gigabytes over whatever connection the
    machine has; a failure here is reported, never raised.

    `progress`, when given, is called with (model, done_bytes, total_bytes,
    status) as the download proceeds. Streaming exists for that: several
    gigabytes with no output at all is indistinguishable from a hang, and the one
    thing a person needs to see on a first launch is that something is happening
    and roughly how much longer. Without a callback it still streams and simply
    does not report, because the alternative - one request that returns after
    twenty minutes - is also what a read timeout looks like.
    """
    import httpx

    last_error = ""
    saw_success = False
    try:
        async with httpx.AsyncClient(base_url=_host(), timeout=httpx.Timeout(None)) as client:
            async with client.stream("POST", "/api/pull",
                                     json={"model": model, "stream": True}) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    return {"ok": False, "error": _pull_error(body, resp.status_code)}
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error"):
                        last_error = str(chunk["error"])
                        break
                    status = str(chunk.get("status") or "")
                    if status == "success":
                        saw_success = True
                    if progress is not None:
                        try:
                            progress(model,
                                     int(chunk.get("completed") or 0),
                                     int(chunk.get("total") or 0),
                                     status)
                        except Exception:
                            # A progress display is never allowed to fail a download.
                            pass
    except Exception as e:
        logger.warning(f"Pulling '{model}' failed ({e}).")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if last_error:
        logger.warning(f"Ollama refused to pull '{model}': {last_error}")
        return {"ok": False, "error": last_error}
    if not saw_success:
        # The stream ended without Ollama saying it had finished. Reported rather
        # than assumed complete - ensure_models checks the model afterwards
        # anyway, and a caller using pull_model directly deserves to know.
        return {"ok": False,
                "error": "the download ended without Ollama reporting success"}
    return {"ok": True, "error": None, "status": "success"}


def _pull_error(body: Any, status_code: int) -> str:
    """What a non-200 from /api/pull said, rather than only its status code."""
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
            return parsed["error"].strip()
    except Exception:
        pass
    return f"Ollama answered HTTP {status_code} to the download request"


# --------------------------------------------------------------------------- #
# Which models are required, and which may be fetched without being asked
#
# This is the one place that answers "what does Leti need downloaded". The
# launcher asks it rather than naming a model, because a model name in the
# launcher is a second configuration source that drifts the moment somebody edits
# settings.yaml - and it would not know which of the four model settings matter.
#
# Roles, not names. Every entry is read from the live configuration, so changing
# ollama.reasoning_model changes what this returns and nothing else has to know.
# --------------------------------------------------------------------------- #

# Leti cannot answer at all without these two: one to think with, one to turn a
# memory into a vector. They are what a launch ensures.
ESSENTIAL_ROLES = ("reasoning_model", "embedding_model")

# Useful, large, and not needed to hold a conversation. The vision model is about
# eight gigabytes and is only read when something asks about the screen; the
# fallback exists precisely for when the primary is missing, so downloading it
# alongside the primary would be downloading a spare tyre to carry in the boot of
# another car. Both are pulled on demand by the code that uses them, not here.
OPTIONAL_ROLES = ("vision_model", "fallback_reasoning_model")


def required_models(include_optional: bool = False) -> List[Dict[str, Any]]:
    """What the CURRENT configuration says Leti needs, as {role, model, essential}.

    Order is ESSENTIAL_ROLES then OPTIONAL_ROLES, so a caller that pulls them in
    order gets the machine working soonest. A role set to an empty string is left
    out rather than reported as a model called "".
    """
    try:
        ollama = get_settings()["ollama"]
    except Exception as e:
        logger.warning(f"Couldn't read the model configuration ({e}).")
        return []
    roles = ESSENTIAL_ROLES + (OPTIONAL_ROLES if include_optional else ())
    out: List[Dict[str, Any]] = []
    for role in roles:
        model = str(ollama.get(role) or "").strip()
        if model:
            out.append({"role": role, "model": model,
                        "essential": role in ESSENTIAL_ROLES})
    return out


def may_pull_unattended(role: str) -> bool:
    """Whether this role may be downloaded without asking the user first.

    The rule lives here rather than in the launcher because it is a statement
    about this module's promise: nothing is downloaded before the first-launch
    screen has been answered, because until then the answer may change which model
    Leti uses and pre-fetching the default would spend several gigabytes on a
    model the user is about to replace.

    Once the choice has been made - whatever it was - the configuration is the
    user's own, and keeping it downloaded needs no further permission.

    The embedding model is the exception in both directions: the first-launch
    screen only ever sets reasoning_model, so no answer to it can change what the
    embedding model should be, and Leti's memory does not work without one.
    """
    if role == "embedding_model":
        return True
    return is_configured()


async def model_is_usable(model: str) -> bool:
    """Whether Ollama can actually load this model, not merely list it.

    /api/tags lists what has a manifest. A download interrupted partway leaves
    one behind with blobs missing, and a model whose blobs were corrupted or
    deleted underneath it lists exactly like a good one - which is the case the
    brief names: a state file saying a model is installed is not evidence that it
    is. /api/show reads the manifest AND resolves the blobs, so it fails for both.
    """
    import httpx

    try:
        async with httpx.AsyncClient(base_url=_host(), timeout=30) as client:
            resp = await client.post("/api/show", json={"model": model})
            if resp.status_code != 200:
                logger.warning(
                    f"Ollama lists '{model}' but cannot describe it "
                    f"(HTTP {resp.status_code}); treating it as not installed.")
                return False
            body = resp.json()
    except Exception as e:
        logger.warning(f"Couldn't check whether '{model}' is usable ({e}).")
        return False
    return isinstance(body, dict) and not body.get("error")


async def missing_models(include_optional: bool = False,
                         verify: bool = True) -> List[Dict[str, Any]]:
    """The required models this machine does not actually have.

    Asks Ollama, never a state file. data/model_setup.json records what was
    chosen, which is a different question from what is on the disk now: a model
    can be removed with `ollama rm`, a download can be interrupted, and a blob
    can be corrupted, and in all three cases the state file still says it is
    there.

    Returns [] when Ollama cannot be reached at all, because "I could not ask" is
    not "nothing is installed" - pulling four models on the strength of a refused
    connection is the wrong way to be wrong.
    """
    wanted = required_models(include_optional)
    if not wanted:
        return []
    installed = await installed_models()
    if not installed:
        logger.info("Ollama listed no models; not treating that as a reason to pull.")
        return []
    missing: List[Dict[str, Any]] = []
    for entry in wanted:
        if not _is_installed(entry["model"], installed):
            missing.append({**entry, "reason": "not installed"})
        elif verify and not await model_is_usable(entry["model"]):
            missing.append({**entry, "reason": "installed but unusable"})
    return missing


async def ensure_models(progress: Optional[Any] = None,
                        include_optional: bool = False,
                        unattended: bool = True) -> Dict[str, Any]:
    """Download whatever is required and not already usable. Never raises.

    `progress`, if given, is called with (model, done_bytes, total_bytes, status)
    as each download proceeds, so a terminal can draw a bar and a window can show
    one. `unattended` false ignores may_pull_unattended and fetches everything
    required, which is what the first-launch screen does once the user has said
    yes to something.

    Returns {"ok", "pulled", "skipped", "failed", "checked"}. `ok` is false only
    when something that had to be downloaded could not be.
    """
    try:
        missing = await missing_models(include_optional)
    except Exception as e:
        logger.warning(f"Couldn't work out which models are missing ({e}).")
        return {"ok": True, "pulled": [], "skipped": [], "failed": [], "checked": False,
                "error": str(e)}

    pulled: List[str] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for entry in missing:
        if unattended and not may_pull_unattended(entry["role"]):
            skipped.append({**entry, "why": "waiting for the first-launch model choice"})
            continue
        result = await pull_model(entry["model"], progress=progress)
        if not result.get("ok"):
            failed.append({**entry, "error": result.get("error")})
            continue
        # Verified rather than assumed: a pull that reports success and leaves
        # nothing loadable behind is exactly the state this function exists to
        # get out of, and reporting it as fixed would hide it until the first
        # question the user asked.
        if not await model_is_usable(entry["model"]):
            failed.append({**entry,
                           "error": "the download reported success but the model "
                                    "still cannot be loaded"})
            continue
        pulled.append(entry["model"])

    return {"ok": not failed, "pulled": pulled, "skipped": skipped,
            "failed": failed, "checked": True}


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
        # "Keep what I have" is not the same as "I have it". Nothing used to check,
        # so a machine with no model downloaded at all could answer this screen,
        # have the answer saved, never be asked again, and then fail on the first
        # question with a model error. The configuration is still not touched -
        # the download is the only thing that happens, and only if it is missing.
        note = "Keeping the models Leti is already configured for. Nothing was changed."
        if kept:
            installed = await installed_models()
            if installed and not _is_installed(kept, installed):
                pull = await pull_model(kept)
                if not pull["ok"]:
                    # Not saved: next launch asks again rather than leaving Leti
                    # pointed at a model that is not there.
                    return {"ok": False, "choice": "keep_current", "model": kept,
                            "error": pull["error"],
                            "note": (f"'{kept}' is the configured model but it is not "
                                     "downloaded, and downloading it failed. Nothing "
                                     "was changed.")}
                note = (f"Keeping '{kept}', which was configured but not downloaded, "
                        "so it was downloaded. No setting was changed.")
        save_setup({"choice": "keep_current", "model": kept})
        return {"ok": True, "choice": "keep_current", "model": kept, "error": None,
                "note": note}

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


# --------------------------------------------------------------------------- #
# Being asked from outside the application
#
# launcher/bootstrap.py runs before Leti does, under whatever interpreter started
# it, and is standard library only on purpose - it is the code that runs when
# nothing is installed. So it cannot import this module, and it must not carry a
# model name of its own: that would be a second answer to "which models does Leti
# need", and it would be wrong the moment somebody edited settings.yaml.
#
# It runs this instead, with the interpreter it has just finished preparing. One
# authority, asked across a process boundary.
#
# The launcher does not capture the output and does not parse it: what is written
# here goes straight to the console it is already writing to, live, and the exit
# code is the whole of what it reads back. That is deliberate - reading a child's
# pipe means starting something and not waiting for it, and the launcher starts
# nothing it does not wait for.
# --------------------------------------------------------------------------- #

BAR_WIDTH = 28


def _bytes_for_people(count: Any) -> str:
    """Bytes as something worth reading. GB once MB stops being a small number."""
    try:
        count = int(count)
    except (TypeError, ValueError):
        return "?"
    if count >= 1024 ** 3:
        return f"{count / 1024 ** 3:.1f} GB"
    return f"{count / 1024 ** 2:.0f} MB"


def progress_bar(model: str, done: int, total: int, width: int = BAR_WIDTH) -> str:
    """One line of download progress. Pure, so it can be tested without a terminal."""
    if total and total > 0:
        fraction = max(0.0, min(1.0, done / total))
        filled = int(round(fraction * width))
        return (f"  {model} [{'#' * filled}{'-' * (width - filled)}] "
                f"{fraction * 100:5.1f}%  {_bytes_for_people(done)} "
                f"of {_bytes_for_people(total)}")
    return f"  {model}  {_bytes_for_people(done)}" if done else f"  {model}  starting"


class _TerminalProgress:
    """Draws the bar, in place on a terminal and one line per tenth otherwise.

    The bar lives here rather than in launcher/bootstrap.py because this is where
    the download happens and this is the only thing that draws one. The launcher
    runs this as a child process and does not capture its output, so what is
    written here goes straight to the console the user is already watching - which
    is the point: three gigabytes with nothing on screen is indistinguishable
    from a hang, and that is the longest thing a first launch does.
    """

    def __init__(self, out: Any = None) -> None:
        self.out = out if out is not None else sys.stdout
        self._open = False
        self._last_tenth: Dict[str, int] = {}
        self._announced: Dict[str, bool] = {}

    def _terminal(self) -> bool:
        try:
            return bool(self.out.isatty())
        except Exception:
            return False

    def line(self, text: str) -> None:
        self.close()
        print(text, file=self.out, flush=True)

    def close(self) -> None:
        if self._open:
            self._open = False
            print("", file=self.out, flush=True)

    def __call__(self, model: str, done: int, total: int, status: str) -> None:
        if not self._announced.get(model):
            self._announced[model] = True
            self.line(f"  Downloading {model} - this can take a while.")
        if not total:
            return
        text = progress_bar(model, done, total)
        if self._terminal():
            self._open = True
            print("\r" + text.ljust(78), end="", file=self.out, flush=True)
            return
        # Not a terminal - a log file or a captured pipe. One line per tenth, so a
        # transcript stays readable instead of holding a thousand redraws.
        tenth = int((done / total) * 10)
        if self._last_tenth.get(model) != tenth:
            self._last_tenth[model] = tenth
            print(text, file=self.out, flush=True)


async def _ensure_from_command_line() -> int:
    show = _TerminalProgress()
    try:
        needed = required_models()
    except Exception as e:
        show.line(f"  Could not read which models Leti needs: {e}")
        return 1

    if not needed:
        show.line("  No models are configured, so there is nothing to download.")
        return 0

    result = await ensure_models(progress=show)
    show.close()

    for model in result.get("pulled", ()):
        show.line(f"  {model} is ready.")
    for entry in result.get("skipped", ()):
        show.line(f"  {entry.get('model')} will be chosen on the model screen "
                  "in a moment.")
    for entry in result.get("failed", ()):
        show.line(f"  Could not download {entry.get('model')}: "
                  f"{entry.get('error') or 'no reason given'}")
    if not result.get("checked"):
        show.line("  Could not ask Ollama what is installed; nothing was downloaded.")
    elif not (result.get("pulled") or result.get("skipped") or result.get("failed")):
        show.line("  Everything Leti needs is already downloaded.")
    return 0 if result.get("ok") else 1


def main(argv: Optional[List[str]] = None) -> int:
    """`python -m core.model_setup --ensure` downloads whatever is missing.

    Deliberately the only thing the command line can do. Choosing a model is a
    question for a person and it has a screen; this is the part that needs no
    answer, because the configuration has already given one.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    if "--ensure" not in args:
        print("usage: python -m core.model_setup --ensure", file=sys.stderr)
        return 2
    try:
        return asyncio.run(_ensure_from_command_line())
    except KeyboardInterrupt:
        # Ctrl-C during a download. Ollama keeps what it has, so the next launch
        # resumes rather than starting again; said so the user knows that.
        print("\n  Stopped. The next launch picks the download up where it left off.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
