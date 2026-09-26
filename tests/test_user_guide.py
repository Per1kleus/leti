"""LETi_USER_GUIDE.txt says true things about the program it documents.

The guide is the one document written for somebody who is not going to read the
code, which makes it the one document where a wrong statement cannot be caught by
the reader. Everything in it that is a fact about the code is checked here: the
settings it names, the values it quotes, the files and commands it tells people
to use, and the promise that it contains no credential.

A statement that is a judgement ("180 is the default, 140 is unhurried") is not
checked; the default it rests on is.
"""
from __future__ import annotations

import decimal
import pathlib
import re

import pytest
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
GUIDE_PATH = PROJECT_ROOT / "LETi_USER_GUIDE.txt"


@pytest.fixture(scope="module")
def guide() -> str:
    return GUIDE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prose(guide) -> str:
    """The guide as one long line, so a claim that happens to wrap still reads as
    the sentence it is. Checking the raw text would make every reflow a failure."""
    return re.sub(r"\s+", " ", guide)


@pytest.fixture(scope="module")
def settings() -> dict:
    # The file, not the merged view: the guide documents the shipped defaults, and
    # a developer's settings.local.yaml must not change what it is checked against.
    return yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def test_the_guide_exists_and_covers_every_section_it_promises(guide):
    """Sixteen sections, each one present as a numbered heading."""
    headings = [
        "GETTING STARTED", "WHERE LETI IS INSTALLED", "THE DESKTOP AND START MENU",
        "THE FIRST LAUNCH", "VOICE SETTINGS", "THE AI MODEL", "MAKING LETI FASTER",
        "PERSONALITY", "TASKS THAT RUN ON THEIR OWN", "PERMISSIONS", "CONNECTIONS",
        "WHEN SOMETHING DOES NOT WORK", "RESETTING AND REPAIRING",
        "CHANGING SETTINGS BY HAND", "SAFE CHANGES AND DANGEROUS ONES",
        "UPDATING LETI",
    ]
    for number, heading in enumerate(headings, start=1):
        assert re.search(rf"^{number}\. {heading}", guide, re.MULTILINE), \
            f"section {number} ({heading}) is missing"


# --- Values it quotes -------------------------------------------------------------

@pytest.mark.parametrize("quoted,path", [
    ("28672", ("ollama", "num_ctx")),
    ("qwen2.5:7b", ("ollama", "reasoning_model")),
    ("mistral-nemo", ("ollama", "fallback_reasoning_model")),
    ("llama3.2-vision", ("ollama", "vision_model")),
    ("nomic-embed-text", ("ollama", "embedding_model")),
    ("0.3", ("ollama", "temperature")),
    ("hey_leti", ("app", "wake_word")),
    ("0.5", ("app", "wake_word_threshold")),
    ("base.en", ("stt", "model_size")),
    ("1.2", ("stt", "silence_timeout_seconds")),
    ("180", ("tts", "rate")),
    ("15", ("safety", "confirmation_timeout_seconds")),
    ("5", ("scheduler", "disable_after_failures")),
])
def test_a_default_the_guide_quotes_is_the_configured_default(guide, settings, quoted, path):
    section, key = path
    actual = settings[section][key]
    assert str(actual) == quoted, f"settings has {key}={actual}, the guide says {quoted}"
    assert quoted in guide, f"the guide no longer states {key}"


def test_the_guide_describes_the_real_confirmation_classes(prose, settings):
    configured = settings["safety"]["require_confirmation_for"]
    assert configured == ["modify", "external", "critical"], (
        "the guide says Leti asks before modify, external and critical; "
        f"settings.yaml now says {configured}")
    assert "modify, external or critical" in prose


def test_the_guide_describes_the_real_unattended_limits(guide, settings):
    allows = settings["scheduler"]["unattended_allows"]
    assert allows == ["read", "execute", "modify"], (
        f"the guide's unattended table assumes read/execute/modify; settings says {allows}")
    from core.safety_guard import SafetyGuard

    # critical is stripped whatever the setting says, which is what the guide
    # promises with "always waits for you".
    assert "critical" not in [c.lower() for c in allows]
    assert hasattr(SafetyGuard, "unattended_classes")


def test_the_guide_quotes_the_real_pressure_thresholds(prose):
    from core import performance

    assert performance.CPU_PRESSURE == 90.0 and performance.RAM_PRESSURE == 92.0
    assert "90% processor or 92% memory" in prose


def test_the_guide_quotes_the_real_scheduler_interval(guide):
    from tools.scheduler import SchedulerRunner
    import inspect

    default = inspect.signature(SchedulerRunner.__init__).parameters["check_interval"].default
    assert default == 30.0, f"the scheduler now checks every {default}s"
    assert "every 30 seconds" in re.sub(r"\s+", " ", guide)


def test_the_guide_quotes_the_real_tool_count_and_window_cost(guide):
    import json
    from unittest.mock import MagicMock

    import main
    from core import modes

    registry = main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())
    assert f"{len(registry.names())} tools" in guide, "the registry's size changed"
    visible = modes.visible_tools(registry, modes.DEFAULT)
    assert f"{len(visible)} of them" in guide, "Default Mode's size changed"
    tokens = len(json.dumps(registry.schemas_for(visible))) // 4
    # Quoted to the nearest hundred, so this allows the rounding and nothing else.
    quoted = int(re.search(r"about ([\d,]+) tokens of the 28,672", guide).group(1).replace(",", ""))
    assert abs(quoted - tokens) <= 100, f"the guide says {quoted} tokens; it is {tokens}"


def test_the_guide_lists_the_real_proactive_levels(guide, prose):
    from core import proactive

    assert "off, suggestions, notifications, active" in prose
    assert proactive.LEVELS == ("off", "suggestions", "notifications", "active")
    assert proactive.DEFAULT_LEVEL == "notifications"
    assert "notifications by default" in guide


def test_the_vram_table_matches_what_the_first_launch_computes(guide, settings):
    """The table people choose a model from, against the code that recommends one."""
    from core import model_setup

    num_ctx = int(settings["ollama"]["num_ctx"])
    rows = re.findall(
        r"(qwen2\.5:\d+b)\s+([\d.]+) GB of weights \+\s+([\d.]+) GB of window\s+=\s+"
        r"([\d.]+) GB", guide)
    assert len(rows) == len(model_setup.CANDIDATES), (
        f"the guide lists {len(rows)} models; there are {len(model_setup.CANDIDATES)}")
    def to_one_decimal(value: float) -> str:
        """As a person rounds: half goes up, and 14.35 reads 14.4 rather than 14.3."""
        return str(decimal.Decimal(str(value)).quantize(decimal.Decimal("0.1"),
                                                        rounding=decimal.ROUND_HALF_UP))

    for model, weights, kv, total in rows:
        candidate = next(c for c in model_setup.CANDIDATES if c["model"] == model)
        assert weights == to_one_decimal(candidate["weights_gib"]), model
        assert kv == to_one_decimal(model_setup._kv_gib(candidate, num_ctx)), model
        assert total == to_one_decimal(model_setup._needs_gib(candidate, num_ctx)), model

    reserved = model_setup._VRAM_RESERVED_GIB + model_setup._VRAM_HEADROOM_GIB
    assert f"less {reserved} GB" in guide, f"the reserve is now {reserved} GB"

    # And the four recommendations, from the function that makes them.
    for vram, expected in ((8, "qwen2.5:3b"), (12, "qwen2.5:7b"),
                           (16, "qwen2.5:7b"), (24, "qwen2.5:14b")):
        advice = model_setup.recommend(
            {"ok": True, "gpu": {"vram_gib": float(vram), "name": "card"},
             "ram_gib": 32.0, "free_disk_gib": 200.0, "platform": "Windows"},
            num_ctx=num_ctx)
        assert advice["recommended"] == expected, (
            f"the guide sends a {vram}GB card to {expected}; "
            f"the first-launch check now says {advice['recommended']}")
        assert re.search(rf"{vram} GB card\s+->\s+{re.escape(expected)}", guide)


# --- Files, settings and commands it names ----------------------------------------

@pytest.mark.parametrize("named", [
    "Launch Leti (Windows).bat",
    "Launch Leti (macOS).command",
    "scripts/install_linux_launcher.sh",
    "scripts\\install_windows_launcher.ps1",
    "config\\settings.yaml",
    "config\\permissions.yaml",
    "README.md",
])
def test_every_file_the_guide_tells_you_to_use_exists(guide, named):
    assert named in guide, f"the guide no longer mentions {named}"
    assert (PROJECT_ROOT / named.replace("\\", "/")).exists(), f"{named} does not exist"


@pytest.mark.parametrize("named", [
    "config\\settings.local.yaml",   # written the first time a setting is saved
    "data\\launch_setup.json",       # written by the launcher on first setup
    "data\\gui_remote_token.txt",    # written when the GUI server first starts
    "data\\personality.json",
    "data\\user_profile.json",
    "data\\todo_list.json",
    "data\\contacts.json",
    "data\\scheduled_tasks.json",
    "data\\task_history.json",
    "data\\security_snapshots\\",
    "logs\\audit.log",
    "logs\\leti.log",
])
def test_every_runtime_file_the_guide_names_is_one_the_code_writes(guide, named):
    """These do not exist in a fresh checkout, so the check is that the code
    names them - a path the guide invented would match nothing."""
    basename = named.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    assert named in guide or basename in guide, f"the guide no longer mentions {named}"
    wanted = named.replace("\\", "/").rstrip("/")
    hits = [p for p in PROJECT_ROOT.rglob("*.py")
            if "leti_env" not in p.parts and "leti_runtime" not in p.parts
            and wanted in p.read_text(encoding="utf-8", errors="replace")]
    assert hits, f"nothing in the code writes {named}"


def test_the_only_startup_option_is_the_one_the_guide_documents(guide):
    """main.py takes --mode and nothing else. A guide that names a flag Leti does
    not have sends people to a traceback."""
    source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    flags = set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', source))
    assert flags == {"--mode"}, f"main.py now also takes {sorted(flags - {'--mode'})}"
    modes = set(re.search(r'choices=\[([^\]]+)\]', source).group(1).replace('"', "").split(", "))
    for mode in modes:
        assert f"--mode {mode}" in guide, f"the guide does not document --mode {mode}"
    for documented in re.findall(r"--mode ([a-z-]+)", guide):
        assert documented in modes, f"the guide documents --mode {documented}, which does not exist"


@pytest.mark.parametrize("command", ["/settings", "/audio"])
def test_the_typed_commands_the_guide_names_are_real(guide, command):
    source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    assert f'startswith("{command}")' in source, f"{command} is no longer handled"
    assert command in guide


def test_the_settings_the_guide_tells_you_to_edit_by_hand_are_real(guide, settings):
    """Two settings are named as things to change in settings.local.yaml."""
    from core.settings_editor import SECTION_SCHEMAS

    assert "enable_remote_access" in guide
    assert "enable_remote_access" in [f["key"] for f in SECTION_SCHEMAS["gui"]["fields"]]
    assert "gui:\n      port: 8500" in guide
    assert "port" in [f["key"] for f in SECTION_SCHEMAS["gui"]["fields"]]
    assert "num_ctx" in guide and "num_ctx" in settings["ollama"]


def test_the_guide_counts_the_settings_that_are_actually_in_use(guide, settings):
    def leaves(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from leaves(value)
        else:
            yield node

    count = len(list(leaves(settings)))
    assert f"{count} settings in use" in guide, (
        f"config/settings.yaml now has {count} active settings; the guide says otherwise")


def test_the_connections_the_guide_lists_are_the_ones_that_exist(guide):
    from core.settings_editor import SECTION_SCHEMAS

    connections = {name for name, schema in SECTION_SCHEMAS.items()
                   if schema.get("kind") == "connection"}
    labels = {"email": "Email", "calendar": "Calendar", "zoom": "Zoom",
              "teams": "Microsoft Teams", "github": "GitHub", "trading": "Market data",
              "youtube": "YouTube", "reddit": "Reddit"}
    assert connections == set(labels), (
        f"the connections have changed: {connections ^ set(labels)}")
    for label in labels.values():
        assert label in guide, f"the guide no longer lists the {label} connection"


# --- The promise that it is safe to share -----------------------------------------

def test_the_guide_contains_no_credential(guide):
    """Section 19 of the audit brief, and the one thing a user guide must never do.

    It talks about passwords and tokens at length, which is the point, so this
    looks for a value rather than for the words.
    """
    assignment = re.compile(
        r"(?i)(password|api[_-]?key|api[_-]?secret|token|client[_-]?secret|bearer)"
        r"\s*[:=]\s*[\"']?([A-Za-z0-9_\-/+.]{12,})")
    found = [m.group(0) for m in assignment.finditer(guide)]
    assert not found, f"the guide appears to contain a credential: {found}"
    # A high-entropy hex or base64 run, which is what a real token looks like.
    assert not re.search(r"\b[0-9a-f]{32,}\b", guide), "a hex secret is in the guide"


def test_the_guide_is_plain_ascii_text(guide):
    """A .txt somebody opens in Notepad. Non-ASCII survives that badly."""
    bad = {ch for ch in guide if ord(ch) > 127}
    assert not bad, f"non-ASCII characters in the guide: {sorted(bad)}"
    assert "\t" not in guide, "tabs render differently everywhere"
