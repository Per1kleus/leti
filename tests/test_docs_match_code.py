"""The numbers in the docs are the numbers the code actually produces.

Three of Leti's documents state measured facts about the tool list: README's
"Fitting your GPU" section, the `num_ctx` comment in config/settings.yaml, and
core/tool_router.py's module docstring all quote how many tools there are and
how many characters their schemas serialise to. Those numbers decide `num_ctx`,
and `num_ctx` decides whether Ollama silently truncates the tool list - so a
stale number here is not a typo, it is the first step of the failure mode the
comments themselves describe.

They have gone stale twice: once at 103 tools and again at 119. Nothing said so
either time, because prose does not run. This does.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import main
from core.config_loader import get_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Documents that quote the measured tool-schema size.
DOCUMENTS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "config" / "settings.yaml",
    PROJECT_ROOT / "core" / "tool_router.py",
]

# What main.py reserves for everything that is not a tool schema - system prompt,
# personality, profile, recalled memories, the rolling buffer, tool results. Kept
# in step with _warn_if_context_is_too_small by the test below.
HEADROOM_TOKENS = 6000

CLAIM = re.compile(r"([\d,]+) tools serialise to ([\d,]+) characters")


def _flatten(text: str) -> str:
    """Strip comment/emphasis markers and line wrapping so a claim reads as one line."""
    return re.sub(r"\s+", " ", text.replace("#", " ").replace("*", " "))


@pytest.fixture(scope="module")
def registry():
    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


def test_documents_quote_the_real_tool_count_and_schema_size(registry):
    """The numbers in the docs are the ones a turn actually pays.

    Both are measured over the biggest MODE rather than the registry: since
    core/modes.py, the registry holds more tools than any single turn is shown,
    and quoting the registry would size num_ctx for a request that cannot happen.
    """
    from core import modes

    widest = max(modes.MODES,
                 key=lambda m: len(json.dumps(registry.schemas_for(
                     modes.visible_tools(registry, m)))))
    visible = modes.visible_tools(registry, widest)
    real_tools = len(visible)
    real_chars = len(json.dumps(registry.schemas_for(visible)))

    claims_per_document = {}
    for path in DOCUMENTS:
        found = CLAIM.findall(_flatten(path.read_text(encoding="utf-8")))
        claims_per_document[path.name] = len(found)
        for claimed_tools, claimed_chars in found:
            assert int(claimed_tools.replace(",", "")) == real_tools, (
                f"{path.name} says {claimed_tools} tools; the registry builds {real_tools}. "
                "Update the document rather than this test."
            )
            assert int(claimed_chars.replace(",", "")) == real_chars, (
                f"{path.name} says {claimed_chars} characters; the schemas are {real_chars:,}. "
                "Update the document rather than this test."
            )
    # Per document rather than as a total: a total of one-per-document is also
    # satisfied by one document claiming it twice and another not at all, which is
    # the case this is here to catch.
    silent = [name for name, count in claims_per_document.items() if count == 0]
    assert not silent, (
        f"{silent} are listed as documents that state the measured tool-schema size "
        "and no longer do. Either restore the claim or drop the document from "
        "DOCUMENTS."
    )


def test_num_ctx_still_fits_every_modes_tool_list(registry):
    """Routing sends a subset, but the fallback sends everything the MODE allows.

    Per mode rather than per registry, because the registry as a whole is never
    sent: Default Mode is not shown the coding tools and Coding Mode is not shown
    the weather. Every mode still has to fit on its own, so this checks each.
    """
    from core import modes

    num_ctx = int(get_settings().get("ollama", {}).get("num_ctx", 0))
    for name in modes.MODES:
        visible = modes.visible_tools(registry, name)
        tokens = len(json.dumps(registry.schemas_for(visible))) // 4
        assert num_ctx >= tokens + HEADROOM_TOKENS, (
            f"ollama.num_ctx is {num_ctx:,} but a full-fallback turn in {name} mode needs "
            f"about {tokens + HEADROOM_TOKENS:,} tokens ({tokens:,} of schemas plus "
            f"{HEADROOM_TOKENS:,} for everything else). Ollama truncates instead of "
            "erroring, so raise num_ctx."
        )


def test_the_documented_size_is_the_biggest_mode_not_the_registry(registry):
    """The number in the docs has to be the one that matters, which is the largest
    thing a turn can actually send. Registering a tool no mode shows would
    otherwise inflate a figure that decides num_ctx."""
    from core import modes

    widest = max(len(json.dumps(registry.schemas_for(modes.visible_tools(registry, m))))
                 for m in modes.MODES)
    assert widest <= len(json.dumps(registry.all_schemas()))
    claimed = [c for path in DOCUMENTS for c in CLAIM.findall(_flatten(path.read_text()))]
    assert claimed, "no document states the measured size"
    for _, chars in claimed:
        assert int(chars.replace(",", "")) == widest, (
            f"the documents quote {chars} characters; the largest mode sends {widest:,}.")


def test_readme_quotes_the_configured_num_ctx():
    num_ctx = int(get_settings().get("ollama", {}).get("num_ctx", 0))
    readme = (PROJECT_ROOT / "README.md").read_text()
    assert f"`ollama.num_ctx` is {num_ctx}" in readme, (
        f"README should name the configured num_ctx ({num_ctx})."
    )
    assert f"`num_ctx` of {num_ctx}" in readme


def test_headroom_here_matches_the_startup_warning():
    """This test's arithmetic is only meaningful while it matches main.py's."""
    source = (PROJECT_ROOT / "main.py").read_text()
    assert f"headroom = {HEADROOM_TOKENS}" in source


def test_the_fourteen_b_option_is_qwen25_14b():
    """One exact identifier, everywhere the 14B is offered or documented."""
    from core import model_setup

    fourteen_b = [c for c in model_setup.CANDIDATES if c["label"] == "14B"]
    assert [c["model"] for c in fourteen_b] == ["qwen2.5:14b"]

    for path in [PROJECT_ROOT / "README.md", PROJECT_ROOT / "config" / "settings.yaml"]:
        text = path.read_text()
        for match in re.findall(r"\bqwen[\d.:]*14b\b", text, flags=re.IGNORECASE):
            assert match == "qwen2.5:14b", f"{path.name} references {match!r}"
