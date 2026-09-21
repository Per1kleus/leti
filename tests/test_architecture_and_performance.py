"""One of each thing, and nothing running when nothing is happening.

Two properties are checked here, and both of them are the kind that erode
quietly. The first is that Leti has one orchestrator, one planner, one memory,
one task manager, one scheduler, one CRM, one permission system, one SafetyGuard,
one Connections Manager, one Computer Use engine, one diagnostics, one voice
pipeline and one interface state - not one plus whichever the newest module
brought with it. The second is that nothing costs anything while Leti is idle:
no loops, no timers, no watchers, no polling, no permanent index.
"""
from __future__ import annotations

import ast
import pathlib
import re
import threading
import time

import pytest

CORE = pathlib.Path("core")
NEW_MODULES = [
    "core/context_engine.py", "core/verification.py", "core/entities.py",
    "core/world_state.py", "core/objectives.py", "core/connections.py",
    "core/autonomy.py",
]


def _tree(path):
    return ast.parse(pathlib.Path(path).read_text())


def _calls(path):
    """Every attribute call and bare call name in a module's code."""
    names = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.add(node.func.id)
    return names


def _imports(path):
    found = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


# --- One of each ----------------------------------------------------------------------

SINGLETONS = {
    "orchestrator": ["core/orchestrator.py"],
    "task manager": ["core/task_manager.py"],
    "scheduler": ["core/system_scheduler.py", "tools/scheduler.py"],
    "SafetyGuard": ["core/safety_guard.py"],
    "permission view": ["core/permission_center.py"],
    "tool router": ["core/tool_router.py"],
    "computer use engine": ["core/computer_use.py"],
    "workflow store": ["core/workflows.py"],
    "watch system": ["core/watches.py"],
    "session memory": ["memory/session_memory.py"],
    "vector memory": ["memory/vector_store.py"],
    "diagnostics": ["core/diagnostics.py"],
}


def test_each_capability_still_has_exactly_one_home():
    for capability, paths in SINGLETONS.items():
        for path in paths:
            assert pathlib.Path(path).exists(), f"{capability}: {path} is gone"


def test_no_new_module_defines_a_second_orchestrator_or_task_manager():
    for path in NEW_MODULES:
        classes = {n.name for n in ast.walk(_tree(path)) if isinstance(n, ast.ClassDef)}
        for forbidden in ("Orchestrator", "TaskRunner", "SafetyGuard", "ToolRegistry",
                          "SchedulerRunner", "SessionMemory", "VectorMemory"):
            assert forbidden not in classes, f"{path} defines a second {forbidden}"


def test_no_new_module_builds_its_own_store():
    """A store means a path, a writer and a format. These read what exists."""
    for path in NEW_MODULES:
        calls = _calls(path)
        for forbidden in ("atomic_write_text", "write_text", "mkdir", "connect"):
            assert forbidden not in calls, f"{path} writes its own store ({forbidden})"


def test_no_new_module_calls_the_model():
    for path in NEW_MODULES:
        assert "core.llm_client" not in _imports(path), f"{path} talks to the model"
        assert "OllamaClient" not in _calls(path), f"{path} builds a model client"


def test_there_is_one_verification_vocabulary():
    """core/coding.py imports it; nothing else redefines it."""
    definitions = []
    for path in CORE.glob("*.py"):
        source = path.read_text()
        if re.search(r'^VERIFIED\s*=\s*["\']VERIFIED["\']', source, re.M):
            definitions.append(path.name)
    assert definitions == ["verification.py"], definitions


def test_there_is_one_failure_classifier():
    """core/task_manager.py asks core/world_state.py rather than keeping a list."""
    assert "_RECOVERABLE" not in pathlib.Path("core/task_manager.py").read_text()
    assert "unlocks_recovery" in _calls("core/task_manager.py")


def test_there_is_one_place_that_decides_between_entity_candidates():
    business = pathlib.Path("core/business.py").read_text()
    assert "entities.choose" in business
    # The old name-overlap decision is gone, not sitting alongside the new one.
    assert "clearly the best match" not in business


def test_the_context_engine_owns_no_memory_of_its_own():
    calls = _calls("core/context_engine.py")
    for forbidden in ("add_memory", "add_turn", "note_referents", "save_tasks",
                      "embed", "index"):
        assert forbidden not in calls, f"the context engine calls {forbidden}"


def test_connection_state_lives_in_one_place():
    """core/connections.py asks settings_editor; it stores nothing durable."""
    assert _imports("core/connections.py") <= {
        "__future__", "logging", "time", "typing", "core"}


# --- Nothing runs while nothing is happening ---------------------------------------------

BACKGROUND = ("Thread", "Timer", "Process", "Pool", "create_task", "ensure_future",
              "run_in_executor", "schedule", "set_interval", "spawn")


@pytest.mark.parametrize("path", NEW_MODULES)
def test_no_new_module_starts_anything_in_the_background(path):
    calls = _calls(path)
    for forbidden in BACKGROUND:
        assert forbidden not in calls, f"{path} starts background work ({forbidden})"


@pytest.mark.parametrize("path", NEW_MODULES)
def test_no_new_module_sleeps_or_loops_waiting(path):
    source = pathlib.Path(path).read_text()
    assert "while True" not in source, f"{path} has an unbounded loop"
    assert "sleep(" not in source, f"{path} sleeps"


@pytest.mark.parametrize("path", NEW_MODULES)
def test_no_new_module_does_work_at_import_time(path):
    """Importing must be cheap: main.py imports everything at startup."""
    started = time.perf_counter()
    module = path.replace("/", ".")[:-3]
    __import__(module)
    assert time.perf_counter() - started < 0.5


def test_importing_every_new_module_starts_no_threads():
    before = threading.active_count()
    for path in NEW_MODULES:
        __import__(path.replace("/", ".")[:-3])
    assert threading.active_count() == before


def test_nothing_new_builds_a_permanent_index_or_a_vector_database():
    for path in NEW_MODULES:
        source = pathlib.Path(path).read_text().lower()
        for forbidden in ("chromadb", "faiss", "sqlite3", "lancedb", "sentence_transformers"):
            assert forbidden not in source, f"{path} mentions {forbidden}"


# --- The context window is not the answer ---------------------------------------------------

def test_num_ctx_was_not_raised_to_pay_for_any_of_this():
    from core.config_loader import get_settings

    assert int(get_settings()["ollama"]["num_ctx"]) == 28_672


def test_no_tool_was_added_to_default_modes_fallback():
    """Default Mode's schema budget is the constraint everything here worked
    around. Adding one tool would have spent what is left of it."""
    from unittest.mock import MagicMock

    import json

    import main
    from core import modes

    registry = main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())
    visible = modes.visible_tools(registry, modes.DEFAULT)
    chars = len(json.dumps(registry.schemas_for(visible)))
    # The count is what the canary is really protecting: a new tool is what
    # would eat the headroom, and no upgrade since has added one.
    assert len(visible) == 129, f"Default Mode now sees {len(visible)} tools, not 129"
    assert chars == 89_345, (
        f"Default Mode's fallback is now {chars:,} characters, not 89,345. Editing a "
        "description is allowed and this number moves with it; adding a tool is not.")
    # And the headroom it buys, which is the thing that actually has to hold.
    from core.config_loader import get_settings

    headroom = int(get_settings()["ollama"]["num_ctx"]) - chars // 4 - 6000
    assert headroom > 0, f"Default Mode has {headroom} tokens of headroom left"


# --- Mode activation is unchanged ---------------------------------------------------------

def test_default_mode_still_has_no_way_to_switch_itself():
    from unittest.mock import MagicMock

    import main
    from core import modes

    registry = main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())
    assert modes.SWITCH_TOOL not in modes.visible_tools(registry, modes.DEFAULT)


@pytest.mark.parametrize("text", [
    "check my customer emails", "what is my pipeline worth",
    "fix the bug in auth.py", "run the tests", "send an invoice to Acme",
    "look at my leads and tell me who to call", "commit this and push it",
    "run full leti diagnostics", "what is the health of my business",
])
def test_no_amount_of_business_or_coding_language_activates_a_mode(text):
    from core import intent as intent_reader

    assert intent_reader.mode_command(text) is None


def test_the_model_is_never_asked_which_mode_to_be_in():
    source = pathlib.Path("core/orchestrator.py").read_text()
    switch = re.search(r"def _switch_mode.*?(?=\n    (?:async )?def )", source, re.S).group(0)
    assert "llm_client" not in switch and "chat(" not in switch


def test_mode_activation_is_matched_before_any_model_call():
    source = pathlib.Path("core/orchestrator.py").read_text()
    turn = re.search(r"async def _handle_one_turn.*?(?=\n    (?:async )?def )",
                     source, re.S).group(0)
    assert turn.index("mode_command") < turn.index("_tool_calling_loop")
