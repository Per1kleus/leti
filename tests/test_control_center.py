"""The Permission Center and the Diagnostics panel.

The Permission Center's whole job is to be a view of SafetyGuard's own settings
rather than a second set of rules, so most of these check that what it shows and
what it writes are the same thing the guard reads. Diagnostics' whole job is to be
honest, so most of those check it says "Unavailable" rather than inventing.
"""
from __future__ import annotations

import ast
import inspect
import sys
import types
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

import main  # noqa: E402
from core import diagnostics, permission_center  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    sys.argv = ["main.py"]
    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


# --- Permission Center reads the guard's own settings -------------------------------

def test_every_category_maps_onto_registered_tools(registry):
    """A category naming a tool that does not exist would promise a control over
    something Leti cannot do."""
    unknown = []
    for name, spec in permission_center.CATEGORIES.items():
        for action, tools in spec["actions"].items():
            unknown += [f"{name}/{action}: {t}" for t in tools if registry.get(t) is None]
    assert not unknown, unknown


def test_the_level_shown_is_what_the_guard_will_actually_do(registry):
    """Not a parallel opinion: derived from the tool's class and the confirmation
    setting, which is exactly what SafetyGuard consults."""
    confirming = permission_center.confirming_classes()

    for tool_name in ("read_file", "delete_file", "send_email", "web_search"):
        action = permission_center.tool_class(tool_name)
        expected = permission_center.ASKS if action in confirming else permission_center.ALLOWED
        assert permission_center.level_for(tool_name) == expected, tool_name


def test_irreversible_actions_always_show_as_asking():
    """The guard asks for critical whether or not it is listed; showing otherwise
    would misrepresent what happens."""
    assert "critical" in permission_center.confirming_classes()


def test_an_unclassified_tool_is_shown_as_asking():
    """The guard treats it as critical, so this must not show it as automatic."""
    assert permission_center.level_for("a_tool_that_does_not_exist") == permission_center.ASKS


def test_a_category_reports_the_strictest_thing_in_it(registry):
    """Saying a whole capability is automatic while part of it asks would be a
    comfortable lie."""
    data = permission_center.overview(registry)
    for category in data["categories"]:
        if any(a["level"] == permission_center.ASKS for a in category["actions"]):
            assert category["level"] == permission_center.ASKS, category["category"]


def test_the_overview_explains_itself_in_plain_words(registry):
    data = permission_center.overview(registry)
    files = next(c for c in data["categories"] if c["category"] == "Files")
    delete = next(a for a in files["actions"] if a["action"] == "Delete")

    assert "Leti" in delete["explanation"]
    assert delete["explanation"].endswith(".")
    assert "critical" not in delete["explanation"], "an internal class leaked into the wording"


def test_the_overview_names_the_tools_behind_each_control(registry):
    data = permission_center.overview(registry)
    files = next(c for c in data["categories"] if c["category"] == "Files")
    read = next(a for a in files["actions"] if a["action"] == "Read")

    assert "read_file" in read["tools"]


# --- Changing a permission changes the guard's settings -------------------------------

def test_changing_a_class_writes_the_setting_the_guard_reads(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr("core.settings_editor._load_overrides", lambda: {})
    monkeypatch.setattr("core.settings_editor._save_overrides",
                        lambda data: saved.update(data))
    monkeypatch.setattr("core.config_loader.reload_settings", lambda: None)

    result = permission_center.set_class_confirmation("execute", True)

    assert result["ok"] is True
    assert "execute" in saved["safety"]["require_confirmation_for"], saved


def test_irreversible_actions_cannot_be_switched_off():
    result = permission_center.set_class_confirmation("critical", False)

    assert result["ok"] is False
    assert "always ask" in result["error"]


def test_an_unknown_class_is_refused():
    assert permission_center.set_class_confirmation("whenever", True)["ok"] is False


def test_reclassifying_an_unknown_tool_is_refused(registry):
    result = permission_center.set_tool_class("not_a_tool", "read", registry)
    assert result["ok"] is False


def test_an_unknown_class_for_a_tool_is_refused(registry):
    result = permission_center.set_tool_class("read_file", "whenever", registry)
    assert result["ok"] is False


def test_the_permission_center_keeps_no_rules_of_its_own():
    """Structural: it reads config, it does not decide. A second source of truth
    is the one thing this must never become."""
    tree = ast.parse(inspect.getsource(permission_center))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("authorize", "audit_result", "run"):
        assert forbidden not in called, f"the permission center calls {forbidden}()"


def test_changing_a_permission_is_itself_a_critical_action(registry):
    """Nothing should be able to loosen its own leash quietly."""
    from core.config_loader import get_permissions

    assert get_permissions()["tools"]["change_permission"]["action"] == "critical"
    assert get_permissions()["tools"]["review_permissions"]["action"] == "read"


# --- Diagnostics is honest ----------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_metrics():
    diagnostics.reset()
    yield
    diagnostics.reset()


def test_with_nothing_measured_everything_says_unavailable():
    performance = diagnostics.performance_section()

    assert performance["last_response_seconds"] == diagnostics.UNAVAILABLE
    assert performance["last_routing_ms"] == diagnostics.UNAVAILABLE
    assert performance["responses_measured"] == 0


def test_recorded_timings_are_reported_as_measured():
    diagnostics.record_turn(1.25, 400)
    diagnostics.record_routing(16, 123, 0.00031, ["read_file"])
    diagnostics.record_tool("read_file", 0.42, True)

    performance = diagnostics.performance_section()
    assert performance["last_response_seconds"] == 1.25
    assert performance["last_routing_ms"] == 0.31
    assert "read_file" in performance["last_tool"]


def test_unmeasurable_metrics_are_never_invented():
    """Responses are not streamed and the server reports no token counts, so a
    figure for either would be a guess wearing a unit."""
    performance = diagnostics.performance_section()

    assert performance["time_to_first_token"] == diagnostics.UNAVAILABLE
    assert performance["tokens_per_second"] == diagnostics.UNAVAILABLE
    assert "not streamed" in performance["time_to_first_token_note"]


def test_the_gpu_is_not_probed_to_fill_the_panel():
    """nvidia-smi costs the better part of a second and this panel refreshes."""
    resources = diagnostics.resources_section()

    assert resources["gpu"] == diagnostics.UNAVAILABLE
    assert resources["vram"] == diagnostics.UNAVAILABLE
    assert "not done" in resources["gpu_note"]


def test_real_resource_numbers_are_real():
    resources = diagnostics.resources_section()

    assert 0 <= resources["cpu_percent"] <= 100 * 64
    assert 0 < resources["ram_percent"] <= 100
    assert resources["leti_ram_mb"] > 0


def test_the_model_section_reports_the_configured_model():
    model = diagnostics.model_section()

    from core.config_loader import get_settings
    assert model["reasoning_model"] == get_settings()["ollama"]["reasoning_model"]
    assert model["context_size"] == get_settings()["ollama"]["num_ctx"]


def test_ollama_is_not_pinged_to_fill_the_panel():
    """It reports what the last real request found, rather than making one."""
    assert diagnostics.model_section()["connected"] == diagnostics.UNAVAILABLE

    diagnostics.record_turn(0.5)
    assert diagnostics.model_section()["connected"] is True


def test_diagnostics_calls_no_model_and_runs_no_loop():
    """Checked structurally: the module's own prose says it takes no screenshots
    and makes no model calls, so a text search would only find its explanation."""
    tree = ast.parse(inspect.getsource(diagnostics))

    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("chat", "handle_user_input", "sleep"):
        assert forbidden not in called, f"diagnostics calls {forbidden}()"

    assert not any(isinstance(n, ast.While) for n in ast.walk(tree)), "diagnostics loops"

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
    for heavy in ("pyautogui", "PIL", "httpx", "subprocess"):
        assert heavy not in imported, f"diagnostics imports {heavy}"


def test_metrics_are_bounded_and_never_written_to_disk():
    for i in range(500):
        diagnostics.record_turn(0.1)
        diagnostics.record_tool(f"t{i}", 0.1, True)

    assert len(diagnostics._turns) == diagnostics._KEEP
    assert len(diagnostics._tools) == diagnostics._KEEP

    tree = ast.parse(inspect.getsource(diagnostics))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "open" not in called and "atomic_write_text" not in called


def test_a_snapshot_has_every_section(registry):
    snapshot = diagnostics.snapshot(registry)

    for section in ("model", "performance", "resources", "tools", "autonomy", "watches"):
        assert section in snapshot, section
    assert snapshot["tools"]["registered"] == len(registry.names())
