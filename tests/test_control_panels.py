"""The five control panels, and what they must keep being views OF.

Every one of them edits something that already existed before it did. What these
protect is that none of them grew a store, a schema or a default of its own:
Connections, AI settings and Voice all render core/settings_editor.py's schema,
Personality writes tools/personality.py's six dials, and Permissions changes the
one setting SafetyGuard reads.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from core import settings_editor as se

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUD = (PROJECT_ROOT / "gui" / "hud.html").read_text()
API = (PROJECT_ROOT / "gui" / "api.py").read_text()

FIVE_CONTROLS = {
    "connectionsBtn": "openConnections",
    "aiSettingsBtn": "openAiSettings",
    "voiceSettingsBtn": "openVoiceSettings",
    "personalityBtn": "openPersonality",
    "permissionsBtn": "openPermissions",
}


def test_the_right_column_has_exactly_the_five_controls():
    grid = re.search(r'<div class="control-grid">(.*?)</div>\s*<div class="control-legend"',
                     HUD, re.S)
    assert grid, "the control grid moved; this test needs updating"
    ids = re.findall(r'id="(\w+)"', grid.group(1))
    assert ids == list(FIVE_CONTROLS), ids


def test_each_control_opens_its_own_panel():
    for element, opener in FIVE_CONTROLS.items():
        assert f"getElementById('{element}').onclick = {opener}" in HUD, element
        assert re.search(rf"function {opener}\(\)\{{", HUD), opener


def test_every_panel_has_an_overlay_and_a_way_out():
    for overlay in ("connections", "ai", "voice", "personality", "permissions"):
        assert f'id="{overlay}Overlay"' in HUD
        assert f'id="{overlay}Content"' in HUD
        assert f'id="{overlay}Close"' in HUD


# --- The settings panels are views of one schema ---------------------------------

def test_the_new_sections_point_at_configuration_that_already_exists():
    """Exposing a setting must not invent one. Every section added for the AI and
    Voice panels has to name a key that is already in config/settings.yaml."""
    settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text())
    for name in ("ai", "tool_routing", "voice", "stt", "tts"):
        schema = se.SECTION_SCHEMAS[name]
        path = schema.get("settings_path", [name])
        node = settings
        for key in path:
            assert isinstance(node, dict) and key in node, f"{name}: no {path} in settings.yaml"
            node = node[key]
        for field in schema["fields"]:
            assert field["key"] in node, f"{name}.{field['key']} is not a real setting"


def test_exposing_the_settings_changed_none_of_them():
    """The panel shows the values Leti is running with. A schema that carried its
    own defaults would quietly become a second source of them."""
    settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text())
    assert settings["ollama"]["reasoning_model"] == "qwen2.5:7b"
    assert settings["ollama"]["num_ctx"] == 28672
    assert settings["ollama"]["max_tool_iterations"] == 8
    assert settings["tool_routing"]["enabled"] is True
    for name in ("ai", "tool_routing", "voice", "stt", "tts"):
        for field in se.SECTION_SCHEMAS[name]["fields"]:
            assert "default" not in field, f"{name}.{field['key']} carries a default of its own"


SECRET = "hunter2-not-a-real-password"


def test_secrets_are_never_read_back(monkeypatch):
    """The Connections manager shows THAT a password is set, never what it is.

    Every secret field in every section is given a value first: on a machine with
    nothing configured this check would otherwise pass by having nothing to leak.
    """
    checked = 0
    for name, schema in se.SECTION_SCHEMAS.items():
        secrets = [f["key"] for f in schema["fields"] if f.get("secret")]
        if not secrets:
            continue
        path = schema.get("settings_path", [name])
        stored = {key: SECRET for key in secrets}
        node = stored
        for key in reversed(path):
            node = {key: node}
        monkeypatch.setattr(se, "get_settings", lambda node=node: node)

        section = se.get_section(name)
        assert SECRET not in str(section), f"{name} echoed a secret back"
        for field in section["fields"]:
            if field.get("secret"):
                checked += 1
                assert field["value"] is None, f"{name}.{field['key']} echoed a secret"
                assert field["is_set"] is True, f"{name}.{field['key']} lost 'is set'"
    assert checked >= 5, "no secret fields were exercised"


def test_connections_are_marked_in_the_schema_not_listed_in_the_page():
    """A list of integrations kept in the interface would be a second list to keep
    in step with the first."""
    connections = {s["name"] for s in se.list_sections() if s["kind"] == "connection"}
    assert {"email", "calendar", "zoom", "teams", "trading", "youtube", "reddit"} <= connections
    assert "s.kind === 'connection'" in HUD
    for name in connections:
        assert f"'{name}'" not in HUD, f"the page names the {name} section directly"


def test_one_form_builder_serves_every_panel():
    """Two builders would be two places for a secret to be echoed by accident."""
    assert HUD.count("function renderSectionForm(") == 1
    assert HUD.count("input.type = f.secret ? 'password' : 'text'") == 1


# --- Personality ------------------------------------------------------------------

def test_the_panel_presets_are_the_same_six_dials():
    from tools import personality

    assert set(personality.UI_PRESETS) == {"Balanced", "Professional", "Friendly",
                                           "Direct", "Concise"}
    for name, preset in personality.UI_PRESETS.items():
        assert set(preset) == set(personality.PARAM_NAMES), name
        assert all(0 <= v <= 10 for v in preset.values()), name


def test_the_presets_did_not_change_the_tool_contract():
    """PRESETS is apply_personality_preset's enum. Renaming or extending it to
    relabel some buttons would change a tool schema."""
    from tools import personality

    assert set(personality.PRESETS) == {"default", "witty_friend", "professional",
                                        "dry_and_sarcastic", "warm_and_supportive",
                                        "blunt_and_brief"}


def test_the_panel_writes_through_the_existing_store(tmp_path, monkeypatch):
    from tools import personality

    monkeypatch.setattr(personality, "_path", lambda: tmp_path / "personality.json")
    saved = personality.save_values({"humor": 9, "verbosity": 0})
    assert saved["humor"] == 9 and saved["verbosity"] == 0
    assert saved["warmth"] == personality.DEFAULT_PERSONALITY["warmth"]
    assert personality.load_values() == saved
    # Out-of-range values are clamped by the same rule set_personality uses.
    assert personality.save_values({"humor": 99})["humor"] == 10


def test_the_panel_has_no_personality_values_of_its_own():
    assert "UI_PRESETS" in API
    assert not re.search(r"presets\s*:\s*\{", HUD), "the page carries its own preset values"


# --- Permissions -------------------------------------------------------------------

def test_the_permission_panel_changes_the_guard_rather_than_shadowing_it():
    assert "permission_center.set_class_confirmation" in API
    permission_center = (PROJECT_ROOT / "core" / "permission_center.py").read_text()
    assert "require_confirmation_for" in permission_center
    # No second store: the panel reads and writes what is already on disk.
    assert "permissions.yaml" in permission_center


def test_critical_cannot_be_switched_off_from_the_panel():
    assert "const locked = cls === 'critical';" in HUD
    assert "if(!locked){" in HUD


# --- Nothing here polls -------------------------------------------------------------

def test_the_panels_read_when_opened_and_not_on_a_timer():
    """Diagnostics is the one panel that refreshes while open, and it clears its
    interval when closed. Nothing added since may hold a timer of its own."""
    intervals = re.findall(r"setInterval\(([^,]+),", HUD)
    assert sorted(intervals) == ["refreshDiagnostics", "refreshStats", "refreshWeather", "tickClock"], intervals
