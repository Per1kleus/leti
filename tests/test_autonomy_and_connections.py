"""Asking well, and knowing what is not set up.

Two modules, one shared constraint: neither of them decides anything. SafetyGuard
still says whether to ask; the Connections Manager still holds the credentials.
Several of these exist specifically to fail if either module ever grows teeth.
"""
from __future__ import annotations

import pytest

from core import autonomy, connections
from core.safety_guard import RiskTier


# --- The confirmation says what it will actually do ------------------------------------

def test_a_mass_email_says_how_many_and_to_whom():
    prompt = autonomy.confirmation_prompt(
        "send the prepared email", "send_email",
        {"to": "a@x.com, b@x.com, c@x.com, d@x.com", "subject": "Q3 follow-up"},
        "external")
    assert "4 recipient(s)" in prompt and "Q3 follow-up" in prompt


def test_records_that_cannot_be_emailed_are_named_before_the_yes():
    prompt = autonomy.confirmation_prompt(
        "send the prepared email", "send_email",
        {"to": "a@x.com, b@x.com, Jane Doe, No Address Ltd"}, "external")
    assert "2 of them have no usable email address" in prompt
    assert "skipped" in prompt


def test_an_irreversible_action_says_so():
    prompt = autonomy.confirmation_prompt("delete the file /tmp/x", "delete_file",
                                          {"path": "/tmp/x"}, "critical")
    assert "cannot be undone" in prompt


def test_overwriting_a_file_is_not_promised_to_be_undoable():
    """Telling somebody a write can be undone is reassurance about the one thing
    they should hesitate over."""
    prompt = autonomy.confirmation_prompt("write to notes.md", "write_file",
                                          {"path": "notes.md"}, "modify")
    assert "can be undone" not in prompt


def test_a_reversible_external_action_says_it_can_be_undone():
    prompt = autonomy.confirmation_prompt("add a contact", "add_contact",
                                          {"name": "Chris"}, "external")
    assert "can be undone" in prompt


def test_the_prompt_always_ends_with_the_question():
    for tool, args, tier in (("send_email", {"to": "a@b.com"}, "external"),
                             ("delete_file", {"path": "/x"}, "critical"),
                             ("run_shell_command", {"command": "ls"}, "execute"),
                             ("some_unknown_tool", {}, "modify")):
        assert autonomy.confirmation_prompt("do the thing", tool, args, tier
                                            ).endswith("Should I go ahead?")


def test_nothing_is_invented_when_the_arguments_say_nothing():
    detail = autonomy.consequences("mystery_tool", {}, "modify")
    assert detail["facts"] == [] and detail["problems"] == []


def test_consequences_never_raise_on_odd_arguments():
    for arguments in (None, {"to": None}, {"to": 5}, {"to": ["a@b.com", None]},
                      {"count": "many"}, {"path": 12}):
        detail = autonomy.consequences("send_email", arguments, "external")
        assert detail["band"] in (autonomy.LOW, autonomy.MEDIUM, autonomy.HIGHER,
                                  autonomy.HIGH)


# --- It decides nothing -----------------------------------------------------------------

def _code_names(path):
    """Every name, attribute and imported module in a file's CODE.

    Deliberately not a substring search over the source: these modules document
    what they are not allowed to do, and a docstring saying "this never touches
    permissions.yaml" would fail a grep for permissions.yaml.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(path).read_text())
    names, imports = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            pass                      # string literals are content, not behaviour
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return names, imports


def test_the_module_cannot_decide_whether_to_ask():
    """No function here decides anything - it only phrases."""
    names, imports = _code_names("core/autonomy.py")
    for forbidden in ("authorize", "requires_confirmation", "PermissionDenied",
                      "ConfirmationDenied", "get_permissions"):
        assert forbidden not in names, f"core/autonomy.py defines or calls {forbidden}"
    for forbidden in ("core.safety_guard", "core.config_loader", "core.permission_center"):
        assert forbidden not in imports, f"core/autonomy.py imports {forbidden}"


def test_the_bands_are_a_description_of_safety_guard_s_tiers():
    described = {b["tier"] for b in autonomy.describe_bands()}
    assert described == {t.value for t in RiskTier
                         if t.value not in ("forbidden",)} - {"forbidden"}
    for band in autonomy.describe_bands():
        assert "safety_guard" in band["decided_by"]


@pytest.mark.asyncio
async def test_an_allowed_low_risk_action_is_still_not_asked_about(guard_factory):
    """The other half of asking well: not asking. The wording layer must not add
    a prompt where SafetyGuard did not want one."""
    guard, prompts = guard_factory(confirm_classes=["critical"])
    auth = await guard.authorize("read_file", {"path": "README.md"})
    assert auth.execute is True and prompts == []


@pytest.mark.asyncio
async def test_the_guard_still_asks_and_now_says_more(guard_factory):
    guard, prompts = guard_factory(confirm_classes=["external", "critical"])
    await guard.authorize("send_email", {"to": "a@x.com, b@x.com", "subject": "hello"})
    assert len(prompts) == 1
    assert "2 recipient(s)" in prompts[0] and prompts[0].endswith("Should I go ahead?")


@pytest.mark.asyncio
async def test_a_broken_wording_layer_never_lets_a_call_through(guard_factory, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("the phrasing broke")

    monkeypatch.setattr(autonomy, "confirmation_prompt", explode)
    guard, prompts = guard_factory(confirm=False, confirm_classes=["critical"])
    from core.safety_guard import ConfirmationDenied

    with pytest.raises(ConfirmationDenied):
        await guard.authorize("delete_file", {"path": "/tmp/x"})
    assert len(prompts) == 1 and "Should I go ahead?" in prompts[0]


# --- Connections ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _forget_outcomes():
    connections.forget_outcomes()
    yield
    connections.forget_outcomes()


def test_an_unconfigured_connection_says_what_will_fail_and_how_to_fix_it(monkeypatch):
    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: False)
    state = connections.status("email")
    assert state["state"] == connections.NOT_CONFIGURED
    assert "will fail" in state["detail"] and "Connections" in state["fix"]


def test_a_configured_but_unused_connection_is_untested_not_working(monkeypatch):
    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: True)
    state = connections.status("email")
    assert state["state"] == connections.CONFIGURED
    assert "untested" in state["detail"]


def test_a_real_success_makes_it_working(monkeypatch):
    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: True)
    connections.record_outcome("email", True)
    assert connections.status("email")["state"] == connections.WORKING


def test_a_real_failure_makes_it_failing_and_says_so(monkeypatch):
    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: True)
    connections.record_outcome("email", False, "authentication rejected")
    state = connections.status("email")
    assert state["state"] == connections.FAILING
    assert "authentication rejected" in state["detail"]


def test_an_old_outcome_stops_counting(monkeypatch):
    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: True)
    connections.record_outcome("email", True)
    later = __import__("time").time() + connections.OUTCOME_TTL_SECONDS + 1
    assert connections.status("email", now=later)["state"] == connections.CONFIGURED


def test_one_failure_does_not_stop_leti_trying_again(monkeypatch):
    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: True)
    connections.record_outcome("email", False, "timeout")
    assert connections.available("email") is True


def test_nothing_is_contacted_to_answer_the_question(monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("connection awareness made a network call")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    connections.overview()
    connections.summary()
    for tool in ("send_email", "schedule_meeting", "github", "read_file"):
        connections.warning_for(tool)


def test_no_credential_is_ever_returned(monkeypatch):
    """The Connections Manager holds the secrets; this returns states."""
    import json

    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: True)
    blob = json.dumps(connections.overview()) + json.dumps(connections.summary())
    for word in ("password", "app_password", "token", "api_key", "secret", "client_secret"):
        assert word not in blob


def test_the_module_never_reads_a_credential_itself():
    """It asks settings_editor whether a section is filled in. It never opens the
    settings, so there is nothing here for a credential to leak out of."""
    names, imports = _code_names("core/connections.py")
    for forbidden in ("get_settings", "get_section", "read_text", "safe_load",
                      "update_section"):
        assert forbidden not in names, f"core/connections.py calls {forbidden}"
    assert imports <= {"logging", "time", "typing", "core", "__future__"}, (
        f"unexpected imports: {imports}")


def test_a_tool_that_needs_no_account_produces_no_warning():
    assert connections.warning_for("read_file") == ""
    assert connections.needed_by("read_file") is None


def test_a_tool_whose_account_is_missing_warns_before_it_is_tried(monkeypatch):
    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: False)
    assert "no email connection is set up" in connections.warning_for("send_email").lower()


def test_an_unknown_capability_is_named_rather_than_assumed_fine():
    state = connections.status("telepathy")
    assert state["state"] == connections.NOT_CONFIGURED and state["problem"]


def test_every_capability_maps_to_a_real_settings_section():
    from core.settings_editor import SECTION_SCHEMAS

    for name, spec in connections.CAPABILITIES.items():
        assert spec["section"] in SECTION_SCHEMAS, f"{name} points at a section that is gone"


def test_every_capability_names_real_tools():
    """The mirror of the test above, on the side that actually went wrong.

    A capability names the tools that need it, and that is how "the calendar is
    not set up yet" gets said before the tool is tried rather than after it fails.
    A tool renamed without updating the table takes its capability's cover with
    it silently: check_calendar_availability and get_market_data were both gone
    for long enough that nothing noticed.
    """
    from unittest.mock import MagicMock

    import main

    registry = main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())
    real = set(registry.names())
    gone = [(name, tool) for name, spec in connections.CAPABILITIES.items()
            for tool in spec["tools"] if tool not in real]
    assert not gone, f"CAPABILITIES names tools that no longer exist: {gone}"


def test_asking_is_cheap_enough_to_do_on_every_tool_call():
    import time

    connections.warning_for("send_email")
    started = time.perf_counter()
    for _ in range(1000):
        connections.warning_for("send_email")
    assert (time.perf_counter() - started) / 1000 < 0.001
