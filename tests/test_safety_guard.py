"""Tests for the authorization choke point every tool call passes through.

These cover the four ways the guard has been wrong rather than the happy path:
a protected-path check defeated by '..', a shell command that names a protected
file in a way no path-shaped argument reveals, a dry_run that authorized AND
executed, and a single spoken "go ahead" standing in for approval of every
action the model chose for the rest of the turn.

Run with: pytest tests/
"""
from __future__ import annotations

import os

import pytest

from core.intent_signals import contains_request_approval, resolve_yes_no
from core.safety_guard import ConfirmationDenied, PermissionDenied

HOME = os.path.expanduser("~")




# --- Protected paths -----------------------------------------------------------

@pytest.mark.parametrize("path", [
    f"{HOME}/.ssh/id_rsa",
    f"{HOME}/.ssh/../.ssh/id_rsa",
    f"{HOME}/../{os.path.basename(HOME)}/.ssh/id_rsa",
    "~/.aws/credentials",
    "/etc/shadow",
    "/tmp/../etc/shadow",
    "/etc/../etc/shadow",
])
def test_protected_paths_survive_traversal(path, guard_factory):
    """'..' must not be a way around the protected list - these all name a
    protected file, however they're spelled."""
    guard, _ = guard_factory()
    assert guard._touches_protected_path(path) is not None, path


@pytest.mark.parametrize("path", [
    "/etcetera/notes.txt",   # merely starts with the string '/etc'
    "/System.md",
    f"{HOME}/notes.txt",
])
def test_protected_paths_dont_false_positive(path, guard_factory):
    """Prefix matching blocked unrelated paths whose names happen to start with a
    protected one. Comparison is per path segment now."""
    guard, _ = guard_factory()
    assert guard._touches_protected_path(path) is None, path


@pytest.mark.parametrize("command", [
    "cat ~/.ssh/id_rsa",
    "cat /etc/shadow",
    "echo pwned > /etc/hosts",
    "cp ~/.aws/credentials /tmp/exfil",
    "tar czf out.tgz /tmp/../etc",
])
def test_shell_commands_cannot_reach_protected_paths(command, guard_factory):
    """run_shell_command has no path-shaped argument to inspect, so the command
    string itself is tokenized and checked."""
    guard, _ = guard_factory()
    assert guard.check_hard_block("run_shell_command", {"command": command}) is not None


@pytest.mark.parametrize("command", ["ls -la ~/projects", "git status", "echo hello world"])
def test_ordinary_shell_commands_still_run(command, guard_factory):
    guard, _ = guard_factory()
    assert guard.check_hard_block("run_shell_command", {"command": command}) is None


@pytest.mark.parametrize("command", ["rm -rf /", "rm  -rf   /", "RM -RF /"])
def test_forbidden_patterns_ignore_whitespace_and_case(command, guard_factory):
    guard, _ = guard_factory()
    assert guard._matches_forbidden_shell_pattern(command) is not None


def test_forbidden_patterns_dont_block_relative_rm(guard_factory):
    """'rm -rf ./build' is ordinary work and must not match the 'rm -rf /' pattern."""
    guard, _ = guard_factory()
    assert guard._matches_forbidden_shell_pattern("rm -rf ./build") is None


# --- dry_run -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dry_run_authorizes_without_executing(guard_factory):
    """The regression that mattered most: dry_run used to return normally from
    authorize(), which the caller reads as 'go ahead and run it'."""
    guard, prompts = guard_factory(dry_run=True)
    auth = await guard.authorize("delete_file", {"path": "/tmp/whatever.txt"})
    assert auth.execute is False
    assert auth.description  # something to tell the user instead
    assert prompts == []     # nothing to confirm; nothing is going to happen


@pytest.mark.asyncio
async def test_dry_run_still_runs_read_only_tools(guard_factory):
    """Otherwise dry_run would make Leti unable to search or read the screen,
    which is the opposite of useful for testing prompts."""
    guard, _ = guard_factory(dry_run=True)
    auth = await guard.authorize("read_file", {"path": "/tmp/whatever.txt"})
    assert auth.execute is True


# --- Voice pre-approval --------------------------------------------------------

@pytest.mark.asyncio
async def test_preapproval_is_reported_as_spent(guard_factory):
    guard, prompts = guard_factory()
    auth = await guard.authorize("write_file", {"path": "/tmp/a.txt"}, preapproved=True)
    assert auth.execute is True
    assert auth.used_preapproval is True   # caller clears its flag on this
    assert prompts == []


@pytest.mark.asyncio
async def test_preapproval_never_covers_destructive_calls(guard_factory):
    """A spoken request that approves one thing must not silently authorize a
    delete the model chose on its own."""
    guard, prompts = guard_factory(confirm=False)
    with pytest.raises(ConfirmationDenied):
        await guard.authorize("delete_file", {"path": "/tmp/a.txt"}, preapproved=True)
    assert len(prompts) == 1               # it asked anyway


@pytest.mark.asyncio
async def test_preapproval_never_bypasses_a_hard_block(guard_factory):
    guard, _ = guard_factory()
    with pytest.raises(PermissionDenied):
        await guard.authorize("read_file", {"path": f"{HOME}/.ssh/id_rsa"}, preapproved=True)


@pytest.mark.parametrize("text", [
    "go ahead and move that file to the archive",
    "sure, send it",
    "yes, delete the old logs",
])
def test_request_approval_recognizes_real_approval(text):
    assert contains_request_approval(text) is True


@pytest.mark.parametrize("text", [
    "is that correct?",
    "whatever works for you",
    "that's fine by me I guess",
    "what does the y flag do",
])
def test_request_approval_ignores_answer_shaped_words(text):
    """These are words that show up in ordinary sentences. Reading them as
    standing permission to act would manufacture consent."""
    assert contains_request_approval(text) is False


@pytest.mark.parametrize("text,expected", [("y", True), ("correct", True), ("nope", False)])
def test_reply_context_still_accepts_short_answers(text, expected):
    """Answering a direct yes/no question is a different context - the full
    phrase list still applies there."""
    assert resolve_yes_no(text) is expected


def test_ambiguous_reply_fails_closed():
    assert resolve_yes_no("not sure") is None


# --- Audit log -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_audit_log_redacts_secret_values(tmp_path, guard_factory):
    """logs/audit.log is permanent, plaintext and unrotated. The fact that a form
    was filled is the auditable event; the password typed into it is not."""
    guard, _ = guard_factory()
    guard._audit_path = tmp_path / "audit.log"

    await guard.authorize(
        "browser_fill_form",
        {"selector": "#password", "value": "hunter2-my-real-password"},
        preapproved=True,
    )

    written = guard._audit_path.read_text()
    assert "hunter2" not in written
    assert "#password" in written          # the action itself stays auditable
    assert "redacted" in written


# --- Action classes ------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("tool,expected_ask", [
    ("read_file", False),          # read
    ("web_search", False),         # execute
    ("add_todo_item", False),      # execute - Leti's own state, reversible
    ("write_file", True),          # modify
    ("send_email", True),          # external
    ("delete_file", True),         # critical
])
async def test_default_confirmation_matches_the_action_class(tool, expected_ask, guard_factory):
    guard, prompts = guard_factory()
    await guard.authorize(tool, {"path": "/tmp/x", "to": "a@b.c"})
    assert bool(prompts) is expected_ask


@pytest.mark.asyncio
async def test_user_can_require_confirmation_for_more_classes(monkeypatch, guard_factory):
    """The point of the setting: someone who wants to approve every search can."""
    from core.config_loader import get_settings

    settings = get_settings()
    settings["safety"]["require_confirmation_for"] = ["execute", "modify", "external", "critical"]
    monkeypatch.setattr("core.safety_guard.get_settings", lambda: settings)

    guard, prompts = guard_factory()
    await guard.authorize("web_search", {"query": "x"})
    assert len(prompts) == 1


@pytest.mark.asyncio
async def test_critical_asks_even_when_removed_from_the_setting(monkeypatch, guard_factory):
    """A setting that could switch off the prompt for irreversible actions would
    defeat the point of having one."""
    from core.config_loader import get_settings

    settings = get_settings()
    settings["safety"]["require_confirmation_for"] = []
    monkeypatch.setattr("core.safety_guard.get_settings", lambda: settings)

    guard, prompts = guard_factory(confirm=False)
    with pytest.raises(ConfirmationDenied):
        await guard.authorize("delete_file", {"path": "/tmp/x"})
    assert len(prompts) == 1


def test_legacy_tier_names_still_resolve():
    """An existing permissions.yaml written in the old vocabulary keeps working."""
    from core.safety_guard import RiskTier

    assert RiskTier.from_config("safe") is RiskTier.READ
    assert RiskTier.from_config("risky") is RiskTier.MODIFY
    assert RiskTier.from_config("destructive") is RiskTier.CRITICAL
    assert RiskTier.from_config("modify") is RiskTier.MODIFY


def test_unclassified_tool_is_treated_as_critical(guard_factory):
    """A tool nobody classified is one nobody thought about."""
    from core.safety_guard import RiskTier

    guard, _ = guard_factory()
    assert guard.get_tier("some_tool_that_does_not_exist") is RiskTier.CRITICAL
