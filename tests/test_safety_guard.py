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
from core.safety_guard import ConfirmationDenied, PermissionDenied, RiskTier, SafetyGuard

HOME = os.path.expanduser("~")


def _guard(dry_run: bool = False, confirm: bool = True):
    """A SafetyGuard with a scripted confirmation callback. Returns (guard, prompts)
    where `prompts` records every confirmation the guard actually asked for."""
    prompts: list[str] = []

    async def callback(prompt: str) -> bool:
        prompts.append(prompt)
        return confirm

    guard = SafetyGuard(confirmation_callback=callback)
    guard.settings = {
        **guard.settings,
        "safety": {**guard.settings["safety"], "dry_run": dry_run},
    }
    return guard, prompts


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
def test_protected_paths_survive_traversal(path):
    """'..' must not be a way around the protected list - these all name a
    protected file, however they're spelled."""
    guard, _ = _guard()
    assert guard._touches_protected_path(path) is not None, path


@pytest.mark.parametrize("path", [
    "/etcetera/notes.txt",   # merely starts with the string '/etc'
    "/System.md",
    f"{HOME}/notes.txt",
])
def test_protected_paths_dont_false_positive(path):
    """Prefix matching blocked unrelated paths whose names happen to start with a
    protected one. Comparison is per path segment now."""
    guard, _ = _guard()
    assert guard._touches_protected_path(path) is None, path


@pytest.mark.parametrize("command", [
    "cat ~/.ssh/id_rsa",
    "cat /etc/shadow",
    "echo pwned > /etc/hosts",
    "cp ~/.aws/credentials /tmp/exfil",
    "tar czf out.tgz /tmp/../etc",
])
def test_shell_commands_cannot_reach_protected_paths(command):
    """run_shell_command has no path-shaped argument to inspect, so the command
    string itself is tokenized and checked."""
    guard, _ = _guard()
    assert guard.check_hard_block("run_shell_command", {"command": command}) is not None


@pytest.mark.parametrize("command", ["ls -la ~/projects", "git status", "echo hello world"])
def test_ordinary_shell_commands_still_run(command):
    guard, _ = _guard()
    assert guard.check_hard_block("run_shell_command", {"command": command}) is None


@pytest.mark.parametrize("command", ["rm -rf /", "rm  -rf   /", "RM -RF /"])
def test_forbidden_patterns_ignore_whitespace_and_case(command):
    guard, _ = _guard()
    assert guard._matches_forbidden_shell_pattern(command) is not None


def test_forbidden_patterns_dont_block_relative_rm():
    """'rm -rf ./build' is ordinary work and must not match the 'rm -rf /' pattern."""
    guard, _ = _guard()
    assert guard._matches_forbidden_shell_pattern("rm -rf ./build") is None


# --- dry_run -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dry_run_authorizes_without_executing():
    """The regression that mattered most: dry_run used to return normally from
    authorize(), which the caller reads as 'go ahead and run it'."""
    guard, prompts = _guard(dry_run=True)
    auth = await guard.authorize("delete_file", {"path": "/tmp/whatever.txt"})
    assert auth.execute is False
    assert auth.description  # something to tell the user instead
    assert prompts == []     # nothing to confirm; nothing is going to happen


@pytest.mark.asyncio
async def test_dry_run_still_runs_read_only_tools():
    """Otherwise dry_run would make Leti unable to search or read the screen,
    which is the opposite of useful for testing prompts."""
    guard, _ = _guard(dry_run=True)
    auth = await guard.authorize("read_file", {"path": "/tmp/whatever.txt"})
    assert auth.execute is True


# --- Voice pre-approval --------------------------------------------------------

@pytest.mark.asyncio
async def test_preapproval_is_reported_as_spent():
    guard, prompts = _guard()
    auth = await guard.authorize("write_file", {"path": "/tmp/a.txt"}, preapproved=True)
    assert auth.execute is True
    assert auth.used_preapproval is True   # caller clears its flag on this
    assert prompts == []


@pytest.mark.asyncio
async def test_preapproval_never_covers_destructive_calls():
    """A spoken request that approves one thing must not silently authorize a
    delete the model chose on its own."""
    guard, prompts = _guard(confirm=False)
    with pytest.raises(ConfirmationDenied):
        await guard.authorize("delete_file", {"path": "/tmp/a.txt"}, preapproved=True)
    assert len(prompts) == 1               # it asked anyway


@pytest.mark.asyncio
async def test_preapproval_never_bypasses_a_hard_block():
    guard, _ = _guard()
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
