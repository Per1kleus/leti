"""Tests for opening things on the user's computer and browsing the web.

The design intent these protect: the user says "open Firefox on YouTube", Leti
confirms the action and it happens. Capability comes from the tool being able to
express what the user meant (apps, files, URLs, arguments); safety comes from the
confirmation prompt, which therefore has to describe what will actually happen.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import types

import pytest

# pyautogui/pygetwindow need a display and aren't importable headless, but the
# launch and URL paths under test don't touch them.
for _name in ("pyautogui", "pygetwindow"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            _stub = types.ModuleType(_name)
            _stub.FAILSAFE = True
            sys.modules[_name] = _stub

from core.safety_guard import _humanize_tool_call  # noqa: E402
from tools.os_control import LaunchAppTool, OpenUrlTool, _resolve_launch  # noqa: E402


# --- Launching -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_launches_a_real_program_with_arguments():
    """The capability the old tool lacked entirely: it took only app_name, so
    "open YouTube in Firefox" could not be expressed."""
    result = await LaunchAppTool().run(app_name="sleep", arguments=["3"])
    assert result.success, result.error
    assert "sleep" in result.output


@pytest.mark.asyncio
async def test_missing_application_is_reported_not_claimed():
    result = await LaunchAppTool().run(app_name="definitely-not-installed-xyz")
    assert result.success is False
    assert "Couldn't find" in result.error


@pytest.mark.asyncio
async def test_program_that_dies_immediately_is_not_reported_as_launched():
    """Popen succeeding is not evidence anything opened - it returns fine for a
    program that exits a moment later."""
    result = await LaunchAppTool().run(app_name="ls", arguments=["/no/such/path/anywhere"])
    assert result.success is False
    assert "exited immediately" in result.error


def test_urls_and_files_go_to_the_system_handler(tmp_path):
    doc = tmp_path / "invoice.txt"
    doc.write_text("x")

    argv, _ = _resolve_launch(str(doc), [])
    assert argv is not None and str(doc) in argv

    argv, _ = _resolve_launch("https://youtube.com", [])
    assert argv is not None and "https://youtube.com" in argv


def test_executable_on_path_is_run_directly():
    argv, how = _resolve_launch("sleep", ["1"])
    assert argv == [shutil.which("sleep"), "1"]
    assert "ran" in how


# --- Opening a URL in the user's own browser -----------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "file:///etc/passwd",       # hands the local filesystem to the browser
    "FILE:/etc/passwd",         # scheme match is case-insensitive
    "javascript:alert(1)",      # runs in whatever page is focused
    "data:text/html,<script>",
    "ftp://example.com",
])
async def test_open_url_rejects_non_web_schemes(url):
    """A '://' test isn't enough - 'javascript:alert(1)' has no slashes, so it
    would fall through and get prefixed with https://."""
    result = await OpenUrlTool().run(url=url)
    assert result.success is False
    assert "Only http" in result.error


# --- What the user is actually agreeing to -------------------------------------

def test_confirmation_names_what_will_be_opened():
    """launch_app can open something IN an app, so "launch firefox" alone doesn't
    describe the action being approved."""
    described = _humanize_tool_call("launch_app", {
        "app_name": "firefox", "arguments": ["https://youtube.com"],
    })
    assert "firefox" in described
    assert "https://youtube.com" in described


def test_confirmation_handles_an_app_with_no_arguments():
    assert _humanize_tool_call("launch_app", {"app_name": "spotify"}) == "open spotify"


def test_close_app_confirmation_names_the_window():
    described = _humanize_tool_call("close_app", {"window_title": "Spotify"})
    assert "Spotify" in described


# --- Tiering: capability under confirmation, not capability withheld -----------

def test_action_classes_match_the_intended_posture():
    from core.config_loader import get_permissions
    from core.safety_guard import DEFAULT_CONFIRMATION_CLASSES

    tools = get_permissions()["tools"]
    # Opening an app starts an arbitrary program -> in a class that confirms.
    assert tools["launch_app"]["action"] in DEFAULT_CONFIRMATION_CLASSES
    # Browsing is meant to be free: none of these are in a confirming class.
    for tool in ("web_search", "open_url", "browser_navigate", "browser_read_page"):
        assert tools[tool]["action"] not in DEFAULT_CONFIRMATION_CLASSES, tool


def test_appdata_block_does_not_cover_installed_applications():
    """C:\\Users\\*\\AppData covers most of what a Windows user has installed -
    VS Code, Discord, Spotify and Slack all live under LocalAppData."""
    from core.config_loader import get_permissions

    protected = get_permissions()["protected_paths"]
    assert "C:\\Users\\*\\AppData" not in protected
    assert any("Chrome\\User Data" in p for p in protected)
