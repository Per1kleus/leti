"""
Desktop automation: launching applications, managing windows, and driving the
mouse/keyboard. Wrapped in individual BaseTool subclasses so each action gets
its own risk tier and audit trail entry (see config/permissions.yaml).

Launching is deliberately capable: the user asks for something ("open Firefox on
YouTube", "open my invoice", "open YouTube"), Leti confirms the action, and it
happens. That means resolving an app the way a person names it rather than
demanding an exact binary, passing arguments through, and reporting honestly when
nothing started. The confirmation prompt is the control here, not a narrow tool.

launch_app opens web pages too, rather than a separate open_url tool beside it:
"open YouTube" and "open Spotify" are one request as far as the user is
concerned, and splitting them left the model choosing between two tools that
both claimed to open things. What differs is the consequence, not the intent, so
that is where the distinction lives now: action_case() reports whether a call is
a web page or a program, and permissions.yaml gives each its own action class -
handing a URL to the browser runs nothing locally and doesn't interrupt, while
starting a program still asks first.
"""
from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyautogui
import pygetwindow as gw

from tools.base import BaseTool, ToolParameter, ToolResult

pyautogui.FAILSAFE = True  # moving mouse to a screen corner aborts pyautogui actions

_WINDOW_CONTROL_UNSUPPORTED = (
    "Window control isn't available on this system. pygetwindow implements it on "
    "Windows and macOS; on Linux it raises NotImplementedError. Leti can still launch "
    "apps, open URLs, type and click - just not enumerate or move windows. "
    "(Workaround on Linux: `run_shell_command` with wmctrl or xdotool, if installed.)"
)


def _matching_windows(window_title: str):
    """Windows whose title contains `window_title`, or a legible error instead.

    pygetwindow raises NotImplementedError on Linux rather than returning nothing,
    which surfaced to the user as a bare exception string with no hint that the
    platform simply doesn't support this.
    """
    try:
        return [w for w in gw.getAllWindows() if window_title.lower() in w.title.lower()], None
    except NotImplementedError:
        return [], _WINDOW_CONTROL_UNSUPPORTED
    except Exception as e:
        return [], f"Couldn't list windows: {e}"

# How long to wait before deciding a launch "worked". A program that is going to
# fail outright (missing binary, bad arguments) exits within a few hundred ms;
# anything still alive after this is a real running application.
_LAUNCH_SETTLE_SECONDS = 0.6


_WEB_SCHEMES = ("http://", "https://")
_OPENABLE_SCHEMES = _WEB_SCHEMES + ("mailto:", "file://")

# A bare domain the user typed as a destination: 'youtube.com', 'news.ycombinator.com/best'.
# Requires a dot-separated hostname with a real-looking TLD so a filename like
# 'report.pdf' or a program like 'python3.11' isn't mistaken for a website.
_BARE_DOMAIN = re.compile(r"^(?:[\w-]+\.)+[a-z]{2,}(?:[:/?#]\S*)?$", re.IGNORECASE)
_NOT_A_DOMAIN_TLD = {"py", "sh", "exe", "app", "pdf", "txt", "md", "json", "yaml", "yml",
                     "png", "jpg", "csv", "zip", "log", "conf", "cfg", "desktop"}


def _looks_like_url(value: str) -> bool:
    return value.startswith(_OPENABLE_SCHEMES)


def _looks_like_bare_domain(value: str) -> bool:
    """'youtube.com' means a website; 'report.pdf' and 'python3.11' do not."""
    if not _BARE_DOMAIN.match(value):
        return False
    host = value.split("/")[0].split(":")[0].split("?")[0]
    return host.rsplit(".", 1)[-1].lower() not in _NOT_A_DOMAIN_TLD


def _as_web_url(value: str) -> Optional[str]:
    """The http(s) URL `value` denotes, or None if it isn't a web destination.

    Bare domains are promoted to https://. Any other scheme is deliberately NOT
    promoted: 'javascript:alert(1)' and 'file:/etc/passwd' contain no '//', so a
    naive test lets them through to be prefixed with https:// and handed to the
    browser anyway - one runs code in whatever page is open, the other exposes the
    local filesystem. They fall through to be treated as a program name, which
    fails with a clear message rather than being executed as a URL.
    """
    if value.startswith(_WEB_SCHEMES):
        return value
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:", value):
        return None
    if _looks_like_bare_domain(value):
        return "https://" + value
    return None


def _web_target(app_name: str, arguments: List[str]) -> Optional[str]:
    """The web page this launch_app call opens, or None if it launches something else.

    A call is "open a web page" only when the whole request is a page: an existing
    local file of the same name wins (someone may really have a file called
    'notes.io'), and arguments mean a program is being driven ('firefox' with a URL
    in `arguments` starts Firefox, which is a program launch however web-shaped its
    argument is).
    """
    if arguments:
        return None
    app_name = app_name.strip()
    if Path(app_name).expanduser().exists():
        return None
    return _as_web_url(app_name)


def _platform_opener() -> Optional[List[str]]:
    """The OS command that opens a file or URL with its registered handler."""
    system = platform.system()
    if system == "Darwin":
        return ["open"]
    if system == "Windows":
        return ["cmd", "/c", "start", ""]
    opener = shutil.which("xdg-open") or shutil.which("gio")
    if opener:
        return [opener, "open"] if opener.endswith("gio") else [opener]
    return None


def _resolve_launch(app_name: str, arguments: List[str]) -> tuple[Optional[List[str]], str]:
    """Work out how to start `app_name`, returning (argv, how) or (None, why-not).

    Tried in order, most specific first:
      1. A URL or an existing file/path -> hand to the OS's registered handler,
         which is what makes "open my invoice" and "open youtube.com" work.
      2. An executable on PATH (or an absolute path to one) -> run it directly.
      3. The platform's own by-name app launcher - `open -a` on macOS, `start` on
         Windows, gtk-launch on Linux desktops - which knows about applications
         that aren't plain binaries on PATH.
    """
    opener = _platform_opener()

    if _looks_like_url(app_name) or Path(app_name).expanduser().exists():
        target = str(Path(app_name).expanduser()) if not _looks_like_url(app_name) else app_name
        if opener:
            return opener + [target] + arguments, f"opened '{target}' with the system handler"
        return None, "No system 'open' handler found (install xdg-utils on Linux)."

    direct = shutil.which(app_name)
    if direct:
        return [direct] + arguments, f"ran '{direct}'"

    system = platform.system()
    if system == "Darwin":
        argv = ["open", "-a", app_name]
        if arguments:
            argv += ["--args"] + arguments
        return argv, f"asked macOS to open the app named '{app_name}'"
    if system == "Windows":
        # No shell=True: app_name comes from the model, and letting cmd.exe parse
        # it as a command line makes '&' and friends into command separators.
        return ["cmd", "/c", "start", "", app_name] + arguments, f"asked Windows to start '{app_name}'"

    gtk_launch = shutil.which("gtk-launch")
    if gtk_launch:
        desktop_id = app_name if app_name.endswith(".desktop") else f"{app_name}.desktop"
        return [gtk_launch, desktop_id] + arguments, f"launched the '{desktop_id}' application entry"

    return None, (
        f"Couldn't find an application called '{app_name}': it isn't a program on PATH, "
        f"an existing file, or a URL. Try the exact command name (e.g. 'firefox', "
        f"'google-chrome') or the full path to the program."
    )


class LaunchAppTool(BaseTool):
    name = "launch_app"
    description = (
        "Open something on the user's computer: an application, a file, or a web page. "
        "Accepts a program name ('firefox', 'spotify', 'code'), a path to a program or "
        "document, or a web address ('youtube.com', 'https://docs.python.org').\n"
        "A web address on its own opens in the user's OWN default browser - the one with "
        "their logins, bookmarks and extensions - which is what they mean by 'open YouTube' "
        "or 'pull up the docs'. Use `arguments` to open something IN a specific app instead: "
        "app_name 'firefox' with arguments ['https://youtube.com'], or 'code' with ['~/projects'].\n"
        "This is for pages the USER looks at. When Leti needs to read a page itself, use "
        "web_search to find one and browser_read_page to read it."
    )
    parameters = [
        ToolParameter(
            name="app_name", type="string",
            description=("Program name (e.g. 'firefox'), a path to a program or file, or a "
                         "web address (e.g. 'youtube.com')."),
        ),
        ToolParameter(
            name="arguments", type="array", items_type="string", required=False,
            description="Arguments to pass, e.g. ['https://youtube.com'] to open a page in a browser.",
        ),
    ]

    def action_case(self, arguments: Dict[str, Any]) -> Optional[str]:
        """Opening a web page and starting a program are not the same act.

        Handing a URL to the browser runs nothing on this machine and is what the
        user asks for many times a day; starting an arbitrary program is the thing
        worth a confirmation prompt. permissions.yaml assigns the class for each.
        """
        args = arguments.get("arguments") or []
        target = _web_target(str(arguments.get("app_name", "")), [str(a) for a in args])
        return "web_page" if target else "program"

    async def run(self, app_name: str, arguments: Optional[List[str]] = None, **kwargs) -> ToolResult:
        arguments = [str(a) for a in (arguments or [])]

        web_url = _web_target(app_name, arguments)
        if web_url:
            return await self._open_in_default_browser(web_url)

        argv, how = _resolve_launch(app_name, arguments)
        if argv is None:
            return ToolResult(success=False, error=how)

        try:
            # Detached and silenced: a launched app must outlive the tool call, and
            # its stdout shouldn't spill into Leti's own terminal.
            popen_kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            proc = await asyncio.get_running_loop().run_in_executor(
                None, lambda: subprocess.Popen(argv, **popen_kwargs)
            )
        except FileNotFoundError:
            return ToolResult(success=False, error=f"Couldn't run '{argv[0]}' - it isn't installed or isn't on PATH.")
        except OSError as e:
            return ToolResult(success=False, error=f"Couldn't start '{app_name}': {e}")

        # Popen returning is not evidence anything opened - it succeeds for a program
        # that exits immediately. Give it a moment, then check it's actually alive,
        # so "Launched X" is never reported for something that died on startup.
        await asyncio.sleep(_LAUNCH_SETTLE_SECONDS)
        if proc.poll() is not None and proc.returncode != 0:
            stderr = b""
            try:
                stderr = proc.stderr.read() if proc.stderr else b""
            except Exception:
                pass
            detail = stderr.decode(errors="replace").strip().splitlines()
            return ToolResult(
                success=False,
                error=(f"'{app_name}' exited immediately with code {proc.returncode}"
                       + (f": {detail[0]}" if detail else ".")),
            )

        opened = f" with {' '.join(arguments)}" if arguments else ""
        return ToolResult(success=True, output=f"Opened '{app_name}'{opened} ({how}).")

    @staticmethod
    async def _open_in_default_browser(url: str) -> ToolResult:
        """Hand a page to the user's registered browser, with their session in it.

        Deliberately webbrowser rather than the generic platform opener used for
        files: it picks the browser the user actually configured (honouring $BROWSER
        on Linux), which is the whole point of showing them a page rather than
        reading it with browser_read_page.
        """
        import webbrowser

        try:
            opened = await asyncio.get_running_loop().run_in_executor(
                None, lambda: webbrowser.open(url)
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Couldn't open the browser: {e}")

        if not opened:
            return ToolResult(
                success=False,
                error="No default browser could be started. On Linux this usually means xdg-utils isn't installed.",
            )
        return ToolResult(success=True, output=f"Opened {url} in your default browser.")


class CloseAppTool(BaseTool):
    name = "close_app"
    description = "Close an application window by matching its window title."
    parameters = [
        ToolParameter(name="window_title", type="string", description="Substring of the window title to close."),
    ]

    async def run(self, window_title: str, **kwargs) -> ToolResult:
        matches, error = _matching_windows(window_title)
        if error:
            return ToolResult(success=False, error=error)
        if not matches:
            return ToolResult(success=False, error=f"No window matching '{window_title}' found.")
        try:
            for w in matches:
                w.close()
            return ToolResult(success=True, output=f"Closed {len(matches)} window(s) matching '{window_title}'.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class FocusWindowTool(BaseTool):
    name = "focus_window"
    description = "Bring a window to the foreground by matching its title."
    parameters = [
        ToolParameter(name="window_title", type="string", description="Substring of the window title to focus."),
    ]

    async def run(self, window_title: str, **kwargs) -> ToolResult:
        matches, error = _matching_windows(window_title)
        if error:
            return ToolResult(success=False, error=error)
        if not matches:
            return ToolResult(success=False, error=f"No window matching '{window_title}' found.")
        try:
            win = matches[0]
            win.activate()
            return ToolResult(success=True, output=f"Focused window '{win.title}'.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class MouseClickTool(BaseTool):
    name = "mouse_click"
    description = "Move the mouse to (x, y) screen coordinates and click."
    parameters = [
        ToolParameter(name="x", type="number", description="X coordinate in pixels."),
        ToolParameter(name="y", type="number", description="Y coordinate in pixels."),
        ToolParameter(
            name="button", type="string", description="Mouse button to click.",
            required=False, enum=["left", "right", "middle"],
        ),
        ToolParameter(name="double", type="boolean", description="Whether to double-click.", required=False),
    ]

    async def run(self, x: float, y: float, button: str = "left", double: bool = False, **kwargs) -> ToolResult:
        try:
            loop = asyncio.get_event_loop()
            if double:
                await loop.run_in_executor(None, lambda: pyautogui.doubleClick(x, y, button=button))
            else:
                await loop.run_in_executor(None, lambda: pyautogui.click(x, y, button=button))
            return ToolResult(success=True, output=f"Clicked ({x}, {y}) with {button} button.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class KeyboardTypeTool(BaseTool):
    name = "keyboard_type"
    description = "Type text at the current cursor/focus location."
    parameters = [
        ToolParameter(name="text", type="string", description="Text to type."),
        ToolParameter(name="interval", type="number", description="Seconds between keystrokes.", required=False),
    ]

    async def run(self, text: str, interval: float = 0.02, **kwargs) -> ToolResult:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: pyautogui.write(text, interval=interval))
            return ToolResult(success=True, output=f"Typed {len(text)} characters.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class KeyboardHotkeyTool(BaseTool):
    name = "keyboard_hotkey"
    description = "Press a keyboard shortcut, e.g. ['ctrl','c'] for copy."
    parameters = [
        ToolParameter(
            name="keys", type="array", items_type="string",
            description="Ordered list of keys to press together, e.g. ['ctrl','s'].",
        ),
    ]

    async def run(self, keys: list, **kwargs) -> ToolResult:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: pyautogui.hotkey(*keys))
            return ToolResult(success=True, output=f"Pressed hotkey: {'+'.join(keys)}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
