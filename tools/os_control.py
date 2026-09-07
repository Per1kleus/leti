"""
Desktop automation: launching applications, managing windows, and driving the
mouse/keyboard. Wrapped in individual BaseTool subclasses so each action gets
its own risk tier and audit trail entry (see config/permissions.yaml).
"""
from __future__ import annotations

import asyncio
import platform
import subprocess
from typing import Optional

import pyautogui
import pygetwindow as gw

from tools.base import BaseTool, ToolParameter, ToolResult

pyautogui.FAILSAFE = True  # moving mouse to a screen corner aborts pyautogui actions


class LaunchAppTool(BaseTool):
    name = "launch_app"
    description = "Launch a desktop application by name (e.g. 'notepad', 'chrome', 'spotify')."
    parameters = [
        ToolParameter(name="app_name", type="string", description="Name or path of the application to launch."),
    ]

    async def run(self, app_name: str, **kwargs) -> ToolResult:
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(["start", "", app_name], shell=True)
            elif system == "Darwin":
                subprocess.Popen(["open", "-a", app_name])
            else:  # Linux
                subprocess.Popen([app_name])
            return ToolResult(success=True, output=f"Launched '{app_name}'.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CloseAppTool(BaseTool):
    name = "close_app"
    description = "Close an application window by matching its window title."
    parameters = [
        ToolParameter(name="window_title", type="string", description="Substring of the window title to close."),
    ]

    async def run(self, window_title: str, **kwargs) -> ToolResult:
        try:
            matches = [w for w in gw.getAllWindows() if window_title.lower() in w.title.lower()]
            if not matches:
                return ToolResult(success=False, error=f"No window matching '{window_title}' found.")
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
        try:
            matches = [w for w in gw.getAllWindows() if window_title.lower() in w.title.lower()]
            if not matches:
                return ToolResult(success=False, error=f"No window matching '{window_title}' found.")
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
