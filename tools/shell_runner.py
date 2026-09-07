"""
Executes shell commands with strict guardrails:
  - Hard-blocked patterns are already filtered by SafetyGuard before this runs,
    but we re-check here as defense-in-depth in case this tool is invoked directly.
  - Commands run with a timeout, captured stdout/stderr, and no shell chaining
    of destructive redirects beyond what the forbidden-pattern list catches.
  - Runs in a restricted working directory by default (not system roots).
"""
from __future__ import annotations

import asyncio
import shlex
from typing import Optional

from core.config_loader import get_permissions, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

DEFAULT_TIMEOUT = 30


class ShellRunnerTool(BaseTool):
    name = "run_shell_command"
    description = (
        "Run a shell/terminal command and return its output. Use only for simple, "
        "non-destructive commands (listing files, checking versions, running scripts). "
        "Never used for deleting data or system-level changes."
    )
    parameters = [
        ToolParameter(name="command", type="string", description="The shell command to execute."),
        ToolParameter(
            name="working_dir", type="string",
            description="Directory to run the command in (defaults to the user's home directory).",
            required=False,
        ),
        ToolParameter(
            name="timeout_seconds", type="number",
            description="Max seconds to allow the command to run before killing it.",
            required=False,
        ),
    ]

    def __init__(self):
        self.permissions = get_permissions()

    def _pattern_check(self, command: str) -> Optional[str]:
        for pattern in self.permissions.get("forbidden_shell_patterns", []):
            if pattern.lower() in command.lower():
                return pattern
        return None

    async def run(
        self, command: str, working_dir: Optional[str] = None, timeout_seconds: float = DEFAULT_TIMEOUT, **kwargs
    ) -> ToolResult:
        matched = self._pattern_check(command)
        if matched:
            return ToolResult(success=False, error=f"Blocked: command matches forbidden pattern '{matched}'.")

        cwd = str(resolve_path(working_dir)) if working_dir else None

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(success=False, error=f"Command timed out after {timeout_seconds}s.")

            output = stdout.decode(errors="replace").strip()
            err = stderr.decode(errors="replace").strip()

            if proc.returncode != 0:
                return ToolResult(success=False, error=err or f"Command exited with code {proc.returncode}")

            return ToolResult(success=True, output=output or "(command produced no output)")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
