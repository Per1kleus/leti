"""
Executes shell commands with strict guardrails:
  - Hard-blocked patterns are already filtered by SafetyGuard before this runs,
    but we re-check here as defense-in-depth in case this tool is invoked directly.
  - Commands run with a timeout, captured stdout/stderr, and no shell chaining
    of destructive redirects beyond what the forbidden-pattern list catches.
  - Runs in the user's home directory unless a working_dir is given.

Kept separate from run_code rather than folded into it, though run_code's bash
mode can execute the same text. They answer different questions: this runs one
command line the way a terminal would, in the user's home directory, and returns
what it printed; run_code runs a named language's runtime over a snippet or a file
in a project directory, with stdin, a chosen interpreter and an exit code, and is
how you check whether code works. Merging them would mean one tool whose contract
changes completely depending on a `language` argument. What they must not have is
two different safety postures, and they don't: SafetyGuard matches the forbidden
shell patterns against `code` as well as `command`, so the same snippet is refused
whichever tool is asked to run it.
"""
from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
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

        # Default to the user's home directory, as this tool's parameter description
        # and module docstring both say. cwd=None inherited Leti's own working
        # directory - the project checkout - so an unqualified 'ls' or a relative
        # write landed inside the app's own source tree.
        cwd = str(resolve_path(working_dir)) if working_dir else str(Path.home())

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
                await proc.wait()   # reap it, or the child lingers as a zombie
                return ToolResult(success=False, error=f"Command timed out after {timeout_seconds}s.")

            output = stdout.decode(errors="replace").strip()
            err = stderr.decode(errors="replace").strip()

            if proc.returncode != 0:
                return ToolResult(success=False, error=err or f"Command exited with code {proc.returncode}")

            return ToolResult(success=True, output=output or "(command produced no output)")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
