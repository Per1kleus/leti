"""Shared helper for shelling out to system utilities from a tool.

Two problems this exists to stop repeating.

Silent failure. The pattern these modules used was:

    def _run(cmd):
        try: return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
        except Exception: return ""

which throws away the exit status and stderr. `sudo -n ufw enable` without a
cached credential returns non-zero and prints to stderr - and the caller, seeing
only an empty string, reported "Firewall enable command issued." A security tool
that says it did something it didn't is worse than one that errors: the user
stops looking. The same shape turned an unreadable /var/log/auth.log into "No
brute-force pattern detected" - a false all-clear.

Blocking the event loop. subprocess.run() inside `async def run` stalls every
other coroutine, and in GUI mode that loop also serves the websocket server, so
chat, voice and every connected phone freeze for the duration with no visible
cause. apply_system_updates used a 30-minute timeout.

CommandResult keeps the exit status and stderr so callers can tell "it worked",
"it failed and here's why", and "the tool isn't installed" apart.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_TIMEOUT = 15


@dataclass
class CommandResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    error: str = ""
    """Set when the command never ran or never finished (not found, timed out,
    permission denied on exec) - as opposed to running and exiting non-zero."""

    @property
    def ok(self) -> bool:
        return self.error == "" and self.returncode == 0

    @property
    def not_found(self) -> bool:
        """True when the executable isn't installed on this machine - usually
        worth reporting differently from a command that ran and failed."""
        return self.error.startswith("not-found:")

    def failure_reason(self) -> str:
        """One-line explanation suitable for a ToolResult error, or "" if it worked."""
        if self.ok:
            return ""
        if self.error:
            return self.error
        detail = (self.stderr or self.stdout).strip().splitlines()
        first = detail[0] if detail else ""
        base = f"exited with code {self.returncode}"
        return f"{base}: {first}" if first else base


async def run_command(cmd: List[str], timeout: float = DEFAULT_TIMEOUT) -> CommandResult:
    """Run `cmd` without blocking the event loop, capturing status and both streams.

    Never raises for an ordinary failure - inspect .ok / .failure_reason(). The
    argument list is passed straight to exec (no shell), so nothing here needs
    quoting and nothing in it is interpreted.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return CommandResult(None, "", "", error=f"not-found: '{cmd[0]}' is not installed or not on PATH")
    except OSError as e:
        return CommandResult(None, "", "", error=f"could not run '{cmd[0]}': {e}")

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()  # reap it - otherwise the child lingers as a zombie
        return CommandResult(None, "", "", error=f"timed out after {timeout}s")

    return CommandResult(
        returncode=proc.returncode,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )
