"""
SafetyGuard is the single choke point every tool call must pass through.

Responsibilities:
  1. Classify a requested tool call into a risk tier (safe/risky/destructive/forbidden).
  2. Block anything matching a forbidden pattern or protected path, unconditionally.
  3. Route risky/destructive calls through a confirmation callback (voice or CLI).
  4. Append an immutable JSON-lines audit record for every tool invocation attempt,
     regardless of outcome (approved, denied, executed, failed).
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from core.config_loader import get_permissions, get_settings, resolve_path


def _humanize_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Turns a tool name + raw arguments into a plain-English description of the action,
    instead of surfacing python-style key=repr(value) pairs to the user."""

    def _get(*keys: str, default: str = "") -> str:
        for k in keys:
            if arguments.get(k):
                return str(arguments[k])
        return default

    humanizers: Dict[str, Callable[[Dict[str, Any]], str]] = {
        "send_email": lambda a: (
            f"send an email to {_get('to', default='someone')}"
            + (f", cc {a['cc']}," if a.get('cc') else "")
            + f' with the subject "{_get("subject", default="(no subject)")}"'
        ),
        "delete_file": lambda a: f"delete the file {_get('path', 'file_path')}",
        "write_file": lambda a: f"write to the file {_get('path', 'file_path')}",
        "move_file": lambda a: f"move {_get('source_path')} to {_get('destination_path')}",
        "run_shell_command": lambda a: f"run this command in the terminal: {_get('command', 'cmd')}",
        "kill_process": lambda a: f"stop the process with ID {_get('pid')}",
        "enable_firewall": lambda a: "turn on the system firewall",
        "apply_system_updates": lambda a: "download and install available system updates (may require a reboot)",
        "create_security_snapshot": lambda a: f"create a security snapshot named '{_get('label')}' of {_get('path')}",
        "restore_from_snapshot": lambda a: (
            f"restore {(_get('only_file') or 'all files')} from the snapshot '{_get('label')}', "
            f"overwriting the current version"
        ),
        "place_paper_order": lambda a: (
            f"place a simulated {_get('side')} order for {_get('qty')} of {_get('symbol')} "
            f"in your paper trading account (no real money involved)"
        ),
        "add_contact": lambda a: f"add a new contact named {_get('name')}" + (f" ({a['email']})" if a.get("email") else ""),
        "update_contact": lambda a: f"update the contact with ID {_get('contact_id')}",
        "delete_contact": lambda a: f"delete the contact with ID {_get('contact_id')}",
        "clear_user_profile": lambda a: "erase everything I remember about you (name and all saved facts)",
        "login_to_social_platform": lambda a: f"open a browser window for you to log in to {_get('platform')} (I won't see your password)",
        "browser_fill_form": lambda a: f"fill out a form on the current page",
        "close_app": lambda a: f"close {_get('app_name', 'name')}",
        "launch_app": lambda a: f"launch {_get('app_name', 'name')}",
    }

    fn = humanizers.get(tool_name)
    if fn:
        try:
            return fn(arguments)
        except Exception:
            pass

    # Generic fallback for any tool without a specific phrasing above: plain "key set to
    # value" pairs, not python repr() syntax, so it still reads as normal text.
    if arguments:
        parts = [f"{k.replace('_', ' ')} set to {v}" for k, v in arguments.items()]
        return f"{tool_name.replace('_', ' ')} ({', '.join(parts)})"
    return tool_name.replace("_", " ")


class RiskTier(str, Enum):
    SAFE = "safe"
    RISKY = "risky"
    DESTRUCTIVE = "destructive"
    FORBIDDEN = "forbidden"


class PermissionDenied(Exception):
    """Raised when a tool call is blocked outright (forbidden pattern/path)."""


class ConfirmationDenied(Exception):
    """Raised when the user explicitly declines or times out on a confirmation prompt."""


@dataclass
class AuditRecord:
    timestamp: float
    tool_name: str
    arguments: Dict[str, Any]
    tier: str
    decision: str          # "auto_approved" | "user_approved" | "user_denied" | "blocked" | "error"
    detail: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
                "tool": self.tool_name,
                "arguments": self.arguments,
                "tier": self.tier,
                "decision": self.decision,
                "detail": self.detail,
            }
        )


# Type for a confirmation callback: given a human-readable prompt, returns True/False.
ConfirmationCallback = Callable[[str], Awaitable[bool]]


class SafetyGuard:
    def __init__(self, confirmation_callback: Optional[ConfirmationCallback] = None):
        self.settings = get_settings()
        self.permissions = get_permissions()
        self._confirmation_callback = confirmation_callback
        self._audit_path = resolve_path(self.settings["safety"]["audit_log_path"])
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def set_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        self._confirmation_callback = callback

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def get_tier(self, tool_name: str) -> RiskTier:
        tool_cfg = self.permissions.get("tools", {}).get(tool_name)
        if not tool_cfg:
            # Unknown tools default to the most cautious tier.
            return RiskTier.DESTRUCTIVE
        return RiskTier(tool_cfg.get("tier", "destructive"))

    def _matches_forbidden_shell_pattern(self, command: str) -> Optional[str]:
        for pattern in self.permissions.get("forbidden_shell_patterns", []):
            if pattern.lower() in command.lower():
                return pattern
        return None

    def _touches_protected_path(self, path_str: str) -> Optional[str]:
        expanded = str(resolve_path(path_str))
        for protected in self.permissions.get("protected_paths", []):
            protected_expanded = str(Path(protected).expanduser())
            if fnmatch.fnmatch(expanded, protected_expanded + "*") or expanded.startswith(
                protected_expanded
            ):
                return protected
        return None

    def check_hard_block(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Returns a reason string if the call must be blocked unconditionally, else None."""
        command = arguments.get("command") or arguments.get("cmd")
        if command:
            matched = self._matches_forbidden_shell_pattern(str(command))
            if matched:
                return f"Command matches forbidden pattern: '{matched}'"

        for key in ("path", "file_path", "target_path", "source_path", "destination_path"):
            if key in arguments and arguments[key]:
                matched = self._touches_protected_path(str(arguments[key]))
                if matched:
                    return f"Path '{arguments[key]}' touches protected location '{matched}'"

        url = arguments.get("url")
        if url:
            for blocked in self.permissions.get("blocked_domains", []):
                if blocked in url:
                    return f"URL matches blocked domain: '{blocked}'"

        return None

    # ------------------------------------------------------------------ #
    # Main entry point used by the orchestrator / tool executor
    # ------------------------------------------------------------------ #
    async def authorize(self, tool_name: str, arguments: Dict[str, Any], preapproved: bool = False) -> None:
        """
        Raises PermissionDenied or ConfirmationDenied if the call should not proceed.
        Returns normally (None) if the call is authorized to execute.

        `preapproved`: set when the user's own request already expressed clear approval
        for this action (e.g. a voice command whose wording already confirms intent - see
        core/intent_signals.py) so the interactive confirmation step can be skipped. This
        NEVER bypasses hard blocks or the FORBIDDEN tier - only the confirmation prompt for
        an otherwise-permitted risky/destructive call.
        """
        tier = self.get_tier(tool_name)

        block_reason = self.check_hard_block(tool_name, arguments)
        if block_reason:
            await self._audit(tool_name, arguments, tier, "blocked", block_reason)
            raise PermissionDenied(block_reason)

        if tier == RiskTier.FORBIDDEN:
            await self._audit(tool_name, arguments, tier, "blocked", "Tool is globally forbidden")
            raise PermissionDenied(f"Tool '{tool_name}' is forbidden by configuration.")

        if tier == RiskTier.SAFE:
            await self._audit(tool_name, arguments, tier, "auto_approved")
            return

        # RISKY or DESTRUCTIVE -> require confirmation, unless already given.
        if self.settings["safety"].get("dry_run"):
            await self._audit(tool_name, arguments, tier, "auto_approved", "dry_run mode")
            return

        if preapproved:
            await self._audit(
                tool_name, arguments, tier, "user_approved",
                "Pre-approved: the user's own request already expressed clear approval.",
            )
            return

        if self._confirmation_callback is None:
            await self._audit(
                tool_name, arguments, tier, "user_denied", "No confirmation channel available"
            )
            raise ConfirmationDenied(
                f"'{tool_name}' requires confirmation but no confirmation channel is configured."
            )

        prompt = self._build_confirmation_prompt(tool_name, arguments, tier)
        timeout = self.settings["safety"].get("confirmation_timeout_seconds", 15)
        try:
            approved = await asyncio.wait_for(self._confirmation_callback(prompt), timeout=timeout)
        except asyncio.TimeoutError:
            await self._audit(tool_name, arguments, tier, "user_denied", "Confirmation timed out")
            raise ConfirmationDenied("Confirmation timed out.")

        if not approved:
            await self._audit(tool_name, arguments, tier, "user_denied", "User declined")
            raise ConfirmationDenied("User declined the action.")

        await self._audit(tool_name, arguments, tier, "user_approved")

    def _build_confirmation_prompt(
        self, tool_name: str, arguments: Dict[str, Any], tier: RiskTier
    ) -> str:
        action_desc = _humanize_tool_call(tool_name, arguments)
        severity = "This can't be undone." if tier == RiskTier.DESTRUCTIVE else \
                   "This will make a change on your system."
        return f"Leti wants to {action_desc}. {severity} Should I go ahead?"

    # ------------------------------------------------------------------ #
    # Audit logging
    # ------------------------------------------------------------------ #
    async def _audit(
        self, tool_name: str, arguments: Dict[str, Any], tier: RiskTier, decision: str, detail: str = ""
    ) -> None:
        record = AuditRecord(
            timestamp=time.time(),
            tool_name=tool_name,
            arguments=arguments,
            tier=tier.value if isinstance(tier, RiskTier) else str(tier),
            decision=decision,
            detail=detail,
        )
        async with self._lock:
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(record.to_json() + "\n")

    async def audit_result(self, tool_name: str, arguments: Dict[str, Any], success: bool, detail: str = "") -> None:
        """Call after execution to log the outcome, separate from the authorization decision."""
        tier = self.get_tier(tool_name)
        decision = "executed" if success else "error"
        await self._audit(tool_name, arguments, tier, decision, detail)
