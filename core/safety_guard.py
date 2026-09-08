"""
SafetyGuard is the single choke point every tool call must pass through.

Responsibilities:
  1. Classify a requested tool call by what it does: read / execute / modify /
     external / critical (see RiskTier), or forbidden.
  2. Block anything matching a forbidden pattern or protected path, unconditionally.
  3. Route whichever classes the user asked to confirm through a confirmation
     callback (voice or CLI) - safety.require_confirmation_for in settings.yaml.
  4. Append an immutable JSON-lines audit record for every tool invocation attempt,
     regardless of outcome (approved, denied, executed, failed).

authorize() returns an Authorization rather than None: "may this run" and "should
this actually execute" are different questions once dry_run exists, and the caller
also needs to know whether a turn's one-shot voice pre-approval was spent.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

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
        # close_app's parameter is window_title, not app_name - naming the wrong key
        # here rendered the prompt as "Leti wants to close ." with the target missing,
        # on a tool that closes windows. Alternatives are kept as fallbacks.
        "close_app": lambda a: f"close the window {_get('window_title', 'app_name', 'name', default='(unspecified)')}",
        # Include the arguments: launch_app can open something IN an app, and
        # "launch firefox" alone doesn't tell the user they're approving opening a
        # particular page. What they confirm should be what happens.
        "launch_app": lambda a: (
            f"open {_get('app_name', 'name', 'window_title', 'url', default='(unspecified)')}"
            + (f" with {' '.join(str(x) for x in a['arguments'])}" if a.get("arguments") else "")
        ),
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


# Argument names whose VALUE is a secret or private content rather than a
# description of the action. The audit log is a permanent, plaintext,
# never-rotated record, so these are recorded as a redaction marker: knowing
# that browser_fill_form ran on a given selector is the auditable fact;
# recording the password typed into it is a liability. Note the confirmation
# prompt already declines to show these (see browser_fill_form above) - without
# this, _audit wrote them to disk anyway.
_REDACTED_ARGUMENT_KEYS = {
    "value",            # browser_fill_form - often a password or card number
    "text",             # keyboard_type - whatever is being typed, incl. credentials
    "content",          # write_file - full file contents
    "body",             # send_email - full message body
    "password", "app_password", "api_key", "api_secret", "secret",
    "client_secret", "token", "access_token", "credentials",
}
_REDACTION_MARKER = "<redacted: {n} chars>"


def _redact_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of `arguments` with secret/bulk values replaced by a length marker."""
    redacted: Dict[str, Any] = {}
    for key, value in arguments.items():
        if key.lower() in _REDACTED_ARGUMENT_KEYS and value is not None:
            redacted[key] = _REDACTION_MARKER.format(n=len(str(value)))
        else:
            redacted[key] = value
    return redacted


class RiskTier(str, Enum):
    """What kind of action a tool performs, in increasing order of consequence.

    These replaced the original safe/risky/destructive tiers rather than sitting
    alongside them - two parallel classification systems for the same question
    would be exactly the duplication the tool layer is meant to avoid. The old
    names still resolve (see from_config) so an existing permissions.yaml keeps
    working, but config/permissions.yaml itself is written in these terms now.

    Which of these actually require confirmation is the user's choice, via
    safety.require_confirmation_for in settings.yaml. The default asks for
    anything that changes something or reaches outside the machine.
    """

    READ = "read"                       # reads information, changes nothing
    EXECUTE = "execute"                 # runs something reversible: a search, a calculation, code
    MODIFY = "modify"                   # changes files, data, or projects on this machine
    EXTERNAL = "external"               # affects an outside service: sends mail, posts, deploys
    CRITICAL = "critical"               # irreversible or potentially damaging
    FORBIDDEN = "forbidden"             # never runs, whatever the confirmation

    @classmethod
    def from_config(cls, value: str) -> "RiskTier":
        """Accepts the current names and the tier names they replaced."""
        legacy = {"safe": cls.READ, "risky": cls.MODIFY, "destructive": cls.CRITICAL}
        text = str(value).lower()
        if text in legacy:
            return legacy[text]
        return cls(text)


# Confirmation is asked for these unless settings say otherwise. READ and EXECUTE
# are excluded deliberately: an assistant that asks before every search or
# calculation is one the user stops reading the prompts of, which makes the
# prompts on the actions that matter worth less.
DEFAULT_CONFIRMATION_CLASSES = ["modify", "external", "critical"]

# What a scheduled task may do when it runs with nobody present to ask. Reading,
# computing and writing files cover the work people actually schedule - research,
# analysis, reports - while `external` (sending mail, posting, deploying) and
# `critical` (irreversible) are left out: an action nobody can decline is not one
# to take on the user's behalf while they're away from the machine.
DEFAULT_UNATTENDED_CLASSES = ["read", "execute", "modify"]


class PermissionDenied(Exception):
    """Raised when a tool call is blocked outright (forbidden pattern/path)."""


class ConfirmationDenied(Exception):
    """Raised when the user explicitly declines or times out on a confirmation prompt."""


@dataclass
class Authorization:
    """The outcome of an allowed tool call. Denials are raised, not returned."""

    execute: bool
    """False means authorized-but-do-not-run: dry_run mode. The caller must report
    the intended action instead of performing it."""

    used_preapproval: bool = False
    """True if this call consumed the turn's voice pre-approval. The caller clears
    it so the next risky call in the same turn prompts normally."""

    description: str = ""
    """Plain-English description of the action, for telling the user what was (or
    would have been) done without exposing raw argument dicts."""


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
        self.permissions = get_permissions()
        self._confirmation_callback = confirmation_callback
        self._audit_path = resolve_path(get_settings()["safety"]["audit_log_path"])
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._unattended = False

    @property
    def settings(self) -> Dict[str, Any]:
        """Read live rather than captured in __init__, so toggling safety.dry_run or
        confirmation_timeout_seconds via /settings takes effect immediately. A guard
        serving a stale copy of its own safety configuration is the last thing that
        should need a restart to notice a change."""
        return get_settings()

    def set_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        self._confirmation_callback = callback

    def set_unattended(self, unattended: bool = True) -> None:
        """Run with nobody present to answer a confirmation prompt.

        Used by `--mode run-scheduled`, where the OS scheduler starts Leti with no
        terminal and no window. Blocking on a prompt there would hang the run
        forever, and auto-approving everything would mean the OS scheduler could
        do things the user never agreed to. Instead the classes listed in
        scheduler.unattended_allows proceed and everything else is refused with a
        reason the user can read in the task's history.
        """
        self._unattended = unattended

    @property
    def unattended_classes(self) -> List[str]:
        configured = self.settings.get("scheduler", {}).get(
            "unattended_allows", DEFAULT_UNATTENDED_CLASSES
        )
        # CRITICAL is never in this set, whatever the setting says - the same
        # reasoning as require_confirmation_for: irreversible actions need a person.
        return [str(c).lower() for c in configured if str(c).lower() != "critical"]

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def get_tier(self, tool_name: str, case: Optional[str] = None) -> RiskTier:
        """The action class configured for a tool, for this particular call.

        An unregistered tool is treated as CRITICAL: a tool nobody classified is
        one nobody thought about, and the safe reading of that is the cautious one.

        `case` comes from the tool's own BaseTool.action_case() and matters only for
        the few tools that cover acts of genuinely different weight - launch_app
        opening a web page versus launch_app starting a program. The class for each
        case is still read from permissions.yaml (`action_by_case:`), so the tool
        says which situation this is and this file says what that situation costs.
        A case nobody configured falls back to the tool's plain `action:`.
        """
        tool_cfg = self.permissions.get("tools", {}).get(tool_name)
        if not tool_cfg:
            return RiskTier.CRITICAL
        raw = tool_cfg.get("action", tool_cfg.get("tier", "critical"))
        if case:
            raw = (tool_cfg.get("action_by_case") or {}).get(str(case), raw)
        try:
            return RiskTier.from_config(raw)
        except ValueError:
            return RiskTier.CRITICAL

    def requires_confirmation(self, tier: RiskTier) -> bool:
        """Whether this action class needs the user to say yes first.

        Reads safety.require_confirmation_for, which the user controls. Note that
        the setting can only be consulted for classes below CRITICAL - see
        authorize() for why removing critical from the list doesn't disarm it.
        """
        configured = self.settings.get("safety", {}).get(
            "require_confirmation_for", DEFAULT_CONFIRMATION_CLASSES
        )
        names = {str(c).lower() for c in configured}
        # Accept the old tier names here too, so an existing setting keeps meaning
        # what it meant: risky -> modify, destructive -> critical.
        if "risky" in names:
            names |= {"modify", "external"}
        if "destructive" in names:
            names.add("critical")
        return tier.value in names

    def _matches_forbidden_shell_pattern(self, command: str) -> Optional[str]:
        """Substring match against the configured pattern list, on a
        whitespace-normalized copy so 'rm  -rf  /' still matches 'rm -rf /'.

        This is a speed bump, NOT a security boundary: a blocklist can't
        enumerate every spelling of a destructive shell command ('rm -fr /',
        'rm -r -f /', 'find / -delete', a base64'd payload piped to sh). What
        actually protects the user is the confirmation prompt every shell call
        goes through plus _command_touches_protected_path below. Patterns here
        just stop the few catastrophic one-liners that shouldn't even reach a
        prompt, where a mistyped 'yes' would be unrecoverable.
        """
        normalized = re.sub(r"\s+", " ", command.lower()).strip()
        for pattern in self.permissions.get("forbidden_shell_patterns", []):
            if re.sub(r"\s+", " ", str(pattern).lower()).strip() in normalized:
                return pattern
        return None

    def _touches_protected_path(self, path_str: str) -> Optional[str]:
        """Returns the matching protected_paths entry if `path_str` names a
        protected location or anything beneath it, else None.

        Compares canonical paths (resolve_path normalizes '..' and follows
        symlinks) segment-by-segment rather than by string prefix. A prefix
        match is wrong in both directions: it misses '/tmp/../etc/shadow' and
        it falsely blocks '/etcetera/notes.txt' because that string happens to
        start with '/etc'.
        """
        try:
            candidate = resolve_path(path_str)
        except (OSError, ValueError):
            return None

        for protected in self.permissions.get("protected_paths", []):
            expanded = os.path.expanduser(os.path.expandvars(str(protected)))
            # Entries like 'C:\Users\*\AppData' are glob patterns, not literal
            # paths, so they can't be resolved or compared segment-wise.
            if any(ch in expanded for ch in "*?["):
                if fnmatch.fnmatch(str(candidate), expanded) or fnmatch.fnmatch(
                    str(candidate), os.path.join(expanded, "*")
                ):
                    return protected
                continue

            try:
                protected_path = Path(expanded).resolve()
            except OSError:
                protected_path = Path(os.path.normpath(expanded))

            if candidate == protected_path or protected_path in candidate.parents:
                return protected
        return None

    def _command_touches_protected_path(self, command: str) -> Optional[str]:
        """Returns the protected location a shell command references, if any.

        permissions.yaml documents protected_paths as places Leti must never
        touch, but checking only the path-shaped ARGUMENT KEYS below leaves the
        one tool that can touch anything unchecked: 'cat ~/.ssh/id_rsa' has no
        'path' argument at all, just a command string. So the command is
        tokenized and every path-shaped token is checked the same way.

        Heuristic by nature - a shell word can be built at runtime from a
        variable or a substitution this never sees. It raises the floor for the
        obvious cases; it does not make run_shell_command safe on its own, which
        is why that tool still requires confirmation on every call.
        """
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            # Unbalanced quotes - fall back to whitespace splitting rather than
            # skipping the check entirely.
            tokens = command.split()

        for raw in tokens:
            # Strip redirection/pipe/separator punctuation that shlex leaves
            # attached to an adjacent word (e.g. '>/etc/hosts' -> '/etc/hosts').
            token = raw.lstrip("<>|&;()").strip("'\"")
            if not token:
                continue
            looks_like_path = (
                token.startswith(("/", "~", "./", "../"))
                or os.sep in token
                or (os.altsep and os.altsep in token)
            )
            if not looks_like_path:
                continue
            matched = self._touches_protected_path(token)
            if matched:
                return matched
        return None

    # Arguments that end up being executed by a shell. `code` is here because
    # run_code takes a bash snippet, which is the same act as run_shell_command's
    # `command` under a different parameter name - checking only `command` meant
    # "rm -rf /" was refused as a command and run as a one-line bash program. The
    # patterns are matched against non-shell code too: `rm -rf /` inside a Python
    # string is either about to be passed to os.system or is a false positive worth
    # one explanation, and the cautious reading is the right one at a hard block.
    _EXECUTED_TEXT_KEYS = ("command", "cmd", "code")

    # Arguments that name somewhere on the web. launch_app is here because it
    # absorbed open_url: the page it opens arrives as `app_name`, and a blocklist
    # that only reads `url` would have stopped applying the moment the tools merged.
    _URL_KEYS = ("url", "app_name")

    def check_hard_block(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Returns a reason string if the call must be blocked unconditionally, else None."""
        for key in self._EXECUTED_TEXT_KEYS:
            text = arguments.get(key)
            if not text:
                continue
            matched = self._matches_forbidden_shell_pattern(str(text))
            if matched:
                return f"Command matches forbidden pattern: '{matched}'"
            matched = self._command_touches_protected_path(str(text))
            if matched:
                return f"Command references protected location '{matched}'"

        for key in ("path", "file_path", "target_path", "source_path", "destination_path"):
            if key in arguments and arguments[key]:
                matched = self._touches_protected_path(str(arguments[key]))
                if matched:
                    return f"Path '{arguments[key]}' touches protected location '{matched}'"

        blocked_domains = self.permissions.get("blocked_domains", [])
        if blocked_domains:
            for key in self._URL_KEYS:
                value = arguments.get(key)
                if not value:
                    continue
                for blocked in blocked_domains:
                    if blocked in str(value):
                        return f"URL matches blocked domain: '{blocked}'"

        return None

    # ------------------------------------------------------------------ #
    # Main entry point used by the orchestrator / tool executor
    # ------------------------------------------------------------------ #
    async def authorize(
        self, tool_name: str, arguments: Dict[str, Any], preapproved: bool = False,
        case: Optional[str] = None,
    ) -> "Authorization":
        """
        Raises PermissionDenied or ConfirmationDenied if the call should not proceed.
        Returns an Authorization if it may - check `.execute` before running the tool,
        since a dry-run authorization deliberately permits the call without executing it.

        `preapproved`: set when the user's own request already expressed clear approval
        for this action (e.g. a voice command whose wording already confirms intent - see
        core/intent_signals.py) so the interactive confirmation step can be skipped. This
        NEVER bypasses hard blocks, the FORBIDDEN tier, or the DESTRUCTIVE tier - only the
        confirmation prompt for an otherwise-permitted RISKY call. The caller is told via
        `.used_preapproval` when it was spent, so a single spoken "go ahead" authorizes a
        single action rather than every action for the rest of the turn.

        `case`: for a tool that covers acts of different weight, which one this call
        is - see get_tier() and BaseTool.action_case().
        """
        tier = self.get_tier(tool_name, case)

        block_reason = self.check_hard_block(tool_name, arguments)
        if block_reason:
            await self._audit(tool_name, arguments, tier, "blocked", block_reason)
            raise PermissionDenied(block_reason)

        if tier == RiskTier.FORBIDDEN:
            await self._audit(tool_name, arguments, tier, "blocked", "Tool is globally forbidden")
            raise PermissionDenied(f"Tool '{tool_name}' is forbidden by configuration.")

        description = _humanize_tool_call(tool_name, arguments)

        # CRITICAL always confirms, whatever the setting says. Letting the user
        # switch off confirmation for irreversible actions would turn one config
        # edit into "delete anything, silently" - the setting is there to let
        # people relax the prompts they find noisy, not to remove the last check.
        if tier != RiskTier.CRITICAL and not self.requires_confirmation(tier):
            await self._audit(tool_name, arguments, tier, "auto_approved")
            return Authorization(execute=True, description=description)

        # dry_run: authorize, but report back that the tool must NOT actually run.
        # Checked before the confirmation prompt because there is nothing to confirm -
        # nothing is going to happen. Read-only SAFE tools above still execute, so the
        # model can research and plan normally while every state-changing call is
        # reported instead of performed.
        if self.settings["safety"].get("dry_run"):
            await self._audit(tool_name, arguments, tier, "dry_run", "dry_run mode - not executed")
            return Authorization(execute=False, description=description)

        # A critical action is never allowed to ride on approval inferred from the
        # phrasing of the original request - the wording that pre-approves "move this
        # file" shouldn't silently cover an unrelated delete the model chose on its own.
        # Those always get their own explicit yes/no.
        if preapproved and tier == RiskTier.CRITICAL:
            preapproved = False

        if preapproved:
            await self._audit(
                tool_name, arguments, tier, "user_approved",
                "Pre-approved: the user's own request already expressed clear approval.",
            )
            return Authorization(execute=True, used_preapproval=True, description=description)

        # Nobody is here to ask (a scheduled run started by the OS scheduler).
        if self._unattended:
            allowed = self.unattended_classes
            if tier.value in allowed:
                await self._audit(
                    tool_name, arguments, tier, "auto_approved",
                    f"Unattended run: '{tier.value}' is permitted by scheduler.unattended_allows.",
                )
                return Authorization(execute=True, description=description)
            await self._audit(
                tool_name, arguments, tier, "user_denied",
                f"Unattended run: '{tier.value}' needs a person to confirm it.",
            )
            raise ConfirmationDenied(
                f"'{tool_name}' is a '{tier.value}' action, and this task ran unattended with "
                f"nobody to confirm it. Unattended runs may do: {', '.join(allowed)}. Either run "
                f"this task while Leti is open, or add '{tier.value}' to scheduler.unattended_allows "
                f"in settings.yaml if you want it to happen without you."
            )

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
        return Authorization(execute=True, description=description)

    def _build_confirmation_prompt(
        self, tool_name: str, arguments: Dict[str, Any], tier: RiskTier
    ) -> str:
        action_desc = _humanize_tool_call(tool_name, arguments)
        severity = {
            RiskTier.CRITICAL: "This can't be undone.",
            RiskTier.EXTERNAL: "This reaches outside your computer.",
            RiskTier.MODIFY: "This will make a change on your system.",
        }.get(tier, "This will run on your computer.")
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
            arguments=_redact_arguments(arguments),
            tier=tier.value if isinstance(tier, RiskTier) else str(tier),
            decision=decision,
            detail=detail,
        )
        line = record.to_json() + "\n"
        async with self._lock:
            # Writing to a permanent record shouldn't stall every other coroutine
            # (the GUI's websocket server shares this loop), so the blocking file
            # I/O goes to a thread. The lock still serializes appends.
            await asyncio.get_running_loop().run_in_executor(None, self._append_audit_line, line)

    def _append_audit_line(self, line: str) -> None:
        with open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())   # an audit record that a crash can lose isn't one

    async def audit_result(self, tool_name: str, arguments: Dict[str, Any], success: bool,
                           detail: str = "", case: Optional[str] = None) -> None:
        """Call after execution to log the outcome, separate from the authorization decision."""
        tier = self.get_tier(tool_name, case)
        decision = "executed" if success else "error"
        await self._audit(tool_name, arguments, tier, decision, detail)
