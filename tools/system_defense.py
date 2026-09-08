"""
Active defense against the most common attack patterns a home/personal
machine faces: exposed services, brute-force login attempts, and
persistence mechanisms (malware/unwanted software that re-launches itself
on startup). Detection is read-only and safe; every remediation action
(enabling the firewall, killing a process, removing a startup entry) is
tiered risky/destructive in permissions.yaml so SafetyGuard requires
confirmation before Leti actually changes anything.

Processes with live outbound connections are not listed here: that is the same
walk of psutil.net_connections as the listening-port scan, so both live in
tools/network_security.py's inspect_network_connections, which answers either
question or both from one pass.
"""
from __future__ import annotations

import platform
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool, ToolParameter, ToolResult
from tools.command_runner import run_command


class EnableFirewallTool(BaseTool):
    name = "enable_firewall"
    description = "Turn on the OS firewall with default deny-incoming rules. Risky: changes system config."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        system = platform.system()
        if system == "Linux":
            cmd = ["sudo", "-n", "ufw", "--force", "enable"]
        elif system == "Darwin":
            cmd = ["sudo", "-n", "/usr/libexec/ApplicationFirewall/socketfilterfw", "--setglobalstate", "on"]
        elif system == "Windows":
            cmd = ["netsh", "advfirewall", "set", "allprofiles", "state", "on"]
        else:
            return ToolResult(success=False, error=f"Unsupported platform: {system}")

        result = await run_command(cmd, timeout=30)
        if not result.ok:
            # Almost always 'sudo -n' with no cached credential, or not being an
            # admin on Windows. Reporting success here told the user their firewall
            # was on when it wasn't.
            hint = ""
            if "sudo" in cmd and ("password" in result.stderr.lower() or result.returncode == 1):
                hint = (" Leti can't enter a sudo password - run this yourself in a terminal: "
                        + " ".join(c for c in cmd if c not in ("sudo", "-n")))
            return ToolResult(
                success=False,
                error=f"Could not enable the firewall - {result.failure_reason()}.{hint}",
            )
        return ToolResult(success=True, output=result.stdout.strip() or "Firewall enabled.")


class DetectBruteForceTool(BaseTool):
    name = "detect_brute_force_attempts"
    description = (
        "Scan local authentication logs (SSH/RDP/login) for repeated failed login attempts "
        "from the same source, which indicates a brute-force attack in progress. Read-only."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(
            name="threshold",
            type="number",
            description="Minimum number of failed attempts from one source to flag it.",
            required=False,
        ),
    ]

    async def run(self, threshold: int = 5, **kwargs) -> ToolResult:
        system = platform.system()
        try:
            attempts_by_ip: Counter = Counter()
            logs_read: List[str] = []
            logs_unreadable: List[str] = []
            if system == "Linux":
                for log_path in ("/var/log/auth.log", "/var/log/secure"):
                    p = Path(log_path)
                    if not p.exists():
                        continue
                    result = await run_command(["sudo", "-n", "tail", "-n", "5000", str(p)])
                    if not result.ok:
                        # Auth logs are usually root:adm 0640, so this is the common
                        # case, not the exception. Saying "no brute-force detected"
                        # after failing to read the log is a false all-clear.
                        logs_unreadable.append(log_path)
                        continue
                    logs_read.append(log_path)
                    text = result.stdout
                    for line in text.splitlines():
                        if "Failed password" in line or "authentication failure" in line:
                            m = re.search(r"from ([\d.]+)", line)
                            if m:
                                attempts_by_ip[m.group(1)] += 1
            elif system == "Windows":
                # Security event log, event ID 4625 = failed logon
                result = await run_command([
                    "powershell", "-NoProfile", "-Command",
                    "Get-WinEvent -FilterHashtable @{LogName='Security';Id=4625} -MaxEvents 2000 "
                    "| Select-Object -ExpandProperty Message"
                ], timeout=60)
                if not result.ok:
                    # Reading the Security log needs an elevated shell.
                    return ToolResult(
                        success=False,
                        error=("Couldn't read the Windows Security event log - "
                               f"{result.failure_reason()}. This usually needs an "
                               "administrator terminal."),
                    )
                logs_read.append("Windows Security event log")
                for m in re.finditer(r"Source Network Address:\s*([\d.]+)", result.stdout):
                    attempts_by_ip[m.group(1)] += 1
            else:
                return ToolResult(success=False, error=f"Log-based brute-force detection not implemented for {system} (macOS: check Console.app manually).")

            if not logs_read:
                return ToolResult(
                    success=False,
                    error=("Couldn't read any authentication log"
                           + (f" ({', '.join(logs_unreadable)} exist but aren't readable - "
                              "they're usually root-owned, so this needs sudo)"
                              if logs_unreadable else " - none found on this system")
                           + ". No conclusion can be drawn about brute-force attempts."),
                )

            flagged = {ip: count for ip, count in attempts_by_ip.items() if count >= threshold}
            summary = (
                f"{len(flagged)} source(s) exceeded {threshold} failed attempts."
                if flagged else "No brute-force pattern detected in the logs that could be read."
            )
            if logs_unreadable:
                summary += f" Note: could not read {', '.join(logs_unreadable)}."
            return ToolResult(
                success=True,
                output={
                    "sources_checked": len(attempts_by_ip),
                    "logs_read": logs_read,
                    "logs_unreadable": logs_unreadable,
                    "flagged_sources": flagged,
                    "summary": summary,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CheckPersistenceTool(BaseTool):
    name = "check_startup_persistence"
    description = (
        "List programs configured to auto-launch at login/boot (startup folders, registry Run "
        "keys, systemd/launch agents, cron) so unfamiliar or suspicious persistence entries - "
        "a hallmark of malware - can be spotted. Read-only."
    )
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        system = platform.system()
        entries: List[str] = []
        try:
            if system == "Linux":
                for path in ("~/.config/autostart", "/etc/xdg/autostart"):
                    p = Path(path).expanduser()
                    if p.exists():
                        entries += [str(f) for f in p.glob("*.desktop")]
                entries.append("--- systemd user services ---")
                entries.append((await run_command(["systemctl", "--user", "list-unit-files", "--state=enabled"])).stdout)
                entries.append("--- crontab ---")
                entries.append((await run_command(["crontab", "-l"])).stdout)
            elif system == "Darwin":
                for path in ("~/Library/LaunchAgents", "/Library/LaunchAgents", "/Library/LaunchDaemons"):
                    p = Path(path).expanduser()
                    if p.exists():
                        entries += [str(f) for f in p.glob("*.plist")]
            elif system == "Windows":
                entries.append((await run_command([
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_StartupCommand | Select-Object Name, Command, Location"
                ], timeout=60)).stdout)
            else:
                return ToolResult(success=False, error=f"Unsupported platform: {system}")

            return ToolResult(success=True, output={"entries": [e for e in entries if e]})
        except Exception as e:
            return ToolResult(success=False, error=str(e))



class KillProcessTool(BaseTool):
    name = "kill_process"
    description = "Terminate a process by PID. Destructive: use only after confirming it's unwanted/malicious."
    parameters: List[ToolParameter] = [
        ToolParameter(name="pid", type="number", description="Process ID to terminate."),
    ]

    async def run(self, pid: int, **kwargs) -> ToolResult:
        try:
            import psutil

            proc = psutil.Process(int(pid))
            name = proc.name()
            proc.terminate()
            return ToolResult(success=True, output=f"Sent terminate signal to PID {pid} ({name}).")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
