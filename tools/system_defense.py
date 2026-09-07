"""
Active defense against the most common attack patterns a home/personal
machine faces: exposed services, brute-force login attempts, and
persistence mechanisms (malware/unwanted software that re-launches itself
on startup). Detection is read-only and safe; every remediation action
(enabling the firewall, killing a process, removing a startup entry) is
tiered risky/destructive in permissions.yaml so SafetyGuard requires
confirmation before Leti actually changes anything.
"""
from __future__ import annotations

import platform
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool, ToolParameter, ToolResult


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return ""


class EnableFirewallTool(BaseTool):
    name = "enable_firewall"
    description = "Turn on the OS firewall with default deny-incoming rules. Risky: changes system config."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        system = platform.system()
        try:
            if system == "Linux":
                out = _run(["sudo", "-n", "ufw", "--force", "enable"])
            elif system == "Darwin":
                out = _run(["sudo", "-n", "/usr/libexec/ApplicationFirewall/socketfilterfw", "--setglobalstate", "on"])
            elif system == "Windows":
                out = _run(["netsh", "advfirewall", "set", "allprofiles", "state", "on"])
            else:
                return ToolResult(success=False, error=f"Unsupported platform: {system}")
            return ToolResult(success=True, output=out or "Firewall enable command issued.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


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
            if system == "Linux":
                for log_path in ("/var/log/auth.log", "/var/log/secure"):
                    p = Path(log_path)
                    if not p.exists():
                        continue
                    text = _run(["sudo", "-n", "tail", "-n", "5000", str(p)]) or ""
                    for line in text.splitlines():
                        if "Failed password" in line or "authentication failure" in line:
                            m = re.search(r"from ([\d.]+)", line)
                            if m:
                                attempts_by_ip[m.group(1)] += 1
            elif system == "Windows":
                # Security event log, event ID 4625 = failed logon
                out = _run([
                    "powershell", "-Command",
                    "Get-WinEvent -FilterHashtable @{LogName='Security';Id=4625} -MaxEvents 2000 "
                    "| Select-Object -ExpandProperty Message"
                ])
                for m in re.finditer(r"Source Network Address:\s*([\d.]+)", out):
                    attempts_by_ip[m.group(1)] += 1
            else:
                return ToolResult(success=False, error=f"Log-based brute-force detection not implemented for {system} (macOS: check Console.app manually).")

            flagged = {ip: count for ip, count in attempts_by_ip.items() if count >= threshold}
            return ToolResult(
                success=True,
                output={
                    "sources_checked": len(attempts_by_ip),
                    "flagged_sources": flagged,
                    "summary": (
                        f"{len(flagged)} source(s) exceeded {threshold} failed attempts."
                        if flagged else "No brute-force pattern detected in available logs."
                    ),
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
                entries.append(_run(["systemctl", "--user", "list-unit-files", "--state=enabled"]))
                entries.append("--- crontab ---")
                entries.append(_run(["crontab", "-l"]))
            elif system == "Darwin":
                for path in ("~/Library/LaunchAgents", "/Library/LaunchAgents", "/Library/LaunchDaemons"):
                    p = Path(path).expanduser()
                    if p.exists():
                        entries += [str(f) for f in p.glob("*.plist")]
            elif system == "Windows":
                entries.append(_run([
                    "powershell", "-Command",
                    "Get-CimInstance Win32_StartupCommand | Select-Object Name, Command, Location"
                ]))
            else:
                return ToolResult(success=False, error=f"Unsupported platform: {system}")

            return ToolResult(success=True, output={"entries": [e for e in entries if e]})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ListSuspiciousProcessesTool(BaseTool):
    name = "list_suspicious_processes"
    description = (
        "List running processes that have an active outbound network connection, sorted so "
        "unfamiliar or unsigned executables making network calls are easy to spot. Read-only."
    )
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        try:
            import psutil

            rows = []
            for conn in psutil.net_connections(kind="inet"):
                if conn.status != psutil.CONN_ESTABLISHED or not conn.pid:
                    continue
                try:
                    proc = psutil.Process(conn.pid)
                    rows.append({
                        "pid": conn.pid,
                        "process": proc.name(),
                        "exe": proc.exe() if proc.exe() else "",
                        "remote": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                    })
                except Exception:
                    continue
            return ToolResult(success=True, output={"connections": rows, "count": len(rows)})
        except ImportError:
            return ToolResult(success=False, error="psutil is required. Add to requirements.txt and pip install.")
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
