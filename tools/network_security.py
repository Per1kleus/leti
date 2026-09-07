"""
Read-only local network / port security diagnostics.

Scope, deliberately: this module only inspects the machine Leti runs on
(open listening ports, the services bound to them, local firewall status)
and does a passive ARP-table read of devices already visible on the LAN.
It never scans, probes, or attempts to connect to other hosts' ports -
that would be indistinguishable from doing recon against someone else's
device and is out of scope for a personal assistant.
"""
from __future__ import annotations

import platform
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List

from tools.base import BaseTool, ToolParameter, ToolResult

# Ports that are commonly targeted / frequently misconfigured to be open
# to the world. Presence alone isn't proof of a problem - it's a prompt
# to check *why* it's open and who can reach it.
WATCH_PORTS: Dict[int, str] = {
    21: "FTP (unencrypted credentials)",
    22: "SSH (brute-force target if exposed to the internet)",
    23: "Telnet (unencrypted, should basically never be open)",
    25: "SMTP (open relay risk)",
    135: "MS RPC (frequent Windows worm vector)",
    139: "NetBIOS (legacy SMB, worm vector)",
    445: "SMB (EternalBlue-class exploits)",
    1433: "MSSQL",
    3306: "MySQL",
    3389: "RDP (top brute-force / ransomware entry point)",
    5432: "PostgreSQL",
    5900: "VNC (often runs with no/weak auth)",
    6379: "Redis (frequently deployed with no auth)",
    27017: "MongoDB (frequently deployed with no auth)",
}


@dataclass
class PortFinding:
    port: int
    proto: str
    process: str
    note: str = ""
    concern: str = ""


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return ""


def _listening_ports_psutil() -> List[PortFinding]:
    import psutil  # optional dep; declared in requirements.txt

    findings: List[PortFinding] = []
    for conn in psutil.net_connections(kind="inet"):
        if conn.status != psutil.CONN_LISTEN or not conn.laddr:
            continue
        port = conn.laddr.port
        proc_name = ""
        if conn.pid:
            try:
                proc_name = psutil.Process(conn.pid).name()
            except Exception:
                proc_name = f"pid:{conn.pid}"
        proto = "tcp" if conn.type == socket.SOCK_STREAM else "udp"
        note = WATCH_PORTS.get(port, "")
        concern = "review" if note else ""
        findings.append(PortFinding(port=port, proto=proto, process=proc_name, note=note, concern=concern))
    return findings


class ScanLocalPortsTool(BaseTool):
    name = "scan_local_ports"
    description = (
        "List all ports currently listening on THIS machine, the process bound to each, "
        "and flag any that are commonly-exploited or frequently misconfigured (e.g. exposed "
        "RDP, SMB, Redis with no auth). Read-only - does not scan or contact other devices."
    )
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        try:
            findings = _listening_ports_psutil()
        except ImportError:
            return ToolResult(
                success=False,
                error="psutil is required for port scanning. Add it to requirements.txt and pip install.",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        flagged = [f for f in findings if f.note]
        return ToolResult(
            success=True,
            output={
                "total_listening": len(findings),
                "all_ports": [f.__dict__ for f in findings],
                "flagged_for_review": [f.__dict__ for f in flagged],
                "summary": (
                    f"{len(findings)} listening port(s), {len(flagged)} flagged for review."
                    if findings
                    else "No listening ports found."
                ),
            },
        )


class FirewallStatusTool(BaseTool):
    name = "check_firewall_status"
    description = "Check whether the OS firewall is enabled and report its current rule summary."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        system = platform.system()
        try:
            if system == "Linux":
                out = _run(["ufw", "status", "verbose"])
                if not out:
                    out = _run(["sudo", "-n", "ufw", "status", "verbose"])
                enabled = "Status: active" in out
                return ToolResult(success=True, output={"engine": "ufw", "enabled": enabled, "raw": out or "ufw not found or needs sudo"})
            elif system == "Darwin":
                out = _run(["/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate"])
                enabled = "enabled" in out.lower()
                return ToolResult(success=True, output={"engine": "pf/ALF", "enabled": enabled, "raw": out})
            elif system == "Windows":
                out = _run(["netsh", "advfirewall", "show", "allprofiles", "state"])
                enabled = "ON" in out.upper()
                return ToolResult(success=True, output={"engine": "Windows Defender Firewall", "enabled": enabled, "raw": out})
            else:
                return ToolResult(success=False, error=f"Unsupported platform: {system}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LanDeviceListTool(BaseTool):
    name = "list_lan_devices"
    description = (
        "Passively list devices already visible in this machine's ARP/neighbor table on the "
        "local network (IP, MAC, vendor if resolvable). Does not ping-sweep or port-scan other "
        "hosts - it only reads entries the OS already knows about from normal traffic."
    )
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        system = platform.system()
        try:
            if system == "Windows":
                out = _run(["arp", "-a"])
            else:
                out = _run(["arp", "-a"]) or _run(["ip", "neigh"])
            return ToolResult(success=True, output={"raw": out or "No ARP entries found (table may be empty)."})
        except Exception as e:
            return ToolResult(success=False, error=str(e))
