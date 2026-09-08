"""
Read-only local network / port security diagnostics.

Scope, deliberately: this module only inspects the machine Leti runs on
(its sockets and the processes behind them, local firewall status) and does
a passive ARP-table read of devices already visible on the LAN. It never
scans, probes, or attempts to connect to other hosts' ports - that would be
indistinguishable from doing recon against someone else's device and is out
of scope for a personal assistant.

Listening ports and established outbound connections are one tool here rather
than one here and one in system_defense.py: they are two filters over the same
walk of psutil.net_connections(), which is the expensive part. Asking both
questions separately paid for that walk twice and could return answers taken a
moment apart that disagreed about the same socket.
"""
from __future__ import annotations

import asyncio
import platform
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools.base import BaseTool, ToolParameter, ToolResult
from tools.command_runner import run_command

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


def _connections(state: str = "all") -> Dict[str, Any]:
    """Both socket views from one walk of the connection table.

    "Which ports are open on this machine" and "which processes are talking to the
    outside world" were two tools reading the same psutil.net_connections() list and
    keeping opposite halves of it. That is a slow call on a busy host - a few hundred
    milliseconds of kernel work plus a process lookup per row - so answering both
    questions used to cost it twice, and the two answers were taken a moment apart
    and could disagree about a socket that opened in between. One pass, filtered here.
    """
    import psutil  # optional dep; declared in requirements.txt

    want_listening = state in ("listening", "all")
    want_established = state in ("established", "all")

    # One Process lookup per pid rather than per socket: a browser with fifty
    # connections is one process, and psutil.Process() re-reads /proc each time.
    processes: Dict[int, Dict[str, str]] = {}

    def describe(pid: Optional[int]) -> Dict[str, str]:
        if not pid:
            return {"process": "", "exe": ""}
        if pid not in processes:
            try:
                proc = psutil.Process(pid)
                processes[pid] = {"process": proc.name(), "exe": proc.exe() or ""}
            except Exception:
                # The process exited between listing the socket and looking it up,
                # or it belongs to another user - the pid is still worth reporting.
                processes[pid] = {"process": f"pid:{pid}", "exe": ""}
        return processes[pid]

    listening: List[PortFinding] = []
    established: List[Dict[str, Any]] = []

    for conn in psutil.net_connections(kind="inet"):
        if want_listening and conn.status == psutil.CONN_LISTEN and conn.laddr:
            port = conn.laddr.port
            note = WATCH_PORTS.get(port, "")
            listening.append(PortFinding(
                port=port,
                proto="tcp" if conn.type == socket.SOCK_STREAM else "udp",
                process=describe(conn.pid)["process"],
                note=note,
                concern="review" if note else "",
            ))
        elif want_established and conn.status == psutil.CONN_ESTABLISHED and conn.pid:
            info = describe(conn.pid)
            established.append({
                "pid": conn.pid,
                "process": info["process"],
                "exe": info["exe"],
                "remote": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
            })

    report: Dict[str, Any] = {}
    if want_listening:
        flagged = [f for f in listening if f.note]
        report["listening"] = {
            "total": len(listening),
            "all_ports": [f.__dict__ for f in listening],
            "flagged_for_review": [f.__dict__ for f in flagged],
        }
    if want_established:
        # Unfamiliar executables first: a process with no resolvable path making
        # outbound calls is the one worth looking at, and sorting by name buries it.
        established.sort(key=lambda r: (bool(r["exe"]), r["process"]))
        report["established"] = {"total": len(established), "connections": established}
    return report


class InspectNetworkConnectionsTool(BaseTool):
    name = "inspect_network_connections"
    description = (
        "What this machine's network sockets are doing. 'listening' shows the ports open on "
        "it and the process behind each, flagging commonly-exploited ones (exposed RDP, SMB, "
        "Redis without auth) - that's the attack surface. 'established' shows which processes "
        "have live outbound connections and to where - that's what's phoning home. 'all' "
        "gives both, which is what you want when investigating rather than checking one "
        "thing.\n"
        "Read-only, and about THIS machine only - it never scans or contacts other devices."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(
            name="state", type="string", required=False,
            enum=["listening", "established", "all"],
            description="Which sockets to report. Default 'all'.",
        ),
    ]

    async def run(self, state: str = "all", **kwargs) -> ToolResult:
        if state not in ("listening", "established", "all"):
            return ToolResult(success=False, error="state must be listening, established, or all.")
        try:
            # net_connections() walks every socket on the machine; on a busy host
            # that's long enough to stutter the GUI if run on the event loop.
            result = await asyncio.get_running_loop().run_in_executor(None, _connections, state)
        except ImportError:
            return ToolResult(success=False, error=(
                "psutil is required to inspect network connections. Add it to requirements.txt "
                "and pip install."
            ))
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        parts = []
        if "listening" in result:
            parts.append(f"{result['listening']['total']} listening port(s), "
                         f"{len(result['listening']['flagged_for_review'])} flagged for review")
        if "established" in result:
            parts.append(f"{result['established']['total']} established outbound connection(s)")
        return ToolResult(success=True, output={**result, "summary": "; ".join(parts) + "."})


class FirewallStatusTool(BaseTool):
    name = "check_firewall_status"
    description = "Check whether the OS firewall is enabled and report its current rule summary."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        system = platform.system()
        try:
            if system == "Linux":
                result = await run_command(["ufw", "status", "verbose"])
                if not result.ok:
                    result = await run_command(["sudo", "-n", "ufw", "status", "verbose"])
                engine, marker = "ufw", "Status: active"
            elif system == "Darwin":
                result = await run_command(
                    ["/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate"]
                )
                engine, marker = "pf/ALF", "enabled"
            elif system == "Windows":
                result = await run_command(["netsh", "advfirewall", "show", "allprofiles", "state"])
                engine, marker = "Windows Defender Firewall", "ON"
            else:
                return ToolResult(success=False, error=f"Unsupported platform: {system}")

            if not result.ok:
                # "The command failed" is not the same fact as "the firewall is off",
                # and reporting the second when we only know the first is how a user
                # ends up believing a machine is protected (or not) on no evidence.
                return ToolResult(
                    success=False,
                    error=(f"Couldn't determine firewall status - {result.failure_reason()}. "
                           f"This usually means the tool isn't installed or the query needs "
                           f"elevated privileges."),
                )

            enabled = marker.lower() in result.stdout.lower()
            return ToolResult(
                success=True,
                output={"engine": engine, "enabled": enabled, "raw": result.stdout.strip()},
            )
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
        try:
            result = await run_command(["arp", "-a"])
            if not result.ok and platform.system() != "Windows":
                result = await run_command(["ip", "neigh"])
            if not result.ok:
                return ToolResult(
                    success=False,
                    error=f"Couldn't read the ARP/neighbor table - {result.failure_reason()}.",
                )
            return ToolResult(
                success=True,
                output={"raw": result.stdout.strip() or "No ARP entries found (table may be empty)."},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
