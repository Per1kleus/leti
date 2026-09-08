"""
System specs, health/performance monitoring, and OS update management.

specs, health and pending updates are one tool with sections rather than three,
because they were never really separate: health already collected the disk and
battery figures specs reported, and already called the update check internally.
Three tools meant asking twice for the same numbers and getting two answers that
could disagree.

This is a sibling to tools/system_defense.py and tools/network_security.py
rather than an addition to either: those two are specifically about
*security* (attack surface, intrusion detection, firewall). This module is
about *health* - hardware specs, resource pressure, and OS patching - a
different concern that happens to also live under "system". Where the two
domains overlap (e.g. a health check surfacing a problem an existing
security tool already knows how to fix), this module points at that tool
by name rather than re-implementing it - see the `related_tool` field on
each issue in system_report's health section.

Update automation is real but scoped: apply_system_updates runs the
platform's actual update mechanism (apt/dnf/pacman, softwareupdate, or
Windows Update) rather than simulating it, and - like every other
system-changing tool here - is gated behind SafetyGuard confirmation, which
is what satisfies "awaits user agreement" for anything it wants to change.
"""
from __future__ import annotations

import asyncio
import platform
import time
from typing import Any, Dict, List, Optional

from core.config_loader import get_settings
from tools.base import BaseTool, ToolParameter, ToolResult
from tools.command_runner import CommandResult, run_command

DEFAULT_THRESHOLDS = {
    "cpu_percent": 90,
    "memory_percent": 85,
    "swap_percent": 60,
    "disk_percent": 90,
    "temperature_celsius": 85,
    "battery_low_percent": 15,
}

# Pseudo/read-only filesystems that show up in disk_partitions() but aren't real,
# user-manageable storage - squashfs in particular is near-universal on Ubuntu (every
# installed snap mounts one, always reported as 100% full by design since it's a
# read-only image). Reporting these as "disk full" would be noise on most real machines.
_EXCLUDED_FSTYPES = {"squashfs", "tmpfs", "devtmpfs", "overlay", "iso9660", "autofs"}
_MIN_DISK_SIZE_GB = 0.5  # also skip anything this small - almost never a real user volume


def _real_disk_partitions():
    import psutil

    for part in psutil.disk_partitions(all=False):
        if part.fstype in _EXCLUDED_FSTYPES:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        if _bytes_to_gb(usage.total) < _MIN_DISK_SIZE_GB:
            continue
        yield part, usage


def _thresholds() -> Dict[str, float]:
    cfg = get_settings().get("system_health", {}).get("thresholds", {})
    return {**DEFAULT_THRESHOLDS, **cfg}


async def _run(cmd: List[str], timeout: int = 20) -> CommandResult:
    """Async so a package manager can't stall the assistant. apply_system_updates
    allows 30 minutes; run synchronously that froze the event loop - and with it
    the GUI's websocket server, so chat, voice and every connected phone went dead
    with no visible cause - for the whole upgrade."""
    return await run_command(cmd, timeout=timeout)


def _bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 2)


# --- Specs ---------------------------------------------------------------------

async def _collect_specs() -> Dict[str, Any]:
    """Static hardware and OS facts - what this machine IS."""
    import psutil

    cpu_freq = psutil.cpu_freq()
    vm = psutil.virtual_memory()

    disks = []
    for part, usage in _real_disk_partitions():
        disks.append({
            "mountpoint": part.mountpoint,
            "filesystem": part.fstype,
            "total_gb": _bytes_to_gb(usage.total),
            "used_gb": _bytes_to_gb(usage.used),
            "free_gb": _bytes_to_gb(usage.free),
            "percent_used": usage.percent,
        })

    battery = None
    try:
        b = psutil.sensors_battery()
        if b:
            battery = {"percent": b.percent, "plugged_in": b.power_plugged}
    except Exception:
        pass

    gpu_info = "not detected (no vendor-specific GPU query implemented for this platform)"
    try:
        nvidia = await _run(["nvidia-smi", "--query-gpu=name,memory.total",
                             "--format=csv,noheader"], timeout=5)
        if nvidia.returncode == 0 and nvidia.stdout.strip():
            gpu_info = nvidia.stdout.strip()
    except Exception:
        pass

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "architecture": platform.machine(),
        "cpu": {
            "model": platform.processor() or "unknown",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency_mhz": cpu_freq.max if cpu_freq else None,
        },
        "memory_total_gb": _bytes_to_gb(vm.total),
        "disks": disks,
        "gpu": gpu_info,
        "battery": battery,
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
    }



# --- Health check ----------------------------------------------------------------

def _top_processes(limit: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    import psutil

    procs = list(psutil.process_iter(["pid", "name"]))
    for p in procs:
        try:
            p.cpu_percent(None)  # prime the internal sample window
        except Exception:
            pass
    time.sleep(0.4)

    rows = []
    for p in procs:
        try:
            rows.append({
                "pid": p.pid,
                "name": p.info.get("name", "?"),
                "cpu_percent": p.cpu_percent(None),
                "memory_percent": round(p.memory_percent(), 1),
            })
        except Exception:
            continue

    by_cpu = sorted(rows, key=lambda r: r["cpu_percent"], reverse=True)[:limit]
    by_mem = sorted(rows, key=lambda r: r["memory_percent"], reverse=True)[:limit]
    return {"top_cpu": by_cpu, "top_memory": by_mem}


class SystemReportTool(BaseTool):
    name = "system_report"
    description = (
        "Report on this machine. Choose which sections you need:\n"
        "  specs   - what the machine IS: CPU, RAM, disks, GPU, OS, uptime, battery\n"
        "  health  - how it's DOING right now: CPU/memory/swap/disk pressure, temperature, "
        "and a list of issues with suggested fixes\n"
        "  updates - pending OS/package updates\n"
        "Defaults to health, which is the usual question. Ask for specs when someone wants "
        "to know what hardware they have, and for all three when diagnosing something.\n"
        "Some issues name a tool that can resolve them (kill_process, close_app, "
        "apply_system_updates) - describe the fix in plain terms and ask before calling it."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(
            name="sections", type="array", items_type="string", required=False,
            description="Any of: specs, health, updates. Default ['health'].",
        ),
    ]

    async def run(self, sections: Optional[List[str]] = None, **kwargs) -> ToolResult:
        wanted = [str(x).lower() for x in (sections or ["health"])]
        if "all" in wanted:
            wanted = ["specs", "health", "updates"]
        unknown = [w for w in wanted if w not in ("specs", "health", "updates")]
        if unknown:
            return ToolResult(success=False, error=(
                f"Unknown section(s): {unknown}. Use specs, health and/or updates."
            ))

        report: Dict[str, Any] = {"sections": wanted}
        if "specs" in wanted:
            try:
                report["specs"] = await _collect_specs()
            except Exception as e:
                report["specs"] = {"error": str(e)}
        if "updates" in wanted and "health" not in wanted:
            # health already collects updates; don't run the package manager twice.
            report["updates"] = await _check_updates_impl()
        if "health" not in wanted:
            return ToolResult(success=True, output=report)

        try:
            import psutil

            t = _thresholds()
            issues: List[Dict[str, Any]] = []

            # interval=1 blocks for a full second by design (it samples). On the
            # event loop that's a visible stall in the GUI, so it goes to a thread.
            cpu_percent = await asyncio.get_running_loop().run_in_executor(
                None, lambda: psutil.cpu_percent(interval=1)
            )
            if cpu_percent >= t["cpu_percent"]:
                top = _top_processes(limit=3)
                worst = top["top_cpu"][0] if top["top_cpu"] else None
                issues.append({
                    "issue": f"CPU usage is at {cpu_percent}% (threshold {t['cpu_percent']}%).",
                    "severity": "warning",
                    "suggestion": (
                        f"Top consumer is '{worst['name']}' (pid {worst['pid']}) at {worst['cpu_percent']}% CPU - "
                        f"consider closing it if unexpected." if worst else "Investigate which process is driving load."
                    ),
                    "related_tool": "kill_process" if worst else None,
                    "related_args": {"pid": worst["pid"]} if worst else None,
                })

            vm = psutil.virtual_memory()
            if vm.percent >= t["memory_percent"]:
                top = _top_processes(limit=3)
                worst = top["top_memory"][0] if top["top_memory"] else None
                issues.append({
                    "issue": f"Memory usage is at {vm.percent}% (threshold {t['memory_percent']}%).",
                    "severity": "warning",
                    "suggestion": (
                        f"Top consumer is '{worst['name']}' (pid {worst['pid']}) at {worst['memory_percent']}% memory - "
                        f"consider closing it, or add more RAM if this is a recurring pattern." if worst
                        else "Consider closing unused applications or adding RAM."
                    ),
                    "related_tool": "kill_process" if worst else None,
                    "related_args": {"pid": worst["pid"]} if worst else None,
                })

            swap = psutil.swap_memory()
            if swap.total > 0 and swap.percent >= t["swap_percent"]:
                issues.append({
                    "issue": f"Swap usage is at {swap.percent}% (threshold {t['swap_percent']}%).",
                    "severity": "warning",
                    "suggestion": "Heavy swap use usually means not enough RAM for your current workload - close memory-heavy apps or add RAM.",
                    "related_tool": None, "related_args": None,
                })

            disk_reports = []
            for part, usage in _real_disk_partitions():
                disk_reports.append({"mountpoint": part.mountpoint, "percent_used": usage.percent})
                if usage.percent >= t["disk_percent"]:
                    issues.append({
                        "issue": f"Disk '{part.mountpoint}' is {usage.percent}% full (threshold {t['disk_percent']}%).",
                        "severity": "critical" if usage.percent >= 97 else "warning",
                        "suggestion": "Free up space: clear downloads/cache, empty trash, or move large files to external/cloud storage.",
                        "related_tool": None, "related_args": None,
                    })

            temps_report = {}
            try:
                temps = psutil.sensors_temperatures()
                for label, entries in (temps or {}).items():
                    for entry in entries:
                        temps_report[f"{label}:{entry.label or 'sensor'}"] = entry.current
                        if entry.current and entry.current >= t["temperature_celsius"]:
                            issues.append({
                                "issue": f"Temperature sensor '{label}' reads {entry.current}\u00b0C (threshold {t['temperature_celsius']}\u00b0C).",
                                "severity": "critical",
                                "suggestion": "Check for dust/blocked vents, ensure fans are spinning, and close CPU/GPU-intensive tasks until it cools.",
                                "related_tool": None, "related_args": None,
                            })
            except (AttributeError, NotImplementedError):
                pass  # not available on this platform (common on macOS/Windows without extra drivers)

            battery_report = None
            try:
                b = psutil.sensors_battery()
                if b:
                    battery_report = {"percent": b.percent, "plugged_in": b.power_plugged}
                    if not b.power_plugged and b.percent <= t["battery_low_percent"]:
                        issues.append({
                            "issue": f"Battery is low at {b.percent}% and not plugged in.",
                            "severity": "warning",
                            "suggestion": "Plug in the charger soon to avoid an unexpected shutdown.",
                            "related_tool": None, "related_args": None,
                        })
            except Exception:
                pass

            updates = await _check_updates_impl()
            if updates.get("pending_count", 0) > 0:
                issues.append({
                    "issue": f"{updates['pending_count']} system update(s) available.",
                    "severity": "info",
                    "suggestion": "Run apply_system_updates to install them.",
                    "related_tool": "apply_system_updates", "related_args": None,
                })

            severity_rank = {"critical": 0, "warning": 1, "info": 2}
            issues.sort(key=lambda i: severity_rank.get(i["severity"], 3))

            report["health"] = {
                "metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": vm.percent,
                    "swap_percent": swap.percent,
                    "disks": disk_reports,
                    "temperatures_celsius": temps_report or "not available on this platform",
                    "battery": battery_report,
                    "pending_updates": updates.get("pending_count", 0),
                },
                "issues": issues,
                "summary": "All systems nominal." if not issues else f"{len(issues)} issue(s) found.",
            }
            # health runs the update check anyway (a pending-updates count is one of
            # the things it reports), so asking for both sections costs one run of the
            # package manager, not two - that sharing is why these are one tool.
            if "updates" in wanted:
                report["updates"] = updates
            return ToolResult(success=True, output=report)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# --- Updates ---------------------------------------------------------------------

async def _check_updates_impl() -> Dict[str, Any]:
    system = platform.system()
    try:
        if system == "Linux":
            if (await _run(["which", "apt"], timeout=5)).returncode == 0:
                out = (await _run(["apt", "list", "--upgradable"], timeout=30)).stdout
                lines = [l for l in out.splitlines() if l and not l.startswith("Listing...")]
                return {"platform": "apt", "pending_count": len(lines), "packages": lines[:30]}
            if (await _run(["which", "dnf"], timeout=5)).returncode == 0:
                result = await _run(["dnf", "check-update", "--quiet"], timeout=45)
                lines = [l for l in result.stdout.splitlines() if l.strip()]
                return {"platform": "dnf", "pending_count": len(lines), "packages": lines[:30]}
            if (await _run(["which", "checkupdates"], timeout=5)).returncode == 0:
                result = await _run(["checkupdates"], timeout=30)
                lines = [l for l in result.stdout.splitlines() if l.strip()]
                return {"platform": "pacman", "pending_count": len(lines), "packages": lines[:30]}
            return {"platform": "unknown", "pending_count": 0, "note": "No supported package manager (apt/dnf/pacman-contrib) found."}

        elif system == "Darwin":
            out = (await _run(["softwareupdate", "-l"], timeout=60)).stdout
            lines = [l.strip() for l in out.splitlines() if l.strip().startswith("*")]
            return {"platform": "softwareupdate", "pending_count": len(lines), "packages": lines}

        elif system == "Windows":
            ps_cmd = (
                "$s=(New-Object -ComObject Microsoft.Update.Session).CreateUpdateSearcher();"
                "$r=$s.Search('IsInstalled=0');"
                "$r.Updates | ForEach-Object { $_.Title }"
            )
            out = (await _run(["powershell", "-NoProfile", "-Command", ps_cmd], timeout=90)).stdout
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            return {"platform": "windows_update", "pending_count": len(lines), "packages": lines[:30]}

        return {"platform": "unknown", "pending_count": 0, "note": f"Unsupported platform: {system}"}
    except FileNotFoundError:
        return {"platform": "unknown", "pending_count": 0, "note": "Update-checking tool not found on this system."}
    except Exception as e:
        return {"platform": "unknown", "pending_count": 0, "note": f"Update check failed: {e}"}



class ApplySystemUpdatesTool(BaseTool):
    name = "apply_system_updates"
    description = (
        "Download and install available OS/package updates. Risky: requires confirmation - "
        "this actually changes installed software and may require a reboot afterward."
    )
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        system = platform.system()
        try:
            if system == "Linux":
                if (await _run(["which", "apt"], timeout=5)).returncode == 0:
                    result = await _run(["sudo", "-n", "bash", "-c", "apt-get update && apt-get -y upgrade"], timeout=1800)
                elif (await _run(["which", "dnf"], timeout=5)).returncode == 0:
                    result = await _run(["sudo", "-n", "dnf", "-y", "upgrade"], timeout=1800)
                elif (await _run(["which", "pacman"], timeout=5)).returncode == 0:
                    result = await _run(["sudo", "-n", "pacman", "-Syu", "--noconfirm"], timeout=1800)
                else:
                    return ToolResult(success=False, error="No supported package manager (apt/dnf/pacman) found.")
            elif system == "Darwin":
                result = await _run(["sudo", "-n", "softwareupdate", "-i", "-a"], timeout=1800)
            elif system == "Windows":
                # No update-installation module ships by default; PSWindowsUpdate must be
                # present ('Install-Module PSWindowsUpdate -Force' as admin, one-time setup).
                ps_cmd = (
                    "if (Get-Module -ListAvailable -Name PSWindowsUpdate) {"
                    " Import-Module PSWindowsUpdate;"
                    " Install-WindowsUpdate -AcceptAll -IgnoreReboot | Out-String"
                    "} else {"
                    " Write-Output 'PSWindowsUpdate module not installed. Run as admin: Install-Module PSWindowsUpdate -Force'"
                    "}"
                )
                result = await _run(["powershell", "-NoProfile", "-Command", ps_cmd], timeout=1800)
            else:
                return ToolResult(success=False, error=f"Unsupported platform: {system}")

            if not result.ok:
                hint = ""
                if "sudo" in result.stderr.lower() or "password" in result.stderr.lower():
                    hint = (" Leti can't enter a sudo password - run the update yourself "
                            "in a terminal.")
                return ToolResult(
                    success=False,
                    error=f"Update command failed - {result.failure_reason()}.{hint}",
                )
            return ToolResult(success=True, output=result.stdout[-3000:] or "Updates applied.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
