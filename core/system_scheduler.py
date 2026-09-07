"""Registering Leti's scheduled tasks with the operating system's scheduler.

This is not a second scheduler. It teaches cron, launchd or Windows Task
Scheduler to run one command periodically:

    python main.py --mode run-scheduled

which starts Leti, runs whatever tools/scheduler.py says is due, and exits. All
the task logic - schedules, retries, history, disabling - stays where it was;
what's added is that the OS starts that check even when Leti isn't open, so a
task scheduled for Monday morning happens on Monday morning rather than the next
time somebody launches the app.

One installer per platform because there is no common mechanism: cron on Linux,
launchd on macOS (cron there needs Full Disk Access granted to /usr/sbin/cron,
which is a worse thing to ask of someone than a plist), and schtasks on Windows.
Each writes a single entry tagged with MARKER so it can be found and removed
again without disturbing anything else the user has scheduled.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKER = "leti-scheduled-tasks"
TASK_NAME = "Leti Scheduled Tasks"
LAUNCHD_LABEL = "com.leti.scheduledtasks"
DEFAULT_INTERVAL_MINUTES = 5


def python_executable() -> str:
    """The interpreter to run Leti with.

    Prefers the venv the launchers create: cron and launchd run with a bare
    environment, so `python3` from PATH there is usually the system one, without
    any of Leti's dependencies installed.
    """
    for candidate in (
        PROJECT_ROOT / "leti_env" / "bin" / "python",
        PROJECT_ROOT / "leti_env" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def log_path() -> Path:
    directory = PROJECT_ROOT / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "scheduled_runs.log"


def _command() -> List[str]:
    return [python_executable(), str(PROJECT_ROOT / "main.py"), "--mode", "run-scheduled"]


def _run(argv: List[str], stdin: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a scheduler command, returning a failed result rather than raising.

    The tool that isn't there is the normal case, not an exception: a container or
    a minimal install has no crontab, and asking for the status of something on a
    machine that can't schedule at all should answer "not installed", not crash.
    """
    try:
        return subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, 127, "", f"{argv[0]} is not installed")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, "", f"{argv[0]} timed out")
    except OSError as e:
        return subprocess.CompletedProcess(argv, 1, "", str(e))


# --- cron (Linux) -----------------------------------------------------------------

def _cron_line(interval_minutes: int) -> str:
    command = " ".join(f'"{part}"' if " " in part else part for part in _command())
    return (
        f"*/{interval_minutes} * * * * cd \"{PROJECT_ROOT}\" && {command} "
        f">> \"{log_path()}\" 2>&1  # {MARKER}"
    )


def _read_crontab() -> List[str]:
    result = _run(["crontab", "-l"])
    # An empty crontab exits non-zero with "no crontab for <user>", which is not
    # an error to report - it just means there's nothing there yet.
    if result.returncode != 0 and "no crontab" not in (result.stderr or "").lower():
        return []
    return [line for line in (result.stdout or "").splitlines()]


def _write_crontab(lines: List[str]) -> subprocess.CompletedProcess:
    body = "\n".join(line for line in lines if line.strip()) + "\n"
    return _run(["crontab", "-"], stdin=body)


def _cron_install(interval_minutes: int) -> Dict[str, Any]:
    if not shutil.which("crontab"):
        return {"ok": False, "error": "cron isn't available on this system (no crontab command)."}
    lines = [line for line in _read_crontab() if MARKER not in line]
    lines.append(_cron_line(interval_minutes))
    result = _write_crontab(lines)
    if result.returncode != 0:
        return {"ok": False, "error": f"crontab rejected the entry: {result.stderr.strip()}"}
    return {"ok": True, "mechanism": "cron", "entry": _cron_line(interval_minutes)}


def _cron_uninstall() -> Dict[str, Any]:
    if not shutil.which("crontab"):
        return {"ok": True, "removed": False, "note": "cron isn't available here."}
    lines = _read_crontab()
    remaining = [line for line in lines if MARKER not in line]
    if len(remaining) == len(lines):
        return {"ok": True, "removed": False, "note": "No Leti entry was in the crontab."}
    result = _write_crontab(remaining) if remaining else _run(["crontab", "-r"])
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip()}
    return {"ok": True, "removed": True}


def _cron_status() -> Dict[str, Any]:
    if not shutil.which("crontab"):
        return {"installed": False, "mechanism": "cron", "entries": [],
                "available": False,
                "note": "cron isn't available on this system, so tasks can only run while Leti is open."}
    entries = [line for line in _read_crontab() if MARKER in line]
    return {"installed": bool(entries), "mechanism": "cron", "available": True, "entries": entries}


# --- launchd (macOS) ----------------------------------------------------------------

def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _launchd_install(interval_minutes: int) -> Dict[str, Any]:
    arguments = "".join(f"        <string>{part}</string>\n" for part in _command())
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{arguments}    </array>
    <key>WorkingDirectory</key><string>{PROJECT_ROOT}</string>
    <key>StartInterval</key><integer>{interval_minutes * 60}</integer>
    <key>StandardOutPath</key><string>{log_path()}</string>
    <key>StandardErrorPath</key><string>{log_path()}</string>
    <!-- Run once at load, and again if the machine was asleep at the due time. -->
    <key>RunAtLoad</key><false/>
</dict>
</plist>
"""
    path = _plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plist)

    # Unload first: launchctl refuses to load a label that's already loaded, and
    # reinstalling with a new interval is the common case.
    _run(["launchctl", "unload", str(path)])
    result = _run(["launchctl", "load", str(path)])
    if result.returncode != 0:
        return {"ok": False, "error": f"launchctl load failed: {result.stderr.strip()}",
                "plist": str(path)}
    return {"ok": True, "mechanism": "launchd", "plist": str(path)}


def _launchd_uninstall() -> Dict[str, Any]:
    path = _plist_path()
    if not path.is_file():
        return {"ok": True, "removed": False, "note": "No Leti launch agent was installed."}
    _run(["launchctl", "unload", str(path)])
    path.unlink()
    return {"ok": True, "removed": True}


def _launchd_status() -> Dict[str, Any]:
    path = _plist_path()
    loaded = False
    if shutil.which("launchctl"):
        result = _run(["launchctl", "list"])
        loaded = LAUNCHD_LABEL in (result.stdout or "")
    return {"installed": path.is_file(), "mechanism": "launchd",
            "plist": str(path) if path.is_file() else None, "loaded": loaded}


# --- Task Scheduler (Windows) --------------------------------------------------------

def _schtasks_install(interval_minutes: int) -> Dict[str, Any]:
    command = " ".join(f'\\"{part}\\"' if " " in part else part for part in _command())
    result = _run([
        "schtasks", "/Create", "/TN", TASK_NAME,
        "/TR", f'cmd /c cd /d "{PROJECT_ROOT}" && {command} >> "{log_path()}" 2>&1',
        "/SC", "MINUTE", "/MO", str(interval_minutes), "/F",
    ])
    if result.returncode != 0:
        return {"ok": False, "error": f"schtasks failed: {(result.stderr or result.stdout).strip()}"}
    return {"ok": True, "mechanism": "Windows Task Scheduler", "task_name": TASK_NAME}


def _schtasks_uninstall() -> Dict[str, Any]:
    result = _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
    if result.returncode != 0:
        if "cannot find" in (result.stderr or result.stdout or "").lower():
            return {"ok": True, "removed": False, "note": "No Leti task was scheduled."}
        return {"ok": False, "error": (result.stderr or result.stdout).strip()}
    return {"ok": True, "removed": True}


def _schtasks_status() -> Dict[str, Any]:
    result = _run(["schtasks", "/Query", "/TN", TASK_NAME])
    return {"installed": result.returncode == 0, "mechanism": "Windows Task Scheduler",
            "task_name": TASK_NAME if result.returncode == 0 else None}


# --- Dispatch ---------------------------------------------------------------------

def _backend():
    system = platform.system()
    if system == "Darwin":
        return _launchd_install, _launchd_uninstall, _launchd_status
    if system == "Windows":
        return _schtasks_install, _schtasks_uninstall, _schtasks_status
    return _cron_install, _cron_uninstall, _cron_status


def install(interval_minutes: int = DEFAULT_INTERVAL_MINUTES) -> Dict[str, Any]:
    """Register the periodic check with this machine's scheduler."""
    interval_minutes = max(1, min(int(interval_minutes), 1440))
    install_fn, _, _ = _backend()
    result = install_fn(interval_minutes)
    result.update({
        "interval_minutes": interval_minutes,
        "command": " ".join(_command()),
        "log": str(log_path()),
    })
    return result


def uninstall() -> Dict[str, Any]:
    _, uninstall_fn, _ = _backend()
    return uninstall_fn()


def status() -> Dict[str, Any]:
    _, _, status_fn = _backend()
    result = status_fn()
    result.update({
        "platform": platform.system(),
        "command": " ".join(_command()),
        "python": python_executable(),
        "log": str(log_path()) if log_path().exists() else None,
    })
    return result


if __name__ == "__main__":
    # Runnable directly, mirroring the launcher installers in scripts/:
    #   python core/system_scheduler.py install [minutes] | uninstall | status
    import json

    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "install":
        minutes = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_INTERVAL_MINUTES
        print(json.dumps(install(minutes), indent=2))
    elif action == "uninstall":
        print(json.dumps(uninstall(), indent=2))
    else:
        print(json.dumps(status(), indent=2))
