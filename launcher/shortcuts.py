"""Leti on the Desktop and in the Start Menu, once and correctly.

A person who installed an assistant looks for it where they look for everything
else. Until now Leti was a file called "Launch Leti (Windows).bat" in a folder
they had to remember, and finding it meant knowing what a .bat is.

WHAT IT MAKES

    Desktop\\Leti.lnk
    Start Menu\\Programs\\Leti\\Leti.lnk

Two shortcuts, one name. Both point at the SAME thing - Leti.exe when the folder
has one, the .bat launcher when it does not - because a person has one Leti and
wants one icon for it. Building the executable later repairs the existing shortcut
to point at it rather than leaving two.

ONCE, AND CORRECTLY

Every one of these is a state, and all four are handled:

    nothing there          -> create it
    there and correct      -> leave it alone
    there and wrong        -> repair it
    there and unreadable   -> replace it

A .lnk is written to a fixed filename, so Windows overwrites rather than making
"Leti (1).lnk". But overwriting a correct shortcut still costs a COM object and a
disk write on every launch, and a shortcut the user deliberately moved would come
back - so what is there is read first, and left alone when it is right.

WHY THE DECISION IS HERE AND THE WRITING IS NOT

Reading and writing a .lnk needs Windows: the format is a COM object's business
and there is no supported way to make one from Linux. So this module decides -
which paths, which target, which icon, and whether anything needs doing - in
plain Python that can be tested anywhere, and hands the two or three resulting
facts to PowerShell, which does the ten lines only Windows can do.

PER-USER, ALWAYS

Everything is under the user's own profile. No administrator prompt, no registry,
no PATH, nothing in Program Files, and nothing another account can see. Removing
Leti is deleting its folder and its two shortcuts.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("leti.shortcuts")

# What the user sees. Not "Launch Leti (Windows).bat", which is a file name, and
# not "Leti Launcher", which is a thing that launches the thing they wanted.
NAME = "Leti"
LINK = f"{NAME}.lnk"
DESCRIPTION = "Leti - your local assistant"

# The Start Menu folder. A folder rather than a loose entry because that is the
# shape Windows applications use, and because it gives an uninstall something to
# remove.
START_MENU_FOLDER = NAME

# The icon, which already exists - seven sizes from 16 to 256 in one file, built
# from gui/icon.svg by scripts/build_icons.py. Nothing here draws or converts
# anything: the visual identity is already decided and committed.
ICON = Path("gui") / "icons" / "leti.ico"

# What a shortcut may point at, best first. The executable when it has been
# built, and the launcher that needs no build when it has not.
TARGETS = ("Leti.exe", "Launch Leti (Windows).bat")

# Minimised for the .bat, normal for the .exe - see `wanted`.
NORMAL_WINDOW = 1
MINIMISED_WINDOW = 7


@dataclass(frozen=True)
class Shortcut:
    """Everything a .lnk needs, and nothing Windows-specific."""

    path: Path                  # where the .lnk goes
    target: Path                # what it runs
    working_directory: Path     # the Leti folder
    icon: Optional[Path]        # gui/icons/leti.ico, or None if it is missing
    description: str = DESCRIPTION
    window_style: int = NORMAL_WINDOW

    def matches(self, existing: Optional[Dict[str, Any]]) -> bool:
        """Is what is already there already this?

        Compared on the three things that decide whether it WORKS - target,
        working directory, icon - and not on the description or the window style.
        A user who edited either of those in the shortcut's properties meant to,
        and having setup quietly undo it would be worse than leaving it.

        Paths are compared case-insensitively, because Windows does.
        """
        if not existing:
            return False
        if not _same_path(existing.get("target"), self.target):
            return False
        if not _same_path(existing.get("working_directory"), self.working_directory):
            return False
        if self.icon is None:
            return True
        return _same_path(_icon_file(existing.get("icon")), self.icon)


def _same_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return str(left).strip().strip('"').rstrip("\\/").lower() == \
        str(right).strip().strip('"').rstrip("\\/").lower()


def _icon_file(location: Any) -> Optional[str]:
    """The file out of an IconLocation, which carries a trailing ",0"."""
    if not location:
        return None
    text = str(location)
    # rsplit, not split: a path can contain a comma and an index cannot.
    if "," in text:
        head, _, tail = text.rpartition(",")
        if head and tail.strip().lstrip("-").isdigit():
            return head
    return text


# --------------------------------------------------------------------------- #
# Where things go
#
# Read from the environment rather than assembled from a user name, so a profile
# on another drive or a redirected Desktop is found. Each one is allowed to be
# missing: a machine with no Desktop folder is unusual and not an error.
# --------------------------------------------------------------------------- #

def desktop_directory(environment: Optional[Dict[str, str]] = None) -> Optional[Path]:
    env = dict(os.environ if environment is None else environment)
    for base in (env.get("USERPROFILE"), env.get("HOME")):
        if not base:
            continue
        candidate = Path(base) / "Desktop"
        if candidate.is_dir():
            return candidate
    return None


def start_menu_directory(environment: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """The per-user Programs folder. Never the all-users one, which needs admin."""
    env = dict(os.environ if environment is None else environment)
    appdata = env.get("APPDATA")
    if appdata:
        candidate = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        if candidate.is_dir():
            return candidate
    base = env.get("USERPROFILE") or env.get("HOME")
    if base:
        candidate = (Path(base) / "AppData" / "Roaming" / "Microsoft" / "Windows"
                     / "Start Menu" / "Programs")
        if candidate.is_dir():
            return candidate
    return None


def best_target(root: Path) -> Optional[Path]:
    """What a shortcut should point at: the executable if built, else the .bat.

    Order matters and is the point. A folder that gains a Leti.exe should have its
    shortcut repaired to use it, which falls out of this being asked every time
    rather than recorded once.
    """
    for name in TARGETS:
        candidate = Path(root) / name
        if candidate.exists():
            return candidate
    return None


def icon_for(root: Path) -> Optional[Path]:
    """Leti's icon, or None. Missing is not an error - a shortcut without one
    still works, and Windows draws the target's own icon instead."""
    candidate = Path(root) / ICON
    return candidate if candidate.exists() else None


def wanted(root: Path, environment: Optional[Dict[str, str]] = None) -> List[Shortcut]:
    """The shortcuts this folder should have. Empty when there is nothing to point
    at - which is a checkout with no executable built and no launcher, and is a
    reason to do nothing rather than to write a broken shortcut."""
    root = Path(root).resolve()
    target = best_target(root)
    if target is None:
        return []
    icon = icon_for(root)
    # The .bat runs in a console it cannot hand back - it starts Leti and waits
    # for it - so its window is sent to the taskbar rather than sitting over the
    # interface. The executable hides its own console once Leti's window is up
    # (see leti_launcher.hide_console), so it opens normally and its setup
    # progress is on screen where somebody waiting can see it.
    style = NORMAL_WINDOW if target.suffix.lower() == ".exe" else MINIMISED_WINDOW

    out: List[Shortcut] = []
    desktop = desktop_directory(environment)
    if desktop is not None:
        out.append(Shortcut(path=desktop / LINK, target=target,
                            working_directory=root, icon=icon, window_style=style))
    programs = start_menu_directory(environment)
    if programs is not None:
        out.append(Shortcut(path=programs / START_MENU_FOLDER / LINK, target=target,
                            working_directory=root, icon=icon, window_style=style))
    return out


# --------------------------------------------------------------------------- #
# Deciding, then doing
# --------------------------------------------------------------------------- #

KEPT = "kept"
CREATED = "created"
REPAIRED = "repaired"
FAILED = "failed"


def decide(shortcut: Shortcut, existing: Optional[Dict[str, Any]]) -> str:
    """KEPT, CREATED or REPAIRED - what needs doing to this one.

    `existing` is what was read off the disk, or None for "not there, or not
    readable". Unreadable is deliberately the same as absent: a .lnk that cannot
    be parsed is one to replace, and telling the two apart would buy nothing.
    """
    if existing is None:
        return CREATED
    return KEPT if shortcut.matches(existing) else REPAIRED


def install(root: Path, read: Optional[Callable] = None,
            write: Optional[Callable] = None,
            environment: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Make this folder's shortcuts right, and report what was done to each.

    Safe to run as often as you like. What is already correct is read and left
    alone, so running it twice is two reads and no writes - and a shortcut the
    user deliberately retargeted is repaired rather than silently duplicated.

    `read` and `write` are the two things only Windows can do, handed in so the
    deciding can be tested without one.
    """
    reader = read or read_shortcut
    writer = write or write_shortcut
    out: List[Dict[str, Any]] = []
    for shortcut in wanted(root, environment):
        try:
            existing = reader(shortcut.path)
        except Exception as e:
            logger.debug(f"Couldn't read {shortcut.path}: {e}")
            existing = None
        action = decide(shortcut, existing)
        if action != KEPT:
            try:
                writer(shortcut)
            except Exception as e:
                logger.debug(f"Couldn't write {shortcut.path}: {e}")
                out.append({"path": str(shortcut.path), "action": FAILED,
                            "why": type(e).__name__})
                continue
        out.append({"path": str(shortcut.path), "action": action,
                    "target": str(shortcut.target)})
    return out


def describe(results: List[Dict[str, Any]]) -> List[str]:
    """One line each, for the setup output. Says nothing when nothing changed."""
    lines: List[str] = []
    for row in results:
        where = "Desktop" if "desktop" in row["path"].lower() else "Start Menu"
        if row["action"] == CREATED:
            lines.append(f"Added Leti to your {where}.")
        elif row["action"] == REPAIRED:
            lines.append(f"Fixed the Leti shortcut on your {where}.")
        elif row["action"] == FAILED:
            lines.append(f"Could not add Leti to your {where} - "
                         "everything else is set up.")
    return lines


# --------------------------------------------------------------------------- #
# The parts that need Windows
#
# Two small PowerShell programs, given their values through the ENVIRONMENT - so a
# folder called "C:\\Users\\O'Brien & Sons" is a path and not a syntax error.
# Nothing here interpolates a path into a script body.
#
# It used to pass them as arguments after `--`, with a param() block to receive
# them, and that never worked: binding trailing arguments to param() is what
# -File does. With -Command, PowerShell joins everything after it onto the
# command text instead, so $LinkPath was never set, Mandatory tried to prompt for
# it, -NonInteractive refused, and PowerShell exited non-zero. Every shortcut
# failed, on both the Desktop and the Start Menu, with "Could not add Leti to
# your Desktop" - reported from a real Windows machine. The tests here inject a
# fake runner, so they checked the arguments were assembled correctly and never
# found out that PowerShell would not read them.
#
# An environment variable is not parsed as code by anything, which makes this
# both the working version and the safer one.
# --------------------------------------------------------------------------- #

_READ_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$LinkPath = $env:LETI_LINK_PATH
if (-not $LinkPath) { exit 4 }
if (-not (Test-Path -LiteralPath $LinkPath)) { exit 3 }
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($LinkPath)
[Console]::Out.WriteLine($sc.TargetPath)
[Console]::Out.WriteLine($sc.WorkingDirectory)
[Console]::Out.WriteLine($sc.IconLocation)
[Console]::Out.WriteLine($sc.Arguments)
"""

_WRITE_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$LinkPath         = $env:LETI_LINK_PATH
$TargetPath       = $env:LETI_TARGET_PATH
$WorkingDirectory = $env:LETI_WORKING_DIRECTORY
$IconLocation     = $env:LETI_ICON_LOCATION
$Description      = $env:LETI_DESCRIPTION
$WindowStyle      = $env:LETI_WINDOW_STYLE
if (-not $LinkPath -or -not $TargetPath -or -not $WorkingDirectory) { exit 4 }
$parent = Split-Path -Parent $LinkPath
if ($parent -and -not (Test-Path -LiteralPath $parent)) {
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
}
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($LinkPath)
$sc.TargetPath       = $TargetPath
$sc.WorkingDirectory = $WorkingDirectory
if ($Description) { $sc.Description = $Description }
if ($WindowStyle)  { $sc.WindowStyle = [int]$WindowStyle }
if ($IconLocation) { $sc.IconLocation = $IconLocation }
$sc.Save()
if (-not (Test-Path -LiteralPath $LinkPath)) { exit 5 }
"""


def _powershell(script: str, values: Dict[str, str],
                run: Optional[Callable] = None) -> Any:
    """Run one of the scripts above, with its values in the environment.

    -Command with the script on the command line, not -File: writing a temporary
    .ps1 would be a file to clean up, and -EncodedCommand would hide what is being
    run from anything looking at the process list. The values go in the
    environment, which -Command does not touch and PowerShell never parses as
    code - see the note above for what passing them as arguments did instead.
    """
    runner = run or subprocess.run
    command = ["powershell", "-NoProfile", "-NonInteractive",
               "-ExecutionPolicy", "Bypass", "-Command", script]
    environment = dict(os.environ)
    environment.update({name: str(value) for name, value in values.items()})
    return runner(command, capture_output=True, text=True, timeout=120,
                  env=environment)


def read_shortcut(path: Path, run: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
    """What the .lnk at `path` points at, or None if there is none to read."""
    if not _is_windows():
        return None
    done = _powershell(_READ_SCRIPT, {"LETI_LINK_PATH": str(path)}, run)
    if getattr(done, "returncode", 1) != 0:
        return None
    lines = (getattr(done, "stdout", "") or "").splitlines()
    while len(lines) < 4:
        lines.append("")
    return {"target": lines[0].strip(), "working_directory": lines[1].strip(),
            "icon": lines[2].strip(), "arguments": lines[3].strip()}


def write_shortcut(shortcut: Shortcut, run: Optional[Callable] = None) -> None:
    """Write the .lnk. Raises if PowerShell could not."""
    if not _is_windows():
        raise OSError("shortcuts can only be written on Windows")
    values = {
        "LETI_LINK_PATH": str(shortcut.path),
        "LETI_TARGET_PATH": str(shortcut.target),
        "LETI_WORKING_DIRECTORY": str(shortcut.working_directory),
        "LETI_ICON_LOCATION": f"{shortcut.icon},0" if shortcut.icon else "",
        "LETI_DESCRIPTION": shortcut.description or "",
        "LETI_WINDOW_STYLE": str(int(shortcut.window_style)),
    }
    done = _powershell(_WRITE_SCRIPT, values, run)
    if getattr(done, "returncode", 1) != 0:
        raise OSError((getattr(done, "stderr", "") or "PowerShell failed")[:300])


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")
