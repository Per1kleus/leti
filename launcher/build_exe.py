"""Build Leti.exe. Run this on Windows, from the project root.

    python launcher\\build_exe.py

It installs PyInstaller if it is not there, builds launcher/Leti.spec, and checks
the result is what it should be. That check is the reason this exists rather than
just a documented command: an executable that builds without error and then cannot
find main.py is a build that looked fine.

The checking is launcher/verify_build.py's, which reads the file's own headers.
This used to check only that something appeared and that its size was plausible -
which cannot tell a Windows build from a Linux one renamed, an ARM build from an
x64 one, or a clean bundle from one that swept a .env file up on the way past.
Pass --keep-going to see every problem without failing the build, which is what
to use while investigating one.

PyInstaller is not in requirements.txt on purpose. It is needed to MAKE a release,
not to run Leti, and every user installing a build tool they will never use is
several megabytes and one more thing to go wrong on first launch.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from launcher import verify_build          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "launcher" / "Leti.spec"
ON_WINDOWS = sys.platform.startswith("win")
BUILT = ROOT / "dist" / ("Leti.exe" if ON_WINDOWS else "Leti")

# Well under a fat build and well over an empty one. A file outside this range
# means the spec's excludes stopped working or its entry point went missing, and
# both are worth failing the build over rather than shipping.
MIN_MEGABYTES = 3
MAX_MEGABYTES = 60


def have_pyinstaller() -> bool:
    """Is PyInstaller available? Asked without importing it - the import costs a
    second and a half and the answer is a filesystem lookup."""
    import importlib.util

    return importlib.util.find_spec("PyInstaller") is not None


def build_environment() -> Dict[str, str]:
    """The environment PyInstaller is run in, fixed so the build is reproducible.

    Two clean builds of the same source produced two different executables:
    measured on this project, 2 MB of differing bytes and a 1,264-byte difference in
    size. The cause is Python's per-process hash randomisation - PyInstaller walks
    sets and dicts of module names while assembling the archive, and the iteration
    order of those follows string hashes, which are seeded differently in every
    process. So the archive comes out in a different order each time and compresses
    differently.

    PYTHONHASHSEED=0 turns that off for the build subprocess, and two clean builds
    then produce byte-identical files with the same SHA-256. It is set here rather
    than left to whoever runs the build, because a release that cannot be
    reproduced cannot be checked against its source by anybody else.

    Nothing else is changed, and this affects only the build: Leti itself still runs
    with hash randomisation on, where it belongs.
    """
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    return environment


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    keep_going = "--keep-going" in arguments

    if not SPEC.exists():
        print(f"Cannot find {SPEC}. Run this from the Leti project folder.")
        return 2
    if not (ROOT / "main.py").exists():
        print("Cannot find main.py. Run this from the Leti project folder.")
        return 2

    if not sys.platform.startswith("win"):
        print("Note: building on " + sys.platform + ". PyInstaller produces an")
        print("executable for the system it runs on, so a Windows .exe has to be")
        print("built on Windows. Carrying on - the result will be for this system.")
        print()

    if not have_pyinstaller():
        print("Installing PyInstaller (needed to build, not to run Leti)...")
        done = subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
        if done.returncode != 0:
            print("PyInstaller could not be installed. Install it and try again.")
            return 1

    print(f"Building from {SPEC.name}...")
    done = subprocess.run([sys.executable, "-m", "PyInstaller", "--clean",
                           "--noconfirm", str(SPEC)], cwd=str(ROOT),
                          env=build_environment())
    if done.returncode != 0:
        print("The build failed - see PyInstaller's output above.")
        return 1

    if not BUILT.exists():
        print(f"The build reported success but {BUILT.name} is not in dist\\.")
        return 1
    megabytes = BUILT.stat().st_size / (1024 * 1024)
    print()
    print(f"Built: {BUILT}  ({megabytes:.1f} MB)")
    if megabytes < MIN_MEGABYTES:
        print(f"That is smaller than {MIN_MEGABYTES} MB, which usually means the")
        print("entry point was not bundled. Treating this as a failure.")
        return 1
    if megabytes > MAX_MEGABYTES:
        print(f"That is larger than {MAX_MEGABYTES} MB, which usually means the")
        print("spec's excludes stopped working and a dependency came along for")
        print("the ride. Treating this as a failure.")
        return 1

    # What was actually built, read out of the file rather than assumed from the
    # fact that PyInstaller exited zero.
    print()
    print("Checking what was built:")
    if ON_WINDOWS:
        result = verify_build.report(BUILT)
    else:
        # Not a PE and never will be - PyInstaller builds for the system it runs
        # on. The leak scan is still worth running, because what goes INTO the
        # bundle is decided by the spec and is the same question on any platform.
        result = {"path": str(BUILT), "pe": None, "problems": [],
                  "leaks": verify_build.scan_for_leaks(BUILT), "ok": True}
        result["ok"] = not result["leaks"]
        print(f"  Not Windows      this is a {sys.platform} build, so the PE checks")
        print("                   do not apply - a Windows Leti.exe has to be built")
        print("                   on Windows. The bundle scan below still applies.")
    print(verify_build.as_text(result))

    if not result["ok"]:
        print()
        if result["leaks"]:
            print("Something is in the bundle that should not ship. That is a")
            print("release blocker, not a warning.")
        print("Not treating this as a successful build.")
        if not keep_going:
            return 1
        print("(--keep-going was passed, so carrying on anyway.)")

    # Put it where everything else expects to find it. launcher/shortcuts.py looks
    # for Leti.exe beside main.py, not in dist\, so leaving it there meant the
    # Desktop shortcut kept pointing at the .bat and the executable was a file the
    # user had to know to move. Copied rather than built there directly, because
    # dist\ is what --clean empties.
    placed = ROOT / BUILT.name
    try:
        if placed.resolve() != BUILT.resolve():
            shutil.copy2(BUILT, placed)
    except OSError as e:
        print()
        print(f"Built, but could not copy it next to main.py ({e}).")
        print(f"Copy {BUILT} there yourself and it will work the same.")
        return 0

    print()
    print(f"Ready: {placed}")
    print("Double-click it, or run scripts\\install_windows_launcher.ps1 to point")
    print("your Desktop and Start Menu shortcuts at it instead of the .bat.")
    print("On first run it prepares Python and the packages, then starts Leti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
