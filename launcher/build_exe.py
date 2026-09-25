"""Build Leti.exe. Run this on Windows, from the project root.

    python launcher\\build_exe.py

It installs PyInstaller if it is not there, builds launcher/Leti.spec, and checks
the result is what it should be. That check is the reason this exists rather than
just a documented command: an executable that builds without error and then cannot
find main.py is a build that looked fine.

PyInstaller is not in requirements.txt on purpose. It is needed to MAKE a release,
not to run Leti, and every user installing a build tool they will never use is
several megabytes and one more thing to go wrong on first launch.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "launcher" / "Leti.spec"
BUILT = ROOT / "dist" / ("Leti.exe" if sys.platform.startswith("win") else "Leti")

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


def main() -> int:
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
                           "--noconfirm", str(SPEC)], cwd=str(ROOT))
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

    print()
    print("Put it in the Leti folder - the one with main.py - and double-click it.")
    print("On first run it prepares Python and the packages, then starts Leti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
