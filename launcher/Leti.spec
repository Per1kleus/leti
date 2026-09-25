# PyInstaller build configuration for Leti.exe.
#
# Build it on Windows, from the project root, with:
#
#     python -m pip install pyinstaller
#     python -m PyInstaller --clean --noconfirm launcher/Leti.spec
#
# The result is dist\Leti.exe. Put it in the Leti folder - the one holding
# main.py - and double-click it.
#
# WHAT IS IN IT, AND WHAT IS NOT
#
# In it: launcher/leti_launcher.py, launcher/bootstrap.py, core/atomic_write.py,
# and the Python standard library. That is about 10 MB and it is the same on
# every release.
#
# Not in it: Leti. Not the application and not one of its twenty-seven
# dependencies. Whisper's tensor library alone would make this several gigabytes,
# it would have to be rebuilt for every dependency bump, and a frozen interpreter
# cannot install anything anyway - PyInstaller unpacks a runtime, not a Python
# installation. So the executable prepares an ordinary Python environment in the
# Leti folder and starts main.py with it, which is also exactly what the
# double-click launcher does. One Leti, two front doors.
#
# It is a console program (console=True) on purpose. The first run downloads a
# few hundred megabytes and the person who started it should be able to see that
# happening; launcher/leti_launcher.py hides the window once Leti's own is up,
# and leaves it visible with the reason on it if setup fails.

import os

project_root = os.path.abspath(os.path.join(os.path.dirname(SPECPATH), "."))
if not os.path.exists(os.path.join(project_root, "main.py")):
    project_root = os.path.abspath(SPECPATH + "/..")

icon = os.path.join(project_root, "gui", "icons", "leti.ico")

a = Analysis(
    [os.path.join(project_root, "launcher", "leti_launcher.py")],
    pathex=[project_root],
    binaries=[],
    # No data files. The executable deliberately reads requirements.txt and
    # main.py from the folder it is in rather than carrying copies: a copy would
    # be a second Leti to keep in step with the first.
    datas=[],
    hiddenimports=[
        # Imported inside functions, so the analyser does not see them.
        "urllib.request",
        "zipfile",
        "hashlib",
        "core.atomic_write",
        # Imported lazily by bootstrap.place_shortcuts and by the launcher's
        # --install-shortcuts branch. Without it the executable builds, runs, and
        # silently never puts Leti on the Desktop.
        "launcher.shortcuts",
    ],
    hookspath=[],
    runtime_hooks=[],
    # Kept out because nothing in the launcher uses them, and leaving them in
    # would add tens of megabytes to a file whose whole point is being small.
    excludes=[
        "tkinter", "numpy", "pandas", "matplotlib", "scipy", "PIL",
        "sympy", "whisper", "torch", "chromadb", "httpx", "playwright",
        "pytest", "sqlite3", "unittest", "pydoc", "doctest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="Leti",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # See the note at the top: the setup output is the point of the window, and
    # leti_launcher.hide_console() puts it away once there is nothing to report.
    console=True,
    disable_windowed_traceback=False,
    icon=icon if os.path.exists(icon) else None,
)
