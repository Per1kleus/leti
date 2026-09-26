"""Leti.exe: what happens when somebody double-clicks Leti for the first time.

This is the whole program inside the executable. It carries a Python of its own -
that is what makes a .exe a .exe - and it uses it for exactly one thing: running
launcher/bootstrap.py, which prepares a real Python environment in the Leti folder
and then starts main.py with it.

WHY IT DOES NOT SIMPLY CONTAIN LETI

An executable with Leti and all twenty-seven dependencies inside it would be
several gigabytes - Whisper and its tensor library alone are most of that - and it
would have to be rebuilt and re-downloaded for every change to any of them. It
also could not install anything, because a frozen interpreter has no venv and no
ensurepip: PyInstaller unpacks a runtime, not a Python installation.

So the executable is a front door rather than a container. It is small, it is the
same on every release, and what it prepares is an ordinary Python environment that
can be inspected, repaired and updated like any other.

THE CONSOLE

It is a console program, because the first run genuinely has something to say: it
downloads a few hundred megabytes and the person who started it should be able to
see that this is what is happening rather than that nothing is. Once Leti's own
window is up the console has nothing left to report, so it is hidden. If setup
fails it stays, with the reason on it, because a window that vanishes is the
failure this is meant to avoid.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    """The Leti folder: the one holding main.py.

    Frozen, sys.executable is Leti.exe, so the folder to look in is the one it
    sits in - and then its parent, so the executable can also live in a bin\\
    subfolder. Unfrozen, this file's own grandparent is the repository, which is
    what makes the launcher runnable from a checkout without being built.
    """
    if getattr(sys, "frozen", False):
        here = Path(sys.executable).resolve().parent
    else:
        here = Path(__file__).resolve().parent.parent
    for candidate in (here, here.parent, Path.cwd()):
        if (candidate / "main.py").exists() and (candidate / "requirements.txt").exists():
            return candidate
    return here


def hide_console() -> None:
    """Put the setup window away once there is nothing left to report.

    Windows only, and never fatal: a console that will not hide is a cosmetic
    problem, and taking the launch down over it would not be.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        window = ctypes.windll.kernel32.GetConsoleWindow()
        if window:
            ctypes.windll.user32.ShowWindow(window, 0)     # SW_HIDE
    except Exception:
        pass


def show_console() -> None:
    """Bring the setup window back, because there is something to read after all.

    The mirror of hide_console(). Leti is started AFTER the console is hidden, so
    anything Leti says on its way out - "Cannot reach Ollama at ...", a traceback,
    a missing model - goes to a window nobody can see, and a user who
    double-clicked Leti.exe watches nothing happen. This is the file whose whole
    stated purpose is that a window which vanishes is the failure to avoid; it just
    did not cover Leti failing rather than setup failing.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        window = ctypes.windll.kernel32.GetConsoleWindow()
        if window:
            ctypes.windll.user32.ShowWindow(window, 5)      # SW_SHOW
            ctypes.windll.user32.SetForegroundWindow(window)
    except Exception:
        pass


def wait_for_the_user(hold: bool = True) -> None:
    """Hold the window open so a message can be read.

    Only ever called when something went wrong. A launcher that closes on failure
    leaves somebody looking at a folder with no idea what happened.

    `hold` is False when this process's output is being read by something else -
    --print-python, which the .bat launcher captures. There the prompt would go
    into the capture instead of onto the screen, and the wait would be a hang
    nobody could see the reason for. The .bat does its own pausing.
    """
    if not hold:
        return
    try:
        input("Press Enter to close this window. ")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    # The project folder goes on the path so `import core` works - the bootstrap
    # reuses core/atomic_write.py, and this file is inside the project when it is
    # run from a checkout rather than from the executable.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from launcher import bootstrap
    except Exception:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from launcher import bootstrap

    arguments = list(argv if argv is not None else sys.argv[1:])
    # --setup-only is for checking that a machine can be prepared without then
    # opening the interface. Everything else is passed to Leti untouched, so the
    # executable can start any mode main.py supports rather than only the GUI.
    setup_only = "--setup-only" in arguments
    if setup_only:
        arguments.remove("--setup-only")
    # --print-python prepares the machine and then prints the interpreter Leti
    # should be started with, and nothing else. The .bat launcher reads it rather
    # than guessing: it cannot tell a venv built from a good system Python from a
    # runtime fetched because the system one was too old, and guessing wrong means
    # running Leti on the interpreter that was rejected.
    print_python = "--print-python" in arguments
    if print_python:
        arguments.remove("--print-python")
        setup_only = True
    # --install-shortcuts is the repair operation: it puts back a Desktop or Start
    # Menu shortcut that was deleted or retargeted, without touching anything
    # else. Separate from setup because setup places them once and then leaves
    # them alone - reading a .lnk on every launch is two PowerShell processes for
    # a question whose answer almost never changes.
    if "--install-shortcuts" in arguments:
        arguments.remove("--install-shortcuts")
        from launcher import shortcuts

        progress = bootstrap.Progress()
        placed = shortcuts.install(root)
        if not placed:
            # Two different nothings, and telling somebody the wrong one sends
            # them to fix the wrong thing.
            if shortcuts.best_target(root) is None:
                progress.say("There is nothing to make a shortcut to yet - build "
                             "Leti.exe, or keep this next to the launcher.")
            else:
                progress.say("Found Leti, but no Desktop or Start Menu to put it "
                             "in. This only works on Windows.")
            return 1
        for line in shortcuts.describe(placed) or ["Your Leti shortcuts are correct."]:
            progress.step(line)
        bootstrap.write_state(root, shortcuts=True)
        return 0 if all(row["action"] != "failed" for row in placed) else 1

    progress = bootstrap.Progress(out=sys.stderr if print_python else sys.stdout)
    if not (root / "main.py").exists():
        progress.problem(
            "Leti's own files are not next to this launcher",
            "find main.py in this folder, or the one above it",
            retry_is_safe=False)
        progress.say("  Keep Leti.exe inside the Leti folder it came with.")
        wait_for_the_user(hold=not print_python)
        return 2

    try:
        python = bootstrap.prepare(root, progress)
    except bootstrap.SetupFailed as failure:
        progress.problem(failure.what, failure.trying, failure.retry_is_safe,
                         logs=root / "logs")
        wait_for_the_user(hold=not print_python)
        return 1
    except KeyboardInterrupt:
        # Closed halfway through on purpose. Nothing is broken: every step records
        # only that it finished, so the next launch carries on from the last one
        # that did.
        progress.say()
        progress.say("Setup stopped. Starting Leti again will carry on from here.")
        return 130
    except Exception as e:
        progress.problem(f"something unexpected went wrong ({type(e).__name__})",
                         "prepare Leti's Python environment", retry_is_safe=True,
                         logs=root / "logs")
        wait_for_the_user(hold=not print_python)
        return 1

    if print_python:
        # The one line of output this mode produces, on its own, so `for /f` in a
        # batch file reads a path and not a progress report.
        print(python)
        return 0
    if setup_only:
        progress.say("Setup finished. Leti was not started (--setup-only).")
        return 0

    hide_console()
    try:
        status = bootstrap.start_leti(python, root, arguments or None)
    except KeyboardInterrupt:
        return 130

    # 0 is a normal close. 130 is the user interrupting on purpose. Anything else
    # means Leti stopped for a reason, and that reason went to a console this
    # function hid a moment ago - so it comes back, rather than the launch simply
    # appearing to do nothing.
    if status not in (0, 130):
        show_console()
        progress.say()
        progress.problem(
            f"Leti started and then stopped (exit code {status})",
            "run Leti itself, after preparing this folder successfully",
            retry_is_safe=True, logs=root / "logs")
        progress.say("  The reason is in the output above, and in logs/leti.log.")
        progress.say("  If it says it cannot reach Ollama: Ollama is what runs the")
        progress.say("  model Leti thinks with. Install it from https://ollama.com,")
        progress.say("  then start Leti again.")
        # hold= even though this branch cannot be reached in --print-python mode -
        # that mode returns before the console is hidden and before Leti is started.
        # Relying on that ordering would make this wait correct by accident, and the
        # next edit to the order above would turn it into a hang with no visible
        # reason. Every wait in this file says whether it is being captured.
        wait_for_the_user(hold=not print_python)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
