"""Is Ollama there, and is it answering? One answer, for both front doors.

WHY THIS EXISTS

Leti cannot answer anything without a model server. "Launch Leti (Windows).bat"
knew that and dealt with it inline - install Ollama with winget, start it, wait
for the port. Leti.exe did not: nothing anywhere under launcher/ mentioned Ollama,
so double-clicking the executable on a clean machine produced a Leti that opened,
could not reach a model, and said so in a log file behind a console the launcher
had just hidden. Two front doors that are supposed to be equivalent, and one of
them did not work on the machine it was written for.

So the step lives here, once, and launcher/bootstrap.py's prepare() runs it - which
is the one thing both front doors go through. The .bat no longer carries its own
copy.

WHAT IT DOES NOT DO

It does not choose a model, download one, or write any configuration.
core/model_setup.py owns that: it asks what this machine can run, shows its
arithmetic, and waits for an answer. This only makes sure there is a server for it
to ask. Keeping that line means there is still exactly one model-selection system.

IT NEVER BLOCKS THE LAUNCH

Reporting and carrying on, rather than refusing to start - because whether Leti
runs is not this step's decision to make, and the machine may be fine in a way this
cannot see (a server on another host, a model already loaded). So every function
here reports through Progress and returns a verdict; none of them raises
SetupFailed.

What happens next is worth being accurate about: main.py EXITS when it cannot reach
a model server, so Leti does not open and explain. That is why
launcher/leti_launcher.py brings the hidden console back when Leti stops with a
non-zero code - without which a user who double-clicked Leti.exe watches nothing
happen at all.

Standard library only, like the rest of launcher/ - it runs before anything is
installed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Optional, Sequence

DEFAULT_HOST = "http://localhost:11434"

# How long to wait for a server that has just been told to start. Ollama is
# usually answering in two or three seconds; thirty is for a cold disk on a
# machine that is also unpacking Python.
START_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 1.0

# One reachability check should not hold a launch up.
PROBE_TIMEOUT_SECONDS = 2.0

# States, weakest first. Returned rather than raised.
ABSENT = "absent"            # no ollama executable on this machine
STOPPED = "stopped"          # installed, but nothing answering
RUNNING = "running"          # answering now
UNKNOWN = "unknown"          # could not be determined; say so rather than guess


def is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def host() -> str:
    """Where to look for the server.

    config/settings.yaml's ollama.host is the authority, and every other module
    reads it through core/config_loader. This cannot: it runs before PyYAML is
    installed. So the environment variable Ollama itself defines is honoured, and
    the default is the same literal the rest of Leti falls back to. A host set
    only in settings.yaml is not seen here, which costs nothing - the check is
    then against localhost, and a wrong answer only means one skipped start
    attempt against a server Leti will reach anyway.
    """
    from_env = (os.environ.get("OLLAMA_HOST") or "").strip()
    if not from_env:
        return DEFAULT_HOST
    if "://" in from_env:
        return from_env.rstrip("/")
    return f"http://{from_env.rstrip('/')}"


def executable() -> Optional[str]:
    """The ollama command, if this machine has one."""
    return shutil.which("ollama")


def responding(at: Optional[str] = None, timeout: float = PROBE_TIMEOUT_SECONDS,
               opener: Optional[Callable] = None) -> bool:
    """Is something answering on Ollama's port?

    A plain GET of the root, which Ollama answers with a one-line banner. Any HTTP
    answer at all counts: the question is whether the server is up, not what it
    thinks of the request. urllib rather than curl because curl is not on every
    Windows 10 build and urllib is in every Python.
    """
    import urllib.error
    import urllib.request

    url = (at or host()).rstrip("/") + "/"
    try:
        open_it = opener or urllib.request.urlopen
        with open_it(url, timeout=timeout) as response:
            getattr(response, "read", lambda *_: b"")(1)
            return True
    except urllib.error.HTTPError:
        # It answered, with a refusal. That still means it is running.
        return True
    except Exception:
        return False


def state(opener: Optional[Callable] = None) -> str:
    """What the situation is, without changing it."""
    if responding(opener=opener):
        return RUNNING
    return STOPPED if executable() else ABSENT


def install(progress: Any, run: Optional[Callable] = None) -> bool:
    """Install Ollama, on a platform where that can be done unattended.

    Windows only, and only through winget, which ships with Windows 10 and 11 and
    installs per-user without an administrator prompt - the same promise the rest
    of the launcher makes. Anywhere else, and on a Windows without winget, this
    says where to get it and returns False. Downloading and running an installer
    ourselves is not something a launcher should do quietly.
    """
    runner = run or subprocess.run
    if not is_windows():
        progress.step("Ollama is not installed. Install it from https://ollama.com "
                      "and start Leti again.")
        return False
    if not shutil.which("winget"):
        progress.step("Ollama is not installed, and winget is not available here to "
                      "install it. Get it from https://ollama.com/download, then "
                      "start Leti again.")
        return False

    progress.step("Installing Ollama, which runs the model Leti thinks with "
                  "(one time, no administrator prompt)...")
    try:
        done = runner(["winget", "install", "--id", "Ollama.Ollama", "-e", "--silent",
                       "--accept-package-agreements", "--accept-source-agreements"],
                      timeout=1800)
    except Exception as e:
        progress.step(f"Ollama could not be installed ({type(e).__name__}). Get it "
                      "from https://ollama.com/download and start Leti again.")
        return False
    if int(getattr(done, "returncode", 1) or 0) != 0:
        progress.step("Ollama could not be installed by winget. Get it from "
                      "https://ollama.com/download and start Leti again.")
        return False

    if not executable():
        # Installed, but this process's PATH was read when it started. The next
        # launch finds it; saying so is better than a start attempt that cannot
        # work and a message about a server instead of about PATH.
        progress.step("Ollama is installed. Start Leti again - this window's PATH "
                      "was read before it existed, so it will be found next time.")
        return False
    progress.step("Ollama installed.")
    return True


def start(progress: Any, run: Optional[Callable] = None,
          opener: Optional[Callable] = None, sleep: Optional[Callable] = None,
          timeout: float = START_TIMEOUT_SECONDS) -> bool:
    """Start the server in the background and wait for it to answer."""
    runner = run or subprocess.run
    wait = sleep or time.sleep
    command = executable()
    if not command:
        return False

    progress.step("Starting Ollama in the background...")
    try:
        _spawn(command, runner)
    except Exception as e:
        progress.step(f"Ollama would not start ({type(e).__name__}). Start it "
                      "yourself with 'ollama serve' and Leti will find it.")
        return False

    deadline = timeout
    spent = 0.0
    while spent < deadline:
        if responding(opener=opener):
            progress.step("Ollama is answering.")
            return True
        wait(POLL_SECONDS)
        spent += POLL_SECONDS
    progress.step(f"Ollama was started but is not answering at {host()} after "
                  f"{int(deadline)} seconds. Leti will open and say so; starting it "
                  "yourself with 'ollama serve' is the fix if this persists.")
    return False


def _spawn(command: str, runner: Callable) -> Any:
    """`ollama serve`, detached, with no console window of its own.

    Popen rather than run: this is a server, and waiting for it to finish would
    wait forever. The Windows flags are what keep a second console from appearing
    behind Leti's window and what let it outlive this process, which is what the
    .bat's Start-Process did.
    """
    spawn = getattr(runner, "popen", None) or subprocess.Popen
    kwargs: Dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if is_windows():
        creation = 0
        for name in ("CREATE_NO_WINDOW", "DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
            creation |= int(getattr(subprocess, name, 0) or 0)
        if creation:
            kwargs["creationflags"] = creation
    else:
        kwargs["start_new_session"] = True
    return spawn([command, "serve"], **kwargs)


def ensure(progress: Any, run: Optional[Callable] = None,
           opener: Optional[Callable] = None, sleep: Optional[Callable] = None,
           installer: Optional[Callable] = None, may_install: bool = True,
           timeout: float = START_TIMEOUT_SECONDS) -> str:
    """Make sure there is a model server, and say what happened.

    Returns one of RUNNING, STOPPED, ABSENT. Never raises: every path reports
    through `progress` and returns, because a launch is not worth refusing over
    this (see the module docstring).

    `may_install` and `timeout` are how the caller keeps this off the critical path
    of a launch that is supposed to be half a second. This function runs on EVERY
    launch, warm ones included, so a machine that cannot install Ollama would
    otherwise re-run winget every time Leti was opened, and one where the server
    will not start would wait the full timeout every time. launcher/bootstrap.py
    remembers the last failure and decides both; here they are just parameters, so
    this module stays a description of the step rather than of the policy.
    """
    try:
        if responding(opener=opener):
            return RUNNING

        if not executable():
            if not may_install:
                progress.step("Ollama is still not installed. Leti will open and say "
                              "it cannot reach a model; installing Ollama from "
                              "https://ollama.com is what fixes it.")
                return ABSENT
            if not (installer or install)(progress, run):
                return ABSENT

        if start(progress, run, opener, sleep, timeout=timeout):
            return RUNNING
        return STOPPED
    except Exception as e:
        # Belt and braces. Whatever went wrong here, it is not a reason for Leti
        # not to open.
        try:
            progress.step(f"Could not check on Ollama ({type(e).__name__}). Leti "
                          "will open and say whether it can reach a model.")
        except Exception:
            pass
        return UNKNOWN


def describe(verdict: str) -> Sequence[str]:
    """What to tell the user about a verdict that is not RUNNING."""
    if verdict == RUNNING:
        return ()
    # Deliberately not "Leti will still open and say so": it will not. main.py
    # exits when it cannot reach a model server, which is why the launcher brings
    # the console back rather than letting the launch appear to do nothing.
    if verdict == ABSENT:
        return ("Leti needs Ollama to think with, and it is not on this machine yet.",
                "Install it from https://ollama.com and start Leti again.")
    if verdict == STOPPED:
        return ("Ollama is installed but not answering yet.",
                "Leti cannot start without it - 'ollama serve' in a terminal is the "
                "quickest way to check why.")
    return ("Whether Ollama is running could not be determined.",)
