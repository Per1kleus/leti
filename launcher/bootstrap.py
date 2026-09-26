"""What has to be true before Leti can start, and how to make it true.

Leti is a Python application with twenty-seven dependencies, a browser binary and
a model server. A developer prepares all of that with commands they already know.
Somebody who downloaded Leti to talk to it does not, and should not have to: no
python.org, no pip, no PowerShell, no PATH, no "activate".

So this module is the one place that knows how to get from "a folder of files" to
"Leti is running". Both Windows front doors go through it - Leti.exe and the
double-click launcher - and both end at the same main.py a developer runs.

WHAT IT DOES, IN ORDER

    an interpreter  ->  a place for packages  ->  the packages  ->  Leti

Each step is CHECKED before it is done. That is the difference between a launcher
somebody uses twice and one they use every day: the first launch downloads a few
hundred megabytes, and the second must notice that and do nothing. Every check is
a file that either exists or does not, or a version that either satisfies
requirements.txt or does not - no timestamps, no trust, nothing that drifts.

AN INTERPRETER

A system Python is used when there is a suitable one, because a venv built from it
is the most ordinary thing on the machine and the least that can go wrong.
Otherwise a runtime is fetched into the project folder. Nothing is installed
system-wide either way: no admin prompt, no PATH edit, and the user's own Python
- if they have one - is never written to.

The fetched runtime is the "embeddable package", which is a zip rather than an
installer for exactly this purpose. It cannot create a venv (it ships no `venv`
and no `ensurepip`), so packages go into it directly once pip is bootstrapped -
which is fine, because the folder IS the isolated environment. Nothing else uses
it and deleting it undoes everything.

RESUMING

Setup can be closed halfway. Every step records that it FINISHED, never that it
started, so an interrupted run leaves the same state as a run that never happened
and the next launch picks up from the last thing that actually completed. A
half-extracted runtime is detected by its interpreter being absent and replaced,
not patched.

WHAT IT DOES NOT DO

It does not install Ollama or pull models - core/model_setup.py already owns the
model question and asks it on first launch, inside Leti, where it can see the
hardware. It does not configure anything. It does not run in the background, hold
a thread, or stay resident: it returns, and then Leti starts.

Standard library only, deliberately. This is the code that runs when nothing is
installed, so it cannot depend on anything being installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# The oldest Python Leti runs on. Below this a system interpreter is passed over
# and a runtime is fetched instead, rather than failing somewhere later with an
# error about syntax.
MINIMUM_PYTHON = (3, 11)

# What is fetched when the machine has no usable Python. A version rather than
# "latest" on purpose: a launcher that silently moves to a new Python whenever
# python.org publishes one is a launcher that breaks on a Tuesday.
EMBED_VERSION = "3.11.9"
EMBED_URL = ("https://www.python.org/ftp/python/{v}/python-{v}-embed-amd64.zip")
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# The SHA-256 of that zip, when it is known. HTTPS is what makes the download
# trustworthy; this is the second lock, and it is only useful if the value was
# verified by a person against python.org's own published checksum. Empty means
# unverified and therefore unchecked - a hash invented here would be worse than
# no hash, because it would look like a check while proving only that the file
# matched whatever was typed. To fill it in:
#
#   curl -sL https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip \
#     | sha256sum
#
# and compare with the checksum on the python.org download page before pasting.
EMBED_SHA256 = ""

# Where things live, relative to the project folder. All three are local to it:
# uninstalling Leti is deleting the folder.
VENV_DIR = "leti_env"            # a venv built from a system Python
RUNTIME_DIR = "leti_runtime"     # a Python fetched because there was none
STATE_FILE = "data/launch_setup.json"
STATE_VERSION = 1

# There is deliberately no table of package names here. An earlier draft had one -
# distribution name to import name, "openai-whisper" to "whisper" - and it was
# both unused and exactly the thing this module must not contain: a second copy of
# the dependency list, to be forgotten when requirements.txt changes.
#
# It is unnecessary because nothing here imports anything. What is installed is
# asked of importlib.metadata, which answers by DISTRIBUTION name - the same name
# pip uses and the same name requirements.txt is written in. So the file is the
# only list, and the launcher reads it.


# --------------------------------------------------------------------------- #
# Reading requirements.txt
#
# The one source of truth for what Leti needs. Parsed rather than duplicated: a
# second list here would be a second list to forget to update.
# --------------------------------------------------------------------------- #

_REQUIREMENT = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9._-]+)\s*(?P<spec>(?:[<>=!~]=?[^,;#\s]+\s*,?\s*)*)")


def requirements(path: Path) -> List[Tuple[str, str]]:
    """Every requirement as (name, specifier), in file order.

    Comments, blank lines and options are skipped. An unparseable line is
    skipped too, with its text kept nowhere - a launcher that stops because
    requirements.txt grew a syntax it does not know is worse than one that
    installs the rest and lets pip complain about the remainder.
    """
    out: List[Tuple[str, str]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        match = _REQUIREMENT.match(stripped)
        if not match:
            continue
        out.append((match.group("name"), (match.group("spec") or "").strip().rstrip(",")))
    return out


def _version_parts(version: str) -> Tuple[int, ...]:
    """A version as numbers, for comparing. Non-numeric tails are dropped."""
    parts: List[int] = []
    for piece in re.split(r"[._-]", str(version).strip()):
        digits = re.match(r"\d+", piece)
        if not digits:
            break
        parts.append(int(digits.group(0)))
    return tuple(parts) or (0,)


def _at_least(installed: str, wanted: str) -> bool:
    left, right = _version_parts(installed), _version_parts(wanted)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def satisfies(installed: Optional[str], spec: str) -> bool:
    """Does `installed` meet `spec`?

    Absent is never satisfied - that is the case this whole module exists for.
    Present with a specifier nobody here can read IS satisfied: pip resolved it
    once and pip is the authority, so the launcher's job is to notice what is
    MISSING rather than to re-adjudicate what is there.
    """
    if installed is None:
        return False
    if not spec:
        return True
    for clause in (c.strip() for c in spec.split(",") if c.strip()):
        if clause.startswith(">="):
            if not _at_least(installed, clause[2:]):
                return False
        elif clause.startswith("=="):
            wanted = clause[2:].rstrip("*").rstrip(".")
            if not str(installed).startswith(wanted):
                return False
        elif clause.startswith(">"):
            if _version_parts(installed) <= _version_parts(clause[1:]):
                return False
        # <, <=, !=, ~= are left to pip. See the docstring.
    return True


def missing(wanted: Sequence[Tuple[str, str]],
            installed: Callable[[str], Optional[str]]) -> List[str]:
    """The names that need installing. `installed` returns a version or None.

    A list rather than a set so the order is requirements.txt's order, which is
    the order a person reading the progress expects.
    """
    return [name for name, spec in wanted if not satisfies(installed(name), spec)]


def survey(python: Path, run: Optional[Callable] = None) -> Dict[str, Dict[str, str]]:
    """Every distribution that interpreter can see: {name: {version, path}}.

    Asked of the interpreter itself rather than guessed from a folder listing,
    because the interpreter is what will import them. Names are lowercased with
    underscores folded to hyphens, which is how pip compares them.

    The path is each distribution's metadata folder. It is recorded after a
    successful install and stat'ed on later launches, which is what lets a quick
    check notice that a package has been uninstalled or deleted without paying
    for this whole query every time.
    """
    runner = run or subprocess.run
    script = (
        "import json\n"
        "try:\n"
        "    from importlib.metadata import distributions\n"
        "except Exception:\n"
        "    print('{}')\n"
        "else:\n"
        "    out = {}\n"
        "    for d in distributions():\n"
        "        try:\n"
        "            name = (d.metadata['Name'] or '').strip()\n"
        "        except Exception:\n"
        "            continue\n"
        "        if not name:\n"
        "            continue\n"
        "        where = ''\n"
        "        try:\n"
        "            where = str(getattr(d, '_path', '') or '')\n"
        "        except Exception:\n"
        "            where = ''\n"
        "        out[name.lower().replace('_', '-')] = {'version': d.version,\n"
        "                                              'path': where}\n"
        "    print(json.dumps(out))\n")
    try:
        done = runner([str(python), "-c", script], capture_output=True, text=True,
                      timeout=120)
        if done.returncode != 0:
            return {}
        found = json.loads((done.stdout or "{}").strip().splitlines()[-1])
        return found if isinstance(found, dict) else {}
    except Exception:
        return {}


def installed_in(python: Path, run: Optional[Callable] = None) -> Dict[str, str]:
    """Every distribution that interpreter can see, as {name: version}."""
    return {name: str(row.get("version", ""))
            for name, row in survey(python, run).items() if isinstance(row, dict)}


def witnesses_for(wanted: Sequence[Tuple[str, str]],
                  found: Dict[str, Dict[str, str]]) -> List[str]:
    """The metadata folder of each required package, for the quick check.

    One path per requirement that has one. A requirement whose location could not
    be read simply has no witness: the quick check then cannot vouch for it, and
    the full check is what finds out - which is the safe direction.
    """
    out: List[str] = []
    for name, _ in wanted:
        row = found.get(str(name).lower().replace("_", "-"))
        if isinstance(row, dict) and row.get("path"):
            out.append(str(row["path"]))
    return out


def all_present(witnesses: Sequence[str]) -> bool:
    """Are all the recorded metadata folders still there?

    This is the whole quick check: a few stat calls, and it is filesystem truth
    rather than a record being trusted. `pip uninstall` removes one of these, so
    a package taken away is noticed on the very next launch and put back.

    No witnesses at all reads as false - an empty list would otherwise vouch for
    everything, which is exactly the wrong way round.
    """
    if not witnesses:
        return False
    return all(Path(w).exists() for w in witnesses)


def installed_in_from(found: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Versions out of a survey already taken, so it is not taken twice."""
    return {name: str(row.get("version", ""))
            for name, row in found.items() if isinstance(row, dict)}


def lookup_from(found: Dict[str, str]) -> Callable[[str], Optional[str]]:
    """A name-tolerant lookup over what installed_in returned."""
    def look(name: str) -> Optional[str]:
        return found.get(str(name).lower().replace("_", "-"))
    return look


# --------------------------------------------------------------------------- #
# Interpreters
# --------------------------------------------------------------------------- #

def is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def python_in(directory: Path) -> Path:
    """Where an interpreter lives inside a venv or an extracted runtime."""
    directory = Path(directory)
    if is_windows():
        for candidate in (directory / "Scripts" / "python.exe", directory / "python.exe"):
            if candidate.exists():
                return candidate
        return directory / "Scripts" / "python.exe"
    for candidate in (directory / "bin" / "python", directory / "bin" / "python3"):
        if candidate.exists():
            return candidate
    return directory / "bin" / "python"


def prepared_python(root: Path) -> Optional[Path]:
    """The interpreter a previous run left ready, or None.

    The venv first: when both exist it is because a system Python appeared after
    a runtime was fetched, and the venv is the better one to be using.
    """
    for directory in (Path(root) / VENV_DIR, Path(root) / RUNTIME_DIR):
        candidate = python_in(directory)
        if candidate.exists():
            return candidate
    return None


def version_of(python: Path, run: Optional[Callable] = None) -> Optional[Tuple[int, ...]]:
    """An interpreter's version, or None if it cannot be asked."""
    runner = run or subprocess.run
    try:
        done = runner([str(python), "-c",
                       "import sys; print('%d.%d.%d' % sys.version_info[:3])"],
                      capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            return None
        return _version_parts((done.stdout or "").strip())
    except Exception:
        return None


def can_build_a_venv(python: Path, run: Optional[Callable] = None) -> bool:
    """Is this interpreter new enough, and does it actually carry venv?

    Both are asked, because a Linux distribution that splits python3-venv into
    its own package produces an interpreter that is new enough and cannot make
    one - and finding that out here is a message, while finding out later is a
    traceback.
    """
    version = version_of(python, run)
    if version is None or version < MINIMUM_PYTHON:
        return False
    runner = run or subprocess.run
    try:
        done = runner([str(python), "-c", "import venv, ensurepip"],
                      capture_output=True, text=True, timeout=60)
        return done.returncode == 0
    except Exception:
        return False


def system_pythons() -> List[Path]:
    """Interpreters worth asking about, best guess first.

    The one running this code comes first when it is a real interpreter rather
    than a frozen executable: inside Leti.exe, sys.executable is Leti.exe, which
    can no more create a venv than a text file can.
    """
    out: List[Path] = []
    if not getattr(sys, "frozen", False):
        out.append(Path(sys.executable))
    for name in ("py", "python3", "python"):
        found = shutil.which(name)
        if found:
            out.append(Path(found))
    if is_windows():
        for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("PROGRAMFILES", "")):
            if not base:
                continue
            for minor in range(13, 10, -1):
                out.append(Path(base) / "Programs" / "Python" / f"Python3{minor}" / "python.exe")
    seen, unique = set(), []
    for candidate in out:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def usable_system_python(run: Optional[Callable] = None,
                         candidates: Optional[Sequence[Path]] = None) -> Optional[Path]:
    """The first system interpreter that can build Leti a venv, or None."""
    for candidate in (candidates if candidates is not None else system_pythons()):
        try:
            if Path(candidate).exists() and can_build_a_venv(Path(candidate), run):
                return Path(candidate)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Setup state
#
# Small, and only ever a record of what FINISHED. Nothing here is trusted over
# the filesystem: the interpreter is checked for by looking for it, and the
# packages by asking the interpreter. The file exists so that the expensive
# checks - which are the ones that shell out - can be skipped when the thing
# they would confirm has not changed.
# --------------------------------------------------------------------------- #

def state_path(root: Path) -> Path:
    return Path(root) / STATE_FILE


def read_state(root: Path) -> Dict[str, Any]:
    try:
        data = json.loads(state_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        return {}
    return data


def write_state(root: Path, **fields: Any) -> Dict[str, Any]:
    """Merge `fields` into the recorded state and save it atomically.

    Atomic because the alternative is a half-written JSON file after a power cut,
    which reads as corrupt, which reads as "nothing is set up" - and that would
    mean downloading everything again. Reuses core/atomic_write.py rather than
    doing it here; it is stdlib-only, so it is available this early.
    """
    state = read_state(root)
    state.update(fields)
    state["version"] = STATE_VERSION
    path = state_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from core.atomic_write import atomic_write_text

        atomic_write_text(path, json.dumps(state, indent=2))
    except Exception:
        # Losing the record costs a slower next launch, never correctness: every
        # check below still works from the filesystem alone.
        pass
    return state


def requirements_fingerprint(path: Path) -> str:
    """A hash of requirements.txt, so an unchanged file can be skipped."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# Progress
#
# Plain lines, because this is read once by somebody waiting for an application
# to open. No progress bars to redraw, no spinner to keep a thread alive, and no
# raw commands - pip's own output is shown when it fails, because that is when it
# is the useful thing to read, and hidden when it works.
# --------------------------------------------------------------------------- #

class Progress:
    """Says what is happening. Replaceable, which is how the tests read it."""

    def __init__(self, out: Any = None) -> None:
        self.out = out if out is not None else sys.stdout
        self.lines: List[str] = []

    def say(self, text: str = "") -> None:
        self.lines.append(text)
        try:
            print(text, file=self.out, flush=True)
        except Exception:
            pass

    def step(self, text: str) -> None:
        self.say(f"  {text}")

    def done(self, text: str = "") -> None:
        self.say(f"  {text} OK" if text else "  OK")

    def problem(self, what: str, trying: str, retry_is_safe: bool,
                logs: Optional[Path] = None) -> None:
        """What failed, what it was doing, whether to try again, where to look.

        Never a traceback and never a command line. Nothing from the environment
        is echoed, so a token in an environment variable or a proxy URL with a
        password in it cannot be printed by this.
        """
        self.say()
        self.say("Leti could not finish setting itself up.")
        self.say(f"  What failed: {what}")
        self.say(f"  It was trying to: {trying}")
        self.say("  Trying again is safe - setup carries on from the last step "
                 "that finished."
                 if retry_is_safe else
                 "  Trying again will not help on its own; the note above has to "
                 "be dealt with first.")
        if logs is not None:
            self.say(f"  Details: {logs}")
        self.say()


# --------------------------------------------------------------------------- #
# Doing the work
# --------------------------------------------------------------------------- #

class SetupFailed(Exception):
    """Something could not be prepared. Carries what to tell the user."""

    def __init__(self, what: str, trying: str, retry_is_safe: bool = True) -> None:
        super().__init__(what)
        self.what = what
        self.trying = trying
        self.retry_is_safe = retry_is_safe


def _run(command: Sequence[str], run: Callable, **kwargs: Any):
    return run([str(c) for c in command], **kwargs)


def build_venv(root: Path, system_python: Path, progress: Progress,
               run: Optional[Callable] = None) -> Path:
    """Make leti_env from a system Python. Returns its interpreter."""
    runner = run or subprocess.run
    target = Path(root) / VENV_DIR
    progress.step("Preparing Leti's own Python environment...")
    if target.exists() and not python_in(target).exists():
        # A run that was interrupted during creation. Replaced rather than
        # repaired: a half-made venv has no supported way to be finished.
        shutil.rmtree(target, ignore_errors=True)
    try:
        done = _run([system_python, "-m", "venv", str(target)], runner,
                    capture_output=True, text=True, timeout=600)
    except Exception as e:
        raise SetupFailed(f"the environment could not be created ({type(e).__name__})",
                          "create a private Python environment in the Leti folder")
    if done.returncode != 0 or not python_in(target).exists():
        raise SetupFailed("the environment could not be created",
                          "create a private Python environment in the Leti folder")
    progress.done()
    return python_in(target)


def fetch_runtime(root: Path, progress: Progress,
                  fetch: Optional[Callable] = None,
                  run: Optional[Callable] = None) -> Path:
    """Download a Python into the Leti folder and give it pip. Returns it.

    For the machine with no Python at all. Nothing is installed system-wide, no
    administrator prompt appears, and PATH is not touched - the interpreter lives
    in leti_runtime\\ and only Leti ever calls it.
    """
    runner = run or subprocess.run
    download = fetch or _download
    target = Path(root) / RUNTIME_DIR
    interpreter = python_in(target)
    if interpreter.exists():
        # Already here - possibly extracted by the .bat launcher, which fetches a
        # Python when there is none but does not know about ._pth files. Opening
        # it up is idempotent, so it is done on this path too rather than only
        # after a fresh extraction. Without it pip would install into a folder
        # the interpreter cannot see.
        _open_up_path_file(target)
        return interpreter

    progress.step(f"Getting Python {EMBED_VERSION} (about 11 MB)...")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)       # a half-extracted attempt
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "python-embed.zip"
    url = EMBED_URL.format(v=EMBED_VERSION)
    try:
        download(url, archive)
    except Exception as e:
        raise SetupFailed(
            f"Python could not be downloaded ({type(e).__name__})",
            "fetch a private copy of Python, because this machine has none Leti can use")
    if EMBED_SHA256:
        got = hashlib.sha256(archive.read_bytes()).hexdigest()
        if got != EMBED_SHA256:
            archive.unlink(missing_ok=True)
            raise SetupFailed("the downloaded Python did not match its checksum",
                              "check that the download was not corrupted or tampered with",
                              retry_is_safe=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
    except Exception as e:
        raise SetupFailed(f"the downloaded Python could not be unpacked "
                          f"({type(e).__name__})", "unpack Python into the Leti folder")
    archive.unlink(missing_ok=True)

    # Resolved AGAIN, after extraction. Before it, python_in had nothing to find
    # and returned its fallback - Scripts\python.exe, which is where a venv keeps
    # its interpreter and not where the embeddable package keeps one. Checking the
    # stale answer failed every time, on the one path a machine with no Python
    # takes.
    interpreter = python_in(target)
    if not interpreter.exists():
        raise SetupFailed("the downloaded Python is not where it was expected",
                          "unpack Python into the Leti folder")
    _open_up_path_file(target)
    progress.done()

    _bootstrap_pip(interpreter, progress, run=runner, fetch=download)
    return interpreter


def _open_up_path_file(runtime: Path) -> None:
    """Let the embeddable Python see installed packages and the project.

    It ships with a `._pth` file that pins sys.path to itself and switches off
    site-packages, which is right for embedding inside another program and wrong
    for being one. Three lines are added: site-packages, so pip's work is
    visible; the project folder, so `import core` works; and `import site`,
    without which the first two are not read.
    """
    for path_file in sorted(runtime.glob("python*._pth")):
        try:
            lines = path_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        wanted = ["Lib\\site-packages", ".."]
        kept = [line for line in lines if line.strip() != "#import site"]
        for entry in wanted:
            if entry not in [line.strip() for line in kept]:
                kept.append(entry)
        if "import site" not in [line.strip() for line in kept]:
            kept.append("import site")
        try:
            path_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
        except OSError:
            continue


def _download(url: str, destination: Path) -> None:
    """One file, over HTTPS, to a temporary name and then into place.

    Written beside its destination and renamed, so an interrupted download is
    never mistaken for a finished one.

    HTTPS is checked rather than assumed, before the request and after it. This
    said "over HTTPS" and did not enforce it: urllib follows redirects and its
    redirect handler allows https -> http, so a redirect could have delivered the
    Python interpreter this is about to run over plaintext. That matters more here
    than it would elsewhere, because EMBED_SHA256 is deliberately empty - HTTPS is
    the whole of the trust, so it has to actually hold. The scheme check before the
    request is for a future caller: nothing but the two constants above is passed
    in today, and a bug that passed a file:// path should not be a way to run
    arbitrary code either.
    """
    import urllib.request

    if not str(url).lower().startswith("https://"):
        raise ValueError(f"refusing to download over anything but HTTPS: {url!r}")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response:      # noqa: S310
        landed = str(getattr(response, "url", "") or url)
        if not landed.lower().startswith("https://"):
            raise ValueError(
                f"{url} redirected to {landed}, which is not HTTPS - refusing it "
                f"rather than trusting a plaintext download")
        with open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    os.replace(partial, destination)


def install_packages(python: Path, root: Path, names: Sequence[str],
                     progress: Progress, run: Optional[Callable] = None) -> None:
    """Install exactly `names`, resolved against requirements.txt.

    The names come from the file and the CONSTRAINTS come from the file too, so
    a package installed here gets the same version a full install would give it.
    Nothing outside requirements.txt is ever asked for.
    """
    if not names:
        return
    runner = run or subprocess.run
    requirements_file = Path(root) / "requirements.txt"
    shown = ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
    progress.step(f"Installing what is missing: {shown}")
    progress.step("The first time, this downloads a few hundred megabytes and "
                  "takes a while.")
    command = [python, "-m", "pip", "install", "--disable-pip-version-check",
               "-c", str(requirements_file), *names]
    try:
        done = _run(command, runner, capture_output=True, text=True, timeout=5400)
    except Exception as e:
        raise SetupFailed(f"the packages could not be installed ({type(e).__name__})",
                          "download and install the packages listed in requirements.txt")
    if done.returncode != 0:
        _write_log(root, "pip", done)
        raise SetupFailed("one or more packages would not install",
                          "download and install the packages listed in requirements.txt")
    progress.done()


def ensure_pip(python: Path, progress: Progress,
               run: Optional[Callable] = None,
               fetch: Optional[Callable] = None) -> None:
    """Make sure the interpreter has pip, however it came to be here.

    Three ways, in order of how little they cost. It already has pip, which is
    every venv and every second launch. It has ensurepip, which is every ordinary
    installation. Or it has neither - which is the embeddable package, and is not
    a corner case: the .bat launcher unpacks one of those itself when the machine
    has no Python, and then this is the only thing that can give it pip.

    An earlier version stopped at ensurepip. The embeddable zip ships no Lib\
    tree, so there is no ensurepip in it, and every install after that failed on
    exactly the machine this was all written for.
    """
    runner = run or subprocess.run
    try:
        done = _run([python, "-m", "pip", "--version"], runner,
                    capture_output=True, text=True, timeout=120)
        if done.returncode == 0:
            return
    except Exception:
        pass
    try:
        done = _run([python, "-m", "ensurepip", "--upgrade"], runner,
                    capture_output=True, text=True, timeout=600)
        if getattr(done, "returncode", 1) == 0:
            return
    except Exception:
        pass
    _bootstrap_pip(python, progress, run=runner, fetch=fetch)


def _bootstrap_pip(python: Path, progress: Progress,
                   run: Optional[Callable] = None,
                   fetch: Optional[Callable] = None) -> None:
    """Fetch get-pip.py and run it. The last resort, and the embeddable one."""
    runner = run or subprocess.run
    download = fetch or _download
    progress.step("Giving it a package installer...")
    get_pip = Path(python).parent / "get-pip.py"
    try:
        download(GET_PIP_URL, get_pip)
        done = _run([python, str(get_pip), "--no-warn-script-location"], runner,
                    capture_output=True, text=True, timeout=900)
    except Exception as e:
        raise SetupFailed(f"pip could not be installed ({type(e).__name__})",
                          "add a package installer to Leti's private Python")
    finally:
        try:
            get_pip.unlink(missing_ok=True)
        except OSError:
            pass
    if getattr(done, "returncode", 1) != 0:
        raise SetupFailed("pip could not be installed",
                          "add a package installer to Leti's private Python")
    progress.done()


def _write_log(root: Path, name: str, done: Any) -> Path:
    """Keep the failed command's own output where the message can point at it.

    The command itself is not written, only what it printed: an argument list can
    carry a proxy url with a password in it, and a log is a file people send to
    other people.
    """
    path = Path(root) / "logs" / f"setup_{name}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (getattr(done, "stdout", "") or "") + (getattr(done, "stderr", "") or "")
        path.write_text(text[-40000:], encoding="utf-8", errors="replace")
    except Exception:
        pass
    return path


def install_browser(python: Path, root: Path, progress: Progress,
                    run: Optional[Callable] = None) -> bool:
    """Playwright's browser binary, which pip does not bring with the package.

    Not fatal. Everything except web browsing works without it, so a failure
    here says so and setup continues - the alternative is refusing to start an
    assistant because one of its tools is unavailable.
    """
    runner = run or subprocess.run
    marker = Path(root) / VENV_DIR / ".playwright_installed"
    alternative = Path(root) / RUNTIME_DIR / ".playwright_installed"
    if marker.exists() or alternative.exists():
        return True
    progress.step("Getting the browser Leti uses for web automation...")
    try:
        done = _run([python, "-m", "playwright", "install", "chromium"], runner,
                    capture_output=True, text=True, timeout=3600)
    except Exception:
        progress.step("The browser could not be fetched - everything except web "
                      "browsing will work.")
        return False
    if done.returncode != 0:
        progress.step("The browser could not be fetched - everything except web "
                      "browsing will work.")
        return False
    target = marker if python_in(Path(root) / VENV_DIR).exists() else alternative
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    except OSError:
        pass
    progress.done()
    return True


# --------------------------------------------------------------------------- #
# The whole thing
# --------------------------------------------------------------------------- #

def prepare(root: Path, progress: Optional[Progress] = None,
            run: Optional[Callable] = None,
            fetch: Optional[Callable] = None,
            skip_browser: bool = False) -> Path:
    """Get this folder ready to run Leti, and return the interpreter to use.

    Safe to call every launch. When everything is already in place it asks the
    interpreter what it has, finds nothing missing, and returns - which is one
    subprocess, and is skipped entirely when requirements.txt has not changed
    since the last time that check passed.

    Raises SetupFailed with something a person can act on. Never leaves the
    folder in a state a second call cannot carry on from.
    """
    root = Path(root)
    progress = progress or Progress()
    runner = run or subprocess.run
    state = read_state(root)
    fingerprint = requirements_fingerprint(root / "requirements.txt")
    if not fingerprint:
        raise SetupFailed("requirements.txt could not be read",
                          "find out which packages Leti needs",
                          retry_is_safe=False)

    progress.say("Preparing Leti...")
    progress.step("Checking Python...")
    python = prepared_python(root)
    if python is not None and (Path(root) / RUNTIME_DIR) in python.parents:
        _open_up_path_file(Path(root) / RUNTIME_DIR)
    if python is None:
        system = usable_system_python(runner)
        if system is not None:
            progress.done(f"found Python {'.'.join(str(n) for n in (version_of(system, runner) or ()))}")
            python = build_venv(root, system, progress, runner)
        else:
            progress.say("  No Python on this machine that Leti can use - "
                         "fetching its own.")
            python = fetch_runtime(root, progress, fetch, runner)
    else:
        progress.done()
    ensure_pip(python, progress, runner, fetch)
    write_state(root, python=str(python))

    # The quick check. Everything below shells out - about a second and a half of
    # it - so it is worth not doing when nothing has changed. What counts as
    # nothing changed is deliberately strict: requirements.txt is the same file it
    # was, the last run finished, AND every package's metadata folder is still on
    # disk. That last clause is what makes a removed dependency repair itself
    # rather than being skipped over by a record that says all is well.
    witnesses = state.get("witnesses") or []
    if (state.get("requirements") == fingerprint and state.get("complete")
            and all_present(witnesses)):
        progress.step("Everything is already installed.")
        # Not on every launch: reading a .lnk costs two PowerShell processes, and
        # a normal launch should be the half-second it is. Done once, for a folder
        # set up before shortcuts existed, and then recorded - after which the
        # explicit repair (--install-shortcuts) is what puts a deleted one back.
        if not state.get("shortcuts"):
            place_shortcuts(root, progress)
        progress.say("Starting Leti...")
        return python

    progress.step("Checking the packages Leti needs...")
    wanted = requirements(root / "requirements.txt")
    if not wanted:
        raise SetupFailed("requirements.txt lists no packages",
                          "find out which packages Leti needs", retry_is_safe=False)
    found = survey(python, runner)
    absent = missing(wanted, lookup_from(installed_in_from(found)))
    if absent:
        install_packages(python, root, absent, progress, runner)
        found = survey(python, runner)
        still = missing(wanted, lookup_from(installed_in_from(found)))
        if still:
            _remember_partial(root, fingerprint, still)
            raise SetupFailed(
                f"{len(still)} package(s) are still not installed after trying",
                "install every package listed in requirements.txt")
    else:
        progress.done("nothing missing -")

    if not skip_browser:
        install_browser(python, root, progress, runner)

    # Written last, and only once everything above actually finished. An
    # interrupted run never gets here, so the next launch checks properly rather
    # than trusting a record of a setup that did not complete.
    write_state(root, requirements=fingerprint, complete=True, partial=None,
                witnesses=witnesses_for(wanted, found))
    place_shortcuts(root, progress)
    progress.say("Starting Leti...")
    return python


def place_shortcuts(root: Path, progress: Progress,
                    install: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """Put Leti on the Desktop and in the Start Menu, if this is Windows.

    Never fatal, and never the reason a launch does not happen: a missing shortcut
    is an inconvenience and a refusal to start is not. What was done is said only
    when something actually changed, because "your shortcut is still fine" is not
    news.
    """
    if not is_windows():
        return []
    try:
        from launcher import shortcuts

        placed = (install or shortcuts.install)(root)
        for line in shortcuts.describe(placed):
            progress.step(line)
        if any(row.get("action") in ("created", "repaired", "kept") for row in placed):
            write_state(root, shortcuts=True)
        return placed
    except Exception as e:
        # Said rather than swallowed - this module has no logger on purpose (it
        # runs before anything is configured) and Progress is where it speaks. A
        # shortcut is a convenience, so this is a note and not a failure.
        progress.step(f"Could not add Leti to your Desktop ({type(e).__name__}) - "
                      "everything else is set up.")
        return []


def _remember_partial(root: Path, fingerprint: str, still_missing: Sequence[str]) -> None:
    write_state(root, requirements=None, complete=False,
                partial={"requirements": fingerprint,
                         "still_missing": list(still_missing)})


def leti_command(python: Path, root: Path, argv: Optional[Sequence[str]] = None) -> List[str]:
    """How Leti is started: the one entry point, with the one interpreter.

    Both front doors call this. There is no second Leti to start - main.py is
    what a developer runs from the repository, and it is what this runs.
    """
    arguments = list(argv) if argv else ["--mode", "gui"]
    return [str(python), str(Path(root) / "main.py"), *arguments]


def start_leti(python: Path, root: Path, argv: Optional[Sequence[str]] = None,
               run: Optional[Callable] = None) -> int:
    """Run Leti and wait for it. Returns its exit code.

    The working directory is the project folder, because Leti resolves its own
    configuration and data relative to it. PYTHONPATH carries the project too, so
    `import core` works from the fetched runtime as well as from a venv.
    """
    runner = run or subprocess.run
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (f"{root}{os.pathsep}{existing}" if existing else str(root))
    done = runner(leti_command(python, root, argv), cwd=str(root), env=environment)
    return int(getattr(done, "returncode", 0) or 0)


def describe() -> Dict[str, Any]:
    """What this machine looks like to the launcher. For diagnostics and tests."""
    return {
        "platform": platform.system(),
        "windows": is_windows(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "minimum_python": ".".join(str(n) for n in MINIMUM_PYTHON),
        "fetches_python": EMBED_VERSION,
        "checksum_pinned": bool(EMBED_SHA256),
    }
