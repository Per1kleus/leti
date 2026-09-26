"""Both front doors make sure there is a model server, and one piece of code does it.

This is the bug these tests exist for: setting Ollama up lived inside
"Launch Leti (Windows).bat" and nothing under launcher/ mentioned it at all, so
double-clicking Leti.exe on a clean Windows machine produced a Leti that opened,
could not reach a model, and said so in a log file behind a console the launcher
had just hidden. Two front doors documented as equivalent, one of which did not
work on the machine it was written for.

Nothing here runs on Windows - see the release report. What is checked is the
logic, with the runner, the HTTP probe and the sleep all injected, which is how the
rest of the launcher is tested.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from launcher import bootstrap, ollama_setup  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def code_of(module) -> str:
    """A module's source with its docstrings removed.

    These tests assert that certain words do not appear in the code - "curl",
    "SetupFailed" - and the code explains at length why they do not. Checking the
    raw source would make the explanation fail its own test.
    """
    import ast
    import inspect

    source = inspect.getsource(module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            text = ast.get_docstring(node, clean=False)
            if text:
                source = source.replace(text, "")
    return source


class Progress:
    """bootstrap.Progress's shape, without printing."""

    def __init__(self):
        self.lines = []

    def say(self, text=""):
        self.lines.append(text)

    def step(self, text):
        self.lines.append(f"  {text}")

    def done(self, text=""):
        self.lines.append("  OK")

    @property
    def text(self):
        return "\n".join(self.lines)


class Done:
    def __init__(self, returncode=0):
        self.returncode = returncode


class Runner:
    """subprocess.run, recorded. `popen` stands in for the detached server."""

    def __init__(self, returncode=0, popen_raises=None):
        self.commands = []
        self.spawned = []
        self.returncode = returncode
        self.popen_raises = popen_raises

    def __call__(self, command, **kwargs):
        self.commands.append([str(c) for c in command])
        return Done(self.returncode)

    def popen(self, command, **kwargs):
        if self.popen_raises:
            raise self.popen_raises
        self.spawned.append(([str(c) for c in command], kwargs))
        return object()

    def ran(self, needle):
        return [c for c in self.commands if needle in " ".join(c)]


def answering(times=None):
    """An opener that reports the server up, or up only after N calls."""
    state = {"calls": 0}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self, *_): return b"Ollama is running"

    def open_it(url, timeout=None):
        state["calls"] += 1
        if times is not None and state["calls"] < times:
            raise OSError("connection refused")
        return Response()

    open_it.calls = lambda: state["calls"]
    return open_it


def never_answering(url, timeout=None):
    raise OSError("connection refused")


# --- What the situation is -----------------------------------------------------------

def test_a_server_that_answers_is_left_alone():
    runner = Runner()
    progress = Progress()
    assert ollama_setup.ensure(progress, runner, opener=answering()) == ollama_setup.RUNNING
    assert runner.commands == [], "it ran a command against a server that was already up"
    assert runner.spawned == [], "it started a second server"
    assert progress.lines == [], "it reported news where there was none"


def test_an_http_error_still_counts_as_running():
    """The question is whether the server is up, not what it thinks of the request."""
    import urllib.error

    def refuses(url, timeout=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    assert ollama_setup.responding(opener=refuses) is True


def test_the_probe_does_not_use_curl():
    """curl is not on every Windows 10 build, and the .bat's check assumed it was.

    That check ran `curl` to decide whether Ollama was up; where curl was absent
    the errorlevel read as "not running", the launcher then waited thirty seconds
    for a server that was already answering, and refused to start Leti.
    """
    source = code_of(ollama_setup)
    assert "curl" not in source
    assert "urllib" in source


# --- Absent -------------------------------------------------------------------------

def test_no_ollama_and_no_winget_is_reported_and_not_fatal(monkeypatch):
    monkeypatch.setattr(ollama_setup, "executable", lambda: None)
    monkeypatch.setattr(ollama_setup, "is_windows", lambda: True)
    monkeypatch.setattr(ollama_setup.shutil, "which", lambda name: None)

    progress = Progress()
    verdict = ollama_setup.ensure(progress, Runner(), opener=never_answering)

    assert verdict == ollama_setup.ABSENT
    assert "ollama.com" in progress.text, "it did not say where to get it"


def test_no_ollama_on_a_platform_winget_cannot_help(monkeypatch):
    monkeypatch.setattr(ollama_setup, "executable", lambda: None)
    monkeypatch.setattr(ollama_setup, "is_windows", lambda: False)

    progress = Progress()
    runner = Runner()
    assert ollama_setup.ensure(progress, runner, opener=never_answering) == ollama_setup.ABSENT
    assert runner.ran("winget") == [], "it tried winget where there is none"
    assert "ollama.com" in progress.text


def test_absent_on_windows_is_installed_with_winget_then_started(monkeypatch):
    """The whole point: the executable now does what the .bat used to do alone."""
    installed = {"yes": False}
    monkeypatch.setattr(ollama_setup, "is_windows", lambda: True)
    monkeypatch.setattr(ollama_setup, "executable",
                        lambda: "C:\\ollama\\ollama.exe" if installed["yes"] else None)

    class InstallingRunner(Runner):
        def __call__(self, command, **kwargs):
            if "winget" in " ".join(str(c) for c in command):
                installed["yes"] = True
            return super().__call__(command, **kwargs)

    runner = InstallingRunner()
    monkeypatch.setattr(ollama_setup.shutil, "which",
                        lambda name: "C:\\winget.exe" if name == "winget" else None)

    progress = Progress()
    # Not answering for the first probe, answering once the server is started.
    verdict = ollama_setup.ensure(progress, runner, opener=answering(times=2))

    assert verdict == ollama_setup.RUNNING
    assert runner.ran("winget"), "winget was never called"
    winget = runner.ran("winget")[0]
    for flag in ("--id", "Ollama.Ollama", "--silent", "--accept-package-agreements"):
        assert flag in winget, f"winget was called without {flag}"
    assert runner.spawned, "the server was never started"
    assert runner.spawned[0][0] == ["C:\\ollama\\ollama.exe", "serve"]


def test_a_winget_install_that_fails_is_reported_and_not_fatal(monkeypatch):
    monkeypatch.setattr(ollama_setup, "is_windows", lambda: True)
    monkeypatch.setattr(ollama_setup, "executable", lambda: None)
    monkeypatch.setattr(ollama_setup.shutil, "which", lambda name: "C:\\winget.exe")

    progress = Progress()
    verdict = ollama_setup.ensure(progress, Runner(returncode=1), opener=never_answering)
    assert verdict == ollama_setup.ABSENT
    assert "ollama.com" in progress.text


def test_installed_but_not_yet_on_this_process_path_says_so(monkeypatch):
    """winget succeeded, but PATH was read when this process started.

    Telling the user to start Leti again is the honest answer. Starting a server
    that cannot be found would report a problem about the server instead.
    """
    monkeypatch.setattr(ollama_setup, "is_windows", lambda: True)
    monkeypatch.setattr(ollama_setup, "executable", lambda: None)
    monkeypatch.setattr(ollama_setup.shutil, "which", lambda name: "C:\\winget.exe")

    progress = Progress()
    verdict = ollama_setup.ensure(progress, Runner(returncode=0), opener=never_answering)
    assert verdict == ollama_setup.ABSENT
    assert "PATH" in progress.text and "again" in progress.text


# --- Stopped ------------------------------------------------------------------------

def test_an_installed_server_that_is_not_answering_is_started(monkeypatch):
    monkeypatch.setattr(ollama_setup, "executable", lambda: "/usr/bin/ollama")
    runner = Runner()
    progress = Progress()

    assert ollama_setup.ensure(progress, runner, opener=answering(times=3),
                               sleep=lambda _s: None) == ollama_setup.RUNNING
    assert runner.spawned[0][0] == ["/usr/bin/ollama", "serve"]
    assert runner.ran("winget") == [], "it tried to install something already installed"


def test_a_server_that_never_answers_reports_stopped_without_blocking(monkeypatch):
    monkeypatch.setattr(ollama_setup, "executable", lambda: "/usr/bin/ollama")
    progress = Progress()
    slept = []

    verdict = ollama_setup.ensure(progress, Runner(), opener=never_answering,
                                  sleep=slept.append)
    assert verdict == ollama_setup.STOPPED
    assert slept, "it did not wait at all"
    assert sum(slept) <= ollama_setup.START_TIMEOUT_SECONDS, "it waited past its own timeout"
    assert "Leti" in progress.text


def test_a_server_that_will_not_spawn_is_reported_and_not_fatal(monkeypatch):
    monkeypatch.setattr(ollama_setup, "executable", lambda: "/usr/bin/ollama")
    progress = Progress()
    verdict = ollama_setup.ensure(progress, Runner(popen_raises=OSError("denied")),
                                  opener=never_answering, sleep=lambda _s: None)
    assert verdict == ollama_setup.STOPPED
    assert "ollama serve" in progress.text, "it did not say what the user could do"


def test_the_server_is_detached_so_it_outlives_the_launcher(monkeypatch):
    """The .bat used Start-Process for this. A child that dies with the console
    would take the model server down when the launcher's window closed."""
    monkeypatch.setattr(ollama_setup, "executable", lambda: "/usr/bin/ollama")
    monkeypatch.setattr(ollama_setup, "is_windows", lambda: False)
    runner = Runner()
    ollama_setup.start(Progress(), runner, opener=answering(), sleep=lambda _s: None)
    assert runner.spawned[0][1].get("start_new_session") is True


def test_on_windows_the_server_gets_no_console_of_its_own(monkeypatch):
    import subprocess as sp

    monkeypatch.setattr(ollama_setup, "executable", lambda: "C:\\ollama.exe")
    monkeypatch.setattr(ollama_setup, "is_windows", lambda: True)
    runner = Runner()
    ollama_setup.start(Progress(), runner, opener=answering(), sleep=lambda _s: None)
    flags = runner.spawned[0][1].get("creationflags", 0)
    # CREATE_NO_WINDOW only exists on Windows, so this asserts the bits that do.
    expected = 0
    for name in ("CREATE_NO_WINDOW", "DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
        expected |= int(getattr(sp, name, 0) or 0)
    assert flags == expected


# --- Nothing here is ever a reason not to start Leti ---------------------------------

def test_ensure_never_raises_whatever_goes_wrong(monkeypatch):
    def explode(*_a, **_k):
        raise RuntimeError("the check itself is broken")

    monkeypatch.setattr(ollama_setup, "responding", explode)
    progress = Progress()
    assert ollama_setup.ensure(progress, Runner()) == ollama_setup.UNKNOWN
    assert progress.text, "it failed silently"


def test_no_path_through_this_module_raises_setupfailed():
    """SetupFailed is what stops a launch. Ollama must not be able to."""
    assert "SetupFailed" not in code_of(ollama_setup)


# --- It is actually wired into the one path both front doors take ---------------------

def test_prepare_makes_sure_there_is_a_model_server(tmp_path, monkeypatch):
    """Both front doors go through prepare(), which is why the step lives there."""
    (tmp_path / "requirements.txt").write_text("pyyaml>=6.0.1\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('leti')\n", encoding="utf-8")

    asked = []
    monkeypatch.setattr(bootstrap, "prepared_python", lambda root: pathlib.Path(sys.executable))
    monkeypatch.setattr(bootstrap, "ensure_pip", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "survey", lambda *a, **k: {
        "pyyaml": {"version": "6.0.2", "path": str(tmp_path / "pyyaml.dist-info")}})
    (tmp_path / "pyyaml.dist-info").mkdir()
    monkeypatch.setattr(bootstrap, "install_browser", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "place_shortcuts", lambda *a, **k: [])
    monkeypatch.setattr(bootstrap, "ensure_model_server",
                        lambda progress, run=None, ensure=None: asked.append("yes") or "running")

    bootstrap.prepare(tmp_path, bootstrap.Progress(out=None), run=Runner())
    assert asked == ["yes"], "prepare() did not make sure there was a model server"


def test_the_quick_path_makes_sure_too(tmp_path, monkeypatch):
    """The second launch is the common case, and it skips almost everything.

    Ollama has to be checked there as well: a machine that was set up yesterday
    has no server running today until something starts one.
    """
    (tmp_path / "requirements.txt").write_text("pyyaml>=6.0.1\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('leti')\n", encoding="utf-8")
    witness = tmp_path / "pyyaml.dist-info"
    witness.mkdir()

    monkeypatch.setattr(bootstrap, "prepared_python", lambda root: pathlib.Path(sys.executable))
    monkeypatch.setattr(bootstrap, "ensure_pip", lambda *a, **k: None)
    bootstrap.write_state(tmp_path,
                          requirements=bootstrap.requirements_fingerprint(
                              tmp_path / "requirements.txt"),
                          complete=True, shortcuts=True, witnesses=[str(witness)])

    asked = []
    monkeypatch.setattr(bootstrap, "ensure_model_server",
                        lambda progress, run=None, ensure=None: asked.append("yes") or "running")
    surveyed = []
    monkeypatch.setattr(bootstrap, "survey",
                        lambda *a, **k: surveyed.append("yes") or {})

    bootstrap.prepare(tmp_path, bootstrap.Progress(out=None), run=Runner())
    assert asked == ["yes"], "the quick path skipped the model server check"
    assert surveyed == [], "the quick path was not actually the quick path"


def test_a_failing_model_server_check_does_not_stop_prepare(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("pyyaml>=6.0.1\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('leti')\n", encoding="utf-8")
    witness = tmp_path / "pyyaml.dist-info"
    witness.mkdir()
    monkeypatch.setattr(bootstrap, "prepared_python", lambda root: pathlib.Path(sys.executable))
    monkeypatch.setattr(bootstrap, "ensure_pip", lambda *a, **k: None)
    bootstrap.write_state(tmp_path,
                          requirements=bootstrap.requirements_fingerprint(
                              tmp_path / "requirements.txt"),
                          complete=True, shortcuts=True, witnesses=[str(witness)])

    progress = bootstrap.Progress(out=None)
    verdict = bootstrap.ensure_model_server(
        progress, Runner(), ensure=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert verdict == "unknown"
    # And prepare still returns an interpreter.
    assert bootstrap.prepare(tmp_path, bootstrap.Progress(out=None), run=Runner())


# --- The frozen build has to carry it ------------------------------------------------

def test_the_spec_names_the_lazily_imported_module():
    """It is imported inside a function, so PyInstaller cannot see it.

    Without this the executable builds, runs, and silently never checks on Ollama -
    which is the bug the module was written to fix, reintroduced in the one build
    nobody can test by importing it.
    """
    spec = (ROOT / "launcher" / "Leti.spec").read_text(encoding="utf-8")
    assert '"launcher.ollama_setup"' in spec
    assert '"urllib.error"' in spec


def test_the_bat_no_longer_carries_its_own_copy():
    """One description of the step, not two that can disagree."""
    bat = (ROOT / "Launch Leti (Windows).bat").read_text(encoding="utf-8")
    assert "winget install" not in bat, "the .bat still installs Ollama itself"
    assert "Start-Process ollama" not in bat, "the .bat still starts the server itself"
    assert "curl" not in bat, "the .bat still probes the server with curl"
    # What it does keep is pre-pulling the configured models, which is not the
    # same job and is not duplicated anywhere.
    assert "ollama pull" in bat


def test_the_bat_reads_settings_as_utf8():
    """Its inline Python read config/settings.yaml in the console code page, with
    2>nul swallowing the decode error - so the model list came back empty and
    nothing was pulled, silently."""
    bat = (ROOT / "Launch Leti (Windows).bat").read_text(encoding="utf-8")
    assert "open('config/settings.yaml', encoding='utf-8')" in bat


def test_this_module_needs_nothing_installed():
    """It runs before pip has. Standard library only, like the rest of launcher/."""
    import ast

    tree = ast.parse((ROOT / "launcher" / "ollama_setup.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"launcher", "core"}
    assert third_party == set(), f"it needs {third_party} to be installed first"


def test_it_does_not_choose_or_download_a_model():
    """core/model_setup.py is the authority for that, and stays the only one."""
    source = code_of(ollama_setup)
    assert "ollama pull" not in source
    assert "/api/pull" not in source
    assert "reasoning_model" not in source


# --- Where Windows actually keeps Python ---------------------------------------------

def test_the_all_users_install_location_is_looked_in(monkeypatch):
    """It was looked for under Programs\\Python too, a path that never exists."""
    import inspect

    source = inspect.getsource(bootstrap._windows_python_installs)
    assert '"Programs/Python/Python3*"' in source, "the per-user layout is gone"
    assert source.count('"Python3*"') >= 2, "the all-users layout is not looked for"
    assert "PROGRAMFILES" in source


@pytest.mark.parametrize("names,expected_first", [
    (["Python39", "Python313"], "Python313"),
    (["Python311", "Python312", "Python314"], "Python314"),
    (["Python38", "Python39"], "Python39"),
])
def test_a_newer_python_is_preferred_over_an_older_one(names, expected_first):
    """Lexically "Python39" sorts above "Python313", which is the wrong way round."""
    ordered = sorted((pathlib.Path("/x") / n for n in names),
                     key=bootstrap._python_dir_order, reverse=True)
    assert ordered[0].name == expected_first


def test_the_search_is_not_a_hand_written_version_range():
    """It was range(13, 10, -1), so 3.14 was invisible the day it shipped."""
    import inspect

    source = inspect.getsource(bootstrap._windows_python_installs)
    assert "range(" not in source
    assert "glob(" in source
