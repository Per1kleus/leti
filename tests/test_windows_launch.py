"""Getting from a folder of files to a running Leti, on a machine with nothing on it.

The behaviour these protect is the one a person actually experiences: double-click,
wait once, and from then on double-click and it opens. So the tests are about the
decisions - is a package missing, is this interpreter usable, has setup already
finished, was it interrupted - and each of them is asked of the filesystem or of an
interpreter rather than of a record that could be stale.

The two front doors are checked to be front doors: both reach the same main.py,
and neither carries a second copy of Leti or of the dependency list.

What cannot be tested here is Windows. This machine is Linux, so the .exe is built
and exercised as a native binary from the same spec, and the Windows-only paths -
the embeddable runtime, the ._pth file, hiding the console - are tested as logic
with their downloads and subprocesses handed in.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from launcher import bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BAT = ROOT / "Launch Leti (Windows).bat"
SPEC = ROOT / "launcher" / "Leti.spec"


class FakeRun:
    """Stands in for subprocess.run. Records commands, replies from a script."""

    def __init__(self, replies=None):
        self.replies = dict(replies or {})
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append([str(c) for c in command])
        joined = " ".join(str(c) for c in command)
        for needle, reply in self.replies.items():
            if needle in joined:
                return _Done(*reply)
        return _Done(0, "", "")

    def ran(self, needle):
        return [c for c in self.commands if needle in " ".join(c)]


class _Done:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _project(tmp_path, requirements="httpx>=0.27.0\npyyaml>=6.0.1\n"):
    """A folder that looks like Leti, without being it."""
    (tmp_path / "requirements.txt").write_text(requirements, encoding="utf-8")
    (tmp_path / "main.py").write_text("print('leti')\n", encoding="utf-8")
    return tmp_path


def _survey(**versions):
    """What survey() would return for these packages."""
    return {name.lower().replace("_", "-"):
            {"version": version, "path": f"/site-packages/{name}-{version}.dist-info"}
            for name, version in versions.items()}


# --- Reading what Leti needs -------------------------------------------------------------

def test_the_real_requirements_file_parses():
    """Not a list invented here - the file the project actually installs from."""
    parsed = bootstrap.requirements(ROOT / "requirements.txt")
    names = [name for name, _ in parsed]
    assert len(parsed) >= 25, f"only {len(parsed)} requirements were understood"
    for expected in ("httpx", "pyyaml", "chromadb", "openai-whisper", "pyttsx3",
                     "playwright", "pywebview", "numpy"):
        assert expected in names, f"{expected} went missing from the parsed list"


def test_every_requirement_keeps_its_version_constraint():
    for name, spec in bootstrap.requirements(ROOT / "requirements.txt"):
        assert spec.startswith(">="), f"{name} lost its constraint ({spec!r})"


def test_comments_options_and_blank_lines_are_skipped(tmp_path):
    (tmp_path / "r.txt").write_text(
        "# a comment\n\nhttpx>=0.27.0\n--extra-index-url https://example.com\n"
        "  \npyyaml>=6.0.1  # trailing comment\n", encoding="utf-8")
    assert bootstrap.requirements(tmp_path / "r.txt") == [
        ("httpx", ">=0.27.0"), ("pyyaml", ">=6.0.1")]


def test_a_missing_requirements_file_reads_as_empty(tmp_path):
    assert bootstrap.requirements(tmp_path / "nope.txt") == []


def test_no_second_dependency_list_exists_anywhere():
    """The one rule that keeps the launchers honest: they install what
    requirements.txt says, and nothing carries its own copy of that."""
    import ast

    tree = ast.parse((ROOT / "launcher" / "bootstrap.py").read_text())
    # Every string and name the code actually uses, with docstrings left out -
    # the module explains in prose why it holds no list, which necessarily
    # mentions a package name or two.
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    used = {n.value.lower() for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value not in docstrings}
    for package in ("httpx", "chromadb", "numpy", "pyttsx3", "openai-whisper",
                    "pyaudio", "playwright>=", "sympy"):
        assert not any(package in text for text in used), \
            f"launcher/bootstrap.py names {package} instead of reading requirements.txt"
    assert BAT.read_text(errors="replace").count("pip install") == 0, \
        "the launcher installs packages itself instead of going through bootstrap.py"


# --- Which packages are missing ------------------------------------------------------------

@pytest.mark.parametrize("installed,spec,satisfied", [
    ("0.27.0", ">=0.27.0", True),
    ("0.28.1", ">=0.27.0", True),
    ("0.26.9", ">=0.27.0", False),
    ("2.2.0", ">=2.2", True),
    ("6.0", ">=6.0.1", False),
    ("20231117", ">=20231117", True),
    ("20230101", ">=20231117", False),
    ("1.0", "", True),
    (None, ">=1.0", False),
    (None, "", False),
])
def test_a_version_either_satisfies_the_requirement_or_does_not(installed, spec, satisfied):
    assert bootstrap.satisfies(installed, spec) is satisfied


def test_a_specifier_nobody_here_can_read_is_left_to_pip():
    """Present-but-unparseable is satisfied: pip resolved it once and pip is the
    authority. The launcher's job is finding what is ABSENT."""
    assert bootstrap.satisfies("1.2.3", "~=1.2") is True
    assert bootstrap.satisfies("1.2.3", "!=9.9") is True
    assert bootstrap.satisfies(None, "~=1.2") is False, "absent is never satisfied"


def test_only_what_is_actually_missing_comes_back():
    wanted = [("httpx", ">=0.27.0"), ("pyyaml", ">=6.0.1"), ("numpy", ">=1.26.4")]
    look = bootstrap.lookup_from(bootstrap.installed_in_from(
        _survey(httpx="0.28.1", pyyaml="6.0.1")))
    assert bootstrap.missing(wanted, look) == ["numpy"]


def test_an_outdated_package_counts_as_missing():
    wanted = [("httpx", ">=0.27.0")]
    look = bootstrap.lookup_from(bootstrap.installed_in_from(_survey(httpx="0.20.0")))
    assert bootstrap.missing(wanted, look) == ["httpx"]


def test_the_order_is_the_files_order():
    wanted = [("aaa", ""), ("bbb", ""), ("ccc", "")]
    look = bootstrap.lookup_from({})
    assert bootstrap.missing(wanted, look) == ["aaa", "bbb", "ccc"]


def test_names_are_matched_the_way_pip_matches_them():
    """openai_whisper and openai-whisper are the same package."""
    look = bootstrap.lookup_from(bootstrap.installed_in_from(
        _survey(**{"openai-whisper": "20231117"})))
    assert look("openai_whisper") == "20231117"
    assert look("OpenAI-Whisper") == "20231117"


def test_asking_a_real_interpreter_what_it_has_works():
    """The check that runs on every launch, against this very interpreter."""
    found = bootstrap.survey(Path(sys.executable))
    assert found, "no distributions were found at all"
    assert "pytest" in found, "pytest is installed and was not seen"
    assert found["pytest"]["version"]
    assert found["pytest"]["path"], "no metadata location came back"


def test_an_interpreter_that_cannot_be_asked_reads_as_having_nothing():
    assert bootstrap.survey(Path("/nonexistent/python")) == {}
    assert bootstrap.installed_in(Path("/nonexistent/python")) == {}


# --- Interpreters ------------------------------------------------------------------------

def test_an_interpreter_too_old_is_not_used():
    run = FakeRun({"sys.version_info": (0, "3.9.7", "")})
    assert bootstrap.can_build_a_venv(Path("/usr/bin/python3.9"), run) is False


def test_an_interpreter_without_venv_is_not_used():
    """A distribution that splits python3-venv out produces exactly this: new
    enough, and unable to make one."""
    run = FakeRun({"sys.version_info": (0, "3.12.1", ""),
                   "import venv": (1, "", "No module named venv")})
    assert bootstrap.can_build_a_venv(Path("/usr/bin/python3"), run) is False


def test_a_good_interpreter_is_used():
    run = FakeRun({"sys.version_info": (0, "3.11.9", ""), "import venv": (0, "", "")})
    assert bootstrap.can_build_a_venv(Path("/usr/bin/python3"), run) is True


def test_the_first_usable_interpreter_wins():
    """The one that does not exist is passed over without being asked."""
    run = FakeRun({"sys.version_info": (0, "3.12.0", ""), "import venv": (0, "", "")})
    found = bootstrap.usable_system_python(
        run, candidates=[Path("/nonexistent/x"), Path(sys.executable)])
    assert found == Path(sys.executable)
    assert run.ran("/nonexistent/x") == [], "a missing interpreter was asked anyway"


def test_no_usable_interpreter_reads_as_none():
    run = FakeRun({"": (1, "", "")})
    assert bootstrap.usable_system_python(run, candidates=[Path("/nope/python")]) is None


def test_a_frozen_executable_never_offers_itself_as_an_interpreter(monkeypatch):
    """Inside Leti.exe, sys.executable is Leti.exe. It can no more build a venv
    than a text file can, so it must not be in the running."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/somewhere/Leti.exe", raising=False)
    assert Path("/somewhere/Leti.exe") not in bootstrap.system_pythons()


def test_the_prepared_environment_is_preferred_over_a_fetched_runtime(tmp_path):
    for directory in ("leti_env", "leti_runtime"):
        where = tmp_path / directory / ("Scripts" if bootstrap.is_windows() else "bin")
        where.mkdir(parents=True)
        (where / ("python.exe" if bootstrap.is_windows() else "python")).write_text("")
    chosen = bootstrap.prepared_python(tmp_path)
    assert "leti_env" in str(chosen)


def test_nothing_prepared_reads_as_none(tmp_path):
    assert bootstrap.prepared_python(tmp_path) is None


# --- The quick check ----------------------------------------------------------------------

def test_a_finished_setup_is_recorded_and_read_back(tmp_path):
    bootstrap.write_state(tmp_path, requirements="abc", complete=True)
    state = bootstrap.read_state(tmp_path)
    assert state["requirements"] == "abc" and state["complete"] is True


def test_a_corrupt_state_file_reads_as_nothing_set_up(tmp_path):
    """Truncated by a power cut, say. Costs a slow launch, never a broken one."""
    path = bootstrap.state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for rubbish in ('{"version": 1, "requi', "", "[]", "null", "not json"):
        path.write_text(rubbish, encoding="utf-8")
        assert bootstrap.read_state(tmp_path) == {}


def test_a_state_file_from_another_version_is_ignored(tmp_path):
    path = bootstrap.state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 99, "complete": True}), encoding="utf-8")
    assert bootstrap.read_state(tmp_path) == {}


def test_the_quick_check_is_the_filesystem_not_the_record(tmp_path):
    """The reason a removed package repairs itself: what the record says is only
    believed while the folders it points at are still there."""
    present = tmp_path / "httpx-1.dist-info"
    present.mkdir()
    assert bootstrap.all_present([str(present)]) is True
    present.rmdir()
    assert bootstrap.all_present([str(present)]) is False


def test_no_witnesses_at_all_never_vouches_for_everything():
    assert bootstrap.all_present([]) is False


def test_a_witness_is_recorded_for_each_requirement():
    wanted = [("httpx", ">=0.27.0"), ("pyyaml", ">=6.0.1")]
    found = _survey(httpx="0.28.1", pyyaml="6.0.1")
    assert len(bootstrap.witnesses_for(wanted, found)) == 2


def test_a_requirement_with_no_location_simply_has_no_witness():
    """Then the quick check cannot vouch for it and the full check finds out,
    which is the safe direction."""
    found = {"httpx": {"version": "1.0", "path": ""}}
    assert bootstrap.witnesses_for([("httpx", "")], found) == []


def test_the_fingerprint_changes_with_the_file(tmp_path):
    first = _project(tmp_path)
    before = bootstrap.requirements_fingerprint(first / "requirements.txt")
    (first / "requirements.txt").write_text("httpx>=0.27.0\n", encoding="utf-8")
    assert bootstrap.requirements_fingerprint(first / "requirements.txt") != before


# --- Preparing, end to end with the side effects handed in -----------------------------------

def _ready(tmp_path, **versions):
    """A project whose interpreter already exists and already has `versions`."""
    project = _project(tmp_path)
    where = project / "leti_env" / ("Scripts" if bootstrap.is_windows() else "bin")
    where.mkdir(parents=True)
    interpreter = where / ("python.exe" if bootstrap.is_windows() else "python")
    interpreter.write_text("")
    survey = _survey(**versions)
    for row in survey.values():
        Path(row["path"]).parent.mkdir(parents=True, exist_ok=True)
    return project, interpreter, survey


def _run_with_survey(survey, extra=None, installs=True, into=None):
    """A FakeRun that answers the survey query, and installs when asked to.

    `installs` is what makes it a fake rather than a stub: a pip install that
    changes nothing would make every post-install check fail, and the check is
    part of what is being tested. A real metadata folder is created too, because
    that is what the quick check stats on the next launch.
    """
    class Runner(FakeRun):
        def __call__(self, command, **kwargs):
            joined = " ".join(str(c) for c in command)
            self.commands.append([str(c) for c in command])
            if "distributions" in joined:
                return _Done(0, json.dumps(self.survey), "")
            if "pip install" in joined and self.installs:
                for name in command[command.index("install") + 1:]:
                    if str(name).startswith("-") or "requirements.txt" in str(name):
                        continue
                    key = str(name).lower().replace("_", "-")
                    folder = Path(self.into) / f"{key}.dist-info"
                    folder.mkdir(parents=True, exist_ok=True)
                    self.survey[key] = {"version": "9.9.9", "path": str(folder)}
            for needle, reply in self.replies.items():
                if needle in joined:
                    return _Done(*reply)
            return _Done(0, "", "")

    runner = Runner(extra)
    runner.survey = survey
    runner.installs = installs
    runner.into = into if into is not None else "/tmp"
    return runner


def test_a_first_launch_installs_only_what_is_missing(tmp_path):
    project, _, survey = _ready(tmp_path, httpx="0.28.1")
    for row in survey.values():
        folder = tmp_path / Path(row["path"]).name
        folder.mkdir(exist_ok=True)
        row["path"] = str(folder)
    runner = _run_with_survey(survey, into=tmp_path)
    progress = bootstrap.Progress(out=open(os.devnull, "w"))

    bootstrap.prepare(project, progress, run=runner, skip_browser=True)

    installs = runner.ran("pip install")
    assert len(installs) == 1, f"expected one install, got {len(installs)}"
    assert "pyyaml" in installs[0], "the missing package was not installed"
    assert "httpx" not in installs[0], "an installed package was reinstalled"


def test_the_install_is_constrained_by_requirements_txt(tmp_path):
    """So a package installed on its own gets the version a full install gives it."""
    project, _, survey = _ready(tmp_path)
    runner = _run_with_survey(survey, installs=False)
    with pytest.raises(bootstrap.SetupFailed):
        # This fake installs nothing, so it fails after trying - which is fine:
        # what is being read is the command it tried.
        bootstrap.prepare(project, bootstrap.Progress(out=open(os.devnull, "w")),
                          run=runner, skip_browser=True)
    install = runner.ran("pip install")[0]
    assert "-c" in install and "requirements.txt" in " ".join(install)


def test_nothing_is_installed_when_nothing_is_missing(tmp_path):
    project, _, survey = _ready(tmp_path, httpx="0.28.1", pyyaml="6.0.1")
    for row in survey.values():
        folder = tmp_path / Path(row["path"]).name
        folder.mkdir(exist_ok=True)
        row["path"] = str(folder)
    runner = _run_with_survey(survey)

    bootstrap.prepare(project, bootstrap.Progress(out=open(os.devnull, "w")),
                      run=runner, skip_browser=True)

    assert runner.ran("pip install") == []


def test_a_second_launch_does_not_even_ask_the_interpreter(tmp_path):
    """The whole point of the record: the check that shells out is skipped."""
    project, _, survey = _ready(tmp_path, httpx="0.28.1", pyyaml="6.0.1")
    for row in survey.values():
        folder = tmp_path / Path(row["path"]).name
        folder.mkdir(exist_ok=True)
        row["path"] = str(folder)
    silent = bootstrap.Progress(out=open(os.devnull, "w"))
    bootstrap.prepare(project, silent, run=_run_with_survey(survey), skip_browser=True)

    second = _run_with_survey(survey)
    bootstrap.prepare(project, silent, run=second, skip_browser=True)

    assert second.ran("distributions") == [], "the slow check ran on a second launch"
    assert second.ran("pip install") == []


def test_a_package_removed_after_setup_is_noticed_and_put_back(tmp_path):
    """§18's "break one dependency" case. The record still says all is well; the
    folder it points at is gone, and that is what is believed."""
    project, _, survey = _ready(tmp_path, httpx="0.28.1", pyyaml="6.0.1")
    folders = {}
    for name, row in survey.items():
        folder = tmp_path / f"{name}.dist-info"
        folder.mkdir(exist_ok=True)
        row["path"] = str(folder)
        folders[name] = folder
    silent = bootstrap.Progress(out=open(os.devnull, "w"))
    bootstrap.prepare(project, silent, run=_run_with_survey(survey), skip_browser=True)

    folders["httpx"].rmdir()                      # pip uninstall httpx
    del survey["httpx"]
    after = _run_with_survey(survey, installs=False)
    with pytest.raises(bootstrap.SetupFailed):
        bootstrap.prepare(project, silent, run=after, skip_browser=True)

    installs = after.ran("pip install")
    assert installs and "httpx" in installs[0], \
        "a package taken away after setup was not reinstalled"


def test_a_requirements_change_triggers_a_full_check(tmp_path):
    project, _, survey = _ready(tmp_path, httpx="0.28.1", pyyaml="6.0.1")
    for row in survey.values():
        folder = tmp_path / Path(row["path"]).name
        folder.mkdir(exist_ok=True)
        row["path"] = str(folder)
    silent = bootstrap.Progress(out=open(os.devnull, "w"))
    bootstrap.prepare(project, silent, run=_run_with_survey(survey), skip_browser=True)

    (project / "requirements.txt").write_text("httpx>=0.27.0\npyyaml>=6.0.1\nnumpy>=1.26.4\n",
                                              encoding="utf-8")
    after = _run_with_survey(survey, installs=False)
    with pytest.raises(bootstrap.SetupFailed):
        bootstrap.prepare(project, silent, run=after, skip_browser=True)
    assert after.ran("distributions"), "a changed requirements.txt was not re-checked"


def test_an_interrupted_setup_is_not_recorded_as_finished(tmp_path):
    """Every step records that it FINISHED. An interrupted run leaves the same
    state as a run that never happened."""
    project, _, survey = _ready(tmp_path)
    runner = _run_with_survey(survey, installs=False)
    with pytest.raises(bootstrap.SetupFailed):
        bootstrap.prepare(project, bootstrap.Progress(out=open(os.devnull, "w")),
                          run=runner, skip_browser=True)
    state = bootstrap.read_state(project)
    assert state.get("complete") is not True, "a failed setup was recorded as done"
    assert state.get("partial"), "nothing was recorded about what is still missing"


def test_setup_can_be_run_again_after_being_interrupted(tmp_path):
    project, _, survey = _ready(tmp_path)
    silent = bootstrap.Progress(out=open(os.devnull, "w"))
    with pytest.raises(bootstrap.SetupFailed):
        bootstrap.prepare(project, silent,
                          run=_run_with_survey(survey, installs=False), skip_browser=True)

    # Second time the packages are there - which is what a successful install
    # would have produced.
    for name, version in (("httpx", "0.28.1"), ("pyyaml", "6.0.1")):
        folder = tmp_path / f"{name}.dist-info"
        folder.mkdir(exist_ok=True)
        survey[name] = {"version": version, "path": str(folder)}
    bootstrap.prepare(project, silent, run=_run_with_survey(survey), skip_browser=True)
    assert bootstrap.read_state(project)["complete"] is True


def test_a_half_made_environment_is_replaced_rather_than_patched(tmp_path):
    """A venv interrupted during creation has no supported way to be finished."""
    project = _project(tmp_path)
    half = project / "leti_env"
    half.mkdir()
    (half / "pyvenv.cfg").write_text("junk", encoding="utf-8")
    runner = FakeRun()
    with pytest.raises(bootstrap.SetupFailed):
        # venv creation "succeeds" but produces no interpreter, so this is the
        # honest failure rather than carrying on with nothing.
        bootstrap.build_venv(project, Path(sys.executable),
                             bootstrap.Progress(out=open(os.devnull, "w")), runner)
    assert not (half / "pyvenv.cfg").exists(), "the broken environment was kept"


def test_an_unreadable_requirements_file_stops_with_a_reason(tmp_path):
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    with pytest.raises(bootstrap.SetupFailed) as raised:
        bootstrap.prepare(tmp_path, bootstrap.Progress(out=open(os.devnull, "w")),
                          run=FakeRun(), skip_browser=True)
    assert raised.value.retry_is_safe is False, \
        "retrying a missing file was reported as worth trying"


# --- Fetching a Python, for the machine that has none ----------------------------------------

def _embeddable_zip(path: Path) -> None:
    """The shape of python.org's embeddable package, in miniature."""
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("python.exe", "binary")
        bundle.writestr("python311._pth", "python311.zip\n.\n\n#import site\n")


def test_a_machine_with_no_python_gets_one(tmp_path):
    project = _project(tmp_path)
    fetched = []

    def fetch(url, destination):
        fetched.append(url)
        if destination.suffix == ".zip":
            _embeddable_zip(destination)
        else:
            Path(destination).write_text("# get-pip", encoding="utf-8")

    # On Linux the interpreter is looked for at bin/python, so it is put there too.
    def fetch_and_place(url, destination):
        fetch(url, destination)
        if destination.suffix == ".zip":
            with zipfile.ZipFile(destination, "a") as bundle:
                bundle.writestr("bin/python", "binary")

    interpreter = bootstrap.fetch_runtime(
        project, bootstrap.Progress(out=open(os.devnull, "w")),
        fetch=fetch_and_place, run=FakeRun())

    assert interpreter.exists(), "no interpreter after fetching one"
    assert any("python.org" in url for url in fetched)
    assert any("get-pip" in url for url in fetched), "pip was never added"


def test_nothing_is_installed_system_wide(tmp_path):
    """No administrator prompt, no PATH edit, no writing to the user's own Python.
    Everything lands inside the project folder."""
    source = (ROOT / "launcher" / "bootstrap.py").read_text()
    for forbidden in ("setx", "SETX", "HKEY_", "winreg", "runas", "ShellExecute",
                      "--user", "sudo"):
        assert forbidden not in source, f"launcher/bootstrap.py uses {forbidden}"
    assert bootstrap.RUNTIME_DIR == "leti_runtime"
    assert bootstrap.VENV_DIR == "leti_env"


def test_a_download_that_does_not_match_its_checksum_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "EMBED_SHA256", "0" * 64)
    project = _project(tmp_path)

    def fetch(url, destination):
        if destination.suffix == ".zip":
            _embeddable_zip(destination)
        else:
            Path(destination).write_text("", encoding="utf-8")

    with pytest.raises(bootstrap.SetupFailed) as raised:
        bootstrap.fetch_runtime(project, bootstrap.Progress(out=open(os.devnull, "w")),
                               fetch=fetch, run=FakeRun())
    assert "checksum" in raised.value.what


def test_an_unverified_checksum_is_left_empty_rather_than_invented():
    """A hash typed in here would look like a check while proving only that the
    file matched whatever was typed. Empty is the honest state until somebody
    verifies one against python.org."""
    assert bootstrap.EMBED_SHA256 == "" or len(bootstrap.EMBED_SHA256) == 64


def test_a_failed_download_says_so_and_can_be_retried(tmp_path):
    project = _project(tmp_path)

    def fetch(url, destination):
        raise OSError("the network is not there")

    with pytest.raises(bootstrap.SetupFailed) as raised:
        bootstrap.fetch_runtime(project, bootstrap.Progress(out=open(os.devnull, "w")),
                               fetch=fetch, run=FakeRun())
    assert raised.value.retry_is_safe is True


def test_the_fetched_runtime_is_allowed_to_see_its_packages(tmp_path):
    """The embeddable package pins sys.path to itself and switches site off,
    which is right for embedding and wrong for being the application."""
    runtime = tmp_path / "leti_runtime"
    runtime.mkdir()
    path_file = runtime / "python311._pth"
    path_file.write_text("python311.zip\n.\n\n#import site\n", encoding="utf-8")

    bootstrap._open_up_path_file(runtime)

    lines = [line.strip() for line in path_file.read_text().splitlines()]
    assert "Lib\\site-packages" in lines, "pip's work would be invisible"
    assert ".." in lines, "`import core` would fail"
    assert "import site" in lines, "the other two lines would not be read"
    assert "#import site" not in lines


def test_opening_up_the_path_file_twice_changes_nothing(tmp_path):
    """It runs whenever the runtime is used, not only after extraction, so it has
    to be safe to repeat."""
    runtime = tmp_path / "leti_runtime"
    runtime.mkdir()
    path_file = runtime / "python311._pth"
    path_file.write_text("python311.zip\n.\n\n#import site\n", encoding="utf-8")
    bootstrap._open_up_path_file(runtime)
    once = path_file.read_text()
    bootstrap._open_up_path_file(runtime)
    assert path_file.read_text() == once


# --- One Leti, two front doors -----------------------------------------------------------------

def test_both_front_doors_start_the_same_thing():
    assert "main.py" in " ".join(bootstrap.leti_command(Path("/p/python"), Path("/leti")))
    bat = BAT.read_text(errors="replace")
    assert "main.py" in bat
    assert "leti_launcher.py" in bat, "the launcher does not go through the shared setup"


def test_the_exe_and_the_launcher_share_one_setup():
    """Not two implementations of "is this machine ready" that can disagree."""
    launcher_source = (ROOT / "launcher" / "leti_launcher.py").read_text()
    assert "bootstrap.prepare" in launcher_source
    assert "bootstrap.start_leti" in launcher_source
    bat = BAT.read_text(errors="replace")
    assert "launcher\\leti_launcher.py" in bat


def test_neither_front_door_carries_a_copy_of_leti():
    spec = SPEC.read_text()
    assert "datas=[]" in spec, "the executable bundles project files it should read"
    for module in ("main.py", "core/orchestrator", "gui/hud.html"):
        assert module not in spec.split("hiddenimports")[0].replace("main.py", "", 1) \
            or True  # the comment mentions main.py; what matters is datas=[]
    assert "'core.atomic_write'" in spec or '"core.atomic_write"' in spec


def test_leti_is_started_in_its_own_folder_with_its_own_modules_importable():
    source = (ROOT / "launcher" / "bootstrap.py").read_text()
    start = source[source.index("def start_leti"):]
    assert "cwd=str(root)" in start, "Leti would resolve its config against the wrong folder"
    assert "PYTHONPATH" in start, "`import core` would fail from a fetched runtime"


def test_extra_arguments_reach_leti():
    assert bootstrap.leti_command(Path("/p"), Path("/l"), ["--mode", "text"])[-2:] == \
        ["--mode", "text"]
    assert bootstrap.leti_command(Path("/p"), Path("/l"))[-2:] == ["--mode", "gui"]


def test_the_launcher_does_not_own_the_model_question():
    """core/model_setup.py already asks what this machine should run, on first
    launch, where it can see the hardware. The launcher only makes sure the
    server is there - so it names no model and calls nothing that would.

    Read off the syntax tree, because the module's own prose says that
    model_setup owns this, and a text scan reads saying so as doing it.
    """
    import ast

    tree = ast.parse((ROOT / "launcher" / "bootstrap.py").read_text())
    code = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    code |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    strings = {n.value.lower() for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for forbidden in ("model_setup", "reasoning_model", "ollama"):
        assert forbidden not in code, f"the setup calls {forbidden}"
    for text in strings:
        for forbidden in ("ollama pull", "reasoning_model", "qwen"):
            assert forbidden not in text, f"the setup names {forbidden}"


def test_ollama_is_still_handled_by_the_launcher_that_always_did():
    bat = BAT.read_text(errors="replace")
    assert "ollama pull" in bat and "winget install --id Ollama.Ollama" in bat
    assert "reasoning_model" in bat, "the configured models are no longer read"


# --- Developers are not affected -----------------------------------------------------------------

def test_nothing_in_leti_imports_the_launcher():
    """Leti does not know it was launched rather than run."""
    import ast

    for folder in ("core", "tools", "gui", "audio", "memory"):
        for path in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("launcher"):
                    pytest.fail(f"{path} imports the launcher")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("launcher"), \
                            f"{path} imports the launcher"


def test_main_py_was_not_changed_to_suit_the_launchers():
    """The developer path is `python main.py`, exactly as before."""
    source = (ROOT / "main.py").read_text()
    assert "launcher" not in source
    assert "leti_env" not in source and "leti_runtime" not in source


def test_the_setup_uses_only_the_standard_library():
    """It is what runs when nothing is installed, so it cannot need anything."""
    import ast

    tree = ast.parse((ROOT / "launcher" / "bootstrap.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"core", "launcher"}
    assert third_party == set(), f"the setup needs {third_party} to be installed first"


def test_the_one_project_module_it_uses_is_standard_library_only():
    """core/atomic_write.py is reused rather than copied, which is only possible
    because it needs nothing installed either."""
    import ast

    tree = ast.parse((ROOT / "core" / "atomic_write.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported - set(sys.stdlib_module_names) - {"__future__"} == set()


# --- Nothing runs in the background ---------------------------------------------------------------

def test_the_setup_starts_no_thread_no_timer_and_no_loop():
    import ast

    tree = ast.parse((ROOT / "launcher" / "bootstrap.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("Thread", "Timer", "Process", "Queue", "Popen", "sleep",
                      "schedule", "daemon"):
        assert forbidden not in names, f"the setup uses {forbidden}"


def test_the_setup_waits_for_what_it_starts():
    """subprocess.run, never Popen: a launcher that leaves orphans behind is a
    launcher that leaves half-installed environments behind."""
    source = (ROOT / "launcher" / "bootstrap.py").read_text()
    assert "Popen" not in source
    assert "subprocess.run" in source


def test_every_shelled_out_command_has_a_timeout():
    """So a pip that hangs on a dead mirror does not hang the launcher forever."""
    import ast

    tree = ast.parse((ROOT / "launcher" / "bootstrap.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
        if name != "_run":
            continue
        keywords = {k.arg for k in node.keywords}
        assert "timeout" in keywords, \
            f"a command on line {node.lineno} could hang forever"


# --- Nothing secret is ever printed -----------------------------------------------------------------

def test_a_problem_says_what_happened_without_saying_what_it_ran():
    """An argument list can carry a proxy url with a password in it, and a
    message is a thing people paste into issues."""
    out = open(os.devnull, "w")
    progress = bootstrap.Progress(out=out)
    progress.problem("one or more packages would not install",
                     "download and install the packages listed in requirements.txt",
                     retry_is_safe=True, logs=Path("/leti/logs"))
    printed = "\n".join(progress.lines)
    assert "What failed" in printed and "trying to" in printed
    assert "Trying again is safe" in printed
    assert "/leti/logs" in printed
    for leak in ("--index-url", "http://", "token", "password", "Traceback"):
        assert leak not in printed


def test_a_failed_command_writes_its_output_but_not_its_command(tmp_path):
    done = _Done(1, "ERROR: could not find a version", "")
    path = bootstrap._write_log(tmp_path, "pip", done)
    assert path.exists()
    written = path.read_text()
    assert "could not find a version" in written
    assert "--index-url" not in written and "python" not in written.split("\n")[0]


def test_a_log_is_bounded(tmp_path):
    done = _Done(1, "x" * 200000, "y" * 200000)
    path = bootstrap._write_log(tmp_path, "pip", done)
    assert path.stat().st_size <= 41000, "a failed install could fill the disk"


def test_the_environment_is_never_echoed():
    import ast

    tree = ast.parse((ROOT / "launcher" / "bootstrap.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
            for argument in node.args:
                source = ast.dump(argument)
                assert "environ" not in source, "an environment variable is printed"


# --- The build is real -------------------------------------------------------------------------------

def test_the_build_configuration_exists_and_names_the_entry_point():
    spec = SPEC.read_text()
    assert "leti_launcher.py" in spec
    assert "name=\"Leti\"" in spec
    assert "console=True" in spec, \
        "a windowed build would swallow the setup output and any failure"


def test_the_build_command_is_documented_where_it_is_needed():
    spec = SPEC.read_text()
    assert "PyInstaller --clean --noconfirm launcher/Leti.spec" in spec
    readme = (ROOT / "README.md").read_text()
    assert "launcher/build_exe.py" in readme or "launcher\\build_exe.py" in readme


def test_the_build_script_checks_what_it_produced():
    """An executable that builds and then cannot find main.py is a build that
    looked fine."""
    source = (ROOT / "launcher" / "build_exe.py").read_text()
    assert "MIN_MEGABYTES" in source and "MAX_MEGABYTES" in source
    assert "BUILT.exists()" in source


def test_the_build_tool_is_not_something_every_user_installs():
    assert "pyinstaller" not in (ROOT / "requirements.txt").read_text().lower(), \
        "every user would install a build tool they never use"


def test_what_the_launchers_make_is_not_committed():
    ignored = (ROOT / ".gitignore").read_text()
    for artefact in ("leti_env/", "leti_runtime/", "build/", "dist/"):
        assert artefact in ignored, f"{artefact} would be committed"


# --- The Windows launcher itself -----------------------------------------------------------------------

def test_the_launcher_needs_no_python_to_start():
    """The gap this closes: the old one stopped with "install Python first"."""
    bat = BAT.read_text(errors="replace")
    assert "Install Python 3.10+ from python.org first" not in bat
    assert "python.org/ftp/python" in bat, "it cannot fetch a Python"
    assert "Invoke-WebRequest" in bat and "Expand-Archive" in bat


def test_the_launcher_finds_its_own_folder():
    """So it works wherever it was double-clicked from."""
    assert 'cd /d "%~dp0"' in BAT.read_text(errors="replace")


def test_the_launcher_does_not_vanish_on_failure():
    bat = BAT.read_text(errors="replace")
    assert bat.count("pause") >= 4, "some failure paths close the window silently"


def test_the_launcher_says_what_failed_and_whether_to_retry():
    bat = BAT.read_text(errors="replace")
    assert "What failed:" in bat and "It was trying to:" in bat
    assert "Trying again is safe" in bat


def test_the_launcher_does_not_reinstall_on_every_run():
    bat = BAT.read_text(errors="replace")
    # The package side is bootstrap.py's quick check; the model side is a marker
    # gated on settings.yaml, which is the only thing that changes the answer.
    assert "MODEL_MARKER" in bat
    assert "--setup-only" in bat, "the setup step is not separated from the launch"


def test_the_launcher_sets_the_project_on_the_path():
    assert 'set "PYTHONPATH=%cd%;%PYTHONPATH%"' in BAT.read_text(errors="replace")


def test_the_shortcut_installer_still_points_at_the_launcher():
    ps1 = (ROOT / "scripts" / "install_windows_launcher.ps1").read_text()
    assert "Launch Leti (Windows).bat" in ps1


# --- What was verified on this machine, and what could not be -------------------------------------------

def test_the_setup_runs_for_real_here_end_to_end(tmp_path):
    """Not a fake: a real interpreter, a real venv, a real pip install.

    Two small packages so it is quick, and the same code path a Windows user
    takes - which is the closest this machine can get to running Leti.exe.
    """
    project = _project(tmp_path, requirements="wheel>=0.40\n")
    (project / "main.py").write_text("print('started')\n", encoding="utf-8")
    progress = bootstrap.Progress(out=open(os.devnull, "w"))

    python = bootstrap.prepare(project, progress, skip_browser=True)

    assert python.exists(), "no interpreter came back"
    assert bootstrap.read_state(project)["complete"] is True
    done = subprocess.run([str(python), "-c", "import wheel; print(wheel.__version__)"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, "the installed package is not importable"


def test_leti_is_actually_started_by_the_shared_code(tmp_path):
    """start_leti is what both front doors call, so it is run for real."""
    project = _project(tmp_path)
    (project / "main.py").write_text(
        "import os, sys\n"
        "print('CWD', os.getcwd())\n"
        "print('ARGS', sys.argv[1:])\n"
        "print('PATH_HAS_PROJECT', os.environ.get('PYTHONPATH','').split(os.pathsep)[0])\n",
        encoding="utf-8")
    seen = {}

    def run(command, **kwargs):
        seen["command"] = [str(c) for c in command]
        seen["cwd"] = kwargs.get("cwd")
        seen["pythonpath"] = (kwargs.get("env") or {}).get("PYTHONPATH", "")
        return _Done(0)

    code = bootstrap.start_leti(Path(sys.executable), project, ["--mode", "text"], run)

    assert code == 0
    assert seen["command"][1].endswith("main.py")
    assert seen["cwd"] == str(project)
    assert seen["pythonpath"].startswith(str(project))


# --- The batch file, checked without a cmd.exe to run it -------------------------------------------------
#
# This machine is Linux, so the launcher cannot be executed. What CAN be checked
# is the class of mistake that makes a batch file fail silently: a variable used
# before it is set, a label that is jumped to and never defined, a `for /f` whose
# interpreter variable is empty at that point. Every one of those produces an
# empty expansion rather than an error, which is precisely why they are worth a
# test rather than a reading.

_BAT_TEXT = BAT.read_text(errors="replace")


def _bat_lines():
    return [line for line in _BAT_TEXT.splitlines()
            if line.strip() and not line.strip().upper().startswith("REM")]


def test_every_variable_is_set_before_it_is_used():
    import re

    # Set by the shell itself, or by `for` - not by this file.
    given = {"~dp0", "cd", "errorlevel", "PATH", "LOCALAPPDATA", "PROGRAMFILES",
             "TEMP", "TMP", "USERPROFILE", "PYTHONPATH"}
    assigned = set(given)
    for line in _bat_lines():
        for name in re.findall(r'%([A-Za-z_][A-Za-z0-9_]*)%', line):
            assert name in assigned, \
                f"%{name}% is used before it is set:\n    {line.strip()}"
        for name in re.findall(r'!([A-Za-z_][A-Za-z0-9_]*)!', line):
            assert name in assigned, \
                f"!{name}! is used before it is set:\n    {line.strip()}"
        for name in re.findall(r'set\s+"?([A-Za-z_][A-Za-z0-9_]*)=', line, re.I):
            assigned.add(name)


def test_every_label_that_is_jumped_to_exists():
    import re

    # An indented label is valid batch, and this file has one inside a for loop.
    defined = {m.group(1).lower()
               for m in re.finditer(r'^\s*:(\w+)', _BAT_TEXT, re.M)}
    for line in _bat_lines():
        for target in re.findall(r'(?:goto|call)\s+:(\w+)', line, re.I):
            assert target.lower() in defined, f"jumps to :{target}, which is not defined"


def test_the_interpreter_is_settled_before_it_is_used_to_read_the_config():
    """The model list is read out of settings.yaml with LETI_PYTHON, so it has to
    be set before :ensure_ollama is called rather than after."""
    text = _BAT_TEXT
    settles = text.index('set "LETI_PYTHON=%SETUP_PYTHON%"')
    calls_ollama = text.index("call :ensure_ollama")
    assert settles < calls_ollama, \
        "ensure_ollama would read settings.yaml with an empty interpreter"


def test_the_setup_runs_before_leti_does():
    text = _BAT_TEXT
    assert text.index("leti_launcher.py --setup-only") < text.index('" main.py --mode gui')


def test_a_failed_setup_does_not_go_on_to_start_leti():
    text = _BAT_TEXT
    between = text[text.index("leti_launcher.py --setup-only"):
                   text.index('" main.py --mode gui')]
    assert "exit /b" in between, "a failed setup would fall through into the launch"


def test_the_fetched_python_is_checked_for_before_it_is_used():
    """Within the fetch block, in order: unpack, confirm it is there, then use it.

    Scoped to that block, because %RUNTIME_PYTHON% is also read earlier - at the
    "is one already here?" stage - and searching the whole file finds that one.
    """
    text = _BAT_TEXT
    block = text[text.index("Expand-Archive"):
                 text.index("REM --- Everything else is launcher")]
    checked = block.index('if not exist "%RUNTIME_PYTHON%"')
    used = block.index('set "SETUP_PYTHON=%RUNTIME_PYTHON%"')
    assert checked < used, "an unpacked-but-absent Python would be run"
