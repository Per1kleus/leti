"""The Windows validation harness cannot report a pass for something it did not do.

scripts/validate_windows_release.py exists to be run on a real Windows machine,
which is the one thing the test suite cannot be. So what is checked here is the
property that makes its report worth reading: a check that did not run reports NOT
RUN, a check a person has to look at reports NEEDS A HUMAN, and neither of those
counts as a pass at any level of the summary.

Run on Linux, every Windows-dependent check must come back NOT RUN - and the
overall verdict must be a failure, not a green run that happened to skip
everything.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import validate_windows_release as harness  # noqa: E402


@pytest.fixture
def report(tmp_path):
    return harness.Report(tmp_path / "validation_report.json")


# --- The verdicts mean what they say --------------------------------------------------

def test_not_run_is_not_a_pass(report):
    report.check("stage", "something that ran", harness.PASS)
    report.check("stage", "something that did not", harness.NOT_RUN)
    assert report.worst() == harness.NOT_RUN


def test_needs_a_human_is_not_a_pass(report):
    report.check("stage", "something that ran", harness.PASS)
    report.check("stage", "something to look at", harness.NEEDS_HUMAN)
    assert report.worst() == harness.NEEDS_HUMAN


def test_a_failure_outranks_everything(report):
    for verdict in (harness.PASS, harness.NEEDS_HUMAN, harness.NOT_RUN,
                    harness.WARNING, harness.FAIL):
        report.check("stage", f"check {verdict}", verdict)
    assert report.worst() == harness.FAIL


def test_all_passes_is_the_only_way_to_pass(report):
    for name in ("a", "b", "c"):
        report.check("stage", name, harness.PASS)
    assert report.worst() == harness.PASS


def test_an_empty_report_is_not_a_pass(report):
    assert report.worst() != harness.PASS


def test_an_invented_verdict_is_refused(report):
    with pytest.raises(AssertionError):
        report.check("stage", "wishful", "PROBABLY FINE")


# --- Re-running replaces a verdict rather than accumulating two -----------------------

def test_a_later_run_replaces_an_earlier_verdict(report):
    report.check("stage", "the same check", harness.NOT_RUN, "no Windows here")
    report.check("stage", "the same check", harness.PASS, "ran on Windows")
    rows = [c for c in report.data["checks"] if c["name"] == "the same check"]
    assert len(rows) == 1, "two verdicts for one check would let a report disagree with itself"
    assert rows[0]["verdict"] == harness.PASS


def test_the_report_survives_being_written_and_read_again(tmp_path):
    path = tmp_path / "validation_report.json"
    first = harness.Report(path)
    first.check("stage", "a check from an earlier sitting", harness.PASS)
    first.measure("first launch", 123.4, "a note")

    second = harness.Report(path)
    names = [c["name"] for c in second.data["checks"]]
    assert "a check from an earlier sitting" in names
    assert second.data["measurements"]["first launch"]["seconds"] == 123.4


def test_a_corrupt_report_does_not_stop_the_run(tmp_path):
    path = tmp_path / "validation_report.json"
    path.write_text("{not json", encoding="utf-8")
    report = harness.Report(path)
    report.check("stage", "still works", harness.PASS)
    assert json.loads(path.read_text(encoding="utf-8"))["checks"]


# --- On anything but Windows, nothing is claimed ---------------------------------------

def test_require_windows_records_every_check_by_name(report, monkeypatch):
    monkeypatch.setattr(harness, "on_windows", lambda: False)
    names = ["one", "two", "three"]
    assert harness.require_windows(report, "stage", names) is False
    for name in names:
        row = next(c for c in report.data["checks"] if c["name"] == name)
        assert row["verdict"] == harness.NOT_RUN
        assert "not Windows" in row["detail"]


def test_require_windows_lets_a_windows_run_through(report, monkeypatch):
    monkeypatch.setattr(harness, "on_windows", lambda: True)
    assert harness.require_windows(report, "stage", ["one"]) is True
    assert report.data["checks"] == [], "it recorded a verdict on Windows"


@pytest.mark.parametrize("stage", ["build", "first", "repair", "warm",
                                  "processes"])
def test_every_windows_stage_refuses_to_claim_anything_here(report, stage, monkeypatch):
    """Run on Linux, these stages must produce NOT RUN rows and no passes."""
    monkeypatch.setattr(harness, "on_windows", lambda: False)
    harness.STAGES[stage](report)
    verdicts = {c["verdict"] for c in report.data["checks"]}
    assert verdicts, f"the {stage} stage recorded nothing at all"
    assert verdicts == {harness.NOT_RUN}, \
        f"the {stage} stage claimed {verdicts - {harness.NOT_RUN}} without Windows"


def test_the_whole_run_fails_on_a_non_windows_machine(tmp_path, monkeypatch, capsys):
    """The property that matters most: it does not go green by skipping everything."""
    monkeypatch.setattr(harness, "REPORT", tmp_path / "validation_report.json")
    monkeypatch.setattr(harness, "ROOT", ROOT)
    monkeypatch.setattr(harness, "on_windows", lambda: False)

    code = harness.main(["preflight", "build", "shortcuts", "report", "--fresh"])
    assert code != 0, "a run that executed nothing reported success"
    printed = capsys.readouterr().out
    assert "NOT WINDOWS" in printed
    assert "NOT a validated release" in printed


# --- The checks that do not need Windows still run -------------------------------------

def test_the_icon_asset_count_is_checked_anywhere(report, monkeypatch):
    """Counting .ico files is reading a directory, so it is answered on any machine -
    and it is the check that says no competing icon was introduced."""
    monkeypatch.setattr(harness, "on_windows", lambda: False)
    monkeypatch.setattr(harness, "ROOT", ROOT)
    harness.stage_shortcuts(report)
    row = next(c for c in report.data["checks"]
               if c["name"] == "no duplicate icon asset exists")
    assert row["verdict"] == harness.PASS, row["detail"]
    assert "leti.ico" in row["detail"]


def test_the_manual_checklist_covers_the_release_criteria():
    """Sections 13 and 20 name things only a person can confirm. Each one has a row
    so that none of them can be quietly assumed."""
    named = " ".join(name for name, _ in harness.MANUAL_CHECKS).lower()
    for required in ("icon", "microphone", "spoken", "stop", "transcript", "image",
                     "graph", "mathematics", "task", "computer use", "verification",
                     "exits cleanly", "console"):
        assert required in named, f"the manual checklist says nothing about {required}"


def test_every_manual_check_says_how_to_do_it():
    for name, how in harness.MANUAL_CHECKS:
        assert len(how) > 20, f"{name!r} has no instruction a person could follow"


def test_the_harness_needs_nothing_installed():
    """It runs on a machine that has not prepared Leti's environment yet."""
    import ast

    tree = ast.parse((ROOT / "scripts" / "validate_windows_release.py")
                     .read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"launcher"}
    assert third_party == set(), f"the harness needs {third_party} installed first"


def test_the_harness_does_not_import_leti():
    """It drives the launchers from outside, the way a user does. Importing Leti
    would mean the harness needed the environment it is supposed to be testing."""
    import ast

    tree = ast.parse((ROOT / "scripts" / "validate_windows_release.py")
                     .read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        for name in ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom)
                     else []):
            assert not name.startswith(("core", "tools", "gui", "audio", "memory")), \
                f"the harness imports {name}"


def test_the_generated_report_is_not_committed():
    """It describes one machine's run, and the next run replaces it."""
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "validation_report.json" in ignored
    assert "validation_report.txt" in ignored
