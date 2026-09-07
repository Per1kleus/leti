"""Tests for running, testing and inspecting code.

The property that matters: a run that failed can never look like one that
passed. Everything else here is about not lying to the model about what happened.
"""
from __future__ import annotations

import pathlib

import pytest

from tools.coding import (
    InspectProjectTool, InstallDependencyTool, RunCodeTool, RunTestsTool,
    detect_test_command, find_interpreter, normalize_language,
)


@pytest.mark.parametrize("given,expected", [
    ("py", "python"), ("PYTHON", "python"), ("js", "javascript"),
    ("node", "javascript"), ("ts", "typescript"), ("octave", "matlab"),
    ("sqlite3", "sql"), ("bash", "bash"),
])
def test_language_aliases(given, expected):
    assert normalize_language(given) == expected


@pytest.mark.asyncio
async def test_code_actually_runs(tmp_path):
    result = await RunCodeTool().run(
        language="python", code="print(sum(range(101)))", working_dir=str(tmp_path)
    )
    assert result.success
    assert result.output["stdout"].strip() == "5050"
    assert result.output["exit_code"] == 0


@pytest.mark.asyncio
async def test_failing_code_is_a_failure_with_the_real_error(tmp_path):
    """The whole point of running code rather than assuming it works."""
    result = await RunCodeTool().run(
        language="python", code="raise ValueError('boom')", working_dir=str(tmp_path)
    )
    assert result.success is False
    assert result.output["exit_code"] != 0
    assert "ValueError: boom" in result.output["stderr"]


@pytest.mark.asyncio
async def test_arguments_and_stdin_reach_the_program(tmp_path):
    result = await RunCodeTool().run(
        language="bash", code='echo "arg=$1"; read x; echo "stdin=$x"',
        arguments=["hello"], stdin="world\n", working_dir=str(tmp_path),
    )
    assert "arg=hello" in result.output["stdout"]
    assert "stdin=world" in result.output["stdout"]


@pytest.mark.asyncio
async def test_a_snippet_leaves_no_file_behind(tmp_path):
    await RunCodeTool().run(language="python", code="pass", working_dir=str(tmp_path))
    assert list(tmp_path.glob("leti_run_*")) == []


@pytest.mark.asyncio
async def test_snippet_runs_in_the_working_directory(tmp_path):
    """Relative imports and data files next to the code are the normal case."""
    (tmp_path / "helper.py").write_text("VALUE = 42\n")
    result = await RunCodeTool().run(
        language="python", code="import helper; print(helper.VALUE)", working_dir=str(tmp_path)
    )
    assert result.output["stdout"].strip() == "42"


@pytest.mark.asyncio
async def test_timeout_is_reported_as_a_failure(tmp_path):
    result = await RunCodeTool().run(
        language="python", code="import time; time.sleep(30)",
        timeout_seconds=1, working_dir=str(tmp_path),
    )
    assert result.success is False
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_unknown_language_says_what_is_supported():
    result = await RunCodeTool().run(language="cobol", code="x")
    assert result.success is False
    assert "python" in result.error


@pytest.mark.asyncio
async def test_missing_interpreter_is_reported_not_worked_around(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.coding.shutil.which", lambda name: None)
    result = await RunCodeTool().run(language="python", code="pass", working_dir=str(tmp_path))
    assert result.success is False
    assert "No interpreter" in result.error


@pytest.mark.asyncio
async def test_running_needs_code_or_a_file():
    assert (await RunCodeTool().run(language="python")).success is False


# --- Test detection --------------------------------------------------------------

def test_detects_pytest_from_a_tests_directory(tmp_path):
    (tmp_path / "tests").mkdir()
    assert detect_test_command(tmp_path)["runner"] == "pytest"


def test_package_json_without_a_test_script_is_not_testable(tmp_path):
    """Otherwise 'run the tests' runs npm test and reports a meaningless failure."""
    (tmp_path / "package.json").write_text('{"name": "x"}')
    assert detect_test_command(tmp_path) is None


def test_package_json_with_a_test_script_is(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
    assert detect_test_command(tmp_path)["runner"] == "npm"


def test_undetectable_project_reports_rather_than_guessing(tmp_path):
    assert detect_test_command(tmp_path) is None


@pytest.mark.asyncio
async def test_failing_tests_are_a_failure(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_ok(): assert True\ndef test_bad(): assert False\n"
    )
    result = await RunTestsTool().run(working_dir=str(tmp_path))
    assert result.success is False
    assert result.output["passed"] is False
    assert "failed" in result.output["stdout"]


@pytest.mark.asyncio
async def test_passing_tests_are_a_success(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_ok(): assert True\n")
    result = await RunTestsTool().run(working_dir=str(tmp_path))
    assert result.success is True
    assert result.output["passed"] is True


@pytest.mark.asyncio
async def test_undetectable_tests_say_so(tmp_path):
    result = await RunTestsTool().run(working_dir=str(tmp_path))
    assert result.success is False
    assert "Couldn't tell how" in result.error


# --- Dependencies ----------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("package", ["--index-url=http://evil", "-e /tmp/x"])
async def test_package_names_cannot_smuggle_flags(package):
    """A flag posing as a package name changes what gets installed, and from where."""
    result = await InstallDependencyTool().run(manager="pip", packages=[package])
    assert result.success is False
    assert "can't start with" in result.error


@pytest.mark.asyncio
async def test_unknown_package_manager_is_refused():
    result = await InstallDependencyTool().run(manager="brew", packages=["wget"])
    assert result.success is False


# --- Inspection -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inspect_skips_generated_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x")
    (tmp_path / "node_modules" / "dep").mkdir(parents=True)
    (tmp_path / "node_modules" / "dep" / "index.js").write_text("x")

    result = await InspectProjectTool().run(path=str(tmp_path))
    assert result.output["file_count"] == 1
    assert not any("node_modules" in f for f in result.output["files"])


@pytest.mark.asyncio
async def test_inspect_reads_manifests_and_test_command(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (tmp_path / "tests").mkdir()
    result = await InspectProjectTool().run(path=str(tmp_path))
    assert "pyproject.toml" in result.output["manifests"]
    assert result.output["test_command"]["runner"] == "pytest"
