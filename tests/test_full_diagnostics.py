"""The full system check: what it reports, and what it refuses to report.

The single guarantee worth having here is that NOT TESTED is a real answer. A
diagnostics run that says everything is fine because it did not look is worse
than having no diagnostics at all, so most of these exist to make sure that
cannot happen quietly.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

import main
from core import diagnostics
from core import intent as intent_reader


@pytest.fixture(scope="module")
def registry():
    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


@pytest.fixture(scope="module")
def report(registry):
    return diagnostics.full_check(registry)


# --- What it covers -------------------------------------------------------------------

EXPECTED_SUBSYSTEMS = {
    "Core", "Model", "Ollama", "Tool registry", "Permissions", "Context window",
    "Memory", "Project Memory", "Coding Mode", "Business Mode", "Calendar",
    "GitHub", "Voice", "Computer Use", "Scheduler", "Workflows", "Connections",
}


def test_every_subsystem_is_checked(report):
    assert {c["subsystem"] for c in report["checks"]} == EXPECTED_SUBSYSTEMS


def test_every_check_reports_one_of_the_six_states(report):
    for check in report["checks"]:
        assert check["state"] in diagnostics.CHECK_STATES, check
        assert check["detail"], f"{check['subsystem']} reported no detail"


def test_the_overall_verdict_is_the_worst_one(report):
    severity = diagnostics._CHECK_SEVERITY
    worst = max(severity[c["state"]] for c in report["checks"])
    assert severity[report["overall"]] == worst


def test_untested_subsystems_are_listed_separately(report):
    listed = set(report["not_tested"])
    from_checks = {c["subsystem"] for c in report["checks"]
                   if c["state"] in (diagnostics.NOT_TESTED, diagnostics.NOT_CONFIGURED,
                                     diagnostics.NOT_AVAILABLE)}
    assert listed == from_checks


def test_not_tested_is_never_reported_as_healthy(monkeypatch, registry):
    """The ordering that matters: a run where nothing could be tested must not
    summarise as PASS."""
    monkeypatch.setattr(diagnostics, "_check_core",
                        lambda: diagnostics._check("Core", diagnostics.NOT_TESTED, "n/a"))
    report = diagnostics.full_check(registry)
    assert report["overall"] != diagnostics.PASS


def test_the_report_says_how_to_read_it(report):
    assert "NOT TESTED means it did not run" in report["how_to_read"]
    assert "not a pass" in report["how_to_read"]


# --- What it must not do ---------------------------------------------------------------

def test_nothing_is_contacted_by_default(monkeypatch, registry):
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("the default diagnostics run made a network call")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    result = diagnostics.full_check(registry)
    assert result["contacted_anything"] is False


def test_no_user_data_is_written_or_deleted(tmp_path, registry):
    """Every check reads. If one of them starts writing, these mtimes move."""
    import pathlib

    from core.config_loader import resolve_path

    data = resolve_path("./data")
    before = {p: p.stat().st_mtime_ns for p in pathlib.Path(data).rglob("*")
              if p.is_file()} if data.exists() else {}
    diagnostics.full_check(registry)
    after = {p: p.stat().st_mtime_ns for p in pathlib.Path(data).rglob("*")
             if p.is_file()} if data.exists() else {}
    assert before == after


def test_a_check_that_raises_becomes_a_failing_row_not_a_dead_run(monkeypatch, registry):
    def explode():
        raise RuntimeError("the check itself is broken")

    monkeypatch.setattr(diagnostics, "_check_workflows", explode)
    report = diagnostics.full_check(registry)
    row = next(c for c in report["checks"] if c["subsystem"] == "Workflows")
    assert row["state"] == diagnostics.FAIL and "broken" in row["detail"]
    assert len(report["checks"]) == len(EXPECTED_SUBSYSTEMS)


def test_without_a_registry_the_tool_checks_say_not_tested_rather_than_guessing():
    report = diagnostics.full_check(None)
    for name in ("Tool registry", "Permissions", "Coding Mode", "Business Mode"):
        row = next(c for c in report["checks"] if c["subsystem"] == name)
        assert row["state"] == diagnostics.NOT_TESTED, name


def test_it_finishes_quickly_enough_to_be_asked_for_by_voice(registry):
    started = time.perf_counter()
    diagnostics.full_check(registry)
    assert time.perf_counter() - started < 5.0


def test_there_is_no_background_health_monitor():
    """Running it must not leave anything behind - no thread, no timer, no task."""
    import threading

    before = threading.active_count()
    diagnostics.full_check(None)
    assert threading.active_count() == before


def test_the_module_starts_nothing_periodic():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/diagnostics.py").read_text())
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    for forbidden in ("Thread", "Timer", "start", "create_task", "schedule",
                      "set_interval"):
        assert forbidden not in called, f"core/diagnostics.py calls {forbidden}"


# --- Reaching it -------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "run full leti diagnostics", "run a system check", "diagnostics",
    "check your health", "run the self-test", "perform a complete health check",
])
def test_the_command_is_recognised(text):
    assert intent_reader.asks_for_diagnostics(text) is not None


@pytest.mark.parametrize("text", [
    "my email is not working", "how healthy is the business",
    "check the diagnostics panel is showing", "is the system ok",
    "run the tests", "what is my system report",
])
def test_an_ordinary_request_is_not_a_diagnostics_command(text):
    assert intent_reader.asks_for_diagnostics(text) is None


def test_asking_for_a_deep_check_is_the_only_way_to_reach_out():
    assert intent_reader.asks_for_diagnostics("run full diagnostics")["reach_out"] is False
    assert intent_reader.asks_for_diagnostics(
        "run a deep diagnostic check including connections")["reach_out"] is True


def test_recognising_the_command_costs_nothing():
    intent_reader.asks_for_diagnostics("hello")
    started = time.perf_counter()
    for _ in range(10_000):
        intent_reader.asks_for_diagnostics("what's the weather like today?")
    assert (time.perf_counter() - started) / 10_000 < 1e-4


def test_it_is_not_a_tool_because_default_mode_has_no_room(registry):
    """Deliberately a command rather than a tool - see core/intent.py."""
    assert "run_diagnostics" not in registry.names()
    assert "full_check" not in registry.names()


def test_the_text_report_names_what_was_not_tested(registry):
    text = diagnostics.full_check_text(diagnostics.full_check(registry))
    assert "Leti diagnostics" in text and "nothing was changed" in text
    assert "Not actually tested:" in text
