"""Which tool results are worth looking at, and - mostly - which are not.

core/artifacts.py decides by shape, with no model call and no per-tool list, so
what these protect is the shape rules themselves and the conservatism around
them. The expensive failure is not a missing panel; it is a window opening over
the interface for something that read perfectly well as a sentence.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from core import artifacts

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- Tables ----------------------------------------------------------------------

def test_uniform_rows_of_scalars_are_a_table():
    visual = artifacts.derive({"quotes": [
        {"symbol": "TSLA", "price": 250.1, "up": True},
        {"symbol": "F", "price": 12.0, "up": False},
    ]})
    assert visual["type"] == "table"
    assert visual["columns"] == ["Symbol", "Price", "Up"]
    assert visual["rows"] == [["TSLA", "250.1", "yes"], ["F", "12", "no"]]


def test_one_row_is_a_sentence_not_a_table():
    assert artifacts.derive({"rows": [{"symbol": "TSLA", "price": 250.1}]}) is None


def test_a_count_named_rows_is_not_a_table():
    """The rule is the shape, not the key - data analysis returns rows as an int."""
    assert artifacts.derive({"rows": 5, "columns": 3, "duplicate_rows": 0}) is None


def test_a_column_that_is_an_object_in_one_row_is_not_a_column():
    visual = artifacts.derive({"records": [
        {"name": "a", "score": 1, "meta": {"x": 1}},
        {"name": "b", "score": 2, "meta": {"x": 2}},
    ]})
    assert visual["columns"] == ["Name", "Score"]


def test_a_list_of_mixed_types_is_not_a_table():
    assert artifacts.derive({"things": [{"a": 1, "b": 2}, "not a row"]}) is None


def test_a_very_wide_result_is_not_forced_into_a_table():
    wide = {f"c{i}": i for i in range(12)}
    assert artifacts.derive({"records": [wide, dict(wide)]}) is None


def test_a_long_table_is_capped_and_says_so():
    rows = [{"n": i, "label": f"row {i}"} for i in range(200)]
    visual = artifacts.derive({"records": rows})
    assert len(visual["rows"]) == artifacts.TABLE_MAX_ROWS
    assert visual["more"] == 200 - artifacts.TABLE_MAX_ROWS


# --- Research --------------------------------------------------------------------

def test_pages_with_a_title_and_a_url_are_research():
    visual = artifacts.derive({"topic": "Tesla", "sources_read": [
        {"url": "https://a.com/x", "title": "A", "domain": "a.com", "text": "hello  world"},
        {"url": "https://b.com/y", "title": "B", "domain": "b.com", "text": "more"},
    ]})
    assert visual["type"] == "research"
    assert visual["title"] == "Tesla"
    assert [i["domain"] for i in visual["items"]] == ["a.com", "b.com"]


def test_research_wins_over_reading_the_same_list_as_a_table():
    """A list of pages has uniform scalar columns too; a table of urls is worse."""
    visual = artifacts.derive({"results": [
        {"url": "https://a.com", "title": "A", "snippet": "s", "domain": "a.com"},
        {"url": "https://b.com", "title": "B", "snippet": "s", "domain": "b.com"},
    ]})
    assert visual["type"] == "research"


# --- Documents -------------------------------------------------------------------

def test_a_long_report_is_a_document():
    visual = artifacts.derive({"report": "x" * (artifacts.DOCUMENT_MIN_CHARS + 1)})
    assert visual["type"] == "document"


def test_a_short_answer_stays_a_sentence():
    assert artifacts.derive({"answer": "It is 17 degrees in Athens."}) is None


# --- Conservatism ----------------------------------------------------------------

@pytest.mark.parametrize("output", [
    {"ok": True},
    {"path": "/tmp/x.txt", "written": True},
    {"status": "done", "seconds": 1.2},
    {},
    None,
    "a string",
    [1, 2, 3],
])
def test_ordinary_results_open_no_window(output):
    assert artifacts.derive(output) is None


def test_a_malformed_result_is_not_an_exception():
    class Exploding(dict):
        def items(self):
            raise RuntimeError("boom")

    assert artifacts.derive(Exploding(a=1)) is None


def test_the_derivation_makes_no_model_call_and_reads_no_tool_names():
    """No LLM, no classifier, and no list of tools to keep in step with the tools."""
    source = inspect.getsource(artifacts)
    tree = ast.parse(source)
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("core.llm", "tools.")) or m == "ollama" for m in imported), imported
    assert "llm_client" not in source and "reasoning_model" not in source


# --- The orchestrator's hook -----------------------------------------------------

def test_a_tool_that_supplies_its_own_visual_still_wins():
    """image_search, the sketch tool and the chart already say what to show. The
    derivation must never override one of them."""
    loop = (PROJECT_ROOT / "core" / "orchestrator.py").read_text()
    assert 'result.output.get("visual") or artifacts.derive(result.output)' in loop


def test_the_hook_is_still_only_reached_for_successful_results():
    loop = (PROJECT_ROOT / "core" / "orchestrator.py").read_text()
    assert "if result.success and self.visual_callback and isinstance(result.output, dict):" in loop
