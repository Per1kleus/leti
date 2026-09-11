"""Which tool results are worth looking at, and - mostly - which are not.

core/artifacts.py decides by shape, with no model call and no per-tool list, so
what these protect is the shape rules themselves and the conservatism around
them. The expensive failure is not a missing panel; it is a window opening over
the interface for something that read perfectly well as a sentence.
"""
from __future__ import annotations

import ast
import inspect
import re
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
    visual = artifacts.derive({"sources_read": [
        {"url": "https://a.com", "title": "A", "snippet": "s", "domain": "a.com"},
        {"url": "https://b.com", "title": "B", "snippet": "s", "domain": "b.com"},
    ]})
    assert visual["type"] == "research"


def test_a_bare_list_of_search_hits_is_not_research():
    """Every web search returns one. It is the raw material for a sentence, not a
    result worth presenting - and treating it as research is most of why a panel
    used to open on "what is the weather"."""
    assert artifacts.derive({"queries": ["weather athens"], "results": [
        {"url": "https://a.com", "title": "A", "snippet": "17C", "domain": "a.com"},
        {"url": "https://b.com", "title": "B", "snippet": "clear", "domain": "b.com"},
    ]}) is None


def test_reading_a_long_file_is_not_a_document():
    """"content", "text" and "answer" are what you asked for, not a document about
    it. Only keys that mean Leti PRODUCED a document count."""
    for key in ("content", "text", "answer"):
        assert artifacts.derive({"path": "/x.txt", key: "x" * 900}) is None
    assert artifacts.derive({"report": "x" * 900})["type"] == "document"


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

# --- Whether it is worth a window --------------------------------------------------
# Recognising a table is not a reason to put one on screen. These are the rules that
# keep the answer to "should this open" as no.

DERIVED = ("table", "research", "document")


@pytest.mark.parametrize("kind", DERIVED)
@pytest.mark.parametrize("text", ["show me the comparison", "draw me a chart of it",
                                  "what does it look like"])
def test_prose_in_a_grid_never_opens_a_window_however_it_was_asked_for(kind, text):
    """A table, a set of sources and a document are all prose in a grid. They are
    offered in the activity log and opened only if you want them - asking to 'see'
    one does not make a panel over the interface the right answer."""
    assert artifacts.should_open({"type": kind}, "search_web", text) is False


def test_a_picture_you_asked_for_opens():
    assert artifacts.should_open({"type": "images"}, "search_images",
                                 "show me pictures of the eiffel tower") is True


def test_a_tool_whose_whole_job_is_visual_has_already_decided():
    """Calling visualize_dataset IS the decision; nothing else has to agree."""
    for tool in artifacts.DELIBERATE_VISUAL_TOOLS:
        assert artifacts.should_open({"type": "images"}, tool, "how did revenue do") is True
    assert artifacts.should_open({"type": "diagram"}, "create_sketch", "explain the flow") is True


def test_images_that_came_along_for_the_ride_do_not_open():
    """A social feed carries thumbnails. Asking to read a feed is not asking to
    look at pictures."""
    assert artifacts.should_open({"type": "images"}, "get_social_content",
                                 "what's new on r/python") is False
    assert artifacts.should_open({"type": "images"}, "get_social_content",
                                 "show me what's new on r/python") is True


def test_a_malformed_call_means_offer_not_window():
    """No name is the safe way round."""
    assert artifacts.should_open({"type": "images"}, "", "how did revenue do") is False
    assert artifacts.should_open(None, "search_images", "show me") is False


@pytest.mark.parametrize("text", [
    "what's the weather in athens?",
    "summarise my unread email",
    "who is in my contacts",
    "imagine a world without traffic",        # not "image"
    "any graphic design tips?",               # not "graph"
    "read me that file",
])
def test_ordinary_questions_do_not_ask_to_see_anything(text):
    assert artifacts.asked_to_see_something(text) is False


def test_the_word_list_can_be_wrong_and_it_costs_nothing_when_it_is():
    """"how many plots of land" reads as asking to see a plot. It does not matter:
    a word match alone opens nothing - the result still has to BE a picture, a
    chart or a diagram, and a question about land is not going to produce one."""
    assert artifacts.asked_to_see_something("how many plots of land do we own") is True
    assert artifacts.should_open({"type": "table"}, "search_web",
                                 "how many plots of land do we own") is False


def test_the_word_list_matches_words_not_substrings():
    """"imagine" is not "image" and "graphic" is not "graph"."""
    assert artifacts.asked_to_see_something("imagine that") is False
    assert artifacts.asked_to_see_something("graphical interface design") is False
    assert artifacts.asked_to_see_something("show me an image") is True


def test_punctuation_cannot_hide_a_phrase():
    assert artifacts.asked_to_see_something("Show me, please.") is True
    assert artifacts.asked_to_see_something("what does   it    look like?") is True


def test_the_deliberate_tools_are_real_tools():
    """A name that stops being a tool would silently stop opening its own charts."""
    from unittest.mock import MagicMock

    import main

    registry = main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())
    missing = artifacts.DELIBERATE_VISUAL_TOOLS - set(registry.names())
    assert not missing, missing


def test_every_tool_that_makes_its_own_visual_is_accounted_for():
    """A new tool that sets output["visual"] is either one whose job is visual, or
    one whose images came along for the ride. Either is fine; being neither means
    nobody decided."""
    setters = set()
    for path in (PROJECT_ROOT / "tools").glob("*.py"):
        source = path.read_text()
        if '"visual"' not in source:
            continue
        setters |= set(re.findall(r'name = "(\w+)"', source))
    # The modules that set a visual, and what each of their tools is for.
    incidental = {"get_social_content", "search_social", "add_social_watch",
                  "list_social_watches", "remove_social_watch", "check_social_watches",
                  "inspect_dataset", "clean_dataset", "export_dataset", "analyze_dataset"}
    unaccounted = setters - artifacts.DELIBERATE_VISUAL_TOOLS - incidental
    assert not unaccounted, (
        f"{sorted(unaccounted)} set their own visual but nothing decided whether "
        "calling them means you asked to see something")


# --- The orchestrator marks every visual with the decision ---------------------------

def test_the_decision_is_made_once_in_the_backend():
    """The page does not get a second opinion: it reads payload.offer and obeys."""
    loop = (PROJECT_ROOT / "core" / "orchestrator.py").read_text()
    assert 'visual["offer"] = not artifacts.should_open(' in loop
    assert "_called_tool_name(call), last_user_message(messages)" in loop
    hud = (PROJECT_ROOT / "gui" / "hud.html").read_text()
    assert "if(payload.offer){ offerArtifact(payload); return; }" in hud
    # Comments stripped: they explain where the decision IS made, and say its name.
    code = re.sub(r"/\*.*?\*/|<!--.*?-->", " ", hud, flags=re.S)
    code = re.sub(r"(?<![:'\"])//[^\n]*", " ", code)
    assert "should_open" not in code, "the page is re-deciding"
    for word in artifacts.DELIBERATE_VISUAL_TOOLS:
        assert word not in code, "the page has its own copy of the rule"


def test_reading_the_name_for_presentation_is_not_a_second_parser():
    """_execute_tool_call still does the real normalisation and is still the only
    thing that decides what runs."""
    from core import orchestrator

    source = inspect.getsource(orchestrator._called_tool_name)
    assert "_normalize_arguments" not in source
    assert "arguments" not in source
    assert orchestrator._called_tool_name({"function": {"name": "plot"}}) == "plot"
    for junk in (None, {}, {"function": None}, {"function": {"name": 7}}, "nope"):
        assert orchestrator._called_tool_name(junk) == ""
