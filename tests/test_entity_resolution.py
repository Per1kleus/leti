"""Which one they meant - and, more often, refusing to decide.

The rule these all circle is the only one that makes resolution safe: one
supported candidate resolves, several is a question, none is "I cannot find
that". Acting on the wrong customer is worse than asking which one.
"""
from __future__ import annotations

import time

import pytest

from core import entities


@pytest.fixture(autouse=True)
def _no_conversation():
    entities.use_conversation_source(None)
    yield
    entities.use_conversation_source(None)


NOW = 1_700_000_000.0


def ctx(**kwargs):
    kwargs.setdefault("now", NOW)
    return entities.Context(**kwargs)


# --- The safety rule -----------------------------------------------------------------

def test_two_comparable_candidates_are_a_question_not_a_choice():
    outcome = entities.choose([{"name": "Acme Corp"}, {"name": "Acme Holdings"}],
                              "acme", ctx(), kind="lead")
    assert outcome["resolved"] is None
    assert "Acme Corp" in outcome["ask"] and "Acme Holdings" in outcome["ask"]


def test_nothing_matching_says_nothing_matches_rather_than_offering_the_nearest():
    outcome = entities.choose([{"name": "Acme Corp"}, {"name": "Beta Co"}],
                              "Wakanda Industries", ctx(), kind="lead")
    assert outcome["resolved"] is None and outcome["candidates"] == []
    assert "Nothing matches" in outcome["problem"]


def test_the_only_candidate_resolves():
    outcome = entities.choose([{"name": "Acme Corp"}], "acme", ctx())
    assert outcome["resolved"]["name"] == "Acme Corp"


def test_a_bare_reference_with_several_candidates_asks():
    outcome = entities.choose([{"name": "Acme Corp"}, {"name": "Beta Co"}],
                              "the customer", ctx(), kind="lead")
    assert outcome["resolved"] is None and len(outcome["candidates"]) == 2


def test_a_bare_reference_with_one_candidate_resolves():
    outcome = entities.choose([{"name": "Acme Corp"}], "the customer", ctx())
    assert outcome["resolved"]["name"] == "Acme Corp"


def test_a_near_tie_is_never_broken_by_the_margin():
    """One signal apart is not enough - MARGIN exists so that it is not."""
    both = [{"name": "Acme Corp", "stage": "proposal"}, {"name": "Acme Corp Ltd"}]
    outcome = entities.choose(both, "acme corp", ctx(), kind="lead")
    assert outcome["resolved"] is None or outcome["resolved"]["support"] - \
        outcome["candidates"][1]["support"] >= entities.MARGIN


# --- Context is what separates them ---------------------------------------------------

def test_what_was_said_earlier_settles_it():
    context = ctx(conversation=["let's follow up with Acme Holdings about the quote"])
    outcome = entities.choose([{"name": "Acme Corp"}, {"name": "Acme Holdings"}],
                              "acme", context, kind="lead")
    assert outcome["resolved"]["name"] == "Acme Holdings"
    assert any("conversation" in because for because in outcome["resolved"]["because"])


def test_the_open_project_settles_it():
    context = ctx(project="Turbine")
    outcome = entities.choose([{"name": "task one", "project": "Turbine"},
                               {"name": "task two", "project": "Other"}],
                              "the task", context, kind="task")
    assert outcome["resolved"]["name"] == "task one"


def test_an_active_task_naming_it_settles_it():
    context = ctx(task_texts=["chase Beta Co for the signed contract"])
    outcome = entities.choose([{"name": "Beta Co"}, {"name": "Beta Industries"}],
                              "beta", context, kind="lead")
    assert outcome["resolved"]["name"] == "Beta Co"


def test_a_date_in_the_reference_settles_a_meeting():
    context = ctx()
    tomorrow = NOW - (NOW % 86400) + 86400 + 3600
    outcome = entities.choose(
        [{"name": "review", "starts": tomorrow},
         {"name": "review", "starts": tomorrow + 86400 * 5}],
        "the review tomorrow", context, kind="meeting")
    assert outcome["resolved"]["starts"] == tomorrow


def test_a_stage_or_company_word_settles_it():
    outcome = entities.choose([{"name": "Chris", "company": "Northwind"},
                               {"name": "Chris", "company": "Contoso"}],
                              "chris at northwind", ctx(), kind="contact")
    assert outcome["resolved"]["company"] == "Northwind"


def test_a_full_name_match_beats_a_partial_one():
    outcome = entities.choose([{"name": "Beta Co"}, {"name": "Beta Industries Group"}],
                              "Beta Co", ctx(), kind="lead")
    assert outcome["resolved"]["name"] == "Beta Co"


# --- What it says about itself ---------------------------------------------------------

def test_resolution_says_why_it_resolved():
    context = ctx(conversation=["Acme Holdings sent the signed quote"])
    outcome = entities.choose([{"name": "Acme Corp"}, {"name": "Acme Holdings"}],
                              "acme", context, kind="lead")
    assert outcome["how"] and "matches" in outcome["how"]


def test_it_says_when_the_context_it_would_have_used_was_missing():
    outcome = entities.choose([{"name": "A"}, {"name": "B"}], "something", ctx(
        unavailable=["nothing has been said in this conversation yet"]))
    assert outcome["context_missing"]


def test_it_does_not_claim_semantic_resolution():
    explained = entities.explain()
    assert "not semantic" in explained["method"]
    assert "embeddings" in explained["not_used"]
    assert "a language model" in explained["not_used"]


def test_the_module_imports_no_model_and_no_vector_store():
    """The claim in explain() has to be true of the code, not just the docstring."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/entities.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in ("core.llm_client", "memory.vector_store", "ollama",
                      "sentence_transformers", "numpy"):
        assert forbidden not in imported, f"core/entities.py imports {forbidden}"


# --- Gathering context ------------------------------------------------------------------

def test_gather_reads_the_conversation_through_the_reference_it_was_given():
    entities.use_conversation_source(
        lambda: [{"role": "user", "content": "about Acme Holdings"}])
    assert "acme holdings" in entities.gather().conversation_text


def test_a_conversation_source_that_raises_is_absent_not_fatal():
    def explode():
        raise RuntimeError("the buffer is gone")

    entities.use_conversation_source(explode)
    context = entities.gather()
    assert context.conversation == []
    assert context.unavailable


def test_gather_never_raises_and_is_cheap():
    started = time.perf_counter()
    for _ in range(50):
        entities.gather()
    assert (time.perf_counter() - started) / 50 < 0.02


# --- The business resolver still honours all of it ---------------------------------------

def test_the_business_resolver_delegates_the_decision(monkeypatch, tmp_path):
    from core import business

    monkeypatch.setattr(business, "_candidates_of",
                        lambda kind, words, now: (
                            [{"kind": "lead", "name": "Acme Corp", "score": 2},
                             {"kind": "lead", "name": "Acme Holdings", "score": 2}]
                            if kind == "lead" else []))
    outcome = business.resolve("acme", kind="lead", now=NOW)
    assert outcome["resolved"] is None and outcome["ask"]


def test_an_unreadable_source_is_reported_rather_than_read_as_empty(monkeypatch):
    from core import business

    def fail(kind, words, now):
        if kind == "meeting":
            raise RuntimeError("no calendar is connected")
        return []

    monkeypatch.setattr(business, "_candidates_of", fail)
    outcome = business.resolve("the review", now=NOW)
    assert outcome["unavailable"] and "calendar" in outcome["unavailable"][0]
