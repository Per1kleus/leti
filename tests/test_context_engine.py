"""What the context engine chooses, and what it refuses to choose.

The guarantees under test are the ones that make a chosen context safer than an
accumulated one: it never drops history a request points back at, it never
fetches something that lives on a network, it says what it left out, and a
source that fails cannot fail the turn.
"""
from __future__ import annotations

import asyncio

import pytest

from core import context_engine
from core import intent as intent_reader

SYSTEM = "You are Leti."


def _request(text, **kwargs):
    return context_engine.Request(text=text, intent=intent_reader.read(text), **kwargs)


def _history(pairs=10):
    out = []
    for i in range(pairs):
        out.append({"role": "user", "content": f"question {i}"})
        out.append({"role": "assistant", "content": f"answer {i}"})
    return out


def _build(request, history=None, **kwargs):
    return asyncio.run(context_engine.build(
        request, system_prompt=SYSTEM, history=history if history is not None else [],
        **kwargs))


# --- What gets chosen ---------------------------------------------------------------

def test_a_self_contained_request_does_not_carry_the_whole_conversation():
    history = _history(10)
    package = _build(_request("convert 40 psi to bar"), history)
    carried = [m for m in package.messages if m["role"] in ("user", "assistant")]
    assert len(carried) < len(history)
    assert any(name == "older conversation" for name, _ in package.skipped)


def test_a_request_that_points_backwards_keeps_every_message():
    history = _history(10)
    package = _build(_request("compare the first three"), history)
    carried = [m for m in package.messages if m["role"] in ("user", "assistant")]
    assert len(carried) == len(history)


@pytest.mark.parametrize("text", [
    "do it", "why?", "yes", "the second one", "and now the other one",
    "keep going", "you said something about that earlier",
])
def test_short_and_referring_requests_always_keep_history(text):
    """The asymmetry that matters: history kept needlessly costs tokens, history
    dropped wrongly costs the conversation."""
    assert context_engine.refers_backwards(_request(text)) is True


def test_history_is_never_trimmed_below_the_floor():
    history = _history(3)          # six messages, under the floor
    package = _build(_request("convert 40 psi to bar"), history)
    carried = [m for m in package.messages if m["role"] in ("user", "assistant")]
    assert len(carried) == len(history)


def test_the_system_prompt_is_always_first_and_always_there():
    package = _build(_request("hello"), _history(2))
    assert package.messages[0] == {"role": "system", "content": SYSTEM}


# --- What is never fetched ----------------------------------------------------------

def test_nothing_on_a_network_is_fetched_only_named():
    package = _build(_request("what's in my calendar tomorrow?"))
    named = {n["source"] for n in package.needs_tools}
    assert "calendar" in named
    # Named, not fetched: no source called "calendar" is in the assembled context.
    assert "calendar" not in package.included


def test_a_web_shaped_request_names_the_web_rather_than_reading_it():
    package = _build(_request("search for the latest news on interest rates"))
    assert any(n["source"] == "the web" for n in package.needs_tools)


def test_long_term_memory_is_not_searched_for_an_impersonal_request():
    searched = []

    async def recall(text):
        searched.append(text)
        return []

    _build(_request("convert 40 psi to bar"), recall=recall)
    assert searched == []


def test_long_term_memory_is_searched_when_the_request_is_personal():
    searched = []

    async def recall(text):
        searched.append(text)
        return [{"text": "prefers metric units"}]

    package = _build(_request("what do you remember about my preferences?"), recall=recall)
    assert searched and "long-term memory" in package.included


def test_memory_is_skipped_under_resource_pressure():
    searched = []

    async def recall(text):
        searched.append(text)
        return []

    _build(_request("remind me what I prefer", allow_optional=False), recall=recall)
    assert searched == []


# --- Honesty about what was left out ------------------------------------------------

def test_every_source_is_either_included_or_explained():
    package = _build(_request("write a report about the project"), _history(4))
    accounted = set(package.included) | {name for name, _ in package.skipped}
    for name, _, _, _ in context_engine.SOURCES:
        assert name in accounted, f"'{name}' was neither used nor explained"


def test_a_source_that_raises_is_reported_not_swallowed(monkeypatch):
    def explode(request):
        raise RuntimeError("the store is on fire")

    monkeypatch.setattr(context_engine, "SOURCES",
                        [("exploding source", lambda r: True, explode, 3)])
    package = _build(_request("hello"))
    assert any(name == "exploding source" and "on fire" in why
               for name, why in package.skipped)


def test_a_source_that_raises_does_not_fail_the_turn(monkeypatch):
    def explode(request):
        raise RuntimeError("nope")

    monkeypatch.setattr(context_engine, "SOURCES",
                        [("exploding source", lambda r: True, explode, 3)])
    package = _build(_request("hello"), _history(2))
    assert package.messages[0]["content"] == SYSTEM
    assert len([m for m in package.messages if m["role"] == "user"]) == 2


def test_a_failing_recall_is_reported_and_the_package_still_builds():
    async def recall(text):
        raise RuntimeError("the vector store is unavailable")

    package = _build(_request("what do you remember about me?"), recall=recall)
    assert any(name == "long-term memory" and "unavailable" in why
               for name, why in package.skipped)
    assert package.messages


# --- The budget ---------------------------------------------------------------------

def test_what_does_not_fit_is_dropped_by_name_not_silently(monkeypatch):
    monkeypatch.setattr(context_engine, "SOURCES", [
        ("huge", lambda r: True, lambda r: "x" * 5000, 5),
        ("also huge", lambda r: True, lambda r: "y" * 5000, 5),
    ])
    package = _build(_request("hello"), budget_chars=len(SYSTEM) + 5200)
    assert len(package.dropped) == 1
    assert package.dropped[0].startswith(("huge", "also huge"))
    assert package.chars <= len(SYSTEM) + 5200


def test_priority_one_context_is_never_dropped_for_budget(monkeypatch):
    monkeypatch.setattr(context_engine, "SOURCES", [
        ("essential", lambda r: True, lambda r: "e" * 4000, 1),
        ("optional", lambda r: True, lambda r: "o" * 4000, 5),
    ])
    package = _build(_request("hello"), budget_chars=len(SYSTEM) + 100)
    assert "essential" in package.included
    assert any(d.startswith("optional") for d in package.dropped)


def test_the_report_says_how_much_of_the_budget_was_used():
    report = _build(_request("hello"), _history(2)).report()
    assert report["chars"] > 0 and report["budget_chars"] == context_engine.BUDGET_CHARS
    assert report["seconds"] >= 0


# --- Cost ---------------------------------------------------------------------------

def test_assembling_context_is_not_where_a_turn_goes():
    """A budget this module cannot exceed without something being visibly wrong."""
    request = _request("what files are in this folder?")
    history = _history(10)
    _build(request, history)                     # warm the imports
    import time

    started = time.perf_counter()
    for _ in range(20):
        _build(request, history)
    per_call = (time.perf_counter() - started) / 20
    assert per_call < 0.05, f"{per_call * 1000:.1f} ms per context assembly"


def test_no_background_work_is_started_by_assembling_context():
    import threading

    before = threading.active_count()
    _build(_request("write a report about the project"), _history(6))
    assert threading.active_count() == before
