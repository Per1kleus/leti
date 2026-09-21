"""What the user is doing by saying something, and what must not follow from it.

The expensive failure this layer exists to prevent has one shape: Leti acting on
something that was not a request. A remark, a hedged suggestion, a thought out
loud. Most of these are that failure, written down.
"""
from __future__ import annotations

import time

import pytest

from core import intent as ir


def kind(text):
    return ir.read(text).kind


# --- A remark is not a request --------------------------------------------------------

@pytest.mark.parametrize("text", [
    "That's interesting.", "That is interesting", "this is cool", "Interesting.",
    "makes sense", "fair enough", "I see", "good to know", "huh", "wow",
    "I agree", "noted", "got it", "of course", "that's true", "nice one",
])
def test_a_reaction_is_a_conversation(text):
    assert kind(text) == ir.CONVERSATION


@pytest.mark.parametrize("text", [
    "Maybe we should open the project.",
    "Perhaps we could run the tests",
    "I wonder if we should email Chris",
    "it might be worth deleting that file",
    "I'm thinking about closing the browser",
    "we could probably send it now",
])
def test_a_hedged_suggestion_is_not_an_instruction(text):
    assert kind(text) == ir.CONVERSATION
    assert ir.read(text).wants_action is False


def test_a_hedged_question_is_still_a_question():
    """"Do you think we should open it?" wants an answer, not silence."""
    assert kind("do you think we should open it?") == ir.QUESTION


@pytest.mark.parametrize("text", [
    "That's interesting.", "Maybe we should open the project.", "hello", "I see",
])
def test_nothing_that_asked_for_nothing_may_need_a_tool(text):
    assert ir.read(text).may_need_tool is False
    assert ir.read(text).wants_action is False


# --- ...but a request behind a remark is still a request --------------------------------

@pytest.mark.parametrize("text,expected", [
    ("interesting - now open spotify", ir.COMMAND),
    ("nice, then run the tests", ir.COMMAND),
    ("cool. cancel the download", ir.CANCELLATION),
    ("ok, continue", ir.CONTINUATION),
    ("thanks. email Chris the report", ir.SINGLE_STEP_TASK),
])
def test_an_instruction_behind_a_remark_still_lands(text, expected):
    assert kind(text) == expected
    assert ir.read(text).wants_action is True


# --- The other kinds ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Open the project.", ir.COMMAND),
    ("close the browser", ir.COMMAND),
    ("what is a turbine?", ir.QUESTION),
    ("how does a jet engine work", ir.QUESTION),
    ("what is the price of AAPL?", ir.INFORMATION_REQUEST),
    ("what's the weather like", ir.INFORMATION_REQUEST),
    ("email stavros the report", ir.SINGLE_STEP_TASK),
    ("research five CRMs, compare their pricing and write a summary", ir.MULTI_STEP_TASK),
    ("tell me when the build finishes", ir.WATCH_REQUEST),
    ("notify me when an email from Chris arrives", ir.WATCH_REQUEST),
    ("stop", ir.CANCELLATION),
    ("cancel this", ir.CANCELLATION),
    ("never mind", ir.CANCELLATION),
    ("continue", ir.CONTINUATION),
    ("resume the website task", ir.CONTINUATION),
    ("no, I meant the other one", ir.CORRECTION),
    ("that's not what I asked for", ir.CORRECTION),
    ("what do you mean?", ir.CLARIFICATION),
    ("which one did you pick?", ir.CLARIFICATION),
])
def test_each_kind_is_recognised(text, expected):
    assert kind(text) == expected


def test_every_kind_is_one_of_the_declared_ones():
    for text in ["hi", "open it", "what is x", "stop", "continue", "no I meant that",
                 "what do you mean", "watch the build", "do a and then b", "nonsense"]:
        assert ir.read(text).kind in ir.KINDS


# --- The structured output -----------------------------------------------------------------

def test_the_layer_reports_everything_it_promises():
    reading = ir.read("email the report to chris and then book a meeting").as_dict()
    for field in ("kind", "confidence", "side_effect_requested",
                  "tool_possibly_required", "clarification_required",
                  "belongs_to_existing_task"):
        assert field in reading


def test_an_external_side_effect_is_flagged():
    assert ir.read("email stavros the report").side_effect is True
    assert ir.read("read the report").side_effect is False


def test_a_remark_never_reports_a_side_effect_even_when_it_names_one():
    """"Maybe we should email them" names sending and asks for none."""
    assert ir.read("maybe we should email them").side_effect is False


def test_a_reference_to_running_work_is_flagged():
    assert ir.read("what happened to that task?").about_task is True
    assert ir.read("stop").about_task is True
    assert ir.read("what is a turbine").about_task is False


def test_confidence_is_higher_for_the_unambiguous_ones():
    assert ir.read("stop").confidence >= 0.95
    assert ir.read("hello").confidence >= 0.95
    assert ir.read("the flange is purple").confidence < 0.8


def test_a_parser_failure_fails_open_not_closed(monkeypatch):
    """A bug here must not silently make Leti refuse to act."""
    def explode(*args, **kwargs):
        raise RuntimeError("the reader is broken")

    monkeypatch.setattr(ir, "_read", explode)
    reading = ir.read("open the project")
    assert reading.may_need_tool is True and reading.wants_action is True


# --- Lifecycle commands ---------------------------------------------------------------------

@pytest.mark.parametrize("text,action", [
    ("stop", ir.STOP), ("stop.", ir.STOP), ("please stop", ir.STOP),
    ("cancel this", ir.STOP), ("stop the download", ir.STOP),
    ("pause", ir.PAUSE), ("pause the research task", ir.PAUSE),
    ("resume", ir.RESUME), ("continue", ir.RESUME),
    ("continue the website task", ir.RESUME),
    ("retry", ir.RETRY), ("try again", ir.RETRY),
    ("skip this step", ir.SKIP), ("skip", ir.SKIP),
    ("what are you waiting for?", ir.WAITING),
    ("what are you doing?", ir.STATUS),
    ("show my active tasks", ir.STATUS),
    ("which task needs me?", ir.STATUS),
])
def test_lifecycle_commands_are_recognised(text, action):
    found = ir.lifecycle_command(text)
    assert found is not None and found["action"] == action


@pytest.mark.parametrize("text", [
    "open the project", "stop worrying about it", "don't cancel that",
    "I stopped it myself", "what is the weather", "email chris",
    "pause the music in spotify and then open blender",
])
def test_an_ordinary_request_is_not_a_lifecycle_command(text):
    assert ir.lifecycle_command(text) is None


def test_the_target_is_passed_through_not_resolved():
    assert ir.lifecycle_command("stop the download")["hint"] == "download"
    assert ir.lifecycle_command("continue the website task")["hint"] == "website task"


def test_stopping_everything_says_so():
    assert ir.lifecycle_command("cancel everything")["everything"] is True
    assert ir.lifecycle_command("stop the download")["everything"] is False


def test_a_status_question_carries_no_target():
    assert ir.lifecycle_command("what are you doing?")["hint"] == ""


# --- Cost -------------------------------------------------------------------------------------

def test_reading_a_request_is_not_where_a_turn_goes():
    ir.read("warm up")
    started = time.perf_counter()
    for _ in range(2000):
        ir.read("email stavros the quarterly report and book a follow-up meeting")
    per_call = (time.perf_counter() - started) / 2000
    assert per_call < 0.001, f"{per_call * 1e6:.0f} us per read"


def test_the_layer_makes_no_model_call():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/intent.py").read_text())
    imports = {n.module for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and n.module}
    assert "core.llm_client" not in imports
    assert not any("ollama" in (m or "").lower() for m in imports)


# --- The veto needs positive evidence, not the absence of evidence -----------------------

@pytest.mark.parametrize("text", [
    "remember that I prefer metric units",
    "note that the deadline moved to Friday",
    "calculate 40 psi in bar",
    "translate it to greek",
    "plan the migration",
    "review my leads",
    "test the auth module",
    "verify the build",
    "track the shipment",
    "export it to csv",
])
def test_an_instruction_is_never_silently_disarmed(text):
    """An earlier version inferred "remark" from the absence of a known verb,
    and every one of these came out as a remark with no tools. Withholding
    tools needs evidence that something WAS a remark."""
    assert ir.read(text).wants_action is True


def test_the_fallback_reading_keeps_its_tools():
    """Nothing pointed anywhere. That is not evidence of a remark."""
    reading = ir.read("the flange is purple")
    assert reading.kind == ir.CONVERSATION
    assert reading.confidence < ir.CONFIDENT_CONVERSATION
    assert reading.wants_action is True


@pytest.mark.parametrize("text", ["hello", "That's interesting.", "I see",
                                  "Maybe we should open the project."])
def test_only_a_confident_remark_withholds_tools(text):
    reading = ir.read(text)
    assert reading.kind == ir.CONVERSATION
    assert reading.confidence >= ir.CONFIDENT_CONVERSATION
    assert reading.wants_action is False
