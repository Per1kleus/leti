"""Reading an answer as the model writes it.

Two things matter here and they pull against each other. The stream has to hand
text over EARLY, because the point of it is that Leti starts speaking before the
model has finished - and it has to reconstruct the answer EXACTLY, because a
streamed answer that differs from the same model's whole-response answer is a
bug nobody will find by reading it.

So most of this file is about the second one: a chunk arriving twice, a chunk
arriving empty, a line that is not JSON, a body that stops in the middle. None of
those may add, drop, reorder or corrupt a character of what did arrive.
"""
from __future__ import annotations

import json

import pytest

from core.llm_client import OllamaClient, _assembled, _read_chunk


def _line(content="", done=False, **extra):
    """One NDJSON line in the shape Ollama sends."""
    chunk = {"model": "qwen", "created_at": "now",
             "message": {"role": "assistant", "content": content}, "done": done}
    chunk.update(extra)
    return json.dumps(chunk)


class FakeResponse:
    def __init__(self, lines, fail_after=None):
        self.lines = list(lines)
        self.fail_after = fail_after

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for index, line in enumerate(self.lines):
            if self.fail_after is not None and index >= self.fail_after:
                raise ConnectionError("the connection went away")
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True
        return False


class FakeHTTP:
    """Stands in for the httpx client. Records that the stream was closed."""

    def __init__(self, lines, fail_after=None, explode=None):
        self.lines = lines
        self.fail_after = fail_after
        self.explode = explode
        self.response = None
        self.payloads = []

    def stream(self, method, url, json=None):
        self.payloads.append(json)
        if self.explode is not None:
            raise self.explode
        self.response = FakeResponse(self.lines, self.fail_after)
        return self.response


def _client(lines, fail_after=None, explode=None):
    client = OllamaClient.__new__(OllamaClient)
    client.host = "http://localhost:11434"
    client.timeout = 60
    client._client = FakeHTTP(lines, fail_after, explode)
    return client


async def _collect(client, **kwargs):
    """Every event the stream produces, in order."""
    return [event async for event in client.stream_response([{"role": "user", "content": "hi"}],
                                                            **kwargs)]


def _text_of(events):
    return "".join(e["text"] for e in events if "text" in e)


def _final(events):
    assert events and events[-1].get("done"), "the stream did not finish"
    return events[-1]


# --- The ordinary case -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fragments_arrive_one_at_a_time():
    events = await _collect(_client([_line("The "), _line("answer "), _line("is four."),
                                     _line(done=True)]))
    assert [e["text"] for e in events if "text" in e] == ["The ", "answer ", "is four."]


@pytest.mark.asyncio
async def test_the_answer_is_reconstructed_exactly():
    """The guarantee the whole feature rests on: the same string chat() would
    have returned, for the same model output."""
    pieces = ["The accele", "ration of the ", "object is F over m."]
    events = await _collect(_client([_line(p) for p in pieces] + [_line(done=True)]))
    expected = "".join(pieces)
    assert _text_of(events) == expected
    assert _final(events)["message"]["content"] == expected


@pytest.mark.asyncio
async def test_the_done_marker_ends_the_stream():
    events = await _collect(_client([_line("one"), _line(done=True), _line("never read")]))
    assert _text_of(events) == "one"
    assert _final(events)["stopped"] is False
    assert not _final(events)["error"]


@pytest.mark.asyncio
async def test_the_assembled_message_looks_like_the_whole_response_one():
    events = await _collect(_client([_line("hello"), _line(done=True)]))
    message = _final(events)["message"]
    assert message == {"role": "assistant", "content": "hello"}


@pytest.mark.asyncio
async def test_tool_calls_survive_the_stream():
    call = {"function": {"name": "write_file", "arguments": {"path": "/tmp/x"}}}
    lines = [_line(""), json.dumps({"message": {"role": "assistant", "content": "",
                                                "tool_calls": [call]}, "done": False}),
             _line(done=True)]
    message = _final(await _collect(_client(lines)))["message"]
    assert message["tool_calls"] == [call]


@pytest.mark.asyncio
async def test_exactly_one_request_is_made():
    client = _client([_line("one"), _line(done=True)])
    await _collect(client)
    assert len(client._client.payloads) == 1, "a streamed answer cost more than one call"
    assert client._client.payloads[0]["stream"] is True


@pytest.mark.asyncio
async def test_the_context_window_is_the_configured_one():
    client = _client([_line(done=True)])
    await _collect(client)
    assert client._client.payloads[0]["options"]["num_ctx"] == 28672


# --- Chunks that are not text ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_empty_chunk_yields_nothing():
    events = await _collect(_client([_line(""), _line("real"), _line(""), _line(done=True)]))
    assert _text_of(events) == "real"
    assert len([e for e in events if "text" in e]) == 1


@pytest.mark.asyncio
async def test_a_metadata_only_chunk_yields_nothing():
    lines = [json.dumps({"model": "qwen", "created_at": "now", "done": False}),
             json.dumps({"eval_count": 42, "done": False}),
             _line("text"), _line(done=True)]
    assert _text_of(await _collect(_client(lines))) == "text"


@pytest.mark.asyncio
async def test_a_chunk_with_no_message_at_all_is_skipped():
    lines = [json.dumps({"done": False}), _line("text"), _line(done=True)]
    assert _text_of(await _collect(_client(lines))) == "text"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "   ", "not json at all", "{half an object",
                                 "[]", "null", '"a string"', "42"])
async def test_an_unreadable_line_is_skipped_not_fatal(bad):
    events = await _collect(_client([_line("before "), bad, _line("after"), _line(done=True)]))
    assert _text_of(events) == "before after", "a bad line took the answer with it"


def test_reading_a_chunk_never_raises():
    for bad in ("", "  ", "nope", "{", "[1,2]", "null", "3", None):
        assert _read_chunk(bad) is None or isinstance(_read_chunk(bad), dict)


@pytest.mark.asyncio
async def test_the_final_chunk_repeating_the_whole_answer_is_not_said_twice():
    """Some servers send the complete text again in the done chunk. Appending it
    would speak the answer, then speak it again."""
    whole = "The answer is four."
    lines = [_line("The answer "), _line("is four."), _line(whole, done=True)]
    events = await _collect(_client(lines))
    assert _text_of(events) == whole
    assert _final(events)["message"]["content"] == whole


@pytest.mark.asyncio
async def test_a_final_chunk_with_genuinely_new_text_is_kept():
    """The mirror of the test above: a done chunk that carries the LAST piece
    rather than a repeat must not be dropped."""
    lines = [_line("The answer "), _line("is four.", done=True)]
    events = await _collect(_client(lines))
    assert _text_of(events) == "The answer is four."


# --- Unicode ---------------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    "Το αποτέλεσμα είναι σωστό.",
    "Η επιτάχυνση ισούται με τη δύναμη διά τη μάζα.",
    "温度は摂氏二十度です。",
    "Ωμέγα, άλφα, βήτα και θήτα.",
    "café — naïve — 日本語 — ✓ — 𝔼[X]",
])
async def test_unicode_survives_being_split_anywhere(answer):
    """Split at every character boundary, which is the worst a stream can do."""
    lines = [_line(ch) for ch in answer] + [_line(done=True)]
    events = await _collect(_client(lines))
    assert _text_of(events) == answer
    assert _final(events)["message"]["content"] == answer


@pytest.mark.asyncio
async def test_greek_split_mid_word_is_reassembled():
    lines = [_line("Το αποτέ"), _line("λεσμα εί"), _line("ναι σωστό."), _line(done=True)]
    assert _text_of(await _collect(_client(lines))) == "Το αποτέλεσμα είναι σωστό."


@pytest.mark.asyncio
async def test_an_expression_split_mid_command_is_reassembled():
    lines = [_line(r"The energy is \("), _line(r"E_k = \fr"), _line(r"ac{1}{2}mv^2"),
             _line(r"\)."), _line(done=True)]
    assert _text_of(await _collect(_client(lines))) == \
        r"The energy is \(E_k = \frac{1}{2}mv^2\)."


# --- When it goes wrong ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_connection_that_dies_keeps_what_arrived():
    client = _client([_line("Το αποτέλεσμα "), _line("είναι"), _line("never")], fail_after=2)
    events = await _collect(client)
    assert _text_of(events) == "Το αποτέλεσμα είναι"
    final = _final(events)
    assert final["error"], "a dead connection was reported as a clean finish"
    assert final["message"]["content"] == "Το αποτέλεσμα είναι"


@pytest.mark.asyncio
async def test_a_body_that_ends_without_a_done_marker_says_so():
    events = await _collect(_client([_line("half an answer")]))
    assert _text_of(events) == "half an answer"
    assert "completion marker" in _final(events)["error"]


@pytest.mark.asyncio
async def test_a_request_that_never_connects_is_still_a_finished_stream():
    events = await _collect(_client([], explode=ConnectionError("refused")))
    assert _text_of(events) == ""
    assert _final(events)["error"]
    assert _final(events)["message"]["content"] == ""


@pytest.mark.asyncio
async def test_an_error_is_never_presented_as_a_complete_answer():
    """The rule from the brief: do not pretend the whole answer was generated."""
    events = await _collect(_client([_line("partial")], fail_after=1))
    assert _final(events)["error"]


# --- Cancellation -------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_stop_ends_the_stream_at_the_next_chunk():
    stop = {"now": False}
    lines = [_line("one "), _line("two "), _line("three "), _line("four"), _line(done=True)]
    client = _client(lines)

    seen = []
    async for event in client.stream_response([], should_stop=lambda: stop["now"]):
        if "text" in event:
            seen.append(event["text"])
            if len(seen) == 2:
                stop["now"] = True
        if event.get("done"):
            assert event["stopped"] is True
            break

    assert seen == ["one ", "two "], f"chunks kept being processed after stop: {seen}"


@pytest.mark.asyncio
async def test_a_stop_keeps_what_had_already_arrived():
    stop = {"now": False}
    client = _client([_line("kept "), _line("also kept "), _line("discarded"), _line(done=True)])
    final = None
    seen = 0
    async for event in client.stream_response([], should_stop=lambda: stop["now"]):
        if "text" in event:
            seen += 1
            if seen == 2:
                stop["now"] = True
        if event.get("done"):
            final = event
            break
    assert final["stopped"] is True
    assert final["message"]["content"] == "kept also kept "


@pytest.mark.asyncio
async def test_a_stop_before_the_first_chunk_yields_no_text_at_all():
    client = _client([_line("never said"), _line(done=True)])
    events = [e async for e in client.stream_response([], should_stop=lambda: True)]
    assert _text_of(events) == ""
    assert _final(events)["stopped"] is True


@pytest.mark.asyncio
async def test_the_response_is_closed_when_a_stop_ends_it():
    """Cooperative: leaving the context manager closes the body, which is what
    cancels the generation. No thread to kill, no socket left half-read."""
    client = _client([_line("one"), _line("two"), _line(done=True)])
    async for event in client.stream_response([], should_stop=lambda: True):
        if event.get("done"):
            break
    assert getattr(client._client.response, "closed", False) is True


# --- The whole-response path is untouched -------------------------------------------------------

def test_the_old_call_is_still_there_and_still_not_streaming():
    import inspect

    source = inspect.getsource(OllamaClient.chat)
    assert '"stream": stream' in source, "chat() stopped honouring its stream argument"
    assert "resp.json()" in source, "chat() no longer returns one whole response"


def test_streaming_is_opt_in():
    """Nothing gained a stream by default. A caller that wants fragments asks
    for them by calling a different method."""
    import inspect

    assert "stream: bool = False" in inspect.getsource(OllamaClient.chat)


def test_assembling_joins_in_order_and_once():
    assert _assembled("assistant", ["a", "b", "c"], []) == \
        {"role": "assistant", "content": "abc"}
    assert "tool_calls" not in _assembled("assistant", [], [])
    assert _assembled("assistant", [], [{"x": 1}])["tool_calls"] == [{"x": 1}]


def test_the_client_starts_nothing_to_stream():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/llm_client.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("threading", "multiprocessing", "subprocess", "queue"):
        assert forbidden not in imported, f"core/llm_client.py imports {forbidden}"
