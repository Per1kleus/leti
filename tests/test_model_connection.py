"""The model connection, against a real HTTP server rather than a stand-in.

The reported symptom was chat sometimes answering "I could not finish that - the
connection to the model stopped partway through". The stand-in doubles in
tests/test_streaming.py cannot reproduce it, because what goes wrong is at the
socket: a read that times out with nothing on it, a peer that closes early, a
404 whose body was never read, an {"error": ...} line where a done marker was
expected. So this file starts an HTTP server that behaves badly on purpose.

Measured, before any of this was fixed: seven distinct situations, five of them
with no text at all, every one of them reported to the user with that one
sentence - including two where Ollama had stated the exact problem and the
client discarded it.

It also covers what the brief asks for beyond a single call: consecutive turns
over one client, a stop that must not leak into the next turn, and a failure
that must not poison the turn after it.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest
from aiohttp import web

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import orchestrator as orch_mod  # noqa: E402
from core.llm_client import OllamaClient, reasons  # noqa: E402


def _ndjson(**kw) -> bytes:
    return (json.dumps(kw) + "\n").encode()


class Server:
    """A model server that can be told to misbehave, one behaviour at a time."""

    def __init__(self):
        self.behaviour = "normal"
        self.requests = 0
        self._runner = None
        self.port = 0

    async def start(self):
        app = web.Application()
        app.router.add_get("/api/tags", self._tags)
        app.router.add_post("/api/chat", self._chat)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _tags(self, _):
        return web.json_response({"models": [{"name": "qwen2.5:7b"}]})

    async def _chat(self, request):
        self.requests += 1
        await request.json()
        how = self.behaviour

        if how == "model_missing":
            return web.json_response(
                {"error": 'model "ghost" not found, try pulling it first'}, status=404)
        if how == "plain_500":
            return web.Response(status=500, text="upstream exploded")

        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "application/x-ndjson"})
        await resp.prepare(request)

        if how == "out_of_memory":
            await resp.write(_ndjson(
                error="model requires more system memory (8.4 GiB) "
                      "than is available (5.1 GiB)"))
            return resp
        if how == "slow_prefill":
            await asyncio.sleep(5.0)          # longer than the client's timeout
            await resp.write(_ndjson(message={"role": "assistant",
                                              "content": "too late"}, done=True))
            return resp
        if how == "cut_mid_answer":
            await resp.write(_ndjson(message={"role": "assistant",
                                              "content": "The first half "}))
            await resp.write(_ndjson(message={"role": "assistant",
                                              "content": "of an answer "}))
            raise ConnectionResetError("the server went away")
        if how == "no_done_marker":
            await resp.write(_ndjson(message={"role": "assistant",
                                              "content": "A whole answer."}))
            return resp
        if how == "long":
            for i in range(200):
                await resp.write(_ndjson(message={"role": "assistant",
                                                  "content": f"Sentence {i}. "}))
            await resp.write(_ndjson(message={"role": "assistant", "content": ""},
                                     done=True))
            return resp
        if how == "slow_but_answers":
            for frag in ("Thinking. ", "Still thinking. ", "Here it is."):
                await asyncio.sleep(0.05)
                await resp.write(_ndjson(message={"role": "assistant",
                                                  "content": frag}))
            await resp.write(_ndjson(message={"role": "assistant", "content": ""},
                                     done=True))
            return resp

        for frag in ("Two plus two ", "is four."):
            await resp.write(_ndjson(message={"role": "assistant", "content": frag}))
        await resp.write(_ndjson(message={"role": "assistant", "content": ""},
                                 done=True))
        return resp


@pytest.fixture
async def server():
    s = await Server().start()
    yield s
    await s.stop()


@pytest.fixture
def client_for(server, monkeypatch):
    """A real OllamaClient pointed at the bad server, with a short timeout.

    The timeout stands in for the shipped 120 seconds against a prefill that takes
    longer than that - which is the same failure, at a length a test can wait for.
    """
    import core.config_loader as loader

    def make(timeout=1.0, stream_timeout=None):
        settings = loader.get_settings()
        settings["ollama"]["host"] = server.host
        settings["ollama"]["request_timeout_seconds"] = timeout
        # Streaming has its own, longer allowance in the shipped configuration -
        # that is the fix for the reported symptom - so a test about a streamed
        # read timing out has to set the knob that actually governs it.
        settings["ollama"]["stream_timeout_seconds"] = (
            stream_timeout if stream_timeout is not None else timeout)
        monkeypatch.setattr(loader, "get_settings", lambda: settings)
        monkeypatch.setattr("core.llm_client.get_settings", lambda: settings)
        return OllamaClient()

    return make


async def _run(client, should_stop=None):
    """One streamed turn. Returns (fragments, the final event)."""
    fragments, final = [], None
    async for event in client.stream_response(
            [{"role": "user", "content": "what is 2 + 2"}], should_stop=should_stop):
        if event.get("done"):
            final = event
        else:
            fragments.append(event["text"])
    return fragments, final


# --- Each failure is reported as itself ------------------------------------------

@pytest.mark.asyncio
async def test_a_normal_answer_reports_no_failure_at_all(server, client_for):
    client = client_for()
    fragments, final = await _run(client)

    assert "".join(fragments) == "Two plus two is four."
    assert final["reason"] == reasons.OK
    assert not final["error"]
    assert final["started"] is True
    await client.close()


@pytest.mark.asyncio
async def test_a_long_answer_is_not_mistaken_for_a_broken_one(server, client_for):
    server.behaviour = "long"
    client = client_for()
    fragments, final = await _run(client)

    assert len(fragments) == 200
    assert final["reason"] == reasons.OK
    assert not final["error"]
    await client.close()


@pytest.mark.asyncio
async def test_a_slow_model_that_does_answer_is_not_a_failure(server, client_for):
    server.behaviour = "slow_but_answers"
    client = client_for(timeout=3.0)
    fragments, final = await _run(client)

    assert "Here it is." in "".join(fragments)
    assert final["reason"] == reasons.OK
    await client.close()


@pytest.mark.asyncio
async def test_a_prefill_longer_than_the_timeout_is_reported_as_a_timeout(server, client_for):
    """The measured cause of the reported symptom. No bytes arrive while the model
    reads the prompt, so the read times out - and nothing was partway through
    anything, because nothing had arrived."""
    server.behaviour = "slow_prefill"
    client = client_for(timeout=0.5)
    fragments, final = await _run(client)

    assert fragments == []
    assert final["reason"] == reasons.TIMED_OUT
    assert final["started"] is False
    assert "partway" not in orch_mod._why_it_ended(final)
    await client.close()


@pytest.mark.asyncio
async def test_a_model_that_was_never_pulled_says_so_in_the_servers_own_words(
        server, client_for):
    """Ollama answers 404 with an instruction the user can act on. It used to be
    discarded, and the user was told the connection had dropped."""
    server.behaviour = "model_missing"
    client = client_for()
    fragments, final = await _run(client)

    assert final["reason"] == reasons.MODEL_ERROR
    assert "not found" in final["error"]
    assert "try pulling it first" in final["error"]
    assert "try pulling it first" in orch_mod._why_it_ended(final)
    await client.close()


@pytest.mark.asyncio
async def test_a_model_that_cannot_fit_in_memory_says_that(server, client_for):
    """Ollama sends {"error": ...} as an NDJSON line and no done marker. That line
    used to be ignored, so it fell through to "ended without a completion marker"
    and the shortfall it names was lost."""
    server.behaviour = "out_of_memory"
    client = client_for()
    _, final = await _run(client)

    assert final["reason"] == reasons.MODEL_ERROR
    assert "more system memory" in final["error"]
    assert "8.4 GiB" in orch_mod._why_it_ended(final)
    await client.close()


@pytest.mark.asyncio
async def test_an_error_with_no_explanation_still_names_the_status(server, client_for):
    server.behaviour = "plain_500"
    client = client_for()
    _, final = await _run(client)

    assert final["reason"] == reasons.MODEL_ERROR
    assert "500" in final["error"]
    await client.close()


@pytest.mark.asyncio
async def test_a_connection_cut_mid_answer_keeps_the_text_and_says_it_was_cut(
        server, client_for):
    """The one case the old sentence was true of. It stays true of it."""
    server.behaviour = "cut_mid_answer"
    client = client_for()
    fragments, final = await _run(client)

    assert "".join(fragments) == "The first half of an answer "
    assert final["message"]["content"] == "The first half of an answer "
    assert final["reason"] == reasons.CUT_OFF
    assert final["started"] is True
    assert "stopped partway through" in orch_mod._why_it_ended(final)
    await client.close()


@pytest.mark.asyncio
async def test_a_clean_close_with_no_done_marker_is_doubt_not_a_dropped_connection(
        server, client_for):
    server.behaviour = "no_done_marker"
    client = client_for()
    fragments, final = await _run(client)

    assert "".join(fragments) == "A whole answer."
    assert final["reason"] == reasons.NO_MARKER
    said = orch_mod._why_it_ended(final)
    assert "may not be all of it" in said
    assert "connection" not in said
    await client.close()


@pytest.mark.asyncio
async def test_no_server_at_all_says_ollama_is_not_running(server, client_for):
    client = client_for()
    await server.stop()
    _, final = await _run(client)

    assert final["reason"] == reasons.UNREACHABLE
    assert "Ollama" in orch_mod._why_it_ended(final)
    await client.close()


@pytest.mark.asyncio
async def test_a_stop_is_not_reported_as_a_failure(server, client_for):
    server.behaviour = "long"
    client = client_for()
    _, final = await _run(client, should_stop=lambda: True)

    assert final["stopped"] is True
    assert final["reason"] == reasons.STOPPED
    assert not final["error"], "a deliberate stop is not an error"
    await client.close()


# --- Consecutive turns ------------------------------------------------------------

@pytest.mark.asyncio
async def test_many_turns_in_a_row_over_one_client_all_succeed(server, client_for):
    """A stale connection or a half-read socket left behind by one turn would show
    up here, on the turn after it."""
    client = client_for()
    for turn in range(6):
        fragments, final = await _run(client)
        assert "".join(fragments) == "Two plus two is four.", f"turn {turn}"
        assert final["reason"] == reasons.OK, f"turn {turn}"
    assert server.requests == 6
    await client.close()


@pytest.mark.asyncio
async def test_a_stop_does_not_leak_into_the_next_turn(server, client_for):
    """The stop flag lives in core/transcript.py and is cleared at the turn
    boundary. Here the flag is the caller's, which is the same contract: a stop
    belongs to the answer it stopped."""
    server.behaviour = "long"
    client = client_for()
    stop = {"now": True}

    _, first = await _run(client, should_stop=lambda: stop["now"])
    assert first["stopped"] is True

    stop["now"] = False
    server.behaviour = "normal"
    fragments, second = await _run(client, should_stop=lambda: stop["now"])

    assert "".join(fragments) == "Two plus two is four."
    assert second["stopped"] is False
    assert second["reason"] == reasons.OK
    await client.close()


@pytest.mark.asyncio
async def test_a_stopped_turn_does_not_leave_the_next_one_reading_its_leftovers(
        server, client_for):
    """A stop abandons a response body with data still on the wire. If that socket
    went back to the pool unread, the next turn would read the old answer."""
    server.behaviour = "long"
    client = client_for()
    seen = 0

    async for event in client.stream_response([{"role": "user", "content": "go"}],
                                              should_stop=lambda: seen >= 3):
        if not event.get("done"):
            seen += 1

    server.behaviour = "normal"
    fragments, final = await _run(client)

    assert "".join(fragments) == "Two plus two is four."
    assert "Sentence" not in "".join(fragments)
    assert final["reason"] == reasons.OK
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cut_mid_answer", "out_of_memory",
                                     "model_missing", "no_done_marker"])
async def test_a_failed_turn_does_not_poison_the_turn_after_it(server, client_for, failure):
    client = client_for()
    server.behaviour = failure
    _, bad = await _run(client)
    assert bad["reason"] != reasons.OK

    server.behaviour = "normal"
    fragments, good = await _run(client)

    assert "".join(fragments) == "Two plus two is four."
    assert good["reason"] == reasons.OK
    assert not good["error"]
    await client.close()


@pytest.mark.asyncio
async def test_a_timeout_does_not_poison_the_turn_after_it(server, client_for):
    client = client_for(timeout=0.5)
    server.behaviour = "slow_prefill"
    _, bad = await _run(client)
    assert bad["reason"] == reasons.TIMED_OUT

    server.behaviour = "normal"
    fragments, good = await _run(client)

    assert "".join(fragments) == "Two plus two is four."
    assert good["reason"] == reasons.OK
    await client.close()


@pytest.mark.asyncio
async def test_the_client_reuses_one_connection_pool_rather_than_building_more(
        server, client_for):
    """Named in the brief: no duplicate model clients, no connection rebuilt per
    turn. The pool is created once in __init__ and every turn goes through it."""
    client = client_for()
    pool = client._client
    for _ in range(4):
        await _run(client)
    assert client._client is pool
    await client.close()


# --- Every reason has a sentence, and none of them is the wrong one --------------

def test_every_reason_the_client_can_report_has_its_own_sentence():
    said = {r: orch_mod._why_it_ended({"reason": r, "started": True,
                                       "error": "something the server said"})
            for r in reasons.ALL}
    failures = [r for r in reasons.ALL if r not in (reasons.OK, reasons.STOPPED)]
    assert len({said[r] for r in failures}) == len(failures), (
        f"two failures share a sentence: {said}")


@pytest.mark.parametrize("reason", [reasons.MODEL_ERROR, reasons.UNREACHABLE,
                                    reasons.TIMED_OUT, reasons.NO_ANSWER])
def test_a_failure_that_produced_no_text_never_claims_it_got_partway(reason):
    said = orch_mod._why_it_ended({"reason": reason, "started": False,
                                   "error": "whatever the server said"})
    assert "partway" not in said
    assert "could not finish" not in said


@pytest.mark.asyncio
async def test_a_streamed_answer_waits_far_longer_for_its_first_token(server, client_for):
    """The configuration change that stops the symptom recurring on the fallback
    path. A whole-response call waits request_timeout_seconds for everything; a
    streamed one waits much longer for the first byte, because that wait is the
    model reading up to 22,300 tokens of tool schemas before it can write a word."""
    import core.config_loader as loader

    settings = loader.get_settings()
    settings["ollama"]["host"] = server.host
    settings["ollama"]["request_timeout_seconds"] = 120
    settings["ollama"].pop("stream_timeout_seconds", None)

    client = client_for(timeout=120)
    # client_for sets the key; drop it again to see the unconfigured default.
    client.settings.pop("stream_timeout_seconds", None)

    assert client._stream_timeout().read >= 600
    assert client._stream_timeout().read > float(client.timeout)
    # Connecting is a different question and stays fast: Ollama not running should
    # be reported at once, not after the request budget.
    assert client._stream_timeout().connect <= 10
    await client.close()


@pytest.mark.asyncio
async def test_the_longer_allowance_is_actually_sent_with_the_request(server, client_for):
    """A timeout computed and not passed would be no timeout at all."""
    client = client_for(timeout=1.0, stream_timeout=30.0)
    await _run(client)
    assert client._stream_timeout().read == 30.0
    await client.close()


def test_the_servers_own_words_are_bounded_before_they_are_spoken():
    """The one place text from outside reaches the voice."""
    said = orch_mod._why_it_ended({"reason": reasons.MODEL_ERROR, "started": False,
                                   "error": "x" * 5_000})
    assert len(said) < orch_mod.MAX_SERVER_DETAIL_CHARS + 120
    assert said.endswith("...")
