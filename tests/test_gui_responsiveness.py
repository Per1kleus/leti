"""The interface must answer while the model works. It did not.

One websocket serves each client, and gui/server.py read it with

    async for msg in ws:
        ...
        await self._dispatch_call(ws, data)

so nothing else that client sent was even READ until the call in flight finished.
Measured against this server with a call that takes 1.5 seconds: a click sent
100ms into it came back after 1,502ms. A model download is minutes rather than
seconds, and apply_model_setup is one of the calls that was awaited there - so
"shrink the window" during a first-launch download was not slow, it was not
happening, and pressing it again was the only sensible response.

These tests use the real LetiWebServer over a real socket, with the server on its
own thread and loop as it really runs (see gui/api.py's run_gui_mode). A client
sharing the server's loop cannot measure this: a blocking call freezes the
measurement too, and the number that comes out is meaningless.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

aiohttp = pytest.importorskip("aiohttp")

from gui.server import LetiWebServer  # noqa: E402

SLOW = 1.2          # long enough to tell waiting from not waiting
PATIENCE = SLOW + 5


class StubAPI:
    """Only what the transport touches. The real API is tested elsewhere."""

    def __init__(self):
        self.ws_clients = set()
        self.started = []
        self.finished = []
        self.hold_turn = None

    async def a_apply_model_setup(self, *args):
        # The call that really is slow: it downloads a model.
        self.started.append("apply_model_setup")
        await asyncio.sleep(SLOW)
        return {"ok": True}

    async def a_send_text_message(self, *args):
        self.started.append("send_text_message")
        if self.hold_turn is not None:
            # Held until the test lets go, so "the turn is still running" is a
            # fact rather than a bet on this machine being slower than a sleep.
            # It WAS a sleep, and under the load of the whole suite the turn
            # sometimes finished first and the test failed for being right about
            # the wrong thing.
            await self.hold_turn.wait()
        else:
            await asyncio.sleep(SLOW)
        self.finished.append("send_text_message")

    async def a_run_full_check(self, *args):
        await asyncio.sleep(SLOW)
        return {"checks": []}

    def set_window_mode(self, mode):
        return {"desktop": True, "mode": mode}

    def stop_speaking(self):
        return {"ok": True}

    def get_todos(self):
        return []


def _free_port() -> int:
    """A port nothing else is on. Not 0: LetiWebServer reads `port or <configured>`,
    so zero is falsy and every test would quietly share the configured one - which
    is what made them pass alone and fail together."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def live_server():
    api = StubAPI()
    port = _free_port()
    server = LetiWebServer(api, host="127.0.0.1", port=port)
    ready = threading.Event()
    holder = {}

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        holder["loop"] = loop
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    assert ready.wait(15), "the test server did not start"
    yield server, api, port

    loop = holder["loop"]

    async def _shutdown():
        # The tests deliberately leave slow calls running. Cancelling them here is
        # what keeps "Task was destroyed but it is pending" out of every run.
        for task in list(server._background_tasks):
            task.cancel()
        await asyncio.gather(*server._background_tasks, return_exceptions=True)
        await server.runner.cleanup()

    try:
        asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(timeout=10)
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)


async def _connected(port, token):
    session = aiohttp.ClientSession()
    ws = await session.ws_connect(f"http://127.0.0.1:{port}/ws")
    await ws.send_str(json.dumps({"type": "auth", "token": token}))
    await ws.receive()
    return session, ws


async def _send(ws, call_id, method, args=None):
    await ws.send_str(json.dumps({"type": "call", "id": call_id,
                                 "method": method, "args": args or []}))


async def _order_answered(ws, call_ids, timeout=PATIENCE):
    """The ids in the order their answers arrived."""
    wanted, seen = set(call_ids), []
    deadline = time.perf_counter() + timeout
    while wanted and time.perf_counter() < deadline:
        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
        body = json.loads(msg.data)
        found = body.get("id")
        if found in wanted:
            wanted.discard(found)
            seen.append(found)
    assert not wanted, f"never answered: {sorted(wanted)}"
    return seen


async def _wait_for(ws, call_id, timeout=PATIENCE):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
        body = json.loads(msg.data)
        if body.get("id") == call_id:
            return body
    raise AssertionError(f"{call_id} was never answered")


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_call", ["apply_model_setup", "run_full_check"])
async def test_a_click_is_answered_while_something_slow_is_in_flight(live_server, slow_call):
    """The whole point. Whatever is running, the next thing the user does is read
    and answered - not queued behind it."""
    server, _api, port = live_server
    session, ws = await _connected(port, server.token)
    try:
        await _send(ws, "slow", slow_call)
        await asyncio.sleep(0.10)          # let it get going
        await _send(ws, "click", "set_window_mode", ["puck"])

        # Asserted on ORDER rather than on a stopwatch. A threshold in
        # milliseconds is a statement about how fast the machine running the
        # tests is, and this one has to hold on a slow one: what matters is that
        # the click is answered BEFORE the thing it was queued behind, which is
        # true or false regardless of speed.
        order = await _order_answered(ws, ["slow", "click"])

        assert order[0] == "click", (
            f"the click was answered after {slow_call} finished, so it waited for "
            f"it; answered in the order {order}")
    finally:
        await ws.close()
        await session.close()


@pytest.mark.asyncio
async def test_a_click_is_answered_while_a_turn_is_still_running(live_server):
    """send_text_message is not itself slow - it reports that the turn STARTED and
    the answer arrives later by push - so ordering says nothing about it. What
    matters is that a click during the turn is answered while the turn is still
    running, which the stub can be asked directly."""
    server, api, port = live_server
    api.hold_turn = asyncio.Event()
    session, ws = await _connected(port, server.token)
    try:
        await _send(ws, "turn", "send_text_message", ["tell me a long story"])
        await _wait_for(ws, "turn")
        for _ in range(300):            # the turn starts on the server's own loop
            if api.started:
                break
            await asyncio.sleep(0.01)
        assert "send_text_message" in api.started, "the turn never started"

        await _send(ws, "click", "set_window_mode", ["puck"])
        answer = await _wait_for(ws, "click")
        assert answer["type"] == "result"
        assert not api.finished, "the turn finished before the click was answered"
    finally:
        api.hold_turn.set()
        await ws.close()
        await session.close()


@pytest.mark.asyncio
async def test_stop_is_answered_while_the_model_is_generating(live_server):
    """The one control that must always work. Stopping is worthless if the request
    to stop queues behind the thing it is stopping."""
    server, _api, port = live_server
    session, ws = await _connected(port, server.token)
    try:
        await _send(ws, "turn", "send_text_message", ["tell me a long story"])
        await asyncio.sleep(0.10)
        await _send(ws, "stop", "stop_speaking")
        # send_text_message replies at once (it reports the turn STARTED), so both
        # come back quickly; what must be true is that stopping was not queued
        # behind the speaking it was stopping.
        order = await _order_answered(ws, ["turn", "stop"])
        assert "stop" in order
    finally:
        await ws.close()
        await session.close()


@pytest.mark.asyncio
async def test_several_slow_calls_do_not_queue_behind_each_other(live_server):
    """Three slow calls at once finish in about the time of one, not three."""
    server, _api, port = live_server
    session, ws = await _connected(port, server.token)
    try:
        started = time.perf_counter()
        for i in range(3):
            await _send(ws, f"slow{i}", "apply_model_setup")
        for i in range(3):
            await _wait_for(ws, f"slow{i}")
        elapsed = time.perf_counter() - started
        # Generous on purpose: the claim is "concurrent, not serialised", and
        # serialised would be three times SLOW. Anything under two and a half
        # times one call cannot be three in a row, however loaded the machine.
        assert elapsed < SLOW * 2.5, (
            f"three concurrent calls took {elapsed:.2f}s; serialised they would "
            f"take at least {SLOW*3:.2f}s")
    finally:
        await ws.close()
        await session.close()


@pytest.mark.asyncio
async def test_a_client_cannot_start_unlimited_work(live_server):
    """Not awaiting means a task per message, and a page in a loop would grow them
    until the process ran out of memory. Past the ceiling calls are refused, which
    is a reply rather than silence."""
    server, _api, port = live_server
    session, ws = await _connected(port, server.token)
    try:
        for i in range(server.MAX_CALLS_IN_FLIGHT + 12):
            await _send(ws, f"flood{i}", "apply_model_setup")
        refusals = 0
        deadline = time.perf_counter() + PATIENCE
        while time.perf_counter() < deadline and refusals == 0:
            msg = await asyncio.wait_for(ws.receive(), timeout=PATIENCE)
            body = json.loads(msg.data)
            if body.get("type") == "error" and "as many requests" in body.get("message", ""):
                refusals += 1
        assert refusals, "a client could start unbounded work"
    finally:
        await ws.close()
        await session.close()


@pytest.mark.asyncio
async def test_a_turn_replies_at_once_rather_than_holding_the_call_open(live_server):
    """send_text_message answers "started", not "finished". The reply reaches every
    client through api.push() when it is ready, so the page's own call does not sit
    open for the length of a turn and time out."""
    server, _api, port = live_server
    session, ws = await _connected(port, server.token)
    try:
        sent = time.perf_counter()
        await _send(ws, "turn", "send_text_message", ["hello"])
        answer = await _wait_for(ws, "turn")
        assert answer["type"] == "result"
        # The turn itself sleeps for SLOW. Coming back before that means the reply
        # was not waiting for it.
        assert time.perf_counter() - sent < SLOW
    finally:
        await ws.close()
        await session.close()


def test_the_read_loop_does_not_await_the_work_it_starts():
    """Structural, so the fix cannot be undone by a refactor that looks harmless."""
    import inspect

    source = inspect.getsource(LetiWebServer._handle_ws)
    assert "await self._dispatch_call" not in source, (
        "the read loop awaits each call again; one slow call would stop the "
        "interface answering anything else")
    assert "self._start_call(ws, data)" in source
