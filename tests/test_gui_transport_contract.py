"""The GUI transport has to fail, not hang.

Every control in the interface is a client of one WebSocket. The promise a
control awaits used to have no way of ever being rejected: a call issued while
the socket was down went onto a queue that was only ever flushed by an auth_ok
that might never come, and a call already in flight when the socket closed was
simply forgotten. Neither settled, so the handler awaiting it never resumed -
and these handlers disable their own button and write "Saving..." first.

The result was an interface where every button was dead, in silence, with the
backend working perfectly. That is the failure this file exists to prevent.
"""
from __future__ import annotations

import collections
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HUD = (ROOT / "gui" / "hud.html").read_text(encoding="utf-8")


def _scripts():
    return re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", HUD, re.S)


# --- A call can always end -------------------------------------------------------

def test_a_closing_socket_fails_every_call_waiting_on_it():
    """A call can only be answered by the socket it was sent on."""
    assert "function failEveryWaitingCall" in HUD
    onclose = HUD[HUD.index("ws.onclose"):HUD.index("ws.onerror")]
    assert "failEveryWaitingCall" in onclose, \
        "closing the socket no longer fails the calls waiting on it"


def test_every_pending_call_is_rejected_not_just_dropped():
    body = HUD[HUD.index("function failEveryWaitingCall"):]
    body = body[:body.index("\n  }")]
    assert "pendingCalls" in body and "readyQueue" in body, \
        "one of the two places a call can be waiting is not cleared"
    assert body.count("reject") >= 2, "a waiting call is dropped rather than rejected"


def test_a_queued_call_cannot_wait_forever():
    """The queue is what lets a window opened before the socket is up still work.
    It must not also be a way for a call to be pending for the life of the page."""
    assert "CONNECT_WAIT_MS" in HUD
    call = HUD[HUD.index("function callApi("):]
    call = call[:call.index("\n  }")]
    assert "setTimeout" in call, "a queued call has no deadline"
    assert "is not connected" in call, "a timed-out call does not say why"


def test_the_queue_still_exists_for_calls_made_before_auth():
    """Rejecting immediately instead would break every call the page makes while
    it is still connecting, which is most of them at startup."""
    call = HUD[HUD.index("function callApi("):]
    call = call[:call.index("\n  }")]
    assert "readyQueue.push" in call
    assert "flushReadyQueue" in HUD


def test_losing_the_connection_is_visible():
    """A dead socket used to look exactly like a working one."""
    assert "function setConnectionState" in HUD
    assert "status-dot.offline" in HUD, "there is no offline styling for the indicator"
    onclose = HUD[HUD.index("ws.onclose"):HUD.index("ws.onerror")]
    assert "setConnectionState(false" in onclose
    assert "setConnectionState(true)" in HUD, "reconnecting never clears the warning"


# --- Backends that RETURN their failure -------------------------------------------

def test_the_permissions_panel_shows_an_error_it_was_handed():
    """a_get_permissions returns {"error": ...} rather than raising, so a try/catch
    never saw it and the panel rendered its headings over empty lists - a
    permissions manager with nothing to manage and no explanation."""
    body = HUD[HUD.index("async function openPermissions"):]
    body = body[:body.index("document.getElementById('permissionsBtn')")]
    assert "data.error" in body, "a returned error is still ignored"
    # The HUD escapes the apostrophe, so match the part that is plain ASCII.
    assert "read permissions: ' + data.error" in body, "the error is not shown"


@pytest.mark.parametrize("panel,marker", [
    ("weather", "data.reason"),
    ("tasks", "tasks.error"),
])
def test_the_other_panels_check_what_they_were_handed(panel, marker):
    assert marker in HUD, f"the {panel} panel ignores what the backend returned"


# --- One name, one function -------------------------------------------------------

def test_no_function_is_declared_twice_in_one_scope():
    """Two declarations with one name is not an error in JavaScript - the later
    one silently wins. renderConnections was declared twice: once returning the
    diagnostics summary, once populating the Connections overlay. Every call
    reached the second, so the diagnostics panel printed "undefined" where its
    connections summary should have been and redrew an unrelated overlay as a
    side effect of being opened.
    """
    for index, script in enumerate(_scripts()):
        names = collections.Counter(re.findall(r"^  function ([A-Za-z_]\w*)\s*\(",
                                               script, re.M))
        duplicates = {n: c for n, c in names.items() if c > 1}
        assert not duplicates, f"script block {index} declares {duplicates} more than once"


def test_the_diagnostics_summary_and_the_panel_are_different_functions():
    assert "function renderConnectionsSummary(c)" in HUD
    assert "renderConnectionsSummary(d.connections)" in HUD
    assert "function renderConnections()" in HUD, "the panel renderer is gone"


# --- The page still parses --------------------------------------------------------

def test_every_script_block_is_valid_javascript(tmp_path):
    for index, script in enumerate(_scripts()):
        path = tmp_path / f"block{index}.js"
        path.write_text(script, encoding="utf-8")
        done = subprocess.run(["node", "--check", str(path)],
                              capture_output=True, text=True)
        assert done.returncode == 0, f"block {index}: {done.stderr[:400]}"
