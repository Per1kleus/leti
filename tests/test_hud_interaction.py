"""Minimising and reopening, driven in a real browser.

The reported bug was that clicking the minimised Leti needed several clicks and
took a while. Source tests can pin that the code no longer awaits the backend;
only a browser can show that one click, aimed where a person aims, changes the
interface - and does it while the backend is still thinking.

Skipped when there is no Chromium to drive. The structural tests in
tests/test_gui_bridge.py cover the same guarantees from the source and always
run, so a machine without a browser is not a machine without the check.
"""
from __future__ import annotations

import os
import pathlib
import threading

import pytest

HUD = pathlib.Path(__file__).resolve().parent.parent / "gui" / "hud.html"

pytest.importorskip("playwright.sync_api",
                    reason="playwright is not installed")

from playwright.sync_api import sync_playwright  # noqa: E402


def _chromium() -> str:
    """Where the browser is, or "" to let playwright find its own."""
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if root:
        found = sorted(pathlib.Path(root).glob("chromium-*/chrome-linux/chrome"))
        if found:
            return str(found[-1])
    return ""


@pytest.fixture(scope="module")
def page_url():
    """The HUD, served over HTTP.

    Not a file:// URL and not set_content: the page reads ?puck=1 and the
    minimised layout from the query string before it renders anything, so it has
    to be fetched from somewhere that has one.
    """
    import http.server
    import socket

    body = HUD.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    executable = _chromium()
    with sync_playwright() as play:
        try:
            launched = play.chromium.launch(
                args=["--no-sandbox"],
                **({"executable_path": executable} if executable else {}))
        except Exception as e:                      # pragma: no cover
            pytest.skip(f"no chromium to drive: {e}")
        yield launched
        launched.close()


SLOW_BACKEND = """() => {
  window.__calls = [];
  window.callApi = (method, ...args) => {
    window.__calls.push(method);
    // Two seconds. The interface must have changed long before this settles.
    return new Promise(r => setTimeout(() => r({desktop:false, mode:args[0]}), 2000));
  };
}"""


def _open(browser, url, width=900, height=620):
    page = browser.new_page(viewport={"width": width, "height": height})
    page.goto(url)
    page.wait_for_timeout(700)
    page.evaluate(SLOW_BACKEND)
    return page


def _minimized(page) -> bool:
    return page.evaluate(
        "() => document.getElementById('leti-root').classList.contains('minimized')")


def test_one_click_minimises_without_waiting_for_the_backend(browser, page_url):
    page = _open(browser, page_url)
    try:
        assert not _minimized(page)
        page.click("#minimizeBtn")
        # 1500ms is well inside the backend's 2000ms, so passing this means the
        # page did not wait for it.
        page.wait_for_function(
            "() => document.getElementById('leti-root').classList.contains('minimized')",
            timeout=1500)
        assert _minimized(page)
    finally:
        page.close()


@pytest.mark.parametrize("aim", ["wordmark", "edge", "centre"])
def test_one_click_anywhere_on_the_puck_reopens_leti(browser, page_url, aim):
    """The whole puck is the target. Aiming at the wordmark is what a person does,
    and the edge is what a person does when they miss."""
    page = _open(browser, page_url)
    try:
        page.click("#minimizeBtn")
        page.wait_for_function(
            "() => document.getElementById('leti-root').classList.contains('minimized')",
            timeout=1500)

        spot = page.evaluate("""(aim) => {
            const puck = document.querySelector('.radar').getBoundingClientRect();
            if(aim === 'edge') return {x: puck.x + 8, y: puck.y + puck.height/2};
            if(aim === 'centre') return {x: puck.x + puck.width/2, y: puck.y + puck.height/2};
            const name = document.querySelector('.puck-name').getBoundingClientRect();
            return {x: name.x + name.width/2, y: name.y + name.height/2};
        }""", aim)

        page.mouse.click(spot["x"], spot["y"])
        page.wait_for_function(
            "() => !document.getElementById('leti-root').classList.contains('minimized')",
            timeout=1500)
        assert not _minimized(page)
    finally:
        page.close()


def test_the_puck_is_drawn_where_its_window_actually_is(browser, page_url):
    """It was not. Measured at x=-95, y=-182 in a 190px window - off its own
    corner, drawing nothing - because `inset:auto` came after left and top in the
    rule and reset them."""
    page = browser.new_page(viewport={"width": 190, "height": 190})
    try:
        page.goto(page_url + "?puck=1&alpha=1")
        page.wait_for_timeout(900)
        box = page.evaluate("""() => {
            const r = document.querySelector('.radar').getBoundingClientRect();
            return {x: Math.round(r.x), y: Math.round(r.y),
                    w: Math.round(r.width), h: Math.round(r.height)};
        }""")
        assert box["w"] == box["h"], f"the puck is not square: {box}"
        assert box["w"] > 100, f"the puck collapsed: {box}"
        assert 0 <= box["x"] and 0 <= box["y"], f"the puck is off its own window: {box}"
        assert box["x"] + box["w"] <= 190 and box["y"] + box["h"] <= 190, \
            f"the puck hangs off the far edge: {box}"
    finally:
        page.close()


def test_the_puck_actually_draws_its_core_and_its_ring(browser, page_url):
    """Everything above can be true of a window with nothing in it."""
    page = browser.new_page(viewport={"width": 190, "height": 190})
    try:
        page.goto(page_url + "?puck=1&alpha=1")
        page.wait_for_timeout(900)
        drawn = page.evaluate("""() => {
            const ring = document.querySelector('#coreRing').getBoundingClientRect();
            const orb = document.querySelector('#orbPath');
            const orbBox = orb.getBoundingClientRect();
            return {ring: Math.round(ring.width), orb: Math.round(orbBox.width),
                    d: (orb.getAttribute('d') || '').length,
                    name: (document.querySelector('.puck-name').textContent || '').trim()};
        }""")
        assert drawn["ring"] > 80, f"the ring is not drawn: {drawn}"
        assert drawn["orb"] > 60, f"the core is not drawn: {drawn}"
        assert drawn["d"] > 100, "the core has no path at all"
        assert drawn["name"] == "LETI"
    finally:
        page.close()


def test_nothing_the_puck_draws_runs_off_the_edge_of_its_window(browser, page_url):
    """"No clipping", from the brief. Measured on the rendered boxes rather than
    on the rule, because the rule looked right when the puck was off screen."""
    page = browser.new_page(viewport={"width": 190, "height": 190})
    try:
        page.goto(page_url + "?puck=1&alpha=1")
        page.wait_for_timeout(900)
        overflowing = page.evaluate("""() => {
            const out = [];
            for(const id of ['coreRing', 'orbPath']){
                const r = document.getElementById(id).getBoundingClientRect();
                if(r.x < -1 || r.y < -1 || r.right > 191 || r.bottom > 191){
                    out.push([id, Math.round(r.x), Math.round(r.y),
                              Math.round(r.right), Math.round(r.bottom)]);
                }
            }
            return out;
        }""")
        assert not overflowing, f"drawn outside the window: {overflowing}"
    finally:
        page.close()
