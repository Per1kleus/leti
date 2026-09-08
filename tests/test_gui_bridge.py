"""The contract between the HUD (gui/hud.html) and the backend behind it.

Both directions are name-addressed across a websocket, so nothing here is checked
by the compiler or by any test that runs only Python: the frontend calls backend
methods by string, and the backend pushes JS function names by string. A rename on
either side produces a dead button or a message that never arrives, silently.

That is exactly what happened when RunHealthCheckTool became SystemReportTool - the
call site in gui/api.py still asked for a key that had moved, the dashboard caught
the error and rendered zeros, and nothing anywhere said so.
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUD = (PROJECT_ROOT / "gui" / "hud.html").read_text()
API = (PROJECT_ROOT / "gui" / "api.py").read_text()
SERVER = (PROJECT_ROOT / "gui" / "server.py").read_text()


def _declared_methods():
    sync = set(re.findall(r"['\"]([a-z_]+)['\"]",
                          re.search(r"SYNC_METHODS\s*=\s*\{(.*?)\}", API, re.S).group(1)))
    asyn = set(re.findall(r"['\"]([a-z_]+)['\"]",
                          re.search(r"ASYNC_METHODS\s*=\s*\{(.*?)\}", API, re.S).group(1)))
    return sync, asyn


def _hud_calls():
    return set(re.findall(r"callApi\(\s*['\"]([A-Za-z_]+)['\"]", HUD))


def _backend_pushes():
    sources = API + (PROJECT_ROOT / "core" / "orchestrator.py").read_text()
    return set(re.findall(r'push\(\s*"([A-Za-z]+)"', sources))


def test_every_method_the_hud_calls_is_declared_and_implemented():
    sync, asyn = _declared_methods()
    for method in sorted(_hud_calls()):
        assert method in sync | asyn, f"hud.html calls '{method}', which gui/api.py doesn't declare"
        handler = f"a_{method}" if method in asyn else method
        assert re.search(rf"def {re.escape(handler)}\b", API), f"no {handler}() in gui/api.py"


def test_no_declared_method_is_unreachable_from_the_hud():
    """A method the interface never calls is either a dead endpoint or a button that
    was renamed on one side only."""
    sync, asyn = _declared_methods()
    assert not (sync | asyn) - _hud_calls()


def test_the_server_routes_every_declared_method():
    sync, asyn = _declared_methods()
    dispatch = re.search(r"async def _dispatch_call.*?(?=\n    async def |\n    def )",
                         SERVER, re.S).group(0)
    for method in sorted(asyn):
        assert f'"{method}"' in dispatch, f"gui/server.py has no branch for async method '{method}'"
    assert "SYNC_METHODS" in dispatch, "sync methods are routed as a group; that route is gone"


def test_every_push_is_allowlisted_and_defined_in_the_page():
    """The HUD dispatches pushes through an allowlist, so a backend push the page
    doesn't know about is dropped with a console warning the user never sees."""
    allowed = set(re.findall(r"['\"]([a-zA-Z]+)['\"]",
                             re.search(r"PUSH_HANDLERS = new Set\(\[(.*?)\]\)", HUD, re.S).group(1)))
    defined = set(re.findall(r"window\.([A-Za-z]+)\s*=\s*function", HUD))
    for fn in sorted(_backend_pushes()):
        assert fn in allowed, f"backend pushes '{fn}', which hud.html's allowlist drops"
        assert fn in defined, f"hud.html allowlists '{fn}' but defines no window.{fn}"


def test_no_allowlisted_push_is_dead():
    assert not set(re.findall(r"['\"]([a-zA-Z]+)['\"]",
                              re.search(r"PUSH_HANDLERS = new Set\(\[(.*?)\]\)", HUD, re.S).group(1))
                   ) - _backend_pushes()


def test_the_dashboard_asks_system_report_for_the_section_it_reads():
    """gui/api.py reads output['health']['metrics']; if it stops asking for the
    health section, or the section is renamed, the dashboard reads a key that isn't
    there and reports zeros rather than failing visibly."""
    stats = re.search(r"async def a_get_system_stats.*?(?=\n    async def |\n    def )",
                      API, re.S).group(0)
    assert 'sections=["health"]' in stats
    assert '"health"' in stats and '"metrics"' in stats
