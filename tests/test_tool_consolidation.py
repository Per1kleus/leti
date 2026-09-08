"""Tests for the tools that absorbed other tools.

Several pairs and triples of tools were doing the same job by different names -
five per-platform "get the latest from X" tools, two "search a social platform"
tools, a navigate tool that was a subset of a read tool, two overlapping views of
the same psutil socket list, and three tools that all sampled the same machine.
Each set was folded into one tool that covers everything the originals did.

What these protect is that "folded in" really means folded in: every platform,
state and section the removed tools handled is still reachable, the dispatch is
one code path rather than a chain of special cases, and nothing was quietly
dropped in the merge. They deliberately assert on the merged tools' contracts
rather than on network results, so they run offline.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from core.config_loader import get_permissions

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The tools that were folded into another tool. Naming them keeps a later merge from
# quietly reintroducing one, and keeps this file honest about what it is protecting.
MERGED_AWAY = [
    "get_youtube_channel_latest", "get_subreddit_posts", "get_instagram_user_latest",
    "get_tiktok_user_latest", "get_facebook_page_latest",   # -> get_social_content
    "search_reddit", "search_youtube_trending",             # -> search_social
    "browser_navigate",                                     # -> browser_read_page
    "scan_local_ports", "list_suspicious_processes",        # -> inspect_network_connections
    "get_system_specs", "run_health_check", "check_for_updates",  # -> system_report
    "open_url",                                             # -> launch_app
]


def _registered_tool_names() -> list:
    """Tool names registered in main.py, read statically.

    Importing main.py needs a display and the full dependency set; the question here
    is about what the source registers, which the AST answers on any machine.
    """
    class_to_name = {}
    for path in sorted((PROJECT_ROOT / "tools").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and getattr(stmt.targets[0], "id", None) == "name"
                        and isinstance(stmt.value, ast.Constant)):
                    class_to_name[node.name] = stmt.value.value

    names = []
    for node in ast.walk(ast.parse((PROJECT_ROOT / "main.py").read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register" and node.args
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)):
            cls = node.args[0].func.id
            assert cls in class_to_name, f"main.py registers {cls}, which no tools/ module defines"
            names.append(class_to_name[cls])
    return names


# --- The registry as a whole ---------------------------------------------------

def test_every_registered_tool_has_an_action_class():
    """A tool nobody classified is treated as CRITICAL at runtime - it starts asking
    for confirmation it never needed, or stops working unattended. A merge that
    renames a tool has to rename it in permissions.yaml too."""
    entries = get_permissions()["tools"]
    missing = [n for n in _registered_tool_names() if n not in entries]
    assert not missing, f"registered with no action class: {missing}"


def test_no_permission_entry_outlives_the_tool_it_governed():
    """The other half: an entry for a tool that no longer exists is dead config that
    reads as though the tool is still there."""
    registered = set(_registered_tool_names())
    orphans = [n for n in get_permissions()["tools"] if n not in registered]
    assert not orphans, f"permissions.yaml governs tools nothing registers: {orphans}"


def test_each_tool_is_registered_once():
    names = _registered_tool_names()
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"registered more than once: {duplicates}"


def test_the_merged_away_tools_are_really_gone():
    entries = get_permissions()["tools"]
    registered = set(_registered_tool_names())
    for name in MERGED_AWAY:
        assert name not in entries, f"permissions.yaml still lists the merged-away {name}"
        assert name not in registered, f"main.py still registers the merged-away {name}"


# --- Social: five fetchers and two searchers became two tools ------------------

def test_get_social_content_covers_every_platform_the_old_tools_did():
    from tools.social_media import CONTENT_PLATFORMS, GetSocialContentTool

    for platform in ("youtube", "reddit_subreddit", "reddit_user", "instagram",
                     "tiktok", "facebook", "webpage"):
        assert platform in CONTENT_PLATFORMS, platform

    enum = next(p for p in GetSocialContentTool().parameters if p.name == "platform").enum
    assert set(enum) == set(CONTENT_PLATFORMS)


@pytest.mark.asyncio
async def test_get_social_content_rejects_an_unknown_platform_without_calling_out():
    from tools.social_media import GetSocialContentTool

    result = await GetSocialContentTool().run(platform="myspace", identifier="x")
    assert result.success is False
    assert "myspace" in result.error


@pytest.mark.asyncio
async def test_reddit_specific_options_survived_the_merge():
    """min_score and sort were the reason get_subreddit_posts existed; losing them in
    the merge would make the merged tool a downgrade."""
    from tools.social_media import GetSocialContentTool

    names = {p.name for p in GetSocialContentTool().parameters}
    assert {"sort", "min_score", "limit"} <= names


def test_search_social_kept_both_platforms_scoring_rules():
    from tools.social_media import SEARCHABLE_PLATFORMS, SearchSocialTool

    assert set(SEARCHABLE_PLATFORMS) >= {"reddit", "youtube"}
    names = {p.name for p in SearchSocialTool().parameters}
    # min_score/time_filter came from search_reddit, days from search_youtube_trending.
    assert {"min_score", "time_filter", "days"} <= names


def test_watches_fetch_through_the_same_dispatcher_as_the_tool():
    """The watch system and the tool used to have separate per-platform branches.
    One dispatcher means a platform added in one place works in both."""
    from tools import social_media

    source = inspect.getsource(social_media._fetch_latest_for_watch)
    assert "fetch_platform_content" in source


# --- Network sockets: two views of one psutil call -----------------------------

class _FakePsutil:
    """Enough of psutil to drive _connections without one installed.

    psutil is a real requirement, but the thing under test is the filtering and the
    number of passes, which is about Leti's code rather than the platform's sockets -
    and a real machine has no reliably-present listening port to assert on.
    """

    CONN_LISTEN = "LISTEN"
    CONN_ESTABLISHED = "ESTABLISHED"

    class _Addr:
        def __init__(self, ip, port):
            self.ip, self.port = ip, port

    class _Conn:
        def __init__(self, status, pid, laddr=None, raddr=None, type_=None):
            self.status, self.pid, self.laddr, self.raddr = status, pid, laddr, raddr
            self.type = type_

    def __init__(self, conns):
        self._conns = conns
        self.connection_walks = 0
        self.process_lookups = 0

    def net_connections(self, kind="inet"):
        self.connection_walks += 1
        return self._conns

    def Process(self, pid):  # noqa: N802 - mirrors psutil's own name
        self.process_lookups += 1
        outer = self

        class _P:
            def name(self):
                return f"proc{pid}"

            def exe(self):
                return f"/usr/bin/proc{pid}" if pid != 999 else ""

        return _P()


@pytest.fixture
def fake_psutil(monkeypatch):
    import socket as socket_module
    import sys

    fake = _FakePsutil([
        # A flagged listening port, an unremarkable one, and outbound connections -
        # two of them from the same process, which used to cost two lookups.
        _FakePsutil._Conn("LISTEN", 10, laddr=_FakePsutil._Addr("0.0.0.0", 3389),
                          type_=socket_module.SOCK_STREAM),
        _FakePsutil._Conn("LISTEN", 11, laddr=_FakePsutil._Addr("127.0.0.1", 51234),
                          type_=socket_module.SOCK_DGRAM),
        _FakePsutil._Conn("ESTABLISHED", 20, raddr=_FakePsutil._Addr("93.184.216.34", 443)),
        _FakePsutil._Conn("ESTABLISHED", 20, raddr=_FakePsutil._Addr("93.184.216.35", 443)),
        _FakePsutil._Conn("ESTABLISHED", 999, raddr=_FakePsutil._Addr("10.0.0.9", 4444)),
        # Neither listening nor established: should appear in no view.
        _FakePsutil._Conn("TIME_WAIT", 30, raddr=_FakePsutil._Addr("1.1.1.1", 80)),
    ])
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


def test_both_socket_views_come_from_one_walk(fake_psutil):
    """scan_local_ports and list_suspicious_processes each walked every socket on the
    machine to answer half the question, and paid for the walk separately."""
    from tools.network_security import _connections

    report = _connections("all")
    assert fake_psutil.connection_walks == 1
    assert set(report) == {"listening", "established"}


def test_asking_for_one_view_does_not_collect_the_other(fake_psutil):
    from tools.network_security import _connections

    assert set(_connections("listening")) == {"listening"}
    assert set(_connections("established")) == {"established"}


def test_listening_view_keeps_what_the_port_scan_reported(fake_psutil):
    from tools.network_security import _connections

    listening = _connections("listening")["listening"]
    assert listening["total"] == 2
    flagged = listening["flagged_for_review"]
    assert [f["port"] for f in flagged] == [3389]      # RDP, from WATCH_PORTS
    assert flagged[0]["proto"] == "tcp" and flagged[0]["process"] == "proc10"
    assert listening["all_ports"][1]["proto"] == "udp"


def test_established_view_keeps_what_the_process_list_reported(fake_psutil):
    from tools.network_security import _connections

    established = _connections("established")["established"]
    assert established["total"] == 3
    rows = established["connections"]
    assert {r["remote"] for r in rows} == {
        "93.184.216.34:443", "93.184.216.35:443", "10.0.0.9:4444",
    }
    # Sockets that are neither listening nor established belong in neither view.
    assert all("1.1.1.1" not in r["remote"] for r in rows)
    # The process with no resolvable executable is the one worth looking at, so it
    # sorts first rather than being buried alphabetically.
    assert rows[0]["pid"] == 999


def test_a_process_is_looked_up_once_however_many_sockets_it_holds(fake_psutil):
    """A browser with fifty connections is one process, and psutil.Process() re-reads
    /proc every time it is constructed."""
    from tools.network_security import _connections

    _connections("all")
    assert fake_psutil.process_lookups == 4  # pids 10, 11, 20, 999 - not 5


def test_a_process_that_exits_mid_walk_still_reports_its_socket(monkeypatch, fake_psutil):
    """The pid is real information even when the process is gone by the time it's
    looked up - dropping the row would hide a connection that existed."""
    from tools.network_security import _connections

    def boom(pid):
        raise Exception("no such process")

    monkeypatch.setattr(fake_psutil, "Process", boom)
    rows = _connections("established")["established"]["connections"]
    assert {r["process"] for r in rows} == {"pid:20", "pid:999"}


@pytest.mark.asyncio
async def test_inspect_network_connections_rejects_an_unknown_state():
    from tools.network_security import InspectNetworkConnectionsTool

    result = await InspectNetworkConnectionsTool().run(state="sideways")
    assert result.success is False


def test_inspect_network_connections_defaults_to_both_views():
    from tools.network_security import InspectNetworkConnectionsTool

    sig = inspect.signature(InspectNetworkConnectionsTool.run)
    assert sig.parameters["state"].default == "all"
    enum = next(p for p in InspectNetworkConnectionsTool().parameters if p.name == "state").enum
    assert set(enum) == {"listening", "established", "all"}


# --- System report: specs + health + updates -----------------------------------

@pytest.mark.asyncio
async def test_system_report_defaults_to_health():
    """The usual question, and what the GUI dashboard asks for."""
    pytest.importorskip("psutil")
    from tools.system_health import SystemReportTool

    result = await SystemReportTool().run()
    assert result.success, result.error
    assert "health" in result.output
    assert "metrics" in result.output["health"]
    # Not asked for, so not paid for: specs and updates each cost real work.
    assert "specs" not in result.output


@pytest.mark.asyncio
async def test_system_report_returns_only_the_sections_asked_for():
    pytest.importorskip("psutil")
    from tools.system_health import SystemReportTool

    result = await SystemReportTool().run(sections=["specs"])
    assert result.success, result.error
    assert "specs" in result.output and "health" not in result.output


@pytest.mark.asyncio
async def test_system_report_names_the_sections_it_does_not_know():
    from tools.system_health import SystemReportTool

    result = await SystemReportTool().run(sections=["weather"])
    assert result.success is False
    assert "weather" in result.error


@pytest.mark.asyncio
async def test_the_dashboard_reads_the_shape_system_report_returns():
    """gui/api.py digs into output['health']['metrics'] - a merge that moved those
    keys would leave the dashboard silently reporting 0% for everything."""
    pytest.importorskip("psutil")
    from tools.system_health import SystemReportTool

    result = await SystemReportTool().run(sections=["health"])
    metrics = result.output["health"]["metrics"]
    assert "cpu_percent" in metrics and "memory_percent" in metrics


# --- launch_app: one tool, two consequences ------------------------------------

def test_launch_app_is_classified_per_call_not_per_tool():
    """The merge only works because opening a page and starting a program can be
    told apart at call time - otherwise absorbing open_url would mean confirming
    every 'open YouTube'."""
    entry = get_permissions()["tools"]["launch_app"]
    assert entry["action_by_case"] == {"web_page": "execute", "program": "modify"}


def test_a_tool_cannot_invent_an_action_class_for_itself():
    """action_case() names a situation; permissions.yaml prices it. A case nobody
    configured falls back to the tool's plain action rather than to nothing."""
    from core.safety_guard import RiskTier, SafetyGuard

    guard = SafetyGuard()
    assert guard.get_tier("launch_app", "some_case_nobody_configured") is RiskTier.MODIFY
    assert guard.get_tier("read_file", "web_page") is RiskTier.READ


# --- Guards that had to follow the tools they were guarding --------------------
# Two tools running the same act under different parameter names is the other face
# of duplication, and the hard-block list only knew one name for each.

def test_a_shell_snippet_is_blocked_whatever_parameter_carries_it():
    """run_code takes a bash snippet - the same act as run_shell_command's `command`
    under a different name. Checking only `command` meant 'rm -rf /' was refused as a
    command and run as a one-line bash program."""
    from core.safety_guard import SafetyGuard

    guard = SafetyGuard()
    blocked = guard.check_hard_block("run_shell_command", {"command": "rm -rf /"})
    assert blocked and "forbidden pattern" in blocked
    assert guard.check_hard_block("run_code", {"language": "bash", "code": "rm -rf /"})


def test_ordinary_code_is_not_caught_by_the_shell_patterns():
    from core.safety_guard import SafetyGuard

    guard = SafetyGuard()
    assert guard.check_hard_block("run_code", {
        "language": "python", "code": "import pandas as pd\nprint(pd.__version__)",
    }) is None


def test_the_domain_blocklist_followed_open_url_into_launch_app(monkeypatch):
    """The page to open arrives as `app_name` now. A blocklist that only read `url`
    would have quietly stopped applying the moment the two tools merged."""
    from core.safety_guard import SafetyGuard

    guard = SafetyGuard()
    monkeypatch.setitem(guard.permissions, "blocked_domains", ["blocked.example"])

    assert guard.check_hard_block("launch_app", {"app_name": "https://blocked.example/x"})
    assert guard.check_hard_block("launch_app", {"app_name": "https://allowed.example/x"}) is None
