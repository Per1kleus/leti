"""The business workspace: calendar, goals, batches, entities and the numbers.

Every capability here had to answer the same question before it was worth
building: what does it say when it does not know? A calendar that cannot be
reached must never read as an empty day. A goal with nothing to count must never
report a percentage. A reference that could mean two customers must ask rather
than pick. A batch must not do anything until somebody says so, and must not be
able to do anything even then.

Those four are most of this file. The capabilities themselves are small; the
refusals are the product.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

import tools.business as business_store  # noqa: E402
import tools.contacts as contacts  # noqa: E402
from core import (business, business_approvals, business_goals,  # noqa: E402
                  modes, task_manager)
from tools import meeting_scheduler, scheduler  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    modes.reset_for_tests()
    business.release()
    monkeypatch.setattr(business_store, "_store_path", lambda: tmp_path / "business.json")
    monkeypatch.setattr(contacts, "_contacts_path", lambda: tmp_path / "contacts.json")
    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(scheduler, "_store_path", lambda: tmp_path / "sched.json")
    yield
    modes.reset_for_tests()
    business.release()


def lead(name, stage="qualified", value=1000, days_idle=0, contact_id=None):
    now = time.time()
    return {"id": f"l-{name}", "name": name, "stage": stage, "value": value,
            "created_at": now - days_idle * 86400,
            "stage_changed_at": now - days_idle * 86400, "contact_id": contact_id}


def save(leads=(), goals=()):
    business_store.save_records({"lead": list(leads), "income": [], "expense": [],
                                 "goal": list(goals)})


# --- Calendar: reading it, and never guessing when it cannot be read -----------------

def _ics(summary, start, end, attendees=()):
    import icalendar

    calendar = icalendar.Calendar()
    event = icalendar.Event()
    event.add("summary", summary)
    event.add("dtstart", start)
    event.add("dtend", end)
    event.add("uid", summary.replace(" ", "-"))
    for who in attendees:
        event.add("attendee", f"mailto:{who}")
    calendar.add_component(event)
    return calendar


class FakeCalDAV:
    """The caldav client, as far as the reader can tell. Records what it was asked."""

    def __init__(self, events=(), fail=None, name="Work"):
        self.events, self.fail, self.name = list(events), fail, name
        self.searched = []

    # -- client
    def principal(self):
        return self

    def calendars(self):
        if self.fail:
            raise self.fail
        return [self]

    # -- calendar
    def search(self, start=None, end=None, **kwargs):
        if self.fail:
            raise self.fail
        self.searched.append((start, end))
        return [types.SimpleNamespace(icalendar_instance=e) for e in self.events]


@pytest.fixture
def calendar(monkeypatch):
    """A configured CalDAV account whose server is a fake."""
    monkeypatch.setattr(meeting_scheduler, "_calendar_settings",
                        lambda: {"caldav_url": "https://dav.example", "username": "u",
                                 "app_password": "p", "calendar_name": ""})
    monkeypatch.setattr(meeting_scheduler, "get_settings",
                        lambda: {"calendar": {"caldav_url": "https://dav.example",
                                              "username": "u", "app_password": "p"}})

    holder = {}

    def install(events=(), fail=None):
        fake = FakeCalDAV(events, fail)
        holder["fake"] = fake
        module = types.ModuleType("caldav")
        module.DAVClient = lambda **kwargs: fake
        monkeypatch.setitem(sys.modules, "caldav", module)
        return fake

    holder["install"] = install
    return holder


def test_events_are_read_from_the_calendar_already_configured(calendar):
    start = datetime(2026, 9, 21, 10, 0)
    calendar["install"]([_ics("Client meeting", start, start + timedelta(hours=1),
                              ["client@example.com"])])

    events = asyncio.run(meeting_scheduler.read_events(start, start + timedelta(days=1)))
    assert len(events) == 1
    assert events[0]["title"] == "Client meeting"
    assert events[0]["duration_minutes"] == 60
    assert events[0]["attendees"] == ["client@example.com"]


def test_a_calendar_that_cannot_be_reached_raises_rather_than_looking_empty(calendar):
    calendar["install"](fail=RuntimeError("the server said no"))
    with pytest.raises(RuntimeError):
        asyncio.run(meeting_scheduler.read_events(datetime.now(),
                                                  datetime.now() + timedelta(days=1)))


@pytest.mark.asyncio
async def test_the_tool_says_it_cannot_see_rather_than_reporting_no_meetings(calendar):
    from tools.business_agent import BusinessCalendarTool

    calendar["install"](fail=RuntimeError("connection refused"))
    result = await BusinessCalendarTool().run("today")

    assert result.success is False
    assert "not an empty calendar" in result.error
    assert "could not read" in result.error


@pytest.mark.asyncio
async def test_no_calendar_configured_is_said_plainly(monkeypatch):
    from tools.business_agent import BusinessCalendarTool

    monkeypatch.setattr(meeting_scheduler, "get_settings", lambda: {"calendar": {}})
    result = await BusinessCalendarTool().run("today")

    assert result.success is False
    assert "Connections" in result.error
    assert "does not know what is in it" in result.error


@pytest.mark.asyncio
async def test_an_empty_window_is_reported_as_empty_when_it_was_actually_read(calendar):
    from tools.business_agent import BusinessCalendarTool

    calendar["install"]([])
    result = await BusinessCalendarTool().run("today")

    assert result.success and result.output["count"] == 0
    assert "was read and there is nothing" in result.output["note"]


@pytest.mark.asyncio
@pytest.mark.parametrize("when,days", [("today", 1), ("tomorrow", 1), ("week", 7)])
async def test_the_window_asked_for_is_the_window_searched(calendar, when, days):
    from tools.business_agent import BusinessCalendarTool

    fake = calendar["install"]([])
    await BusinessCalendarTool().run(when)
    start, end = fake.searched[-1]
    assert (end - start).days == days


@pytest.mark.asyncio
async def test_an_absurd_range_is_refused(calendar):
    from tools.business_agent import BusinessCalendarTool

    calendar["install"]([])
    result = await BusinessCalendarTool().run("range", start="2026-01-01", end="2027-01-01")
    assert result.success is False and "shorter window" in result.error


def test_overlapping_meetings_are_found_and_all_day_events_are_not_a_clash():
    events = [
        {"title": "A", "starts": "2026-09-21T10:00:00", "ends": "2026-09-21T11:00:00",
         "all_day": False},
        {"title": "B", "starts": "2026-09-21T10:30:00", "ends": "2026-09-21T11:30:00",
         "all_day": False},
        {"title": "Holiday", "starts": "2026-09-21", "ends": "2026-09-22", "all_day": True},
    ]
    clashes = meeting_scheduler.overlapping(events)
    assert len(clashes) == 1 and set(clashes[0]["between"]) == {"A", "B"}


def test_an_event_with_unreadable_times_is_skipped_not_guessed():
    assert meeting_scheduler.overlapping(
        [{"title": "A", "starts": "soon", "ends": "later", "all_day": False}]) == []


# --- Goals: never a percentage from nothing -----------------------------------------

def test_a_goal_is_a_record_in_the_store_that_already_exists():
    save()
    goal = business_goals.create("Increase monthly sales", measure="qualified leads",
                                 target_value=20)
    raw = business_store.load_records()
    assert [g["id"] for g in raw["goal"]] == [goal["id"]]
    assert set(raw) == {"lead", "income", "expense", "goal"}


def test_writing_leads_does_not_wipe_goals():
    """load_records/save_records round-trip every key, which is why `goal` is in
    RECORD_TYPES - a type missing from that tuple is erased by the next write."""
    save()
    business_goals.create("Keep me")
    records = business_store.load_records()
    records["lead"].append(lead("Acme"))
    business_store.save_records(records)
    assert len(business_store.load_records()["goal"]) == 1


def test_a_goal_with_nothing_to_count_refuses_to_report_progress():
    save()
    goal = business_goals.create("Vague ambition")
    found = business_goals.progress(goal)
    assert found["basis"] == business_goals.UNMEASURABLE
    assert found["percent"] is None
    assert "cannot be determined" in found["how"]


def test_progress_from_a_measured_result_says_it_is_measured():
    save()
    goal = business_goals.create("More leads", measure="qualified leads", target_value=20)
    business_goals.record_result(goal["id"], value=12)
    found = business_goals.progress(business_goals.get(goal["id"]))
    assert found["basis"] == business_goals.MEASURED
    assert found["percent"] == 60.0
    assert found["counted_from"] == "results recorded on this goal"


def test_progress_from_tasks_says_it_counts_work_not_achievement():
    save()
    goal = business_goals.create("Ship the thing")
    first = task_manager.create_task("a", ["x"], name="A")
    second = task_manager.create_task("b", ["x"], name="B")
    business_goals.link(goal["id"], task_id=first["id"])
    business_goals.link(goal["id"], task_id=second["id"])
    task_manager._set_status(first["id"], task_manager.COMPLETED)

    found = business_goals.progress(business_goals.get(goal["id"]))
    assert found["basis"] == business_goals.BY_TASKS
    assert found["percent"] == 50.0
    assert "not the same as the goal being achieved" in found["how"]


def test_a_measured_result_outranks_counting_tasks():
    save()
    goal = business_goals.create("Both", measure="leads", target_value=10)
    task = task_manager.create_task("a", ["x"], name="A")
    business_goals.link(goal["id"], task_id=task["id"])
    business_goals.record_result(goal["id"], value=5)
    assert business_goals.progress(business_goals.get(goal["id"]))["basis"] == \
        business_goals.MEASURED


def test_linking_a_task_that_does_not_exist_is_refused():
    save()
    goal = business_goals.create("A goal")
    with pytest.raises(ValueError) as refused:
        business_goals.link(goal["id"], task_id="nonexistent")
    assert "no task with id" in str(refused.value).lower()


def test_a_goal_knowing_about_a_task_is_not_owning_it():
    save()
    goal = business_goals.create("A goal")
    task = task_manager.create_task("a", ["x"], name="A")
    business_goals.link(goal["id"], task_id=task["id"])

    business_goals.delete(goal["id"])
    assert task_manager.get_task(task["id"]) is not None, "deleting a goal ate a task"


def test_statuses_are_the_four_and_nothing_else():
    save()
    goal = business_goals.create("A goal")
    assert business_goals.set_status(goal["id"], business_goals.BLOCKED,
                                     reason="waiting on legal")["status"] == "blocked"
    with pytest.raises(ValueError):
        business_goals.set_status(goal["id"], "vibes")


def test_blocked_and_neglected_goals_are_surfaced():
    save()
    blocked = business_goals.create("Blocked one")
    business_goals.set_status(blocked["id"], business_goals.BLOCKED, reason="waiting")
    business_goals.create("Forgotten one")
    # Age it in the store. Going through _replace would stamp updated_at with now,
    # which is right - a goal is neglected because nothing touched it, and this is
    # what "nothing touched it for forty days" looks like on disk.
    records = business_store.load_records()
    for stored in records["goal"]:
        if stored["name"] == "Forgotten one":
            stored["updated_at"] = time.time() - 40 * 86400
    business_store.save_records(records)

    flagged = {f["goal"]: f["why"] for f in business_goals.needs_attention()}
    assert "waiting" in flagged["Blocked one"]
    assert "nothing recorded for" in flagged["Forgotten one"]


def test_a_completed_goal_stops_asking_for_attention():
    save()
    goal = business_goals.create("Done")
    business_goals.set_status(goal["id"], business_goals.COMPLETED)
    assert business_goals.needs_attention() == []


# --- Batches: preview, approve, and never execute ------------------------------------

def proposal():
    return business_approvals.propose("follow up with quiet leads", [
        {"kind": "send_email", "about": "Acme Ltd", "to": "a@acme.test"},
        {"kind": "send_email", "about": "Beta Co", "to": "b@beta.test"},
        {"kind": "send_email", "about": "Gamma Inc",
         "blocked_because": "no email address on the contact"},
    ])


def test_a_proposal_does_nothing_and_says_so():
    preview = proposal()
    assert preview["found"] == 3
    assert preview["can_be_done"] == 2
    assert preview["cannot_be_done"][0]["why"] == "no email address on the contact"
    assert preview["nothing_has_happened"] is True


def test_an_action_kind_nobody_reviewed_cannot_be_smuggled_in():
    preview = business_approvals.propose("odd", [{"kind": "rm -rf", "about": "everything"}])
    assert preview["can_be_done"] == 0
    assert "is not an action this can propose" in preview["cannot_be_done"][0]["why"]


def test_only_the_approved_items_come_back_to_be_done():
    preview = proposal()
    business_approvals.decide(preview["batch"], approve=[1], reject=[2])
    to_do = business_approvals.approved_items(preview["batch"])
    assert [i["about"] for i in to_do] == ["Acme Ltd"]


def test_deciding_still_does_nothing():
    preview = proposal()
    decision = business_approvals.decide(preview["batch"], approve=[1, 2])
    assert decision["nothing_has_happened"] is True
    assert "not permission" in decision["next"]


def test_a_blocked_item_cannot_be_approved_into_existence():
    preview = proposal()
    business_approvals.decide(preview["batch"], approve=[1, 2, 3])
    assert [i["n"] for i in business_approvals.approved_items(preview["batch"])] == [1, 2]


def test_an_item_is_done_because_a_tool_said_so_not_because_it_was_approved():
    preview = proposal()
    business_approvals.decide(preview["batch"], approve=[1, 2])
    business_approvals.record_outcome(preview["batch"], 1, True, "sent")
    business_approvals.record_outcome(preview["batch"], 2, False, "SMTP refused")

    report = business_approvals.report(preview["batch"])
    assert [d["n"] for d in report["done"]] == [1]
    assert [f["n"] for f in report["failed"]] == [2]
    assert "do not round either of them up" in report["how_to_report"]


def test_an_approved_item_never_attempted_is_reported_as_such():
    preview = proposal()
    business_approvals.decide(preview["batch"], approve=[1, 2])
    report = business_approvals.report(preview["batch"])
    assert [n["n"] for n in report["never_attempted"]] == [1, 2]
    assert "not yet attempted" in report["summary"]


def test_cancelling_drops_what_was_outstanding():
    preview = proposal()
    business_approvals.decide(preview["batch"], approve=[1, 2])
    cancelled = business_approvals.cancel(preview["batch"])
    assert cancelled["dropped"] == [1, 2]
    assert "Nothing outstanding was performed" in cancelled["note"]
    assert business_approvals.get(preview["batch"]) is None


def test_the_queue_cannot_send_write_or_authorise_anything():
    """The hard requirement. The queue is orchestration; SafetyGuard is authority."""
    import ast

    tree = ast.parse(open("core/business_approvals.py").read())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("authorize", "send_email", "run", "execute_tool", "save_records",
                      "handle_user_input", "set_unattended", "start_in_background"):
        assert forbidden not in called, f"the approval queue calls {forbidden}()"
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)]


def test_batches_do_not_outlive_the_mode():
    preview = proposal()
    assert business_approvals.get(preview["batch"]) is not None
    business.release()
    assert business_approvals.get(preview["batch"]) is None


# --- Entity resolution: ask rather than guess ----------------------------------------

def test_one_match_resolves():
    save([lead("Beta Co"), lead("Gamma")])
    found = business.resolve("the Beta lead", kind="lead")
    assert found["resolved"]["name"] == "Beta Co"


def test_several_matches_ask_instead_of_picking():
    save([lead("Acme Ltd"), lead("Acme Holdings")])
    found = business.resolve("the Acme customer", kind="lead")
    assert found["resolved"] is None
    assert sorted(c["name"] for c in found["candidates"]) == ["Acme Holdings", "Acme Ltd"]
    assert "which one" in found["ask"]


def test_nothing_matching_says_nothing_matches():
    save([lead("Acme Ltd")])
    found = business.resolve("Wakanda Industries", kind="lead")
    assert found["resolved"] is None and found["candidates"] == []
    assert "Nothing matches" in found["problem"]


def test_an_irrelevant_record_is_not_offered_as_a_candidate():
    """The first version returned every lead for a name that matched none of them,
    which turns "I cannot find that" into "here are four things it might be"."""
    save([lead("Acme Ltd"), lead("Beta Co"), lead("Gamma"), lead("Delta")])
    assert business.resolve("Wakanda", kind="lead")["candidates"] == []


def test_a_bare_reference_offers_what_there_is():
    save([lead("Acme Ltd"), lead("Beta Co")])
    found = business.resolve("the customer", kind="lead")
    assert found["resolved"] is None and len(found["candidates"]) == 2


def test_a_bare_reference_with_one_of_that_kind_resolves():
    save([lead("Only Co")])
    assert business.resolve("the customer", kind="lead")["resolved"]["name"] == "Only Co"


def test_a_source_that_cannot_be_read_is_named_not_silently_skipped(monkeypatch):
    save([lead("Acme Ltd")])
    monkeypatch.setattr(meeting_scheduler, "get_settings", lambda: {"calendar": {}})
    found = business.resolve("the meeting tomorrow", kind="meeting")
    assert found["resolved"] is None
    assert any("calendar" in u for u in (found.get("unavailable") or []))


# --- Metrics: observed, calculated, assumed ------------------------------------------

def test_every_number_says_where_it_came_from():
    save([lead("A", stage="qualified", value=1000), lead("B", stage="won", value=2000),
          lead("C", stage="lost", value=500)])
    found = business.metrics()

    assert found["observed"]["leads_total"] == 3
    assert found["observed"]["leads_won"] == 1
    assert found["calculated"]["conversion_rate_percent"] == 50.0
    assert found["calculated"]["conversion_counted_from"] == "1 won of 2 decided"
    assert "stage_weights" in found["assumed"]
    assert "never fill one in" in found["how_to_report"]


def test_a_rate_with_no_denominator_is_none_not_zero():
    save([lead("A", stage="qualified")])
    found = business.metrics()
    assert found["calculated"]["conversion_rate_percent"] is None
    assert "no lead has been marked won or lost" in found["calculated"]["conversion_counted_from"]


def test_unreadable_records_produce_no_figures_at_all(monkeypatch):
    def explode():
        raise OSError("gone")

    monkeypatch.setattr(business_store, "load_records", explode)
    found = business.metrics()
    assert "problem" in found and "observed" not in found
    assert "figures from nothing" in found["note"]


# --- The briefing, with everything in it ---------------------------------------------

def test_the_briefing_has_the_sections_a_person_needs():
    save([lead("Quiet Co", days_idle=30)])
    brief = business.briefing()
    for section in ("needs_your_attention", "waiting_on_you", "leads_needing_attention",
                    "todays_meetings", "goals", "pipeline", "scheduled", "blocked",
                    "cannot_see"):
        assert section in brief, section


def test_attention_leads_with_what_is_blocking_the_user():
    save([lead("Quiet Co", days_idle=30)])
    task = task_manager.create_task("send it", ["draft"], name="Proposal")
    task_manager._set_status(task["id"], task_manager.WAITING_FOR_USER,
                             blocked_reason="needs your confirmation")

    attention = business.briefing()["needs_your_attention"]
    assert attention[0]["kind"] == "waiting for your approval"
    assert any(a["kind"] == "follow-up" for a in attention)


def test_a_blocked_goal_reaches_the_attention_list():
    save()
    goal = business_goals.create("Stuck")
    business_goals.set_status(goal["id"], business_goals.BLOCKED, reason="legal review")
    attention = business.briefing()["needs_your_attention"]
    assert any(a["kind"] == "goal" and "legal review" in a["why"] for a in attention)


def test_the_briefing_never_claims_an_empty_calendar(monkeypatch):
    monkeypatch.setattr(meeting_scheduler, "get_settings", lambda: {"calendar": {}})
    meetings = business.briefing()["todays_meetings"]
    assert meetings["available"] is False
    assert meetings["events"] == []
    assert "not the same as having none" in meetings["problem"]


def test_a_connected_calendar_puts_meetings_in_the_briefing(calendar):
    start = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    calendar["install"]([_ics("Client meeting", start, start + timedelta(hours=1))])
    meetings = business.briefing()["todays_meetings"]
    assert meetings["available"] is True
    assert [e["title"] for e in meetings["events"]] == ["Client meeting"]


# --- Mode isolation still holds ------------------------------------------------------

@pytest.fixture(scope="module")
def registry():
    from unittest.mock import MagicMock

    import main

    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


def test_the_new_business_tools_are_invisible_in_default_mode(registry):
    visible = modes.visible_tools(registry, modes.DEFAULT)
    for name in ("business_calendar", "business_goals", "business_batch",
                 "business_briefing"):
        assert name not in visible, f"{name} leaked into Default Mode"
        assert registry.get(name) is not None


def test_default_mode_is_exactly_what_it_was(registry):
    """Business Mode grew; Default Mode did not. This is the number that proves it."""
    visible = modes.visible_tools(registry, modes.DEFAULT)
    assert len(visible) == 129
    assert len(json.dumps(registry.schemas_for(visible))) == 89_298


def test_coding_mode_never_sees_a_business_tool(registry):
    coding_tools = modes.visible_tools(registry, modes.CODING)
    assert not [n for n in coding_tools if n.startswith("business_")]


def test_business_mode_never_sees_a_coding_tool(registry):
    business_tools = modes.visible_tools(registry, modes.BUSINESS)
    assert not ({"code_map", "git_workspace", "github"} & business_tools)


@pytest.mark.parametrize("text", [
    "check my calendar", "what meetings do I have today", "show me my goals",
    "follow up with my leads", "what is my conversion rate",
    "prepare a business report", "the customer wants a quote",
])
def test_none_of_the_new_capabilities_trigger_business_mode(text):
    from core import intent

    assert intent.mode_command(text) is None
    assert modes.current() == modes.DEFAULT


def test_every_new_tool_has_a_permission_entry(registry):
    from core.config_loader import get_permissions

    entries = get_permissions()["tools"]
    for name in ("business_calendar", "business_goals", "business_batch"):
        assert name in entries, f"{name} has no action class"
    assert entries["business_goals"]["action_by_case"]["write"] == "modify"


def test_changing_a_goal_is_a_write_and_reading_one_is_not():
    from tools.business_agent import BusinessGoalsTool

    tool = BusinessGoalsTool()
    assert tool.action_case({"action": "list"}) == "read"
    assert tool.action_case({"action": "show"}) == "read"
    for writing in ("create", "link", "result", "status", "remove"):
        assert tool.action_case({"action": writing}) == "write"


def test_nothing_new_runs_on_a_clock():
    import ast

    for module in ("core/business.py", "core/business_goals.py",
                   "core/business_approvals.py", "tools/business_agent.py"):
        source = open(module).read()
        tree = ast.parse(source)
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)], module
        for forbidden in ("Thread(", "setInterval", "watchdog", "inotify"):
            assert forbidden not in source, f"{module} uses {forbidden}"


def test_no_business_module_authorises_anything():
    import ast

    for module in ("core/business.py", "core/business_goals.py",
                   "core/business_approvals.py", "tools/business_agent.py"):
        tree = ast.parse(open(module).read())
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for forbidden in ("authorize", "set_unattended", "execute_tool"):
            assert forbidden not in called, f"{module} calls {forbidden}()"
