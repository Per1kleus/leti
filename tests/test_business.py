"""Tests for business records and the metrics computed from them."""
from __future__ import annotations

import json
import re
import time

import pytest

import tools.business as business
import tools.contacts as contacts
from tools.business import (
    BusinessDashboardTool, ListBusinessDataTool, NextActionsTool,
    RecordBusinessDataTool, UpdateLeadTool, contact_name, summarize,
)
from tools.contacts import AddContactTool


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(business, "_store_path", lambda: tmp_path / "business.json")
    monkeypatch.setattr(contacts, "_contacts_path", lambda: tmp_path / "contacts.json")
    return tmp_path


async def _contact(name="Maria Kostas", email="maria@example.gr"):
    result = await AddContactTool().run(name=name, email=email)
    return re.search(r"id=(\w+)", result.output).group(1)


# --- People live in the contact book -------------------------------------------

@pytest.mark.asyncio
async def test_a_lead_links_to_a_contact_rather_than_copying_it(isolated_store):
    """One place for someone's details - updating the contact updates everything."""
    contact_id = await _contact()
    result = await RecordBusinessDataTool().run(
        record_type="lead", name="Taverna website", amount=2500, contact_id=contact_id,
    )
    assert result.output["linked_contact"] == "Maria Kostas"

    stored = json.dumps(business.load_records())
    assert "maria@example.gr" not in stored, "contact details were copied into business records"


@pytest.mark.asyncio
async def test_a_link_to_a_missing_contact_is_refused():
    result = await RecordBusinessDataTool().run(
        record_type="lead", name="Ghost", contact_id="nonexistent",
    )
    assert result.success is False
    assert "No contact" in result.error


# --- Validation -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_records_are_validated():
    assert (await RecordBusinessDataTool().run(record_type="lead")).success is False
    assert (await RecordBusinessDataTool().run(record_type="income", name="x")).success is False
    assert (await RecordBusinessDataTool().run(record_type="nonsense", name="x")).success is False
    bad_stage = await RecordBusinessDataTool().run(
        record_type="lead", name="x", stage="almost_signed")
    assert bad_stage.success is False


# --- Metrics --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conversion_rate_counts_decided_leads_only():
    """Counting open leads as unconverted understates the rate and makes it move
    whenever new leads arrive, which makes periods incomparable."""
    for stage in ("won", "lost", "new", "proposal"):
        await RecordBusinessDataTool().run(record_type="lead", name=stage, amount=100, stage=stage)

    metrics = summarize()
    assert metrics["conversion_rate_percent"] == 50.0     # 1 won of 2 decided
    assert "2 still open" in metrics["conversion_basis"]


@pytest.mark.asyncio
async def test_pipeline_is_weighted_by_stage():
    await RecordBusinessDataTool().run(record_type="lead", name="a", amount=1000, stage="new")
    await RecordBusinessDataTool().run(record_type="lead", name="b", amount=1000, stage="negotiation")

    metrics = summarize()
    assert metrics["pipeline_value"] == 2000.0
    assert metrics["pipeline_weighted"] == 900.0          # 1000*0.1 + 1000*0.8


@pytest.mark.asyncio
async def test_won_and_lost_leads_leave_the_pipeline():
    await RecordBusinessDataTool().run(record_type="lead", name="w", amount=500, stage="won")
    await RecordBusinessDataTool().run(record_type="lead", name="l", amount=500, stage="lost")
    assert summarize()["pipeline_value"] == 0.0


@pytest.mark.asyncio
async def test_profit_and_revenue_by_service():
    await RecordBusinessDataTool().run(record_type="income", name="i", amount=1500, service="websites")
    await RecordBusinessDataTool().run(record_type="income", name="i", amount=500, service="hosting")
    await RecordBusinessDataTool().run(record_type="expense", name="e", amount=200, category="ads")

    metrics = summarize()
    assert metrics["revenue"] == 2000.0
    assert metrics["profit"] == 1800.0
    assert metrics["revenue_by_service"]["websites"] == 1500.0
    assert list(metrics["revenue_by_service"])[0] == "websites"   # biggest first


@pytest.mark.asyncio
async def test_channel_performance_is_grouped_by_source():
    await RecordBusinessDataTool().run(record_type="lead", name="a", amount=1000,
                                       stage="won", source="referral")
    await RecordBusinessDataTool().run(record_type="lead", name="b", amount=1000,
                                       stage="lost", source="google_ads")

    by_source = summarize()["by_source"]
    assert by_source["referral"]["conversion_rate_percent"] == 100.0
    assert by_source["google_ads"]["conversion_rate_percent"] == 0.0


@pytest.mark.asyncio
async def test_dashboard_compares_periods():
    await RecordBusinessDataTool().run(record_type="income", name="now", amount=1000)
    result = await BusinessDashboardTool().run(period="this_month", compare_with="last_month")
    assert result.output["comparison"]["changes"]["revenue"]["now"] == 1000.0
    assert result.output["comparison"]["changes"]["revenue"]["then"] == 0.0


@pytest.mark.asyncio
async def test_no_records_says_so_rather_than_reporting_zeroes():
    result = await BusinessDashboardTool().run(period="all")
    assert "no business records yet" in result.output["note"]


# --- Next actions ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_next_actions_ranks_by_weighted_value():
    await RecordBusinessDataTool().run(record_type="lead", name="small", amount=500, stage="negotiation")
    await RecordBusinessDataTool().run(record_type="lead", name="big", amount=5000, stage="negotiation")

    actions = (await NextActionsTool().run()).output["next_actions"]
    assert actions[0]["name"] == "big"


@pytest.mark.asyncio
async def test_stale_leads_come_first(isolated_store):
    """A lead going cold is time-sensitive; weighted value alone would keep
    recommending the same top deal forever."""
    await RecordBusinessDataTool().run(record_type="lead", name="fresh_big", amount=9000, stage="proposal")
    await RecordBusinessDataTool().run(record_type="lead", name="stale_small", amount=100, stage="new")

    records = business.load_records()
    for lead in records["lead"]:
        if lead["name"] == "stale_small":
            lead["created_at"] = time.time() - 60 * 86400
    business.save_records(records)

    actions = (await NextActionsTool().run()).output["next_actions"]
    assert actions[0]["name"] == "stale_small"
    assert actions[0]["stale"] is True


@pytest.mark.asyncio
async def test_closed_leads_are_not_suggested():
    await RecordBusinessDataTool().run(record_type="lead", name="done", amount=5000, stage="won")
    assert (await NextActionsTool().run()).output["next_actions"] == []


@pytest.mark.asyncio
async def test_moving_a_lead_to_won_changes_the_metrics():
    created = await RecordBusinessDataTool().run(
        record_type="lead", name="deal", amount=3000, stage="proposal")
    assert summarize()["leads_won"] == 0

    moved = await UpdateLeadTool().run(lead_id=created.output["id"], stage="won")
    assert moved.output["moved_from"] == "proposal"
    assert summarize()["leads_won"] == 1


@pytest.mark.asyncio
async def test_updating_an_unknown_lead_is_reported():
    assert (await UpdateLeadTool().run(lead_id="nope", stage="won")).success is False


@pytest.mark.asyncio
async def test_listing_resolves_contact_names():
    contact_id = await _contact()
    await RecordBusinessDataTool().run(record_type="lead", name="deal", contact_id=contact_id)
    listed = (await ListBusinessDataTool().run(record_type="lead")).output["lead"]
    assert listed[0]["contact_name"] == "Maria Kostas"
