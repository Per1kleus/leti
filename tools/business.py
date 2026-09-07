"""Business records and the metrics computed from them.

People are not stored here. A lead, a prospect and a client are all a person or
an organisation Leti may also need to email, resolve by name, or schedule a
meeting with - and that is what tools/contacts.py is for. A record here carries a
`contact_id` pointing into the contact book, so a client's email address lives in
exactly one place and updating it updates it everywhere.

What this module owns is the business layer over those people: where each one is
in the pipeline, what has been earned and spent, and what that adds up to. Three
record types, deliberately:

  lead     - someone who might buy, with a stage, a value, and a source
  income   - money received, optionally tied to a lead and a service
  expense  - money spent, with a category

Everything the spec asks for is computed from those rather than stored
separately: conversion rate is won leads over total, pipeline value is open
leads weighted by stage, profitability is income minus expenses grouped however
you ask, and channel performance is leads and revenue grouped by source. Storing
a derived number is how a dashboard starts disagreeing with the data.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.atomic_write import atomic_write_json
from core.config_loader import get_settings, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

# Pipeline stages in order, with the probability each is worth when weighting the
# pipeline. Conventional sales-funnel values; override in settings if your
# business converts differently.
DEFAULT_STAGES: Dict[str, float] = {
    "new": 0.10,
    "contacted": 0.20,
    "qualified": 0.40,
    "proposal": 0.60,
    "negotiation": 0.80,
    "won": 1.00,
    "lost": 0.00,
}
OPEN_STAGES = [s for s in DEFAULT_STAGES if s not in ("won", "lost")]
RECORD_TYPES = ("lead", "income", "expense")


def _store_path() -> Path:
    cfg = get_settings().get("business", {})
    path = resolve_path(cfg.get("file_path", "./data/business.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def stage_weights() -> Dict[str, float]:
    configured = get_settings().get("business", {}).get("stage_weights") or {}
    return {**DEFAULT_STAGES, **{k: float(v) for k, v in configured.items()}}


def load_records() -> Dict[str, List[Dict[str, Any]]]:
    import json

    path = _store_path()
    if not path.is_file():
        return {kind: [] for kind in RECORD_TYPES}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {kind: [] for kind in RECORD_TYPES}
    return {kind: list(data.get(kind, [])) for kind in RECORD_TYPES}


def save_records(records: Dict[str, List[Dict[str, Any]]]) -> None:
    atomic_write_json(_store_path(), records)


def parse_date(value: Optional[str]) -> float:
    """A date string to a timestamp, defaulting to now.

    Accepts YYYY-MM-DD and full ISO 8601, because the model will produce both.
    """
    if not value:
        return time.time()
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return time.time()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")


def within(records: List[Dict[str, Any]], since: Optional[float], until: Optional[float],
           field: str = "date") -> List[Dict[str, Any]]:
    return [
        r for r in records
        if (since is None or r.get(field, 0) >= since) and (until is None or r.get(field, 0) < until)
    ]


def period_bounds(period: str) -> tuple:
    """(start, end, label) for 'this_month', 'last_month', 'this_year', 'all'.

    Used by both the dashboard and comparisons so "compared with last month"
    means the same span in both.
    """
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if period == "this_month":
        return start_of_month.timestamp(), None, start_of_month.strftime("%B %Y")
    if period == "last_month":
        end = start_of_month
        start = (end - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp(), end.timestamp(), start.strftime("%B %Y")
    if period == "this_year":
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp(), None, str(now.year)
    if period == "last_30_days":
        return (now - timedelta(days=30)).timestamp(), None, "last 30 days"
    return None, None, "all time"


def contact_name(contact_id: Optional[str]) -> Optional[str]:
    """Resolve a contact id through the contact book - people live there, not here."""
    if not contact_id:
        return None
    from tools.contacts import _load

    for contact in _load():
        if contact.get("id") == contact_id:
            return contact.get("name")
    return None


def summarize(since: Optional[float] = None, until: Optional[float] = None) -> Dict[str, Any]:
    """Every headline metric, computed from the records rather than stored."""
    records = load_records()
    weights = stage_weights()

    leads = within(records["lead"], since, until, field="created_at")
    income = within(records["income"], since, until)
    expenses = within(records["expense"], since, until)

    revenue = sum(float(r.get("amount", 0)) for r in income)
    spend = sum(float(r.get("amount", 0)) for r in expenses)
    won = [l for l in leads if l.get("stage") == "won"]
    lost = [l for l in leads if l.get("stage") == "lost"]
    open_leads = [l for l in leads if l.get("stage") in OPEN_STAGES]
    decided = len(won) + len(lost)

    by_service: Dict[str, float] = {}
    for record in income:
        by_service[record.get("service") or "unspecified"] = (
            by_service.get(record.get("service") or "unspecified", 0) + float(record.get("amount", 0))
        )
    by_category: Dict[str, float] = {}
    for record in expenses:
        by_category[record.get("category") or "unspecified"] = (
            by_category.get(record.get("category") or "unspecified", 0) + float(record.get("amount", 0))
        )
    by_source: Dict[str, Dict[str, Any]] = {}
    for lead in leads:
        source = lead.get("source") or "unspecified"
        entry = by_source.setdefault(source, {"leads": 0, "won": 0, "revenue": 0.0})
        entry["leads"] += 1
        if lead.get("stage") == "won":
            entry["won"] += 1
            entry["revenue"] += float(lead.get("value", 0))
    for entry in by_source.values():
        entry["conversion_rate_percent"] = (
            round(entry["won"] / entry["leads"] * 100, 1) if entry["leads"] else 0.0
        )

    return {
        "revenue": round(revenue, 2),
        "expenses": round(spend, 2),
        "profit": round(revenue - spend, 2),
        "margin_percent": round((revenue - spend) / revenue * 100, 1) if revenue else None,
        "leads_total": len(leads),
        "leads_open": len(open_leads),
        "leads_won": len(won),
        "leads_lost": len(lost),
        # Over decided leads, not all leads: counting still-open leads as
        # not-yet-converted understates the rate and moves whenever new leads
        # arrive, which makes it useless for comparing periods.
        "conversion_rate_percent": round(len(won) / decided * 100, 1) if decided else None,
        "conversion_basis": f"{len(won)} won of {decided} decided ({len(open_leads)} still open)",
        "pipeline_value": round(sum(float(l.get("value", 0)) for l in open_leads), 2),
        "pipeline_weighted": round(
            sum(float(l.get("value", 0)) * weights.get(l.get("stage", "new"), 0) for l in open_leads), 2
        ),
        "average_deal_size": round(
            sum(float(l.get("value", 0)) for l in won) / len(won), 2) if won else None,
        "revenue_by_service": {k: round(v, 2) for k, v in sorted(by_service.items(), key=lambda kv: -kv[1])},
        "expenses_by_category": {k: round(v, 2) for k, v in sorted(by_category.items(), key=lambda kv: -kv[1])},
        "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1]["revenue"])),
    }


class RecordBusinessDataTool(BaseTool):
    name = "record_business_data"
    description = (
        "Record a lead, income, or an expense.\n"
        "For a lead give `name` and optionally value, stage, source and contact_id. Link a "
        "lead to a saved contact with contact_id (from resolve_contact or add_contact) rather "
        "than retyping their email here - people live in the contact book, and a link keeps "
        "one copy of their details.\n"
        "Stages, in order: new, contacted, qualified, proposal, negotiation, won, lost."
    )
    parameters = [
        ToolParameter(name="record_type", type="string", enum=list(RECORD_TYPES),
                      description="lead, income, or expense."),
        ToolParameter(name="name", type="string", required=False,
                      description="Lead/company name, or what the income or expense was for."),
        ToolParameter(name="amount", type="number", required=False,
                      description="Amount for income/expense, or deal value for a lead."),
        ToolParameter(name="stage", type="string", required=False, enum=list(DEFAULT_STAGES),
                      description="Pipeline stage, for leads."),
        ToolParameter(name="source", type="string", required=False,
                      description="Where a lead came from: referral, google_ads, linkedin, cold_email."),
        ToolParameter(name="service", type="string", required=False,
                      description="Which service earned this income, e.g. 'website build'."),
        ToolParameter(name="category", type="string", required=False,
                      description="Expense category, e.g. 'software', 'advertising'."),
        ToolParameter(name="contact_id", type="string", required=False,
                      description="Id of the saved contact this relates to."),
        ToolParameter(name="date", type="string", required=False,
                      description="YYYY-MM-DD. Defaults to today."),
        ToolParameter(name="notes", type="string", required=False, description="Anything else worth keeping."),
    ]

    async def run(self, record_type: str, name: str = "", amount: float = 0,
                  stage: str = "", source: str = "", service: str = "", category: str = "",
                  contact_id: str = "", date: str = "", notes: str = "", **kwargs) -> ToolResult:
        kind = (record_type or "").lower()
        if kind not in RECORD_TYPES:
            return ToolResult(success=False, error=f"record_type must be one of {RECORD_TYPES}.")
        if kind == "lead" and not name:
            return ToolResult(success=False, error="A lead needs a name.")
        if kind in ("income", "expense") and not amount:
            return ToolResult(success=False, error=f"An {kind} record needs an amount.")
        if stage and stage not in DEFAULT_STAGES:
            return ToolResult(success=False, error=(
                f"Unknown stage '{stage}'. Use one of: {', '.join(DEFAULT_STAGES)}."
            ))
        if contact_id and contact_name(contact_id) is None:
            return ToolResult(success=False, error=(
                f"No contact with id '{contact_id}'. Use list_contacts or add_contact first, "
                f"or leave contact_id out."
            ))

        records = load_records()
        now = time.time()
        record: Dict[str, Any] = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "notes": notes,
            "contact_id": contact_id or None,
            "created_at": now,
        }
        if kind == "lead":
            record.update({"stage": stage or "new", "value": float(amount or 0), "source": source or ""})
            record["created_at"] = parse_date(date)
        else:
            record.update({"amount": float(amount), "date": parse_date(date)})
            record["service" if kind == "income" else "category"] = (service or category or "")

        records[kind].append(record)
        save_records(records)
        return ToolResult(success=True, output={
            "recorded": kind,
            "id": record["id"],
            "linked_contact": contact_name(contact_id),
            "record": {**record, "created_at": _iso(record["created_at"])},
        })


class UpdateLeadTool(BaseTool):
    name = "update_lead"
    description = (
        "Move a lead to a different stage, change its value, or link it to a contact. "
        "Moving a lead to 'won' is what makes it count towards conversion and revenue, so "
        "keep stages current."
    )
    parameters = [
        ToolParameter(name="lead_id", type="string", description="Lead id, from list_business_data."),
        ToolParameter(name="stage", type="string", required=False, enum=list(DEFAULT_STAGES),
                      description="New stage."),
        ToolParameter(name="value", type="number", required=False, description="New deal value."),
        ToolParameter(name="contact_id", type="string", required=False, description="Contact to link."),
        ToolParameter(name="notes", type="string", required=False, description="Replacement notes."),
    ]

    async def run(self, lead_id: str, stage: str = "", value: Optional[float] = None,
                  contact_id: str = "", notes: Optional[str] = None, **kwargs) -> ToolResult:
        if stage and stage not in DEFAULT_STAGES:
            return ToolResult(success=False, error=f"Unknown stage '{stage}'.")
        records = load_records()
        for lead in records["lead"]:
            if lead["id"] == lead_id:
                previous = lead.get("stage")
                if stage:
                    lead["stage"] = stage
                    lead["stage_changed_at"] = time.time()
                if value is not None:
                    lead["value"] = float(value)
                if contact_id:
                    if contact_name(contact_id) is None:
                        return ToolResult(success=False, error=f"No contact with id '{contact_id}'.")
                    lead["contact_id"] = contact_id
                if notes is not None:
                    lead["notes"] = notes
                save_records(records)
                return ToolResult(success=True, output={
                    "lead": lead["name"],
                    "stage": lead.get("stage"),
                    "moved_from": previous if stage and previous != stage else None,
                })
        return ToolResult(success=False, error=f"No lead with id '{lead_id}'.")


class ListBusinessDataTool(BaseTool):
    name = "list_business_data"
    description = (
        "List leads, income or expenses, optionally filtered by stage, source or period. "
        "Use this to see the records behind a number before acting on it."
    )
    parameters = [
        ToolParameter(name="record_type", type="string", enum=list(RECORD_TYPES) + ["all"],
                      required=False, description="What to list. Default leads."),
        ToolParameter(name="stage", type="string", required=False, description="Only leads in this stage."),
        ToolParameter(name="source", type="string", required=False, description="Only this lead source."),
        ToolParameter(name="period", type="string", required=False,
                      enum=["this_month", "last_month", "this_year", "last_30_days", "all"],
                      description="Time span. Default all."),
    ]

    async def run(self, record_type: str = "lead", stage: str = "", source: str = "",
                  period: str = "all", **kwargs) -> ToolResult:
        records = load_records()
        since, until, label = period_bounds(period)
        kinds = list(RECORD_TYPES) if record_type == "all" else [record_type or "lead"]
        output: Dict[str, Any] = {"period": label}

        for kind in kinds:
            if kind not in RECORD_TYPES:
                return ToolResult(success=False, error=f"Unknown record type '{kind}'.")
            field = "created_at" if kind == "lead" else "date"
            rows = within(records[kind], since, until, field=field)
            if kind == "lead":
                if stage:
                    rows = [r for r in rows if r.get("stage") == stage]
                if source:
                    rows = [r for r in rows if r.get("source") == source]
            output[kind] = [
                {**row, field: _iso(row.get(field, 0)), "contact_name": contact_name(row.get("contact_id"))}
                for row in sorted(rows, key=lambda r: r.get(field, 0), reverse=True)
            ]
        return ToolResult(success=True, output=output)


class BusinessDashboardTool(BaseTool):
    name = "business_dashboard"
    description = (
        "The overview: revenue, expenses, profit and margin; leads by stage; conversion rate; "
        "pipeline value both raw and weighted by stage; revenue by service; expenses by "
        "category; and performance by lead source. Set `compare_with` to see what changed "
        "against another period - that comparison is what answers 'what changed since last "
        "month'.\n"
        "Every figure is computed from the recorded leads, income and expenses, so if a number "
        "looks wrong the records are wrong; list_business_data shows them."
    )
    parameters = [
        ToolParameter(name="period", type="string", required=False,
                      enum=["this_month", "last_month", "this_year", "last_30_days", "all"],
                      description="Period to report on. Default this_month."),
        ToolParameter(name="compare_with", type="string", required=False,
                      enum=["last_month", "this_year", "all"],
                      description="Period to compare against."),
    ]

    async def run(self, period: str = "this_month", compare_with: str = "", **kwargs) -> ToolResult:
        since, until, label = period_bounds(period)
        current = summarize(since, until)
        output: Dict[str, Any] = {"period": label, "metrics": current}

        if compare_with:
            other_since, other_until, other_label = period_bounds(compare_with)
            previous = summarize(other_since, other_until)
            changes = {}
            for key in ("revenue", "expenses", "profit", "leads_total", "leads_won", "pipeline_value"):
                now_value, then_value = current.get(key) or 0, previous.get(key) or 0
                changes[key] = {
                    "now": now_value,
                    "then": then_value,
                    "change": round(now_value - then_value, 2),
                    "percent_change": (round((now_value - then_value) / then_value * 100, 1)
                                       if then_value else None),
                }
            output["comparison"] = {"against": other_label, "changes": changes,
                                    "their_metrics": previous}

        records = load_records()
        if not any(records[kind] for kind in RECORD_TYPES):
            output["note"] = (
                "There are no business records yet, so every figure is zero. Add leads, "
                "income and expenses with record_business_data - these numbers only "
                "describe what has been recorded."
            )
        return ToolResult(success=True, output=output)


class NextActionsTool(BaseTool):
    name = "business_next_actions"
    description = (
        "Which leads to contact next, and why. Ranks open leads by value weighted by stage "
        "probability, and surfaces the ones that have gone quiet - a high-value lead untouched "
        "for weeks is usually the most valuable thing to do today. Use this for 'who should I "
        "contact?' or 'where are the opportunities?'."
    )
    parameters = [
        ToolParameter(name="limit", type="number", required=False, description="How many to return (default 10)."),
        ToolParameter(name="stale_after_days", type="number", required=False,
                      description="Days without an update before a lead counts as gone quiet (default 14)."),
    ]

    async def run(self, limit: int = 10, stale_after_days: float = 14, **kwargs) -> ToolResult:
        records = load_records()
        weights = stage_weights()
        now = time.time()
        stale_seconds = float(stale_after_days) * 86400

        ranked = []
        for lead in records["lead"]:
            if lead.get("stage") not in OPEN_STAGES:
                continue
            last_touch = lead.get("stage_changed_at") or lead.get("created_at", now)
            idle_days = (now - last_touch) / 86400
            value = float(lead.get("value", 0))
            weight = weights.get(lead.get("stage", "new"), 0)
            reasons = []
            if value * weight > 0:
                reasons.append(f"weighted value {round(value * weight, 2)} ({lead.get('stage')} stage)")
            if now - last_touch > stale_seconds:
                reasons.append(f"no movement for {int(idle_days)} days")
            ranked.append({
                "lead_id": lead["id"],
                "name": lead.get("name"),
                "stage": lead.get("stage"),
                "value": value,
                "weighted_value": round(value * weight, 2),
                "days_since_update": int(idle_days),
                "stale": now - last_touch > stale_seconds,
                "contact_name": contact_name(lead.get("contact_id")),
                "contact_id": lead.get("contact_id"),
                "source": lead.get("source"),
                "why": "; ".join(reasons) or "open lead",
            })

        # Stale first among equals: a lead going cold is the time-sensitive one,
        # and weighted value alone would keep recommending the same top deal.
        ranked.sort(key=lambda r: (r["stale"], r["weighted_value"]), reverse=True)
        return ToolResult(success=True, output={
            "next_actions": ranked[:max(1, int(limit))],
            "open_leads": len(ranked),
            "stale_threshold_days": stale_after_days,
            "note": ("Ranked by stage-weighted value, with leads that have gone quiet first. "
                     "Use send_email or schedule_meeting to act, resolving the person through "
                     "their contact_id."),
        })
