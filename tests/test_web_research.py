"""Tests for the web tool: search, research, and corroboration.

The research output reads authoritative, so what's tested here is mostly the
places it could mislead - a failed fetch counted as a source, one site's pages
posing as several sources, a claim "confirmed" by a single domain.
"""
from __future__ import annotations

import pytest

from tools.web_search import ResearchTopicTool, WebSearchTool, _claim_support, _domain, run_searches


class FakeSession:
    """Stands in for BrowserSession. Research must work against whatever the web
    returns, including pages that don't load."""

    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    async def read_many(self, urls, **kwargs):
        self.requested = list(urls)
        return [self.pages.get(u, {"ok": False, "url": u, "error": "not found"}) for u in urls]


def _hit(url, domain, found_by=("q1",)):
    return {"title": f"T {domain}", "url": url, "domain": domain,
            "snippet": "snip", "found_by": list(found_by)}


@pytest.mark.parametrize("url,expected", [
    ("https://www.example.com/a", "example.com"),
    ("http://sub.example.co.uk/b", "sub.example.co.uk"),
    ("not a url", ""),
])
def test_domain_extraction(url, expected):
    assert _domain(url) == expected


# --- Corroboration -------------------------------------------------------------

def test_claim_supported_by_several_domains_is_counted_once_per_domain():
    sources = [
        {"domain": "a.com", "text": "The market grew 18% in 2024 across the region."},
        {"domain": "b.org", "text": "Reports say the market grew 18% in 2024."},
        {"domain": "a.com", "text": "Another page: the market grew 18% in 2024."},
    ]
    result = _claim_support("The market grew 18% in 2024", sources)
    assert result["domain_count"] == 2, "same domain twice is one source, not two"
    assert set(result["supporting_domains"]) == {"a.com", "b.org"}


def test_unsupported_claim_gets_no_support():
    sources = [{"domain": "a.com", "text": "Entirely unrelated content about gardening."}]
    assert _claim_support("Patras is the capital of France", sources)["domain_count"] == 0


def test_a_single_shared_word_is_not_support():
    """Otherwise almost any page 'supports' almost any claim."""
    sources = [{"domain": "a.com", "text": "The market is closed on Sundays."}]
    assert _claim_support("The market grew 18% in 2024", sources)["domain_count"] == 0


def test_claim_with_no_content_words_is_reported_not_crashed():
    assert _claim_support("the a of in", [{"domain": "a.com", "text": "x"}])["domain_count"] == 0


# --- Research ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pages_that_fail_to_load_are_reported_not_counted(monkeypatch):
    """A source that couldn't be read must never look like one that was."""
    import tools.web_search as ws

    async def fake(queries, max_results=5):
        return {"results": [_hit("https://a.com/1", "a.com"), _hit("https://b.com/1", "b.com")],
                "failed_queries": []}

    monkeypatch.setattr(ws, "run_searches", fake)
    session = FakeSession({
        "https://a.com/1": {"ok": True, "url": "https://a.com/1", "title": "A", "text": "content"},
        "https://b.com/1": {"ok": False, "url": "https://b.com/1", "error": "HTTP 404 Not Found"},
    })

    result = await ResearchTopicTool(session).run(topic="t", queries=["q"], max_sources=2)

    assert [s["domain"] for s in result.output["sources_read"]] == ["a.com"]
    assert [u["domain"] for u in result.output["sources_unread"]] == ["b.com"]
    assert "404" in result.output["sources_unread"][0]["reason"]


@pytest.mark.asyncio
async def test_reading_spreads_across_domains_before_going_deep(monkeypatch):
    """Six pages from one site is one source wearing six hats."""
    import tools.web_search as ws

    async def fake(queries, max_results=5):
        return {"results": [
            _hit("https://a.com/1", "a.com"), _hit("https://a.com/2", "a.com"),
            _hit("https://a.com/3", "a.com"), _hit("https://b.com/1", "b.com"),
            _hit("https://c.com/1", "c.com"),
        ], "failed_queries": []}

    monkeypatch.setattr(ws, "run_searches", fake)
    session = FakeSession({})
    await ResearchTopicTool(session).run(topic="t", queries=["q"], max_sources=3)

    domains = [_domain(u) for u in session.requested]
    assert sorted(domains) == ["a.com", "b.com", "c.com"], domains


@pytest.mark.asyncio
async def test_research_reports_when_every_search_failed(monkeypatch):
    import tools.web_search as ws

    async def fake(queries, max_results=5):
        return {"results": [], "failed_queries": [{"query": "q", "error": "network down"}]}

    monkeypatch.setattr(ws, "run_searches", fake)
    result = await ResearchTopicTool(FakeSession({})).run(topic="t", queries=["q"])
    assert result.success is False


@pytest.mark.asyncio
async def test_research_needs_at_least_one_query():
    result = await ResearchTopicTool(FakeSession({})).run(topic="t", queries=[])
    assert result.success is False


# --- Multi-query search --------------------------------------------------------

@pytest.mark.asyncio
async def test_merged_results_keep_which_queries_found_them(monkeypatch):
    """Overlap between queries is a signal a result is central to the topic."""
    import tools.web_search as ws

    def fake_sync(query, max_results):
        common = {"title": "Shared", "href": "https://shared.com/x", "body": "b"}
        return [common, {"title": query, "href": f"https://{query}.com/x", "body": "b"}]

    monkeypatch.setattr(ws, "_search_sync", fake_sync)
    merged = await run_searches(["alpha", "beta"], 5)

    top = merged["results"][0]
    assert top["url"] == "https://shared.com/x"
    assert sorted(top["found_by"]) == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_one_failing_query_does_not_lose_the_others(monkeypatch):
    import tools.web_search as ws

    def fake_sync(query, max_results):
        if query == "bad":
            raise RuntimeError("rate limited")
        return [{"title": "ok", "href": "https://ok.com/1", "body": "b"}]

    monkeypatch.setattr(ws, "_search_sync", fake_sync)
    merged = await run_searches(["good", "bad"], 5)

    assert len(merged["results"]) == 1
    assert merged["failed_queries"][0]["query"] == "bad"


@pytest.mark.asyncio
async def test_web_search_accepts_one_query_or_many():
    tool = WebSearchTool()
    assert (await tool.run()).success is False   # neither given
