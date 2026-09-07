"""Leti's web tool: search, and research built on top of search.

This is one tool module rather than a search tool plus a separate research
system. research_topic runs the same search backend web_search does and reads
pages through the same BrowserSession the browser_* tools drive (its headless
path - see tools/browser.py), so there is exactly one way Leti reaches the web.

What research adds over search is the part a single query can't do: several
queries at once so a question gets approached from more than one angle,
following the results onto the actual pages, and reporting which claims more
than one independent site actually carries. That last part is the difference
between "the internet says X" and "four unrelated domains say X" - the tool
computes corroboration mechanically and leaves interpretation to the model,
rather than inventing a confidence number.

Honest limits, since research output reads authoritative:
  - Corroboration counts distinct domains mentioning a claim's terms. Sites copy
    each other, so agreement is evidence of consensus, not of truth.
  - Page text is what the DOM renders. Sites that require login, block
    automation, or build content after load may come back empty; those are
    reported as unread sources rather than silently dropped.
"""
from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ddgs import DDGS

from tools.base import BaseTool, ToolParameter, ToolResult


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _search_sync(query: str, max_results: int) -> List[Dict[str, Any]]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


async def run_searches(queries: List[str], max_results: int = 5) -> Dict[str, Any]:
    """Run several searches concurrently and merge them, keeping provenance.

    Shared by web_search and research_topic so multi-query behaviour has one
    implementation. A URL found by more than one query keeps every query that
    found it - that overlap is itself a signal the result is central to the topic.
    """
    loop = asyncio.get_running_loop()
    results = await asyncio.gather(
        *(loop.run_in_executor(None, _search_sync, q, max_results) for q in queries),
        return_exceptions=True,
    )

    merged: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, str]] = []
    for query, outcome in zip(queries, results):
        if isinstance(outcome, BaseException):
            failures.append({"query": query, "error": str(outcome)})
            continue
        for hit in outcome:
            url = hit.get("href") or ""
            if not url:
                continue
            entry = merged.setdefault(url, {
                "title": hit.get("title"),
                "url": url,
                "domain": _domain(url),
                "snippet": hit.get("body"),
                "found_by": [],
            })
            if query not in entry["found_by"]:
                entry["found_by"].append(query)

    ranked = sorted(merged.values(), key=lambda r: len(r["found_by"]), reverse=True)
    return {"results": ranked, "failed_queries": failures}


def _claim_support(claim: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How many distinct domains carry the distinctive words of a claim.

    Deliberately crude and deliberately transparent: it matches the claim's
    content words against each source's text. It cannot tell "X is safe" from "X
    is not safe", so it is reported as 'sources mentioning this', never as proof.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for", "to",
        "and", "or", "it", "its", "this", "that", "with", "by", "as", "at", "from",
        "has", "have", "had", "be", "been", "will", "can", "may", "than", "then",
    }
    terms = [w for w in re.findall(r"[a-zA-Z0-9%.$-]{3,}", claim.lower()) if w not in stopwords]
    if not terms:
        return {"claim": claim, "supporting_domains": [], "domain_count": 0, "matched_terms": []}

    supporting, matched = [], set()
    for source in sources:
        text = (source.get("text") or "").lower()
        if not text:
            continue
        hits = [t for t in terms if t in text]
        # Most of the claim's distinctive words, not just one - a single shared
        # word makes almost any page look like it supports almost any claim.
        if len(hits) >= max(2, int(len(terms) * 0.6)):
            supporting.append(source["domain"])
            matched.update(hits)

    unique = sorted(set(d for d in supporting if d))
    return {
        "claim": claim,
        "supporting_domains": unique,
        "domain_count": len(unique),
        "matched_terms": sorted(matched),
    }


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web. Pass one query, or several at once to approach a question from "
        "different angles in a single call - results are merged and the ones found by more "
        "than one query are ranked first. This returns links and snippets; to answer from "
        "what a page actually says, follow up with browser_read_page, or use research_topic "
        "to search, read and cross-check in one step."
    )
    parameters = [
        ToolParameter(
            name="query", type="string", required=False,
            description="Search query. Use this for a single search.",
        ),
        ToolParameter(
            name="queries", type="array", items_type="string", required=False,
            description="Several queries to run at once, e.g. different phrasings or sub-questions.",
        ),
        ToolParameter(
            name="max_results", type="number", required=False,
            description="Results per query (default 5).",
        ),
    ]

    async def run(self, query: str = "", queries: Optional[List[str]] = None,
                  max_results: int = 5, **kwargs) -> ToolResult:
        all_queries = [q for q in ([query] if query else []) + list(queries or []) if str(q).strip()]
        if not all_queries:
            return ToolResult(success=False, error="Give a query, or a list of queries.")
        try:
            found = await run_searches(all_queries, int(max_results))
            if not found["results"] and found["failed_queries"]:
                return ToolResult(success=False, error=(
                    f"Every search failed: {found['failed_queries'][0]['error']}"
                ))
            return ToolResult(success=True, output={
                "queries": all_queries,
                "results": found["results"],
                "failed_queries": found["failed_queries"],
            })
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ResearchTopicTool(BaseTool):
    name = "research_topic"
    description = (
        "Research something properly: run several searches, open the most relevant pages, "
        "and return their actual content with sources attributed. Use this instead of "
        "web_search whenever the answer needs more than a snippet - researching a company, "
        "a competitor, a product, a market, a technical subject, or building a list of "
        "candidates that has to be filtered from real pages rather than guessed.\n"
        "Give several `queries` covering different angles of the question; that is what "
        "makes the research broad rather than one search repeated. Pass `verify` with "
        "specific claims to see how many independent domains carry them.\n"
        "Every finding comes back with the URL it came from - cite those, and say plainly "
        "when something rests on a single source."
    )
    parameters = [
        ToolParameter(name="topic", type="string", description="What is being researched, in a phrase."),
        ToolParameter(
            name="queries", type="array", items_type="string",
            description="Search queries covering different angles. Three to six works well.",
        ),
        ToolParameter(
            name="max_sources", type="number", required=False,
            description="How many pages to open and read (default 6, max 12).",
        ),
        ToolParameter(
            name="verify", type="array", items_type="string", required=False,
            description="Specific claims to check for corroboration across the sources found.",
        ),
        ToolParameter(
            name="results_per_query", type="number", required=False,
            description="Search results to consider per query before picking pages to read (default 5).",
        ),
    ]

    def __init__(self, session):
        # The same BrowserSession the browser_* tools use - see the module docstring.
        self.session = session

    async def run(self, topic: str, queries: List[str], max_sources: int = 6,
                  verify: Optional[List[str]] = None, results_per_query: int = 5,
                  **kwargs) -> ToolResult:
        queries = [str(q).strip() for q in (queries or []) if str(q).strip()]
        if not queries:
            return ToolResult(success=False, error="research_topic needs at least one query.")
        max_sources = max(1, min(int(max_sources), 12))

        try:
            found = await run_searches(queries, int(results_per_query))
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed: {e}")

        candidates = found["results"]
        if not candidates:
            return ToolResult(success=False, error=(
                f"No search results for {topic!r}. Tried: {queries}."
                + (f" Errors: {found['failed_queries']}" if found["failed_queries"] else "")
            ))

        # Prefer breadth of sources over depth on one site: a research answer built
        # from six pages of the same domain is one source wearing six hats.
        by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for result in candidates:
            by_domain[result["domain"]].append(result)
        chosen: List[Dict[str, Any]] = []
        round_index = 0
        while len(chosen) < max_sources:
            added = False
            for domain_results in by_domain.values():
                if round_index < len(domain_results) and len(chosen) < max_sources:
                    chosen.append(domain_results[round_index])
                    added = True
            if not added:
                break
            round_index += 1

        pages = await self.session.read_many([c["url"] for c in chosen])

        sources, unread = [], []
        for candidate, page in zip(chosen, pages):
            if page.get("ok") and page.get("text"):
                sources.append({
                    "url": page["url"],
                    "domain": candidate["domain"],
                    "title": page.get("title") or candidate["title"],
                    "found_by": candidate["found_by"],
                    "text": page["text"],
                    "truncated": page.get("truncated", False),
                })
            else:
                unread.append({
                    "url": candidate["url"],
                    "domain": candidate["domain"],
                    "title": candidate["title"],
                    "snippet": candidate["snippet"],
                    "reason": page.get("error", "no readable text on the page"),
                })

        verification = [_claim_support(c, sources) for c in (verify or []) if str(c).strip()]

        return ToolResult(success=True, output={
            "topic": topic,
            "queries": queries,
            "sources_read": sources,
            "sources_unread": unread,
            "distinct_domains": sorted({s["domain"] for s in sources if s["domain"]}),
            "other_results_not_read": [
                {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
                for r in candidates if r["url"] not in {c["url"] for c in chosen}
            ][:15],
            "verification": verification,
            "how_to_use": (
                "Answer from sources_read and cite each source's url. A claim carried by "
                "one domain is uncertain - say so. sources_unread were found but couldn't "
                "be read; their snippets are not a substitute for the page. Corroboration "
                "counts domains that mention a claim's terms, which shows consensus, not "
                "correctness."
            ),
        })
