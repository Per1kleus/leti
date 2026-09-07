"""
Live web search so the local model can answer questions beyond its training
data (current events, prices, documentation lookups) without needing a
hosted search API key.
"""
from __future__ import annotations

import asyncio
from typing import List

from ddgs import DDGS

from tools.base import BaseTool, ToolParameter, ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for current information and return top results with snippets."
    parameters = [
        ToolParameter(name="query", type="string", description="Search query."),
        ToolParameter(
            name="max_results", type="number", description="Number of results to return (default 5).",
            required=False,
        ),
    ]

    async def run(self, query: str, max_results: int = 5, **kwargs) -> ToolResult:
        try:
            loop = asyncio.get_event_loop()

            def _search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))

            results: List[dict] = await loop.run_in_executor(None, _search)
            formatted = [
                {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
                for r in results
            ]
            return ToolResult(success=True, output=formatted)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
