"""
Image search - for when the user actually wants to SEE something (a
picture of a flower, what a landmark looks like, a product photo), as
opposed to tools/web_search.py's text results for factual questions.
Same no-key DuckDuckGo backend as web search, just its image endpoint.

The result carries a "visual" payload the orchestrator pushes directly to
the GUI (see core/orchestrator.py's visual_callback) so the images show up
regardless of how the model phrases its reply - the pop-up doesn't depend
on the model describing the picture in words.
"""
from __future__ import annotations

import asyncio
from typing import List

from ddgs import DDGS

from tools.base import BaseTool, ToolParameter, ToolResult


class SearchImagesTool(BaseTool):
    name = "search_images"
    description = (
        "Search the web for images and show them to the user. Use whenever the user asks to "
        "see, find, look up, or search for a picture/photo/image of something - e.g. 'show me "
        "a flower', 'what does the Eiffel Tower look like', 'find pictures of golden retrievers'. "
        "Do not use this for factual/text questions - use web_search for those."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="query", type="string", description="What to find images of."),
        ToolParameter(name="max_results", type="number", required=False, description="Default 6, max useful is around 8."),
    ]

    async def run(self, query: str, max_results: int = 6, **kwargs) -> ToolResult:
        try:
            loop = asyncio.get_event_loop()

            def _search():
                with DDGS() as ddgs:
                    return list(ddgs.images(query, max_results=max_results))

            results = await loop.run_in_executor(None, _search)
            if not results:
                return ToolResult(success=True, output={"results": [], "note": f"No images found for '{query}'."})

            items = [
                {
                    "url": r.get("image"),
                    "thumbnail": r.get("thumbnail") or r.get("image"),
                    "title": r.get("title", ""),
                    "source": r.get("source", ""),
                    "page_url": r.get("url", ""),
                }
                for r in results
                if r.get("image")
            ]

            return ToolResult(success=True, output={
                "results": items,
                "visual": {"type": "images", "query": query, "items": items},
            })
        except Exception as e:
            return ToolResult(success=False, error=str(e))
