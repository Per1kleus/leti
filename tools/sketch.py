"""
Simple diagram/sketch generation - for when explaining a process, flow, or
structure would genuinely land better as a picture than a paragraph
(a multi-step process, a decision tree, a sequence of events). The model
writes Mermaid syntax (flowchart/sequence/etc.) directly since it's plain
text it can reason about like any other output; the GUI renders it.

This is deliberately narrow: it's for genuinely diagram-shaped content,
not a general "draw me a picture" tool (see tools/image_search.py for
"show me a picture of X" instead) - the system prompt is what actually
decides when a sketch is warranted, not this tool itself.
"""
from __future__ import annotations

from typing import List

from tools.base import BaseTool, ToolParameter, ToolResult


class CreateSketchTool(BaseTool):
    name = "create_sketch"
    description = (
        "Show a simple diagram (flowchart, sequence, or decision tree) using Mermaid syntax. "
        "Use when explaining a multi-step process, a decision flow, or a sequence of events "
        "would genuinely be clearer as a picture than a paragraph - not for every explanation, "
        "just ones that are actually diagram-shaped. For 'show me a picture of X' use "
        "search_images instead - this is for sketches you draw, not photos."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(
            name="mermaid_code", type="string",
            description=(
                "Valid Mermaid syntax, e.g. \"flowchart TD\\nA[Start] --> B{Decision}\\n"
                "B -->|Yes| C[Do X]\\nB -->|No| D[Do Y]\". Keep it simple - a handful of nodes, "
                "not an exhaustive diagram."
            ),
        ),
        ToolParameter(name="title", type="string", required=False, description="Short title for the diagram."),
    ]

    async def run(self, mermaid_code: str, title: str = "", **kwargs) -> ToolResult:
        if not mermaid_code.strip():
            return ToolResult(success=False, error="mermaid_code was empty.")
        return ToolResult(success=True, output={
            "note": "Sketch shown to the user.",
            "visual": {"type": "diagram", "title": title, "mermaid": mermaid_code},
        })
