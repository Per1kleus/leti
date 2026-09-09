"""
BaseTool defines the contract every tool implements so the orchestrator can:
  1. Advertise tools to the LLM in Ollama's function-calling schema format.
  2. Route a model-issued tool call to the right Python coroutine.
  3. Enforce SafetyGuard authorization uniformly before any tool body runs.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolParameter:
    name: str
    type: str                      # "string" | "number" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    items_type: Optional[str] = None   # only used when type == "array"


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "output": self.output, "error": self.error}


class BaseTool(abc.ABC):
    name: str = "base_tool"
    description: str = "Base tool. Override in subclasses."
    parameters: List[ToolParameter] = field(default_factory=list)

    @abc.abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Execute the tool. Must NOT perform its own permission checks -
        the orchestrator/safety guard handles authorization before calling this."""
        raise NotImplementedError

    def action_case(self, arguments: Dict[str, Any]) -> Optional[str]:
        """Which *kind* of call this is, for tools whose consequence varies by argument.

        A tool that does one thing returns None and is classified purely by its
        `action:` in permissions.yaml. A tool that legitimately covers two acts of
        different weight - launch_app opening a web page in the browser is not the
        same as launch_app starting an arbitrary program - names the case here.

        This deliberately reports a case, not a risk class: permissions.yaml still
        decides what each case is worth (`action_by_case:`), so the answer to "what
        needs confirming" stays in the one file the user edits, and a tool cannot
        lower its own gate by returning a laxer class. An unrecognised case falls
        back to the tool's plain `action:`.
        """
        return None

    def to_ollama_schema(self) -> Dict[str, Any]:
        """Converts this tool's definition into Ollama's function-calling tool schema."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: Dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            if p.type == "array" and p.items_type:
                prop["items"] = {"type": p.items_type}
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Holds all instantiated tools and exposes lookup + schema aggregation."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def all_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_ollama_schema() for t in self._tools.values()]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def approx_schema_tokens(self) -> int:
        """Roughly how much of the context window the tool list occupies.

        Every schema is sent on every call and on every iteration of the
        tool-calling loop, so this is a fixed toll on num_ctx before the system
        prompt, memory or the conversation get any. Four characters per token is
        the usual rule of thumb and errs low for JSON, which is denser - that is
        the right direction for a check whose job is to warn early.
        """
        import json

        return len(json.dumps(self.all_schemas())) // 4
