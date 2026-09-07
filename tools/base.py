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
