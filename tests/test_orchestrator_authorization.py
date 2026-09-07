"""Tests for how the orchestrator acts on a SafetyGuard decision.

The guard can only report what it decided; whether the tool body actually runs
is the orchestrator's call, so the dry_run and one-shot pre-approval guarantees
are only real if this layer honors them.
"""
from __future__ import annotations

import sys
import types

import pytest

# core.orchestrator imports memory.vector_store, which imports chromadb at module
# level. The tests below never touch long-term memory, so a stub keeps them
# runnable without a multi-hundred-MB dependency installed.
sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core.orchestrator import Orchestrator  # noqa: E402
from tools.base import BaseTool, ToolParameter, ToolRegistry, ToolResult  # noqa: E402


class RecordingWriteTool(BaseTool):
    """Stands in for any risky tool; counts how often its body actually ran."""

    name = "write_file"
    description = "test double"
    parameters = [
        ToolParameter(name="path", type="string", description="path"),
        ToolParameter(name="content", type="string", description="content"),
    ]

    def __init__(self):
        self.calls = 0

    async def run(self, **kwargs) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, output="written")


def _orchestrator(guard_factory, dry_run: bool = False, confirm: bool = False):
    """A bare Orchestrator wired to a scripted guard. Built with __new__ to skip
    __init__, which would construct an Ollama client and a vector store."""
    guard, prompts = guard_factory(dry_run=dry_run, confirm=confirm)

    tool = RecordingWriteTool()
    registry = ToolRegistry()
    registry.register(tool)

    orch = Orchestrator.__new__(Orchestrator)
    orch.tool_registry = registry
    orch.safety_guard = guard
    orch._preapproved_this_turn = False
    return orch, tool, prompts


CALL = {"function": {"name": "write_file", "arguments": {"path": "/tmp/a.txt", "content": "x"}}}


@pytest.mark.asyncio
async def test_one_spoken_approval_authorizes_exactly_one_action(guard_factory):
    orch, tool, prompts = _orchestrator(guard_factory, confirm=False)
    orch._preapproved_this_turn = True   # e.g. "go ahead and write that file"

    first = await orch._execute_tool_call(CALL)
    assert first.success
    assert prompts == []                      # rode the pre-approval
    assert tool.calls == 1
    assert orch._preapproved_this_turn is False   # and spent it

    second = await orch._execute_tool_call(CALL)
    assert not second.success                 # prompted, and the user declined
    assert len(prompts) == 1
    assert tool.calls == 1                    # declined call never executed


@pytest.mark.asyncio
async def test_dry_run_reports_the_action_without_performing_it(guard_factory):
    orch, tool, prompts = _orchestrator(guard_factory, dry_run=True)

    result = await orch._execute_tool_call(CALL)

    assert tool.calls == 0                    # the whole point
    assert result.success                     # not an error - it did what was asked
    assert result.output["dry_run"] is True
    assert "/tmp/a.txt" in result.output["would_have_done"]
    assert prompts == []


@pytest.mark.asyncio
async def test_blocked_call_never_reaches_the_tool(guard_factory):
    orch, tool, _ = _orchestrator(guard_factory)
    blocked = {"function": {"name": "write_file",
                            "arguments": {"path": "~/.ssh/authorized_keys", "content": "x"}}}

    result = await orch._execute_tool_call(blocked)

    assert not result.success
    assert "Blocked" in (result.error or "")
    assert tool.calls == 0
