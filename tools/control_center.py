"""Tools behind the Permission Center, the Diagnostics panel and the mode switch.

Both read what already exists. The permission tools read and write
config/permissions.yaml and safety.require_confirmation_for - the two things
SafetyGuard itself reads - so there is no second copy of the rules anywhere. The
diagnostics tool reports timings other parts of Leti recorded while working, and
says "Unavailable" for anything it cannot honestly obtain.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from core import diagnostics, permission_center
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.control_center")


class SwitchModeTool(BaseTool):
    name = "switch_mode"
    description = (
        "Report which mode Leti is in, or leave the current one. Only for an explicit "
        "request - 'what mode am I in', 'exit', 'switch to business mode'. Doing a coding "
        "or business task is not asking for a mode; never switch because of what a "
        "request is about."
    )
    parameters = [
        ToolParameter(name="mode", type="string",
                      description="Which mode to switch to. 'status' just reports.",
                      enum=["default", "coding", "business", "status"]),
        ToolParameter(name="project", type="string", required=False,
                      description="Project to work in, for coding or business."),
        ToolParameter(name="path", type="string", required=False,
                      description="Folder to work in, for coding."),
        ToolParameter(name="repository", type="string", required=False,
                      description="GitHub repository as owner/name, for coding."),
        ToolParameter(name="business_name", type="string", required=False,
                      description="Which business this session is about, for business."),
    ]

    async def run(self, mode: str, project: str = "", path: str = "", repository: str = "",
                  business_name: str = "", **kwargs) -> ToolResult:
        from core import modes

        wanted = str(mode or "status").strip().lower()
        if wanted in ("status", ""):
            return ToolResult(success=True, output=modes.describe(_REGISTRY))

        if wanted in ("exit", "leave", "off", modes.DEFAULT):
            result = modes.leave()
            return ToolResult(success=True, output={
                **result, "note": ("Back to the general assistant. The specialised "
                                   "workspace is released; projects, memory, tasks and "
                                   "permissions are untouched.")})

        result = modes.enter(wanted)
        if not result.get("ok"):
            return ToolResult(success=False, error=result.get("error", "Couldn't switch mode."))

        details = await _open_workspace_for(wanted, project=project, path=path,
                                            repository=repository,
                                            business_name=business_name)
        return ToolResult(success=True, output={**result, **details})


async def _open_workspace_for(mode_name: str, project: str = "", path: str = "",
                              repository: str = "", business_name: str = "") -> dict:
    """Set up whichever workspace the mode uses. Reads; starts nothing."""
    from core import modes

    if mode_name == modes.CODING:
        from tools.coding_agent import open_coding_workspace

        return await open_coding_workspace(path=path, project=project, repository=repository)
    if mode_name == modes.BUSINESS:
        from core import business

        space = business.open_workspace(business_name=business_name, project=project)
        return {"workspace": space.summary(), "sources": business.connected_sources(),
                "note": ("Business Mode reads the records, contacts, documents and tasks "
                         "Leti already has. Anything it cannot see is listed in "
                         "'sources' rather than reported as empty.")}
    return {}

_REGISTRY: Optional[Any] = None


def set_registry(registry) -> None:
    global _REGISTRY
    _REGISTRY = registry


class ReviewPermissionsTool(BaseTool):
    name = "review_permissions"
    description = (
        "Show what Leti is currently allowed to do, grouped the way a person would "
        "think about it - files, browser, email, calendar, computer, system - with "
        "whether each is automatic or asks first. Use for 'what can you do without "
        "asking', 'what are your permissions', or before changing one."
    )
    parameters = []

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=permission_center.overview(_REGISTRY))


class ChangePermissionTool(BaseTool):
    name = "change_permission"
    description = (
        "Change what Leti may do without asking. Either make a whole class of action "
        "confirm or stop confirming (read, execute, modify, external, critical), or "
        "reclassify one tool. This edits the same settings SafetyGuard enforces, so it "
        "takes effect immediately. Irreversible actions always ask and cannot be "
        "switched off. Confirm with the user before loosening anything."
    )
    parameters = [
        ToolParameter(name="scope", type="string",
                      description="'class' for a whole class, or 'tool' for one tool.",
                      enum=["class", "tool"]),
        ToolParameter(name="target", type="string",
                      description="The class (e.g. 'modify') or the tool name."),
        ToolParameter(name="must_confirm", type="boolean", required=False,
                      description="For scope=class: true to make it ask, false to stop asking."),
        ToolParameter(name="action_class", type="string", required=False,
                      description="For scope=tool: the class to give it.",
                      enum=list(permission_center.CLASSES)),
    ]

    async def run(self, scope: str, target: str, must_confirm: bool = True,
                  action_class: str = "", **kwargs) -> ToolResult:
        if scope == "class":
            result = permission_center.set_class_confirmation(target, must_confirm)
        elif scope == "tool":
            if not action_class:
                return ToolResult(success=False,
                                  error="Changing one tool needs an action_class.")
            result = permission_center.set_tool_class(target, action_class, _REGISTRY)
        else:
            return ToolResult(success=False, error="scope must be 'class' or 'tool'.")

        if not result.get("ok"):
            return ToolResult(success=False, error=result.get("error"))
        return ToolResult(success=True, output=result)


class DiagnosticsTool(BaseTool):
    name = "get_diagnostics"
    description = (
        "Report how Leti is running: the configured model and context size, how long "
        "the last response and the last tool call took, how many tools were exposed for "
        "the last request, CPU and memory, active tasks and watches. Use for 'how are "
        "you doing', 'why are you slow', 'what are you working on'. Anything that cannot "
        "be measured honestly is reported as Unavailable rather than guessed."
    )
    parameters = []

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=diagnostics.snapshot(_REGISTRY))
