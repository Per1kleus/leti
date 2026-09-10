"""Tools for the controlled GUI mode.

None of these clicks anything. They decide whether the GUI is the right layer,
check what is on screen, and keep a sequence bounded; the clicking is done by the
existing mouse_click, keyboard_type, focus_window and launch_app tools, which the
orchestrator calls and SafetyGuard authorises exactly as always.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from core import computer_use
from tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("leti.tools.computer_use")


class ChooseComputerApproachTool(BaseTool):
    name = "choose_computer_approach"
    description = (
        "Before driving the screen with the mouse and keyboard, ask this which way to do "
        "the job. It answers 'tool', 'browser' or 'gui' and names the better option when "
        "there is one - a dedicated tool beats browser automation, and browser automation "
        "beats moving the mouse. Only when it answers 'gui' is clicking and typing on "
        "screen the right approach, and it opens a bounded session for it. Use this for "
        "requests like 'open this application and click Settings'."
    )
    parameters = [
        ToolParameter(name="request", type="string",
                      description="What the user actually asked for, in their words."),
    ]

    async def run(self, request: str, **kwargs) -> ToolResult:
        decision = computer_use.choose_layer(request)
        output = dict(decision)
        if decision["layer"] == computer_use.LAYER_GUI:
            session = computer_use.open_session(request)
            output["session_id"] = session.id
            output["steps_allowed"] = computer_use.MAX_STEPS
            output["next"] = ("Call read_screen, then verify_screen to say what you "
                              "expected, before the first click.")
        else:
            output["next"] = f"Use {decision['suggestion']} instead of the GUI."
        return ToolResult(success=True, output=output)


class VerifyScreenTool(BaseTool):
    name = "verify_screen"
    description = (
        "Check that the screen is what you expected before acting on it, and that the "
        "result happened after. Give the session id, what you expected to see, and what "
        "read_screen actually reported. Says whether it matches and whether the next step "
        "is allowed - it refuses once a session has taken too many steps or when the same "
        "action has already been tried without the screen changing, so a sequence stops "
        "rather than clicking blindly. Leti does not capture the screen on its own; you "
        "call read_screen when you need a look."
    )
    parameters = [
        ToolParameter(name="session_id", type="string",
                      description="From choose_computer_approach."),
        ToolParameter(name="expected", type="string",
                      description="What should be on screen, e.g. 'Settings window, General tab'."),
        ToolParameter(name="observed", type="string",
                      description="What read_screen reported."),
        ToolParameter(name="next_action", type="string", required=False,
                      description="The action you intend next, e.g. 'click Save'."),
    ]

    async def run(self, session_id: str, expected: str, observed: str,
                  next_action: str = "", **kwargs) -> ToolResult:
        session = computer_use.get_session(session_id)
        if session is None:
            return ToolResult(success=False,
                              error=f"No GUI session '{session_id}'. Start one with "
                                    "choose_computer_approach.")

        session.observe(observed)
        matches, why = session.expectation_holds(expected)
        if not matches:
            session.close("the screen was not what was expected")
            return ToolResult(success=False,
                              error=f"Stopping: {why}.",
                              output={"matches": False, "session": session.summary(),
                                      "note": ("Do not carry on clicking. Tell the user "
                                               "what you saw instead of what you expected.")})

        allowed, reason = session.may_act(next_action or "step", next_action)
        if next_action and allowed:
            session.record(next_action or "step", next_action)

        return ToolResult(success=True, output={
            "matches": True,
            "may_continue": allowed,
            "reason": reason,
            "session": session.summary(),
            "note": ("Go ahead with the existing tool for that action - mouse_click, "
                     "keyboard_type, focus_window - and verify again afterwards."
                     if allowed else f"Stop here: {reason}."),
        })


class EndComputerSessionTool(BaseTool):
    name = "end_computer_session"
    description = (
        "Close a GUI session when the errand is done or has to be abandoned, and get a "
        "summary of what was actually done on screen."
    )
    parameters = [
        ToolParameter(name="session_id", type="string", description="Which session."),
        ToolParameter(name="reason", type="string", required=False,
                      description="Why it ended, e.g. 'finished' or 'the dialog never appeared'."),
    ]

    async def run(self, session_id: str, reason: str = "finished", **kwargs) -> ToolResult:
        session = computer_use.get_session(session_id)
        if session is None:
            return ToolResult(success=False, error=f"No GUI session '{session_id}'.")
        return ToolResult(success=True, output=session.close(reason))
