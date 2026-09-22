"""The new layers, exercised through a real turn rather than in isolation.

A fake model server stands in for Ollama and a real ToolRegistry, SafetyGuard,
SessionMemory and Orchestrator do everything else, so these fail if the wiring
comes apart even when every module still passes its own tests.
"""
from __future__ import annotations

import sys
import types

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import connections, context_engine, diagnostics, entities, modes  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from tools.base import BaseTool, ToolParameter, ToolRegistry, ToolResult  # noqa: E402


class FakeLLM:
    """Replays a script of assistant messages and records what it was shown."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    async def chat(self, messages, tools=None):
        self.seen.append({"messages": list(messages), "tools": list(tools or [])})
        return {"message": self.script.pop(0) if self.script
                else {"content": "done", "tool_calls": None}}


class WriteTool(BaseTool):
    name = "write_file"
    description = "write a file"
    parameters = [ToolParameter(name="path", type="string", description="path"),
                  ToolParameter(name="content", type="string", description="content")]

    async def run(self, path="", content="", **kwargs):
        return ToolResult(success=True, output={"success": True, "message": "Saved!"})


class Memory:
    """The real shape, without sqlite."""

    def __init__(self):
        self.turns = []
        self._referents = []

    def add_turn(self, role, content, session_id="default"):
        self.turns.append({"role": role, "content": content})

    def get_recent_messages(self):
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def recent_referents(self):
        return list(self._referents)

    def note_referents(self, items):
        self._referents = list(items)


def _orchestrator(guard_factory, script=(), confirm=True):
    guard, prompts = guard_factory(confirm=confirm, confirm_classes=[])
    registry = ToolRegistry()
    registry.register(WriteTool())
    llm = FakeLLM(script)

    orch = Orchestrator.__new__(Orchestrator)
    orch.llm_client = llm
    orch.tool_registry = registry
    orch.safety_guard = guard
    orch.session_memory = Memory()
    orch.vector_memory = None
    orch.speak_callback = None
    orch.visual_callback = None
    orch.state = None
    orch._state_listeners = []
    orch._preapproved_this_turn = False
    orch._checked_watches_this_session = True
    import asyncio

    orch._turn_lock = asyncio.Lock()
    entities.use_conversation_source(orch.session_memory.get_recent_messages)
    return orch, llm, prompts


@pytest.fixture(autouse=True)
def _default_mode():
    modes.reset_for_tests()
    yield
    modes.reset_for_tests()


# --- A turn assembles chosen context ---------------------------------------------------

@pytest.mark.asyncio
async def test_a_turn_goes_through_the_context_engine(guard_factory):
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "It is 12 degrees."}])
    await orch.handle_user_input("what's the weather?")
    assert orch._context is not None
    assert orch._context.report()["included"]
    assert llm.seen[0]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_the_context_decision_reaches_the_diagnostics_panel(guard_factory):
    diagnostics.reset()
    orch, _, _ = _orchestrator(guard_factory, [{"content": "ok"}])
    await orch.handle_user_input("what do you remember about my preferences?")
    section = diagnostics.context_section()
    assert section["status"] == "ok" and section["chars"] > 0


@pytest.mark.asyncio
async def test_a_broken_context_engine_does_not_break_the_turn(guard_factory, monkeypatch):
    async def explode(*args, **kwargs):
        raise RuntimeError("the engine is down")

    monkeypatch.setattr(context_engine, "build", explode)
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "still answered"}])
    answer = await orch.handle_user_input("hello")
    assert answer == "still answered"
    assert llm.seen[0]["messages"][0]["role"] == "system"


# --- Verification reaches the model ------------------------------------------------------

@pytest.mark.asyncio
async def test_an_unconfirmed_write_is_flagged_in_the_tool_result(guard_factory, tmp_path):
    call = {"content": None, "tool_calls": [
        {"function": {"name": "write_file",
                      "arguments": {"path": str(tmp_path / "nope.txt"), "content": "x"}}}]}
    orch, llm, _ = _orchestrator(guard_factory, [call, {"content": "I tried."}])
    await orch.handle_user_input("save that to a file")
    tool_messages = [m for m in llm.seen[-1]["messages"] if m.get("role") == "tool"]
    assert tool_messages, "the tool result never reached the model"
    assert "FAILED" in tool_messages[-1]["content"]
    assert "Do not report this as done" in tool_messages[-1]["content"]


@pytest.mark.asyncio
async def test_a_confirmed_write_adds_no_noise(guard_factory, tmp_path):
    path = tmp_path / "yes.txt"
    path.write_text("x")
    call = {"content": None, "tool_calls": [
        {"function": {"name": "write_file",
                      "arguments": {"path": str(path), "content": "x"}}}]}
    orch, llm, _ = _orchestrator(guard_factory, [call, {"content": "Saved."}])
    await orch.handle_user_input("save that to a file")
    tool_messages = [m for m in llm.seen[-1]["messages"] if m.get("role") == "tool"]
    assert "NOT VERIFIED" not in tool_messages[-1]["content"]
    assert "how_to_report" not in tool_messages[-1]["content"]


@pytest.mark.asyncio
async def test_verification_never_turns_a_success_into_a_failure(guard_factory, tmp_path):
    path = tmp_path / "ok.txt"
    path.write_text("x")
    call = {"content": None, "tool_calls": [
        {"function": {"name": "write_file",
                      "arguments": {"path": str(path), "content": "x"}}}]}
    orch, llm, _ = _orchestrator(guard_factory, [call, {"content": "done"}])
    answer = await orch.handle_user_input("save it")
    assert answer == "done"


# --- Connections learn from real calls ------------------------------------------------------

@pytest.mark.asyncio
async def test_a_failing_external_call_is_remembered(guard_factory, monkeypatch):
    connections.forget_outcomes()
    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: True)

    class FailingEmail(BaseTool):
        name = "send_email"
        description = "send"
        parameters = [ToolParameter(name="to", type="string", description="to")]

        async def run(self, **kwargs):
            return ToolResult(success=False, error="authentication rejected")

    orch, llm, _ = _orchestrator(guard_factory, [
        {"content": None, "tool_calls": [
            {"function": {"name": "send_email", "arguments": {"to": "a@b.com"}}}]},
        {"content": "I couldn't send it."}])
    orch.tool_registry.register(FailingEmail())
    await orch.handle_user_input("email a@b.com")
    assert connections.status("email")["state"] == connections.FAILING
    connections.forget_outcomes()


# --- Commands are answered without the model ---------------------------------------------------

@pytest.mark.asyncio
async def test_run_full_diagnostics_never_reaches_the_model(guard_factory):
    orch, llm, _ = _orchestrator(guard_factory, [])
    answer = await orch.handle_user_input("run full leti diagnostics")
    assert llm.seen == [], "the diagnostics command went through the model"
    assert "Leti diagnostics" in answer and "nothing was changed" in answer


@pytest.mark.asyncio
async def test_the_diagnostics_answer_names_what_was_not_tested(guard_factory):
    orch, _, _ = _orchestrator(guard_factory, [])
    answer = await orch.handle_user_input("run a system check")
    assert "NOT TESTED" in answer or "NOT CONFIGURED" in answer


@pytest.mark.asyncio
async def test_a_mode_command_still_never_reaches_the_model(guard_factory):
    orch, llm, _ = _orchestrator(guard_factory, [])
    await orch.handle_user_input("enter coding mode")
    assert llm.seen == [] and modes.current() == modes.CODING
    await orch.handle_user_input("exit")     # what the mode description tells you to say
    assert llm.seen == [] and modes.current() == modes.DEFAULT


@pytest.mark.asyncio
async def test_a_business_sounding_request_still_gets_an_ordinary_turn(guard_factory):
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "Here they are."}])
    await orch.handle_user_input("check my customer emails and update the invoice")
    assert modes.current() == modes.DEFAULT
    assert llm.seen, "the request should have gone to the model"


# --- Entity resolution sees this conversation ----------------------------------------------------

@pytest.mark.asyncio
async def test_what_was_said_this_turn_is_available_to_entity_resolution(guard_factory):
    orch, _, _ = _orchestrator(guard_factory, [{"content": "ok"}])
    await orch.handle_user_input("let's chase Acme Holdings about the quote")
    assert "acme holdings" in entities.gather().conversation_text


@pytest.mark.asyncio
async def test_the_conversation_source_is_a_reference_not_a_copy(guard_factory):
    orch, _, _ = _orchestrator(guard_factory, [{"content": "ok"}, {"content": "ok"}])
    await orch.handle_user_input("first message about Northwind")
    before = entities.gather().conversation_text
    await orch.handle_user_input("second message about Contoso")
    after = entities.gather().conversation_text
    assert "northwind" in before and "contoso" in after and "contoso" not in before


@pytest.mark.asyncio
async def test_a_bare_exit_works_as_the_mode_description_says_it_does(guard_factory):
    """The description says: Say "exit" to go back. It has to be true."""
    orch, llm, _ = _orchestrator(guard_factory, [])
    await orch.handle_user_input("enter business mode")
    assert modes.current() == modes.BUSINESS
    assert "exit" in orch._describe_mode()
    await orch.handle_user_input("exit")
    assert modes.current() == modes.DEFAULT and llm.seen == []


@pytest.mark.asyncio
async def test_a_bare_exit_in_default_mode_is_an_ordinary_request(guard_factory):
    """There is nothing to exit, so "exit" means whatever the model makes of it -
    it must not be swallowed as a command."""
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "Exit what?"}])
    answer = await orch.handle_user_input("exit")
    assert answer == "Exit what?" and llm.seen


@pytest.mark.asyncio
async def test_a_failure_with_no_account_configured_says_which_account(guard_factory,
                                                                      monkeypatch):
    monkeypatch.setattr("core.settings_editor.section_is_configured", lambda name: False)
    connections.forget_outcomes()

    class Unreachable(BaseTool):
        name = "send_email"
        description = "send"
        parameters = [ToolParameter(name="to", type="string", description="to")]

        async def run(self, **kwargs):
            return ToolResult(success=False, error="connection refused")

    orch, llm, _ = _orchestrator(guard_factory, [
        {"content": None, "tool_calls": [
            {"function": {"name": "send_email", "arguments": {"to": "a@b.com"}}}]},
        {"content": "There's no mail account set up."}])
    orch.tool_registry.register(Unreachable())
    await orch.handle_user_input("email a@b.com")
    tool_messages = [m for m in llm.seen[-1]["messages"] if m.get("role") == "tool"]
    assert "No email connection is set up" in tool_messages[-1]["content"]
    assert "Connections" in tool_messages[-1]["content"]
    connections.forget_outcomes()


@pytest.mark.asyncio
async def test_a_failure_from_a_tool_that_needs_no_account_is_left_alone(guard_factory,
                                                                        tmp_path):
    class Broken(BaseTool):
        name = "write_file"
        description = "write"
        parameters = [ToolParameter(name="path", type="string", description="path")]

        async def run(self, **kwargs):
            return ToolResult(success=False, error="disk full")

    orch, llm, _ = _orchestrator(guard_factory, [
        {"content": None, "tool_calls": [
            {"function": {"name": "write_file", "arguments": {"path": "/x"}}}]},
        {"content": "The disk is full."}])
    orch.tool_registry.register(Broken())
    await orch.handle_user_input("save it")
    tool_messages = [m for m in llm.seen[-1]["messages"] if m.get("role") == "tool"]
    assert "Connections" not in tool_messages[-1]["content"]


@pytest.mark.asyncio
async def test_the_context_package_names_what_the_request_names(guard_factory):
    orch, _, _ = _orchestrator(guard_factory, [{"content": "ok"}])
    await orch.handle_user_input("read turbine.py and summarise it")
    assert "turbine.py" in orch._context.report()["entities"]


# --- The Intent Layer's veto, end to end ------------------------------------------------

@pytest.mark.asyncio
async def test_a_remark_is_shown_no_tools_at_all(guard_factory):
    """The structural version of "don't act on that": there is nothing to call."""
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "Glad you think so."}])
    await orch.handle_user_input("That's interesting.")
    assert llm.seen[0]["tools"] == []


@pytest.mark.asyncio
async def test_a_hedged_suggestion_is_shown_no_tools(guard_factory):
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "We could."}])
    await orch.handle_user_input("Maybe we should open the project.")
    assert llm.seen[0]["tools"] == []


@pytest.mark.asyncio
async def test_a_real_instruction_still_gets_its_tools(guard_factory, tmp_path):
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "Done."}])
    await orch.handle_user_input(f"write hello into {tmp_path / 'x.txt'}")
    assert llm.seen[0]["tools"], "an instruction was shown no tools"


@pytest.mark.asyncio
async def test_an_instruction_behind_a_remark_still_gets_its_tools(guard_factory,
                                                                   tmp_path):
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "Done."}])
    await orch.handle_user_input(
        f"interesting - now write hello into {tmp_path / 'x.txt'}")
    assert llm.seen[0]["tools"], "the instruction behind the remark lost its tools"


@pytest.mark.asyncio
async def test_the_reading_reaches_the_diagnostics_panel(guard_factory):
    from core import diagnostics

    diagnostics.reset()
    orch, _, _ = _orchestrator(guard_factory, [{"content": "ok"}])
    await orch.handle_user_input("open the project")
    section = diagnostics.intent_section()
    assert section["status"] == "ok" and section["kind"] == "command"


# --- Lifecycle commands, end to end -------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_never_reaches_the_model(guard_factory, tmp_path, monkeypatch):
    from core import task_history, task_manager

    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "history.json")
    task = task_manager.create_task("o", ["a"], "the download")

    orch, llm, _ = _orchestrator(guard_factory, [])
    answer = await orch.handle_user_input("stop")
    assert llm.seen == [], "a stop went through the model"
    assert task_manager.get_task(task["id"])["status"] == "cancelled"
    assert "Nothing further will start" in answer


@pytest.mark.asyncio
async def test_a_lifecycle_command_with_nothing_running_still_answers(guard_factory,
                                                                     tmp_path,
                                                                     monkeypatch):
    from core import task_history, task_manager

    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "history.json")
    orch, llm, _ = _orchestrator(guard_factory, [])
    answer = await orch.handle_user_input("pause")
    assert "nothing" in answer.lower() and llm.seen == []


@pytest.mark.asyncio
async def test_what_are_you_doing_is_answered_from_the_task_store(guard_factory,
                                                                  tmp_path,
                                                                  monkeypatch):
    from core import task_history, task_manager

    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "history.json")
    task_manager.create_task("o", ["a"], "the download")
    orch, llm, _ = _orchestrator(guard_factory, [])
    answer = await orch.handle_user_input("what are you doing?")
    assert "the download" in answer and llm.seen == []


@pytest.mark.asyncio
async def test_an_ordinary_request_mentioning_stopping_is_not_a_stop(guard_factory,
                                                                     tmp_path,
                                                                     monkeypatch):
    from core import task_history, task_manager

    monkeypatch.setattr(task_manager, "store_path", lambda: tmp_path / "tasks.json")
    monkeypatch.setattr(task_history, "store_path", lambda: tmp_path / "history.json")
    task = task_manager.create_task("o", ["a"], "the download")
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "Here's how."}])
    await orch.handle_user_input("how do I stop a systemd service?")
    assert llm.seen, "an ordinary question was swallowed as a lifecycle command"
    assert task_manager.get_task(task["id"])["status"] != "cancelled"


# --- Runtime resource claims at the tool boundary -------------------------------------

@pytest.mark.asyncio
async def test_a_tool_call_claims_and_releases_what_it_touches(guard_factory, tmp_path):
    from core import resources

    resources.clear()
    path = tmp_path / "x.txt"
    call = {"content": None, "tool_calls": [
        {"function": {"name": "write_file",
                      "arguments": {"path": str(path), "content": "x"}}}]}
    orch, llm, _ = _orchestrator(guard_factory, [call, {"content": "done"}])
    await orch.handle_user_input("save it")
    # Released on the way out, whatever happened.
    assert resources.snapshot()["held"] == []
    resources.clear()


@pytest.mark.asyncio
async def test_a_held_resource_blocks_a_second_writer(guard_factory, tmp_path):
    from core import resources

    resources.clear()
    path = tmp_path / "shared.txt"
    resources.acquire(resources.identity(resources.FILE, str(path)),
                      resources.WRITE, "some-other-task")
    call = {"content": None, "tool_calls": [
        {"function": {"name": "write_file",
                      "arguments": {"path": str(path), "content": "x"}}}]}
    orch, llm, _ = _orchestrator(guard_factory, [call, {"content": "I waited."}])
    await orch.handle_user_input("save it")
    tool_messages = [m for m in llm.seen[-1]["messages"] if m.get("role") == "tool"]
    assert "waiting" in tool_messages[-1]["content"]
    assert "Nothing was changed" in tool_messages[-1]["content"]
    resources.clear()


@pytest.mark.asyncio
async def test_a_half_taken_claim_is_given_back_not_left_locked(guard_factory, tmp_path):
    """move_file wants two resources. If the second is held by someone else, the
    first was already taken - and a tool call that never runs must not leave a
    lock behind. Every other release path runs after the tool; this one is the
    only cover for a claim that was abandoned halfway."""
    from core import resources
    from tools.base import BaseTool, ToolParameter, ToolResult

    class MoveTool(BaseTool):
        name = "move_file"
        description = "move a file"
        parameters = [
            ToolParameter(name="source_path", type="string", description="from"),
            ToolParameter(name="destination_path", type="string", description="to")]

        async def run(self, **kwargs):
            return ToolResult(success=True, output={"success": True, "message": "Moved"})

    resources.clear()
    source = tmp_path / "from.txt"
    destination = tmp_path / "to.txt"
    # Somebody else has the destination. The source is free, and gets taken first.
    resources.acquire(resources.identity(resources.FILE, str(destination)),
                      resources.WRITE, "some-other-task")
    call = {"content": None, "tool_calls": [
        {"function": {"name": "move_file",
                      "arguments": {"source_path": str(source),
                                    "destination_path": str(destination)}}}]}
    orch, llm, _ = _orchestrator(guard_factory, [call, {"content": "I waited."}])
    orch.tool_registry.register(MoveTool())
    await orch.handle_user_input("move it")

    still_held = resources.snapshot()["held"]
    assert {hold["task"] for hold in still_held} <= {"some-other-task"}, (
        f"a half-taken claim was left locked: {still_held}")
    assert not any(hold["resource"].endswith("from.txt") for hold in still_held), (
        "the source end was taken and never given back")
    resources.clear()


@pytest.mark.asyncio
async def test_a_failing_tool_still_releases_its_resources(guard_factory, tmp_path):
    from core import resources
    from tools.base import BaseTool, ToolParameter, ToolResult

    resources.clear()

    class Exploding(BaseTool):
        name = "write_file"
        description = "write"
        parameters = [ToolParameter(name="path", type="string", description="path"),
                      ToolParameter(name="content", type="string", description="c")]

        async def run(self, **kwargs):
            raise RuntimeError("the disk caught fire")

    call = {"content": None, "tool_calls": [
        {"function": {"name": "write_file",
                      "arguments": {"path": str(tmp_path / "y"), "content": "x"}}}]}
    orch, llm, _ = _orchestrator(guard_factory, [call, {"content": "It failed."}])
    orch.tool_registry.register(Exploding())
    await orch.handle_user_input("save it")
    assert resources.snapshot()["held"] == [], "a crashed tool leaked its lock"
    resources.clear()


@pytest.mark.asyncio
async def test_a_refused_call_never_holds_anything(guard_factory, tmp_path):
    """A lock is not a permission, and must never be able to look like one."""
    from core import resources
    from core.safety_guard import ConfirmationDenied

    resources.clear()
    guard, prompts = guard_factory(confirm=False, confirm_classes=["modify", "critical"])
    call = {"content": None, "tool_calls": [
        {"function": {"name": "write_file",
                      "arguments": {"path": str(tmp_path / "z"), "content": "x"}}}]}
    orch, llm, _ = _orchestrator(guard_factory, [call, {"content": "Not allowed."}])
    orch.safety_guard = guard
    await orch.handle_user_input("save it")
    assert resources.snapshot()["held"] == []
    resources.clear()


@pytest.mark.asyncio
async def test_a_task_owns_the_resources_its_steps_touch(guard_factory):
    orch, llm, _ = _orchestrator(guard_factory, [{"content": "ok"}])
    seen = {}

    original = orch._claim_resources

    def watch(tool_name, arguments):
        seen["owner"] = orch._resource_owner()
        return original(tool_name, arguments)

    orch._claim_resources = watch
    await orch.handle_user_input("hello", owner="task-42")
    assert orch._resource_owner() == "conversation", "the owner leaked past the turn"
