"""
The Orchestrator is Leti's central nervous system. It:
  1. Runs the state machine: IDLE -> LISTENING -> THINKING -> EXECUTING -> SPEAKING -> IDLE
  2. Drives the LLM tool-calling loop: send messages+tools, execute any tool
     calls the model requests (via SafetyGuard authorization), feed results
     back, repeat until the model produces a final text answer.
  3. Coordinates memory (session buffer + long-term recall) and speech I/O.

This module is transport-agnostic: main.py decides whether input comes from
voice (wake word + STT) or text (CLI/typed), and orchestrator just consumes
"user said X" events and produces "Leti responds Y" events.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from core.config_loader import get_settings
from core.intent_signals import contains_explicit_denial, contains_request_approval
from core.llm_client import OllamaClient
from core import artifacts, diagnostics
from core.safety_guard import ConfirmationDenied, PermissionDenied, SafetyGuard
from core.tool_router import last_user_message, select_tools_for
from memory.session_memory import SessionMemory
from memory.vector_store import VectorMemory
from tools.base import ToolRegistry, ToolResult
from tools.personality import describe_personality
from tools.projects import get_active_project, project_context
from tools.user_profile import profile_summary

logger = logging.getLogger("leti.orchestrator")


class AgentState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    SPEAKING = "speaking"


SYSTEM_PROMPT = """You are Leti, a helpful, concise local AI assistant running on the user's own
computer. You can see the user's screen, control their mouse/keyboard, run shell commands, manage
files, browse and research the web, write and run code, analyse datasets, do engineering
calculations, track business data, keep persistent project workspaces, and schedule work to
happen automatically - all via the tools provided to you.

Combine these when a task needs it; they are one toolbox, not separate modes. "Analyse this
company's performance and tell me what to do" is business data plus research. "Find the optimal
operating point in this turbine dataset" is the project's files plus data analysis plus
engineering calculation, with the numerical work run as real code. Pick what the task actually
needs rather than announcing which capability you are using.

Guidelines:
- Use tools whenever a request requires current information, screen context, or system actions.
- Prefer the smallest number of tool calls that accomplish the task.
- Never claim to have done something you did not actually call a tool to do.
- If a tool call fails or is denied, tell the user plainly what happened and suggest an alternative.
- Keep spoken responses short and natural; save detailed output (like file contents) for when asked.
- Before emailing or messaging someone by name (not by a literal email address), ALWAYS call
  resolve_contact first with the name and the topic/context of the message. If it returns
  "ambiguous", stop and ask the user which specific person they meant (list the candidates'
  distinguishing details, e.g. tags/company) - never guess between two people who share a name.
  If it returns "not_found", tell the user and offer to add the contact. The same applies to
  meeting participants named by name in schedule_meeting or send_meeting_invite_email.
- When system_report returns issues, present each one plainly with its suggestion. If an
  issue has a related_tool, ask the user whether to apply that fix before calling it - even
  though the tool's own risk tier will also require confirmation, describe the fix in plain
  terms first rather than just invoking it silently.
- Proactively call remember_about_user when the user shares something durable worth carrying
  into future sessions (name, preferences, interests, goals, communication style) - don't wait
  to be asked, and don't announce it every time either; just do it naturally. Use set_user_name
  the first time they introduce themselves or ask to be called something. If asked what you know
  about them, use view_user_profile rather than guessing from memory.
- If the user asks you to change your tone (funnier, more sarcastic, more serious, blunter, more
  formal/casual, more or less detailed, etc.), call set_personality or apply_personality_preset
  rather than just trying to act differently in this one response - the change should persist.
- When the user says something like "notify me when X uploads/posts", that's a request to call
  add_social_watch, not a one-time check - it should keep working next session too. Any time
  check_social_watches reports new items, tell the user plainly which watch it was and what's new.
- When the user wants something opened on their computer, just do it - call launch_app
  (they'll be asked to confirm a program launch, which is the point; don't ask permission
  yourself first, and don't explain that you need permission). launch_app opens apps, files
  and web pages: "open YouTube" is launch_app with app_name 'youtube.com', which goes to
  their own browser, and `arguments` opens something IN a particular app - "open YouTube in
  Firefox" is app_name 'firefox' with arguments ['https://youtube.com'].
- For browsing: web_search finds pages, browser_read_page reads one so you can answer from
  what it actually says rather than a snippet, and launch_app puts a page in front of the
  user to look at themselves. When a search result doesn't clearly answer the question, read
  the page instead of guessing from the snippet.
- Work inside the user's projects. If a project is active its instructions and files are
  in your context above - follow those instructions, put new files in that project's
  folder, and read what's already there before adding to it. When the user refers to a
  project by name ("continue the MATLAB project"), call open_project first. Only create
  a project when the user asks for one or is clearly starting durable new work.
- When you write code, run it. The workflow is PLAN -> WRITE/MODIFY -> RUN -> TEST ->
  FIX -> VERIFY -> REPORT: use run_code to execute what you wrote, run_tests after
  changing a project, and inspect_project before editing a codebase you haven't seen.
  Report what the output actually said. Code that ran and printed the right answer is
  evidence; code that looks correct is not, and "this should work" is not a report.
  When something fails, read the real error in stderr rather than guessing at the cause.
- For datasets: inspect_dataset first, always - what's missing, duplicated or the wrong
  type decides which analysis is honest. Then clean_dataset with the specific operations
  the data needs, analyze_dataset for statistics, visualize_dataset to show it. Say what
  cleaning changed, since conclusions depend on it, and report correlations as association
  rather than cause.
- For engineering and physics, compute - never do the arithmetic yourself. Use
  engineering_calculate (it carries units, so unit errors surface as errors),
  convert_units, check_dimensions to test an equation before trusting it, and
  solve_symbolic to rearrange or differentiate. For anything heavier, write the Python
  or MATLAB and run it with run_code. Present engineering work as INPUTS -> ASSUMPTIONS
  -> EQUATIONS -> UNITS -> CALCULATIONS -> RESULTS -> CHECKS, stating assumptions
  explicitly, and when asked to check someone's work run check_dimensions first - an
  equation that fails it is wrong whatever the numbers look like.
- For business questions, read the records before answering: business_dashboard for
  where things stand and what changed, business_next_actions for who to contact,
  list_business_data for the records behind a number. Every figure is computed from
  recorded leads, income and expenses, so if there are no records say so rather than
  reporting zeroes as if they were results. When recording a lead for someone, link
  them with contact_id from resolve_contact or add_contact instead of retyping their
  details - the contact book is where people live.
- "Every Monday...", "every month...", "each morning..." is a request for
  create_scheduled_task, not something to do once now. Write the instruction so it
  stands alone weeks later with none of this conversation around it, and attach the
  project it belongs to. A scheduled task can use everything you can, so
  "every Friday analyse my business performance and write a report" is one task.
  If a message arrives labelled as a scheduled task, just do it and report the result.
- Use search_images whenever the user wants to SEE something (a picture/photo of X) rather than
  read about it - the images appear automatically once the tool runs, so just call it, you don't
  need to also describe the images in detail afterward. Use create_sketch only for genuinely
  diagram-shaped explanations (a process, a decision flow, a sequence) - not as a substitute for
  a normal explanation, and not for every technical answer.
"""


# --------------------------------------------------------------------------- #
# Tool-call normalisation
#
# Ollama hands back a tool call as {"function": {"name": ..., "arguments": {...}}}
# and that is what the execution path below consumes. Templates vary in how they
# represent the SAME call, though, and the two variants seen in practice are
# arguments arriving as a JSON string rather than an object, and a call landing in
# the message text because the template failed to lift it out. Neither is a
# different call - only a different spelling of one.
#
# These normalise the spelling and nothing else. They never repair a call, never
# guess a tool, and never invent an argument: anything that does not resolve
# cleanly is handed back as unusable so the caller can fail it through the error
# path it already has. There is no branching on which model is configured.
# --------------------------------------------------------------------------- #

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_JSON_FENCE = re.compile(r"\A```(?:json)?\s*(\{.*\})\s*```\Z", re.S)


def _normalize_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    """One call's arguments as a mapping, or None if they are not one.

    None means unusable, and is deliberately distinct from {} - a tool that takes
    no arguments is a legitimate call, a tool whose arguments could not be read is
    not. Callers must not treat the two the same.
    """
    if not raw:
        # What the previous `fn.get("arguments", {}) or {}` did: absent, null and
        # empty all mean a call with no arguments.
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _call_from_object(blob: str, find_tool: Callable[[str], Any],
                      require_arguments: bool) -> Optional[Dict[str, Any]]:
    """One JSON blob turned into a tool call, or None if it is not clearly one."""
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if require_arguments and "arguments" not in obj:
        return None
    name = obj.get("name")
    # An exact match against a registered tool, never a resemblance to one.
    if not isinstance(name, str) or find_tool(name) is None:
        return None
    arguments = _normalize_arguments(obj.get("arguments"))
    if arguments is None:
        return None
    return {"function": {"name": name, "arguments": arguments}}


def _recover_tool_calls_from_text(content: Any,
                                  find_tool: Callable[[str], Any]) -> List[Dict[str, Any]]:
    """Tool calls a template left in the message text. Almost always empty.

    This is the conservative half of the parser, and it is meant to stay that way:
    an assistant that says "I can use the weather tool for that" is talking, not
    calling, and must keep being treated as talking. Recovery therefore needs
    machine-readable evidence of an intended invocation, not a mention:

      - a <tool_call>...</tool_call> block, whose delimiters are the evidence, or
      - a message that is ENTIRELY one JSON object carrying both a name and an
        arguments key. A JSON object quoted inside a sentence is the model
        describing a call, not making one, so it does not qualify.

    In both cases the name must match a registered tool exactly and the arguments
    must resolve to a mapping. Nothing here scores, ranks or guesses.
    """
    if not isinstance(content, str) or "{" not in content:
        return []

    tagged = _TOOL_CALL_TAG.findall(content)
    if tagged:
        calls = [_call_from_object(blob, find_tool, require_arguments=False) for blob in tagged]
        return [call for call in calls if call is not None]

    stripped = content.strip()
    fenced = _JSON_FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        call = _call_from_object(stripped, find_tool, require_arguments=True)
        return [call] if call is not None else []
    return []


class Orchestrator:
    def __init__(
        self,
        llm_client: OllamaClient,
        tool_registry: ToolRegistry,
        safety_guard: SafetyGuard,
        session_memory: SessionMemory,
        vector_memory: VectorMemory,
        speak_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        visual_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.safety_guard = safety_guard
        self.session_memory = session_memory
        self.vector_memory = vector_memory
        self.speak_callback = speak_callback
        self.visual_callback = visual_callback

        self.state = AgentState.IDLE
        self._state_listeners: List[Callable[[AgentState], None]] = []
        self._preapproved_this_turn = False
        self._checked_watches_this_session = False
        # One turn at a time. The desktop window, a phone, and the voice loop are all
        # clients of the same orchestrator, and a turn mutates shared state (the memory
        # buffer, the agent state machine, the one-shot pre-approval flag). Two
        # concurrent turns interleave their tool calls and their history.
        self._turn_lock = asyncio.Lock()

    @property
    def settings(self) -> Dict[str, Any]:
        """Read live rather than cached at construction: /settings edits call
        reload_settings(), and a value captured in __init__ would keep serving the
        old one until restart - which is exactly what config_loader.reload_settings
        promises doesn't happen."""
        return get_settings()["ollama"]

    def on_state_change(self, callback: Callable[[AgentState], None]) -> None:
        self._state_listeners.append(callback)

    def _set_state(self, state: AgentState) -> None:
        self.state = state
        logger.debug(f"State -> {state.value}")
        for cb in self._state_listeners:
            cb(state)

    # ------------------------------------------------------------------ #
    # Main entry point: handle one user utterance/turn end-to-end
    # ------------------------------------------------------------------ #
    async def handle_user_input(self, user_text: str, session_id: str = "default", voice_mode: bool = False) -> str:
        async with self._turn_lock:
            return await self._handle_one_turn(user_text, session_id, voice_mode)

    async def _handle_one_turn(self, user_text: str, session_id: str, voice_mode: bool) -> str:
        self._set_state(AgentState.THINKING)
        self.session_memory.add_turn("user", user_text, session_id)

        # In voice mode, if what the user said already reads as clear approval/intent to
        # proceed (and doesn't also contain a denial, e.g. "no wait"), the interactive
        # confirmation prompt can be skipped for ONE risky tool call - there's no natural
        # way to type '-y' while talking, so the approval has to come from the request
        # itself. Text mode is unaffected: it still shows the confirmation prompt, which
        # now also accepts '-y' as a shorthand yes (see main.py).
        #
        # Scope matters here. A turn can trigger up to max_tool_iterations rounds of tool
        # calls, and the model chooses them - so treating one "sure, go ahead" as blanket
        # permission for the whole turn hands out approval for actions the user never
        # described. Instead this is a single-use budget: SafetyGuard reports when it's
        # spent (see _execute_tool_call), and everything after it prompts normally.
        # SafetyGuard additionally refuses to apply it to DESTRUCTIVE calls at all.
        self._preapproved_this_turn = (
            voice_mode
            and contains_request_approval(user_text)
            and not contains_explicit_denial(user_text)
        )

        messages = await self._build_messages(user_text)
        _turn_started = time.perf_counter()
        final_answer = await self._tool_calling_loop(messages)
        try:
            diagnostics.record_turn(time.perf_counter() - _turn_started,
                                    len(final_answer or ""))
        except Exception:
            pass

        self.session_memory.add_turn("assistant", final_answer, session_id)
        await self._maybe_persist_to_long_term(user_text, final_answer)

        if self.speak_callback:
            self._set_state(AgentState.SPEAKING)
            await self.speak_callback(final_answer)

        self._set_state(AgentState.IDLE)
        return final_answer

    # ------------------------------------------------------------------ #
    # Message construction: system prompt + recalled memory + recent turns
    # ------------------------------------------------------------------ #
    async def _build_messages(self, user_text: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        try:
            messages.append({"role": "system", "content": describe_personality()})
        except Exception as e:
            logger.warning(f"Failed to load personality settings (continuing with defaults): {e}")

        # The active project's instructions and file list, so "continue the MATLAB
        # project" resolves to something concrete rather than a name the model has
        # to guess the contents of.
        try:
            active = get_active_project()
            if active:
                # The request is passed so the file list can be about this turn
                # rather than the whole folder every time.
                messages.append({"role": "system",
                                 "content": project_context(active, request=user_text)})
        except Exception as e:
            logger.warning(f"Failed to load project context (continuing without it): {e}")

        try:
            profile_text = profile_summary()
            if profile_text:
                messages.append({"role": "system", "content": f"What you know about the user so far:\n{profile_text}"})
        except Exception as e:
            logger.warning(f"Failed to load user profile (continuing without it): {e}")

        if not self._checked_watches_this_session:
            messages.append({
                "role": "system",
                "content": (
                    "This is the start of a new session. If any social media watches are "
                    "configured, call check_social_watches now, before anything else, to catch "
                    "the user up on anything missed since last time - regardless of how long "
                    "it's been. If it reports nothing new (or no watches exist), just continue "
                    "with the user's actual request without mentioning the check."
                ),
            })
            self._checked_watches_this_session = True

        try:
            recalled = await self.vector_memory.search(user_text)
            if recalled:
                memory_context = "\n".join(f"- {m['text']}" for m in recalled)
                messages.append(
                    {
                        "role": "system",
                        "content": f"Relevant things you remember about the user:\n{memory_context}",
                    }
                )
        except Exception as e:
            logger.warning(f"Long-term memory recall failed (continuing without it): {e}")

        # The rolling buffer already ends with this turn's user message - handle_user_input
        # records it before calling this, so that a failed LLM call doesn't lose what the
        # user said. Appending user_text again here sent every utterance to the model twice.
        messages.extend(self.session_memory.get_recent_messages())
        return messages

    # ------------------------------------------------------------------ #
    # Tool-calling loop
    # ------------------------------------------------------------------ #
    async def _tool_calling_loop(self, messages: List[Dict[str, Any]]) -> str:
        max_iterations = self.settings.get("max_tool_iterations", 8)
        # Which tools this request is shown - see core/tool_router.py. Chosen ONCE
        # per turn, not per iteration: the loop's whole point is that the prompt
        # grows by a tool result each time round, and a tool list that changed
        # underneath it would throw away the model server's cached prefix on every
        # pass. Routing never raises and its worst case is the full registry, which
        # is exactly what this line used to be.
        _routing_started = time.perf_counter()
        routing = select_tools_for(last_user_message(messages), self.tool_registry)
        tool_schemas = self.tool_registry.schemas_for(routing.tool_names)
        # Timings for the diagnostics panel, taken while doing the real work rather
        # than by measuring anything extra. Never allowed to affect the turn.
        try:
            diagnostics.record_routing(routing.count, len(self.tool_registry.names()),
                                       time.perf_counter() - _routing_started,
                                       routing.tool_names)
        except Exception:
            pass
        logger.info(
            f"Tools for this turn: {routing.count}/{len(self.tool_registry.names())}"
            f"{' (full fallback)' if routing.full_fallback else ''} - {routing.reason}"
        )

        for iteration in range(max_iterations):
            response = await self.llm_client.chat(messages, tools=tool_schemas)
            message = response.get("message", {})
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # A call the template failed to lift out of the text is still a
                # call; returning it verbatim would report the request as answered
                # without ever running anything. Empty for ordinary prose.
                tool_calls = _recover_tool_calls_from_text(
                    message.get("content"), self.tool_registry.get
                )
                if tool_calls:
                    logger.info(
                        f"Recovered {len(tool_calls)} tool call(s) from message text: "
                        f"{[c['function']['name'] for c in tool_calls]}"
                    )
            if not tool_calls:
                return message.get("content", "").strip() or "I don't have a response for that."

            messages.append(message)
            self._set_state(AgentState.EXECUTING)

            for call in tool_calls:
                result = await self._execute_tool_call(call)
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result.to_dict()),
                    }
                )
                # Visual content (image results, diagrams) is pushed directly to whatever
                # surface can show it (the GUI), independent of what the model ends up
                # saying in text - a tool result carrying a "visual" key is shown
                # immediately rather than waiting on/depending on the model to describe it.
                # A tool that supplies its own "visual" always wins; core/artifacts.py
                # only reads the SHAPE of results that don't have one (a table of
                # uniform rows, a set of sources, one long document) so a result
                # worth looking at is shown rather than only described. No model
                # call, no classifier, and nothing for a tool to opt into.
                if result.success and self.visual_callback and isinstance(result.output, dict):
                    visual = result.output.get("visual") or artifacts.derive(result.output)
                    if visual:
                        try:
                            await self.visual_callback(visual)
                        except Exception:
                            logger.exception("visual_callback failed")

            self._set_state(AgentState.THINKING)

        logger.warning("Max tool iterations reached without a final answer.")
        return "I made several attempts but couldn't complete that within my step limit. Want me to keep going?"

    async def _execute_tool_call(self, call: Dict[str, Any]) -> ToolResult:
        # Read defensively: a malformed call is a failed tool result, never an
        # exception that ends the turn.
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict):
            fn = {}
        raw_name = fn.get("name")
        tool_name = raw_name if isinstance(raw_name, str) else ""
        arguments = _normalize_arguments(fn.get("arguments"))

        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: '{tool_name}'")

        if arguments is None:
            # Unreadable arguments. Running the tool anyway would mean inventing
            # what the user asked for, so the model is told what was wrong instead.
            return ToolResult(
                success=False,
                error=(
                    f"Malformed arguments for '{tool_name}': expected a JSON object, got "
                    f"{type(fn.get('arguments')).__name__}. Call it again with an object."
                ),
            )

        # A few tools cover acts of genuinely different weight depending on their
        # arguments (launch_app opening a web page vs. starting a program). The tool
        # says which case this call is; permissions.yaml still says what each case costs.
        try:
            case = tool.action_case(arguments)
        except Exception:
            logger.exception(f"'{tool_name}'.action_case failed; using its configured action class")
            case = None

        try:
            auth = await self.safety_guard.authorize(
                tool_name, arguments, preapproved=self._preapproved_this_turn, case=case
            )
        except PermissionDenied as e:
            return ToolResult(success=False, error=f"Blocked: {e}")
        except ConfirmationDenied as e:
            return ToolResult(success=False, error=f"Not confirmed: {e}")

        # A spoken pre-approval covers ONE action, not the whole turn. Spending it
        # here means any further risky call this turn prompts normally - the user
        # said yes to something specific, not to everything that follows from it.
        if auth.used_preapproval:
            self._preapproved_this_turn = False

        if not auth.execute:
            # dry_run mode: report the intended action rather than performing it.
            return ToolResult(
                success=True,
                output={
                    "dry_run": True,
                    "would_have_done": auth.description,
                    "note": (
                        "DRY RUN - nothing was actually done. safety.dry_run is enabled in "
                        "config/settings.yaml. Tell the user plainly what this tool would "
                        "have done; do not claim it happened."
                    ),
                },
            )

        _tool_started = time.perf_counter()
        try:
            result = await tool.run(**arguments)
            await self.safety_guard.audit_result(tool_name, arguments, result.success, result.error or "", case=case)
            try:
                diagnostics.record_tool(tool_name, time.perf_counter() - _tool_started,
                                        result.success)
            except Exception:
                pass
            return result
        except Exception as e:
            logger.exception(f"Tool '{tool_name}' raised an exception")
            await self.safety_guard.audit_result(tool_name, arguments, False, str(e), case=case)
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------ #
    # Long-term memory writeback
    # ------------------------------------------------------------------ #
    async def _maybe_persist_to_long_term(self, user_text: str, answer: str) -> None:
        """Naive heuristic: persist statements that look like preferences/facts.
        A more advanced version could ask the LLM to decide what's worth remembering."""
        lowered = user_text.lower()
        memorable_triggers = ("remember that", "i prefer", "i like", "i always", "i never", "my ", "call me")
        if any(trigger in lowered for trigger in memorable_triggers):
            try:
                await self.vector_memory.add_memory(user_text, metadata={"source": "user_statement"})
            except Exception as e:
                logger.warning(f"Failed to persist long-term memory: {e}")
