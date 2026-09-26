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
from core import intent as intent_reader
from core import context_engine, entities, modes, performance, task_control
from core.intent_signals import contains_explicit_denial, contains_request_approval
from core.llm_client import OllamaClient
from core import (artifacts, connections, diagnostics, math_render, resources,
                  speech, transcript, verification)
from core.safety_guard import ConfirmationDenied, PermissionDenied, SafetyGuard
from core.tool_router import Routing, last_user_message, select_tools_for
from memory.session_memory import SessionMemory
from memory.vector_store import VectorMemory
from tools.base import ToolRegistry, ToolResult

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


def _called_tool_name(call: Any) -> str:
    """The tool name off a call, for presentation only.

    Deliberately not the parser: _execute_tool_call still does the real
    normalisation and is the only thing that decides what runs. This reads the
    same field defensively so a malformed call means "no name" - which makes the
    result an offer rather than a window, the safe way round.
    """
    function = call.get("function") if isinstance(call, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    return name if isinstance(name, str) else ""


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


# A numbered or bulleted line, with the marker stripped. Deliberately only these
# two shapes: a paragraph that happens to contain "1990" is not a list, and a
# guess about what counts as an item is worse than having no referents at all.
_LIST_ITEM = re.compile(r"^\s*(?:\d{1,2}[.)]\s+|[-*\u2022]\s+)(.{3,})$")


def _enumerated_items(answer: str) -> List[str]:
    """The ordered items of a list in an answer, or nothing."""
    if not isinstance(answer, str) or len(answer) > 20_000:
        return []
    items = []
    for line in answer.splitlines():
        match = _LIST_ITEM.match(line)
        if match:
            # The name, not the whole entry: "1. Dell XPS 15 - 1,049 EUR, 32GB"
            # is remembered as the laptop, which is what gets referred to.
            items.append(re.split(r"\s+[\u2013\u2014-]\s+|:\s", match.group(1).strip())[0])
    return items if len(items) >= 2 else []


# The reading and the budget a turn falls back on. Class-level defaults so an
# Orchestrator built without __init__ still has them: nothing here is mutated in
# place, only ever replaced at the start of a turn.
def _task_runner():
    """Whatever is running tasks right now, or nothing.

    Read through tools/autonomous.py, which is where the one runner has always
    been registered - the orchestrator does not hold a second reference to it.
    """
    try:
        from tools.autonomous import get_runner

        return get_runner()
    except Exception:
        return None


_DEFAULT_INTENT = intent_reader.Intent()
_DEFAULT_MODE = performance.Mode()


def _is_stop(user_text: str) -> bool:
    """Is this message nothing but "stop"?

    Read with the lifecycle reader that already decides what a stop is, so there
    is one answer to that question rather than two that can disagree. Only a bare
    stop jumps the queue: "stop the research task" names work, and work is the
    task controls' business on the ordinary path.
    """
    try:
        command = intent_reader.lifecycle_command(user_text)
    except Exception:
        return False
    return bool(command) and command.get("action") == intent_reader.STOP \
        and not command.get("hint")


class Orchestrator:
    _owning_task = None              # set per turn; see handle_user_input
    interrupt_callback = None        # the speech engine's own interrupt, if any
    _spoke_while_streaming = False   # set per turn; see _ask
    _intent = _DEFAULT_INTENT
    _mode = _DEFAULT_MODE
    _context = None                  # the last context package, for diagnostics

    def __init__(
        self,
        llm_client: OllamaClient,
        tool_registry: ToolRegistry,
        safety_guard: SafetyGuard,
        session_memory: SessionMemory,
        vector_memory: VectorMemory,
        speak_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        visual_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        interrupt_callback: Optional[Callable[[], None]] = None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.safety_guard = safety_guard
        self.session_memory = session_memory
        self.vector_memory = vector_memory
        self.speak_callback = speak_callback
        self.visual_callback = visual_callback
        # Set by whichever surface owns the speech engine (gui/api.py, main.py).
        # None means there is nothing playing to cut, which is the text-only case.
        self.interrupt_callback = interrupt_callback

        self.state = AgentState.IDLE
        self._state_listeners: List[Callable[[AgentState], None]] = []
        self._preapproved_this_turn = False
        self._checked_watches_this_session = False
        # This turn's reading of the request and its budget, replaced at the start
        # of every turn. The class defaults above are what a turn costs if either
        # ever fails, or when something builds an orchestrator without __init__.
        self._intent = _DEFAULT_INTENT
        self._mode = _DEFAULT_MODE
        # One turn at a time. The desktop window, a phone, and the voice loop are all
        # clients of the same orchestrator, and a turn mutates shared state (the memory
        # buffer, the agent state machine, the one-shot pre-approval flag). Two
        # concurrent turns interleave their tool calls and their history.
        self._turn_lock = asyncio.Lock()
        # "Which Acme did they mean" is answered partly by what was said two turns
        # ago. core/entities.py reads the conversation through this reference - the
        # buffer stays the one in memory/session_memory.py, and nothing is copied.
        try:
            entities.use_conversation_source(self.session_memory.get_recent_messages)
        except Exception as e:
            logger.debug(f"Entity resolution will run without conversation context: {e}")

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
    async def handle_user_input(self, user_text: str, session_id: str = "default",
                                voice_mode: bool = False, preapproved: bool = False,
                                owner: str = "") -> str:
        """One turn. `preapproved` is the caller saying the user has already agreed
        to what this turn will do - the task runner passes it for a step the user
        approved in the interface. It feeds the SAME pre-approval SafetyGuard
        already honours for voice, with the same limit: an irreversible action is
        never covered by it and still stops to ask."""
        # "Stop" cannot queue behind the thing it is stopping. Every other
        # message waits its turn, and should - two turns interleaving their tool
        # calls is the reason the lock exists. A stop is the one message whose
        # whole meaning is that the turn in flight should end, so it is read
        # before the lock and acted on immediately.
        #
        # Only while a turn is actually running. With nothing in flight there is
        # nothing to jump, and the ordinary path answers it properly - including
        # the task controls, which stay the authority on stopping work.
        if self._turn_lock.locked() and _is_stop(user_text):
            return self._stop_now(session_id)

        async with self._turn_lock:
            # Who owns whatever this turn touches - a task id when the task
            # runner is driving, the conversation when a person is. Set inside
            # the lock, which is what makes it safe: one turn runs at a time, so
            # there is never a moment where two owners are in play.
            self._owning_task = str(owner) or None
            try:
                return await self._handle_one_turn(user_text, session_id, voice_mode,
                                                   preapproved)
            finally:
                self._owning_task = None

    async def _handle_one_turn(self, user_text: str, session_id: str, voice_mode: bool,
                               preapproved: bool = False) -> str:
        self._set_state(AgentState.THINKING)
        # A new request, so whatever the last answer was told to stop saying is
        # spent. Here rather than in transcript.begin() because a turn answered
        # without the model - showing the text, switching mode - does not call
        # begin() and still speaks a confirmation. The stop shortcut above returns
        # before this, so a stop aimed at the turn in flight is not cleared by the
        # turn it is stopping.
        transcript.clear_stop()
        self.session_memory.add_turn("user", user_text, session_id)

        # What this request IS, read deterministically before anything is sent -
        # no model call, tens of microseconds (core/intent.py) - and what that
        # means this turn may spend (core/performance.py). Both are hints: they
        # change what the model is shown, never what it is allowed to do.
        self._intent = intent_reader.read(
            user_text, self.session_memory.get_recent_messages())
        self._mode = performance.for_turn(self._intent)
        try:
            diagnostics.record_intent(self._intent.as_dict())
        except Exception:
            pass
        if self._mode.under_pressure:
            logger.info(f"Adaptive mode: {self._mode.reason}")

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
        self._preapproved_this_turn = bool(preapproved) or (
            voice_mode
            and contains_request_approval(user_text)
            and not contains_explicit_denial(user_text)
        )

        # An explicit mode command is a command: handled here, deterministically,
        # without a model round-trip. This is also what makes the rule that nothing
        # ELSE switches modes enforceable - a regex cannot be talked into Business
        # Mode by a sentence full of invoices, and "check my customer emails" gets
        # the ordinary Default Mode turn it asked for.
        if intent_reader.asks_which_mode(user_text):
            answer = self._describe_mode()
            self.session_memory.add_turn("assistant", answer, session_id)
            if self.speak_callback:
                self._set_state(AgentState.SPEAKING)
                await self.speak_callback(answer)
            self._set_state(AgentState.IDLE)
            return answer

        wanted_mode = intent_reader.mode_command(user_text)
        if wanted_mode is not None:
            answer = self._switch_mode(wanted_mode)
            self.session_memory.add_turn("assistant", answer, session_id)
            if self.speak_callback:
                self._set_state(AgentState.SPEAKING)
                await self.speak_callback(answer)
            self._set_state(AgentState.IDLE)
            return answer

        # A lifecycle command - "stop", "pause", "resume", "what are you waiting
        # for" - is about work already running, and is answered here for the
        # same reason a mode command is: a task that stops only if the model
        # agrees that "stop" means stop is a task that does not reliably stop.
        #
        # It falls through when nothing matches: a command naming a task that
        # does not exist gets an ordinary turn rather than an error, because the
        # user may have meant something else entirely by it.
        control = intent_reader.lifecycle_command(user_text)
        if control is not None:
            # "Stop" means stop talking too, not only stop working. Setting the
            # flag rather than reaching for the engine keeps this on one path:
            # the speak callback reads it between utterances, and the surface
            # that owns the engine interrupts the one already playing.
            was_speaking = False
            if control.get("action") == intent_reader.STOP:
                was_speaking = transcript.is_speaking()
                transcript.stop_speaking()
            handled = task_control.apply(control, runner=_task_runner())
            if handled is not None:
                answer = handled["answer"]
                # "Stop" while Leti is talking and nothing is running is not
                # "there is nothing to stop" - the talking stopped. Saying
                # otherwise would read as the stop having failed.
                if (was_speaking and handled.get("ok") is False
                        and not handled.get("needs_choice")):
                    answer = ("Stopped. The answer is still here if you want to "
                              "read it.")
                self.session_memory.add_turn("assistant", answer, session_id)
                if self.speak_callback:
                    self._set_state(AgentState.SPEAKING)
                    await self.speak_callback(answer)
                self._set_state(AgentState.IDLE)
                return answer

        # Asking to see the text, or to stop seeing it, is answered from what
        # Leti already said. No model call, no regeneration: the words shown are
        # the same string that was spoken, because showing is a panel opening
        # over a buffer that never went anywhere.
        #
        # Handled here, with the other deterministic commands, for the reason
        # they are: a request that only works when the model agrees it was a
        # request is a request that sometimes gets an essay instead.
        showing = intent_reader.transcript_command(user_text)
        if showing is not None:
            handled = await self._show_or_hide(showing)
            if handled is not None:
                answer = handled["answer"]
                self.session_memory.add_turn("assistant", answer, session_id)
                if self.speak_callback:
                    self._set_state(AgentState.SPEAKING)
                    await self.speak_callback(answer)
                self._set_state(AgentState.IDLE)
                return answer

        # "Run full Leti diagnostics" is a command too, and answered the same way:
        # deterministically, here, without a model round trip and without costing
        # Default Mode a tool schema. Every check reads; none of them changes,
        # sends or deletes anything (see core/diagnostics.py's full_check).
        diagnostics_request = intent_reader.asks_for_diagnostics(user_text)
        if diagnostics_request is not None:
            answer = await self._run_diagnostics(**diagnostics_request)
            self.session_memory.add_turn("assistant", answer, session_id)
            if self.speak_callback:
                self._set_state(AgentState.SPEAKING)
                await self.speak_callback(answer)
            self._set_state(AgentState.IDLE)
            return answer

        # From here on this is an ordinary turn, so it gets its own response to
        # collect against. Started BEFORE the tools run: a chart drawn halfway
        # through belongs to this answer, and starting the response afterwards
        # would throw it away just as "show that graph again" needed it.
        transcript.begin()
        self._spoke_while_streaming = False

        messages = await self._build_messages(user_text)
        _turn_started = time.perf_counter()
        final_answer = await self._tool_calling_loop(messages)
        try:
            diagnostics.record_turn(time.perf_counter() - _turn_started,
                                    len(final_answer or ""))
        except Exception:
            pass

        # The answer becomes the current response: held, hidden, and ready to be
        # shown the moment somebody asks. This replaces nothing - the session
        # memory below still keeps the conversation exactly as it did.
        if transcript.should_stop_speaking():
            # Stopped part-way. What arrived is already in the buffer (see _ask),
            # and the turn ends here rather than recording an empty answer over
            # it, persisting it or looking for a visual in it.
            self._set_state(AgentState.IDLE)
            return final_answer

        transcript.said(final_answer)

        # If the request was for the mathematics rather than the answer - "derive
        # the bending equation and show me the formulas" - and the answer has
        # notation in it, the notation is rendered. Both halves are required:
        # asking alone renders nothing when there is nothing to render, and an
        # equation alone opens no panel, because "what is two plus two" is
        # answered in one spoken word and a window for it is decoration.
        if artifacts.asked_to_see_mathematics(user_text):
            equations = math_render.visual(final_answer)
            if equations is not None:
                transcript.add_visual(equations)
                await self._push_visual(equations)

        self.session_memory.add_turn("assistant", final_answer, session_id)
        # If this answer was a list, remember its order, so "compare the first
        # three" next turn points at the same three. Purely a reading of the text
        # Leti just produced; nothing is asked and nothing is stored that was not
        # already in the buffer.
        try:
            self.session_memory.note_referents(_enumerated_items(final_answer))
        except Exception:
            logger.debug("Couldn't note what this answer listed.")
        await self._maybe_persist_to_long_term(user_text, final_answer)

        # Said already, sentence by sentence, while the model was still writing
        # it - so saying it again here would be the answer twice. The streamed
        # path speaks through the same one callback; what changes is only WHEN.
        if self.speak_callback and not self._spoke_while_streaming:
            self._set_state(AgentState.SPEAKING)
            await self.speak_callback(final_answer)

        self._set_state(AgentState.IDLE)
        return final_answer

    async def _run_diagnostics(self, reach_out: bool = False) -> str:
        """Check every subsystem and say what was and was not actually tested.

        Run off the event loop: the checks read files and, when asked, contact the
        model server, and blocking the loop would freeze the interface and the
        voice pipeline while it happened.
        """
        self._set_state(AgentState.EXECUTING)
        try:
            report = await asyncio.get_running_loop().run_in_executor(
                None, lambda: diagnostics.full_check(self.tool_registry, reach_out))
            return diagnostics.full_check_text(report)
        except Exception as e:
            logger.exception("The diagnostics run itself failed")
            return (f"The diagnostics run failed before it could report: {e}. That is "
                    "not a verdict on any subsystem - nothing was tested.")

    def _describe_mode(self) -> str:
        active = modes.mode()
        if active.name == modes.DEFAULT:
            return ("Default Mode - the general assistant. Say \"enter coding mode\" or "
                    "\"enter business mode\" to switch to a specialised workspace.")
        return (f"{active.label}. {active.description} Say \"exit\" to go back to the "
                "general assistant.")

    def _switch_mode(self, wanted: str) -> str:
        """Do what the mode command asked, and say so in one line.

        Changes a string, a tool list and a system note. Nothing is restarted, the
        model is not reloaded, memory and projects carry across, and running tasks
        keep running - see core/modes.py.
        """
        result = modes.leave() if wanted == modes.DEFAULT else modes.enter(wanted)
        if not result.get("ok"):
            return result.get("error", "I couldn't switch to that mode.")
        if not result.get("changed"):
            return result.get("note", f"Already in {modes.mode().label}.")
        try:
            from core import business

            if wanted == modes.BUSINESS:
                business.open_workspace()
        except Exception as e:
            logger.debug(f"Workspace setup on mode switch: {e}")
        return result.get("note", f"{modes.mode().label} is on.")

    # ------------------------------------------------------------------ #
    # Message construction: system prompt + recalled memory + recent turns
    # ------------------------------------------------------------------ #
    async def _build_messages(self, user_text: str) -> List[Dict[str, Any]]:
        """The context for this turn, chosen rather than accumulated.

        core/context_engine.py decides which of Leti's context sources this
        particular request could use, reads only those, and fits what comes back
        in a budget. What it leaves out it says it left out - the report goes to
        the diagnostics panel, so "why did it not know about my project" has an
        answer that is not a guess.

        The fallback is deliberately the thing that always worked: the system
        prompt and the conversation buffer. A context engine that can fail a turn
        would be worse than no context engine.
        """
        request = context_engine.Request(
            text=user_text,
            intent=self._intent,
            mode=modes.current(),
            allow_optional=self._mode.recall_memory,
            first_turn_of_session=not self._checked_watches_this_session,
        )
        self._checked_watches_this_session = True
        try:
            package = await context_engine.build(
                request,
                system_prompt=SYSTEM_PROMPT,
                history=self.session_memory.get_recent_messages(),
                referents=self.session_memory.recent_referents(),
                recall=(self.vector_memory.search if self.vector_memory else None),
            )
        except Exception as e:
            logger.warning(f"Context assembly failed; falling back to the basics: {e}")
            return ([{"role": "system", "content": SYSTEM_PROMPT}]
                    + self.session_memory.get_recent_messages())

        self._context = package
        try:
            diagnostics.record_context(package.report())
        except Exception:
            pass
        logger.debug(f"Context: {', '.join(package.included) or 'nothing extra'}"
                     f" ({package.chars} chars)")
        return package.messages

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
        # The Intent Layer's one veto. A remark is not a request, and a hedged
        # suggestion is a request for an opinion - so neither is shown a tool at
        # all. Not "the model should know better": there is nothing to call, so
        # "that's interesting" cannot become a web search however the sampler
        # rolls. See core/intent.py's kinds.
        #
        # This narrows what is OFFERED and nothing else. Every permission, every
        # guard and every tool stays exactly as it was; a request that turns out
        # to need something says so and the next turn has the full list back.
        if not self._intent.wants_action:
            routing = Routing(tool_names=[], no_tools=True,
                              reason=f"{self._intent.kind}: nothing was asked for")
        else:
            routing = select_tools_for(last_user_message(messages), self.tool_registry,
                                       budget=self._mode.tool_budget,
                                       allowed=modes.visible_tools(self.tool_registry))
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
            message = await self._ask(messages, tool_schemas)
            if message is None:
                return ""              # stopped; the turn unwinds above
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
                # worth looking at can be shown rather than only described. No model
                # call, no classifier, and nothing for a tool to opt into.
                #
                # Recognising one is not the same as putting it on screen. should_open
                # decides that, and usually says no: a panel opens by itself only for
                # something a sentence cannot carry, and only when it was asked for or
                # deliberately made. Everything else is marked as an offer and appears
                # as one line in the activity log, openable if it turns out to be
                # wanted. Answering in words is the normal case.
                if (result.success and self.visual_callback and self._mode.derive_visuals
                        and isinstance(result.output, dict)):
                    visual = result.output.get("visual") or artifacts.derive(result.output)
                    if visual:
                        visual = dict(visual)
                        visual["offer"] = not artifacts.should_open(
                            visual, _called_tool_name(call), last_user_message(messages))
                        try:
                            await self.visual_callback(visual)
                            # Kept so "show that graph again" shows THAT graph
                            # rather than recomputing one. Remembering is not
                            # showing: an offered visual is remembered too, and
                            # is what a later "show me the chart" opens.
                            transcript.add_visual(visual)
                        except Exception:
                            logger.exception("visual_callback failed")

            self._set_state(AgentState.THINKING)

        logger.warning("Max tool iterations reached without a final answer.")
        return "I made several attempts but couldn't complete that within my step limit. Want me to keep going?"

    def _stop_now(self, session_id: str = "default") -> str:
        """End the answer in flight, without waiting for it.

        Three things, none of which can block: the flag that the speech loop and
        the model stream both read between pieces, the engine's own interrupt for
        the utterance already playing, and a reply. The turn itself notices at
        its next check and unwinds on its own - cooperative, so there is no
        thread to kill and no socket left half-read.

        It stops the ANSWER. Work that is running is stopped by the task
        controls, which a stop with nothing in flight still goes through.
        """
        transcript.stop_speaking()
        if self.interrupt_callback is not None:
            try:
                self.interrupt_callback()
            except Exception:
                logger.exception("Couldn't interrupt the voice.")
        answer = "Stopped."
        try:
            self.session_memory.add_turn("assistant", answer, session_id)
        except Exception:
            logger.debug("Couldn't record the stop in the session.")
        self._set_state(AgentState.IDLE)
        return answer

    async def _show_or_hide(self, action: str) -> Optional[Dict[str, Any]]:
        """Answer a show/hide request from what is already held, or None.

        None means "this was not something I can show", and the caller falls
        through to an ordinary turn rather than insisting - "show me the graph"
        when no graph was made is a request to MAKE one, and refusing it would
        be worse than the panel it was trying to avoid.

        Nothing here generates, regenerates or asks anything. Every branch reads
        core/transcript.py and pushes what it finds through the visual_callback
        that already exists.
        """
        if action == intent_reader.HIDE_TEXT:
            transcript.hide()
            await self._push_visual({"type": "transcript", "visible": False})
            return {"answer": "Hidden."}

        if action == intent_reader.SHOW_TEXT:
            shown = transcript.reveal()
            if not shown["shown"]:
                return None
            await self._push_visual({"type": "transcript", "visible": True,
                               "text": shown["text"]})
            return {"answer": shown["answer"]}

        if action == intent_reader.SHOW_MATH:
            # The equation that was already in the answer, rendered. If there
            # was none, this was not a request to show one - it was a request
            # to work one out, and that is an ordinary turn.
            visual = math_render.visual(transcript.current().text)
            if visual is None:
                return None
            await self._push_visual(visual)
            transcript.add_visual(visual)
            return {"answer": "There it is."}

        if action == intent_reader.SHOW_BOTH:
            # Both were asked for, so both are shown, and the answer names only
            # what actually appeared. Nothing is regenerated for either half.
            parts = []
            shown = transcript.reveal()
            if shown["shown"]:
                await self._push_visual({"type": "transcript", "visible": True,
                                         "text": shown["text"]})
                parts.append("the text")
            for half in (intent_reader.SHOW_MATH, intent_reader.SHOW_LAST_VISUAL):
                answered = await self._show_or_hide(half)
                if answered is not None and "Which one" not in answered["answer"]:
                    parts.append("math" if half == intent_reader.SHOW_MATH else "the visual")
                    break
            if not parts:
                return None
            return {"answer": "Here " + ("they are." if len(parts) > 1 else "it is.")}

        if action == intent_reader.SHOW_LAST_VISUAL:
            remembered = [v for v in transcript.visuals()
                          if v.get("type") != "transcript"]
            if not remembered:
                return None
            if len({v.get("type") for v in remembered}) > 1:
                # Several different things were made and "show me the graph"
                # does not say which. Asking beats guessing, which is what the
                # existing entity rules do everywhere else.
                kinds = sorted({str(v.get("type")) for v in remembered})
                return {"answer": "I have " + " and a ".join(kinds) +
                                  " from that. Which one do you want?"}
            visual = dict(remembered[-1])
            visual["offer"] = False          # asked for by name: open it
            await self._push_visual(visual)
            return {"answer": "Here it is again."}

        return None

    async def _push_visual(self, visual: Dict[str, Any]) -> None:
        """Send a payload to whatever surface can render it, if there is one.

        Awaited rather than scheduled: this is a deterministic command with no
        model call to overlap with, and a panel that opens some time after the
        turn ends is a panel that races the next thing the user says. A page
        that is not connected is not an error, and a surface that throws must
        not take the turn down with it.
        """
        if not self.visual_callback:
            return
        try:
            await self.visual_callback(visual)
        except Exception:
            logger.exception("Couldn't show a visual.")

    async def _ask(self, messages: List[Dict[str, Any]],
                   tool_schemas: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """One model call. Streamed where that buys something, whole where it does not.

        Streaming is what lets Leti start speaking before the model has finished
        writing - the fragments feed core/speech.py's buffer, which hands back
        whole sentences and nothing smaller. It is the same ONE call either way:
        this replaces the whole-response request, it does not add to it.

        The whole-response path stays for a client that has no stream_response -
        every test double, and any caller that wants a message in one piece.

        None means the user said stop. The caller returns and the turn unwinds.
        """
        if not hasattr(self.llm_client, "stream_response") or self.speak_callback is None:
            response = await self.llm_client.chat(messages, tools=tool_schemas)
            return response.get("message", {})

        buffer = speech.Stream()
        said_anything = False
        stopped = False
        cut_short = ""
        message: Dict[str, Any] = {}

        async for event in self.llm_client.stream_response(
                messages, tools=tool_schemas,
                should_stop=transcript.should_stop_speaking):
            if event.get("done"):
                stopped = bool(event.get("stopped"))
                message = event.get("message") or {}
                if event.get("error") and not stopped:
                    cut_short = str(event["error"])
                    logger.warning(f"Streamed answer ended early: {cut_short}")
                break
            fragment = event.get("text")
            if not fragment:
                continue
            # A tool call in flight is not an answer. Whatever prose came with it
            # is a preamble the loop is about to throw away, so it is not spoken.
            for utterance in buffer.feed(fragment):
                if transcript.should_stop_speaking():
                    stopped = True
                    break
                await self._say(utterance)
                said_anything = True
            if stopped:
                break

        if stopped:
            # Silenced, not discarded. What the model managed to write is kept, so
            # "stop" then "show me the answer" shows it - including the part that
            # was never said out loud.
            partial = (message.get("content") or buffer.text).strip()
            if partial:
                transcript.said(partial)
            return None
        if not message.get("tool_calls"):
            for utterance in buffer.flush():
                if transcript.should_stop_speaking():
                    return None
                await self._say(utterance)
                said_anything = True

        if cut_short:
            # Half an answer presented as a whole one is the failure mode to
            # avoid: the user hears something that stops mid-thought and has no
            # way to know the model was interrupted rather than finished. So it
            # is said, and it is part of the answer that gets kept.
            #
            # Nothing is retried here. core/llm_client.py's fallback model is the
            # one retry policy and it covers the REQUEST; a connection that died
            # halfway through an answer is not a request to make again, and a
            # second call on its own initiative is a second call.
            note = ("I could not finish that - the connection to the model "
                    "stopped partway through.")
            message["content"] = ((message.get("content") or "").rstrip()
                                  + ("\n\n" if message.get("content") else "") + note)
            if not message.get("tool_calls"):
                await self._say(note)
                said_anything = True

        # What was said is not said again at the end of the turn.
        self._spoke_while_streaming = said_anything
        return message

    async def _say(self, utterance: str) -> None:
        if self.speak_callback is None:
            return
        self._set_state(AgentState.SPEAKING)
        await self.speak_callback(utterance)

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

        # What this call is actually about to touch - see core/resources.py. The
        # tool boundary is the one instrumentation point: the arguments already
        # say what the tool will use, so nothing has to be traced, watched or
        # scanned. Most tools touch nothing worth tracking and this is empty.
        #
        # The claim is made AFTER authorisation, deliberately. A lock is not a
        # permission and must never be able to look like one: a call SafetyGuard
        # refuses never gets as far as holding anything.
        claimed = self._claim_resources(tool_name, arguments)
        if claimed.get("blocked"):
            return ToolResult(success=False, error=claimed["explain"],
                              output={"waiting_on": claimed["held_by"],
                                      "resource": claimed["resource"],
                                      "note": ("Another piece of work is using this. "
                                               "Nothing was changed. Wait for it to "
                                               "finish, or stop the other one.")})

        _tool_started = time.perf_counter()
        try:
            result = await tool.run(**arguments)
            await self.safety_guard.audit_result(tool_name, arguments, result.success, result.error or "", case=case)
            try:
                diagnostics.record_tool(tool_name, time.perf_counter() - _tool_started,
                                        result.success)
                connections.record_tool_outcome(tool_name, result.success,
                                                result.error or "")
            except Exception:
                pass
            if not result.success:
                result = self._explain_failure(tool_name, result)
            return self._verified(tool_name, arguments, result)
        except Exception as e:
            logger.exception(f"Tool '{tool_name}' raised an exception")
            await self.safety_guard.audit_result(tool_name, arguments, False, str(e), case=case)
            try:
                connections.record_tool_outcome(tool_name, False, str(e))
            except Exception:
                pass
            return self._explain_failure(tool_name, ToolResult(success=False, error=str(e)))
        finally:
            # Released whatever happened - returned, raised, or was cancelled.
            # A resource let go only on the happy path is a resource that
            # deadlocks on the unhappy one.
            self._release_resources(claimed)

    def _claim_resources(self, tool_name: str,
                         arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Take what this call needs, or report who has it.

        Atomic: core/resources.acquire has no await in it, so two tasks on this
        event loop cannot both succeed. Never raises - a resource reading that
        fails must not stop a tool running, because the worst case of not
        tracking is the behaviour Leti had before runtime tracking existed.
        """
        owner = self._resource_owner()
        try:
            wanted = resources.for_tool(tool_name, arguments)
            if not wanted:
                return {"owner": owner, "taken": []}
            taken: List[str] = []
            for entry in wanted:
                outcome = resources.acquire(entry["resource"], entry["mode"], owner,
                                            reason=tool_name)
                if not outcome["acquired"]:
                    # Give back whatever this call already took, so a partial
                    # claim never becomes a lock nobody releases.
                    for name in taken:
                        resources.release(name, owner)
                    return {"owner": owner, "taken": [], "blocked": True,
                            "resource": outcome["resource"],
                            "held_by": outcome["held_by"],
                            "explain": resources.describe_conflict(
                                outcome, self._task_name(outcome["held_by"]))}
                taken.append(entry["resource"])
            return {"owner": owner, "taken": taken}
        except Exception as e:
            logger.debug(f"Resource claim for {tool_name} skipped: {e}")
            return {"owner": owner, "taken": []}

    def _release_resources(self, claimed: Dict[str, Any]) -> None:
        try:
            for name in claimed.get("taken") or []:
                resources.release(name, claimed["owner"])
        except Exception as e:
            logger.debug(f"Couldn't release a resource: {e}")

    def _resource_owner(self) -> str:
        """Who is holding this - the running task, or this conversation.

        A turn typed by the user is as much a holder as a task step is: the
        point is that two things do not write the same file, and one of them
        being a person at the keyboard does not change that.
        """
        return getattr(self, "_owning_task", None) or "conversation"

    @staticmethod
    def _task_name(task_id: str) -> str:
        if not task_id or task_id == "conversation":
            return "Something you asked for directly"
        try:
            from core import task_manager

            task = task_manager.get_task(task_id)
            return f"'{task.get('name') or task.get('objective')}'" if task else task_id
        except Exception:
            return task_id

    def _explain_failure(self, tool_name: str, result: ToolResult) -> ToolResult:
        """When an external call fails and its account is not set up, say so.

        A failed send_email with no mail account configured reads to the model as
        a mystery, and the answer that follows is a guess about servers. One line
        from core/connections.py - which reads settings, contacts nothing and
        returns no credential - turns it into "there is no email account; add one
        under Connections". Silent for every tool that needs no account, and for
        every account that IS set up, which is the common case.
        """
        try:
            warning = connections.warning_for(tool_name)
            if warning and result.error:
                result.error = f"{result.error} - {warning}"
        except Exception as e:
            logger.debug(f"Couldn't add connection context to a failure: {e}")
        return result

    def _verified(self, tool_name: str, arguments: Dict[str, Any],
                  result: ToolResult) -> ToolResult:
        """Say what is actually known about what that call did.

        core/verification.py answers "did the intended thing happen", which is not
        the question "did the call return". When it cannot confirm something, the
        model is told so IN the tool result, where it cannot be missed on the way
        to writing the answer - and only then: a VERIFIED or NOT APPLICABLE check
        adds nothing, so most calls come back untouched and cost nothing.

        A verification never changes success/failure. It is evidence about the
        world, not a second opinion about the call.
        """
        try:
            found = verification.verify(tool_name, arguments, result.output)
            note = verification.note_for(found)
            if not note:
                return result
            if isinstance(result.output, dict):
                result.output = {**result.output, "verification": found,
                                 "how_to_report": note}
            else:
                result.output = {"result": result.output, "verification": found,
                                 "how_to_report": note}
        except Exception as e:
            logger.debug(f"Verification of {tool_name} skipped: {e}")
        return result

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
