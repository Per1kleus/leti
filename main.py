"""
Leti entry point.

Usage:
    python main.py --mode text       # typed input/output in the terminal (no mic/speaker needed)
    python main.py --mode voice      # push-to-talk voice mode
    python main.py --mode continuous # always-listening wake-word mode

Run `ollama pull <model>` for the models configured in config/settings.yaml
before starting, and make sure `ollama serve` is running.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Dict

from core.config_loader import ensure_data_dirs, get_settings
from core.llm_client import OllamaClient
from core.orchestrator import Orchestrator
from core.safety_guard import SafetyGuard
from memory.session_memory import SessionMemory
from memory.vector_store import VectorMemory
from tools.base import ToolRegistry
from tools.browser import (
    BrowserClickTool,
    BrowserFillFormTool,
    BrowserReadPageTool,
    BrowserSession,
    set_shared_session,
)
from tools.file_manager import DeleteFileTool, ListFilesTool, MoveFileTool, ReadFileTool, WriteFileTool
from tools.os_control import (
    CloseAppTool,
    FocusWindowTool,
    KeyboardHotkeyTool,
    KeyboardTypeTool,
    LaunchAppTool,
    MouseClickTool,
)
from tools.coding import (
    RunCodeTool,
    RunTestsTool,
    InstallDependencyTool,
    InspectProjectTool,
)
from tools.shell_runner import ShellRunnerTool
from tools.vision import ReadScreenTool
from tools.web_search import ResearchTopicTool, WebSearchTool
from tools.network_security import InspectNetworkConnectionsTool, FirewallStatusTool, LanDeviceListTool
from tools.system_defense import (
    EnableFirewallTool,
    DetectBruteForceTool,
    CheckPersistenceTool,
    KillProcessTool,
)
from tools.backup_restore import CreateSecuritySnapshotTool, CheckIntegrityTool, RestoreFromSnapshotTool
from tools.email_client import ListNewEmailsTool, SendEmailTool
from tools.trading_platform import (
    GetWatchlistTool,
    SetWatchlistTool,
    GetQuoteTool,
    EvaluateStrategyTool,
    GetPositionsTool,
    PlacePaperOrderTool,
)
from core.confirmation import cli_confirmation_callback, make_voice_confirmation_callback
from tools.contacts import (
    AddContactTool,
    ListContactsTool,
    ResolveContactTool,
    UpdateContactTool,
    DeleteContactTool,
)
from tools.venture_scout import ScoutFindTrendsTool, ScoutStressTestTool
from tools.meeting_scheduler import ScheduleMeetingTool, CreateVideoMeetingLinkTool, SendMeetingInviteEmailTool
from tools.system_health import SystemReportTool, ApplySystemUpdatesTool
from tools.personality import (
    GetPersonalitySettingsTool,
    SetPersonalityTool,
    ApplyPersonalityPresetTool,
    ResetPersonalityTool,
)
from tools.user_profile import (
    RememberAboutUserTool,
    SetUserNameTool,
    ViewUserProfileTool,
    ForgetUserFactTool,
    ClearUserProfileTool,
)
from tools.social_media import (
    GetSocialContentTool,
    SearchSocialTool,
    AddSocialWatchTool,
    ListSocialWatchesTool,
    RemoveSocialWatchTool,
    CheckSocialWatchesTool,
)
from tools.social_login import (
    SocialLoginManager,
    LoginToSocialPlatformTool,
    LogoutSocialPlatformTool,
)
from tools.weather import GetWeatherTool
from tools.todo_list import AddTodoItemTool, ListTodoItemsTool, CompleteTodoItemTool, DeleteTodoItemTool
from tools.projects import (
    CreateProjectTool,
    ListProjectsTool,
    OpenProjectTool,
    CloseProjectTool,
    UpdateProjectTool,
    GetProjectContextTool,
    DeleteProjectTool,
)
from tools.data_analysis import (
    InspectDatasetTool,
    CleanDatasetTool,
    AnalyzeDatasetTool,
    VisualizeDatasetTool,
    ExportDatasetTool,
)
from tools.engineering import (
    EngineeringCalculateTool,
    ConvertUnitsTool,
    CheckDimensionsTool,
    SolveSymbolicTool,
)
from tools.business import (
    RecordBusinessDataTool,
    UpdateLeadTool,
    ListBusinessDataTool,
    BusinessDashboardTool,
    NextActionsTool,
)
from tools.scheduler import (
    CreateScheduledTaskTool,
    ListScheduledTasksTool,
    UpdateScheduledTaskTool,
    DeleteScheduledTaskTool,
    RunScheduledTaskNowTool,
    SystemSchedulingTool,
    SchedulerRunner,
    set_runner,
)
from tools.image_search import SearchImagesTool
from tools.sketch import CreateSketchTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("leti.main")


def _apply_log_level() -> None:
    """Honor app.log_level from settings.yaml, which was documented but unread.

    Called after config is available; basicConfig above still runs first so that
    a failure to load config is itself logged.
    """
    level_name = str(get_settings().get("app", {}).get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, None)
    if isinstance(level, int):
        logging.getLogger().setLevel(level)
    else:
        logger.warning(f"Unknown app.log_level '{level_name}' - keeping INFO.")


def build_tool_registry(llm_client: OllamaClient, browser_session: BrowserSession, social_login_manager: SocialLoginManager) -> ToolRegistry:
    registry = ToolRegistry()
    # Modules that need page text (webpage watches) reach this same browser.
    set_shared_session(browser_session)

    registry.register(ReadScreenTool(llm_client))
    registry.register(WebSearchTool())
    registry.register(ResearchTopicTool(browser_session))
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListFilesTool())
    registry.register(DeleteFileTool())
    registry.register(MoveFileTool())
    registry.register(LaunchAppTool())
    registry.register(CloseAppTool())
    registry.register(FocusWindowTool())
    registry.register(MouseClickTool())
    registry.register(KeyboardTypeTool())
    registry.register(KeyboardHotkeyTool())
    registry.register(ShellRunnerTool())

    # Coding: run/test/inspect. Reading and writing source is the file tools;
    # git, builds and deploys are run_shell_command. Neither is duplicated here.
    registry.register(RunCodeTool())
    registry.register(RunTestsTool())
    registry.register(InstallDependencyTool())
    registry.register(InspectProjectTool())
    registry.register(BrowserReadPageTool(browser_session))
    registry.register(BrowserClickTool(browser_session))
    registry.register(BrowserFillFormTool(browser_session))

    # Cybersecurity: network diagnostics, active defense, integrity/backup.
    registry.register(InspectNetworkConnectionsTool())
    registry.register(FirewallStatusTool())
    registry.register(LanDeviceListTool())
    registry.register(EnableFirewallTool())
    registry.register(DetectBruteForceTool())
    registry.register(CheckPersistenceTool())
    registry.register(KillProcessTool())
    registry.register(CreateSecuritySnapshotTool())
    registry.register(CheckIntegrityTool())
    registry.register(RestoreFromSnapshotTool())

    # Email: read/categorize/send.
    registry.register(ListNewEmailsTool())
    registry.register(SendEmailTool())

    # Trading: paper/sandbox monitoring + simple strategies only.
    registry.register(GetWatchlistTool())
    registry.register(SetWatchlistTool())
    registry.register(GetQuoteTool())
    registry.register(EvaluateStrategyTool())
    registry.register(GetPositionsTool())
    registry.register(PlacePaperOrderTool())

    # Contacts: name -> disambiguated contact resolution.
    registry.register(AddContactTool())
    registry.register(ListContactsTool())
    registry.register(ResolveContactTool())
    registry.register(UpdateContactTool())
    registry.register(DeleteContactTool())

    # Venture Scout / Red-Team Strategist / COO persona - local reasoning-model sub-calls.
    registry.register(ScoutFindTrendsTool(llm_client))
    registry.register(ScoutStressTestTool(llm_client))

    # Meeting scheduling: CalDAV calendar + Zoom/Teams link generation + invite email.
    registry.register(ScheduleMeetingTool())
    registry.register(CreateVideoMeetingLinkTool())
    registry.register(SendMeetingInviteEmailTool())

    # System specs, health/performance monitoring, and OS updates.
    registry.register(SystemReportTool())
    registry.register(ApplySystemUpdatesTool())

    # Personalization: personality dials + structured, editable user profile.
    registry.register(GetPersonalitySettingsTool())
    registry.register(SetPersonalityTool())
    registry.register(ApplyPersonalityPresetTool())
    registry.register(ResetPersonalityTool())
    registry.register(RememberAboutUserTool())
    registry.register(SetUserNameTool())
    registry.register(ViewUserProfileTool())
    registry.register(ForgetUserFactTool())
    registry.register(ClearUserProfileTool())

    # Social media: one content tool across every platform (YouTube, Reddit,
    # Instagram/TikTok/Facebook, any web page), one search tool, and the watch list.
    # Logging in to the cookie-based platforms is separate, in social_login.
    registry.register(GetSocialContentTool())
    registry.register(SearchSocialTool())
    registry.register(AddSocialWatchTool())
    registry.register(ListSocialWatchesTool())
    registry.register(RemoveSocialWatchTool())
    registry.register(CheckSocialWatchesTool())
    registry.register(LoginToSocialPlatformTool(social_login_manager))
    registry.register(LogoutSocialPlatformTool(social_login_manager))

    # Dashboard data sources: weather + to-do list (also usable conversationally).
    registry.register(GetWeatherTool())
    registry.register(AddTodoItemTool())
    registry.register(ListTodoItemsTool())
    registry.register(CompleteTodoItemTool())
    registry.register(DeleteTodoItemTool())

    # Project workspaces: a persistent folder + standing instructions per project.
    # Everything a project "contains" is files in its folder, so the file, coding
    # and data tools operate on projects without needing to know about them.
    registry.register(CreateProjectTool())
    registry.register(ListProjectsTool())
    registry.register(OpenProjectTool())
    registry.register(CloseProjectTool())
    registry.register(UpdateProjectTool())
    registry.register(GetProjectContextTool())
    registry.register(DeleteProjectTool())

    # Data analysis: one loader covers CSV/Excel/JSON/TXT/Parquet/MATLAB, so every
    # tool here is format-blind. Output lands in the active project's folder.
    registry.register(InspectDatasetTool())
    registry.register(CleanDatasetTool())
    registry.register(AnalyzeDatasetTool())
    registry.register(VisualizeDatasetTool())
    registry.register(ExportDatasetTool())

    # Engineering: units and symbolic maths computed by pint and sympy rather than
    # by the model. Numerical work beyond them runs as real code through run_code.
    registry.register(EngineeringCalculateTool())
    registry.register(ConvertUnitsTool())
    registry.register(CheckDimensionsTool())
    registry.register(SolveSymbolicTool())

    # Business intelligence: leads, income and expenses, with every metric computed
    # from them. People are not stored here - a record links to a contact by id, so
    # the contact book stays the one place someone's details live.
    registry.register(RecordBusinessDataTool())
    registry.register(UpdateLeadTool())
    registry.register(ListBusinessDataTool())
    registry.register(BusinessDashboardTool())
    registry.register(NextActionsTool())

    # Scheduling: a task is an instruction plus a schedule, and it runs through the
    # orchestrator when it fires - so it reaches every capability above without the
    # scheduler knowing any of them exist. Scheduled meetings are just tasks whose
    # instruction calls the existing schedule_meeting tools.
    registry.register(CreateScheduledTaskTool())
    registry.register(ListScheduledTasksTool())
    registry.register(UpdateScheduledTaskTool())
    registry.register(DeleteScheduledTaskTool())
    registry.register(RunScheduledTaskNowTool())
    registry.register(SystemSchedulingTool())

    # Visual output: image search + simple diagrams, pushed straight to the GUI via
    # the orchestrator's visual_callback (see core/orchestrator.py) when present.
    registry.register(SearchImagesTool())
    registry.register(CreateSketchTool())
    return registry


async def _run_settings_editor_cli() -> None:
    """Interactive /settings flow for text mode - lists sections, lets the user
    pick one, prompts field-by-field. Deliberately plain input()/print() rather
    than routing through the orchestrator: this is a deterministic form-filling
    task, not something that benefits from an LLM call, and secrets shouldn't
    pass through a chat history either."""
    from core import settings_editor as se
    from core.console_input import read_line

    async def ask(prompt: str) -> str:
        answer = await read_line(prompt)
        return (answer or "").strip()

    sections = se.list_sections()
    print("\n--- Leti settings ---")
    for i, s in enumerate(sections, 1):
        status = "configured" if s["configured"] else "not set"
        print(f"  {i}. {s['label']} ({status})")
    print("  Type a number/name to edit it, 'clear <name>' to reset a section, or 'cancel'.\n")

    choice = (await ask("> ")).lower()
    if choice in ("cancel", ""):
        print("Cancelled.\n")
        return

    if choice.startswith("clear "):
        target = choice[len("clear "):].strip()
        matched = next((s["name"] for s in sections if s["name"] == target or s["label"].lower() == target), None)
        if not matched:
            print(f"Unknown section: '{target}'\n")
            return
        se.clear_section(matched)
        print(f"Cleared {matched} settings.\n")
        return

    name = None
    if choice.isdigit() and 1 <= int(choice) <= len(sections):
        name = sections[int(choice) - 1]["name"]
    else:
        name = next((s["name"] for s in sections if s["name"] == choice or s["label"].lower() == choice), None)
    if not name:
        print(f"Didn't recognize '{choice}'.\n")
        return

    section = se.get_section(name)
    print(f"\n--- {section['label']} ---")
    print("Press Enter to leave a field unchanged. Type 'cancel' at any point to abort.\n")

    values: Dict[str, str] = {}
    for field in section["fields"]:
        if field.get("secret"):
            status = "(currently set - Enter to keep, or type a new value)" if field["is_set"] else "(not set)"
        elif field["is_set"]:
            status = f"(current: {field['value']})"
        elif field.get("example"):
            status = f"(e.g. {field['example']})"
        elif field.get("default") is not None:
            status = f"(default: {field['default']})"
        else:
            status = "(not set)"

        raw = await ask(f"{field['label']} {status}: ")
        if raw.lower() == "cancel":
            print("Cancelled - nothing saved.\n")
            return
        if raw:
            values[field["key"]] = raw

    if not values:
        print("Nothing changed.\n")
        return

    try:
        se.update_section(name, values)
        print(f"\nSaved {section['label']} settings. They're active immediately - no restart needed.\n")
    except ValueError as e:
        print(f"\nCouldn't save: {e}\n")


async def run_text_mode(orchestrator: Orchestrator) -> None:
    from core.console_input import read_line

    print("Leti is ready. Type your message ('/settings' to configure integrations, "
          "'/audio' for microphone and speakers, 'exit' to quit).\n")
    while True:
        # Shared stdin reader, not input() on an executor - see core/console_input.py.
        line = await read_line("You: ")
        if line is None:      # Ctrl-D / end of piped input
            break
        user_text = line.strip()
        if user_text.lower() in ("exit", "quit"):
            break
        if not user_text:
            continue
        if user_text.lower().startswith("/settings"):
            await _run_settings_editor_cli()
            continue
        if user_text.lower().startswith("/audio"):
            # Same flow as the first launch, on purpose: one implementation of
            # "ask, test, save", whether it's the first time or a re-check.
            from audio.setup import run_console_setup

            await run_console_setup(force=True)
            continue
        answer = await orchestrator.handle_user_input(user_text)
        print(f"Leti: {answer}\n")


async def run_voice_mode(orchestrator: Orchestrator, safety_guard: SafetyGuard, continuous: bool) -> None:
    from audio import load_voice_stack

    tts, transcriber, voice_error = load_voice_stack()
    if voice_error:
        # Unlike GUI mode, there's nothing to fall back to here - voice IS the mode
        # that was asked for. But "you said no" and "espeak isn't installed" are
        # different problems with different fixes, and a traceback names neither.
        from audio.setup import blocked_reason

        if blocked_reason():
            print(f"\nVoice mode needs the microphone, which is currently turned off.\n"
                  f"  {voice_error}\n"
                  f"Run `python main.py --mode text` and type /audio to turn it back on.\n")
        else:
            print(f"\nVoice mode needs a working microphone and speaker, and this machine "
                  f"couldn't provide one:\n  {voice_error}\n\n"
                  f"On Linux, text-to-speech needs espeak (`sudo apt install espeak`).\n"
                  f"To use Leti without voice, run:  python main.py --mode text\n"
                  f"                            or:  python main.py --mode gui\n")
        return

    # Route risky/destructive confirmations through voice instead of terminal typing -
    # see make_voice_confirmation_callback for how it interprets a spoken yes/no.
    safety_guard.set_confirmation_callback(make_voice_confirmation_callback(tts, transcriber))

    async def speak(text: str) -> None:
        print(f"Leti: {text}")
        await tts.speak(text)

    orchestrator.speak_callback = speak

    if continuous:
        from audio.wake_word import WakeWordListener

        async def on_wake():
            print("Wake word detected, listening...")
            audio = await transcriber.record_until_silence()
            text = await transcriber.transcribe(audio)
            if text:
                print(f"You said: {text}")
                await orchestrator.handle_user_input(text, voice_mode=True)

        listener = WakeWordListener(on_wake=on_wake)
        print("Leti is listening for the wake word. Press Ctrl+C to stop.")
        await listener.start()
    else:
        from core.console_input import read_line

        print("Push-to-talk mode: press Enter to start talking; recording stops on silence.")
        while True:
            if await read_line("\nPress Enter to talk (Ctrl+D to quit)...") is None:
                break
            print("Recording... (stops automatically when you stop speaking)")
            # Simplified PTT: record until silence to avoid needing a separate key-release thread.
            audio = await transcriber.record_until_silence()
            text = await transcriber.transcribe(audio)
            if text:
                print(f"You said: {text}")
                await orchestrator.handle_user_input(text, voice_mode=True)
            else:
                print("(no speech detected)")


async def build_app(start_scheduler: bool = True):
    """Shared async setup for every mode - constructs and returns everything a mode
    needs. Factored out so GUI mode (see run_gui() below) can run this via
    run_until_complete() on its own dedicated loop, instead of the loop
    asyncio.run(main()) creates - the two must never nest (see run_gui's docstring)."""
    ensure_data_dirs()
    _apply_log_level()
    settings = get_settings()

    llm_client = OllamaClient()
    if not await llm_client.is_available():
        logger.error(
            f"Cannot reach Ollama at {settings['ollama']['host']}. "
            "Make sure 'ollama serve' is running and models are pulled."
        )
        sys.exit(1)

    browser_session = BrowserSession()
    social_login_manager = SocialLoginManager()
    tool_registry = build_tool_registry(llm_client, browser_session, social_login_manager)

    safety_guard = SafetyGuard(confirmation_callback=cli_confirmation_callback)
    session_memory = SessionMemory()
    vector_memory = VectorMemory(llm_client)

    orchestrator = Orchestrator(
        llm_client=llm_client,
        tool_registry=tool_registry,
        safety_guard=safety_guard,
        session_memory=session_memory,
        vector_memory=vector_memory,
    )

    # The scheduler runs tasks through this orchestrator, so it can only start once
    # the orchestrator exists. Failure notifications go wherever replies go, which
    # in GUI mode is the chat and in voice mode is spoken.
    async def notify_failure(message: str) -> None:
        logger.warning(message)
        if orchestrator.speak_callback:
            await orchestrator.speak_callback(message)

    scheduler = SchedulerRunner(orchestrator, notify=notify_failure)
    set_runner(scheduler)
    if start_scheduler:
        scheduler.start()

    return llm_client, browser_session, social_login_manager, orchestrator, safety_guard, scheduler


async def run_scheduled_once() -> int:
    """Run whatever scheduled tasks are due, then exit.

    This is what the OS scheduler invokes (see core/system_scheduler.py), which is
    how a task scheduled for Monday morning happens on Monday morning rather than
    the next time someone opens Leti.

    Two things differ from a normal session. The in-process loop is not started -
    this process checks once and leaves. And SafetyGuard runs unattended: there is
    no terminal and no window here, so a confirmation prompt would hang forever.
    Unattended runs may do what scheduler.unattended_allows permits and refuse the
    rest with a reason that lands in the task's history.
    """
    llm_client, browser_session, social_login_manager, orchestrator, safety_guard, scheduler = \
        await build_app(start_scheduler=False)
    safety_guard.set_unattended(True)

    try:
        outcomes = await scheduler.run_due_tasks()
        if not outcomes:
            logger.info("No scheduled tasks were due.")
            return 0
        for outcome in outcomes:
            if outcome["status"] == "succeeded":
                logger.info(f"Task '{outcome['name']}' succeeded in {outcome['duration_seconds']}s.")
            else:
                logger.error(f"Task '{outcome['name']}' {outcome['status']}: {outcome.get('error')}")
        # Non-zero if anything failed, so the OS scheduler's own log shows it.
        return 0 if all(o["status"] == "succeeded" for o in outcomes) else 1
    finally:
        await scheduler.stop()
        await llm_client.close()
        await browser_session.close()
        await social_login_manager.close()


async def main(args) -> None:
    """CLI modes only (text/voice/continuous). GUI mode is handled by run_gui()
    instead, called directly from __main__ - see its docstring for why."""
    llm_client, browser_session, social_login_manager, orchestrator, safety_guard, scheduler = await build_app()

    # Ask about the microphone and speakers once, on the first launch, before any
    # mode starts using them. On macOS and Windows this is also what makes the OS's
    # own permission dialog appear while the user is looking at an explanation of
    # why - rather than mid-conversation, behind the window. See audio/setup.py.
    from audio.setup import run_console_setup

    await run_console_setup()

    try:
        if args.mode == "text":
            await run_text_mode(orchestrator)
        elif args.mode == "voice":
            await run_voice_mode(orchestrator, safety_guard, continuous=False)
        elif args.mode == "continuous":
            await run_voice_mode(orchestrator, safety_guard, continuous=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await scheduler.stop()
        await llm_client.close()
        await browser_session.close()
        await social_login_manager.close()


def run_gui() -> None:
    """GUI mode's entry point - deliberately NOT run through asyncio.run(), because
    pywebview's blocking webview.start() must own the main thread's native GUI loop,
    and llm_client/orchestrator need to be constructed on the same asyncio loop that
    will later service them from a background thread (see gui/api.py's docstring).
    Nesting a second run_until_complete() inside asyncio.run(main())'s already-running
    loop on the same thread is fragile and unsupported, so this sidesteps that
    entirely by never calling asyncio.run() at all in this path."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    llm_client, browser_session, social_login_manager, orchestrator, safety_guard, scheduler = \
        loop.run_until_complete(build_app())

    from gui.api import run_gui_mode
    try:
        run_gui_mode(orchestrator, safety_guard, loop, continuous_voice=True)
    finally:
        loop.run_until_complete(asyncio.gather(
            scheduler.stop(), llm_client.close(), browser_session.close(),
            social_login_manager.close(), return_exceptions=True,
        ))
        loop.close()


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Leti - local-first AI desktop assistant")
    cli_parser.add_argument(
        "--mode", choices=["text", "voice", "continuous", "gui", "run-scheduled"], default="text",
        help=(
            "Interaction mode: text (typed), voice (push-to-talk), continuous (wake word), "
            "gui (HUD window), run-scheduled (run due tasks once and exit - what the OS "
            "scheduler invokes)"
        ),
    )
    cli_args = cli_parser.parse_args()

    if cli_args.mode == "gui":
        run_gui()
    elif cli_args.mode == "run-scheduled":
        sys.exit(asyncio.run(run_scheduled_once()))
    else:
        asyncio.run(main(cli_args))
