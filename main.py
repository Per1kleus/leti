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
from tools.browser import BrowserClickTool, BrowserFillFormTool, BrowserNavigateTool, BrowserSession
from tools.file_manager import DeleteFileTool, ListFilesTool, MoveFileTool, ReadFileTool, WriteFileTool
from tools.os_control import (
    CloseAppTool,
    FocusWindowTool,
    KeyboardHotkeyTool,
    KeyboardTypeTool,
    LaunchAppTool,
    MouseClickTool,
)
from tools.shell_runner import ShellRunnerTool
from tools.vision import ReadScreenTool
from tools.web_search import WebSearchTool
from tools.network_security import ScanLocalPortsTool, FirewallStatusTool, LanDeviceListTool
from tools.system_defense import (
    EnableFirewallTool,
    DetectBruteForceTool,
    CheckPersistenceTool,
    ListSuspiciousProcessesTool,
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
from tools.system_health import GetSystemSpecsTool, RunHealthCheckTool, CheckForUpdatesTool, ApplySystemUpdatesTool
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
    GetSubredditPostsTool,
    SearchRedditTool,
    GetYouTubeChannelLatestTool,
    SearchYouTubeTrendingTool,
    AddSocialWatchTool,
    ListSocialWatchesTool,
    RemoveSocialWatchTool,
    CheckSocialWatchesTool,
)
from tools.social_login import (
    SocialLoginManager,
    LoginToSocialPlatformTool,
    LogoutSocialPlatformTool,
    GetInstagramUserLatestTool,
    GetTikTokUserLatestTool,
    GetFacebookPageLatestTool,
)
from tools.weather import GetWeatherTool
from tools.todo_list import AddTodoItemTool, ListTodoItemsTool, CompleteTodoItemTool, DeleteTodoItemTool
from tools.image_search import SearchImagesTool
from tools.sketch import CreateSketchTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("leti.main")


def build_tool_registry(llm_client: OllamaClient, browser_session: BrowserSession, social_login_manager: SocialLoginManager) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadScreenTool(llm_client))
    registry.register(WebSearchTool())
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
    registry.register(BrowserNavigateTool(browser_session))
    registry.register(BrowserClickTool(browser_session))
    registry.register(BrowserFillFormTool(browser_session))

    # Cybersecurity: network diagnostics, active defense, integrity/backup.
    registry.register(ScanLocalPortsTool())
    registry.register(FirewallStatusTool())
    registry.register(LanDeviceListTool())
    registry.register(EnableFirewallTool())
    registry.register(DetectBruteForceTool())
    registry.register(CheckPersistenceTool())
    registry.register(ListSuspiciousProcessesTool())
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
    registry.register(GetSystemSpecsTool())
    registry.register(RunHealthCheckTool())
    registry.register(CheckForUpdatesTool())
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

    # Social media: YouTube + Reddit (official, no login), watches, and
    # Instagram/TikTok/Facebook (session-cookie login via a dedicated browser manager).
    registry.register(GetSubredditPostsTool())
    registry.register(SearchRedditTool())
    registry.register(GetYouTubeChannelLatestTool())
    registry.register(SearchYouTubeTrendingTool())
    registry.register(AddSocialWatchTool())
    registry.register(ListSocialWatchesTool())
    registry.register(RemoveSocialWatchTool())
    registry.register(CheckSocialWatchesTool())
    registry.register(LoginToSocialPlatformTool(social_login_manager))
    registry.register(LogoutSocialPlatformTool(social_login_manager))
    registry.register(GetInstagramUserLatestTool(social_login_manager))
    registry.register(GetTikTokUserLatestTool(social_login_manager))
    registry.register(GetFacebookPageLatestTool(social_login_manager))

    # Dashboard data sources: weather + to-do list (also usable conversationally).
    registry.register(GetWeatherTool())
    registry.register(AddTodoItemTool())
    registry.register(ListTodoItemsTool())
    registry.register(CompleteTodoItemTool())
    registry.register(DeleteTodoItemTool())

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

    print("Leti is ready. Type your message ('/settings' to configure integrations, 'exit' to quit).\n")
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
        answer = await orchestrator.handle_user_input(user_text)
        print(f"Leti: {answer}\n")


async def run_voice_mode(orchestrator: Orchestrator, safety_guard: SafetyGuard, continuous: bool) -> None:
    from audio.stt import WhisperTranscriber
    from audio.tts import Pyttsx3TTS

    transcriber = WhisperTranscriber()
    tts = Pyttsx3TTS()

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


async def build_app():
    """Shared async setup for every mode - constructs and returns everything a mode
    needs. Factored out so GUI mode (see run_gui() below) can run this via
    run_until_complete() on its own dedicated loop, instead of the loop
    asyncio.run(main()) creates - the two must never nest (see run_gui's docstring)."""
    ensure_data_dirs()
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
    return llm_client, browser_session, social_login_manager, orchestrator, safety_guard


async def main(args) -> None:
    """CLI modes only (text/voice/continuous). GUI mode is handled by run_gui()
    instead, called directly from __main__ - see its docstring for why."""
    llm_client, browser_session, social_login_manager, orchestrator, safety_guard = await build_app()

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

    llm_client, browser_session, social_login_manager, orchestrator, safety_guard = \
        loop.run_until_complete(build_app())

    from gui.api import run_gui_mode
    try:
        run_gui_mode(orchestrator, safety_guard, loop, continuous_voice=True)
    finally:
        loop.run_until_complete(asyncio.gather(
            llm_client.close(), browser_session.close(), social_login_manager.close(),
            return_exceptions=True,
        ))
        loop.close()


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Leti - local-first AI desktop assistant")
    cli_parser.add_argument(
        "--mode", choices=["text", "voice", "continuous", "gui"], default="text",
        help="Interaction mode: text (typed), voice (push-to-talk), continuous (wake word), gui (HUD window)",
    )
    cli_args = cli_parser.parse_args()

    if cli_args.mode == "gui":
        run_gui()
    else:
        asyncio.run(main(cli_args))
