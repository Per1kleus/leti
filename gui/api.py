"""
GUI backend: the orchestrator-facing side of the HUD. Both the desktop
pywebview window AND any phone/browser connect through the SAME websocket
protocol (see gui/server.py) - the pywebview window is just a native
wrapper pointed at http://127.0.0.1:<port>/, not a special bridge, so
there's exactly one code path for "the UI talks to Leti" regardless of
which device is asking. That's what makes "launchable from Android" a
real feature rather than a second, divergent implementation: a phone on
the same Wi-Fi opens the printed URL and gets the identical live session.

Threading model: the orchestrator, tools, and voice pipeline are all
asyncio-based and run on a dedicated background thread with its own event
loop, started here. gui/server.py's websocket handler runs ON THAT SAME
LOOP (not a separate one), so it can call the async methods below with a
plain `await` - no thread-hop needed for the request/response path. The
only thing that still crosses a thread boundary is push() broadcasting an
update out to clients, which is fire-and-forget (never blocks), so it's
safe to call from any thread.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Set

from core.orchestrator import Orchestrator
from core.safety_guard import SafetyGuard
from tools.system_health import RunHealthCheckTool
from tools.todo_list import add_todo, delete_todo, load_todos, toggle_todo
from tools.weather import get_current_weather

logger = logging.getLogger("leti.gui")

# Methods that need real async work (network calls, the orchestrator) get an
# "a_" coroutine below, awaited directly by gui/server.py's handler. Methods
# that are just fast local JSON file I/O (todos, session history) don't need
# that treatment - they're safe to call synchronously from within the async
# handler too, so they're just plain methods.
SYNC_METHODS = {
    "get_todos", "add_todo_item", "toggle_todo_item", "delete_todo_item", "get_session_messages",
    "list_settings_sections", "get_settings_section", "update_settings_section", "clear_settings_section",
}
ASYNC_METHODS = {"send_text_message", "get_system_stats", "get_weather"}


class LetiAPI:
    def __init__(self, orchestrator: Orchestrator, loop: asyncio.AbstractEventLoop):
        self.orchestrator = orchestrator
        self.loop = loop
        self.ws_clients: Set[Any] = set()  # every connected client: desktop window + any phones/browsers

    def push(self, fn_name: str, *args: Any) -> None:
        """Broadcasts a call to a named JS function (e.g. appendLetiReply) to every
        connected client at once - one shared live session across every surface."""
        if not self.ws_clients:
            return
        payload = json.dumps({"type": "push", "fn": fn_name, "args": list(args)})
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)

    async def _broadcast(self, payload: str) -> None:
        dead = set()
        for ws in list(self.ws_clients):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.add(ws)
        self.ws_clients -= dead

    # ---- Async: real work, awaited directly by gui/server.py ----

    async def a_send_text_message(self, text: str) -> None:
        try:
            await self.orchestrator.handle_user_input(text, voice_mode=False)
        except Exception as e:
            logger.exception("Error handling typed message")
            self.push("appendLetiReply", f"Something went wrong: {e}")

    async def a_get_system_stats(self) -> dict:
        try:
            result = await RunHealthCheckTool().run()
            metrics = result.output.get("metrics", {}) if result.success else {}
            disks = metrics.get("disks") or []
            return {
                "cpu": round(metrics.get("cpu_percent", 0)),
                "mem": round(metrics.get("memory_percent", 0)),
                "disk": round(disks[0]["percent_used"]) if disks else 0,
            }
        except Exception as e:
            logger.warning(f"get_system_stats failed: {e}")
            return {"cpu": 0, "mem": 0, "disk": 0, "error": str(e)}

    async def a_get_weather(self) -> dict:
        try:
            return await get_current_weather()
        except Exception as e:
            logger.warning(f"get_weather failed: {e}")
            return {"error": str(e)}

    # ---- Sync: fast local file I/O, safe to call from anywhere. Return plain
    # Python values (not json.dumps strings) - gui/server.py's response envelope
    # already handles JSON-encoding the whole message once, so double-encoding
    # here would make the frontend parse it twice. ----

    def get_todos(self) -> list:
        return load_todos()

    def add_todo_item(self, text: str) -> dict:
        return add_todo(text)

    def toggle_todo_item(self, item_id: str) -> bool:
        return toggle_todo(item_id)

    def delete_todo_item(self, item_id: str) -> bool:
        return delete_todo(item_id)

    def get_session_messages(self) -> list:
        """Current session only (the in-process rolling buffer) - not past sessions,
        per the user's explicit answer when this was designed."""
        return self.orchestrator.session_memory.get_recent_messages()

    # ---- Settings editor (the /settings command) - deterministic, deliberately
    # not routed through the orchestrator/LLM. See core/settings_editor.py. ----

    def list_settings_sections(self) -> list:
        from core import settings_editor as se
        return se.list_sections()

    def get_settings_section(self, name: str) -> dict:
        from core import settings_editor as se
        try:
            return se.get_section(name)
        except KeyError as e:
            return {"error": str(e)}

    def update_settings_section(self, name: str, values: dict) -> dict:
        from core import settings_editor as se
        try:
            return se.update_section(name, values)
        except (KeyError, ValueError) as e:
            return {"error": str(e)}

    def clear_settings_section(self, name: str) -> bool:
        from core import settings_editor as se
        try:
            se.clear_section(name)
            return True
        except KeyError:
            return False


def _make_gui_speak_callback(api: LetiAPI, tts):
    """Unified reply handler for every input channel (typed, voice, from the desktop
    window or a phone): shows the reply as a bubble, speaks it, and focuses the text
    input once Leti finishes - satisfying "focus text after Leti finishes speaking"
    identically regardless of which surface triggered the turn."""

    async def _speak(text: str) -> None:
        api.push("appendLetiReply", text)
        api.push("setHudState", "speaking")
        await tts.speak(text)
        api.push("setHudState", "idle")
        api.push("focusChatInput")

    return _speak


def _make_gui_visual_callback(api: LetiAPI):
    """Pushes image/diagram results straight to every connected client, independent
    of whatever the model ends up saying in text - see core/orchestrator.py's
    visual_callback hook."""

    async def _show(visual: dict) -> None:
        api.push("showVisual", visual)

    return _show


async def _run_voice_loop(orchestrator: Orchestrator, api: LetiAPI, continuous: bool, transcriber) -> None:
    """Same shape as main.py's run_voice_mode, with two additions: pushing the
    user's transcribed utterance to every connected client (typed messages show
    themselves; spoken ones need Python to report them), and focusing the text
    input the moment the user finishes talking - before Leti's reply even starts.

    `transcriber` is passed in already-constructed (see run_gui_mode) rather than
    built here: loading the Whisper model is genuinely slow (real disk I/O plus
    model init - seconds at minimum), and doing that inside a coroutine on the
    SAME event loop the websocket server runs on would freeze every connected
    client's chat messages for however long it takes. Loading it up front,
    before the server is marked ready, means nothing is ever silently frozen
    once the app looks interactive."""
    loop = asyncio.get_event_loop()

    async def handle_utterance(text: str) -> None:
        if not text:
            return
        api.push("appendUserMessage", text)
        api.push("focusChatInput")  # user just finished speaking
        await orchestrator.handle_user_input(text, voice_mode=True)

    if continuous:
        from audio.wake_word import WakeWordListener

        async def on_wake():
            api.push("setHudState", "listening")
            audio = await transcriber.record_until_silence()
            text = await transcriber.transcribe(audio)
            api.push("setHudState", "idle")
            await handle_utterance(text)

        # WakeWordListener's constructor also loads a (much smaller) model - still
        # offloaded to a thread rather than assumed harmless, for the same reason.
        listener = await loop.run_in_executor(None, lambda: WakeWordListener(on_wake=on_wake))
        await listener.start()
    else:
        while True:
            api.push("setHudState", "listening")
            audio = await transcriber.record_until_silence()
            text = await transcriber.transcribe(audio)
            api.push("setHudState", "idle")
            await handle_utterance(text)


def run_gui_mode(orchestrator: Orchestrator, safety_guard: SafetyGuard, loop: asyncio.AbstractEventLoop, continuous_voice: bool = True) -> None:
    """Entry point called from main.py's run_gui() for `--mode gui`. `loop` is an
    already-created event loop that `orchestrator` (and everything it depends on)
    was constructed on via run_until_complete - this keeps that SAME loop alive on
    a background thread (so bound resources like the LLM client's HTTP connections
    keep working) while the web server and, if available, a native window run.
    Blocks until the app is closed (window close, or Ctrl+C in headless/browser-only
    mode when pywebview isn't installed)."""
    from audio.tts import Pyttsx3TTS
    from audio.stt import WhisperTranscriber
    from gui.server import LetiWebServer

    def _run_background_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run_background_loop, daemon=True)
    thread.start()

    api = LetiAPI(orchestrator, loop)

    # Load speech models up front, synchronously, on THIS (main) thread, before the
    # server or window exist at all - deliberately blocking here, not later. The
    # alternative (loading inside a coroutine on the shared background loop) is
    # what used to cause chat messages to silently hang: that loop is what the
    # websocket server runs on too, so anything blocking on it freezes every
    # connected client for however long model loading takes. One Whisper model is
    # loaded once and reused for both voice input and confirmation prompts -
    # loading it twice was also just wasted time and memory.
    print("Loading voice models (Whisper + TTS) - this happens once per launch...")
    tts = Pyttsx3TTS()
    transcriber = WhisperTranscriber()
    print("Voice models ready.")

    orchestrator.speak_callback = _make_gui_speak_callback(api, tts)
    orchestrator.visual_callback = _make_gui_visual_callback(api)

    # Route risky/destructive confirmations through voice, same as CLI voice mode -
    # there's no phone-side confirmation UI (yet), so a risky/destructive action
    # requested from a phone still confirms via this machine's own voice/mic. Worth
    # knowing if you're driving Leti remotely and not standing next to the desktop.
    from core.confirmation import make_voice_confirmation_callback
    safety_guard.set_confirmation_callback(make_voice_confirmation_callback(tts, transcriber))

    server = LetiWebServer(api)
    server_ready = threading.Event()

    async def _start_server_and_voice():
        await server.start()
        server_ready.set()
        try:
            await _run_voice_loop(orchestrator, api, continuous_voice, transcriber)
        except Exception:
            # A voice/mic failure (e.g. no audio input device available) shouldn't
            # take the whole app down - the server above is already up and fully
            # usable via text regardless of what happens to voice.
            logger.exception("Voice pipeline stopped unexpectedly - text chat is unaffected.")

    asyncio.run_coroutine_threadsafe(_start_server_and_voice(), loop)
    if not server_ready.wait(timeout=15):
        raise RuntimeError("The Leti web server didn't start within 15s - check the logs above.")

    try:
        import webview
    except ImportError:
        webview = None

    if webview:
        webview.create_window(
            "Leti", f"http://127.0.0.1:{server.port}/", width=1180, height=760,
            background_color="#050b14",
        )
        webview.start()  # blocks until the window is closed
    else:
        print("(pywebview not installed - no desktop window will open, but the web")
        print(" interface above is fully usable from any browser, including this")
        print(" machine's. Press Ctrl+C to stop Leti.)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")

    asyncio.run_coroutine_threadsafe(server.stop(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
