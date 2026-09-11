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
from pathlib import Path
from typing import Any, Set

from core.orchestrator import Orchestrator
from core.safety_guard import SafetyGuard
from tools.system_health import SystemReportTool
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
    "get_audio_setup", "set_window_mode",
    "get_personality", "set_personality", "get_activity",
}
ASYNC_METHODS = {"send_text_message", "get_system_stats", "get_weather",
                 "check_audio", "save_audio_setup",
                 "get_model_setup", "apply_model_setup",
                 "get_permissions", "set_permission", "get_diagnostics"}


class LetiAPI:
    def __init__(self, orchestrator: Orchestrator, loop: asyncio.AbstractEventLoop):
        self.orchestrator = orchestrator
        self.loop = loop
        self.ws_clients: Set[Any] = set()  # every connected client: desktop window + any phones/browsers
        # Set while a confirmation is waiting for a typed yes/no - see
        # make_text_confirmation_callback. The next message the user sends answers
        # the question instead of starting a new turn.
        self._pending_confirmation: Any = None
        # Wired by run_gui_mode: a coroutine that loads the voice stack and starts
        # listening, so accepting the first-run audio prompt takes effect now rather
        # than at the next launch. None means voice isn't available at all here.
        self.enable_voice: Any = None
        self.voice_active = False
        self._background: Set[Any] = set()   # strong refs; asyncio only holds weak ones
        # Set by run_gui_mode when there are native windows to switch between.
        # None means the page is being viewed in a browser (or a phone), where
        # minimising can only collapse the layout inside the window it already has.
        self.desktop: Any = None

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
        # A confirmation in flight owns the next thing the user types: they were
        # asked a yes/no question and answered it, and treating that answer as a
        # fresh request would both lose the answer and act on "yes" as a prompt.
        pending = self._pending_confirmation
        if pending is not None and not pending.done():
            self.push("appendUserMessage", text)
            pending.set_result(text)
            return
        try:
            await self.orchestrator.handle_user_input(text, voice_mode=False)
        except Exception as e:
            logger.exception("Error handling typed message")
            self.push("appendLetiReply", f"Something went wrong: {e}")

    async def a_get_system_stats(self) -> dict:
        try:
            result = await SystemReportTool().run(sections=["health"])
            health = result.output.get("health", {}) if result.success else {}
            metrics = health.get("metrics", {})
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

    # ---- Audio permission (audio/setup.py). The same ask-test-save flow the
    # terminal runs, driven from the HUD instead, so the question is answered
    # wherever the user actually is - including a phone, which has no terminal
    # to read a console prompt from. ----

    def get_audio_setup(self) -> dict:
        """What we know, and what there is to choose from. Drives the first-run card."""
        from audio import setup as audio_setup

        return {
            "configured": audio_setup.is_configured(),
            "setup": audio_setup.load_setup(),
            "devices": audio_setup.list_devices(),
            # Whether voice is live in THIS session, which is not the same as whether
            # it's permitted: accepting mid-session still has to load Whisper.
            "voice_active": self.voice_active,
        }

    async def a_get_permissions(self) -> dict:
        """What Leti may do, read from the same files SafetyGuard enforces."""
        from core import permission_center

        try:
            return permission_center.overview(self.orchestrator.tool_registry)
        except Exception as e:
            logger.warning(f"Couldn't read permissions: {e}")
            return {"error": str(e), "categories": [], "all_classes": [],
                    "confirming_classes": []}

    async def a_set_permission(self, action_class: str, must_confirm: bool) -> dict:
        """Change which classes stop and ask. Writes the guard's own setting."""
        from core import permission_center

        try:
            return permission_center.set_class_confirmation(action_class, bool(must_confirm))
        except Exception as e:
            logger.exception("Couldn't change a permission")
            return {"ok": False, "error": str(e)}

    async def a_get_diagnostics(self) -> dict:
        """A snapshot for the diagnostics panel. Reads only what already exists."""
        from core import diagnostics

        try:
            return diagnostics.snapshot(self.orchestrator.tool_registry)
        except Exception as e:
            logger.warning(f"Couldn't build a diagnostics snapshot: {e}")
            return {"error": str(e)}

    async def a_get_model_setup(self) -> dict:
        """Detected hardware and the recommended model. Drives the first-run card.

        Read-only: nothing is downloaded and nothing is configured by asking.
        A failure here is reported to the card, which then offers to keep the
        current models - it must never stop the interface from loading.
        """
        from core import model_setup

        try:
            return await model_setup.gather()
        except Exception as e:
            logger.warning(f"Model setup check failed: {e}")
            return {"configured": model_setup.is_configured(), "error": str(e),
                    "hardware": {"ok": False, "error": str(e)},
                    "recommendation": {"current": model_setup.current_model(),
                                       "recommended": None, "confident": False,
                                       "reasons": [], "tradeoffs": [], "considered": []}}

    async def a_apply_model_setup(self, choice: str, model: str = "") -> dict:
        """Act on the user's answer. The only method here that writes anything."""
        from core import model_setup

        try:
            return await model_setup.apply_choice(choice, model or None)
        except Exception as e:
            logger.exception("Applying the model choice failed")
            return {"ok": False, "choice": choice, "model": model, "error": str(e),
                    "note": "Your existing model configuration has not been changed."}

    async def a_check_audio(self, what: str, device_index=None) -> dict:
        """Run one hardware check. Blocking audio I/O, so it goes to a thread -
        the websocket server shares this loop, and a 3-second recording on it
        would freeze every connected client for 3 seconds."""
        from audio import setup as audio_setup

        loop = asyncio.get_running_loop()
        if what == "microphone":
            index = int(device_index) if device_index not in (None, "", "default") else None
            return await loop.run_in_executor(None, lambda: audio_setup.measure_microphone(index))
        if what == "speakers":
            return await loop.run_in_executor(None, audio_setup.play_test_tone)
        return {"ok": False, "error": f"Unknown check: {what}"}

    async def a_save_audio_setup(self, choice: dict) -> dict:
        """Persist the answer, and start voice now if it was yes.

        Starting it here rather than telling the user to restart is the whole point
        of asking at first launch: the next thing they do should be able to be
        talking to Leti.
        """
        from audio import setup as audio_setup

        index = choice.get("device_index")
        record = audio_setup.record_choice(
            mic_allowed=bool(choice.get("microphone_allowed")),
            speakers_ok=bool(choice.get("speakers_ok")),
            device_index=int(index) if index not in (None, "", "default") else None,
            device_name=str(choice.get("device_name", "")),
            mic_measurement=choice.get("mic_measurement") or {},
            output_name=str(choice.get("output_name", "")),
            note=str(choice.get("note", "")),
        )
        started = False
        if record["microphone"]["allowed"] and self.enable_voice is not None and not self.voice_active:
            started = True
            task = asyncio.create_task(self.enable_voice())
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        return {"saved": record, "starting_voice": started}

    def set_window_mode(self, mode: str) -> dict:
        """Switch between the full window and the always-on-top puck.

        Returns whether a native swap actually happened. It won't have in a
        browser tab or on a phone, and the page then collapses its own layout
        instead - the same control doing the best available version of the same
        thing, rather than a button that silently does nothing on half the
        surfaces this interface runs on.
        """
        if self.desktop is None:
            return {"desktop": False, "mode": mode}
        ok = self.desktop.show_puck() if mode == "puck" else self.desktop.show_full()
        return {"desktop": bool(ok), "mode": mode}

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

    # ---- Personality. The panel is a view onto tools/personality.py's six dials
    # and its own store; there is no second set of values and no second file. ----

    def get_personality(self) -> dict:
        from tools import personality

        return {
            "dials": personality.PARAM_NAMES,
            "values": personality.load_values(),
            "defaults": dict(personality.DEFAULT_PERSONALITY),
            # Each preset is just six numbers, so the panel can show what one would
            # change before it is applied rather than after.
            "presets": personality.UI_PRESETS,
        }

    def set_personality(self, values: dict) -> dict:
        """Save the dials. A preset is applied by sending its six values, which is
        exactly what applying a preset means - see tools/personality.UI_PRESETS."""
        from tools import personality

        saved = personality.save_values(values or {})
        return {"values": saved, "description": personality.describe_personality(saved)}

    # ---- The activity feed behind the RT-LOG. Live entries are PUSHED as they
    # happen (see run_gui_mode); this is only the tail, read once by a client that
    # connected after some of it had already scrolled past. ----

    def get_activity(self) -> list:
        from core import diagnostics

        return diagnostics.recent_activity()


def _serve_headless(note: str = "", url: str = "") -> None:
    """The browser fallback: open the interface in the default browser, then serve.

    Reached whenever the desktop window cannot be used - pywebview missing, no
    display, or the window failing to open or run. The outcome is the same in
    every case, which is why they share one path: the server is already up, so the
    application stays usable; it just lives in a browser tab instead of its own
    window.

    Opening the browser rather than printing a URL is the difference between a
    fallback and an error message. If nothing can be opened (a headless box has no
    browser either), the URL is still printed and the server still runs, because a
    phone on the same network can reach it even when this machine cannot show it.
    """
    if note:
        print(note)
    if url:
        try:
            import webbrowser

            if webbrowser.open(url):
                print(f" Opened {url} in your browser.")
            else:
                print(f" No browser could be started here - open {url} yourself,")
                print(" or reach it from another device at the address above.")
        except Exception as e:
            logger.warning(f"Couldn't open a browser ({e}).")
            print(f" Couldn't open a browser automatically - open {url} yourself.")
    print(" Press Ctrl+C to stop Leti.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def _make_gui_speak_callback(api: LetiAPI, tts):
    """Unified reply handler for every input channel (typed, voice, from the desktop
    window or a phone): shows the reply as a bubble, speaks it, and focuses the text
    input once Leti finishes - satisfying "focus text after Leti finishes speaking"
    identically regardless of which surface triggered the turn."""

    async def _speak(text: str) -> None:
        api.push("appendLetiReply", text)
        if tts is None:
            # Text-only: the reply is already on screen, and pretending to speak
            # would leave the orb stuck in its speaking animation.
            api.push("focusChatInput")
            return
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
    from audio import load_voice_stack, setup as audio_setup
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
    # On a machine that has never been asked, DON'T touch the microphone yet: the
    # first open is what makes macOS and Windows show their permission dialog, and
    # that dialog should appear while the user is looking at the card in the HUD
    # explaining it - not seconds after launch with nothing on screen to connect it
    # to. The interface starts text-only and the card asks; accepting starts voice
    # in this same session, via api.enable_voice below.
    if not audio_setup.is_configured():
        print("Audio hasn't been set up yet - the interface will ask. Starting in text-only mode.")
        tts, transcriber, voice_error = None, None, "First-run audio setup hasn't been completed yet."
    else:
        print("Loading voice models (Whisper + TTS) - this happens once per launch...")
        tts, transcriber, voice_error = load_voice_stack()
        if voice_error:
            # Deliberately not fatal. The HUD is fully usable by typing, and the
            # server below is what serves it - a machine with no microphone, no
            # speakers, or no espeak installed should get a working text interface
            # and an honest note about what's missing, not a traceback instead of
            # an application.
            print(f"Voice is unavailable, continuing in text-only mode: {voice_error}")
            logger.warning(f"Voice unavailable, text-only: {voice_error}")
        else:
            print("Voice models ready.")

    orchestrator.speak_callback = _make_gui_speak_callback(api, tts)
    orchestrator.visual_callback = _make_gui_visual_callback(api)

    # The interface's core and its RT-LOG read the state machine that already
    # exists, through the listener hook that already exists. Nothing new decides
    # what state Leti is in; this only forwards the transitions the orchestrator
    # makes anyway, which is why "thinking" and "executing" can be shown at all -
    # before this, only the reply and voice paths ever said anything.
    from core import diagnostics

    orchestrator.on_state_change(lambda state: api.push("setHudState", state.value))
    diagnostics.on_activity(lambda entry: api.push("letiActivity", entry))

    # Confirmations go through voice when there IS voice - hands-free is the point
    # of GUI mode, and that's how CLI voice mode behaves. Without a working mic they
    # go to the chat window instead: the alternative was a user sitting in front of
    # a working interface, unable to approve the action they had just asked for.
    from core.confirmation import make_text_confirmation_callback, make_voice_confirmation_callback

    def _logged(callback):
        """The same confirmation callback, with the RT-LOG told it is waiting.

        Wrapped here rather than inside SafetyGuard or core/confirmation.py: being
        asked is a fact about this turn that the interface wants to show, and it is
        not SafetyGuard's business to know an interface exists. The decision, the
        timeout and the fail-closed behaviour are entirely unchanged - this adds a
        log line on the way in and passes the answer straight back out.
        """
        async def _wrapped(prompt: str) -> bool:
            diagnostics.record_activity("approval", "Waiting for your approval")
            answer = await callback(prompt)
            diagnostics.record_activity(
                "approval", "Approved" if answer else "Declined")
            return answer

        return _wrapped

    def _use_voice(tts_engine, stt_engine) -> None:
        """Point the reply and confirmation paths at voice, or back at text.

        Called once at startup and again if first-run setup turns voice on
        mid-session, so both paths switch together - a session that speaks its
        replies but asks for confirmation in text, or the reverse, would be a
        confusing half-state.
        """
        orchestrator.speak_callback = _make_gui_speak_callback(api, tts_engine)
        if tts_engine and stt_engine:
            safety_guard.set_confirmation_callback(
                _logged(make_voice_confirmation_callback(tts_engine, stt_engine)))
        else:
            safety_guard.set_confirmation_callback(
                _logged(make_text_confirmation_callback(api)))

    _use_voice(tts, transcriber)

    server = LetiWebServer(api)
    server_ready = threading.Event()

    async def _listen(stt_engine) -> None:
        api.voice_active = True
        try:
            await _run_voice_loop(orchestrator, api, continuous_voice, stt_engine)
        except Exception:
            # A voice/mic failure (e.g. the audio device disappearing mid-session)
            # shouldn't take the whole app down - the server is already up and fully
            # usable via text regardless of what happens to voice.
            logger.exception("Voice pipeline stopped unexpectedly - text chat is unaffected.")
        finally:
            api.voice_active = False

    async def enable_voice() -> None:
        """Turn voice on in THIS session, after the user accepts the audio prompt.

        Loading Whisper takes seconds of real work, and this runs on the loop the
        websocket server shares - so it goes to a thread, exactly as the startup
        path loads it before the server exists. Doing it inline here is the same
        bug that used to freeze every connected client's chat.
        """
        if api.voice_active:
            return
        new_tts, new_stt, error = await loop.run_in_executor(None, load_voice_stack)
        if error:
            logger.warning(f"Voice couldn't start after audio setup: {error}")
            api.push("appendLetiReply", f"I saved that, but voice couldn't start: {error}")
            return
        _use_voice(new_tts, new_stt)
        api.push("appendLetiReply", "Voice is on - say the wake word whenever you're ready.")
        await _listen(new_stt)

    api.enable_voice = enable_voice

    async def _start_server_and_voice():
        await server.start()
        server_ready.set()
        if transcriber is None:
            return  # text-only for now; enable_voice() can still start it later
        await _listen(transcriber)

    asyncio.run_coroutine_threadsafe(_start_server_and_voice(), loop)
    if not server_ready.wait(timeout=15):
        raise RuntimeError("The Leti web server didn't start within 15s - check the logs above.")

    try:
        import webview
    except ImportError:
        webview = None

    from gui.desktop import display_available

    local_url = f"http://127.0.0.1:{server.port}"

    no_window_reason = None
    if webview is None:
        no_window_reason = "pywebview isn't installed"
    elif not display_available():
        # pywebview is here, but there is no window server to put a window on.
        # Deliberately checked rather than attempted - see display_available().
        no_window_reason = "there's no graphical display (normal over SSH, or on a headless machine)"
        webview = None

    if webview:
        # The desktop application is what this mode is: a real window is attempted
        # first, every time, and the browser is only ever what we fall back TO.
        from gui.desktop import DesktopWindows

        try:
            # Both native windows live here: the full interface, and the frameless
            # always-on-top puck that minimising switches to. See gui/desktop.py.
            api.desktop = DesktopWindows(webview, local_url)
            api.desktop.create_main()
            # Window/taskbar icon. pywebview takes this on its GTK and Qt backends;
            # older versions don't accept the argument at all, and on Windows/macOS
            # the window icon comes from the launcher shortcut or app bundle instead
            # (see scripts/install_windows_launcher.ps1 and install_macos_icon.sh).
            # Falling back to a plain start() keeps a missing icon from being the
            # reason the app won't open.
            icon_path = Path(__file__).parent / "icons" / "leti-512.png"
            try:
                webview.start(icon=str(icon_path))
            except TypeError:
                webview.start()  # blocks until the window is closed
        except Exception as e:
            # Creating the window counts as part of the attempt, so it is inside
            # this: a failure there used to end the process instead of falling
            # back. The server is already up either way, so losing the window is
            # never a reason to lose the application with it.
            api.desktop = None
            logger.warning(f"The desktop window couldn't run ({e}); falling back to the browser.")
            _serve_headless(f"\n(The desktop window couldn't start: {e}.", url=local_url)
    else:
        print(f"\n(No desktop window: {no_window_reason}.")
        _serve_headless(url=local_url)

    asyncio.run_coroutine_threadsafe(server.stop(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
