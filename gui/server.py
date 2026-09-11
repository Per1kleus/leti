"""
Serves the HUD (gui/hud.html) over HTTP + WebSocket so it can be opened
from ANY device on the local network, not just the desktop pywebview
window - which is now just one more client of this same server rather
than a separate bridge. This is what makes Leti launchable from a phone:
open the printed URL in a mobile browser, then "Add to Home Screen" for
an app-like icon (a manifest + minimal service worker are served below
specifically so that install prompt is available on Android Chrome).

Security model - worth reading before enabling this on a shared network:
  - Loopback (127.0.0.1/::1) is trusted implicitly. That's the desktop
    pywebview window and anyone with a shell on this exact machine - no
    different from any other localhost-bound dev server.
  - Every other connection (a phone, another computer on the LAN) must
    present a token, generated once into data/gui_remote_token.txt and
    printed to the console on startup.
  - This is LAN-trust-level security, comparable to a home router's admin
    page - it is NOT internet-facing. Never port-forward this. Set
    gui.enable_remote_access: false in settings.yaml to disable non-loopback
    access entirely and restrict Leti's GUI to this machine only.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
from pathlib import Path
from typing import Any, Dict, Optional, Set

from aiohttp import web, WSMsgType

from core.atomic_write import atomic_write_text
from core.config_loader import get_settings, resolve_path

logger = logging.getLogger("leti.gui.server")

GUI_DIR = Path(__file__).parent
DEFAULT_TOKEN_PATH = "./data/gui_remote_token.txt"
LOOPBACK_ADDRESSES = {"127.0.0.1", "::1"}


def _get_or_create_token() -> str:
    cfg = get_settings().get("gui", {})
    path = resolve_path(cfg.get("token_file_path", DEFAULT_TOKEN_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text().strip()
        if existing:
            return existing
    token = secrets.token_hex(16)
    # Owner-only: this is the shared secret that grants a LAN device a full session.
    atomic_write_text(path, token, secret=True)
    return token


def _lan_ip() -> str:
    """Best-effort local network IP (not loopback), so the printed URL is one a
    phone on the same Wi-Fi can actually reach. Doesn't send any real traffic -
    just asks the OS's routing table what interface would be used."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def is_authenticated(peer_ip: str, provided_token: Optional[str], configured_token: str, allow_remote: bool) -> bool:
    """Pure decision function (kept separate from the handler so it's directly
    testable without spinning up a real socket): loopback is always trusted;
    anything else needs allow_remote AND a matching token."""
    if peer_ip in LOOPBACK_ADDRESSES:
        return True
    if not allow_remote:
        return False
    return provided_token == configured_token


class LetiWebServer:
    def __init__(self, api, host: Optional[str] = None, port: Optional[int] = None):
        settings = get_settings().get("gui", {})
        self.api = api
        self.port = port or settings.get("port", 8420)
        self.allow_remote = settings.get("enable_remote_access", True)
        # enable_remote_access: false used to gate only the websocket handshake while
        # the server still bound 0.0.0.0 - so the HUD, the manifest and an open port
        # were on the LAN anyway. "Restrict Leti's GUI to this machine" now means not
        # listening anywhere else in the first place.
        self.host = host if host is not None else ("0.0.0.0" if self.allow_remote else "127.0.0.1")
        self.token = _get_or_create_token()

        self.app = web.Application()
        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_get("/manifest.json", self._handle_manifest)
        self.app.router.add_get("/sw.js", self._handle_service_worker)
        self.app.router.add_get("/icon.svg", self._handle_icon)
        # Raster icons for the PWA install prompt and the browser tab. Android
        # Chrome wants a real PNG for the home-screen icon; an SVG alone is
        # accepted inconsistently across versions.
        icons_dir = GUI_DIR / "icons"
        if icons_dir.is_dir():
            self.app.router.add_static("/icons/", path=str(icons_dir), name="icons")
        # Optional local copies of third-party assets, so the HUD can be made to
        # work with no internet at all - see the mermaid comment in hud.html.
        # Only created if the user actually vendors something.
        vendor_dir = GUI_DIR / "vendor"
        if vendor_dir.is_dir():
            self.app.router.add_static("/vendor/", path=str(vendor_dir), name="vendor")
        self.app.router.add_get("/ws", self._handle_ws)
        self.runner: Optional[web.AppRunner] = None
        # asyncio only holds a weak reference to a running task, so a bare
        # create_task() can be garbage-collected mid-flight. Keep them alive until
        # they finish (see _dispatch_call).
        self._background_tasks: Set[asyncio.Task] = set()

    async def _handle_index(self, request: web.Request) -> web.Response:
        html = (GUI_DIR / "hud.html").read_text()
        return web.Response(text=html, content_type="text/html")

    async def _handle_manifest(self, request: web.Request) -> web.Response:
        manifest = {
            "name": "Leti", "short_name": "Leti", "start_url": "/", "display": "standalone",
            "background_color": "#050b14", "theme_color": "#050b14",
            "icons": [
                {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"},
                {"src": "/icons/leti-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
                {"src": "/icons/leti-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
                # Maskable: Android crops the icon to the launcher's shape, so this
                # entry promises the mark stays inside the safe zone when it does.
                {"src": "/icons/leti-maskable-512.png", "sizes": "512x512", "type": "image/png",
                 "purpose": "maskable"},
            ],
        }
        return web.json_response(manifest)

    async def _handle_service_worker(self, request: web.Request) -> web.Response:
        # Deliberately minimal - just enough to satisfy "installable" criteria on
        # Android Chrome. No real offline caching: Leti needs a live connection to
        # the backend regardless, so there's nothing meaningful to serve offline.
        return web.Response(text="self.addEventListener('fetch', function(){});", content_type="application/javascript")

    async def _handle_icon(self, request: web.Request) -> web.Response:
        svg_path = GUI_DIR / "icon.svg"
        svg = svg_path.read_text() if svg_path.exists() else (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<rect width="100" height="100" fill="#050b14"/>'
            '<circle cx="50" cy="50" r="32" fill="none" stroke="#4fe3ff" stroke-width="6"/>'
            '</svg>'
        )
        return web.Response(text=svg, content_type="image/svg+xml")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        peer_ip = request.remote or ""
        authenticated = peer_ip in LOOPBACK_ADDRESSES

        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")

            if msg_type == "auth":
                if is_authenticated(peer_ip, data.get("token"), self.token, self.allow_remote):
                    authenticated = True
                    self.api.ws_clients.add(ws)
                    await ws.send_str(json.dumps({"type": "auth_ok"}))
                else:
                    await ws.send_str(json.dumps({"type": "error", "message": "Not authorized. Check the token in your URL."}))
                    await ws.close()
                continue

            if not authenticated:
                await ws.send_str(json.dumps({"type": "error", "message": "Not authenticated."}))
                continue

            if msg_type == "call":
                await self._dispatch_call(ws, data)

        self.api.ws_clients.discard(ws)
        return ws

    async def _dispatch_call(self, ws: web.WebSocketResponse, data: Dict[str, Any]) -> None:
        from gui.api import ASYNC_METHODS, SYNC_METHODS

        call_id = data.get("id")
        method = data.get("method", "")
        args = data.get("args", [])
        try:
            if method == "send_text_message":
                # Answering a turn takes as long as it takes; the websocket must stay
                # responsive meanwhile, so this is deliberately not awaited. The reply
                # reaches every client through api.push() when it's ready.
                task = asyncio.create_task(self.api.a_send_text_message(*args))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                value: Any = True
            elif method == "get_system_stats":
                value = await self.api.a_get_system_stats()
            elif method == "get_weather":
                value = await self.api.a_get_weather()
            elif method == "check_audio":
                value = await self.api.a_check_audio(*args)
            elif method == "save_audio_setup":
                value = await self.api.a_save_audio_setup(*args)
            elif method == "get_permissions":
                value = await self.api.a_get_permissions()
            elif method == "set_permission":
                value = await self.api.a_set_permission(*args)
            elif method == "get_diagnostics":
                value = await self.api.a_get_diagnostics()
            elif method == "get_model_setup":
                value = await self.api.a_get_model_setup()
            elif method == "apply_model_setup":
                # Downloading a model takes as long as it takes; the card waits on
                # this call rather than polling, and localhost has nothing in
                # between to time it out.
                value = await self.api.a_apply_model_setup(*args)
            elif method in SYNC_METHODS:
                value = getattr(self.api, method)(*args)
            else:
                await ws.send_str(json.dumps({"type": "error", "id": call_id, "message": f"Unknown method: {method}"}))
                return
            await ws.send_str(json.dumps({"type": "result", "id": call_id, "value": value}))
        except Exception as e:
            logger.exception(f"Error handling call to '{method}'")
            await ws.send_str(json.dumps({"type": "error", "id": call_id, "message": str(e)}))

    async def start(self) -> None:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        try:
            site = web.TCPSite(self.runner, self.host, self.port)
            await site.start()
        except OSError as e:
            raise RuntimeError(
                f"Couldn't start Leti's web server on port {self.port} ({e}). Another program "
                f"may already be using it - set a different gui.port in config/settings.yaml."
            )

        print(f"\nLeti's interface is running.")
        print(f"  On this computer: http://127.0.0.1:{self.port}/")
        if not self.allow_remote:
            print(f"  (Listening on 127.0.0.1 only - nothing else on the network can reach it.)")
        if self.allow_remote:
            print(f"  From your phone (same Wi-Fi): http://{_lan_ip()}:{self.port}/?token={self.token}")
            print(f"  Open that in your phone's browser, then use its menu to 'Add to Home Screen'")
            print(f"  for an app-like icon. This is LAN-only - never port-forward it to the internet.\n")
        else:
            print(f"  Remote access is disabled (gui.enable_remote_access: false in settings.yaml).\n")

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()
