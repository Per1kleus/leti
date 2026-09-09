"""The two native windows: the full interface, and the always-on-top puck.

Minimising inside the page can only ever shrink things *within* the browser
window - the page cannot float over other applications, and it cannot move a
window it does not own. Doing what "stays on top and can be dragged anywhere"
actually means takes a second native window, which is what this module owns:

    full  - the ordinary application window, 1180x760, framed, in the taskbar
    puck  - 190x190, frameless, always-on-top, dragged by its own surface

Both load the SAME page from the same local server; the puck adds `?puck=1`, and
gui/hud.html renders itself collapsed when it sees that. So there is one
interface, one renderer and one websocket session per window - the puck is not a
second, smaller app that has to be kept in step with the first.

Only one is visible at a time. Swapping is a hide and a show rather than a
create and a destroy, so the puck keeps its socket, its place on screen and its
unread count for as long as the app runs.

Everything here is best-effort by design. pywebview's window methods vary by
platform and backend - transparency in particular is unsupported on Windows -
and a window control that fails is not a reason to take down an assistant that
is otherwise working. Failures are logged and reported back to the page, which
falls back to collapsing inside its own window.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("leti.gui.desktop")

PUCK_SIZE = 190
# Where the puck first appears, as an inset from the bottom-right of the screen.
# Bottom-right because that is where system trays and notifications live, so it is
# the corner people already expect small persistent things to occupy.
PUCK_MARGIN = 40


def display_available() -> bool:
    """Whether this machine can open a window at all.

    Checked BEFORE creating one, because on Linux there is no second chance: GTK
    prints "cannot open display" and takes the process down with it rather than
    raising something Python can catch, so an app that only tried and handled the
    failure would simply die - taking the web server, and with it the interface
    every other device was using, along with it.

    Windows and macOS always have a window server when there is a user session, so
    the question only arises on Linux, where "no DISPLAY and no WAYLAND_DISPLAY"
    is exactly the SSH session, the headless server and the systemd unit.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class DesktopWindows:
    """Owns the native windows. One instance per run; None when there is no GUI."""

    def __init__(self, webview_module: Any, base_url: str):
        self._webview = webview_module
        self._base_url = base_url.rstrip("/")
        self._lock = threading.Lock()
        self.main = None
        self.puck = None

    # ---- creation ------------------------------------------------------------

    def create_main(self, icon_path: Optional[Path] = None):
        self.main = self._webview.create_window(
            "Leti", f"{self._base_url}/", width=1180, height=760,
            background_color="#050b14",
        )
        return self.main

    def _create_puck(self):
        """The puck window. Made once, then hidden and shown.

        transparent lets the round puck sit on the desktop without a square of
        background around it. It is honoured on GTK and Cocoa and ignored on
        Windows, so the page keeps its own opaque circular backing and looks
        deliberate either way rather than relying on it.

        easy_drag is off: with it on, pywebview moves the window on any drag
        anywhere, which swallows the click that reopens the interface. Dragging is
        scoped to a region the page marks instead (see `pywebview-drag-region` in
        gui/hud.html).
        """
        options = dict(
            width=PUCK_SIZE, height=PUCK_SIZE,
            frameless=True, on_top=True, easy_drag=False, resizable=False,
            background_color="#050b14",
        )
        try:
            window = self._webview.create_window(
                "Leti", f"{self._base_url}/?puck=1", transparent=True, **options)
        except TypeError:
            # An older pywebview without `transparent`. The page's own backing
            # makes this a cosmetic difference, not a broken window.
            logger.info("This pywebview has no transparent window support; using an opaque puck.")
            window = self._webview.create_window("Leti", f"{self._base_url}/?puck=1", **options)

        self._place_bottom_right(window)
        return window

    def _place_bottom_right(self, window) -> None:
        """Open in the bottom-right corner of the screen, if we can find out where
        that is. Screen enumeration differs between pywebview versions, so a
        failure here just leaves the window wherever the toolkit put it."""
        try:
            screens = self._webview.screens
            if not screens:
                return
            screen = screens[0]
            window.move(screen.width - PUCK_SIZE - PUCK_MARGIN,
                        screen.height - PUCK_SIZE - PUCK_MARGIN)
        except Exception as e:
            logger.debug(f"Couldn't place the puck in the corner ({e}); leaving it where it opened.")

    # ---- switching -----------------------------------------------------------

    def show_puck(self) -> bool:
        """Shrink to the always-on-top puck. True if the native swap happened."""
        with self._lock:
            try:
                if self.puck is None:
                    self.puck = self._create_puck()
                else:
                    self.puck.show()
                    # Re-assert on_top: a window manager can drop the hint while a
                    # window is hidden, and a puck that other windows cover is not
                    # doing the one thing it exists to do.
                    try:
                        self.puck.on_top = True
                    except Exception:
                        pass
                if self.main is not None:
                    self.main.hide()
                return True
            except Exception:
                logger.exception("Couldn't switch to the puck window.")
                return False

    def show_full(self) -> bool:
        """Back to the full interface."""
        with self._lock:
            try:
                if self.main is not None:
                    self.main.show()
                if self.puck is not None:
                    self.puck.hide()
                return True
            except Exception:
                logger.exception("Couldn't restore the main window.")
                return False

    def destroy(self) -> None:
        for window in (self.puck, self.main):
            try:
                window and window.destroy()
            except Exception:
                pass
