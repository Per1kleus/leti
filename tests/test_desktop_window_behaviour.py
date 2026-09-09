"""How Leti behaves as a desktop application: one process, two windows, and a
minimise that shrinks rather than disappears.

The window toolkit itself cannot be exercised here - there is no window manager
in this environment, so nothing can actually iconify a window. What these test is
the part that is Leti's rather than GTK's: that the titlebar's minimise button is
wired to the same swap the page's own control performs, that coming back from it
un-iconifies the window it hid, that the swap is a hide and a show rather than a
second process, and that closing closes everything.
"""
from __future__ import annotations

import sys
import types

import pytest

from gui.desktop import APP_ID, DesktopWindows, claim_application_identity


class FakeEvent:
    """pywebview's Event: handlers added with +=, called on fire."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def fire(self, *args):
        for handler in list(self.handlers):
            handler(*args)


class FakeWindow:
    def __init__(self, title, url, **options):
        self.title, self.url, self.options = title, url, options
        self.events = types.SimpleNamespace(minimized=FakeEvent(), closed=FakeEvent())
        self.log = []
        self.on_top = options.get("on_top", False)
        self.destroyed = False

    def show(self):    self.log.append("show")
    def hide(self):    self.log.append("hide")
    def restore(self): self.log.append("restore")
    def minimize(self): self.log.append("minimize")
    def move(self, x, y): self.log.append(("move", x, y))
    def destroy(self): self.destroyed = True; self.log.append("destroy")


class FakeWebview:
    """Stands in for the pywebview module. Records every window ever created."""

    def __init__(self):
        self.created = []
        self.screens = [types.SimpleNamespace(width=1920, height=1080)]

    def create_window(self, title, url, **options):
        window = FakeWindow(title, url, **options)
        self.created.append(window)
        return window


@pytest.fixture
def desktop():
    webview = FakeWebview()
    windows = DesktopWindows(webview, "http://127.0.0.1:8420")
    windows.create_main()
    return webview, windows


# --- The window the operating system sees ------------------------------------------

def test_the_main_window_is_an_ordinary_framed_resizable_window(desktop):
    """Minimise, maximise, restore, move and resize are the window manager's to
    provide, and it only provides them to a window that asked to be ordinary."""
    webview, _ = desktop
    main = webview.created[0]

    assert main.options.get("frameless") in (None, False)
    assert main.options.get("on_top") in (None, False)
    assert main.options.get("resizable") in (None, True)
    assert main.title == "Leti"


def test_windows_is_told_this_process_is_leti_not_python(monkeypatch):
    """Without an explicit AppUserModelID, Windows files the taskbar button under
    python.exe - wrong icon, wrong group, and pinning pins Python."""
    calls = []

    class FakeShell32:
        def SetCurrentProcessExplicitAppUserModelID(self, app_id):
            calls.append(app_id)

    monkeypatch.setattr(sys, "platform", "win32")
    fake_ctypes = types.ModuleType("ctypes")
    fake_ctypes.windll = types.SimpleNamespace(shell32=FakeShell32())
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    assert claim_application_identity() is True
    assert calls == [APP_ID]


def test_a_locked_down_windows_still_opens_the_window(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    broken = types.ModuleType("ctypes")

    class Boom:
        def SetCurrentProcessExplicitAppUserModelID(self, _):
            raise OSError("access denied")

    broken.windll = types.SimpleNamespace(shell32=Boom())
    monkeypatch.setitem(sys.modules, "ctypes", broken)

    assert claim_application_identity() is False      # reported, never raised


def test_the_identity_is_claimed_before_the_window_exists(monkeypatch):
    """Windows reads the id when the window is created; setting it afterwards is
    too late to affect the taskbar button."""
    import inspect

    import gui.desktop

    source = inspect.getsource(gui.desktop.DesktopWindows.create_main)
    assert source.index("claim_application_identity") < source.index("create_window")


# --- Minimise becomes the floating circle ------------------------------------------

def test_the_titlebar_minimise_button_shrinks_to_the_puck(desktop):
    """The whole point: the two minimises must mean the same thing. Using the
    titlebar button used to drop Leti into the taskbar while the page's own
    control turned it into the puck."""
    webview, windows = desktop
    main = webview.created[0]

    main.events.minimized.fire()

    assert len(webview.created) == 2, "a puck window was not created"
    puck = webview.created[1]
    assert "hide" in main.log, "the full window is still on screen"
    assert puck.options["frameless"] and puck.options["on_top"]
    assert "puck=1" in puck.url


def test_the_puck_is_the_same_application_not_a_second_one(desktop):
    """Two windows, one process, one server, one session - the puck loads the
    same page from the same URL rather than starting anything."""
    webview, windows = desktop
    windows.show_puck()
    windows.show_full()
    windows.show_puck()

    assert len(webview.created) == 2, "switching created extra windows"
    assert webview.created[1].url.startswith(webview.created[0].url.rstrip("/"))


def test_going_back_and_forth_reuses_the_same_two_windows(desktop):
    webview, windows = desktop
    main, _ = webview.created[0], windows.show_puck()
    puck = webview.created[1]

    for _ in range(3):
        windows.show_full()
        windows.show_puck()

    assert len(webview.created) == 2
    assert not main.destroyed and not puck.destroyed, "a window was destroyed and remade"


def test_restoring_un_iconifies_a_window_that_was_minimised(desktop):
    """Reached through the titlebar button the window was iconified BEFORE it was
    hidden, and showing it again brings it back still iconified - a click on the
    puck that appears to do nothing."""
    webview, windows = desktop
    main = webview.created[0]
    main.events.minimized.fire()
    main.log.clear()

    windows.show_full()

    assert "show" in main.log and "restore" in main.log
    assert main.log.index("show") < main.log.index("restore")


def test_the_puck_is_hidden_again_when_the_full_window_returns(desktop):
    webview, windows = desktop
    windows.show_puck()
    puck = webview.created[1]
    puck.log.clear()

    windows.show_full()

    assert "hide" in puck.log


def test_the_puck_reasserts_always_on_top_every_time_it_returns(desktop):
    """A window manager can drop the hint while a window is hidden, and a puck
    other windows cover is not doing the one thing it exists for."""
    webview, windows = desktop
    windows.show_puck()
    puck = webview.created[1]
    windows.show_full()
    puck.on_top = False                      # as a window manager might leave it

    windows.show_puck()

    assert puck.on_top is True


def test_a_failing_window_toolkit_reports_rather_than_raises(desktop):
    """A window control that fails is not a reason to take down an assistant that
    is otherwise working."""
    webview, windows = desktop
    windows.show_puck()

    def boom():
        raise RuntimeError("the toolkit said no")

    webview.created[0].show = boom
    assert windows.show_full() is False      # reported, not raised


def test_a_native_minimise_that_fails_does_not_escape_into_the_toolkit(desktop):
    """This handler is called from the toolkit's own thread, where an exception
    has nowhere useful to go."""
    webview, windows = desktop
    windows.show_puck = lambda: (_ for _ in ()).throw(RuntimeError("no"))

    webview.created[0].events.minimized.fire()   # must not raise


def test_an_older_pywebview_without_the_event_still_opens(monkeypatch):
    """No minimised event means the titlebar button keeps doing its ordinary
    thing. A lesser interface, not a broken one."""
    webview = FakeWebview()
    original = webview.create_window

    def no_events(title, url, **options):
        window = original(title, url, **options)
        window.events = types.SimpleNamespace()      # no .minimized at all
        return window

    webview.create_window = no_events
    windows = DesktopWindows(webview, "http://127.0.0.1:8420")

    assert windows.create_main() is not None


# --- Closing -----------------------------------------------------------------------

def test_closing_the_main_window_takes_the_puck_with_it(desktop):
    """webview.start() returns when the LAST window closes, and a hidden puck is
    still a window. Measured before this was wired: the main window shut and the
    process was still running six seconds later with nothing on screen, holding the
    port, the loaded model and the microphone. Closing has to actually close."""
    webview, windows = desktop
    main = webview.created[0]
    windows.show_puck()
    puck = webview.created[1]
    assert not puck.destroyed

    main.events.closed.fire()

    assert puck.destroyed, "the floating circle outlived the application"
    assert windows.puck is None


def test_closing_with_no_puck_ever_opened_is_uneventful(desktop):
    webview, windows = desktop

    webview.created[0].events.closed.fire()   # must not raise

    assert len(webview.created) == 1, "closing created a window"


def test_a_puck_that_refuses_to_close_is_reported_not_raised(desktop):
    """Raised from the toolkit's own thread, an exception here has nowhere to go."""
    webview, windows = desktop
    windows.show_puck()

    def boom():
        raise RuntimeError("the toolkit said no")

    webview.created[1].destroy = boom
    webview.created[0].events.closed.fire()   # must not raise


def test_closing_destroys_both_windows(desktop):
    """The floating circle must not outlive the application that owns it."""
    webview, windows = desktop
    windows.show_puck()

    windows.destroy()

    assert all(w.destroyed for w in webview.created), "a window survived the close"
