"""What to click, described well enough to be sure of it.

Leti's Computer Use has matched targets by reading text off a screenshot. That
works and it is the wrong thing to rely on: a screenshot says "the word Settings
appears somewhere", not "there is an enabled button named Settings in this
window". The difference matters when two things are called Settings, when one of
them is greyed out, when the word is in a tooltip, or when the window scrolled
between looking and clicking.

So this module asks the operating system instead, when the operating system can
be asked, and falls back to what already worked when it cannot. The order is
always the same and is never skipped:

    native accessibility / UI automation
    structured application information
    visible text plus window context
    screenshot / OCR text
    coordinates

WHAT THIS IS NOT. Not a second Computer Use system: core/computer_use.py still
owns the session, the step budget, the mismatch counter and the decision to act.
This answers one question for it - "which element is that, and can I be sure?" -
and returns a verdict, never an action. Nothing here clicks, types, captures a
screen or authorises anything.

NOTHING RUNS WHEN NOTHING IS HAPPENING. There is no accessibility daemon, no
tree cache that outlives a session, no desktop scan and no polling. A provider is
imported the first time a resolution actually needs it and the import result is
remembered so the failure is not paid for twice. A query walks one window's
descendants, filtered by what is being looked for, and throws the result away.

AND THE TREE NEVER REACHES THE MODEL. A real window has hundreds of elements.
What leaves this module is at most a handful of candidates, each a few fields
wide - which is the whole point of resolving locally instead of serialising a
tree into a prompt and asking.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("leti.ui")

# --------------------------------------------------------------------------- #
# What came back
# --------------------------------------------------------------------------- #

UNIQUE_MATCH = "UNIQUE_MATCH"     # exactly one element, and it is safe to act on
AMBIGUOUS = "AMBIGUOUS"           # several equally good; the user has to choose
PARTIAL_MATCH = "PARTIAL_MATCH"   # something close, not close enough to click
NOT_FOUND = "NOT_FOUND"           # nothing like it is there
DISABLED = "DISABLED"             # it is there and cannot be used
STALE = "STALE"                   # it was there; it is not the same thing now
UNSUPPORTED = "UNSUPPORTED"       # this machine cannot answer the question

STATES = (UNIQUE_MATCH, AMBIGUOUS, PARTIAL_MATCH, NOT_FOUND, DISABLED, STALE,
          UNSUPPORTED)

# States where acting is allowed. Deliberately one.
ACTIONABLE = (UNIQUE_MATCH,)

# How a target was identified, strongest first. Reported so a resolution that had
# to fall back says so rather than looking like one that did not.
BY_ACCESSIBILITY = "accessibility"      # the OS named it
BY_STRUCTURE = "structure"              # the application named it
BY_TEXT = "visible text"                # it was on screen, in a window we know
BY_OCR = "screen text"                  # it was in the screenshot
BY_COORDINATES = "coordinates"          # nothing named it at all

METHODS = (BY_ACCESSIBILITY, BY_STRUCTURE, BY_TEXT, BY_OCR, BY_COORDINATES)
_METHOD_RANK = {name: i for i, name in enumerate(METHODS)}

# How long a resolved element may be held before it has to be resolved again.
# Short: the cost of asking twice is one query, and the cost of acting on a
# stale element is a click in the wrong place.
TARGET_MAX_AGE_SECONDS = 20

# How many candidates ever leave this module. A person choosing between more
# than this is not choosing, and the model does not need the rest.
MAX_CANDIDATES = 5

# How many elements one query may walk before it gives up. A real window has
# hundreds; a runaway tree has hundreds of thousands, and walking one of those
# would be the freeze this module exists to avoid.
MAX_ELEMENTS_SCANNED = 2000


@dataclass
class UITarget:
    """One element, normalised across whatever named it.

    Every field is optional except `name` and `how`, because no two backends
    supply the same set and a field nobody filled in has to read as unknown
    rather than as absent.
    """
    name: str
    how: str                                  # one of METHODS
    role: Optional[str] = None                # button, tab, menu item, edit...
    automation_id: Optional[str] = None       # AutomationId / accessibility id
    class_name: Optional[str] = None
    application: Optional[str] = None
    window: Optional[str] = None
    enabled: Optional[bool] = None
    visible: Optional[bool] = None
    rect: Optional[Tuple[int, int, int, int]] = None   # left, top, right, bottom
    parent: Optional[str] = None
    handle: Optional[str] = None              # whatever the backend can re-find it by
    observed_at: float = field(default_factory=time.time)

    def identity(self) -> Tuple:
        """What has to stay the same for this to be the same element.

        Used to detect a target going stale between resolving it and acting on
        it. Position is deliberately NOT part of it: a window that moved still
        holds the same button, and a button whose name or automation id changed
        is a different button however little it moved.
        """
        return (self.application, self.window, self.role, self.name,
                self.automation_id, self.class_name)

    def age(self, now: Optional[float] = None) -> float:
        return (now if now is not None else time.time()) - self.observed_at

    def describe(self) -> Dict[str, Any]:
        """The few fields that go to the model. Never the whole element."""
        out: Dict[str, Any] = {"name": self.name, "how": self.how}
        for key in ("role", "application", "window", "automation_id", "enabled",
                    "visible", "parent"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass
class Resolution:
    """The answer, and everything needed to argue with it."""
    state: str
    target: Optional[UITarget] = None
    candidates: List[UITarget] = field(default_factory=list)
    method: Optional[str] = None
    detail: str = ""
    wanted: str = ""

    @property
    def may_act(self) -> bool:
        return self.state in ACTIONABLE and self.target is not None

    def describe(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "wanted": self.wanted,
            "method": self.method,
            "detail": self.detail,
            "target": self.target.describe() if self.target else None,
            "candidates": [c.describe() for c in self.candidates[:MAX_CANDIDATES]],
            "may_act": self.may_act,
        }


# --------------------------------------------------------------------------- #
# Providers
#
# A provider answers two questions: which window is in front, and which of its
# descendants look like what is being asked for. Each is imported the first time
# it is needed and never before; a machine with no accessibility stack pays one
# failed import for the whole session and then nothing.
# --------------------------------------------------------------------------- #

class Provider:
    """The interface. A provider that cannot work says so and is skipped."""

    name = "none"
    method = BY_ACCESSIBILITY

    def available(self) -> bool:
        raise NotImplementedError

    def active_window(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def find(self, wanted: str, role: str = "",
             window: str = "") -> List[UITarget]:
        raise NotImplementedError

    def still_there(self, target: UITarget) -> Optional[UITarget]:
        """Re-resolve one element. None means it is gone."""
        matches = [t for t in self.find(target.name, target.role or "",
                                        target.window or "")
                   if t.identity() == target.identity()]
        return matches[0] if len(matches) == 1 else None


class _LazyImport:
    """One import attempt, remembered. Never retried in a loop."""

    def __init__(self, *modules: str):
        self.modules = modules
        self._loaded: Any = None
        self._tried = False

    def get(self) -> Any:
        if not self._tried:
            self._tried = True
            for module in self.modules:
                try:
                    self._loaded = __import__(module, fromlist=["*"])
                    break
                except Exception as e:
                    logger.debug(f"UI provider import {module} failed: {e}")
        return self._loaded


class WindowsProvider(Provider):
    """Windows UI Automation, through whichever binding is installed.

    uiautomation and pywinauto both wrap the same OS API; neither is a
    dependency of Leti and both are used only if the user already has one. The
    walk is depth-limited and count-limited because UIA will happily hand back a
    tree with a hundred thousand nodes in it.
    """

    name = "windows-uia"
    method = BY_ACCESSIBILITY

    def __init__(self):
        self._uia = _LazyImport("uiautomation")
        self._pywinauto = _LazyImport("pywinauto")

    def available(self) -> bool:
        if not sys.platform.startswith("win"):
            return False
        return bool(self._uia.get() or self._pywinauto.get())

    def active_window(self) -> Optional[Dict[str, Any]]:
        uia = self._uia.get()
        if uia is None:
            return None
        try:
            window = uia.GetForegroundControl().GetTopLevelControl()
            if window is None:
                return None
            return {"window": window.Name, "application": window.ClassName,
                    "handle": str(getattr(window, "NativeWindowHandle", "") or "")}
        except Exception as e:
            logger.debug(f"Windows active window failed: {e}")
            return None

    def find(self, wanted: str, role: str = "", window: str = "") -> List[UITarget]:
        uia = self._uia.get()
        if uia is None:
            return []
        try:
            root = uia.GetForegroundControl().GetTopLevelControl()
            if root is None:
                return []
            return self._walk(root, wanted, role)
        except Exception as e:
            logger.debug(f"Windows UI query failed: {e}")
            return []

    def _walk(self, root: Any, wanted: str, role: str) -> List[UITarget]:
        found: List[UITarget] = []
        scanned = 0
        stack = [(root, None)]
        window_name = getattr(root, "Name", "") or ""
        application = getattr(root, "ClassName", "") or ""
        while stack and scanned < MAX_ELEMENTS_SCANNED:
            element, parent = stack.pop()
            scanned += 1
            try:
                children = element.GetChildren()
            except Exception:
                children = []
            stack.extend((child, element) for child in children)
            try:
                name = (element.Name or "").strip()
            except Exception:
                continue
            if not name or not _could_be(name, wanted):
                continue
            try:
                rect = element.BoundingRectangle
                box = (rect.left, rect.top, rect.right, rect.bottom)
            except Exception:
                box = None
            found.append(UITarget(
                name=name,
                how=self.method,
                role=_normalise_role(getattr(element, "ControlTypeName", "") or ""),
                automation_id=(getattr(element, "AutomationId", "") or "") or None,
                class_name=(getattr(element, "ClassName", "") or "") or None,
                application=application or None,
                window=window_name or None,
                enabled=bool(getattr(element, "IsEnabled", True)),
                visible=not bool(getattr(element, "IsOffscreen", False)),
                rect=box,
                parent=(getattr(parent, "Name", None) if parent is not None else None),
                handle=str(getattr(element, "NativeWindowHandle", "") or "") or None,
            ))
        return found


class MacProvider(Provider):
    """macOS accessibility, if the user has the PyObjC bridge and has granted
    Leti accessibility permission. Absent either, this reports unavailable and
    the text path is used."""

    name = "macos-ax"
    method = BY_ACCESSIBILITY

    def __init__(self):
        self._appkit = _LazyImport("AppKit")

    def available(self) -> bool:
        return sys.platform == "darwin" and self._appkit.get() is not None

    def active_window(self) -> Optional[Dict[str, Any]]:
        appkit = self._appkit.get()
        if appkit is None:
            return None
        try:
            app = appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
            return {"application": str(app.localizedName()), "window": None,
                    "handle": str(app.processIdentifier())}
        except Exception as e:
            logger.debug(f"macOS active application failed: {e}")
            return None

    def find(self, wanted: str, role: str = "", window: str = "") -> List[UITarget]:
        # Enumerating elements needs ApplicationServices and an accessibility
        # grant. Without a real grant this returns nothing rather than pretending,
        # and the caller falls through to the text path.
        return []


class LinuxProvider(Provider):
    """AT-SPI, which exists on a desktop Linux session with assistive
    technologies enabled and nowhere else. Headless and most server installs
    report unavailable, which is correct."""

    name = "linux-atspi"
    method = BY_ACCESSIBILITY

    def __init__(self):
        self._atspi = _LazyImport("pyatspi")

    def available(self) -> bool:
        return sys.platform.startswith("linux") and self._atspi.get() is not None

    def active_window(self) -> Optional[Dict[str, Any]]:
        atspi = self._atspi.get()
        if atspi is None:
            return None
        try:
            for app in atspi.Registry.getDesktop(0):
                for window in app:
                    states = window.getState()
                    if states.contains(atspi.STATE_ACTIVE):
                        return {"application": app.name, "window": window.name,
                                "handle": None}
        except Exception as e:
            logger.debug(f"AT-SPI active window failed: {e}")
        return None

    def find(self, wanted: str, role: str = "", window: str = "") -> List[UITarget]:
        atspi = self._atspi.get()
        if atspi is None:
            return []
        found: List[UITarget] = []
        scanned = 0
        try:
            for app in atspi.Registry.getDesktop(0):
                for top in app:
                    states = top.getState()
                    if not states.contains(atspi.STATE_ACTIVE):
                        continue
                    stack = [(top, None)]
                    while stack and scanned < MAX_ELEMENTS_SCANNED:
                        element, parent = stack.pop()
                        scanned += 1
                        try:
                            stack.extend((child, element) for child in element)
                        except Exception:
                            pass
                        name = (getattr(element, "name", "") or "").strip()
                        if not name or not _could_be(name, wanted):
                            continue
                        element_states = element.getState()
                        found.append(UITarget(
                            name=name,
                            how=self.method,
                            role=_normalise_role(element.getRoleName()),
                            application=app.name,
                            window=top.name,
                            enabled=element_states.contains(atspi.STATE_ENABLED),
                            visible=element_states.contains(atspi.STATE_VISIBLE),
                            parent=(getattr(parent, "name", None)
                                    if parent is not None else None),
                        ))
        except Exception as e:
            logger.debug(f"AT-SPI query failed: {e}")
        return found


class WindowListProvider(Provider):
    """Not accessibility - just which windows exist, from the same pygetwindow
    the existing tools use. It cannot find a button, but it CAN say which
    application is in front, which is what turns "the word Settings is on
    screen" into "the word Settings is in the Settings window"."""

    name = "window-list"
    method = BY_STRUCTURE

    def available(self) -> bool:
        try:
            from tools.os_control import _matching_windows

            _, problem = _matching_windows("")
            return problem is None
        except Exception:
            return False

    def active_window(self) -> Optional[Dict[str, Any]]:
        try:
            import pygetwindow as gw

            window = gw.getActiveWindow()
            if window is None:
                return None
            return {"window": window.title, "application": None, "handle": None}
        except Exception as e:
            logger.debug(f"Active window lookup failed: {e}")
            return None

    def find(self, wanted: str, role: str = "", window: str = "") -> List[UITarget]:
        """Windows whose title matches. A window IS a target for focus_window."""
        try:
            from tools.os_control import _matching_windows

            windows, problem = _matching_windows(wanted)
            if problem:
                return []
            return [UITarget(name=w.title, how=self.method, role="window",
                             window=w.title, visible=True, enabled=True)
                    for w in windows]
        except Exception as e:
            logger.debug(f"Window match failed: {e}")
            return []


# Built once, held as instances rather than classes so each keeps its own
# remembered import result. Nothing in a constructor touches the OS.
_PROVIDERS: List[Provider] = []


def providers() -> List[Provider]:
    global _PROVIDERS
    if not _PROVIDERS:
        _PROVIDERS = [WindowsProvider(), MacProvider(), LinuxProvider(),
                      WindowListProvider()]
    return _PROVIDERS


def reset_providers(replacement: Optional[Sequence[Provider]] = None) -> None:
    """Only the tests and a platform change call this."""
    global _PROVIDERS
    _PROVIDERS = list(replacement) if replacement is not None else []


def available_provider() -> Optional[Provider]:
    """The strongest provider this machine can actually use, or None.

    Asked on demand. Each provider's availability check is its own remembered
    import, so a machine with no accessibility stack answers None in microseconds
    after the first time.
    """
    for provider in providers():
        try:
            if provider.available():
                return provider
        except Exception as e:
            logger.debug(f"Provider {provider.name} availability check failed: {e}")
    return None


def capabilities() -> Dict[str, Any]:
    """What Leti can actually ask this machine. For diagnostics; contacts nothing."""
    out = []
    for provider in providers():
        try:
            ok = provider.available()
        except Exception as e:
            ok = False
            logger.debug(f"{provider.name}: {e}")
        out.append({"provider": provider.name, "method": provider.method,
                    "available": ok})
    best = available_provider()
    return {
        "platform": sys.platform,
        "providers": out,
        "best": best.name if best else None,
        "method": best.method if best else BY_OCR,
        "note": ("No accessibility provider is available, so targets are matched "
                 "against the text read off the screen - which is weaker and is "
                 "reported as such on every resolution."
                 if best is None or best.method != BY_ACCESSIBILITY else
                 "Targets are resolved through the operating system's own UI "
                 "information."),
    }


# --------------------------------------------------------------------------- #
# Matching and scoring
#
# Deterministic, local, and never a model call. The scoring exists so that two
# genuinely comparable candidates come back as AMBIGUOUS instead of one of them
# being picked - which is the single most important behaviour in this file.
# --------------------------------------------------------------------------- #

_WORD = re.compile(r"[a-z0-9]+")

# Role words a person uses, mapped onto what backends call them.
_ROLES = {
    "button": ("button", "pushbutton", "splitbutton"),
    "tab": ("tab", "tabitem", "pagetab"),
    "menu": ("menu", "menubar"),
    "menu item": ("menuitem",),
    "field": ("edit", "text", "entry", "textbox", "passwordtext"),
    "checkbox": ("checkbox", "check box"),
    "link": ("hyperlink", "link"),
    "list item": ("listitem", "list item"),
    "window": ("window", "dialog", "frame", "pane"),
    "combo box": ("combobox", "combo box", "dropdown"),
}
_ROLE_OF = {alias: canonical for canonical, aliases in _ROLES.items()
            for alias in aliases}


def _normalise_role(raw: str) -> Optional[str]:
    cleaned = re.sub(r"controltype\.|\s+", "", str(raw or "").lower())
    return _ROLE_OF.get(cleaned, cleaned or None)


def _normalise(text: str) -> str:
    """A label reduced to what a person would say. Accelerators and ellipses go.

    "&Save as..." and "Save As" are the same menu item, and failing to click it
    because of an ampersand is the kind of thing that makes people stop trusting
    UI automation.
    """
    lowered = str(text or "").lower()
    lowered = lowered.replace("&", "").replace("…", "")
    lowered = re.sub(r"\.\.\.$", "", lowered.strip())
    lowered = re.sub(r"\s*\([^)]*\)\s*$", "", lowered)     # trailing "(Ctrl+S)"
    return " ".join(_WORD.findall(lowered))


def _could_be(name: str, wanted: str) -> bool:
    """A cheap pre-filter, run inside the tree walk so most nodes cost nothing."""
    if not wanted:
        return True
    normalised, target = _normalise(name), _normalise(wanted)
    if not target:
        return True
    return target in normalised or normalised in target or bool(
        set(target.split()) & set(normalised.split()))


# What each kind of agreement is worth. Coarse on purpose: a scale fine enough
# to separate two genuinely comparable candidates would be a scale fine enough
# to pick between them, which is exactly what must not happen.
SCORES = {
    "exact_name": 10,
    "normalised_name": 8,
    "prefix": 4,
    "word_overlap": 2,
    "role": 3,
    "automation_id": 6,
    "window": 2,
    "enabled": 1,
    "visible": 1,
}

# How far ahead the best candidate has to be before it is the answer rather than
# one of several. One whole signal's worth.
MARGIN = 3


def score(candidate: UITarget, wanted: str, role: str = "",
          window: str = "") -> Tuple[int, List[str]]:
    """How well this element matches, and why. Never a model call."""
    points = 0
    because: List[str] = []

    name, target = candidate.name or "", wanted or ""
    if name == target:
        points += SCORES["exact_name"]
        because.append("the name matches exactly")
    elif _normalise(name) == _normalise(target):
        points += SCORES["normalised_name"]
        because.append("the name matches")
    elif _normalise(name).startswith(_normalise(target)) and _normalise(target):
        points += SCORES["prefix"]
        because.append("the name starts with it")
    else:
        shared = set(_normalise(name).split()) & set(_normalise(target).split())
        if shared:
            points += SCORES["word_overlap"] * len(shared)
            because.append(f"shares {', '.join(sorted(shared)[:3])}")

    if role and candidate.role and _normalise_role(role) == candidate.role:
        points += SCORES["role"]
        because.append(f"it is a {candidate.role}")

    if candidate.automation_id and _normalise(candidate.automation_id) == _normalise(target):
        points += SCORES["automation_id"]
        because.append("its automation id matches")

    if window and candidate.window and _normalise(window) in _normalise(candidate.window):
        points += SCORES["window"]
        because.append("it is in the window named")

    if candidate.enabled:
        points += SCORES["enabled"]
    if candidate.visible:
        points += SCORES["visible"]

    return points, because


def choose(candidates: Sequence[UITarget], wanted: str, role: str = "",
           window: str = "") -> Resolution:
    """The element, the question, or the absence. The only place that decides.

    Four outcomes and no fifth:

      UNIQUE_MATCH   one candidate is clearly ahead, enabled and visible
      DISABLED       the best match is there and cannot be used
      AMBIGUOUS      two or more are comparable - the user picks, never this
      PARTIAL_MATCH  something is close and not close enough to act on
      NOT_FOUND      nothing resembles it
    """
    usable = [c for c in candidates if c is not None]
    if not usable:
        return Resolution(state=NOT_FOUND, wanted=wanted,
                          detail=f"Nothing named '{wanted}' is there.")

    scored = []
    for candidate in usable:
        points, because = score(candidate, wanted, role, window)
        scored.append((points, because, candidate))
    scored.sort(key=lambda row: -row[0])

    # Anything that shares nothing with the request is not a candidate at all.
    strong = [row for row in scored if row[0] >= SCORES["word_overlap"]]
    if not strong:
        return Resolution(state=NOT_FOUND, wanted=wanted,
                          candidates=[row[2] for row in scored[:MAX_CANDIDATES]],
                          detail=f"Nothing named '{wanted}' is there.")

    best_points, best_why, best = strong[0]
    runner_up = strong[1][0] if len(strong) > 1 else 0
    method = best.how

    # A name match that is only a word or two shared is not enough to click on.
    if best_points < SCORES["normalised_name"]:
        return Resolution(state=PARTIAL_MATCH, target=None, method=method,
                          candidates=[row[2] for row in strong[:MAX_CANDIDATES]],
                          wanted=wanted,
                          detail=(f"The closest thing to '{wanted}' is "
                                  f"'{best.name}', which is not close enough to "
                                  "act on. Say exactly what to click, or look again."))

    if best_points - runner_up < MARGIN:
        return Resolution(state=AMBIGUOUS, target=None, method=method,
                          candidates=[row[2] for row in strong[:MAX_CANDIDATES]],
                          wanted=wanted,
                          detail=(f"{len(strong)} elements match '{wanted}' about "
                                  "equally. Leti will not choose between them - "
                                  "ask which one is meant."))

    if best.enabled is False:
        return Resolution(state=DISABLED, target=best, method=method,
                          candidates=[best], wanted=wanted,
                          detail=(f"'{best.name}' is there and is disabled. "
                                  "Clicking it would do nothing - say so rather "
                                  "than clicking."))
    if best.visible is False:
        return Resolution(state=PARTIAL_MATCH, target=None, method=method,
                          candidates=[best], wanted=wanted,
                          detail=(f"'{best.name}' exists but is not on screen. "
                                  "Scroll to it or open what contains it first."))

    return Resolution(state=UNIQUE_MATCH, target=best, method=method,
                      candidates=[row[2] for row in strong[:MAX_CANDIDATES]],
                      wanted=wanted,
                      detail=f"'{best.name}' - {'; '.join(best_why[:3])}.")


# --------------------------------------------------------------------------- #
# Resolving, and checking a resolution is still true
# --------------------------------------------------------------------------- #

def resolve(wanted: str, role: str = "", window: str = "",
            observation: str = "") -> Resolution:
    """Which element is meant, through the strongest route this machine has.

    `observation` is whatever the screen was last read as - the existing
    screenshot path. It is the fallback, used only when no provider can answer,
    and a resolution that came from it says BY_OCR so nothing downstream mistakes
    it for the operating system's own answer.
    """
    if not str(wanted or "").strip():
        return Resolution(state=NOT_FOUND, wanted=wanted,
                          detail="No target was named, so there is nothing to resolve.")

    provider = available_provider()
    if provider is not None:
        try:
            candidates = provider.find(wanted, role, window)
        except Exception as e:
            logger.debug(f"Provider {provider.name} query failed: {e}")
            candidates = []
        if candidates:
            return choose(candidates, wanted, role, window)

    return _from_text(wanted, role, window, observation, had_provider=provider is not None)


def _from_text(wanted: str, role: str, window: str, observation: str,
               had_provider: bool) -> Resolution:
    """The existing behaviour, kept as the fallback it should always have been.

    A screenshot says a word is somewhere on the screen. That is real evidence
    and it is weaker evidence, so the state it produces is never better than
    PARTIAL_MATCH unless the whole label is present - and it always reports
    BY_OCR so the difference is visible.
    """
    seen = str(observation or "")
    if not seen.strip():
        return Resolution(
            state=UNSUPPORTED, wanted=wanted, method=BY_OCR,
            detail=("Nothing has been read off the screen and this machine has no "
                    "accessibility provider, so there is no way to tell whether "
                    f"'{wanted}' is there. Read the screen first."
                    if not had_provider else
                    f"The accessibility provider returned nothing for '{wanted}' "
                    "and the screen has not been read. Read it and try again."))

    lowered, target = seen.lower(), _normalise(wanted)
    if not target:
        return Resolution(state=NOT_FOUND, wanted=wanted, method=BY_OCR,
                          detail="There is nothing identifiable in that target.")

    occurrences = lowered.count(target)
    if occurrences == 1:
        return Resolution(
            state=UNIQUE_MATCH, method=BY_OCR, wanted=wanted,
            target=UITarget(name=wanted, how=BY_OCR, role=_normalise_role(role),
                            window=window or None, enabled=True, visible=True),
            detail=(f"'{wanted}' appears once in what was read off the screen. "
                    "This is screen text, not the interface's own answer - check "
                    "the result afterwards."))
    if occurrences > 1:
        return Resolution(
            state=AMBIGUOUS, method=BY_OCR, wanted=wanted,
            detail=(f"'{wanted}' appears {occurrences} times in what was read off "
                    "the screen, and screen text cannot tell them apart. Ask which "
                    "one is meant."))

    words = [w for w in target.split() if len(w) > 2]
    if words and all(w in lowered for w in words):
        return Resolution(
            state=PARTIAL_MATCH, method=BY_OCR, wanted=wanted,
            detail=(f"Every word of '{wanted}' is on screen but not together. "
                    "That is not the same element - look again or say exactly "
                    "what to click."))
    return Resolution(state=NOT_FOUND, method=BY_OCR, wanted=wanted,
                      detail=f"'{wanted}' is not in what was read off the screen.")


def still_valid(target: Optional[UITarget], now: Optional[float] = None,
                observation: str = "") -> Resolution:
    """Is this resolved element still the thing it was? Asked before acting.

    A target that was resolved and then sat while the model thought is not
    evidence about the screen now. Windows close, lists scroll, dialogs appear
    on top. So the element is looked up again and its identity compared - and
    anything that does not come back identical is STALE, which is a refusal.
    """
    if target is None:
        return Resolution(state=NOT_FOUND,
                          detail="There is no resolved target to act on.")

    age = target.age(now)
    if age > TARGET_MAX_AGE_SECONDS:
        fresh = _reresolve(target, observation)
        if fresh is None:
            return Resolution(
                state=STALE, wanted=target.name, method=target.how,
                detail=(f"'{target.name}' was resolved {int(age)}s ago and cannot be "
                        "found now. Look again before acting; do not click where it "
                        "used to be."))
        if fresh.identity() != target.identity():
            return Resolution(
                state=STALE, wanted=target.name, method=target.how,
                detail=(f"Something named '{target.name}' is there, and it is not the "
                        "same element as before. Resolve it again."))
        target = fresh

    if target.enabled is False:
        return Resolution(state=DISABLED, target=target, method=target.how,
                          wanted=target.name,
                          detail=f"'{target.name}' is disabled now.")
    if target.visible is False:
        return Resolution(state=STALE, target=None, method=target.how,
                          wanted=target.name,
                          detail=f"'{target.name}' is no longer on screen.")
    return Resolution(state=UNIQUE_MATCH, target=target, method=target.how,
                      wanted=target.name,
                      detail=f"'{target.name}' is still there.")


def _reresolve(target: UITarget, observation: str) -> Optional[UITarget]:
    provider = available_provider()
    if provider is not None:
        try:
            return provider.still_there(target)
        except Exception as e:
            logger.debug(f"Re-resolution failed: {e}")
            return None
    # No provider: the screen text is all there is, and it can only say the
    # label is still somewhere. That is enough to not be STALE and not enough to
    # be a different answer, so the target comes back with a fresh timestamp.
    if observation and _normalise(target.name) in str(observation).lower():
        return UITarget(**{**target.__dict__, "observed_at": time.time()})
    return None


def active_window() -> Optional[Dict[str, Any]]:
    """What is in front, if anything can say. On demand; nothing is cached."""
    for provider in providers():
        try:
            if not provider.available():
                continue
            found = provider.active_window()
            if found:
                return {**found, "via": provider.name}
        except Exception as e:
            logger.debug(f"Active window via {provider.name} failed: {e}")
    return None


def confidence(method: Optional[str]) -> str:
    """How much a resolution by this route is worth saying out loud."""
    if method == BY_ACCESSIBILITY:
        return "high - the operating system named this element"
    if method == BY_STRUCTURE:
        return "medium - the application named this window"
    if method in (BY_TEXT, BY_OCR):
        return "low - this is text read off the screen, not the element itself"
    return "lowest - nothing named this; it is a position on screen"


def is_weaker(method: Optional[str], than: str) -> bool:
    return _METHOD_RANK.get(method or BY_COORDINATES, 99) > _METHOD_RANK.get(than, 0)
