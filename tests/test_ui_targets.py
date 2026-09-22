"""Which element that is, and refusing to guess when the answer is two of them.

The one behaviour worth protecting: only UNIQUE_MATCH may act. Everything else -
ambiguous, disabled, partial, stale, missing, unsupported - is a refusal with a
reason, and no amount of "probably that one" turns any of them into a click.
"""
from __future__ import annotations

import time

import pytest

from core import ui_targets as ui


class FakeProvider(ui.Provider):
    """A provider whose tree is whatever a test says it is."""

    name = "fake"
    method = ui.BY_ACCESSIBILITY

    def __init__(self, elements=(), window=None, ok=True, explode=False):
        self.elements = list(elements)
        self.window = window or {"window": "Settings", "application": "settings.exe"}
        self.ok = ok
        self.explode = explode
        self.queries = 0

    def available(self):
        if self.explode:
            raise RuntimeError("the provider is broken")
        return self.ok

    def active_window(self):
        return self.window

    def find(self, wanted, role="", window=""):
        self.queries += 1
        return [e for e in self.elements if ui._could_be(e.name, wanted)]


def element(name, **kwargs):
    kwargs.setdefault("how", ui.BY_ACCESSIBILITY)
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("visible", True)
    kwargs.setdefault("window", "Settings")
    return ui.UITarget(name=name, **kwargs)


@pytest.fixture(autouse=True)
def _no_real_providers():
    ui.reset_providers([])
    yield
    ui.reset_providers()


# --- Found, and found unambiguously ---------------------------------------------------

def test_one_matching_element_resolves():
    ui.reset_providers([FakeProvider([element("Settings", role="button"),
                                      element("Help", role="link")])])
    found = ui.resolve("Settings", "button")
    assert found.state == ui.UNIQUE_MATCH
    assert found.target.name == "Settings"
    assert found.method == ui.BY_ACCESSIBILITY
    assert found.may_act is True


def test_an_exact_name_beats_a_partial_one():
    ui.reset_providers([FakeProvider([element("Settings", role="button"),
                                      element("Settings and privacy", role="button")])])
    found = ui.resolve("Settings", "button")
    assert found.state == ui.UNIQUE_MATCH and found.target.name == "Settings"


def test_the_role_separates_two_things_with_one_name():
    ui.reset_providers([FakeProvider([element("Settings", role="button"),
                                      element("Settings", role="tab")])])
    found = ui.resolve("Settings", "button")
    assert found.state == ui.UNIQUE_MATCH and found.target.role == "button"


def test_an_automation_id_is_strong_evidence():
    ui.reset_providers([FakeProvider([
        element("Save", role="button", automation_id="Save"),
        element("Save as", role="button")])])
    assert ui.resolve("Save", "button").state == ui.UNIQUE_MATCH


def test_an_accelerator_does_not_stop_a_match():
    """"&Save as..." and "Save As" are the same menu item."""
    ui.reset_providers([FakeProvider([element("&Save as...", role="menu item")])])
    assert ui.resolve("Save As", "menu item").state == ui.UNIQUE_MATCH


# --- Refusing --------------------------------------------------------------------------

def test_two_comparable_elements_are_ambiguous_not_a_choice():
    ui.reset_providers([FakeProvider([element("Settings", role="button"),
                                      element("Settings", role="button")])])
    found = ui.resolve("Settings", "button")
    assert found.state == ui.AMBIGUOUS
    assert found.may_act is False
    assert found.target is None
    assert "will not choose between them" in found.detail


def test_nothing_like_it_is_not_found():
    ui.reset_providers([FakeProvider([element("Help"), element("About")])])
    found = ui.resolve("Checkout", "button")
    assert found.state == ui.NOT_FOUND and found.may_act is False


def test_a_disabled_element_is_reported_not_clicked():
    ui.reset_providers([FakeProvider([element("Save", role="button", enabled=False)])])
    found = ui.resolve("Save", "button")
    assert found.state == ui.DISABLED
    assert found.may_act is False
    assert "would do nothing" in found.detail


def test_an_offscreen_element_is_not_actionable():
    ui.reset_providers([FakeProvider([element("Save", role="button", visible=False)])])
    found = ui.resolve("Save", "button")
    assert found.state == ui.PARTIAL_MATCH and found.may_act is False


def test_a_weak_word_overlap_is_a_partial_match_not_a_click():
    ui.reset_providers([FakeProvider([element("Save all open documents",
                                              role="button")])])
    found = ui.resolve("Save draft", "button")
    assert found.state in (ui.PARTIAL_MATCH, ui.NOT_FOUND)
    assert found.may_act is False


def test_an_empty_target_resolves_to_nothing():
    assert ui.resolve("").state == ui.NOT_FOUND


@pytest.mark.parametrize("state", [s for s in ui.STATES if s != ui.UNIQUE_MATCH])
def test_only_a_unique_match_may_act(state):
    assert state not in ui.ACTIONABLE


# --- Falling back ------------------------------------------------------------------------

def test_no_provider_falls_back_to_the_screen_text():
    found = ui.resolve("Settings", observation="A window with a Settings button")
    assert found.state == ui.UNIQUE_MATCH
    assert found.method == ui.BY_OCR
    assert "screen text" in found.detail or "read off the screen" in found.detail


def test_screen_text_that_says_it_twice_is_ambiguous():
    found = ui.resolve("Settings",
                       observation="Settings in the sidebar and Settings in the menu")
    assert found.state == ui.AMBIGUOUS
    assert "cannot tell them apart" in found.detail


def test_no_provider_and_no_observation_is_unsupported():
    found = ui.resolve("Settings")
    assert found.state == ui.UNSUPPORTED
    assert "Read the screen first" in found.detail


def test_a_provider_that_finds_nothing_falls_through_to_text():
    ui.reset_providers([FakeProvider([])])
    found = ui.resolve("Settings", observation="A Settings button is here")
    assert found.state == ui.UNIQUE_MATCH and found.method == ui.BY_OCR


def test_a_provider_that_raises_never_stops_a_resolution():
    ui.reset_providers([FakeProvider(explode=True)])
    found = ui.resolve("Settings", observation="A Settings button is here")
    assert found.state in ui.STATES


def test_accessibility_is_preferred_over_text():
    provider = FakeProvider([element("Settings", role="button")])
    ui.reset_providers([provider])
    found = ui.resolve("Settings", "button",
                       observation="a Settings label somewhere")
    assert found.method == ui.BY_ACCESSIBILITY
    assert provider.queries == 1


def test_the_method_ranking_says_which_is_weaker():
    assert ui.is_weaker(ui.BY_OCR, ui.BY_ACCESSIBILITY) is True
    assert ui.is_weaker(ui.BY_ACCESSIBILITY, ui.BY_OCR) is False
    assert ui.is_weaker(ui.BY_COORDINATES, ui.BY_OCR) is True


def test_confidence_names_what_each_route_is_worth():
    assert "high" in ui.confidence(ui.BY_ACCESSIBILITY)
    assert "low" in ui.confidence(ui.BY_OCR)
    assert "lowest" in ui.confidence(ui.BY_COORDINATES)


# --- Staleness ---------------------------------------------------------------------------

def test_a_fresh_target_is_still_valid():
    ui.reset_providers([FakeProvider([element("Settings", role="button")])])
    target = ui.resolve("Settings", "button").target
    assert ui.still_valid(target).may_act is True


def test_an_old_target_that_is_gone_is_stale():
    target = element("Settings", role="button")
    target.observed_at = time.time() - ui.TARGET_MAX_AGE_SECONDS - 1
    ui.reset_providers([FakeProvider([])])
    found = ui.still_valid(target)
    assert found.state == ui.STALE and found.may_act is False
    assert "do not click where it used to be" in found.detail.lower()


def test_an_old_target_that_changed_identity_is_stale():
    target = element("Settings", role="button", automation_id="settings-1")
    target.observed_at = time.time() - ui.TARGET_MAX_AGE_SECONDS - 1
    ui.reset_providers([FakeProvider([element("Settings", role="button",
                                              automation_id="settings-2")])])
    assert ui.still_valid(target).state == ui.STALE


def test_still_valid_checks_identity_itself_not_only_the_provider():
    """Defence in depth, pinned directly.

    Provider.still_there already filters by identity, so a mutation that removes
    still_valid's own identity check survives - the base class catches it first.
    This provider deliberately hands back a DIFFERENT element from still_there,
    which is what a backend doing its own re-resolution on name alone would do,
    and proves the second check is real work rather than decoration.
    """
    class LyingProvider(FakeProvider):
        def still_there(self, target):
            return element("Settings", role="button", automation_id="different")

    target = element("Settings", role="button", automation_id="original")
    target.observed_at = time.time() - ui.TARGET_MAX_AGE_SECONDS - 1
    ui.reset_providers([LyingProvider([])])
    found = ui.still_valid(target)
    assert found.state == ui.STALE
    assert found.may_act is False
    assert "not the same element" in found.detail


def test_an_old_target_that_is_identical_is_refreshed():
    target = element("Settings", role="button", automation_id="settings-1")
    target.observed_at = time.time() - ui.TARGET_MAX_AGE_SECONDS - 1
    ui.reset_providers([FakeProvider([element("Settings", role="button",
                                              automation_id="settings-1")])])
    assert ui.still_valid(target).may_act is True


def test_a_target_that_became_disabled_is_refused():
    target = element("Save", role="button")
    target.enabled = False
    assert ui.still_valid(target).state == ui.DISABLED


def test_nothing_resolved_is_nothing_to_act_on():
    assert ui.still_valid(None).state == ui.NOT_FOUND


def test_identity_ignores_position_but_not_name():
    a = element("Save", role="button", rect=(0, 0, 10, 10))
    b = element("Save", role="button", rect=(500, 500, 510, 510))
    c = element("Save as", role="button", rect=(0, 0, 10, 10))
    assert a.identity() == b.identity()
    assert a.identity() != c.identity()


# --- Efficiency ---------------------------------------------------------------------------

def test_nothing_is_scanned_until_something_asks():
    provider = FakeProvider([element("Settings")])
    ui.reset_providers([provider])
    assert provider.queries == 0
    ui.resolve("Settings")
    assert provider.queries == 1


def test_no_background_work_is_started():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/ui_targets.py").read_text())
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("Thread", "Timer", "start", "create_task", "sleep",
                      "schedule", "run_in_executor"):
        assert forbidden not in called, f"core/ui_targets.py calls {forbidden}"


def test_no_unbounded_walk():
    import pathlib

    source = pathlib.Path("core/ui_targets.py").read_text()
    assert "MAX_ELEMENTS_SCANNED" in source
    assert source.count("while stack and scanned < MAX_ELEMENTS_SCANNED") >= 1


def test_only_a_handful_of_candidates_ever_leave():
    ui.reset_providers([FakeProvider([element(f"Settings {i}") for i in range(40)])])
    found = ui.resolve("Settings")
    assert len(found.candidates) <= ui.MAX_CANDIDATES
    assert len(found.describe()["candidates"]) <= ui.MAX_CANDIDATES


def test_a_described_target_is_a_few_fields_not_a_tree():
    described = element("Settings", role="button", automation_id="x").describe()
    assert set(described) <= {"name", "how", "role", "application", "window",
                              "automation_id", "enabled", "visible", "parent"}


def test_resolution_is_fast_enough_to_do_before_every_click():
    ui.reset_providers([FakeProvider([element(f"Thing {i}") for i in range(200)])])
    ui.resolve("Thing 7")
    started = time.perf_counter()
    for _ in range(200):
        ui.resolve("Thing 7")
    assert (time.perf_counter() - started) / 200 < 0.01


def test_a_providers_availability_is_asked_once_not_once_per_click():
    """The window-list provider answers "can I work here?" by enumerating the
    desktop. Paying that on every resolution made resolution the slow path, so
    the answer is remembered - and remembered by arithmetic, not by a timer."""
    provider = FakeProvider([element("Settings", role="button")])
    asked = []
    provider.available = lambda: (asked.append(1), True)[1]
    ui.reset_providers([provider])
    for _ in range(50):
        ui.resolve("Settings", "button")
    assert len(asked) == 1, f"the desktop was enumerated {len(asked)} times for 50 clicks"


def test_the_remembered_availability_does_expire():
    """Remembered is not frozen: a machine that gains an accessibility stack
    must be noticed without restarting Leti."""
    provider = FakeProvider([])
    asked = []
    provider.available = lambda: (asked.append(1), True)[1]
    assert provider.is_available(now=1000.0) is True
    assert provider.is_available(now=1000.0 + ui.AVAILABILITY_MEMO_SECONDS - 1) is True
    assert len(asked) == 1
    assert provider.is_available(now=1000.0 + ui.AVAILABILITY_MEMO_SECONDS + 1) is True
    assert len(asked) == 2, "the answer was kept forever"


def test_a_provider_that_forgets_to_call_super_still_answers():
    """The memo's state lives on the class, so a provider with its own __init__
    degrades to asking rather than raising AttributeError mid-click."""
    class OwnConstructor(ui.Provider):
        name = "own"
        method = ui.BY_ACCESSIBILITY

        def __init__(self):
            self.thing = object()

        def available(self):
            return True

    assert OwnConstructor().is_available() is True


def test_one_provider_remembering_does_not_answer_for_another():
    a, b = FakeProvider([], ok=True), FakeProvider([], ok=False)
    assert a.is_available() is True
    assert b.is_available() is False


def test_a_broken_availability_check_reads_as_unavailable():
    exploding = FakeProvider([], explode=True)
    assert exploding.is_available() is False


def test_capabilities_reports_what_this_machine_can_do():
    reported = ui.capabilities()
    assert "platform" in reported and "providers" in reported
    assert reported["method"] in ui.METHODS


def test_the_module_makes_no_model_or_network_call():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("core/ui_targets.py").read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    for forbidden in ("core.llm_client", "httpx", "requests", "urllib.request"):
        assert forbidden not in imports
