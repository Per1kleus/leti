"""Shared fixtures.

SafetyGuard.settings reads through to config_loader on every access, so that a
/settings edit applies without a restart. That means a test can't simply assign
to it - the seam is the module-level get_settings the guard resolves, patched
per-test so nothing leaks into the next one.
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from core.config_loader import get_settings
from core.safety_guard import SafetyGuard


@pytest.fixture(scope="session", autouse=True)
def audit_somewhere_else():
    """Keep the test suite out of the real logs/audit.log.

    Measured before this existed: one full run appended 83 entries to the project's
    own audit log, and 23,549 had accumulated in it. That file is the permanent
    record of every action Leti took that changed something, and the user guide
    tells people to read it - so a copy filling up with test calls against
    /tmp/x.csv is an audit trail with fiction in it.

    Session-scoped and autouse because the leak is not from one test: any test that
    builds a SafetyGuard or drives the orchestrator writes a record, and there are
    hundreds of those.

    Redirecting the SETTING is not enough, which is worth knowing before anyone
    simplifies this. core/model_setup.py, core/permission_center.py and
    core/settings_editor.py all call reload_settings(), which drops the cached
    settings dict and re-reads settings.yaml from disk - so an override put there is
    thrown away partway through a run and every guard built afterwards goes back to
    the real file. Tried, and the log still grew by the same 18 KB. So the redirect
    is applied where the path is actually used, at construction, which no reload can
    undo. The one test that asserts on the audit file sets _audit_path itself after
    construction and is unaffected.
    """
    original_init = SafetyGuard.__init__

    with tempfile.TemporaryDirectory(prefix="leti-test-audit-") as directory:
        somewhere_else = pathlib.Path(directory) / "audit.log"

        def init_elsewhere(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._audit_path = somewhere_else

        SafetyGuard.__init__ = init_elsewhere
        try:
            yield
        finally:
            SafetyGuard.__init__ = original_init


@pytest.fixture
def guard_factory(monkeypatch):
    """Returns make(dry_run=..., confirm=..., confirm_classes=...) -> (guard, prompts).

    `confirm_classes` overrides safety.require_confirmation_for for the test, since
    the guard reads that setting live through the same patched get_settings.

    `prompts` records every confirmation the guard actually asked for, which is
    usually the thing under test: whether it asked at all.
    """

    def make(dry_run: bool = False, confirm: bool = True, confirm_classes=None):
        prompts: list[str] = []

        async def callback(prompt: str) -> bool:
            prompts.append(prompt)
            return confirm

        settings = get_settings()
        settings["safety"]["dry_run"] = dry_run
        if confirm_classes is not None:
            settings["safety"]["require_confirmation_for"] = confirm_classes
        monkeypatch.setattr("core.safety_guard.get_settings", lambda: settings)

        return SafetyGuard(confirmation_callback=callback), prompts

    return make
