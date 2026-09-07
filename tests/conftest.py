"""Shared fixtures.

SafetyGuard.settings reads through to config_loader on every access, so that a
/settings edit applies without a restart. That means a test can't simply assign
to it - the seam is the module-level get_settings the guard resolves, patched
per-test so nothing leaks into the next one.
"""
from __future__ import annotations

import pytest

from core.config_loader import get_settings
from core.safety_guard import SafetyGuard


@pytest.fixture
def guard_factory(monkeypatch):
    """Returns make(dry_run=..., confirm=...) -> (guard, prompts).

    `prompts` records every confirmation the guard actually asked for, which is
    usually the thing under test: whether it asked at all.
    """

    def make(dry_run: bool = False, confirm: bool = True):
        prompts: list[str] = []

        async def callback(prompt: str) -> bool:
            prompts.append(prompt)
            return confirm

        settings = get_settings()
        settings["safety"]["dry_run"] = dry_run
        monkeypatch.setattr("core.safety_guard.get_settings", lambda: settings)

        return SafetyGuard(confirmation_callback=callback), prompts

    return make
