"""Tests for the state files Leti keeps on disk.

Every JSON store in the project loads with `except JSONDecodeError: return []`,
which turns a corrupt file into silent data loss rather than a visible error -
so the write path is where correctness has to live.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from core.atomic_write import atomic_write_json, atomic_write_text


def test_write_replaces_content(tmp_path):
    target = tmp_path / "todos.json"
    atomic_write_json(target, [{"id": "a", "text": "first"}])
    atomic_write_json(target, [{"id": "b", "text": "second"}])
    assert json.loads(target.read_text()) == [{"id": "b", "text": "second"}]


def test_failed_write_leaves_original_intact(tmp_path, monkeypatch):
    """The point of writing via a temp file: an interrupted write must not be
    able to leave a truncated file where the user's data was."""
    target = tmp_path / "contacts.json"
    atomic_write_json(target, [{"name": "original"}])

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(target, [{"name": "replacement"}])

    assert json.loads(target.read_text()) == [{"name": "original"}]


def test_failed_write_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    target = tmp_path / "contacts.json"
    atomic_write_json(target, [{"name": "original"}])

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(OSError):
        atomic_write_json(target, [{"name": "replacement"}])

    assert [p.name for p in tmp_path.iterdir()] == ["contacts.json"]


def test_secret_files_are_owner_only(tmp_path):
    """settings.local.yaml holds app passwords and API keys; the GUI token grants
    a LAN device a full session. Neither should inherit a world-readable umask."""
    target = tmp_path / "settings.local.yaml"
    atomic_write_text(target, "email:\n  app_password: hunter2\n", secret=True)

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_all_state_files_are_owner_only(tmp_path):
    """Not just the credential files: contacts, the profile and conversation
    state are all personal, and these land 0600 rather than at the umask's mercy."""
    target = tmp_path / "todo_list.json"
    atomic_write_json(target, [])
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "data" / "nested" / "profile.json"
    atomic_write_json(target, {"name": "x"})
    assert json.loads(target.read_text()) == {"name": "x"}


# --- Personality and user profile (the files that carry who Leti is talking to) ---

@pytest.mark.asyncio
async def test_personality_persists_and_reaches_the_system_prompt(tmp_path, monkeypatch):
    import tools.personality as personality

    monkeypatch.setattr(personality, "_path", lambda: tmp_path / "personality.json")

    await personality.SetPersonalityTool().run(sarcasm=9, directness=9)
    assert json.loads((tmp_path / "personality.json").read_text())["sarcasm"] == 9

    described = personality.describe_personality()
    assert "Sarcasm (9/10)" in described
    # The dials shape tone only - that boundary is stated in the injected text
    # itself rather than left to be inferred.
    assert "TONE only" in described

    await personality.ResetPersonalityTool().run()
    assert json.loads((tmp_path / "personality.json").read_text())["sarcasm"] == 2


@pytest.mark.asyncio
async def test_user_profile_round_trips(tmp_path, monkeypatch):
    import tools.user_profile as profile

    monkeypatch.setattr(profile, "_path", lambda: tmp_path / "user_profile.json")

    await profile.SetUserNameTool().run(name="Perikles")
    await profile.RememberAboutUserTool().run(fact="prefers blunt answers", category="preference")

    summary = profile.profile_summary()
    assert "Perikles" in summary
    assert "prefers blunt answers" in summary

    viewed = await profile.ViewUserProfileTool().run()
    fact_id = viewed.output["facts"][0]["id"]

    assert (await profile.ForgetUserFactTool().run(fact_id=fact_id)).success
    assert (await profile.ForgetUserFactTool().run(fact_id="nope")).success is False

    await profile.ClearUserProfileTool().run()
    assert profile.profile_summary() == ""
