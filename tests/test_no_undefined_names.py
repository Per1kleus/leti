"""Nothing in the application refers to a name that does not exist.

Both of the bugs this suite was written for were the same shape: a function that
looks right, passes review, is covered by no test, and raises NameError the first
time a user reaches it. `add_social_watch` could not save a watch at all, and the
failure surfaced as a tool error rather than a crash, so it looked like a
configuration problem.

pyflakes finds exactly this class of thing in a second, so it runs here on every
test run rather than being something somebody remembers to do.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

APPLICATION = ["core", "tools", "memory", "gui", "main.py"]


def _pyflakes(paths):
    result = subprocess.run([sys.executable, "-m", "pyflakes", *paths],
                            capture_output=True, text=True, cwd=pathlib.Path.cwd())
    return result.stdout.splitlines()


@pytest.mark.parametrize("marker", ["undefined name", "redefinition of unused"])
def test_no_module_refers_to_something_that_is_not_there(marker):
    problems = [line for line in _pyflakes(APPLICATION) if marker in line]
    assert not problems, "\n".join(problems)


def test_adding_a_social_watch_actually_saves_it(tmp_path, monkeypatch):
    """The regression: _save_watches called a function nobody had imported."""
    import asyncio
    import json

    import tools.social_media as social_media

    path = tmp_path / "watches.json"
    monkeypatch.setattr(social_media, "_watches_path", lambda: path)
    result = asyncio.run(social_media.AddSocialWatchTool().run(
        platform="youtube", identifier="@someone"))
    assert result.success is True
    saved = json.loads(path.read_text())
    assert len(saved) == 1 and saved[0]["identifier"] == "@someone"
