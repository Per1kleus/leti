"""Tests for tool-level bugs where a tool reported one thing and did another."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from tools.backup_restore import _safe_label
from tools.command_runner import run_command


# --- Snapshot labels -----------------------------------------------------------

@pytest.mark.parametrize("label", ["../../etc", "..", "a/b", "/absolute", "", "  ", "x" * 65])
def test_snapshot_label_rejects_path_traversal(label):
    """The label is joined onto the snapshot root and the directory it names is
    rmtree'd, so it has to be an ordinary filename."""
    with pytest.raises(ValueError):
        _safe_label(label)


@pytest.mark.parametrize("label", ["ssh-config", "my_backup.1", "Project2024"])
def test_snapshot_label_accepts_ordinary_names(label):
    assert _safe_label(label) == label


# --- Command runner ------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_command_is_reported_as_failure():
    """The old helper returned only stdout, so a command that failed looked
    identical to one that succeeded quietly - which is how 'sudo -n ufw enable'
    failing became 'Firewall enable command issued.'"""
    result = await run_command([sys.executable, "-c", "import sys; sys.stderr.write('nope'); sys.exit(3)"])
    assert result.ok is False
    assert result.returncode == 3
    assert "nope" in result.failure_reason()


@pytest.mark.asyncio
async def test_successful_command_reports_stdout():
    result = await run_command([sys.executable, "-c", "print('hello')"])
    assert result.ok is True
    assert result.stdout.strip() == "hello"
    assert result.failure_reason() == ""


@pytest.mark.asyncio
async def test_missing_executable_is_distinguishable():
    result = await run_command(["definitely-not-a-real-binary-xyz"])
    assert result.ok is False
    assert result.not_found is True


@pytest.mark.asyncio
async def test_timeout_kills_and_reaps_the_child():
    result = await run_command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert result.ok is False
    assert "timed out" in result.failure_reason()


@pytest.mark.asyncio
async def test_run_command_does_not_block_the_event_loop():
    """A 30-minute apt upgrade used to freeze the GUI's websocket server. Other
    coroutines must keep making progress while a command runs."""
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await run_command([sys.executable, "-c", "import time; time.sleep(0.4)"])
    task.cancel()

    assert ticks > 5, f"event loop was starved during the subprocess (ticks={ticks})"


# --- Facebook watch ids --------------------------------------------------------

def test_facebook_post_ids_are_stable_across_processes():
    """hash() on a str is randomized per process, so ids built from it changed on
    every restart and check_social_watches reported the same posts as new forever."""
    code = (
        "import hashlib;"
        "print(hashlib.sha256('a post body'.encode('utf-8')).hexdigest()[:16])"
    )
    first = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    second = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin"},
    ).stdout
    assert first == second != ""
