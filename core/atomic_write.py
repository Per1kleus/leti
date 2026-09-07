"""Crash-safe writes for the small JSON/YAML files Leti keeps its state in.

Every state file in the project was written with `path.write_text(json.dumps(...))`,
which truncates the file and then writes into it. Interrupt that - a crash, a
power cut, or simply the GUI's checkbox handler and a tool call saving the same
list at the same moment - and what's left on disk is a truncated file. Every
loader in the project then catches JSONDecodeError and returns an empty list, so
the failure mode is the user's contacts, to-dos or profile silently becoming
empty rather than an error anyone can see.

Writing to a temporary file in the same directory and then os.replace()-ing it
over the target makes the swap atomic: a reader sees either the old file or the
new one, never a half-written one.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str, secret: bool = False) -> None:
    """Replace `path`'s contents with `text` atomically.

    `secret=True` restricts the file to the owner (0600) - use it for anything
    holding credentials or session tokens, which would otherwise be created with
    whatever the process umask happens to be (commonly world-readable 0644).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Same directory as the target: os.replace is only atomic within a filesystem.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())   # the rename is pointless if the bytes aren't down yet
        if secret:
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        # Never leave the temp file behind on failure; the original is untouched.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any, secret: bool = False) -> None:
    """Serialize `data` as indented JSON and write it atomically."""
    atomic_write_text(path, json.dumps(data, indent=2), secret=secret)
