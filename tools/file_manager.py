"""
Safe filesystem operations. Protected-path enforcement is handled centrally
by SafetyGuard before these run; this module focuses on correct, bounded
file I/O (size limits, encoding safety, non-recursive-by-default search).

read_file reads TEXT. Pointed at a PDF it used to decode the bytes with
errors="replace" and report success, handing the model a page of mojibake that
reads like a file whose contents are unknowable - which is a worse answer than a
refusal, because the model has no way to tell it apart from a real reading. It
now sends anything core/documents.py knows how to open through that reader
instead, and returns the labelled sections. Same tool, same argument, same
answer for a text file; a correct answer instead of nonsense for the rest.
"""
from __future__ import annotations

import fnmatch
import os
import shutil
from pathlib import Path
from typing import Optional

from core.config_loader import resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

MAX_READ_BYTES = 200_000  # ~200KB cap to avoid dumping huge files into the LLM context


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a text file whole - code, configuration, notes, logs. For a PDF, Word or "
        "Excel file it hands back the readable part with the page or sheet it came from, "
        "and for anything long use read_document instead, which takes a question and "
        "returns only the part that answers it."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="Path to the file to read."),
    ]

    async def run(self, path: str, **kwargs) -> ToolResult:
        try:
            p = resolve_path(path)
            if not p.exists():
                return ToolResult(success=False, error=f"File not found: {p}")

            from core import documents

            kind = documents.kind_of(p)
            if kind == "image":
                return ToolResult(success=False, error=(
                    f"{p.name} is an image, not text. Leti has no OCR, so there is no text "
                    "to read out of it - use look_at_image to have the vision model "
                    "describe what it shows."))
            # A format whose bytes are not its text. Decoding those would be the
            # mojibake this exists to stop, so it goes through the reader that
            # knows the format - same tool, correct answer.
            if kind in documents.BINARY_KINDS:
                result = documents.extract(p, max_chars=documents.DEFAULT_EXTRACT_CHARS)
                if not result.get("ok"):
                    return ToolResult(success=False, error=result.get("error"), output=result)
                return ToolResult(success=True, output=result)

            if p.stat().st_size > MAX_READ_BYTES:
                # Too big to hand over whole, but not unreadable: return the start,
                # labelled, and say how to ask for the rest.
                result = documents.extract(p, max_chars=documents.DEFAULT_EXTRACT_CHARS)
                if result.get("ok"):
                    result["note"] = (
                        f"{p.name} is {p.stat().st_size:,} bytes, too much to return whole. "
                        "This is the start of it; use read_document with a question to get "
                        "the part you need.")
                    return ToolResult(success=True, output=result)
                return ToolResult(
                    success=False,
                    error=f"File too large to read directly ({p.stat().st_size} bytes).")

            content = p.read_text(encoding="utf-8", errors="replace")
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write (or overwrite) text content to a file."
    parameters = [
        ToolParameter(name="path", type="string", description="Path to the file to write."),
        ToolParameter(name="content", type="string", description="Text content to write."),
        ToolParameter(
            name="append", type="boolean", description="Append instead of overwrite.", required=False
        ),
    ]

    async def run(self, path: str, content: str, append: bool = False, **kwargs) -> ToolResult:
        try:
            p = resolve_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(p, mode, encoding="utf-8") as f:
                f.write(content)
            return ToolResult(success=True, output=f"Wrote {len(content)} characters to {p}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List files in a directory, optionally filtered by a glob pattern."
    parameters = [
        ToolParameter(name="directory", type="string", description="Directory to list."),
        ToolParameter(
            name="pattern", type="string", description="Glob pattern, e.g. '*.py'.", required=False
        ),
        ToolParameter(
            name="recursive", type="boolean", description="Search subdirectories too.", required=False
        ),
    ]

    async def run(
        self, directory: str, pattern: str = "*", recursive: bool = False, **kwargs
    ) -> ToolResult:
        try:
            d = resolve_path(directory)
            if not d.is_dir():
                return ToolResult(success=False, error=f"Not a directory: {d}")
            glob_fn = d.rglob if recursive else d.glob
            matches = [str(p) for p in glob_fn(pattern) if p.is_file()]
            return ToolResult(success=True, output=matches[:500])  # cap results
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "Permanently delete a file. Destructive - requires confirmation."
    parameters = [
        ToolParameter(name="path", type="string", description="Path to the file to delete."),
    ]

    async def run(self, path: str, **kwargs) -> ToolResult:
        try:
            p = resolve_path(path)
            if not p.exists():
                return ToolResult(success=False, error=f"File not found: {p}")
            p.unlink()
            return ToolResult(success=True, output=f"Deleted {p}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class MoveFileTool(BaseTool):
    name = "move_file"
    description = "Move or rename a file."
    parameters = [
        ToolParameter(name="source_path", type="string", description="Current file path."),
        ToolParameter(name="destination_path", type="string", description="New file path."),
    ]

    async def run(self, source_path: str, destination_path: str, **kwargs) -> ToolResult:
        try:
            src = resolve_path(source_path)
            dst = resolve_path(destination_path)
            if not src.exists():
                return ToolResult(success=False, error=f"Source not found: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return ToolResult(success=True, output=f"Moved {src} -> {dst}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
