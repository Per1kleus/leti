"""
Safe filesystem operations. Protected-path enforcement is handled centrally
by SafetyGuard before these run; this module focuses on correct, bounded
file I/O (size limits, encoding safety, non-recursive-by-default search).
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
    description = "Read the text contents of a file."
    parameters = [
        ToolParameter(name="path", type="string", description="Path to the file to read."),
    ]

    async def run(self, path: str, **kwargs) -> ToolResult:
        try:
            p = resolve_path(path)
            if not p.exists():
                return ToolResult(success=False, error=f"File not found: {p}")
            if p.stat().st_size > MAX_READ_BYTES:
                return ToolResult(success=False, error=f"File too large to read directly ({p.stat().st_size} bytes).")
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
