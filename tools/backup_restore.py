"""
Integrity baselining and backup/restore for files/paths the user cares
about protecting. Lets Leti:
  1. Snapshot a set of paths (hash + copy) as a known-good baseline.
  2. Later diff the current state against that baseline to "find the
     fault" - files that were modified, added, or deleted since.
  3. Restore any changed/missing file from the snapshot.

This does not attempt whole-disk/system-image backup - it's a scoped,
auditable snapshot of specific files or directories the user registers
(e.g. config files, a project folder, browser bookmarks export).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from core.config_loader import resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

SNAPSHOT_ROOT = "./data/security_snapshots"
MANIFEST_NAME = "manifest.json"


def _snapshot_dir() -> Path:
    d = resolve_path(SNAPSHOT_ROOT)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path):
    if root.is_file():
        yield root
    else:
        yield from (f for f in root.rglob("*") if f.is_file())


class CreateSecuritySnapshotTool(BaseTool):
    name = "create_security_snapshot"
    description = (
        "Create a hashed, copied baseline snapshot of the given file(s)/directory so future "
        "tampering, corruption, or unwanted changes can be detected and reverted."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="path", type="string", description="File or directory to baseline."),
        ToolParameter(name="label", type="string", description="Short name for this snapshot (e.g. 'ssh-config')."),
    ]

    async def run(self, path: str, label: str, **kwargs) -> ToolResult:
        try:
            src = resolve_path(path)
            if not src.exists():
                return ToolResult(success=False, error=f"Path not found: {src}")

            snap_dir = _snapshot_dir() / label
            data_dir = snap_dir / "data"
            if data_dir.exists():
                shutil.rmtree(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

            manifest: Dict[str, Any] = {"source_root": str(src), "created": time.time(), "files": {}}
            for f in _iter_files(src):
                rel = f.relative_to(src) if src.is_dir() else Path(f.name)
                dest = data_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                manifest["files"][str(rel)] = _hash_file(f)

            (snap_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
            return ToolResult(
                success=True,
                output=f"Snapshot '{label}' created: {len(manifest['files'])} file(s) baselined from {src}.",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CheckIntegrityTool(BaseTool):
    name = "check_integrity"
    description = (
        "Compare the current state of a path against its stored security snapshot to find "
        "modified, added, or deleted files ('find the fault'). Read-only."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="label", type="string", description="Snapshot label to check against."),
    ]

    async def run(self, label: str, **kwargs) -> ToolResult:
        try:
            snap_dir = _snapshot_dir() / label
            manifest_path = snap_dir / MANIFEST_NAME
            if not manifest_path.exists():
                return ToolResult(success=False, error=f"No snapshot found with label '{label}'.")

            manifest = json.loads(manifest_path.read_text())
            src = Path(manifest["source_root"])

            modified, deleted, added = [], [], []
            current_files = set()
            if src.exists():
                for f in _iter_files(src):
                    rel = str(f.relative_to(src) if src.is_dir() else Path(f.name))
                    current_files.add(rel)
                    if rel not in manifest["files"]:
                        added.append(rel)
                    elif _hash_file(f) != manifest["files"][rel]:
                        modified.append(rel)
                deleted = [rel for rel in manifest["files"] if rel not in current_files]
            else:
                deleted = list(manifest["files"].keys())

            clean = not (modified or deleted or added)
            return ToolResult(
                success=True,
                output={
                    "clean": clean,
                    "modified": modified,
                    "deleted": deleted,
                    "added": added,
                    "summary": "No changes detected." if clean else (
                        f"{len(modified)} modified, {len(deleted)} deleted, {len(added)} added since snapshot."
                    ),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class RestoreFromSnapshotTool(BaseTool):
    name = "restore_from_snapshot"
    description = (
        "Restore file(s) to their last known-good state from a security snapshot, overwriting "
        "current (potentially tampered/corrupted) versions. Destructive: requires confirmation."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="label", type="string", description="Snapshot label to restore from."),
        ToolParameter(
            name="only_file", type="string", required=False,
            description="If set, restore only this single relative file path instead of everything.",
        ),
    ]

    async def run(self, label: str, only_file: str = "", **kwargs) -> ToolResult:
        try:
            snap_dir = _snapshot_dir() / label
            manifest_path = snap_dir / MANIFEST_NAME
            if not manifest_path.exists():
                return ToolResult(success=False, error=f"No snapshot found with label '{label}'.")

            manifest = json.loads(manifest_path.read_text())
            src_root = Path(manifest["source_root"])
            data_dir = snap_dir / "data"

            targets = [only_file] if only_file else list(manifest["files"].keys())
            restored = []
            for rel in targets:
                if rel not in manifest["files"]:
                    continue
                backup_file = data_dir / rel
                dest = src_root / rel if src_root.is_dir() else src_root
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_file, dest)
                restored.append(rel)

            return ToolResult(success=True, output=f"Restored {len(restored)} file(s) from snapshot '{label}': {restored}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
