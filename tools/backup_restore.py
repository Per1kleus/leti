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
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from core.atomic_write import atomic_write_json
from core.config_loader import resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

SNAPSHOT_ROOT = "./data/security_snapshots"
MANIFEST_NAME = "manifest.json"

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _safe_label(label: str) -> str:
    """Validate a snapshot label before it is used as a directory name.

    The label comes from the model, and it was being joined straight onto the
    snapshot root - so '../../..' escaped the snapshot directory entirely, at
    which point create_security_snapshot's shutil.rmtree(data_dir) would run
    against whatever it landed on. Restricted to a single, ordinary filename.
    """
    label = (label or "").strip()
    if not _SAFE_LABEL.match(label):
        raise ValueError(
            f"Invalid snapshot label {label!r}: use only letters, numbers, dots, "
            f"dashes and underscores (max 64 characters, no path separators)."
        )
    return label


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
            label = _safe_label(label)
            src = resolve_path(path)
            if not src.exists():
                return ToolResult(success=False, error=f"Path not found: {src}")

            snap_dir = _snapshot_dir() / label
            data_dir = snap_dir / "data"
            if data_dir.exists():
                shutil.rmtree(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

            manifest: Dict[str, Any] = {
                "source_root": str(src),
                # Whether the source was a directory or a single file decides how
                # restore rebuilds the paths. Recording it here means restore
                # doesn't have to infer it from a filesystem that may since have
                # changed - see RestoreFromSnapshotTool.
                "source_is_dir": src.is_dir(),
                "created": time.time(),
                "files": {},
            }
            for f in _iter_files(src):
                rel = f.relative_to(src) if src.is_dir() else Path(f.name)
                dest = data_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                manifest["files"][str(rel)] = _hash_file(f)

            atomic_write_json(snap_dir / MANIFEST_NAME, manifest)
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
            label = _safe_label(label)
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
            label = _safe_label(label)
            snap_dir = _snapshot_dir() / label
            manifest_path = snap_dir / MANIFEST_NAME
            if not manifest_path.exists():
                return ToolResult(success=False, error=f"No snapshot found with label '{label}'.")

            manifest = json.loads(manifest_path.read_text())
            src_root = Path(manifest["source_root"])
            data_dir = snap_dir / "data"

            # Whether the source was a directory is read from the manifest, not from
            # the current filesystem. Asking src_root.is_dir() here got it wrong in
            # exactly the case restore exists for: if the directory had since been
            # deleted, is_dir() was False and every file in the manifest was copied
            # over src_root itself in turn, leaving one file where a tree should be.
            source_is_dir = manifest.get("source_is_dir")
            if source_is_dir is None:      # snapshot taken before this was recorded
                source_is_dir = len(manifest["files"]) > 1 or src_root.is_dir()

            targets = [only_file] if only_file else list(manifest["files"].keys())
            restored, missing = [], []
            for rel in targets:
                if rel not in manifest["files"]:
                    missing.append(rel)
                    continue
                backup_file = data_dir / rel
                if not backup_file.exists():
                    missing.append(rel)
                    continue
                dest = (src_root / rel) if source_is_dir else src_root
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_file, dest)
                restored.append(rel)

            if missing and not restored:
                return ToolResult(
                    success=False,
                    error=f"Nothing restored - not in snapshot '{label}': {missing}",
                )
            output = f"Restored {len(restored)} file(s) from snapshot '{label}': {restored}"
            if missing:
                output += f" (skipped, not in the snapshot: {missing})"
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
