"""Persistent project workspaces.

A project is a folder plus a small amount of metadata. Everything a project
"contains" - files, code, data, reports - is just files in that folder, so the
existing file tools, coding tools and data tools all work on a project without
knowing projects exist. What this module adds is the part those tools can't
supply on their own: a stable home on disk, instructions that follow the project
into every conversation about it, and a notion of which project is currently
active so "continue the MATLAB project" resolves to something concrete.

Projects nest, which is what makes "University / MATLAB" and "Business /
Clients" work as the user described them. A project is addressed by its path:
"University/MATLAB". Nesting is a real directory hierarchy rather than a
parent-id field, so a project's folder is the obvious place its files live and
the user can open it in a file manager.

What deliberately isn't here: conversation history has a home already
(memory/session_memory.py) and tasks have one (tools/todo_list.py and
tools/scheduler.py). Those are referenced by project, not reimplemented -
see the `project` argument on the scheduler's tools and get_project_context
below, which gathers the state a project has rather than storing its own copy.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.atomic_write import atomic_write_json
from core.config_loader import get_settings, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

MANIFEST_NAME = ".leti-project.json"
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")


def projects_root() -> Path:
    cfg = get_settings().get("projects", {})
    root = resolve_path(cfg.get("root", "./data/projects"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _state_path() -> Path:
    """Where the active-project pointer lives - outside the projects tree, so
    deleting a project can't take the pointer's file with it."""
    return resolve_path("./data/active_project.json")


def safe_project_path(name: str) -> List[str]:
    """Split "University/MATLAB" into validated path segments.

    Each segment is checked rather than the whole string: the name is joined onto
    the projects root to build a real directory, so a segment of '..' would walk
    out of it, and create/delete both act on whatever that resolves to.
    """
    raw = (name or "").strip().strip("/")
    if not raw:
        raise ValueError("A project name is required.")
    segments = [seg.strip() for seg in raw.split("/") if seg.strip()]
    if not segments:
        raise ValueError("A project name is required.")
    if len(segments) > 4:
        raise ValueError("Projects can nest at most four levels deep.")
    for seg in segments:
        if not _SAFE_SEGMENT.match(seg):
            raise ValueError(
                f"Invalid project name segment {seg!r}: use letters, numbers, spaces, "
                f"dots, dashes and underscores only."
            )
    return segments


def project_dir(name: str) -> Path:
    return projects_root().joinpath(*safe_project_path(name))


def project_exists(name: str) -> bool:
    try:
        return (project_dir(name) / MANIFEST_NAME).is_file()
    except ValueError:
        return False


def load_manifest(name: str) -> Dict[str, Any]:
    path = project_dir(name) / MANIFEST_NAME
    if not path.is_file():
        raise KeyError(f"No project named '{name}'. Use list_projects to see what exists.")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # A corrupt manifest shouldn't make the project's files unreachable.
        return {"name": name, "instructions": "", "created_at": None, "corrupt_manifest": True}


def save_manifest(name: str, manifest: Dict[str, Any]) -> None:
    atomic_write_json(project_dir(name) / MANIFEST_NAME, manifest)


def own_files(directory: Path) -> List[Path]:
    """Files belonging to this project, excluding its manifest and anything owned
    by a nested project.

    Without the nesting check a parent reports its children's files as its own -
    "University" claiming the two files that actually belong to "University/MATLAB".
    """
    nested_roots = [m.parent for m in directory.rglob(MANIFEST_NAME) if m.parent != directory]
    files = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        if any(root in path.parents for root in nested_roots):
            continue
        files.append(path)
    return files


def set_archived(name: str, archived: bool) -> Dict[str, Any]:
    """Archive or restore a project.

    Archiving is not deleting: the folder and everything in it stay exactly where
    they are, the project simply stops showing up in the everyday list and stops
    being offered as context. Finished work that might still be needed should not
    have to be thrown away to get out of the way.
    """
    manifest = load_manifest(name)
    manifest["archived"] = bool(archived)
    save_manifest(name, manifest)
    if archived and get_active_project() == name:
        set_active_project(None)
    return manifest


def list_projects(include_archived: bool = False) -> List[Dict[str, Any]]:
    """Every project, as "Parent/Child" paths, shallowest first."""
    root = projects_root()
    found = []
    for manifest_path in sorted(root.rglob(MANIFEST_NAME)):
        rel = manifest_path.parent.relative_to(root)
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
        archived = bool(manifest.get("archived"))
        if archived and not include_archived:
            continue
        found.append({
            "name": str(rel).replace("\\", "/"),
            "description": manifest.get("description", ""),
            "updated_at": manifest.get("updated_at"),
            "archived": archived,
            "file_count": len(own_files(manifest_path.parent)),
        })
    return sorted(found, key=lambda p: (p["name"].count("/"), p["name"].lower()))


def get_active_project() -> Optional[str]:
    path = _state_path()
    if not path.is_file():
        return None
    try:
        name = json.loads(path.read_text()).get("active")
    except json.JSONDecodeError:
        return None
    # A project can be deleted while it's active; don't keep pointing at it.
    return name if name and project_exists(name) else None


def set_active_project(name: Optional[str]) -> None:
    atomic_write_json(_state_path(), {"active": name, "set_at": time.time()})


def resolve_project(name: Optional[str]) -> Optional[str]:
    """The project a tool call should work in: the one named, else the active one.

    Every capability that can belong to a project takes an optional `project`
    argument and runs it through here, so "analyse this dataset" after "work on
    the MATLAB project" lands in the right place without the user repeating it.
    """
    if name:
        return name
    return get_active_project()


def project_context(name: str, max_files: int = 40, request: str = "") -> str:
    """A project's instructions and contents, as text for the system prompt.

    This is what makes "continue the MATLAB project" mean something: the model
    sees the project's own instructions and what's actually in its folder, rather
    than being told a name and left to guess.
    """
    try:
        manifest = load_manifest(name)
    except KeyError:
        return ""

    directory = project_dir(name)
    lines = [f"Active project: {name}"]
    if manifest.get("description"):
        lines.append(f"Description: {manifest['description']}")
    lines.append(f"Folder: {directory}")

    if manifest.get("instructions"):
        lines.append("\nProject instructions (follow these for work in this project):")
        lines.append(manifest["instructions"])

    files = sorted(own_files(directory), key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        # Naming every file on every turn is how project context turns into the
        # thing dynamic tool routing exists to avoid. When the request mentions
        # particular files those come first and the list is short; otherwise the
        # most recently touched ones stand in for "what we were doing".
        shown, budget = _relevant_files(files, directory, request, max_files)
        lines.append(f"\nFiles in this project ({len(files)} total"
                     + (", most relevant to this request first):" if request
                        else ", most recently changed first):"))
        for path in shown:
            size = path.stat().st_size
            lines.append(f"- {path.relative_to(directory)} ({size:,} bytes)")
        if len(files) > len(shown):
            lines.append(f"- ... and {len(files) - len(shown)} more "
                         "(ask for a listing if you need them)")
    else:
        lines.append("\nThis project has no files yet.")
    return "\n".join(lines)


def _relevant_files(files, directory, request: str, max_files: int):
    """The files worth naming for this request, and how many that is.

    References, never contents: a project may point at large files and copying
    them into the prompt would be the opposite of useful.
    """
    import re as _re

    if not request:
        return files[:max_files], max_files
    words = {w for w in _re.findall(r"[a-z0-9_.]+", request.lower()) if len(w) > 2}
    if not words:
        return files[:12], 12
    named, others = [], []
    for path in files:
        text = str(path.relative_to(directory)).lower()
        (named if any(w in text for w in words) else others).append(path)
    # A short recent tail keeps "carry on where we left off" working even when the
    # request names nothing.
    return (named + others[:6])[:max_files], max_files


# --- Tools ---------------------------------------------------------------------

class CreateProjectTool(BaseTool):
    name = "create_project"
    description = (
        "Create a persistent project workspace. Use a path to nest one inside another, "
        "e.g. 'University/MATLAB' or 'Business/Clients'. Project instructions are "
        "included in your context every time this project is active, so put standing "
        "requirements there ('always use SI units', 'this codebase is TypeScript')."
    )
    parameters = [
        ToolParameter(name="name", type="string", description="Project name or path, e.g. 'University/MATLAB'."),
        ToolParameter(name="description", type="string", required=False, description="One line on what this project is."),
        ToolParameter(
            name="instructions", type="string", required=False,
            description="Standing instructions to follow whenever working in this project.",
        ),
        ToolParameter(
            name="make_active", type="boolean", required=False,
            description="Switch to this project immediately (default true).",
        ),
    ]

    async def run(self, name: str, description: str = "", instructions: str = "",
                  make_active: bool = True, **kwargs) -> ToolResult:
        try:
            segments = safe_project_path(name)
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

        canonical = "/".join(segments)
        if project_exists(canonical):
            return ToolResult(success=False, error=f"Project '{canonical}' already exists.")

        directory = project_dir(canonical)
        directory.mkdir(parents=True, exist_ok=True)
        now = time.time()
        save_manifest(canonical, {
            "name": canonical,
            "description": description,
            "instructions": instructions,
            "created_at": now,
            "updated_at": now,
        })
        if make_active:
            set_active_project(canonical)

        return ToolResult(success=True, output={
            "project": canonical,
            "folder": str(directory),
            "active": make_active,
            "summary": f"Created project '{canonical}'" + (" and made it active." if make_active else "."),
        })


class ListProjectsTool(BaseTool):
    name = "list_projects"
    description = "List every project workspace, nested paths included, and say which one is active."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={
            "projects": list_projects(),
            "active": get_active_project(),
        })


class OpenProjectTool(BaseTool):
    name = "open_project"
    description = (
        "Switch to a project, so its instructions and files are in context and later "
        "work lands in its folder by default. Use this when the user refers to an "
        "existing project - 'continue the MATLAB project'."
    )
    parameters = [
        ToolParameter(name="name", type="string", description="Project name or path."),
    ]

    async def run(self, name: str, **kwargs) -> ToolResult:
        try:
            canonical = "/".join(safe_project_path(name))
        except ValueError as e:
            return ToolResult(success=False, error=str(e))

        if not project_exists(canonical):
            existing = [p["name"] for p in list_projects()]
            # Partial match: "MATLAB" should find "University/MATLAB" rather than
            # failing on a name the user reasonably shortened.
            matches = [p for p in existing if p.lower().endswith("/" + canonical.lower())
                       or p.lower() == canonical.lower()]
            if len(matches) == 1:
                canonical = matches[0]
            elif len(matches) > 1:
                return ToolResult(success=False, error=(
                    f"'{name}' matches several projects: {matches}. Say which one."
                ))
            else:
                return ToolResult(success=False, error=(
                    f"No project named '{name}'. Existing projects: {existing or 'none yet'}."
                ))

        set_active_project(canonical)
        return ToolResult(success=True, output={
            "project": canonical,
            "context": project_context(canonical),
        })


class CloseProjectTool(BaseTool):
    name = "close_project"
    description = "Stop working in the current project. Its files are untouched."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        previous = get_active_project()
        set_active_project(None)
        return ToolResult(success=True, output=(
            f"Closed project '{previous}'." if previous else "No project was active."
        ))


class UpdateProjectTool(BaseTool):
    name = "update_project"
    description = (
        "Change a project's description or standing instructions. Instructions are "
        "replaced, not appended - read them first if you're adding to them."
    )
    parameters = [
        ToolParameter(name="name", type="string", required=False,
                      description="Project to update. Defaults to the active project."),
        ToolParameter(name="description", type="string", required=False, description="New description."),
        ToolParameter(name="instructions", type="string", required=False, description="New standing instructions."),
    ]

    async def run(self, name: str = "", description: Optional[str] = None,
                  instructions: Optional[str] = None, **kwargs) -> ToolResult:
        target = resolve_project(name)
        if not target:
            return ToolResult(success=False, error="No project named and none is active.")
        try:
            manifest = load_manifest(target)
        except KeyError as e:
            return ToolResult(success=False, error=str(e))

        if description is not None:
            manifest["description"] = description
        if instructions is not None:
            manifest["instructions"] = instructions
        manifest["updated_at"] = time.time()
        save_manifest(target, manifest)
        return ToolResult(success=True, output=f"Updated project '{target}'.")


class GetProjectContextTool(BaseTool):
    name = "get_project_context"
    description = (
        "Show what a project contains: its instructions, its folder, and the files in "
        "it. Use this when you need to know what's already in a project before working "
        "on it."
    )
    parameters = [
        ToolParameter(name="name", type="string", required=False,
                      description="Project to describe. Defaults to the active project."),
    ]

    async def run(self, name: str = "", **kwargs) -> ToolResult:
        target = resolve_project(name)
        if not target:
            return ToolResult(success=False, error="No project named and none is active.")
        if not project_exists(target):
            return ToolResult(success=False, error=f"No project named '{target}'.")
        return ToolResult(success=True, output={
            "project": target,
            "folder": str(project_dir(target)),
            "context": project_context(target),
        })


class ArchiveProjectTool(BaseTool):
    name = "archive_project"
    description = (
        "Put a finished project out of the way, or bring one back. Archiving keeps every "
        "file exactly where it is and only stops the project appearing in the everyday "
        "list and being offered as context - use it instead of delete_project when the "
        "work is done but might still be wanted."
    )
    parameters = [
        ToolParameter(name="name", type="string", description="The project."),
        ToolParameter(name="restore", type="boolean", required=False,
                      description="True to bring an archived project back."),
    ]

    async def run(self, name: str, restore: bool = False, **kwargs) -> ToolResult:
        resolved = resolve_project(name)
        if resolved is None:
            return ToolResult(success=False, error=f"No project called '{name}'.")
        try:
            set_archived(resolved, not restore)
        except KeyError:
            return ToolResult(success=False, error=f"No project called '{name}'.")
        return ToolResult(success=True, output={
            "project": resolved,
            "archived": not restore,
            "note": (f"'{resolved}' is back in the list." if restore else
                     f"'{resolved}' is archived. Nothing was deleted."),
        })


class DeleteProjectTool(BaseTool):
    name = "delete_project"
    description = (
        "Permanently delete a project and every file in it. Irreversible - the files "
        "are removed from disk, not moved to a recycle bin."
    )
    parameters = [
        ToolParameter(name="name", type="string", description="Project to delete."),
        ToolParameter(
            name="confirm_name", type="string",
            description="The project's name again, exactly, to confirm the right one is going.",
        ),
    ]

    async def run(self, name: str, confirm_name: str = "", **kwargs) -> ToolResult:
        import shutil

        try:
            canonical = "/".join(safe_project_path(name))
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        if not project_exists(canonical):
            return ToolResult(success=False, error=f"No project named '{canonical}'.")
        # Deleting the wrong project is unrecoverable, and the model picks the
        # argument. Naming it twice makes a mistaken pick fail instead of delete.
        if confirm_name.strip() != canonical:
            return ToolResult(success=False, error=(
                f"confirm_name must repeat the project name exactly ('{canonical}') - "
                f"got '{confirm_name}'. Nothing was deleted."
            ))

        directory = project_dir(canonical)
        # Report what the user loses: their files, and any nested projects that go
        # with the folder. The manifest isn't one of their files.
        file_count = len(own_files(directory))
        nested = [p["name"] for p in list_projects()
                  if p["name"].startswith(canonical + "/")]
        shutil.rmtree(directory)
        if get_active_project() == canonical:
            set_active_project(None)
        summary = f"Deleted project '{canonical}' and {file_count} file(s)."
        if nested:
            summary += f" This also removed nested projects: {', '.join(nested)}."
        return ToolResult(success=True, output=summary)
