"""
Structured, editable memory of who the user is - a sibling to, not a
replacement for, the free-form semantic vector memory in
memory/vector_store.py. Facts stored here are always injected into the
system prompt (see profile_summary(), used by core/orchestrator.py), not
just recalled when a similarity search happens to match - so something
like the user's name or a stated preference is reliably present every
turn, not only when the current topic happens to be related.

Storage is a single human-editable JSON file (data/user_profile.json),
same shape as tools/contacts.py, so it's easy to inspect or hand-edit.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from core.config_loader import get_settings, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

CATEGORIES = ["preference", "fact", "interest", "dislike", "goal", "other"]


def _path() -> Path:
    cfg = get_settings().get("user_profile", {})
    p = resolve_path(cfg.get("file_path", "./data/user_profile.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load() -> Dict[str, Any]:
    p = _path()
    if not p.exists():
        return {"name": "", "facts": []}
    try:
        data = json.loads(p.read_text())
        data.setdefault("name", "")
        data.setdefault("facts", [])
        return data
    except json.JSONDecodeError:
        return {"name": "", "facts": []}


def _save(data: Dict[str, Any]) -> None:
    _path().write_text(json.dumps(data, indent=2))


def profile_summary(max_facts: int = 12) -> str:
    """Compact, system-prompt-ready summary of what's known about the user so far.
    Returns "" when there's nothing yet, so callers can skip the message entirely."""
    data = _load()
    lines = []
    if data.get("name"):
        lines.append(f"Name: {data['name']}")
    facts = data.get("facts", [])[-max_facts:]
    if facts:
        lines.append("Known about the user:")
        for f in facts:
            lines.append(f"- {f['text']} ({f['category']})")
    return "\n".join(lines)


class RememberAboutUserTool(BaseTool):
    name = "remember_about_user"
    description = (
        "Save a fact/preference about the user for future sessions - name, likes/dislikes, "
        "interests, goals, communication style, anything durable. Call this whenever the user "
        "shares something worth remembering long-term - don't wait to be asked, this is "
        "low-stakes and reviewable/deletable any time via view_user_profile. Skip throwaway "
        "details relevant only to the current task."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="fact", type="string", description="The fact, written plainly, e.g. 'prefers dark humor' or 'works as a nurse'."),
        ToolParameter(name="category", type="string", enum=CATEGORIES, description="Best-fit category."),
    ]

    async def run(self, fact: str, category: str = "other", **kwargs) -> ToolResult:
        try:
            data = _load()
            entry = {
                "id": uuid.uuid4().hex[:8],
                "text": fact,
                "category": category if category in CATEGORIES else "other",
                "learned_at": time.time(),
            }
            data["facts"].append(entry)
            _save(data)
            return ToolResult(success=True, output=f"Remembered: {fact}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SetUserNameTool(BaseTool):
    name = "set_user_name"
    description = "Save the user's preferred name so Leti can address them by it."
    parameters: List[ToolParameter] = [
        ToolParameter(name="name", type="string", description="The name to use."),
    ]

    async def run(self, name: str, **kwargs) -> ToolResult:
        try:
            data = _load()
            data["name"] = name
            _save(data)
            return ToolResult(success=True, output=f"I'll call you {name}.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ViewUserProfileTool(BaseTool):
    name = "view_user_profile"
    description = "Show everything Leti currently remembers about the user (name + saved facts, with their ids)."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=_load())


class ForgetUserFactTool(BaseTool):
    name = "forget_user_fact"
    description = "Delete one specific remembered fact about the user by its id (get ids from view_user_profile)."
    parameters: List[ToolParameter] = [
        ToolParameter(name="fact_id", type="string", description="The fact's id."),
    ]

    async def run(self, fact_id: str, **kwargs) -> ToolResult:
        try:
            data = _load()
            before = len(data["facts"])
            data["facts"] = [f for f in data["facts"] if f["id"] != fact_id]
            if len(data["facts"]) == before:
                return ToolResult(success=False, error=f"No fact with id '{fact_id}'.")
            _save(data)
            return ToolResult(success=True, output=f"Forgot fact {fact_id}.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ClearUserProfileTool(BaseTool):
    name = "clear_user_profile"
    description = "Erase everything Leti remembers about the user (name + all facts). Destructive, cannot be undone."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        _save({"name": "", "facts": []})
        return ToolResult(success=True, output="Cleared the user profile.")
