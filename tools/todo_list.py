"""
A simple to-do list, storage-shared between the GUI dashboard's direct
checkbox interactions (via gui/api.py) and conversational requests
("add X to my to-do list") via the tool classes below. Same JSON-file
pattern as tools/contacts.py - one source of truth either way.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from core.config_loader import get_settings, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult


def _path() -> Path:
    cfg = get_settings().get("todo_list", {})
    p = resolve_path(cfg.get("file_path", "./data/todo_list.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_todos() -> List[Dict[str, Any]]:
    p = _path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return []


def save_todos(todos: List[Dict[str, Any]]) -> None:
    _path().write_text(json.dumps(todos, indent=2))


def add_todo(text: str) -> Dict[str, Any]:
    todos = load_todos()
    item = {"id": uuid.uuid4().hex[:8], "text": text, "done": False, "created_at": time.time()}
    todos.append(item)
    save_todos(todos)
    return item


def toggle_todo(item_id: str) -> bool:
    todos = load_todos()
    for t in todos:
        if t["id"] == item_id:
            t["done"] = not t["done"]
            save_todos(todos)
            return True
    return False


def delete_todo(item_id: str) -> bool:
    todos = load_todos()
    remaining = [t for t in todos if t["id"] != item_id]
    if len(remaining) == len(todos):
        return False
    save_todos(remaining)
    return True


class AddTodoItemTool(BaseTool):
    name = "add_todo_item"
    description = "Add an item to the user's to-do list."
    parameters: List[ToolParameter] = [
        ToolParameter(name="text", type="string", description="The task text."),
    ]

    async def run(self, text: str, **kwargs) -> ToolResult:
        item = add_todo(text)
        return ToolResult(success=True, output=f"Added to your to-do list: {text}")


class ListTodoItemsTool(BaseTool):
    name = "list_todo_items"
    description = "List the user's current to-do items."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={"items": load_todos()})


class CompleteTodoItemTool(BaseTool):
    name = "complete_todo_item"
    description = "Mark a to-do item as done (or not done, toggling it) by its id (from list_todo_items)."
    parameters: List[ToolParameter] = [
        ToolParameter(name="item_id", type="string", description="The item's id."),
    ]

    async def run(self, item_id: str, **kwargs) -> ToolResult:
        if not toggle_todo(item_id):
            return ToolResult(success=False, error=f"No to-do item with id '{item_id}'.")
        return ToolResult(success=True, output=f"Toggled item {item_id}.")


class DeleteTodoItemTool(BaseTool):
    name = "delete_todo_item"
    description = "Remove an item from the to-do list by its id."
    parameters: List[ToolParameter] = [
        ToolParameter(name="item_id", type="string", description="The item's id."),
    ]

    async def run(self, item_id: str, **kwargs) -> ToolResult:
        if not delete_todo(item_id):
            return ToolResult(success=False, error=f"No to-do item with id '{item_id}'.")
        return ToolResult(success=True, output=f"Removed item {item_id}.")
