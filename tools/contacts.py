"""
Contact book: lets Leti resolve a name mentioned in conversation ("email
stavros about the mechanic news") to the right saved contact, even when
multiple contacts share that first name, by matching the surrounding
context (a task description, a topic, whatever prompted the lookup)
against each candidate's tags/notes.

Storage is a single human-editable JSON file (data/contacts.json by
default) - no DB needed for what's realistically a few dozen/hundred
entries, and it's easy for the user to hand-edit if they want to bulk-add
contacts.

Resolution logic (see resolve_contact):
  1. Find every contact whose name or alias matches the given name
     (case-insensitive; first-name-only queries match against the first
     token of the full name too).
  2. Zero matches -> not_found.
  3. Exactly one match -> resolved immediately (no ambiguity to break).
  4. Multiple matches -> score each candidate against the supplied context
     string using its tags/notes/company (simple keyword containment).
     If exactly one candidate has a strictly-higher score than every
     other, it's resolved via context. Otherwise it comes back
     "ambiguous" with the full candidate list so the caller (the LLM, per
     the system prompt, or the user directly) can pick one - Leti should
     never silently guess between two same-scored people.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.atomic_write import atomic_write_json
from core.config_loader import get_settings, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

DEFAULT_CONTACTS_PATH = "./data/contacts.json"


@dataclass
class Contact:
    id: str
    name: str
    email: str = ""
    phone: str = ""
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)      # e.g. ["mechanic", "work"]
    company: str = ""
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def _contacts_path() -> Path:
    cfg = get_settings().get("contacts", {})
    path = resolve_path(cfg.get("file_path", DEFAULT_CONTACTS_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load() -> List[Dict[str, Any]]:
    path = _contacts_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def _save(contacts: List[Dict[str, Any]]) -> None:
    atomic_write_json(_contacts_path(), contacts)


def _matches_name(contact: Dict[str, Any], query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    full_name = contact.get("name", "").lower()
    if q == full_name:
        return True
    if q in [a.lower() for a in contact.get("aliases", [])]:
        return True
    # First-name-only match: "stavros" should match "Stavros Papadakis"
    first_token = full_name.split()[0] if full_name.split() else ""
    if q == first_token:
        return True
    return False


def _context_score(contact: Dict[str, Any], context: str) -> int:
    if not context:
        return 0
    ctx = context.lower()
    haystack_fields = contact.get("tags", []) + [contact.get("notes", ""), contact.get("company", "")]
    score = 0
    for field_val in haystack_fields:
        if not field_val:
            continue
        for token in re.findall(r"[a-zA-Z]{3,}", str(field_val).lower()):
            if token in ctx:
                score += 1
    return score


def _redact(contact: Dict[str, Any]) -> Dict[str, Any]:
    """Candidate list previews shouldn't leak full contact details into a disambiguation
    prompt unnecessarily - name/tags/company are enough to tell people apart."""
    return {
        "id": contact["id"],
        "name": contact["name"],
        "tags": contact.get("tags", []),
        "company": contact.get("company", ""),
    }


class AddContactTool(BaseTool):
    name = "add_contact"
    description = "Add a new contact to the contact book."
    parameters: List[ToolParameter] = [
        ToolParameter(name="name", type="string", description="Full name, e.g. 'Stavros Papadakis'."),
        ToolParameter(name="email", type="string", required=False, description="Email address."),
        ToolParameter(name="phone", type="string", required=False, description="Phone number."),
        ToolParameter(
            name="tags", type="array", items_type="string", required=False,
            description="Keywords that describe who they are, e.g. ['mechanic','car repair'].",
        ),
        ToolParameter(name="company", type="string", required=False, description="Company/organization."),
        ToolParameter(name="notes", type="string", required=False, description="Free-text notes."),
        ToolParameter(
            name="aliases", type="array", items_type="string", required=False,
            description="Other names/nicknames this person is referred to by.",
        ),
    ]

    async def run(
        self, name: str, email: str = "", phone: str = "", tags: Optional[List[str]] = None,
        company: str = "", notes: str = "", aliases: Optional[List[str]] = None, **kwargs
    ) -> ToolResult:
        try:
            contacts = _load()
            contact = Contact(
                id=uuid.uuid4().hex[:8], name=name, email=email, phone=phone,
                tags=tags or [], company=company, notes=notes, aliases=aliases or [],
            )
            contacts.append(asdict(contact))
            _save(contacts)
            return ToolResult(success=True, output=f"Added contact '{name}' (id={contact.id}).")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ListContactsTool(BaseTool):
    name = "list_contacts"
    description = "List all saved contacts, optionally filtered by a name/tag substring."
    parameters: List[ToolParameter] = [
        ToolParameter(name="filter_text", type="string", required=False, description="Substring to filter by."),
    ]

    async def run(self, filter_text: str = "", **kwargs) -> ToolResult:
        try:
            contacts = _load()
            if filter_text:
                ft = filter_text.lower()
                contacts = [
                    c for c in contacts
                    if ft in c.get("name", "").lower()
                    or any(ft in t.lower() for t in c.get("tags", []))
                    or ft in c.get("company", "").lower()
                ]
            return ToolResult(success=True, output={"contacts": contacts, "count": len(contacts)})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ResolveContactTool(BaseTool):
    name = "resolve_contact"
    description = (
        "Look up a contact by the name mentioned in conversation and disambiguate between "
        "multiple people who share that name using surrounding context (e.g. the topic of the "
        "message). ALWAYS call this before emailing/messaging someone by name - never guess an "
        "email address. If the result is 'ambiguous', ask the user which person they meant "
        "instead of picking one yourself."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="name", type="string", description="The name as mentioned, e.g. 'stavros'."),
        ToolParameter(
            name="context", type="string", required=False,
            description="What the message/task is about, used to disambiguate same-named contacts "
                        "(e.g. 'the new mechanic news' -> matches a contact tagged 'mechanic').",
        ),
    ]

    async def run(self, name: str, context: str = "", **kwargs) -> ToolResult:
        try:
            contacts = _load()
            candidates = [c for c in contacts if _matches_name(c, name)]

            if not candidates:
                return ToolResult(success=True, output={"status": "not_found", "query": name})

            if len(candidates) == 1:
                return ToolResult(success=True, output={"status": "resolved", "method": "unique_name", "contact": candidates[0]})

            scored = [(c, _context_score(c, context)) for c in candidates]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            top_score = scored[0][1]
            top_matches = [c for c, s in scored if s == top_score]

            if top_score > 0 and len(top_matches) == 1:
                return ToolResult(success=True, output={"status": "resolved", "method": "context_matched", "contact": top_matches[0]})

            return ToolResult(success=True, output={
                "status": "ambiguous",
                "query": name,
                "candidates": [_redact(c) for c in candidates],
                "note": "Multiple contacts share this name and context didn't clearly point to one. Ask the user to clarify.",
            })
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class UpdateContactTool(BaseTool):
    name = "update_contact"
    description = "Update fields on an existing contact by id."
    parameters: List[ToolParameter] = [
        ToolParameter(name="contact_id", type="string", description="The contact's id."),
        ToolParameter(name="name", type="string", required=False, description="New name."),
        ToolParameter(name="email", type="string", required=False, description="New email."),
        ToolParameter(name="phone", type="string", required=False, description="New phone."),
        ToolParameter(name="tags", type="array", items_type="string", required=False, description="Replace tags."),
        ToolParameter(name="company", type="string", required=False, description="New company."),
        ToolParameter(name="notes", type="string", required=False, description="New notes."),
    ]

    async def run(self, contact_id: str, **kwargs) -> ToolResult:
        try:
            contacts = _load()
            for c in contacts:
                if c["id"] == contact_id:
                    for field_name in ("name", "email", "phone", "tags", "company", "notes"):
                        if field_name in kwargs and kwargs[field_name] is not None:
                            c[field_name] = kwargs[field_name]
                    c["updated_at"] = time.time()
                    _save(contacts)
                    return ToolResult(success=True, output=f"Updated contact {contact_id}.")
            return ToolResult(success=False, error=f"No contact with id '{contact_id}'.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class DeleteContactTool(BaseTool):
    name = "delete_contact"
    description = "Delete a contact by id. Destructive."
    parameters: List[ToolParameter] = [
        ToolParameter(name="contact_id", type="string", description="The contact's id."),
    ]

    async def run(self, contact_id: str, **kwargs) -> ToolResult:
        try:
            contacts = _load()
            remaining = [c for c in contacts if c["id"] != contact_id]
            if len(remaining) == len(contacts):
                return ToolResult(success=False, error=f"No contact with id '{contact_id}'.")
            _save(remaining)
            return ToolResult(success=True, output=f"Deleted contact {contact_id}.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
