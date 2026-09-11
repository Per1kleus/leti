"""What Leti is allowed to do, in words a person can act on.

This is a VIEW. The permissions themselves live where they always have:
config/permissions.yaml says what class of action each tool is, and settings.yaml's
safety.require_confirmation_for says which classes stop and ask. SafetyGuard reads
those two and nothing else, so there is exactly one source of truth and this reads
and writes it rather than keeping its own copy.

What it adds is translation. "delete_file: critical" is precise and means nothing
to anyone who has not read the guard; "Leti must ask before deleting files" is the
same fact. Categories group tools by what a person would call the capability -
Files, Browser, Email, Calendar, Computer, System - and each one reports the
strictest thing in it, because a category that says "allowed automatically" while
one of its tools asks first would be a comfortable lie.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.permissions")

# The classes SafetyGuard already uses, weakest first.
CLASSES = ("read", "execute", "modify", "external", "critical")

ALLOWED = "allowed"            # runs without asking
ASKS = "asks"                  # stops for confirmation
BLOCKED = "blocked"            # refused outright
LEVELS = (ALLOWED, ASKS, BLOCKED)

# Capabilities as a person would name them, mapped onto the tools that implement
# them. Tools are named explicitly rather than by module so a category can span
# modules (Computer covers the GUI primitives AND reading the screen).
CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Files": {
        "blurb": "Reading, writing and deleting files on this computer.",
        "actions": {
            "Read": ["read_file", "list_files"],
            "Create or modify": ["write_file", "move_file"],
            "Delete": ["delete_file"],
            "Back up and restore": ["create_security_snapshot", "restore_from_snapshot",
                                    "check_integrity"],
        },
    },
    "Browser": {
        "blurb": "Searching the web and using pages in a browser.",
        "actions": {
            "Search and read": ["web_search", "research_topic", "browser_read_page",
                                "search_images"],
            "Click and fill forms": ["browser_click", "browser_fill_form"],
            "Sign in to sites": ["login_to_social_platform", "logout_social_platform"],
        },
    },
    "Email": {
        "blurb": "Reading your mail and sending messages as you.",
        "actions": {
            "Read": ["list_new_emails"],
            "Send": ["send_email", "send_meeting_invite_email"],
        },
    },
    "Calendar": {
        "blurb": "Meetings and scheduled work.",
        "actions": {
            "Create meetings": ["schedule_meeting", "create_video_meeting_link"],
            "Scheduled tasks": ["create_scheduled_task", "update_scheduled_task",
                                "delete_scheduled_task", "run_scheduled_task_now"],
        },
    },
    "Computer": {
        "blurb": "Seeing your screen and using the mouse and keyboard.",
        "actions": {
            "Look at the screen": ["read_screen", "choose_computer_approach",
                                   "verify_screen", "end_computer_session"],
            "Use the mouse": ["mouse_click"],
            "Use the keyboard": ["keyboard_type", "keyboard_hotkey"],
            "Open and close applications": ["launch_app", "close_app", "focus_window"],
        },
    },
    "System": {
        "blurb": "Information about this machine, and changing how it runs.",
        "actions": {
            "Read system information": ["system_report", "inspect_network_connections",
                                        "check_firewall_status", "list_lan_devices",
                                        "check_startup_persistence",
                                        "detect_brute_force_attempts"],
            "Run commands": ["run_shell_command", "run_code", "install_dependency"],
            "Change system settings": ["enable_firewall", "apply_system_updates",
                                       "kill_process", "system_scheduling"],
        },
    },
}


def _permissions() -> Dict[str, Any]:
    from core.config_loader import get_permissions

    return get_permissions()


def _settings() -> Dict[str, Any]:
    from core.config_loader import get_settings

    return get_settings()


def confirming_classes() -> List[str]:
    """Which action classes stop and ask. SafetyGuard's own setting, read live."""
    configured = _settings().get("safety", {}).get("require_confirmation_for", [])
    # critical always asks whether or not it is listed - the guard enforces this,
    # and showing it any other way here would misrepresent what will happen.
    return sorted(set(list(configured) + ["critical"]), key=CLASSES.index)


def tool_class(tool_name: str) -> Optional[str]:
    entry = _permissions().get("tools", {}).get(tool_name)
    return entry.get("action") if isinstance(entry, dict) else None


def level_for(tool_name: str) -> str:
    """What will actually happen when Leti tries this tool."""
    action = tool_class(tool_name)
    if action is None:
        # Unclassified is treated as critical by the guard, so it asks.
        return ASKS
    if action in confirming_classes():
        return ASKS
    return ALLOWED


def explain(category: str, action_name: str, level: str) -> str:
    """One sentence a person can act on, rather than a class name."""
    verb = {
        "Read": "read files", "Create or modify": "create and change files",
        "Delete": "delete files", "Back up and restore": "back up and restore files",
        "Search and read": "search the web and read pages",
        "Click and fill forms": "click and fill in forms on pages",
        "Sign in to sites": "sign in to sites on your behalf",
        "Send": "send email as you",
        "Create meetings": "create meetings", "Scheduled tasks": "schedule its own work",
        "Look at the screen": "look at your screen",
        "Use the mouse": "move and click the mouse",
        "Use the keyboard": "type and press keys",
        "Open and close applications": "open and close applications",
        "Read system information": "read information about this machine",
        "Run commands": "run commands on this machine",
        "Change system settings": "change how this machine is set up",
    }.get(action_name, action_name.lower())
    if category == "Email" and action_name == "Read":
        verb = "read your mail"
    if level == ALLOWED:
        return f"Leti can {verb} without asking."
    if level == BLOCKED:
        return f"Leti cannot {verb}."
    return f"Leti asks before it will {verb}."


def overview(registry: Any = None) -> Dict[str, Any]:
    """Every category, what it covers, and what Leti may currently do.

    A category reports the STRICTEST level among its actions: saying a whole
    capability is automatic when part of it asks would be the wrong way round.
    """
    entries = _permissions().get("tools", {})
    categories = []
    for name, spec in CATEGORIES.items():
        actions = []
        for action_name, tool_names in spec["actions"].items():
            present = [t for t in tool_names
                       if t in entries and (registry is None or registry.get(t) is not None)]
            if not present:
                continue
            level = ASKS if any(level_for(t) == ASKS for t in present) else ALLOWED
            actions.append({
                "action": action_name,
                "level": level,
                "explanation": explain(name, action_name, level),
                "tools": sorted(present),
                "classes": sorted({tool_class(t) for t in present if tool_class(t)}),
            })
        if actions:
            categories.append({
                "category": name,
                "blurb": spec["blurb"],
                "level": ASKS if any(a["level"] == ASKS for a in actions) else ALLOWED,
                "actions": actions,
            })

    covered = {t for spec in CATEGORIES.values()
               for names in spec["actions"].values() for t in names}
    return {
        "categories": categories,
        "confirming_classes": confirming_classes(),
        "all_classes": list(CLASSES),
        "dry_run": bool(_settings().get("safety", {}).get("dry_run", False)),
        "uncategorised_tools": sorted(set(entries) - covered),
        "note": ("These are the same settings SafetyGuard enforces - "
                 "config/permissions.yaml and safety.require_confirmation_for. "
                 "Changing one here changes what Leti will actually do."),
    }


def set_class_confirmation(action_class: str, must_confirm: bool) -> Dict[str, Any]:
    """Make a whole class of action ask first, or stop asking.

    Writes safety.require_confirmation_for - the setting SafetyGuard already reads -
    through the same override file the settings editor uses, so settings.yaml keeps
    its comments and there is still one source of truth.
    """
    action_class = str(action_class or "").strip().lower()
    if action_class not in CLASSES:
        return {"ok": False, "error": f"'{action_class}' is not one of: {', '.join(CLASSES)}."}
    if action_class == "critical" and not must_confirm:
        # The guard ignores this anyway; saying so is better than appearing to obey.
        return {"ok": False,
                "error": ("Irreversible actions always ask. A setting that could switch "
                          "that off would defeat the point of having it.")}

    from core.config_loader import reload_settings
    from core.settings_editor import _deep_set, _load_overrides, _save_overrides

    current = [c for c in _settings().get("safety", {}).get("require_confirmation_for", [])]
    if must_confirm and action_class not in current:
        current.append(action_class)
    elif not must_confirm and action_class in current:
        current = [c for c in current if c != action_class]

    ordered = sorted(set(current), key=CLASSES.index)
    overrides = _load_overrides()
    _deep_set(overrides, ["safety"], {"require_confirmation_for": ordered})
    _save_overrides(overrides)
    reload_settings()
    return {"ok": True, "confirming_classes": confirming_classes()}


def set_tool_class(tool_name: str, action_class: str, registry: Any = None) -> Dict[str, Any]:
    """Reclassify one tool - the other half of what SafetyGuard reads.

    Writes config/permissions.yaml itself, because that file IS the classification;
    keeping a second copy somewhere else is exactly the thing this must not do.
    """
    action_class = str(action_class or "").strip().lower()
    if action_class not in CLASSES:
        return {"ok": False, "error": f"'{action_class}' is not one of: {', '.join(CLASSES)}."}
    if registry is not None and registry.get(tool_name) is None:
        return {"ok": False, "error": f"'{tool_name}' is not a registered tool."}

    from pathlib import Path

    import yaml

    from core.atomic_write import atomic_write_text
    # get_permissions() reads the file on every call rather than caching it,
    # so writing it is all that is needed for the change to take effect.
    from core.config_loader import CONFIG_DIR

    path = Path(CONFIG_DIR) / "permissions.yaml"
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as e:
        return {"ok": False, "error": f"Couldn't read permissions.yaml: {e}"}

    tools = data.setdefault("tools", {})
    if tool_name not in tools:
        return {"ok": False, "error": f"'{tool_name}' has no entry in permissions.yaml."}
    previous = tools[tool_name].get("action")
    tools[tool_name]["action"] = action_class

    try:
        atomic_write_text(path, yaml.safe_dump(data, sort_keys=False))
    except Exception as e:
        return {"ok": False, "error": f"Couldn't save permissions.yaml: {e}"}

    return {"ok": True, "tool": tool_name, "was": previous, "now": action_class,
            "level": level_for(tool_name)}
