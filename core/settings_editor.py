"""
The /settings command: lets the user configure integrations (email,
calendar, Zoom, weather, etc.) without manually opening settings.yaml and
finding the right line. Deliberately NOT an LLM tool - editing config,
especially anything with a password/API key, is a deterministic,
structured task that doesn't benefit from going through the model (and
shouldn't risk the model mangling a value or explaining rather than
acting). main.py's text-mode loop and gui/api.py's websocket handler both
call straight into this module.

Values are written to config/settings.local.yaml (see core/config_loader.py)
rather than settings.yaml itself, so the heavily-commented, example-filled
main file never gets rewritten or risks losing its comments - the override
file is the one and only thing this module ever touches.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml

from core.config_loader import CONFIG_DIR, get_settings, reload_settings

# Each entry: settings_path locates the section in the merged settings dict
# (defaults to [name] if not given). Fields marked secret=True are never
# echoed back with their real value once set - only "(currently set)".
SECTION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "email": {
        "label": "Email (IMAP/SMTP)",
        "fields": [
            {"key": "imap_host", "label": "IMAP host", "example": "imap.gmail.com"},
            {"key": "imap_port", "label": "IMAP port", "type": "number", "default": 993},
            {"key": "smtp_host", "label": "SMTP host", "example": "smtp.gmail.com"},
            {"key": "smtp_port", "label": "SMTP port", "type": "number", "default": 465},
            {"key": "username", "label": "Email address", "example": "you@gmail.com"},
            {"key": "app_password", "label": "App password", "secret": True},
        ],
    },
    "calendar": {
        "label": "Calendar (CalDAV)",
        "fields": [
            {"key": "caldav_url", "label": "CalDAV URL", "example": "https://caldav.icloud.com"},
            {"key": "username", "label": "Username", "example": "you@icloud.com"},
            {"key": "app_password", "label": "App password", "secret": True},
            {"key": "calendar_name", "label": "Calendar name (optional)", "required": False},
        ],
    },
    "zoom": {
        "label": "Zoom (Server-to-Server OAuth)",
        "fields": [
            {"key": "account_id", "label": "Account ID"},
            {"key": "client_id", "label": "Client ID"},
            {"key": "client_secret", "label": "Client secret", "secret": True},
        ],
    },
    "teams": {
        "label": "Microsoft Teams",
        "fields": [
            {"key": "tenant_id", "label": "Azure tenant ID"},
            {"key": "client_id", "label": "App client ID"},
            {"key": "client_secret", "label": "Client secret", "secret": True},
            {"key": "organizer_user_id", "label": "Organizer user object ID"},
        ],
    },
    "trading": {
        "label": "Trading (Alpaca paper account)",
        "fields": [
            {"key": "api_key", "label": "Alpaca paper API key"},
            {"key": "api_secret", "label": "Alpaca paper API secret", "secret": True},
        ],
    },
    "weather": {
        "label": "Weather location",
        "fields": [
            {"key": "latitude", "label": "Latitude", "type": "number", "required": False},
            {"key": "longitude", "label": "Longitude", "type": "number", "required": False},
            {"key": "location_label", "label": "Location label (optional)", "required": False},
            {"key": "unit", "label": "Unit (celsius/fahrenheit)", "default": "celsius", "required": False},
        ],
    },
    "youtube": {
        "label": "YouTube",
        "settings_path": ["social_media", "youtube"],
        "fields": [{"key": "api_key", "label": "YouTube Data API v3 key"}],
    },
    "reddit": {
        "label": "Reddit",
        "settings_path": ["social_media", "reddit"],
        "fields": [{"key": "user_agent", "label": "User agent string", "example": "leti-assistant/1.0 (by /u/you)"}],
    },
    "gui": {
        "label": "GUI web server",
        "fields": [
            {"key": "port", "label": "Port", "type": "number", "default": 8420},
            {"key": "enable_remote_access", "label": "Allow phones/other devices (true/false)", "type": "bool", "default": True},
        ],
    },
}


def _get_path(schema: Dict[str, Any], name: str) -> List[str]:
    return schema.get("settings_path", [name])


def _deep_get(d: Dict[str, Any], path: List[str]) -> Optional[Dict[str, Any]]:
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d if isinstance(d, dict) else None


def _deep_set(d: Dict[str, Any], path: List[str], value: Dict[str, Any]) -> None:
    for key in path[:-1]:
        d = d.setdefault(key, {})
    existing = d.get(path[-1])
    if isinstance(existing, dict):
        existing.update(value)
    else:
        d[path[-1]] = value


def _load_overrides() -> Dict[str, Any]:
    path = CONFIG_DIR / "settings.local.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _save_overrides(data: Dict[str, Any]) -> None:
    path = CONFIG_DIR / "settings.local.yaml"
    header = (
        "# Written by Leti's /settings command. Safe to hand-edit too, but it's\n"
        "# plain YAML with no comments explaining each field - see settings.yaml\n"
        "# for that. Anything set here overrides settings.yaml for the same key.\n"
    )
    path.write_text(header + yaml.safe_dump(data, sort_keys=False))


def _coerce(value: str, field: Dict[str, Any]) -> Any:
    field_type = field.get("type", "string")
    if field_type == "number":
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            raise ValueError(f"'{value}' isn't a valid number.")
    if field_type == "bool":
        low = value.strip().lower()
        if low in ("true", "yes", "y", "1"):
            return True
        if low in ("false", "no", "n", "0"):
            return False
        raise ValueError(f"'{value}' isn't true/false.")
    return value


def list_sections() -> List[Dict[str, Any]]:
    """Every editable section, with whether it's currently configured (has all
    its required fields set) - used to render the section picker."""
    settings = get_settings()
    out = []
    for name, schema in SECTION_SCHEMAS.items():
        current = _deep_get(settings, _get_path(schema, name)) or {}
        required_fields = [f for f in schema["fields"] if f.get("required", True) and not f.get("default")]
        configured = bool(current) and all(current.get(f["key"]) for f in required_fields)
        out.append({"name": name, "label": schema["label"], "configured": configured})
    return out


def get_section(name: str) -> Dict[str, Any]:
    """Current field values for one section, with secret fields redacted to a
    boolean "is_set" rather than ever echoing the real value back."""
    if name not in SECTION_SCHEMAS:
        raise KeyError(f"Unknown settings section: '{name}'")
    schema = SECTION_SCHEMAS[name]
    current = _deep_get(get_settings(), _get_path(schema, name)) or {}
    fields = []
    for f in schema["fields"]:
        entry = {**f, "value": None, "is_set": False}
        raw = current.get(f["key"])
        if raw not in (None, ""):
            entry["is_set"] = True
            if not f.get("secret"):
                entry["value"] = raw
        fields.append(entry)
    return {"name": name, "label": schema["label"], "fields": fields}


def update_section(name: str, values: Dict[str, str]) -> Dict[str, Any]:
    """Validates and saves new values for a section. `values` maps field key ->
    raw string input; blank/omitted values leave that field untouched (not
    cleared) - see clear_field() to explicitly remove one. Returns the
    reloaded settings for that section (secrets still redacted)."""
    if name not in SECTION_SCHEMAS:
        raise KeyError(f"Unknown settings section: '{name}'")
    schema = SECTION_SCHEMAS[name]
    path = _get_path(schema, name)

    to_save: Dict[str, Any] = {}
    for field in schema["fields"]:
        key = field["key"]
        raw = values.get(key)
        if raw in (None, ""):
            continue
        to_save[key] = _coerce(raw, field)

    if not to_save:
        raise ValueError("No values given to save.")

    overrides = _load_overrides()
    _deep_set(overrides, path, to_save)
    _save_overrides(overrides)
    reload_settings()
    return get_section(name)


def clear_section(name: str) -> None:
    """Removes a section from the override file entirely (settings.yaml's
    commented-out defaults take over again, i.e. the integration goes back
    to unconfigured)."""
    if name not in SECTION_SCHEMAS:
        raise KeyError(f"Unknown settings section: '{name}'")
    schema = SECTION_SCHEMAS[name]
    path = _get_path(schema, name)
    overrides = _load_overrides()
    d = overrides
    for key in path[:-1]:
        if key not in d:
            reload_settings()
            return
        d = d[key]
    d.pop(path[-1], None)
    _save_overrides(overrides)
    reload_settings()
