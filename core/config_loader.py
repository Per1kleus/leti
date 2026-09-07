"""
Loads and caches settings.yaml and permissions.yaml so every module reads
from the same source of truth without repeated disk I/O.

settings.local.yaml (optional, not shipped, created on first edit) layers
on top of settings.yaml - it's what core/settings_editor.py writes to when
the user edits settings via the /settings command, so real values (API
keys, passwords) never have to be hand-edited into the heavily-commented,
example-filled settings.yaml. Same idea as a local override file in most
dev tooling: the shipped file stays a clean reference/template, actual
secrets live somewhere that's easy to point elsewhere or wipe separately.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override into base, returning a new dict. Nested dicts
    merge key-by-key (so setting one field in a section doesn't wipe its
    siblings); anything else in override replaces the base value outright."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@functools.lru_cache(maxsize=1)
def get_settings() -> Dict[str, Any]:
    base = _load_yaml(CONFIG_DIR / "settings.yaml")
    local_path = CONFIG_DIR / "settings.local.yaml"
    if local_path.exists():
        overrides = _load_yaml(local_path) or {}
        return _deep_merge(base, overrides)
    return base


def reload_settings() -> Dict[str, Any]:
    """Clears the cache and reloads from disk. Called after
    core/settings_editor.py saves a change, so edits made via /settings take
    effect immediately rather than requiring a full restart - everything
    else in the app calls get_settings() fresh each time it needs a value
    (no module holds onto a stale copy), so clearing this one cache is enough."""
    get_settings.cache_clear()
    return get_settings()


@functools.lru_cache(maxsize=1)
def get_permissions() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "permissions.yaml")


def resolve_path(relative_or_absolute: str) -> Path:
    """Resolve a path to a single canonical form: expand ~ and env vars, anchor
    relative paths to the project root, then normalize away '..'/'.' segments and
    follow symlinks.

    Normalizing ABSOLUTE paths too is what makes core/safety_guard.py's
    protected-path check sound. Comparing an unnormalized path against the
    protected list lets '/tmp/../etc/shadow' and '~/../<user>/.ssh/id_rsa' slip
    past a guard that blocks '/etc/shadow' and '~/.ssh/id_rsa' - they name the
    same files. Symlinks are followed for the same reason: a link into a
    protected directory must not read as a path outside it.

    resolve(strict=False) is deliberate - paths that don't exist yet (a file
    about to be written, a snapshot directory about to be created) still
    normalize, with the non-existent tail appended to the resolved prefix.
    """
    expanded = os.path.expanduser(os.path.expandvars(relative_or_absolute))
    p = Path(expanded)
    if not p.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        p = project_root / p
    try:
        return p.resolve()
    except OSError:
        # Unreadable parent, a symlink loop, or a path too long to resolve. Fall
        # back to purely lexical normalization rather than returning the raw
        # input - callers (the safety guard above all) must never see '..'.
        return Path(os.path.normpath(str(p)))


def ensure_data_dirs() -> None:
    settings = get_settings()
    for key in ("data_dir", "logs_dir"):
        resolve_path(settings["paths"][key]).mkdir(parents=True, exist_ok=True)
    resolve_path(settings["memory"]["persist_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_path(settings["vision"]["screenshot_dir"]).mkdir(parents=True, exist_ok=True)
