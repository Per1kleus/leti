"""
Leti's personality: a small set of numeric dials (0-10) that shape tone -
humor, sarcasm, formality, warmth, directness, verbosity - translated into
plain-language behavioral instructions injected into the system prompt.
Stored at data/personality.json (plain, hand-editable JSON, same pattern as
contacts.json), changeable any time via the tools below.

Important boundary: these dials only ever affect TONE. They never change
what Leti will or won't do - cranking up sarcasm/humor doesn't make
refusals, safety explanations, or factual accuracy any less serious or
clear. That instruction is baked into describe_personality()'s output
itself, not left to be inferred.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config_loader import get_settings, resolve_path
from tools.base import BaseTool, ToolParameter, ToolResult

DEFAULT_PERSONALITY: Dict[str, int] = {
    "humor": 5,
    "sarcasm": 2,
    "formality": 4,
    "warmth": 6,
    "directness": 6,
    "verbosity": 4,
}
PARAM_NAMES = list(DEFAULT_PERSONALITY.keys())

PRESETS: Dict[str, Dict[str, int]] = {
    "default": dict(DEFAULT_PERSONALITY),
    "witty_friend": {"humor": 8, "sarcasm": 6, "formality": 2, "warmth": 7, "directness": 7, "verbosity": 4},
    "professional": {"humor": 1, "sarcasm": 0, "formality": 9, "warmth": 5, "directness": 7, "verbosity": 5},
    "dry_and_sarcastic": {"humor": 6, "sarcasm": 9, "formality": 3, "warmth": 3, "directness": 8, "verbosity": 3},
    "warm_and_supportive": {"humor": 4, "sarcasm": 0, "formality": 3, "warmth": 9, "directness": 5, "verbosity": 5},
    "blunt_and_brief": {"humor": 2, "sarcasm": 3, "formality": 4, "warmth": 3, "directness": 10, "verbosity": 1},
}

# (low 0-3, mid 4-7, high 8-10) phrasing for each dial.
_PHRASES: Dict[str, tuple] = {
    "humor": (
        "Keep humor minimal - stay focused and matter-of-fact.",
        "Light, occasional humor is welcome when it fits naturally.",
        "Be playful and witty - look for genuine opportunities for humor.",
    ),
    "sarcasm": (
        "Avoid sarcasm - mean what you say plainly.",
        "A bit of dry wit is fine occasionally.",
        "Lean into dry, sarcastic wit as part of your voice - but never at the user's expense, "
        "and never about anything they're seriously asking for help with.",
    ),
    "formality": (
        "Speak casually, like a friend - contractions and informal phrasing are fine.",
        "Balance casual and professional language.",
        "Speak formally and precisely, with minimal slang.",
    ),
    "warmth": (
        "Stay neutral and businesslike - skip emotional language.",
        "Be friendly and considerate.",
        "Be warm, encouraging, and emotionally attentive.",
    ),
    "directness": (
        "Soften feedback and hedge more; prioritize tact over bluntness.",
        "Balance honesty with tact.",
        "Be blunt and direct - say what you actually think, skip the hedging.",
    ),
    "verbosity": (
        "Keep responses short - a sentence or two when possible.",
        "Give moderately detailed answers.",
        "Feel free to elaborate and give thorough, detailed answers.",
    ),
}


def _path() -> Path:
    cfg = get_settings().get("personality", {})
    p = resolve_path(cfg.get("file_path", "./data/personality.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load() -> Dict[str, int]:
    p = _path()
    if not p.exists():
        return dict(DEFAULT_PERSONALITY)
    try:
        data = json.loads(p.read_text())
        return {**DEFAULT_PERSONALITY, **{k: v for k, v in data.items() if k in PARAM_NAMES}}
    except json.JSONDecodeError:
        return dict(DEFAULT_PERSONALITY)


def _save(values: Dict[str, int]) -> None:
    _path().write_text(json.dumps(values, indent=2))


def _clamp(v: Any) -> int:
    return max(0, min(10, int(round(float(v)))))


def _level_phrase(value: int, phrases: tuple) -> str:
    low, mid, high = phrases
    if value <= 3:
        return low
    if value <= 7:
        return mid
    return high


def describe_personality(values: Optional[Dict[str, int]] = None) -> str:
    """Translates the numeric dials into a system-prompt-ready instruction block."""
    values = values or _load()
    lines = [
        f"- {param.capitalize()} ({values[param]}/10): {_level_phrase(values[param], _PHRASES[param])}"
        for param in PARAM_NAMES
    ]
    return (
        "Personality settings (these shape your TONE only - never what you will or won't help "
        "with; safety-relevant explanations and factual accuracy stay clear and serious no "
        "matter how these are set):\n" + "\n".join(lines)
    )


class GetPersonalitySettingsTool(BaseTool):
    name = "get_personality_settings"
    description = "Show Leti's current personality dial values (humor, sarcasm, formality, warmth, directness, verbosity)."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        values = _load()
        return ToolResult(success=True, output={"values": values, "description": describe_personality(values)})


class SetPersonalityTool(BaseTool):
    name = "set_personality"
    description = (
        "Adjust one or more of Leti's personality dials (0-10 each): humor, sarcasm, formality, "
        "warmth, directness, verbosity. Only the ones given are changed - others stay as they "
        "are. Use when the user asks to be funnier, more serious, more sarcastic, blunter, more "
        "formal/casual, more or less detailed, etc."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="humor", type="number", required=False, description="0-10."),
        ToolParameter(name="sarcasm", type="number", required=False, description="0-10."),
        ToolParameter(name="formality", type="number", required=False, description="0-10."),
        ToolParameter(name="warmth", type="number", required=False, description="0-10."),
        ToolParameter(name="directness", type="number", required=False, description="0-10."),
        ToolParameter(name="verbosity", type="number", required=False, description="0-10."),
    ]

    async def run(self, **kwargs) -> ToolResult:
        try:
            values = _load()
            changed = {}
            for param in PARAM_NAMES:
                if kwargs.get(param) is not None:
                    values[param] = _clamp(kwargs[param])
                    changed[param] = values[param]
            if not changed:
                return ToolResult(success=False, error="No valid personality parameters were given.")
            _save(values)
            return ToolResult(success=True, output={"changed": changed, "values": values, "description": describe_personality(values)})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ApplyPersonalityPresetTool(BaseTool):
    name = "apply_personality_preset"
    description = f"Apply a named personality preset in one step: {', '.join(PRESETS.keys())}."
    parameters: List[ToolParameter] = [
        ToolParameter(name="preset", type="string", enum=list(PRESETS.keys()), description="Preset name."),
    ]

    async def run(self, preset: str, **kwargs) -> ToolResult:
        if preset not in PRESETS:
            return ToolResult(success=False, error=f"Unknown preset '{preset}'. Options: {', '.join(PRESETS.keys())}")
        values = dict(PRESETS[preset])
        _save(values)
        return ToolResult(success=True, output={"preset": preset, "values": values, "description": describe_personality(values)})


class ResetPersonalityTool(BaseTool):
    name = "reset_personality"
    description = "Reset Leti's personality dials back to the default balance."
    parameters: List[ToolParameter] = []

    async def run(self, **kwargs) -> ToolResult:
        _save(dict(DEFAULT_PERSONALITY))
        return ToolResult(success=True, output={"values": DEFAULT_PERSONALITY, "description": describe_personality(DEFAULT_PERSONALITY)})
