"""
Lightweight, local, keyword-level matcher for "the user already said yes/no"
in a piece of transcribed or typed text.

Honest limitation: this reads WORDS, not actual vocal tone/prosody - there
is no audio sentiment/emotion model in this project, so "tone" in practice
means "the phrasing of what was transcribed" rather than pitch, pace, or
emphasis. If real prosodic confidence scoring is wanted later, it'd need a
speech-emotion model in the STT pipeline (audio/stt.py) feeding a
confidence score in here - out of scope for this change.
"""
from __future__ import annotations

import re
from typing import Optional

APPROVAL_PHRASES = [
    "y", "yes", "yeah", "yep", "yup", "sure", "confirm", "confirmed", "i confirm",
    "go ahead", "go for it", "do it", "please proceed", "proceed", "affirmative",
    "sounds good", "that's fine", "that works", "approved", "correct", "please do",
]

DENIAL_PHRASES = [
    "n", "no", "nope", "don't", "do not", "cancel", "stop", "never mind", "nevermind",
    "hold off", "wait", "not yet", "deny", "denied", "negative",
    # Negated/uncertain forms that would otherwise false-positive as approval via a bare
    # word like "sure" (e.g. "not sure" should never read as approval).
    "not sure", "unsure", "not really", "not confirmed",
]


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def contains_explicit_approval(text: Optional[str]) -> bool:
    if not text:
        return False
    norm = f" {_normalize(text)} "
    return any(f" {phrase} " in norm for phrase in APPROVAL_PHRASES)


def contains_explicit_denial(text: Optional[str]) -> bool:
    if not text:
        return False
    norm = f" {_normalize(text)} "
    return any(f" {phrase} " in norm for phrase in DENIAL_PHRASES)


def resolve_yes_no(text: Optional[str]) -> Optional[bool]:
    """Best-effort yes/no read of a short reply: True/False, or None if unclear
    (caller should treat None as 'not confirmed' - fail closed)."""
    approved = contains_explicit_approval(text)
    denied = contains_explicit_denial(text)
    if approved and not denied:
        return True
    if denied and not approved:
        return False
    return None
