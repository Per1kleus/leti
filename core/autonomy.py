"""Asking well, when something has to be asked.

"Leti wants to send an email. This reaches outside your computer. Should I go
ahead?" is a question nobody can answer. It does not say to whom, how many, what
happens to the ones it cannot send, or whether saying yes can be taken back. A
person faced with it either says yes reflexively - which makes the prompt
decoration rather than a check - or says no and does it themselves.

So this module works out what a call actually means, in concrete terms, from the
arguments it was given: how many, to whom, which file, what will be skipped, and
whether it can be undone. core/safety_guard.py puts that into the prompt.

WHAT THIS IS NOT, AND CANNOT BECOME. It is not a permission system. It decides
nothing. SafetyGuard still classifies every call, still consults
config/permissions.yaml and safety.require_confirmation_for, still blocks what is
forbidden, still refuses to let a CRITICAL action ride on an inferred approval,
and still raises ConfirmationDenied when nobody is there. The Permission Center
is still where a person changes any of that. Everything here is text.

That includes the graduated tiers below. They are a way of SAYING how heavy an
action is, mapped from the tiers SafetyGuard already assigned. Nothing reads them
to decide anything, and a "low impact" label on a call SafetyGuard classified as
CRITICAL would change the wording of the prompt and not one thing else.

The other half of asking well is not asking. A low-risk action the user has
already allowed must not be confirmed again out of politeness: SafetyGuard's
requires_confirmation_for is what decides that, and this module deliberately has
no way to add a prompt where SafetyGuard did not ask for one.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("leti.autonomy")

LOW = "low impact"
MEDIUM = "medium impact"
HIGHER = "higher impact"
HIGH = "high impact"

# SafetyGuard's tier -> how it reads to a person. A presentation mapping only.
TIERS: Dict[str, Dict[str, str]] = {
    "read": {"band": LOW,
             "means": "reads something; changes nothing"},
    "execute": {"band": MEDIUM,
                "means": "runs something on this computer"},
    "modify": {"band": HIGHER,
               "means": "changes something on this computer"},
    "external": {"band": HIGHER,
                 "means": "reaches outside this computer, where Leti cannot take it back"},
    "critical": {"band": HIGH,
                 "means": "cannot be undone"},
}

REVERSIBLE = "reversible"
IRREVERSIBLE = "not reversible"
UNKNOWN_REVERSIBILITY = "unclear whether this can be undone"


def _recipients(arguments: Dict[str, Any]) -> List[str]:
    """Everyone an address field names, however it was spelled."""
    found: List[str] = []
    for key in ("to", "recipient", "recipients", "cc", "bcc", "attendees",
                "participants", "emails"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            found.extend(part.strip() for part in value.replace(";", ",").split(",")
                         if part.strip())
        elif isinstance(value, (list, tuple)):
            found.extend(str(v).strip() for v in value if str(v).strip())
    return found


def _looks_like_an_address(value: str) -> bool:
    return "@" in value and "." in value.split("@")[-1] and " " not in value


def consequences(tool_name: str, arguments: Optional[Dict[str, Any]] = None,
                 tier: str = "") -> Dict[str, Any]:
    """What this call actually does, in facts rather than adjectives.

    Everything comes from the arguments in hand: nothing is fetched, nothing is
    counted by asking a service, and a fact that cannot be established from the
    call is left out rather than estimated. Never raises.
    """
    arguments = arguments if isinstance(arguments, dict) else {}
    name = str(tool_name or "")
    band = TIERS.get(str(tier or "").lower(), {}).get("band", MEDIUM)
    means = TIERS.get(str(tier or "").lower(), {}).get("means", "")

    facts: List[str] = []
    problems: List[str] = []
    reversible = UNKNOWN_REVERSIBILITY

    try:
        people = _recipients(arguments)
        if people:
            valid = [p for p in people if _looks_like_an_address(p)]
            unusable = [p for p in people if not _looks_like_an_address(p)]
            if len(valid) == 1:
                facts.append(f"one recipient: {valid[0]}")
            elif valid:
                facts.append(f"{len(valid)} recipient(s): "
                             + ", ".join(valid[:4])
                             + (f" and {len(valid) - 4} more" if len(valid) > 4 else ""))
            if unusable:
                problems.append(
                    (f"{unusable[0]} has no usable email address and would be skipped"
                     if len(unusable) == 1 else
                     f"{len(unusable)} of them have no usable email address "
                     f"({', '.join(unusable[:3])}) and would be skipped"))

        for key, label in (("path", "file"), ("file_path", "file"),
                           ("source_path", "from"), ("destination_path", "to"),
                           ("directory", "folder")):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                facts.append(f"{label}: {value.strip()}")

        for key in ("count", "quantity", "qty", "limit"):
            value = arguments.get(key)
            if isinstance(value, (int, float)) and value:
                facts.append(f"{key}: {value}")

        command = arguments.get("command") or arguments.get("cmd")
        if isinstance(command, str) and command.strip():
            facts.append(f"command: {command.strip()[:120]}")

        subject = arguments.get("subject")
        if isinstance(subject, str) and subject.strip():
            facts.append(f'subject: "{subject.strip()[:80]}"')

        reversible = _reversibility(name, str(tier or "").lower())
    except Exception as e:                                   # pragma: no cover - guard
        logger.debug(f"Consequence reading for {name} failed: {e}")

    return {"band": band, "means": means, "facts": facts, "problems": problems,
            "reversible": reversible}


# What can be taken back, and what cannot. Named per tool rather than derived,
# because "modify" covers both writing a file (a backup away from reversible) and
# turning on the firewall (a switch), and guessing between them is how a prompt
# reassures somebody about something it should not have.
_IRREVERSIBLE_TOOLS = frozenset({
    "send_email", "send_meeting_invite_email", "delete_file", "delete_contact",
    "place_paper_order", "clear_user_profile", "apply_system_updates",
    "kill_process", "post_to_social", "create_pull_request",
})
# write_file is deliberately absent. Writing over a file that already had
# something in it is not undoable without a snapshot, and telling somebody it is
# would be the prompt reassuring them about the one thing they should hesitate
# over. It falls through to "unclear", which is the truth.
_REVERSIBLE_TOOLS = frozenset({
    "move_file", "add_contact", "update_contact", "enable_firewall",
    "create_scheduled_task", "update_scheduled_task", "add_business_data",
    "launch_app", "close_app", "focus_window",
})


def _reversibility(tool_name: str, tier: str) -> str:
    if tool_name in _IRREVERSIBLE_TOOLS or tier == "critical":
        return IRREVERSIBLE
    if tool_name in _REVERSIBLE_TOOLS:
        return REVERSIBLE
    return UNKNOWN_REVERSIBILITY


def confirmation_prompt(action_description: str, tool_name: str,
                        arguments: Optional[Dict[str, Any]] = None,
                        tier: str = "") -> str:
    """The question to put in front of the user.

    Built around the description SafetyGuard already produces, with the concrete
    consequences after it and the caveats last, because the thing a person needs
    in order to answer is the thing that would make them say no.
    """
    detail = consequences(tool_name, arguments, tier)
    parts = [f"Leti wants to {action_description}."]

    if detail["facts"]:
        parts.append("This means: " + "; ".join(detail["facts"][:5]) + ".")
    if detail["problems"]:
        parts.append("Worth knowing: " + "; ".join(detail["problems"][:3]) + ".")

    if detail["reversible"] == IRREVERSIBLE:
        parts.append("This cannot be undone.")
    elif detail["reversible"] == REVERSIBLE and detail["band"] in (HIGHER, HIGH):
        parts.append("This can be undone afterwards.")
    elif detail["means"]:
        parts.append(detail["means"].capitalize() + ".")

    parts.append("Should I go ahead?")
    return " ".join(parts)


def describe_bands() -> List[Dict[str, str]]:
    """What the tiers mean, for the Permission Center and for diagnostics.

    A description of SafetyGuard's classification, not a second one: the keys are
    its tier names and nothing here can change what any of them are allowed to do.
    """
    return [{"tier": tier, "band": spec["band"], "means": spec["means"],
             "decided_by": "core/safety_guard.py and config/permissions.yaml"}
            for tier, spec in TIERS.items()]
