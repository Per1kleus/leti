"""
Confirmation callbacks for SafetyGuard, shared between CLI modes (main.py)
and GUI mode (gui/api.py). Lives here rather than in main.py so neither
imports the other - main.py is the entry point (run as __main__, not a
safely-importable module), and gui/api.py needs these too.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from core.console_input import read_line
from core.intent_signals import resolve_yes_no


async def cli_confirmation_callback(prompt: str) -> bool:
    """Terminal-based confirmation for text mode. Accepts 'yes'/'y' as well as the
    '-y' shorthand (and 'no'/'n'/'-n' to explicitly decline).

    Reads through core.console_input rather than input() on an executor: SafetyGuard
    wraps this in asyncio.wait_for, and a cancelled executor thread would stay blocked
    on stdin, eating the user's next line. See that module's docstring.
    """
    print(f"\n[CONFIRM NEEDED] {prompt}")
    answer = await read_line("Type 'yes' (or -y) to proceed, 'no' (or -n) to cancel: ")
    if answer is None:
        return False  # end of input - fail closed
    answer = answer.strip().lower()
    if answer in ("-y", "y", "yes"):
        return True
    if answer in ("-n", "n", "no"):
        return False
    return False  # anything unrecognized fails closed


def make_voice_confirmation_callback(tts, transcriber) -> Callable[[str], Awaitable[bool]]:
    """Voice-mode confirmation: speaks the prompt and listens for a reply, interpreting a
    broad range of natural affirmative/negative phrasing (see core/intent_signals.py) rather
    than requiring an exact word - there's no '-y' to type while talking. Fails closed
    (treated as 'no') if the reply is unclear or nothing is heard, since this only runs
    when pre-approval from the original request wasn't already granted."""

    async def _callback(prompt: str) -> bool:
        print(f"\n[CONFIRM NEEDED] {prompt}")
        await tts.speak(prompt)
        audio = await transcriber.record_until_silence()
        reply_text = await transcriber.transcribe(audio)
        print(f"You said: {reply_text}")
        decision = resolve_yes_no(reply_text)
        if decision is None:
            print("(Couldn't tell if that was a yes or no - treating as declined.)")
            return False
        return decision

    return _callback
