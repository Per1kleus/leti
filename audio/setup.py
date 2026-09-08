"""First-run audio permission: ask once, check it actually works, remember the answer.

Leti records through the machine's own microphone and speaks through its own
speakers, and on macOS and Windows the first attempt to open an input stream is
what triggers the operating system's own permission dialog. Left to itself that
dialog appears at a random moment - mid-sentence, behind the window, the first
time somebody happens to say the wake word - and if it's missed, voice simply
never works with nothing on screen to explain why.

So the first launch asks deliberately: it opens the microphone at a moment the
user is looking at a prompt about the microphone, measures whether any sound
actually arrives, plays a tone and asks whether it was audible, and writes the
answer to data/audio_setup.json. Every launch after that reads the file.

Three states, not two. "Allowed and working", "declined", and "never asked" are
different: only the last one prompts, and a decline is remembered rather than
re-asked every launch. Nothing here is irreversible - `/audio` in text mode and
the Audio button in the HUD re-run the same flow.

What this deliberately does NOT do is offer a speaker picker. pyttsx3 hands audio
to the OS's default output device and gives no way to choose another, so a picker
here would be a control that silently does nothing. The check reports which device
the system default actually is, and says to change it in the OS to change Leti's.
"""
from __future__ import annotations

import json
import logging
import math
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

from core.atomic_write import atomic_write_text
from core.config_loader import resolve_path

logger = logging.getLogger("leti.audio.setup")

SETUP_PATH = "./data/audio_setup.json"
SETUP_VERSION = 1

# Recording parameters for the microphone check. 16 kHz mono matches what
# WhisperTranscriber captures, so a level measured here is a level Whisper sees.
CHECK_SAMPLE_RATE = 16000
CHECK_CHUNK = 1024
CHECK_SECONDS = 3.0

# Peak amplitude (0.0-1.0) above which we consider real sound to have arrived
# rather than the noise floor of a live-but-silent input. A quiet room on an
# open mic sits around 0.001-0.01; speech at a normal distance clears 0.05 easily.
SIGNAL_THRESHOLD = 0.02

TEST_TONE_HZ = 440.0
TEST_TONE_SECONDS = 1.2
TEST_TONE_RATE = 44100


def setup_path():
    return resolve_path(SETUP_PATH)


def load_setup() -> Optional[Dict[str, Any]]:
    """The saved answers, or None if this machine has never been asked."""
    path = setup_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        # A corrupt file means "we don't know", which is the same as never having
        # asked - better one extra prompt than voice silently disabled forever.
        logger.warning(f"Couldn't read {path} ({e}); treating audio as un-configured.")
        return None
    return data if isinstance(data, dict) else None


def save_setup(data: Dict[str, Any]) -> Dict[str, Any]:
    data = {**data, "version": SETUP_VERSION, "completed_at": time.time()}
    path = setup_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2))
    return data


def is_configured() -> bool:
    return load_setup() is not None


def microphone_allowed() -> bool:
    """True unless the user has been asked and said no.

    Un-configured counts as allowed so that nothing about voice changes until the
    first-run prompt has actually run - the prompt is what asks, not this.
    """
    setup = load_setup()
    if setup is None:
        return True
    return bool(setup.get("microphone", {}).get("allowed", True))


def speakers_allowed() -> bool:
    setup = load_setup()
    if setup is None:
        return True
    return bool(setup.get("speakers", {}).get("allowed", True))


def chosen_input_device() -> Optional[int]:
    """The saved microphone's index, or None to use the system default.

    Returned as an index because that is what PyAudio's `input_device_index`
    takes; None is meaningful there too, so "no preference" needs no special case.
    """
    setup = load_setup()
    if not setup:
        return None
    index = setup.get("microphone", {}).get("device_index")
    return int(index) if isinstance(index, (int, float)) else None


def blocked_reason() -> Optional[str]:
    """Why voice is off, phrased for the user, or None if it isn't."""
    setup = load_setup()
    if setup is None:
        return None
    mic_ok = setup.get("microphone", {}).get("allowed", True)
    spk_ok = setup.get("speakers", {}).get("allowed", True)
    if mic_ok and spk_ok:
        return None
    if not mic_ok and not spk_ok:
        missing = "the microphone or the speakers"
    else:
        missing = "the microphone" if not mic_ok else "the speakers"
    return (f"You chose not to let Leti use {missing}. Run /audio in text mode, or "
            f"press Audio in the interface, to change that.")


# --- Devices ---------------------------------------------------------------------

def _pyaudio():
    import pyaudio

    return pyaudio.PyAudio()


def list_devices() -> Dict[str, Any]:
    """Input devices to choose from, and which output the system will actually use.

    Errors are returned rather than raised: "there is no sound card" is the normal
    case on a server, and the caller's job is to say so, not to crash.
    """
    try:
        pa = _pyaudio()
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}",
                "inputs": [], "default_input": None, "default_output": None}

    try:
        inputs: List[Dict[str, Any]] = []
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                continue
            if int(info.get("maxInputChannels", 0)) > 0:
                inputs.append({
                    "index": i,
                    "name": str(info.get("name", f"device {i}")),
                    "channels": int(info["maxInputChannels"]),
                    "sample_rate": int(info.get("defaultSampleRate", 0)),
                })

        def _default(getter):
            try:
                info = getter()
                return {"index": int(info["index"]), "name": str(info["name"])}
            except Exception:
                return None

        return {
            "available": True,
            "inputs": inputs,
            "default_input": _default(pa.get_default_input_device_info),
            "default_output": _default(pa.get_default_output_device_info),
        }
    finally:
        pa.terminate()


# --- Checking that it actually works ----------------------------------------------

def measure_microphone(device_index: Optional[int] = None,
                       seconds: float = CHECK_SECONDS) -> Dict[str, Any]:
    """Record briefly and report how loud it was.

    This is the call that makes the OS show its permission dialog on macOS and
    Windows, which is the point of doing it during a prompt that explains why.

    A stream that opens but delivers silence is the failure people actually hit -
    a muted input, the wrong device, permission granted to the terminal but not to
    Python - and it looks identical to success unless the level is measured. So
    the answer is a peak, not a boolean.
    """
    import pyaudio

    try:
        pa = _pyaudio()
    except Exception as e:
        return {"ok": False, "error": f"Couldn't open the audio system: {e}"}

    stream = None
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=CHECK_SAMPLE_RATE,
                         input=True, frames_per_buffer=CHECK_CHUNK,
                         input_device_index=device_index)
        peak = 0.0
        total_squares = 0.0
        samples = 0
        for _ in range(max(1, int(CHECK_SAMPLE_RATE / CHECK_CHUNK * seconds))):
            raw = stream.read(CHECK_CHUNK, exception_on_overflow=False)
            values = struct.unpack(f"{len(raw) // 2}h", raw)
            for v in values:
                level = abs(v) / 32768.0
                peak = max(peak, level)
                total_squares += level * level
            samples += len(values)
        rms = math.sqrt(total_squares / samples) if samples else 0.0
        return {
            "ok": True,
            "peak": round(peak, 4),
            "rms": round(rms, 4),
            "heard_sound": peak >= SIGNAL_THRESHOLD,
            "threshold": SIGNAL_THRESHOLD,
        }
    except Exception as e:
        # A denied permission surfaces here as an OSError from the audio layer,
        # which is worth passing through verbatim - it names the real reason.
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()


def play_test_tone(seconds: float = TEST_TONE_SECONDS) -> Dict[str, Any]:
    """Play a tone through the system's default output.

    Deliberately a generated tone rather than only a spoken sentence: it isolates
    the audio path from the speech engine, so "I heard nothing" after this means
    the speakers, while "I heard the beep but not the words" means espeak or a
    voice setting. Fading the ends avoids the click a truncated sine makes.
    """
    import pyaudio

    try:
        pa = _pyaudio()
    except Exception as e:
        return {"ok": False, "error": f"Couldn't open the audio system: {e}"}

    stream = None
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=TEST_TONE_RATE, output=True)
        frames = int(TEST_TONE_RATE * seconds)
        fade = max(1, int(TEST_TONE_RATE * 0.02))
        samples = bytearray()
        for n in range(frames):
            envelope = min(1.0, n / fade, (frames - n) / fade)
            value = int(0.35 * envelope * 32767 * math.sin(2 * math.pi * TEST_TONE_HZ * n / TEST_TONE_RATE))
            samples += struct.pack("<h", value)
        stream.write(bytes(samples))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()


def record_choice(mic_allowed: bool, speakers_ok: bool,
                  device_index: Optional[int] = None, device_name: str = "",
                  mic_measurement: Optional[Dict[str, Any]] = None,
                  output_name: str = "", note: str = "") -> Dict[str, Any]:
    """Write the outcome of a setup run. This is what makes it a first-run prompt."""
    measurement = mic_measurement or {}
    return save_setup({
        "microphone": {
            "allowed": bool(mic_allowed),
            "device_index": device_index,
            "device_name": device_name,
            "verified": bool(measurement.get("heard_sound")),
            "peak_level": measurement.get("peak"),
        },
        "speakers": {
            "allowed": bool(speakers_ok),
            "device_name": output_name,
            "verified": bool(speakers_ok),
        },
        "note": note,
    })


# --- The first-run conversation ----------------------------------------------------

async def run_console_setup(force: bool = False) -> Optional[Dict[str, Any]]:
    """Ask, test, and save - in the terminal. Returns the saved record, or None if
    this machine was already set up and `force` wasn't given.

    Reads through core.console_input and interprets the answers with
    core.intent_signals, the same reader and the same yes/no understanding the
    confirmation prompts use, so "yeah" works here exactly as it does there.
    """
    from core.console_input import read_line
    from core.intent_signals import resolve_yes_no

    if is_configured() and not force:
        return None

    class NoOneToAsk(Exception):
        """stdin ended: this launch has nobody at a keyboard to answer."""

    async def ask(question: str, default: Optional[bool] = None) -> bool:
        suffix = " [Y/n]: " if default is True else (" [y/N]: " if default is False else " [y/n]: ")
        while True:
            answer = await read_line(question + suffix)
            if answer is None:
                raise NoOneToAsk()
            if not answer.strip() and default is not None:
                return default
            decision = resolve_yes_no(answer)
            if decision is not None:
                return decision
            print("  Please answer yes or no.")

    try:
        return await _ask_and_check(ask, force)
    except NoOneToAsk:
        # Nothing was saved on purpose. A launcher, a service or a piped stdin has
        # nobody to answer, and recording a decline there would silently turn voice
        # off forever for a user who was never actually asked. Leave it un-answered
        # so the next launch with a person at the keyboard asks properly.
        print("\n(No answer available on this launch - Leti will ask again next time. "
              "Voice stays off until then.)\n")
        return None


async def _ask_and_check(ask, force: bool) -> Dict[str, Any]:
    """The flow itself. Split out so `ask` raising on end-of-input unwinds cleanly
    from wherever in the conversation it happens, rather than each question needing
    to check."""
    from core.console_input import read_line

    print("\n" + "=" * 68)
    print("Setting up audio - this happens once.")
    print("=" * 68)
    print(
        "Leti listens through this computer's microphone and replies through its\n"
        "speakers. Your operating system may show its own permission dialog the\n"
        "first time the microphone opens - allow it there too, or voice stays off.\n"
        "Nothing is uploaded: speech is transcribed locally by Whisper.\n"
        "You can skip this and use Leti entirely by typing.\n"
    )

    if not await ask("Let Leti use the microphone and speakers?", default=True):
        record = record_choice(mic_allowed=False, speakers_ok=False,
                              note="Declined at first-run setup.")
        print("\nNo problem - voice is off and Leti won't ask again.")
        print("Run /audio any time to turn it on.\n")
        return record

    devices = list_devices()
    if not devices.get("available"):
        print(f"\nThis machine has no working audio system ({devices.get('error')}).")
        print("Leti will run in text-only mode.\n")
        return record_choice(mic_allowed=False, speakers_ok=False,
                             note=f"No audio system: {devices.get('error')}")

    # --- Microphone ---
    inputs = devices["inputs"]
    default_input = devices.get("default_input")
    device_index: Optional[int] = None
    device_name = ""

    if not inputs:
        print("\nNo microphone was found on this machine.")
        return record_choice(mic_allowed=False, speakers_ok=True,
                             output_name=(devices.get("default_output") or {}).get("name", ""),
                             note="No input device present.")

    if len(inputs) == 1:
        device_index, device_name = inputs[0]["index"], inputs[0]["name"]
        print(f"\nMicrophone: {device_name}")
    else:
        print("\nMicrophones found:")
        for position, device in enumerate(inputs, start=1):
            marker = "  (system default)" if default_input and device["index"] == default_input["index"] else ""
            print(f"  {position}. {device['name']}{marker}")
        choice = await read_line(f"Which one? [1-{len(inputs)}, Enter for the system default]: ")
        picked = None
        if choice and choice.strip().isdigit():
            position = int(choice.strip())
            if 1 <= position <= len(inputs):
                picked = inputs[position - 1]
        if picked is None and default_input:
            picked = next((d for d in inputs if d["index"] == default_input["index"]), inputs[0])
        picked = picked or inputs[0]
        device_index, device_name = picked["index"], picked["name"]
        print(f"Using: {device_name}")

    measurement: Dict[str, Any] = {}
    while True:
        print(f"\nSay something for {int(CHECK_SECONDS)} seconds, starting now...")
        measurement = measure_microphone(device_index, seconds=CHECK_SECONDS)
        if not measurement.get("ok"):
            print(f"  Couldn't record: {measurement.get('error')}")
            if not await ask("  Try again?", default=True):
                return record_choice(mic_allowed=False, speakers_ok=True,
                                     device_index=device_index, device_name=device_name,
                                     output_name=(devices.get("default_output") or {}).get("name", ""),
                                     note=f"Microphone check failed: {measurement.get('error')}")
            continue
        if measurement["heard_sound"]:
            print(f"  Heard you - peak level {measurement['peak']:.2f}. Microphone works.")
            break
        print(f"  Nothing came through (peak {measurement['peak']:.3f}). The microphone opened,")
        print("  but no sound arrived - it may be muted, or the OS may have denied access.")
        if not await ask("  Try again?", default=True):
            print("  Keeping the microphone enabled anyway - you can re-check with /audio.")
            break

    # --- Speakers ---
    output_name = (devices.get("default_output") or {}).get("name", "the system default output")
    speakers_ok = True
    while True:
        print(f"\nPlaying a test tone through {output_name}...")
        tone = play_test_tone()
        if not tone.get("ok"):
            print(f"  Couldn't play it: {tone.get('error')}")
            speakers_ok = False
            break
        if await ask("  Did you hear it?", default=True):
            break
        print("  Turn the volume up, or change the system's default output device")
        print("  (Leti speaks through whatever the OS default is - it can't pick another).")
        if not await ask("  Try again?", default=True):
            speakers_ok = False
            break

    record = record_choice(mic_allowed=True, speakers_ok=speakers_ok,
                          device_index=device_index, device_name=device_name,
                          mic_measurement=measurement, output_name=output_name)
    print("\nAudio is set up. Leti won't ask again - run /audio to change it.\n")
    return record
