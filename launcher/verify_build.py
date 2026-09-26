"""Check that the thing PyInstaller just built is the thing we meant to ship.

launcher/build_exe.py used to check two things: that a file appeared, and that its
size was in a plausible range. Both are worth checking and neither answers the
question a release needs answered - an executable can be the right size, exist in
the right place, and still be for the wrong operating system, carry the wrong
architecture, have no icon, or have swept a .env file into the bundle on its way
past.

So this reads the file. It is deliberately dependency-free and platform-neutral:
the PE header is the same bytes whichever machine looks at them, which means the
parser can be written and tested anywhere even though only Windows can produce the
file it is about.

Two questions, kept apart because they fail for different reasons:

    describe_pe()      what this binary IS - format, machine, subsystem, whether
                       it carries an icon and version metadata
    scan_for_leaks()   what it CONTAINS that it should not - credentials, .env
                       files, development paths, test fixtures, .git

Nothing here is Leti-specific enough to belong in the application, and nothing in
the application imports it. It is build tooling.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# What the binary is
# --------------------------------------------------------------------------- #

# IMAGE_FILE_MACHINE_*, the values that matter for a desktop release.
MACHINES = {
    0x014C: "x86 (32-bit)",
    0x8664: "x64 (AMD64)",
    0xAA64: "ARM64",
    0x01C4: "ARMv7",
}
WANTED_MACHINE = 0x8664

# IMAGE_SUBSYSTEM_*. Leti.exe is a console program on purpose - the first run has
# something to say - and leti_launcher.hide_console() puts the window away once
# Leti's own is up. A GUI subsystem here would mean that output had nowhere to go.
SUBSYSTEMS = {1: "native", 2: "Windows GUI", 3: "Windows console"}
WANTED_SUBSYSTEM = 3

PE32_PLUS = 0x20B          # 64-bit optional header
PE32 = 0x10B

# Resource type 14 is RT_GROUP_ICON, 3 is RT_ICON, 16 is RT_VERSION.
RT_ICON = 3
RT_GROUP_ICON = 14
RT_VERSION = 16


class NotAPortableExecutable(Exception):
    """The file is not a Windows PE binary. Says what it looks like instead."""


def _looks_like(head: bytes) -> str:
    """A name for what was found, so the failure is useful rather than a denial."""
    if head[:4] == b"\x7fELF":
        return "an ELF binary (Linux)"
    if head[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe") or head[:4] == b"\xca\xfe\xba\xbe":
        return "a Mach-O binary (macOS)"
    if head[:2] == b"#!":
        return "a script with a shebang"
    if head[:2] == b"PK":
        return "a zip archive"
    return "not a recognised executable format"


def describe_pe(path: Path) -> Dict[str, Any]:
    """Read the PE headers. Raises NotAPortableExecutable for anything else.

    Only the fields a release check needs, read at their fixed offsets rather than
    through a library, because adding a build-time dependency to answer "is this a
    .exe" would be the wrong trade.
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise NotAPortableExecutable(
            f"{path.name} does not start with the MZ signature - it is {_looks_like(data[:8])}")

    # e_lfanew at 0x3C points at the PE signature.
    (pe_offset,) = struct.unpack_from("<I", data, 0x3C)
    if pe_offset + 24 > len(data) or data[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise NotAPortableExecutable(
            f"{path.name} has an MZ header but no PE signature where it points")

    machine, sections, timestamp = struct.unpack_from("<HHI", data, pe_offset + 4)
    (optional_size,) = struct.unpack_from("<H", data, pe_offset + 20)
    (characteristics,) = struct.unpack_from("<H", data, pe_offset + 22)
    optional = pe_offset + 24
    (magic,) = struct.unpack_from("<H", data, optional)
    bits = 64 if magic == PE32_PLUS else 32
    # AddressOfEntryPoint is at +16 in the optional header for both widths.
    (entry_point,) = struct.unpack_from("<I", data, optional + 16)
    # Subsystem sits at +68 for PE32 and PE32+ alike.
    (subsystem,) = struct.unpack_from("<H", data, optional + 68)

    return {
        "path": str(path),
        "size_bytes": len(data),
        "size_mb": round(len(data) / (1024 * 1024), 2),
        "format": "PE32+" if magic == PE32_PLUS else ("PE32" if magic == PE32 else f"unknown ({magic:#x})"),
        "bits": bits,
        "machine": MACHINES.get(machine, f"unknown ({machine:#x})"),
        "machine_id": machine,
        "sections": sections,
        "timestamp": timestamp,
        "characteristics": characteristics,
        "entry_point": entry_point,
        "subsystem": SUBSYSTEMS.get(subsystem, f"unknown ({subsystem})"),
        "subsystem_id": subsystem,
        "optional_header_size": optional_size,
        "resource_types": sorted(_resource_types(data, pe_offset, magic)),
        "has_icon": RT_GROUP_ICON in _resource_types(data, pe_offset, magic),
        "has_version_info": RT_VERSION in _resource_types(data, pe_offset, magic),
        "has_pyinstaller_bootloader": _carries_pyinstaller(data),
    }


def _resource_types(data: bytes, pe_offset: int, magic: int) -> List[int]:
    """The resource type IDs in the .rsrc directory, or [] if it cannot be read.

    Enough to answer "is there an icon" and "is there version metadata" without
    walking the whole tree. A file whose resource directory cannot be parsed
    reports nothing rather than guessing, and the caller treats that as unknown.
    """
    try:
        optional = pe_offset + 24
        # NumberOfRvaAndSizes, then the data directories. Resources are entry 2.
        directories = optional + (112 if magic == PE32_PLUS else 96)
        resource_rva, resource_size = struct.unpack_from("<II", data, directories + 2 * 8)
        if not resource_rva or not resource_size:
            return []

        (sections,) = struct.unpack_from("<H", data, pe_offset + 6)
        (optional_size,) = struct.unpack_from("<H", data, pe_offset + 20)
        table = optional + optional_size
        offset = None
        for i in range(sections):
            entry = table + i * 40
            virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from(
                "<IIII", data, entry + 8)
            if virtual_address <= resource_rva < virtual_address + max(virtual_size, raw_size):
                offset = raw_pointer + (resource_rva - virtual_address)
                break
        if offset is None or offset + 16 > len(data):
            return []

        # IMAGE_RESOURCE_DIRECTORY: 12 bytes, then NumberOfNamedEntries and
        # NumberOfIdEntries, then the entries.
        named, by_id = struct.unpack_from("<HH", data, offset + 12)
        found: List[int] = []
        for i in range(named + by_id):
            entry = offset + 16 + i * 8
            if entry + 8 > len(data):
                break
            (name, _child) = struct.unpack_from("<II", data, entry)
            if not name & 0x80000000:          # an integer type ID, not a string
                found.append(name)
        return found
    except Exception:
        return []


def _carries_pyinstaller(data: bytes) -> bool:
    """Is this a PyInstaller build?

    The bootloader leaves its cookie near the end of the file and its own strings
    throughout. Either is enough; both being absent means whatever this is, it was
    not built from the spec.
    """
    tail = data[-131072:] if len(data) > 131072 else data
    return b"MEI\014\013\012\013\016" in tail or b"pyi-" in data or b"PyInstaller" in data


def check_pe(path: Path, want_machine: int = WANTED_MACHINE,
             want_subsystem: Optional[int] = WANTED_SUBSYSTEM,
             require_icon: bool = True) -> List[str]:
    """Everything wrong with this binary as a Windows release, as sentences.

    An empty list means it passed. Returning problems rather than raising on the
    first one means one build report lists all of them.
    """
    problems: List[str] = []
    try:
        pe = describe_pe(path)
    except NotAPortableExecutable as e:
        return [str(e)]
    except OSError as e:
        return [f"{path} could not be read ({e})"]

    if pe["machine_id"] != want_machine:
        problems.append(
            f"built for {pe['machine']}, not {MACHINES.get(want_machine, want_machine)}")
    if pe["bits"] != 64:
        problems.append(f"a {pe['bits']}-bit image; a 64-bit one was expected")
    if not pe["entry_point"]:
        problems.append("the entry point address is zero, so there is nothing to run")
    if want_subsystem is not None and pe["subsystem_id"] != want_subsystem:
        problems.append(
            f"subsystem is {pe['subsystem']}, not {SUBSYSTEMS.get(want_subsystem)} - "
            "the first run's output would have nowhere to go")
    if require_icon and not pe["has_icon"]:
        problems.append("no icon is embedded, so Explorer will show the default one")
    if not pe["has_pyinstaller_bootloader"]:
        problems.append("no PyInstaller bootloader found, so this was not built from the spec")
    return problems


# --------------------------------------------------------------------------- #
# What the binary contains that it should not
# --------------------------------------------------------------------------- #

# Each rule is (name, pattern, why it matters). Deliberately conservative: a false
# alarm on a release check costs a minute of looking, and a missed credential in a
# shipped binary costs rather more.
LEAK_RULES: Sequence[Any] = (
    ("a .env file", re.compile(rb"(?i)\.env(?:\.[a-z]+)?\b"),
     "environment files hold credentials and are never needed in a build"),
    ("an AWS access key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
     "that is the shape of a live AWS key"),
    ("a private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
     "a private key is in the bundle"),
    ("a GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{16,}"),
     "that is the shape of a GitHub personal access token"),
    ("a Slack token", re.compile(rb"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
     "that is the shape of a Slack token"),
    ("an OpenAI-style key", re.compile(rb"\bsk-[A-Za-z0-9]{32,}"),
     "that is the shape of an API key"),
    ("a bearer token", re.compile(rb"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]{16,}"),
     "a hard-coded Authorization header"),
    ("Leti's own settings override", re.compile(rb"settings\.local\.yaml"),
     "that file holds the user's saved passwords and must never be bundled"),
    ("the GUI's remote token file", re.compile(rb"gui_remote_token"),
     "that file is the shared secret for LAN access"),
    ("a test fixture", re.compile(rb"(?:^|[^A-Za-z0-9_.-])tests?[\\/]test_[a-z0-9_]+\.py"),
     "tests are not part of a release"),
    ("a .git directory", re.compile(rb"[\\/]\.git[\\/](?:HEAD|config|index)"),
     "repository internals are not part of a release"),
    ("a developer home directory", re.compile(rb"(?i)[a-z]:\\users\\(?!(?:\*|<|%|public\b))[a-z0-9._-]+\\"),
     "an absolute path from the machine that built it"),
    ("a Unix developer home directory", re.compile(rb"/(?:home|Users)/(?!user/leti)[a-z0-9._-]+/"),
     "an absolute path from the machine that built it"),
)

# Strings a PyInstaller build legitimately carries, which the rules above would
# otherwise flag. Each one is here because it was checked, not to quiet a warning.
ALLOWED = (
    # The spec excludes these; their names still appear in PyInstaller's own module
    # tables. A name is not a file.
    b"pytest", b"unittest", b"doctest",
)


def scan_for_leaks(path: Path, extra_rules: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    """Anything in the file that should not ship. Empty list means clean.

    Reads the whole binary as bytes. It is under ten megabytes by design, so there
    is no reason to be clever about it.
    """
    data = Path(path).read_bytes()
    findings: List[Dict[str, Any]] = []
    for name, pattern, why in list(LEAK_RULES) + list(extra_rules):
        for match in pattern.finditer(data):
            found = match.group(0)
            if any(allowed in found for allowed in ALLOWED):
                continue
            findings.append({
                "what": name,
                "why": why,
                "offset": match.start(),
                # Decoded lossily and truncated: this goes on a terminal, and a
                # finding is a pointer to look, not the secret itself.
                "excerpt": found[:80].decode("utf-8", "replace"),
            })
            break          # one report per rule; the build is already failing
    return findings


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def report(path: Path, want_machine: int = WANTED_MACHINE,
           want_subsystem: Optional[int] = WANTED_SUBSYSTEM,
           require_icon: bool = True) -> Dict[str, Any]:
    """Everything a release note needs about one built executable."""
    problems = check_pe(path, want_machine, want_subsystem, require_icon)
    leaks = scan_for_leaks(path)
    try:
        described: Optional[Dict[str, Any]] = describe_pe(path)
    except Exception:
        described = None
    return {"path": str(path), "pe": described, "problems": problems,
            "leaks": leaks, "ok": not problems and not leaks}


def as_text(result: Dict[str, Any]) -> str:
    """The report, for a terminal."""
    lines: List[str] = []
    pe = result.get("pe")
    if pe:
        lines.append(f"  Format          {pe['format']} ({pe['bits']}-bit)")
        lines.append(f"  Architecture    {pe['machine']}")
        lines.append(f"  Subsystem       {pe['subsystem']}")
        lines.append(f"  Entry point     {pe['entry_point']:#x}")
        lines.append(f"  Sections        {pe['sections']}")
        lines.append(f"  Size            {pe['size_mb']} MB")
        lines.append(f"  Icon            {'yes' if pe['has_icon'] else 'NO'}")
        lines.append(f"  Version info    {'yes' if pe['has_version_info'] else 'no'}")
        lines.append(f"  Bootloader      "
                     f"{'PyInstaller' if pe['has_pyinstaller_bootloader'] else 'NOT FOUND'}")
    for problem in result["problems"]:
        lines.append(f"  PROBLEM         {problem}")
    for leak in result["leaks"]:
        lines.append(f"  LEAK            {leak['what']} at {leak['offset']:#x}: "
                     f"{leak['excerpt']!r} - {leak['why']}")
    if not result["ok"]:
        lines.append("  Verdict         NOT releasable - see above")
    elif pe is None:
        # Said explicitly rather than left to be inferred: this is the case where
        # the PE checks did not run, and a verdict that reads like they passed is
        # the specific wrong claim this whole module exists to prevent.
        lines.append("  Verdict         the bundle is clean; whether this is a "
                     "Windows executable was NOT checked")
    else:
        lines.append("  Verdict         looks like a releasable Windows executable")
    return "\n".join(lines)
