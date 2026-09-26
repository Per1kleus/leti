"""The build checks that it built a Windows executable, not that a file appeared.

launcher/build_exe.py checked two things: that dist/Leti.exe existed and that its
size was plausible. Both are worth checking; neither can tell a Windows executable
from a Linux one renamed, an ARM build from an x64 one, an executable with an icon
from one without, or a clean bundle from one that swept a .env file up on the way
past. launcher/verify_build.py reads the file instead.

The PE header is the same bytes whichever machine reads them, which is why this can
be tested here. Two kinds of fixture are used:

  * a PE assembled byte by byte in this file, so every offset the parser reads is
    one this test put there deliberately
  * whatever real Windows binaries happen to be on this machine - pip ships
    several launcher stubs - checked against `file` and `objdump` when they are
    present, and skipped when they are not

Neither is a Windows build of Leti. See the release report for what that needs.
"""
from __future__ import annotations

import pathlib
import struct
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from launcher import verify_build as vb  # noqa: E402


# --------------------------------------------------------------------------- #
# A PE, assembled here
# --------------------------------------------------------------------------- #

def make_pe(machine: int = 0x8664, subsystem: int = 3, plus: bool = True,
            entry_point: int = 0x1000, sections: int = 4,
            bootloader: bool = True, resources: bool = True,
            payload: bytes = b"") -> bytes:
    """The smallest byte string this parser should read as a PE.

    Built rather than borrowed so that a parser reading the wrong offset produces a
    wrong answer here, where it is visible, instead of agreeing with a real file by
    coincidence.
    """
    optional_size = 240 if plus else 224
    pe_offset = 0x80

    head = bytearray(b"\0" * pe_offset)
    head[0:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, pe_offset)

    coff = struct.pack("<4sHHIIIHH", b"PE\0\0", machine, sections, 0x5F000000, 0, 0,
                       optional_size, 0x0022)

    optional = bytearray(b"\0" * optional_size)
    struct.pack_into("<H", optional, 0, 0x20B if plus else 0x10B)
    struct.pack_into("<I", optional, 16, entry_point)
    struct.pack_into("<H", optional, 68, subsystem)
    # NumberOfRvaAndSizes, then the data directories. Resources are entry 2.
    directories_at = 112 if plus else 96
    if resources:
        struct.pack_into("<II", optional, directories_at + 2 * 8, 0x4000, 0x200)

    # One section covering the resource RVA, so the directory can be found.
    section = struct.pack("<8sIIII12x", b".rsrc\0\0\0", 0x200, 0x4000, 0x200, 0x600)
    section += b"\0" * (40 - len(section)) if len(section) < 40 else b""

    built = bytearray(bytes(head) + coff + bytes(optional) + section)
    built.extend(b"\0" * (0x600 - len(built)))

    if resources:
        # IMAGE_RESOURCE_DIRECTORY at file offset 0x600: 12 reserved bytes, then
        # the two entry counts, then one entry per type.
        directory = bytearray(b"\0" * 16)
        struct.pack_into("<HH", directory, 12, 0, 2)
        directory += struct.pack("<II", vb.RT_GROUP_ICON, 0)
        directory += struct.pack("<II", vb.RT_VERSION, 0)
        built.extend(directory)

    if bootloader:
        built.extend(b"PyInstaller bootloader, pyi-something\0")
    built.extend(payload)
    # Pad past the tail window the bootloader cookie is looked for in, so a test
    # that asks for no bootloader really has none in range.
    built.extend(b"\0" * 64)
    return bytes(built)


@pytest.fixture
def exe(tmp_path):
    def write(**kwargs):
        path = tmp_path / "Leti.exe"
        path.write_bytes(make_pe(**kwargs))
        return path
    return write


# --- Reading it ---------------------------------------------------------------------

def test_a_windows_x64_console_build_reads_as_one(exe):
    pe = vb.describe_pe(exe())
    assert pe["format"] == "PE32+"
    assert pe["bits"] == 64
    assert pe["machine"] == "x64 (AMD64)"
    assert pe["subsystem"] == "Windows console"
    assert pe["entry_point"] == 0x1000
    assert pe["sections"] == 4
    assert pe["has_icon"] is True
    assert pe["has_version_info"] is True
    assert pe["has_pyinstaller_bootloader"] is True


def test_such_a_build_has_nothing_wrong_with_it(exe):
    assert vb.check_pe(exe()) == []
    result = vb.report(exe())
    assert result["ok"] is True
    assert "releasable" in vb.as_text(result)


@pytest.mark.parametrize("kwargs,expected", [
    ({"machine": 0x014C, "plus": False}, "x86 (32-bit)"),
    ({"machine": 0xAA64}, "ARM64"),
])
def test_the_architecture_is_read_not_assumed(exe, kwargs, expected):
    assert vb.describe_pe(exe(**kwargs))["machine"] == expected


def test_a_gui_subsystem_is_read_as_one(exe):
    assert vb.describe_pe(exe(subsystem=2))["subsystem"] == "Windows GUI"


# --- Refusing what is not one --------------------------------------------------------

@pytest.mark.parametrize("content,described", [
    (b"\x7fELF\x02\x01\x01\x00" + b"\0" * 200, "ELF"),
    (b"\xcf\xfa\xed\xfe" + b"\0" * 200, "Mach-O"),
    (b"#!/bin/sh\necho hello\n" + b"\0" * 200, "shebang"),
    (b"PK\x03\x04" + b"\0" * 200, "zip"),
    (b"", "not a recognised"),
    (b"MZ", "not a recognised"),
])
def test_anything_that_is_not_a_pe_is_refused_by_name(tmp_path, content, described):
    """"This is not a .exe" is not a useful build failure. Saying what it IS is."""
    path = tmp_path / "Leti.exe"
    path.write_bytes(content)
    with pytest.raises(vb.NotAPortableExecutable) as raised:
        vb.describe_pe(path)
    assert described in str(raised.value)
    # And through the checking front door, as a problem rather than an exception.
    problems = vb.check_pe(path)
    assert problems and described in problems[0]


def test_a_linux_binary_renamed_to_exe_does_not_pass(tmp_path):
    """The thing the brief explicitly rules out. Cross-compiling cannot produce a
    PE, so a Linux build renamed is the mistake that would look like success."""
    path = tmp_path / "Leti.exe"
    path.write_bytes(pathlib.Path(sys.executable).read_bytes()[:4096])
    assert vb.check_pe(path), "a Linux executable named Leti.exe was accepted"


def test_an_mz_header_pointing_nowhere_is_refused(tmp_path):
    body = bytearray(make_pe())
    struct.pack_into("<I", body, 0x3C, 0x7FFFFF)       # past the end of the file
    path = tmp_path / "Leti.exe"
    path.write_bytes(bytes(body))
    with pytest.raises(vb.NotAPortableExecutable):
        vb.describe_pe(path)


# --- The release rules ---------------------------------------------------------------

def test_the_wrong_architecture_fails_the_check(exe):
    problems = vb.check_pe(exe(machine=0xAA64))
    assert any("ARM64" in p for p in problems)


def test_a_32_bit_build_fails_the_check(exe):
    problems = vb.check_pe(exe(machine=0x014C, plus=False))
    assert any("32-bit" in p for p in problems)


def test_a_windowed_build_fails_the_check(exe):
    """Leti.exe is a console program on purpose: the first run has something to
    say, and leti_launcher.hide_console() puts the window away afterwards."""
    problems = vb.check_pe(exe(subsystem=2))
    assert any("subsystem" in p and "nowhere to go" in p for p in problems)


def test_a_missing_icon_fails_the_check(exe):
    problems = vb.check_pe(exe(resources=False))
    assert any("icon" in p for p in problems)
    assert vb.check_pe(exe(resources=False), require_icon=False) == \
        [p for p in vb.check_pe(exe(resources=False)) if "icon" not in p]


def test_a_zero_entry_point_fails_the_check(exe):
    problems = vb.check_pe(exe(entry_point=0))
    assert any("entry point" in p for p in problems)


def test_something_not_built_from_the_spec_fails_the_check(exe):
    problems = vb.check_pe(exe(bootloader=False))
    assert any("bootloader" in p for p in problems)


def test_every_problem_is_reported_not_just_the_first(exe):
    problems = vb.check_pe(exe(machine=0x014C, plus=False, subsystem=2,
                               resources=False, bootloader=False))
    assert len(problems) >= 4, f"only {len(problems)} reported: {problems}"


# --- What it must not contain ---------------------------------------------------------

@pytest.mark.parametrize("planted,what", [
    (b"path/to/.env", "a .env file"),
    (b"AKIAIOSFODNN7EXAMPLE", "an AWS access key"),
    (b"-----BEGIN RSA PRIVATE KEY-----", "a private key"),
    (b"ghp_0123456789abcdefghij", "a GitHub token"),
    (b"xoxb-1234567890-abcdefg", "a Slack token"),
    (b"sk-" + b"a" * 40, "an OpenAI-style key"),
    (b"Authorization: Bearer abcdef0123456789xyz", "a bearer token"),
    (b"config\\settings.local.yaml", "Leti's own settings override"),
    (b"data/gui_remote_token.txt", "the GUI's remote token file"),
    (b"tests/test_orchestrator.py", "a test fixture"),
    (b"project/.git/HEAD", "a .git directory"),
    (b"C:\\Users\\alice\\leti\\build", "a developer home directory"),
    (b"/home/someone/leti/build", "a Unix developer home directory"),
])
def test_a_planted_secret_is_found(exe, planted, what):
    findings = vb.scan_for_leaks(exe(payload=planted))
    assert [f for f in findings if f["what"] == what], \
        f"{planted!r} was not reported as {what}; got {[f['what'] for f in findings]}"


def test_a_finding_says_where_and_why_without_printing_the_secret(exe):
    findings = vb.scan_for_leaks(exe(payload=b"ghp_" + b"z" * 40))
    finding = findings[0]
    assert finding["offset"] > 0
    assert finding["why"]
    assert len(finding["excerpt"]) <= 80, "a finding printed an unbounded excerpt"


def test_a_clean_build_reports_nothing(exe):
    assert vb.scan_for_leaks(exe()) == []


def test_an_excluded_module_name_is_not_a_finding(exe):
    """The spec excludes pytest and unittest; PyInstaller's module tables still
    mention them. A name is not a bundled file."""
    assert vb.scan_for_leaks(exe(payload=b"pytest\0unittest\0doctest\0")) == []


def test_a_leak_fails_the_report_even_when_the_pe_is_perfect(exe):
    result = vb.report(exe(payload=b"-----BEGIN PRIVATE KEY-----"))
    assert result["problems"] == []
    assert result["ok"] is False
    assert "LEAK" in vb.as_text(result)


# --- Against real Windows binaries, when this machine has any --------------------------

def _real_pe_files():
    """Windows launcher stubs that pip's own dependencies ship, if present."""
    found = []
    for base in (pathlib.Path("/root/.local/share/uv"), pathlib.Path("/usr/local/lib"),
                 pathlib.Path(sys.prefix)):
        if not base.exists():
            continue
        try:
            found.extend(p for p in base.rglob("*.exe") if p.is_file())
        except OSError:
            continue
        if len(found) > 8:
            break
    return found[:8]


@pytest.mark.parametrize("path", _real_pe_files() or [None])
def test_the_parser_agrees_with_file_about_a_real_windows_binary(path):
    """Ground truth from something that did not come out of this file.

    `file` reads the same header independently. Where they disagree, the parser is
    wrong - and an offset that is right for a PE built in this test and wrong for a
    real one is exactly the mistake a self-built fixture cannot catch.
    """
    if path is None:
        pytest.skip("no real Windows binaries on this machine to check against")
    described = subprocess.run(["file", "-b", str(path)],
                               capture_output=True, text=True).stdout.strip()
    if "for MS Windows" not in described:
        pytest.skip(f"{path.name} is not a PE according to file(1)")

    pe = vb.describe_pe(path)
    assert pe["format"] in described, f"file says {described!r}, parser says {pe['format']}"
    if "x86-64" in described:
        assert pe["machine_id"] == 0x8664
    elif "Aarch64" in described:
        assert pe["machine"] == "ARM64"
    elif "80386" in described:
        assert pe["machine_id"] == 0x014C
    if "(console)" in described:
        assert pe["subsystem_id"] == 3
    elif "(GUI)" in described:
        assert pe["subsystem_id"] == 2
    if "sections" in described:
        counted = int(described.split(",")[-1].strip().split()[0])
        assert pe["sections"] == counted, "the section count disagrees"
    assert pe["entry_point"] > 0


def test_the_build_script_actually_runs_these_checks():
    """A verifier nothing calls is a verifier that does not verify."""
    import ast

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "launcher" / "build_exe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    assert any("verify_build" in name for name in imported), \
        "build_exe.py does not use launcher/verify_build.py"
    # And actually calls it, rather than importing it and moving on.
    called = {ast.unparse(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}
    assert {"verify_build.report", "verify_build.scan_for_leaks"} & called, \
        "build_exe.py imports the verifier without running it"


# --- The build is reproducible --------------------------------------------------------

def test_the_build_pins_the_hash_seed():
    """Two clean builds of the same source produced two different executables.

    Measured on this project before the fix: 2 MB of differing bytes and a
    1,264-byte difference in size between consecutive clean builds. The cause is
    Python's per-process hash randomisation - PyInstaller walks sets and dicts of
    module names while assembling the archive, so the member order follows string
    hashes and changes every process. With PYTHONHASHSEED=0 two clean builds come
    out byte-identical with the same SHA-256.

    A release nobody can reproduce is a release nobody can check against its
    source, so this is pinned rather than trusted to whoever runs the build.
    """
    from launcher import build_exe

    environment = build_exe.build_environment()
    assert environment.get("PYTHONHASHSEED") == "0"
    # The ambient environment is carried through, not replaced - PATH and the
    # proxy variables a build may need are still there.
    import os

    for name in list(os.environ)[:5]:
        assert name in environment


def test_the_build_actually_uses_that_environment():
    """A function nothing passes to subprocess.run is a function that fixes nothing."""
    import ast

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "launcher" / "build_exe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) != "subprocess.run":
            continue
        if "PyInstaller" not in ast.unparse(node):
            continue
        passed = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
        assert passed.get("env") == "build_environment()", \
            "the PyInstaller subprocess does not get the pinned environment"
        return
    pytest.fail("no subprocess.run call building PyInstaller was found")


def test_leti_itself_is_not_run_with_a_fixed_hash_seed():
    """Only the BUILD is pinned. Hash randomisation is a defence at runtime, and
    launcher/bootstrap.py starts Leti - it must not inherit this."""
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "launcher" / "bootstrap.py").read_text(encoding="utf-8")
    assert "PYTHONHASHSEED" not in source


# --- The built executable ends up where everything looks for it ------------------------

def test_the_build_puts_the_executable_beside_main_py():
    """PyInstaller writes to dist\\, and nothing looks for it there.

    launcher/shortcuts.py's TARGETS names "Leti.exe" relative to the project root,
    so a build left in dist\\ meant the Desktop shortcut kept pointing at the .bat
    and the executable was a file the user had to know to move. dist\\ is also what
    --clean empties, so it is copied out rather than built there.
    """
    import ast

    from launcher import shortcuts

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "launcher" / "build_exe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert "shutil.copy2" in calls, "the build never copies the executable out of dist/"
    assert "ROOT / BUILT.name" in source, "it is not copied to the project root"

    # And that is the name the shortcut looks for.
    assert "Leti.exe" in shortcuts.TARGETS
    assert shortcuts.TARGETS[0] == "Leti.exe", \
        "the executable is no longer preferred over the .bat"


def test_there_is_a_double_click_way_to_build_it():
    """The project's whole premise is not needing a terminal; building was the one
    thing that did."""
    root = pathlib.Path(__file__).resolve().parent.parent
    batch = root / "Build Leti.exe (Windows).bat"
    assert batch.is_file(), "there is no double-click build"
    text = batch.read_text(encoding="utf-8")
    assert "launcher\\build_exe.py" in text
    # It must use the environment a launcher prepared, not a system Python.
    assert "leti_env\\Scripts\\python.exe" in text
    assert "leti_runtime\\python.exe" in text
    # And say what to do when there is none, rather than failing obscurely.
    assert "Launch Leti (Windows).bat" in text
    assert "pause" in text
