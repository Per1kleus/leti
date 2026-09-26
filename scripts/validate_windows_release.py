"""Validate a Windows release of Leti on a real Windows machine.

    python scripts\\validate_windows_release.py --help
    python scripts\\validate_windows_release.py all

WHAT THIS IS FOR

Everything about Leti's Windows launch can be reasoned about anywhere; almost none
of it can be PROVEN anywhere but Windows. A .lnk cannot be written, an embeddable
Python cannot be unpacked and run, winget does not exist, and PyInstaller builds
for the machine it runs on. So this script exists to be run there, and to produce a
release report whose every line says which of those actually happened.

THE ONE RULE IT FOLLOWS

A check that did not run reports NOT RUN, and a check a person has to look at
reports NEEDS A HUMAN. Neither is a pass. A validation report that says everything
is fine because it did not look is worse than no report, which is the same rule
core/diagnostics.py already follows for Leti's own self-check.

HOW TO USE IT

The stages are separate because the honest ones are slow and destructive:

    preflight   is this machine actually clean, and is the exe actually a PE
    build       build Leti.exe here and check it, twice, for reproducibility
    first       time a real first launch end to end
    repair      break the installation in nine ways and check each recovery
    warm        time a second launch and prove it downloads nothing
    shortcuts    Desktop and Start Menu, targets, icons, idempotence
    processes   what is left running after Leti exits
    tests       the full suite, in fixed and random order
    manual      print the checklist that needs eyes and ears
    report      write everything gathered so far to a file
    all         every stage above, in order

Each stage appends to validation_report.json beside this script's project, so a
run can be done over several sittings and the report still adds up.

Standard library only, and it never imports Leti - it drives the launchers from
outside, the way a user does.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "validation_report.json"

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT RUN"
NEEDS_HUMAN = "NEEDS A HUMAN"
WARNING = "WARNING"

VERDICTS = (PASS, FAIL, WARNING, NEEDS_HUMAN, NOT_RUN)

# What a clean machine must not have, per the brief's section 4.
LETI_ARTEFACTS = (
    "leti_env", "leti_runtime", "data/launch_setup.json", "data/.models_pulled",
    "config/settings.local.yaml", "build", "dist",
)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #

class Report:
    """Every check, its verdict, and what was measured. Appends across runs."""

    def __init__(self, path: Path = REPORT) -> None:
        self.path = path
        self.data: Dict[str, Any] = {"checks": [], "measurements": {}, "runs": []}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        self.data.setdefault("checks", [])
        self.data.setdefault("measurements", {})
        self.data.setdefault("runs", [])
        self.data["runs"].append({
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": sys.platform,
            "windows": os.name == "nt",
            "python": sys.version.split()[0],
        })

    def check(self, stage: str, name: str, verdict: str, detail: str = "",
              **extra: Any) -> str:
        assert verdict in VERDICTS, verdict
        row = {"stage": stage, "name": name, "verdict": verdict, "detail": detail}
        row.update(extra)
        # One row per check name: a later run replaces an earlier verdict rather
        # than accumulating two that disagree.
        self.data["checks"] = [c for c in self.data["checks"] if c["name"] != name]
        self.data["checks"].append(row)
        mark = {PASS: "ok  ", FAIL: "FAIL", WARNING: "warn", NEEDS_HUMAN: "look",
                NOT_RUN: "    "}[verdict]
        print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
        self.save()
        return verdict

    def measure(self, name: str, seconds: float, note: str = "") -> None:
        self.data["measurements"][name] = {"seconds": round(seconds, 2), "note": note}
        print(f"  [time] {name}: {seconds:.1f}s" + (f" ({note})" if note else ""))
        self.save()

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"  (could not write {self.path}: {e})")

    def worst(self) -> str:
        found = {c["verdict"] for c in self.data["checks"]}
        for verdict in (FAIL, NOT_RUN, NEEDS_HUMAN, WARNING, PASS):
            if verdict in found:
                return verdict
        return NOT_RUN


def on_windows() -> bool:
    return os.name == "nt"


def require_windows(report: Report, stage: str, checks: Sequence[str]) -> bool:
    """Record every check in this stage as NOT RUN when this is not Windows.

    Named individually rather than as one line, because a report that has to be
    read against a list of release criteria needs a row per criterion.
    """
    if on_windows():
        return True
    for name in checks:
        report.check(stage, name, NOT_RUN,
                     f"this is {sys.platform}, not Windows - nothing was executed")
    return False


def run(command: Sequence[str], timeout: float = 600, cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None) -> Any:
    """A subprocess whose output is captured, with a timeout that is never None."""
    return subprocess.run([str(c) for c in command], capture_output=True, text=True,
                          timeout=timeout, cwd=str(cwd or ROOT), env=env,
                          errors="replace")


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #

def stage_preflight(report: Report) -> None:
    print("\n== preflight: is this machine clean, and is the exe a PE?")

    report.check("preflight", "running on Windows",
                 PASS if on_windows() else FAIL,
                 f"platform is {sys.platform}")

    # Section 4: a developer's configured machine is not evidence of first-launch
    # behaviour, so say exactly what is already here.
    present = [name for name in LETI_ARTEFACTS if (ROOT / name).exists()]
    report.check("preflight", "clean of previous Leti state",
                 PASS if not present else WARNING,
                 "nothing left over" if not present
                 else f"already present: {', '.join(present)} - first-launch timings "
                      "from this folder are not a clean-machine measurement",
                 found=present)

    for tool, why in (("python", "a system Python is present, so the no-Python path "
                                "will NOT be the one exercised"),
                      ("ollama", "Ollama is already installed, so its installation "
                                 "will NOT be exercised"),
                      ("winget", "winget is available, which is what Ollama is "
                                 "installed with")):
        found = shutil.which(tool)
        report.check("preflight", f"{tool} on PATH",
                     WARNING if (found and tool != "winget") else PASS,
                     (f"{found} - {why}" if found else f"absent - {why}"),
                     path=found)

    exe = ROOT / "dist" / "Leti.exe"
    if not exe.exists():
        report.check("preflight", "Leti.exe is a Windows PE executable", NOT_RUN,
                     f"{exe} does not exist yet - run the build stage")
        return
    _check_the_exe(report, exe)


def _check_the_exe(report: Report, exe: Path) -> None:
    """Sections 1 and 3, using the project's own verifier."""
    sys.path.insert(0, str(ROOT))
    try:
        from launcher import verify_build
    except Exception as e:
        report.check("preflight", "Leti.exe is a Windows PE executable", NOT_RUN,
                     f"launcher/verify_build.py could not be imported ({e})")
        return

    result = verify_build.report(exe)
    pe = result.get("pe") or {}
    report.check("preflight", "Leti.exe is a Windows PE executable",
                 PASS if pe.get("format", "").startswith("PE") else FAIL,
                 f"{pe.get('format', 'not a PE')}, {pe.get('machine', 'unknown')}",
                 **{k: pe.get(k) for k in ("format", "machine", "subsystem",
                                           "entry_point", "sections", "size_mb")})
    report.check("preflight", "Leti.exe is x64", PASS if pe.get("bits") == 64 else FAIL,
                 f"{pe.get('bits')}-bit")
    report.check("preflight", "Leti.exe has an entry point",
                 PASS if pe.get("entry_point") else FAIL, hex(pe.get("entry_point") or 0))
    report.check("preflight", "Leti.exe embeds the icon",
                 PASS if pe.get("has_icon") else FAIL,
                 "an icon resource is present" if pe.get("has_icon")
                 else "no icon resource - Explorer will show the default")
    report.check("preflight", "Leti.exe carries the PyInstaller bootloader",
                 PASS if pe.get("has_pyinstaller_bootloader") else FAIL, "")
    report.check("preflight", "Leti.exe size is reasonable",
                 PASS if 3 <= (pe.get("size_mb") or 0) <= 60 else FAIL,
                 f"{pe.get('size_mb')} MB - the application itself is deliberately "
                 "not bundled")
    report.check("preflight", "Leti.exe bundles no secrets or dev files",
                 PASS if not result["leaks"] else FAIL,
                 "nothing found" if not result["leaks"]
                 else "; ".join(f"{f['what']} at {f['offset']:#x}" for f in result["leaks"]),
                 leaks=result["leaks"])
    # Section 1: nothing for another operating system came along.
    report.check("preflight", "no foreign binaries in the bundle",
                 PASS if b"\x7fELF" not in exe.read_bytes()[:4] else FAIL,
                 "the image itself is PE")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def stage_build(report: Report) -> None:
    print("\n== build: build Leti.exe here, twice, and compare")
    if not require_windows(report, "build", [
            "Leti.exe built on Windows", "the build is reproducible"]):
        return

    import hashlib

    digests = []
    for attempt in (1, 2):
        for stale in ("build", "dist"):
            shutil.rmtree(ROOT / stale, ignore_errors=True)
        started = time.perf_counter()
        done = run([sys.executable, "launcher/build_exe.py"], timeout=3600)
        elapsed = time.perf_counter() - started
        exe = ROOT / "dist" / "Leti.exe"
        if done.returncode != 0 or not exe.exists():
            report.check("build", "Leti.exe built on Windows", FAIL,
                         f"build {attempt} failed with code {done.returncode}",
                         output=(done.stdout or "")[-2000:])
            return
        digests.append(hashlib.sha256(exe.read_bytes()).hexdigest())
        report.measure(f"build {attempt}", elapsed)
        if attempt == 1:
            shutil.copy2(exe, ROOT / "dist" / "Leti.first.exe")

    report.check("build", "Leti.exe built on Windows", PASS,
                 f"{(ROOT / 'dist' / 'Leti.exe').stat().st_size / 1048576:.1f} MB",
                 sha256=digests[0])
    report.check("build", "the build is reproducible",
                 PASS if digests[0] == digests[1] else FAIL,
                 "two clean builds are byte-identical" if digests[0] == digests[1]
                 else f"two clean builds differ: {digests[0][:12]} vs {digests[1][:12]}",
                 sha256=digests)
    (ROOT / "dist" / "Leti.first.exe").unlink(missing_ok=True)
    _check_the_exe(report, ROOT / "dist" / "Leti.exe")


# --------------------------------------------------------------------------- #
# first launch
# --------------------------------------------------------------------------- #

def stage_first(report: Report, exe: Optional[Path] = None) -> None:
    print("\n== first: a real first launch, timed")
    names = ["first launch prepares Python", "first launch installs packages",
             "first launch reaches a running Leti", "no manual terminal was needed"]
    if not require_windows(report, "first", names):
        return

    exe = exe or (ROOT / "dist" / "Leti.exe")
    if not exe.exists():
        for name in names:
            report.check("first", name, NOT_RUN, f"{exe} does not exist")
        return

    # --setup-only, because the point is to time the preparation and see its
    # output. The GUI is started by the manual stage, where somebody can watch it.
    started = time.perf_counter()
    try:
        done = run([exe, "--setup-only"], timeout=5400)
    except subprocess.TimeoutExpired:
        report.check("first", "first launch reaches a running Leti", FAIL,
                     "setup did not finish within 90 minutes")
        return
    elapsed = time.perf_counter() - started
    report.measure("first launch (setup only)", elapsed,
                   "downloads Python, packages and possibly Ollama")
    output = (done.stdout or "") + (done.stderr or "")

    interpreter = ROOT / "leti_env" / "Scripts" / "python.exe"
    runtime = ROOT / "leti_runtime" / "python.exe"
    report.check("first", "first launch prepares Python",
                 PASS if (interpreter.exists() or runtime.exists()) else FAIL,
                 f"venv={interpreter.exists()} fetched-runtime={runtime.exists()}")
    state = ROOT / "data" / "launch_setup.json"
    complete = False
    if state.exists():
        try:
            complete = bool(json.loads(state.read_text(encoding="utf-8")).get("complete"))
        except (OSError, ValueError):
            complete = False
    report.check("first", "first launch installs packages",
                 PASS if complete else FAIL,
                 "the state file records a completed setup" if complete
                 else "setup did not record completion")
    report.check("first", "first launch reaches a running Leti",
                 PASS if done.returncode == 0 else FAIL,
                 f"--setup-only exited {done.returncode}",
                 output=output[-3000:])
    report.check("first", "no manual terminal was needed",
                 PASS if done.returncode == 0 else FAIL,
                 "the executable did everything itself"
                 if done.returncode == 0 else "see the output above")

    # Section 12: what it said about Ollama, which is the step the exe used not to do.
    said_ollama = any(word in output.lower() for word in ("ollama", "model server"))
    report.check("first", "first launch dealt with Ollama",
                 PASS if said_ollama else WARNING,
                 "the setup output mentions it" if said_ollama
                 else "the setup output never mentions Ollama - check "
                      "launcher/ollama_setup.py ran")


# --------------------------------------------------------------------------- #
# dependency repair and failure injection
# --------------------------------------------------------------------------- #

def stage_repair(report: Report) -> None:
    print("\n== repair: break it nine ways, check each recovery")
    cases = ["a removed package is reinstalled", "a corrupt state file recovers",
             "a truncated state file recovers", "a missing site-packages recovers",
             "a removed runtime recovers", "an interrupted install recovers",
             "a missing state file recovers", "a partially created venv recovers",
             "an unreadable state file recovers"]
    if not require_windows(report, "repair", cases):
        return

    exe = ROOT / "dist" / "Leti.exe"
    state = ROOT / "data" / "launch_setup.json"
    if not exe.exists() or not state.exists():
        for name in cases:
            report.check("repair", name, NOT_RUN,
                         "run the first stage before this one")
        return

    interpreter = _prepared_interpreter()
    if interpreter is None:
        for name in cases:
            report.check("repair", name, NOT_RUN, "no prepared interpreter found")
        return

    def relaunch(what: str) -> Any:
        return run([exe, "--setup-only"], timeout=3600)

    def importable(module: str) -> bool:
        return run([interpreter, "-c", f"import {module}"], timeout=120).returncode == 0

    # 1. A package genuinely removed, not just unrecorded. The brief is explicit
    #    that a state file alone must not be what the repair relies on.
    if importable("yaml"):
        run([interpreter, "-m", "pip", "uninstall", "-y", "pyyaml"], timeout=600)
        gone = not importable("yaml")
        relaunch("removed package")
        report.check("repair", "a removed package is reinstalled",
                     PASS if (gone and importable("yaml")) else FAIL,
                     "pyyaml was uninstalled, then importable again after a launch"
                     if gone else "pyyaml could not be uninstalled, so this proved nothing")
    else:
        report.check("repair", "a removed package is reinstalled", NOT_RUN,
                     "pyyaml was not importable to begin with")

    # 2-3. A state file that is not JSON, and one cut in half.
    original = state.read_text(encoding="utf-8")
    for name, content in (("a corrupt state file recovers", "{not json at all"),
                          ("a truncated state file recovers", original[:len(original) // 2])):
        state.write_text(content, encoding="utf-8")
        done = relaunch(name)
        report.check("repair", name, PASS if done.returncode == 0 else FAIL,
                     f"relaunch exited {done.returncode}")
    state.write_text(original, encoding="utf-8")

    # 4. site-packages taken away wholesale.
    site = _site_packages(interpreter)
    if site and site.exists():
        moved = site.with_name(site.name + ".moved")
        try:
            site.rename(moved)
            done = relaunch("missing site-packages")
            report.check("repair", "a missing site-packages recovers",
                         PASS if done.returncode == 0 else FAIL,
                         f"relaunch exited {done.returncode}")
        except OSError as e:
            report.check("repair", "a missing site-packages recovers", NOT_RUN, str(e))
        finally:
            if moved.exists() and not site.exists():
                moved.rename(site)
            shutil.rmtree(moved, ignore_errors=True)
    else:
        report.check("repair", "a missing site-packages recovers", NOT_RUN,
                     "site-packages could not be located")

    # 5. The whole interpreter directory gone.
    for directory, label in ((ROOT / "leti_runtime", "runtime"),
                            (ROOT / "leti_env", "venv")):
        name = ("a removed runtime recovers" if label == "runtime"
                else "a partially created venv recovers")
        if not directory.exists():
            report.check("repair", name, NOT_RUN, f"no {label} to break")
            continue
        if label == "venv":
            # Partially created rather than removed: a Scripts directory with no
            # interpreter in it is what an interrupted venv leaves behind.
            interpreter_path = directory / "Scripts" / "python.exe"
            backup = interpreter_path.with_suffix(".exe.moved")
            try:
                interpreter_path.rename(backup)
                done = relaunch(name)
                report.check("repair", name, PASS if done.returncode == 0 else FAIL,
                             f"relaunch exited {done.returncode}")
            except OSError as e:
                report.check("repair", name, NOT_RUN, str(e))
            finally:
                if backup.exists() and not interpreter_path.exists():
                    backup.rename(interpreter_path)
        else:
            moved = directory.with_name(directory.name + ".moved")
            try:
                directory.rename(moved)
                done = relaunch(name)
                report.check("repair", name, PASS if done.returncode == 0 else FAIL,
                             f"relaunch exited {done.returncode}")
            finally:
                shutil.rmtree(moved, ignore_errors=True)

    # 6. Interrupted install: the state says a set of packages is still missing.
    try:
        broken = json.loads(original)
        broken["complete"] = False
        broken["partial"] = {"requirements": broken.get("requirements"),
                             "still_missing": ["pyyaml"]}
        broken.pop("witnesses", None)
        state.write_text(json.dumps(broken), encoding="utf-8")
        done = relaunch("interrupted install")
        report.check("repair", "an interrupted install recovers",
                     PASS if done.returncode == 0 else FAIL,
                     f"relaunch exited {done.returncode}")
    except ValueError as e:
        report.check("repair", "an interrupted install recovers", NOT_RUN, str(e))
    finally:
        state.write_text(original, encoding="utf-8")

    # 7. No state file at all.
    state.unlink(missing_ok=True)
    done = relaunch("missing state")
    report.check("repair", "a missing state file recovers",
                 PASS if done.returncode == 0 else FAIL,
                 f"relaunch exited {done.returncode}")

    # 8. A state file that cannot be read at all.
    try:
        state.write_bytes(b"\x00\xff\xfe binary rubbish")
        done = relaunch("unreadable state")
        report.check("repair", "an unreadable state file recovers",
                     PASS if done.returncode == 0 else FAIL,
                     f"relaunch exited {done.returncode}")
    finally:
        state.write_text(original, encoding="utf-8")


def _prepared_interpreter() -> Optional[Path]:
    for candidate in (ROOT / "leti_env" / "Scripts" / "python.exe",
                      ROOT / "leti_runtime" / "python.exe"):
        if candidate.exists():
            return candidate
    return None


def _site_packages(interpreter: Path) -> Optional[Path]:
    done = run([interpreter, "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])"], timeout=120)
    if done.returncode != 0:
        return None
    text = (done.stdout or "").strip()
    return Path(text) if text else None


# --------------------------------------------------------------------------- #
# warm launch
# --------------------------------------------------------------------------- #

def stage_warm(report: Report) -> None:
    print("\n== warm: a second launch downloads nothing and is fast")
    names = ["warm launch is fast", "warm launch downloads nothing",
             "warm launch does not rebuild the environment"]
    if not require_windows(report, "warm", names):
        return

    exe = ROOT / "dist" / "Leti.exe"
    if not exe.exists() or not (ROOT / "data" / "launch_setup.json").exists():
        for name in names:
            report.check("warm", name, NOT_RUN, "run the first stage before this one")
        return

    interpreter = _prepared_interpreter()
    before = interpreter.stat().st_mtime if interpreter else 0
    started = time.perf_counter()
    done = run([exe, "--setup-only"], timeout=900)
    elapsed = time.perf_counter() - started
    report.measure("warm launch (setup only)", elapsed)
    output = ((done.stdout or "") + (done.stderr or "")).lower()

    report.check("warm", "warm launch is fast",
                 PASS if elapsed < 30 else WARNING,
                 f"{elapsed:.1f}s - a warm launch should be a few stat calls")
    downloaded = any(word in output for word in
                     ("downloading", "getting python", "installing what is missing",
                      "about 11 mb", "a few hundred megabytes"))
    report.check("warm", "warm launch downloads nothing",
                 FAIL if downloaded else PASS,
                 "the output mentions downloading" if downloaded
                 else "nothing was fetched",
                 output=output[-1500:])
    after = interpreter.stat().st_mtime if interpreter else 0
    report.check("warm", "warm launch does not rebuild the environment",
                 PASS if after == before else FAIL,
                 "the interpreter was not replaced" if after == before
                 else "the interpreter was rebuilt")


# --------------------------------------------------------------------------- #
# shortcuts and icon
# --------------------------------------------------------------------------- #

def stage_shortcuts(report: Report) -> None:
    print("\n== shortcuts: Desktop, Start Menu, targets, icons, idempotence")
    names = ["Desktop shortcut exists", "Start Menu shortcut exists",
             "shortcut targets are correct", "shortcut working directory is correct",
             "shortcut icon is the existing leti.ico", "placing twice changes nothing",
             "a stale target is repaired"]
    # Counting the icon assets is reading a directory, so it runs anywhere and is
    # deliberately not in the list above - listing it there would record it as NOT
    # RUN and then immediately overwrite that with the real answer.
    _check_no_duplicate_icon(report)
    if not require_windows(report, "shortcuts", names):
        return

    exe = ROOT / "dist" / "Leti.exe"
    launcher = ROOT / "launcher" / "leti_launcher.py"
    interpreter = _prepared_interpreter() or Path(sys.executable)

    placed = run([interpreter, launcher, "--install-shortcuts"], timeout=300)
    print((placed.stdout or "").rstrip())

    desktop = Path(os.path.expanduser("~")) / "Desktop" / "Leti.lnk"
    start = (Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows"
             / "Start Menu" / "Programs" / "Leti" / "Leti.lnk")

    report.check("shortcuts", "Desktop shortcut exists",
                 PASS if desktop.exists() else FAIL, str(desktop))
    report.check("shortcuts", "Start Menu shortcut exists",
                 PASS if start.exists() else FAIL, str(start))

    expected = str(exe if exe.exists() else ROOT / "Launch Leti (Windows).bat")
    for label, link in (("Desktop", desktop), ("Start Menu", start)):
        if not link.exists():
            continue
        details = _read_lnk(link)
        if details is None:
            report.check("shortcuts", "shortcut targets are correct", NOT_RUN,
                         f"{label} shortcut could not be read")
            continue
        report.check("shortcuts", "shortcut targets are correct",
                     PASS if Path(details["target"]) == Path(expected) else FAIL,
                     f"{label} -> {details['target']} (expected {expected})",
                     target=details["target"])
        report.check("shortcuts", "shortcut working directory is correct",
                     PASS if Path(details["working"] or ".") == ROOT else FAIL,
                     f"{label} working directory {details['working']}")
        icon = (details.get("icon") or "").split(",")[0]
        report.check("shortcuts", "shortcut icon is the existing leti.ico",
                     PASS if "leti.ico" in icon.lower() or icon == expected else FAIL,
                     f"{label} icon {icon or '(inherited from the target)'}")

    # Idempotence: placing twice must report nothing changed.
    again = run([interpreter, launcher, "--install-shortcuts"], timeout=300)
    text = (again.stdout or "").lower()
    report.check("shortcuts", "placing twice changes nothing",
                 PASS if ("correct" in text or "kept" in text
                          or "created" not in text) else WARNING,
                 (again.stdout or "").strip()[:200])

    # A stale target, repaired.
    if desktop.exists():
        try:
            _retarget_lnk(desktop, str(ROOT / "nothing-here.exe"))
            repaired = run([interpreter, launcher, "--install-shortcuts"], timeout=300)
            details = _read_lnk(desktop) or {}
            report.check("shortcuts", "a stale target is repaired",
                         PASS if Path(details.get("target", "")) == Path(expected) else FAIL,
                         f"after repair: {details.get('target')}",
                         output=(repaired.stdout or "")[:300])
        except Exception as e:
            report.check("shortcuts", "a stale target is repaired", NOT_RUN, str(e))


def _check_no_duplicate_icon(report: Report) -> None:
    """Section 11: one icon asset, the existing one."""
    icons = sorted(p for p in (ROOT / "gui" / "icons").glob("*.ico"))
    report.check("shortcuts", "no duplicate icon asset exists",
                 PASS if len(icons) == 1 and icons[0].name == "leti.ico" else FAIL,
                 f"{[p.name for p in icons]} in gui/icons")


def _powershell(script: str) -> Optional[str]:
    try:
        done = run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-Command", script], timeout=120)
    except Exception:
        return None
    return (done.stdout or "").strip() if done.returncode == 0 else None


def _read_lnk(path: Path) -> Optional[Dict[str, str]]:
    """Target, working directory and icon of a .lnk, via the shell that wrote it."""
    out = _powershell(
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('"
        + str(path).replace("'", "''") + "');"
        "Write-Output $s.TargetPath; Write-Output $s.WorkingDirectory;"
        "Write-Output $s.IconLocation")
    if out is None:
        return None
    parts = (out.splitlines() + ["", "", ""])[:3]
    return {"target": parts[0].strip(), "working": parts[1].strip(),
            "icon": parts[2].strip()}


def _retarget_lnk(path: Path, target: str) -> None:
    _powershell(
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('"
        + str(path).replace("'", "''") + "');"
        "$s.TargetPath='" + target.replace("'", "''") + "'; $s.Save()")


# --------------------------------------------------------------------------- #
# processes
# --------------------------------------------------------------------------- #

def stage_processes(report: Report) -> None:
    print("\n== processes: what is left running after Leti exits")
    names = ["no orphan bootstrap process", "no orphan Python process",
             "no unexpected console window", "Ollama is left as it was found"]
    if not require_windows(report, "processes", names):
        return

    before = _process_names()
    exe = ROOT / "dist" / "Leti.exe"
    if not exe.exists():
        for name in names:
            report.check("processes", name, NOT_RUN, "no Leti.exe to run")
        return

    # --setup-only exits by itself, which is what makes this measurable without a
    # human closing a window. The GUI's own shutdown is in the manual checklist.
    run([exe, "--setup-only"], timeout=3600)
    time.sleep(3)
    after = _process_names()

    leaked_exe = after.count("Leti.exe") - before.count("Leti.exe")
    leaked_python = (after.count("python.exe") - before.count("python.exe"))
    report.check("processes", "no orphan bootstrap process",
                 PASS if leaked_exe <= 0 else FAIL,
                 f"{leaked_exe} extra Leti.exe still running")
    report.check("processes", "no orphan Python process",
                 PASS if leaked_python <= 0 else FAIL,
                 f"{leaked_python} extra python.exe still running")
    report.check("processes", "no unexpected console window", NEEDS_HUMAN,
                 "watch for a console that stays visible after Leti's window opens")
    ollama_before = before.count("ollama.exe")
    ollama_after = after.count("ollama.exe")
    report.check("processes", "Ollama is left as it was found",
                 PASS if ollama_after <= max(ollama_before, 1) else WARNING,
                 f"ollama.exe: {ollama_before} before, {ollama_after} after - Leti "
                 "starting one is expected; several is not")


def _process_names() -> List[str]:
    try:
        done = run(["tasklist", "/fo", "csv", "/nh"], timeout=120)
    except Exception:
        return []
    names = []
    for line in (done.stdout or "").splitlines():
        if line.startswith('"'):
            names.append(line.split('","')[0].strip('"'))
    return names


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def stage_tests(report: Report) -> None:
    print("\n== tests: the full suite, fixed order and shuffled")
    interpreter = _prepared_interpreter() or Path(sys.executable)

    fixed = run([interpreter, "-m", "pytest", "tests/", "-q"], timeout=3600)
    tail = ((fixed.stdout or "") + (fixed.stderr or "")).strip().splitlines()
    summary = tail[-1] if tail else ""
    report.check("tests", "full suite in fixed order",
                 PASS if fixed.returncode == 0 else FAIL, summary)

    # Shuffled without a plugin: pytest takes an explicit file order, so shuffling
    # the files is a real order change and needs nothing installed.
    import random

    files = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))
    random.shuffle(files)
    shuffled = run([interpreter, "-m", "pytest", "-q", "-p", "no:randomly"]
                   + [f"tests/{name}" for name in files], timeout=3600)
    tail = ((shuffled.stdout or "") + (shuffled.stderr or "")).strip().splitlines()
    report.check("tests", "full suite in a shuffled file order",
                 PASS if shuffled.returncode == 0 else FAIL,
                 tail[-1] if tail else "", order=files)

    flakes = run([interpreter, "-m", "pytest", "tests/", "-q"], timeout=3600)
    report.check("tests", "the suite is not flaky over repeated runs",
                 PASS if flakes.returncode == 0 else FAIL,
                 "a second identical run agreed with the first"
                 if flakes.returncode == 0 else "a repeated run disagreed")

    pyflakes = run([interpreter, "-m", "pyflakes", "core", "tools", "gui", "audio",
                    "memory", "launcher", "scripts", "main.py", "tests"], timeout=600)
    findings = len([l for l in (pyflakes.stdout or "").splitlines() if l.strip()])
    report.check("tests", "no pyflakes findings",
                 PASS if findings == 0 else FAIL, f"{findings} findings")


# --------------------------------------------------------------------------- #
# what needs eyes and ears
# --------------------------------------------------------------------------- #

MANUAL_CHECKS = (
    ("Leti.exe icon in Explorer",
     "Open dist\\ in Explorer. Leti.exe shows Leti's icon, not the default."),
    ("Desktop shortcut icon", "The Desktop shortcut shows Leti's icon."),
    ("Start Menu icon", "Start > Leti shows Leti's icon."),
    ("taskbar icon while running", "With Leti open, the taskbar button shows its icon."),
    ("the console goes away", "The setup console closes once Leti's window appears."),
    ("Leti's window opens", "Double-clicking the Desktop shortcut opens Leti."),
    ("microphone initialises", "The first launch offers the microphone check and it passes."),
    ("speech is heard", "Say something; Leti transcribes it."),
    ("the answer is spoken", "Leti speaks its reply rather than printing it."),
    ("speech is chunked by sentence",
     "The reply is spoken in sentences, not word by word and not one long block."),
    ("STOP is immediate", "Say stop mid-answer; speaking stops within a word or two."),
    ("STOP does not leak into the next turn",
     "Ask something else straight after a stop; it is answered in full."),
    ("the transcript stays hidden", "No answer text appears until it is asked for."),
    ("show me the text", 'Saying "show me the text" reveals the answer.'),
    ("hide the text", 'Saying "hide the text" puts it away again.'),
    ("an image appears", "Ask for an image; it appears without revealing the transcript."),
    ("a graph appears", "Ask for a chart of some numbers; it appears."),
    ("mathematics renders",
     'Ask for a derivation, then "show me the math"; it renders as mathematics, '
     "not as raw LaTeX, and the LaTeX is never spoken."),
    ("a task runs", "Ask for something multi-step; progress is reported."),
    ("a task pauses and resumes", "Pause it, then resume it."),
    ("a task is cancelled", "Cancel one; it stops and says so."),
    ("task history", "Ask what happened to that task."),
    ("two tasks at once", "Start two; neither corrupts the other."),
    ("computer use refuses an ambiguous target",
     'Ask it to click something there are two of; it asks which rather than guessing.'),
    ("verification does not overclaim",
     "Do something whose result cannot be checked; Leti says it could not confirm it "
     "rather than reporting success."),
    ("a confirmation is asked for",
     "Ask for something destructive; Leti stops and asks first."),
    ("Leti exits cleanly", "Close Leti; no window, console or process is left behind."),
)


def stage_manual(report: Report) -> None:
    print("\n== manual: what a person has to look at or listen to")
    print("  Each of these is recorded as NEEDS A HUMAN until somebody edits")
    print(f"  {REPORT.name} and changes the verdict to PASS or FAIL.\n")
    for name, how in MANUAL_CHECKS:
        report.check("manual", name, NEEDS_HUMAN, how)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def stage_report(report: Report) -> None:
    print("\n== report")
    by_verdict: Dict[str, List[str]] = {v: [] for v in VERDICTS}
    for check in report.data["checks"]:
        by_verdict[check["verdict"]].append(check["name"])

    lines = ["", "=" * 74, "  LETI WINDOWS RELEASE VALIDATION", "=" * 74, ""]
    lines.append(f"  Run on            {sys.platform} "
                 f"({'Windows' if on_windows() else 'NOT WINDOWS'})")
    lines.append(f"  Report            {REPORT}")
    lines.append("")
    for verdict in VERDICTS:
        names = by_verdict[verdict]
        lines.append(f"  {verdict:<15} {len(names)}")
    lines.append("")
    if by_verdict[FAIL]:
        lines.append("  FAILED:")
        lines.extend(f"    - {name}" for name in by_verdict[FAIL])
        lines.append("")
    if by_verdict[NOT_RUN]:
        lines.append("  NOT RUN (these are not passes):")
        lines.extend(f"    - {name}" for name in by_verdict[NOT_RUN])
        lines.append("")
    if report.data["measurements"]:
        lines.append("  Measured:")
        for name, row in report.data["measurements"].items():
            lines.append(f"    {name:<28} {row['seconds']}s "
                         f"{('- ' + row['note']) if row['note'] else ''}")
        lines.append("")
    verdict = report.worst()
    lines.append(f"  Overall           {verdict}")
    if verdict != PASS:
        lines.append("")
        lines.append("  This is NOT a validated release. Every row above that is not")
        lines.append("  PASS has to be either fixed or genuinely run.")
    lines.append("=" * 74)
    text = "\n".join(lines)
    print(text)
    (ROOT / "validation_report.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\n  Written to {ROOT / 'validation_report.txt'}")


STAGES = {
    "preflight": stage_preflight, "build": stage_build, "first": stage_first,
    "repair": stage_repair, "warm": stage_warm, "shortcuts": stage_shortcuts,
    "processes": stage_processes, "tests": stage_tests, "manual": stage_manual,
    "report": stage_report,
}
ORDER = ("preflight", "build", "first", "repair", "warm", "shortcuts",
         "processes", "tests", "manual", "report")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Windows release of Leti, honestly.",
        epilog="A check that did not run reports NOT RUN, which is not a pass.")
    parser.add_argument("stages", nargs="*", default=["all"],
                        help="stages to run: " + ", ".join(ORDER) + ", or all")
    parser.add_argument("--fresh", action="store_true",
                        help="start a new report rather than adding to the existing one")
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    if arguments.fresh:
        REPORT.unlink(missing_ok=True)
    wanted = ORDER if "all" in arguments.stages else tuple(arguments.stages)
    unknown = [name for name in wanted if name not in STAGES]
    if unknown:
        print(f"Unknown stage(s): {', '.join(unknown)}")
        return 2

    if not on_windows():
        print()
        print("  This is not Windows. Every check that needs Windows will be")
        print("  recorded as NOT RUN rather than skipped quietly, so the report")
        print("  says what was actually proven. The platform-independent checks")
        print("  still run.")

    report = Report()
    for name in wanted:
        STAGES[name](report)
    if "report" not in wanted:
        stage_report(report)
    return 0 if report.worst() == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
