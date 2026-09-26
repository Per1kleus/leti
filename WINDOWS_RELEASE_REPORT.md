# Leti — Windows release validation

**Status: NOT a validated Windows release.** No Windows build was produced and no
Windows behaviour was executed, because this environment is Linux and no Windows
machine is reachable from it. What follows separates what was proven from what was
prepared, and maps both onto the 25 release criteria.

Branch `claude/code-review-rcfna5`, commits `9f329c0` and `c03d3ef`.
12 files changed, +2788 / −69.

---

## 1. Why no Windows validation happened

Established before starting, not assumed:

```
uname -a          Linux vm 6.18.44-fc-v37 x86_64
/etc/os-release   Ubuntu 24.04.4 LTS
wine / wine64     absent
python.exe        absent
list_environments 3 environments, all kind=anthropic_cloud (Linux)
```

PyInstaller's own installation here carries exactly one bootloader —
`PyInstaller/bootloader/Linux-64bit-intel/` — so it cannot emit a PE image even in
principle. There is no Windows machine, no Windows Python, and no emulation layer.

**So: criteria 1 and 2 are NOT MET, and criteria 3–23 are NOT TESTED.** They are
not marked as passed anywhere in this report or in the code.

---

## 2. The release blocker, found and fixed

`Launch Leti (Windows).bat` installed Ollama with winget, started the server, and
waited for the port. **Nothing anywhere under `launcher/` mentioned Ollama at all.**
So `Leti.exe` did none of it.

On a clean Windows machine, double-clicking `Leti.exe` produced a Leti that opened,
could not reach a model, and reported that in a log file behind a console the
launcher had just hidden. Two front doors documented as *"one Leti, two front
doors"*, and only one of them worked on the machine it was written for. The stated
goal of this task — *a normal Windows user with no Python can double-click Leti.exe
and then use Leti normally* — was not reachable through the `.exe`.

**Fixed:** `launcher/ollama_setup.py` is that step, once — is the server answering,
is Ollama installed, install it with winget if not, start it detached, wait for the
port. `bootstrap.prepare()` runs it, which is the one function both front doors go
through, and the `.bat`'s copy is deleted.

Three constraints held deliberately:

- It **chooses no model and downloads none.** `core/model_setup.py` stays the only
  thing that decides what a machine should run. A test asserts the module contains
  no `ollama pull`, no `/api/pull`, and no `reasoning_model`.
- It **never blocks the launch.** No path raises `SetupFailed` — a test asserts the
  name does not appear in its code. Leti opens and reports an unreachable model
  server itself, on screen, which a console that is about to be hidden cannot.
- `launcher.ollama_setup` is in the spec's `hiddenimports`, because it is imported
  inside a function. Without that the executable would build, run, and silently
  skip the step — this exact bug reintroduced in the one build that cannot be
  caught by importing it. A test asserts the spec names it.

---

## 3. Three more Windows-only defects, found by reading the paths that only run there

| Defect | Consequence |
|---|---|
| `bootstrap.system_pythons()` looked for an all-users Python under `%PROGRAMFILES%\Programs\Python\Python3NN` — that is the **per-user** layout; the real all-users path is `%PROGRAMFILES%\Python3NN`. And the version range was hand-written `range(13, 10, -1)`, so 3.14 was invisible the day it shipped. | A Python installed for all users, or any version ≥3.14, was never found. Not fatal — Leti fetched its own 11 MB runtime — but wrong, and wasteful. Now globbed over both real layouts, newest first, and `Python39` no longer sorts above `Python313`. |
| The `.bat` probed Ollama with `curl`, absent on early Windows 10 builds. | There the errorlevel read as "not running", so it waited 30 s for a server that was already answering, then **refused to start Leti**. The shared module uses `urllib`, which is in every Python. |
| The `.bat`'s inline Python read `config/settings.yaml` in the console code page, with `2^>nul` swallowing the error. | A settings file with any non-ASCII in it produced an empty model list and pulled **nothing, silently**. |

---

## 4. The build now checks what it built

`build_exe.py` checked that a file appeared and that its size was plausible.
Neither can tell a Windows build from a Linux one renamed, an x64 build from an ARM
one, an executable with an icon from one without, or a clean bundle from one that
swept a `.env` up on the way past.

`launcher/verify_build.py` reads the file: PE signature, PE32/PE32+, machine,
subsystem, entry point, section count, embedded icon, version metadata, PyInstaller
bootloader — and scans the bundle for credentials, AWS/GitHub/Slack/OpenAI-shaped
keys, private keys, `settings.local.yaml`, the GUI token, test fixtures, `.git`, and
absolute developer paths.

It is pure byte-reading with no dependencies, which is why it could be **validated
here**. Checked field by field against four real Windows binaries found on this
machine, spanning PE32 and PE32+, x86/x64/ARM64, and console/GUI subsystems, with
`file(1)` and `objdump` as independent ground truth:

| Binary | `file(1)` | parser |
|---|---|---|
| `t64.exe` | PE32+ console x86-64, 6 sections | PE32+ / x64 (AMD64) / Windows console / 6 |
| `w64.exe` | PE32+ GUI x86-64, 6 sections | PE32+ / x64 (AMD64) / Windows GUI / 6 |
| `w32.exe` | PE32 GUI Intel 80386, 5 sections | PE32 / x86 (32-bit) / Windows GUI / 5 |
| `w64-arm.exe` | PE32+ GUI Aarch64, 6 sections | PE32+ / ARM64 / Windows GUI / 6 |

Entry points matched `objdump`'s start address less the image base in every case.
Plus a PE assembled byte by byte in the tests, so every offset the parser reads is
one a test put there deliberately.

It also **refuses to overclaim**: on a non-Windows build it prints *"the bundle is
clean; whether this is a Windows executable was NOT checked"* rather than a verdict
that reads like the PE checks passed. That wording is itself a fix — the first
version said "looks like a releasable Windows executable" on a Linux build.

**Verified negative:** the Linux build produced here is correctly rejected —
*"Leti does not start with the MZ signature — it is an ELF binary (Linux)"* — as is
a Linux executable renamed to `Leti.exe`.

---

## 5. Build reproducibility — a real defect, diagnosed and fixed

§2 asked for two clean builds and for the cause to be fixed if they differed. They
did:

```
build 1   7dcc029f6294…   7,970,696 bytes
build 2   1b5f2dd8c1bf…   7,969,432 bytes
          2,039,432 differing bytes
```

**Cause:** Python's per-process hash randomisation. PyInstaller walks sets and
dicts of module names while assembling the archive, so the member order follows
string hashes and changes in every process, and the archive compresses differently.

**Proven** by building twice with `PYTHONHASHSEED=0` — byte-identical, same
SHA-256, zero differing bytes. The build subprocess now sets it, and two clean
builds through `build_exe.py` with nothing set by hand come out identical:

```
ceb9e2a7495311f3d14fa92198366c8560c3dba98c87d72ee5a59b213548978f   (both)
0 differing bytes
```

Only the build is pinned. Leti itself still runs with randomisation on, and a test
asserts `bootstrap.py` never sets it.

*(Measured on the Linux build. The cause is platform-independent, so the fix
applies to the Windows build; that the Windows build is reproducible has not been
observed.)*

---

## 6. What is prepared for the real Windows run

`scripts/validate_windows_release.py` — ten stages, run on the Windows machine:

```
python scripts\validate_windows_release.py all
```

| Stage | What it does |
|---|---|
| `preflight` | is the machine genuinely clean (names every leftover artefact), is the exe a PE, x64, iconed, bootloadered, leak-free |
| `build` | builds twice from clean, compares SHA-256 for reproducibility |
| `first` | times a real first launch; checks Python was prepared, packages installed, setup recorded complete, and that Ollama was dealt with |
| `repair` | breaks the installation **nine** ways and checks each recovery — including a package genuinely `pip uninstall`ed rather than merely unrecorded, a corrupt state file, a truncated one, an unreadable one, a missing one, missing site-packages, a removed runtime, a partially created venv, and an interrupted install |
| `warm` | times a second launch and proves it downloaded nothing and did not rebuild the interpreter |
| `shortcuts` | Desktop and Start Menu existence, targets, working directories, icons, idempotence, and stale-target repair — read back through the same shell that wrote the `.lnk` |
| `processes` | orphan `Leti.exe`, `python.exe` and `ollama.exe` counts before and after |
| `tests` | full suite fixed order, shuffled file order, a repeat run for flakiness, and pyflakes |
| `manual` | the **27** things that need eyes and ears |
| `report` | writes `validation_report.txt` and a machine-readable `.json` |

The design constraint, tested: **it cannot report a pass for something it did not
do.** NOT RUN and NEEDS A HUMAN are verdicts, neither counts as a pass at any level
of the summary, and run on Linux every Windows stage produces NOT RUN rows and the
run exits non-zero. It does not go green by skipping everything. Standard library
only, and it never imports Leti — it drives the launchers from outside, the way a
user does.

Run here, it reports exactly that:

```
  Run on            linux (NOT WINDOWS)
  PASS            3        NOT RUN         8
  FAIL            1  (running on Windows)
  Overall           FAIL
  This is NOT a validated release.
```

---

## 7. The 25 release criteria

| # | Criterion | Status |
|---|---|---|
| 1 | A real Windows `Leti.exe` was built | **NOT MET** — no Windows machine |
| 2 | Confirmed to be a Windows PE executable | **NOT MET** — no artefact to confirm. The *checker* exists and is validated against four real PEs |
| 3 | Launches on a clean Windows machine | **NOT TESTED** |
| 4 | Works without pre-installed Python | **NOT TESTED** — the detection defect on that path is fixed and unit-tested |
| 5 | Installs required packages automatically | **NOT TESTED** — logic unit-tested with injected runners |
| 6 | Handles a missing/invalid dependency | **NOT TESTED** — nine-case matrix written and ready |
| 7 | Ollama/model setup works as designed | **NOT TESTED** — and it *did not exist* on the `.exe` path; now written, unit-tested, and wired into both front doors |
| 8 | Desktop shortcut works | **NOT TESTED** — no `.lnk` can be written here |
| 9 | Start Menu shortcut works | **NOT TESTED** |
| 10 | Existing icon is used | **PARTIALLY** — `gui/icons/leti.ico` is the only `.ico` in the repo (asserted by test) and the spec points at it. That it *embeds* and *displays* is NOT TESTED; PyInstaller reports `Ignoring icon; supported only on Windows and macOS` on Linux |
| 11 | Voice works | **NOT TESTED** |
| 12 | Streaming voice works | **NOT TESTED** on Windows; 51 tests cover it against a fake model |
| 13 | STOP is immediate and reliable | **NOT TESTED** on Windows; covered by tests, and a race here was fixed earlier in this branch |
| 14 | Transcript hidden by default | **NOT TESTED** on Windows; covered by tests |
| 15 | Images work | **NOT TESTED** |
| 16 | Graphs work | **NOT TESTED** |
| 17 | Mathematics renders | **NOT TESTED** |
| 18 | Tasks work | **NOT TESTED** |
| 19 | Computer Use safety works | **NOT TESTED** |
| 20 | Verification does not falsely claim success | **NOT TESTED** on Windows; covered by tests |
| 21 | No security-critical regression | **HELD** — see §8 |
| 22 | No orphan processes after exit | **NOT TESTED** — the harness measures it |
| 23 | Warm startup is fast, reinstalls nothing | **NOT TESTED** — the harness measures and asserts it |
| 24 | Full regression suite passes | **MET** — 2795 passing; fixed order ×1, three shuffled file orders, pyflakes 0 |
| 25 | No known critical Windows-specific issue remains | **CANNOT BE ASSERTED.** Four were found by reading; nothing found by running, because nothing ran |

---

## 8. Security

No regression, and two hardening items carried from the audit still hold: HTTPS is
enforced on every launcher download, before the request and on the landed URL after
redirects, so a redirect cannot downgrade the interpreter fetch to plaintext. The
new bundle scan adds a release-time check that no credential, key, `settings.local.yaml`,
GUI token, test fixture or `.git` internals ship inside the executable — it runs on
every build and fails it.

`EMBED_SHA256` remains deliberately empty. Filling it from a download made here,
through this session's proxy, would create the appearance of verification while
pinning whatever that channel served. It stays a task for a person with
python.org's published checksum in front of them.

New attack surface reviewed: `ollama_setup` invokes `winget` with a fixed argument
list and no shell, and `ollama serve` with a path from `shutil.which`. No string
interpolation reaches a shell.

---

## 9. Tests

**2795 passing** (was 2723 at the start of this task), pyflakes **0**.

| Suite | |
|---|---|
| Full suite, fixed order | 2795 passed |
| Shuffled file order, 3 seeds | 2795 passed, 2795 passed, 2795 passed |
| `test_ollama_setup.py` (new) | 28 |
| `test_verify_build.py` (new) | 49 |
| `test_windows_validation_harness.py` (new) | 23 |
| `test_windows_launch.py` | 164 |

Two existing tests were changed, and both had pinned the defect rather than the
behaviour: one asserted the `.bat` installs Ollama itself, the other that the word
`ollama` appears nowhere in `bootstrap.py`. That arrangement *was* the bug. They now
pin the division that matters — the model server is shared by both front doors, and
choosing a model remains `core/model_setup.py`'s alone. No test was weakened.

---

## 10. What a person has to do next

1. On a clean Windows machine with no Python, no Ollama and no Leti state, put this
   folder there.
2. `python launcher\build_exe.py` — it builds, verifies the PE, scans the bundle,
   and fails on any problem.
3. `python scripts\validate_windows_release.py all`
4. Work the 27 rows of the `manual` stage and edit their verdicts in
   `validation_report.json`.
5. Any FAIL or NOT RUN left in `validation_report.txt` is a release blocker.

Until that has been done, criteria 1–23 remain unproven, whatever the code review
says.
