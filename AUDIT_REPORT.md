# Leti — final engineering audit

Branch `claude/code-review-rcfna5`, six commits on top of `b4033b9`.
73 files changed, +2074 / -152.

Every fix below was made, tested, and left in a green suite. Nothing here is a
recommendation for later.

---

## 1. What was inspected

The whole tree, not only what changed recently: `core/` (50 modules), `tools/`,
`gui/`, `audio/`, `memory/`, `launcher/`, `scripts/`, `tests/`, `main.py`,
`config/settings.yaml` (342 lines), `config/permissions.yaml` (584 lines),
`requirements.txt`, `README.md`, `.gitignore`, `pytest.ini`, `launcher/Leti.spec`,
the two `.bat`/`.command` launchers, `scripts/install_windows_launcher.ps1`, and
`gui/icons/`.

Method, in this order: static analysis (pyflakes + AST passes written for each
question), then the dependency graph, then behavioural probes against the real
code, then targeted reading. Findings were confirmed by reproduction before
being fixed — two things that looked like bugs were not, and are recorded as
such in §9.

**Measured at the end of the pass**

| | |
|---|---|
| Test suite | **2694 passed**, 0 failed (was 2610) |
| pyflakes over everything incl. tests | **0 findings** |
| `compileall` | all modules compile |
| Module imports | 106 import cleanly, 2 skipped for absent optional deps (`whisper`, `openwakeword`), 0 failures |
| Module-level import cycles | none |
| Registered tools | 137, all classified, no duplicates |
| New dependencies | none — `requirements.txt` is byte-identical |

---

## 2. Architecture — one authority per subsystem

`tests/test_architecture_and_performance.py` (80 tests) already machine-checks
this, and it passes: one orchestrator, one planner, one memory, one task manager,
one scheduler, one CRM, one permission system, one SafetyGuard, one Connections
Manager, one Computer Use engine, one diagnostics, one voice pipeline, one
interface state. No duplicate system was introduced by this pass and none was
found.

What the pass *did* find was four **stale references** between authorities — the
failure mode that a single-authority design is still vulnerable to (§3, item 4).

Ollama has one chat/embed/stream authority (`core/llm_client.py`), one
model-selection authority (`core/model_setup.py`, which owns tags and pulls), and
one read-only reachability probe (`core/diagnostics.py`). All three honour
`ollama.host`. No second provider was introduced and no model was substituted.

`core/context_engine.py` implements the documented six stages — UNDERSTAND,
ENTITIES, SOURCES, RETRIEVE, RANK, PACKAGE — and is the only thing that assembles
a turn's context.

Two mechanisms do the same small job and are left that way deliberately:
`gui/server.py` holds its background tasks in an instance-scoped set tied to the
server's lifetime, while `core/task_manager.run_detached()` (new, §3 item 3) holds
process-lifetime ones. Consolidating them would tie a task's lifetime to the
wrong object. Noted rather than merged.

---

## 3. Bugs found and fixed

### 1. Two paths to arbitrary code execution — the most serious finding

`engineering_calculate` evaluated the model's expression with `eval()` and an
empty `__builtins__`, on the belief that this is a sandbox. It is not.
**Reproduced:**

```
[c for c in ().__class__.__base__.__subclasses__()
 if c.__name__ == 'BuiltinImporter'][0].load_module('os').getcwd()
```

returned the working directory. `solve_symbolic` handed the same string to
sympy's `parse_expr` and `sympify`, which are `eval()` underneath and documented
as unsafe on unsanitised input — `__import__('os').getcwd()` and
`open('/etc/hostname').read()` both ran.

Both tools are action class `execute`, so **neither stops to ask**, and the
expression comes from a model that reads web pages.

Fixed by parsing and walking the formula instead of evaluating it: only numbers,
names, arithmetic, and calls to the functions the namespace provides. The
symbolic path is both walked *and* parsed against a namespace holding sympy's own
names plus an empty `__builtins__` — needed because `eval()` supplies real
builtins when the globals it is handed have no entry for one. Every legitimate
calculation and every symbolic operation still works.

The existing test named `test_expressions_cannot_reach_the_interpreter` passed
throughout, because it only tried the one attack that was already refused. It now
tries the ones that worked.

### 2. A stop the user just issued could be wiped

An answer is spoken as a series of utterances and the stop flag is read between
them — but `transcript.start_speaking()` ran at the head of *each* utterance and
cleared it. A stop arriving after the orchestrator's check for utterance N had
passed was erased by utterance N beginning, and the rest of the answer was spoken
**after the user had said stop**. Exactly the failure the streaming work was
meant to make impossible.

Clearing a stop is a turn boundary, not an utterance boundary. It now happens
once per turn in `_handle_one_turn` — every turn, including ones that do not call
the model, because "show me the text" also speaks a confirmation. The stop
shortcut still returns before the turn starts, so a stop aimed at the turn in
flight is not cleared by the turn it is stopping.

The test covering "a new answer clears a leftover stop" drove it through
`start_speaking()`, which is why this was never caught.

### 3. Work reported as started that could vanish

`tools/scheduler.py`'s "run this task now" and `core/task_manager.py`'s
notifications used a bare `asyncio.create_task()`. asyncio holds only a weak
reference, so the work could be collected mid-flight after the caller had already
reported success. One `run_detached()` helper holds them, and says so when there
is no loop rather than losing the work silently. `task_manager` also used
`get_event_loop()`, which raises outside a loop — into a bare `except` that
discarded the notification.

### 4. Six tool names that no longer exist, in the tables that reason about tools

Four modules keep tool names as data. A name that is no longer a tool does not
fail — it silently stops covering the tool that replaced it:

| Dead name | Consequence |
|---|---|
| `create_document`, `add_business_data`, `update_business_data`, `save_workflow` | recording a business record, updating a lead and saving a workflow all came back **unverifiable**, while `VERIFIERS` looked like it covered them |
| `add_business_data`, `update_business_data` | the business store had **no write resource**, so two tasks writing it were not serialised |
| `check_calendar_availability`, `get_market_data` | the calendar and market tools lost the cheap "is this configured?" check that exists to answer *before* a round trip |

All replaced with the tools that exist. `update_lead` now returns the lead's id so
its verifier can read the record back instead of only reporting that no id came
back. One test checks all five tables against the registry; a second checks that
no tool is listed as both having an effect and having none.

### 5. A chart somebody asked for, dropped because the machine was busy

`core/performance.py` turns `derive_visuals` off above 90% CPU or 92% memory, and
the orchestrator read that flag before asking whether anyone had asked for a
visual. So "plot these measurements" produced the chart, dropped it, and said
nothing — and whether it happened depended on how loaded the machine was.

Found as an **intermittent** test failure: passing alone, passing in its own file,
failing about one full-suite run in three. Deliberate visuals are now shown
whatever the load; the saving pressure was meant to make is untouched. Both
halves are pinned with the mode *forced* rather than measured, and the whole
streaming suite passes with every turn forced into PRESSURE mode. Three
consecutive full runs green.

### 6. Routing could answer with another registry's tools

`core/tool_router._index_for` cached its index under `id(registry)`. An id is
unique only while its object is alive. **Reproduced:** a registry allocated at a
dead one's address with the same tool count inherited its index, and routing then
named tools that are not in that registry at all — which `select_tools_for` would
hand to the model as available. The tests had to call `_CACHE.clear()` to work
around it, which was the smell. Now a weak key: no reuse, and no dict that only
grows.

### 7. Text read in the platform's encoding where its counterpart used UTF-8

`config/settings.yaml` is read as UTF-8 by `config_loader` (the authority) and as
cp1252 by the settings editor; the launchd plist declared UTF-8 while being
written in whatever the locale was; a user's `.sql` file was read in the locale
encoding. Four files corrected. Latent today — both YAML files are pure ASCII and
every JSON writer escapes — but triggered by the user typing a non-ASCII value,
which is what the settings editor is *for*.

### 8. The launcher said it downloaded over HTTPS without enforcing it

`launcher/bootstrap.py` fetches an embeddable Python and then **runs** it.
`EMBED_SHA256` is deliberately empty and the comment explaining why is right, so
the transport is the whole of the trust — and `_download`'s docstring said "over
HTTPS" while nothing checked. urllib follows redirects and its redirect handler
allows https → http, so a redirect could have delivered that interpreter over
plaintext with no integrity check behind it. The scheme is now checked before the
request and the landed URL after it.

### 9. Three defects and 52 dead imports from the static pass

`audio/stt.py`, `audio/tts.py` and `audio/wake_word.py` annotated returns with
`Any` without importing it (latent rather than fatal only because of
`from __future__ import annotations`); `tools/todo_list.py` discarded the id it had
just created, so "mark that one done" needed a second call; `core/documents.py`
declared a `nonlocal` it only read. Plus 52 unused imports across 37 files, 5 dead
test bindings, and 7 redundant f-strings — pyflakes went 69 → 0.

---

## 4. Two things a person could never see

**The context-window warning.** `main.py` measures the tool list against `num_ctx`
at startup and logs a warning — a warning that exists *because nothing said
anything* the first time the list outgrew the window. It was a `logger.warning`,
on stderr, in a console the Windows launchers hide the moment Leti's own window
appears. It is a diagnostics check now ("Context window"), so it reaches the panel
and "run full Leti diagnostics".

**No application log existed at all.** `basicConfig` sent everything to stderr and
nothing else, while README and the user guide both point at `logs/` when something
needs explaining. There is a rotating `logs/leti.log` now (2 MB, three kept), and
a `logs/` that cannot be written to is a reason to carry on with stderr rather
than not to start.

---

## 5. Configuration audit

All **39 active leaf settings** in `config/settings.yaml` have a real reader in
the code. **Zero** obsolete, unread, or written-never-read parameters. No
duplicate or conflicting definitions. All action classes in
`config/permissions.yaml` are valid, including every `action_by_case` override.

Tool/permission mapping is exact: **137 registered tools, 137 entries, no tool
without one and no entry without a tool.** (An earlier pass of mine reported 133
orphans; that was my own naive YAML flattener reading the dangerous-path and
command denylists as tool names. Re-run against the file's real structure, it is
clean.)

Three configuration statements were wrong and are fixed:

1. `settings.yaml` advised a 16 GB card to switch to `qwen2.5:14b`, "and it fits
   there with room to spare". **It does not**: at the configured 28,672-token
   window the 14b needs 14.35 GiB against a 13.7 GiB budget, and Ollama answers
   that by moving layers onto the CPU rather than refusing — the exact failure the
   comment exists to warn about. The arithmetic in it had been done at 24,576, two
   windows earlier. It now states `core/model_setup.py`'s own figures, and a test
   checks them against that module, including all four card recommendations.
2. `settings.yaml` restated the measured tool-schema size a second time, which
   `tests/test_docs_match_code.py` counts. The numbers live once now, in the
   `num_ctx` note. That test also localised a missing claim to its document
   instead of only counting them.
3. README's prerequisites told people to `ollama pull qwen2.5:14b`, which is not
   the configured reasoning model.

`num_ctx` is **unchanged at 28672**, and the new check reports why that matters:
Default Mode's whole tool list is 22,336 tokens of 28,672, leaving 336 beyond what
one turn can need. An ordinary routed turn is nowhere near that; the turn routing
cannot narrow sends all of it and is close to the edge. The measurement is the
deliverable — raising `num_ctx` would invalidate the VRAM figures the model
recommendation is built on.

---

## 6. Performance

Measured rather than assumed, on the real registry:

| | |
|---|---|
| `schemas_for` (129 Default Mode tools) | 0.57 ms |
| `route()` | 0.08 ms |
| `approx_schema_tokens()` | 0.95 ms |

Nothing here is worth caching against an LLM call. No blocking call
(`time.sleep`, `subprocess.run`, `requests`) inside any `async def`. No mutable
default arguments anywhere. Idle cost is one 30-second scheduler tick that reads
one small file. The `while True: time.sleep(1)` parking loop in `gui/api.py` looks
wasteful and is not — it is the standard way to keep Ctrl+C working on Windows,
and was left alone after checking.

The one unbounded cache found (`tool_router._CACHE`) is fixed as part of §3 item 6.

---

## 7. Voice, streaming, transcript

The STOP invariants were re-derived from the code rather than trusted:

- A stop **cannot** carry into a later turn — cleared once per turn at the turn
  boundary.
- A stop **cannot** be lost within its own turn — this was broken and is fixed
  (§3 item 2).
- An interrupted stream cannot keep speaking: the check sits between utterances in
  both the streaming and non-streaming paths, and the utterance already playing is
  cut by the engine's own interrupt.
- A stopped answer is kept, so "stop" then "show me the text" shows the whole
  thing including the part never said aloud.

Tool errors were examined against §16 and deliberately left alone: 74 sites pass
`str(e)` as a failure, and the reader of a `ToolResult.error` is **the model**,
which writes the user-facing sentence — and the orchestrator already enriches it
with "there is no email account; add one under Connections". Rewriting them would
be churn, not a fix. The surfaces a person actually reads — the launcher's
`problem()`, the startup warnings, the confirmation prompts, the frozen binary's
failures — do say what happened, what was being attempted, and what to do.

---

## 8. Build, and the Windows shortcut path

A **real PyInstaller build** was produced (PyInstaller 6.22.3, `launcher/Leti.spec`
unchanged): 7.6 MB. Built outside the repository, so nothing generated is
committed; `.gitignore` already covers `build/` and `dist/`.

The frozen binary's own paths were exercised:

- No `main.py` beside it → names what failed, what it was trying to do, and what
  to do about it; exits **2**.
- `--install-shortcuts` → finds Leti, says shortcuts are Windows-only, exits **1**
  — which also proves the `launcher.shortcuts` hiddenimport is doing its job,
  since without it the executable would build, run, and silently never place a
  shortcut.

`gui/icons/leti.ico` is present (71 KB) and is the only icon; no competing icon was
created. 163 launcher/shortcut tests pass.

---

## 9. Two things that looked like bugs and were not

Recorded because a report that only lists confirmed bugs hides the work of ruling
things out.

- `model_setup.recommend()` appeared to return `None` for every machine, including
  a 24 GB card where three candidates fit. My probe was missing `"ok": True` in
  the hardware dict. The function is correct.
- `gui/api.py`'s `while True: time.sleep(1)` looked like a busy-wait to replace
  with `Event().wait()`. It is the documented workaround for Ctrl+C on Windows;
  replacing it would have broken shutdown.

---

## 10. Platform limitations — what was NOT tested

**This audit ran on Linux. No Windows behaviour was executed.** Specifically:

- **No `.lnk` was created or read.** Desktop and Start Menu shortcut placement,
  repair and idempotence are covered by 163 tests and by reading
  `launcher/shortcuts.py`, but the PowerShell that writes a `.lnk` did not run.
  On this platform `install()` correctly returns `[]` and the launcher correctly
  reports "no Desktop or Start Menu to put it in".
- **The built executable is an ELF binary, not a `.exe`.** PyInstaller reported
  `Ignoring icon; supported only on Windows and macOS` — so **the icon embedded in
  `Leti.exe` was not verified**, only that the spec points at an icon that exists.
  The 7.6 MB figure is the Linux build; the spec's "about 10 MB" refers to Windows
  and was not checked.
- **No embeddable Python was fetched and no venv was built**, so the first-launch
  download path was exercised only through its tests and its error paths.
- `msvcrt` file locking, `winreg`, and console hiding via `ShowWindow` are
  Windows-only branches that were read, not run. The PyInstaller warning file
  confirms only the expected Windows-only modules are missing on a Linux build.
- **Ollama was not running**, so no live model call was made. Streaming, chunking
  and cancellation are covered by the fake-model tests.

---

## 11. Known, deliberate, and left alone

- **`EMBED_SHA256` is empty.** The comment explaining why is correct: a hash
  invented in that file would prove only that the download matched what somebody
  typed. Filling it in from a download made *here*, through this session's proxy,
  would create the appearance of verification while pinning whatever that channel
  served. It stays a task for a person with python.org's published checksum in
  front of them. HTTPS is now genuinely enforced (§3 item 8).
- **The GUI listens on `0.0.0.0` by default**, so Leti's window can be opened from
  a phone on the same Wi-Fi. A non-loopback device must present a token. This is a
  documented design decision with a documented off switch
  (`gui.enable_remote_access: false`), so it is documented rather than changed —
  and the user guide now says plainly that it is on, what protects it, and never
  to forward it through a router.
- **Routing sends the full tool list when nothing matches** (e.g. "what is 2+2").
  That is the documented safe fallback, not a regression, and was not narrowed.
- **74 `error=str(e)` sites** — see §7.

---

## 12. Security sweep

No hardcoded credentials, API keys, tokens or private keys. No personal machine
paths. No `shell=True`, no `os.system`, no `os.popen`. No `pickle`, no
`yaml.load`, no `marshal` — every YAML read is `safe_load`. No debug endpoints,
no `debug=True`, no `reload=True`. Nothing under `data/` or `logs/` is tracked by
git, and `.gitignore` covers `config/settings.local.yaml`, the GUI token, and the
social-session cookies. The two arbitrary-code-execution paths in §3 item 1 were
the sweep's real finding, and the unverified-interpreter question is §3 item 8 and
§11.

`LETi_USER_GUIDE.txt` contains no credential, and a test asserts it — by looking
for a *value*, since the guide discusses passwords and tokens at length by design.

---

## 13. `LETi_USER_GUIDE.txt`

All sixteen mandated sections, written for somebody who is not going to read the
code. `tests/test_user_guide.py` (48 tests) keeps it honest the way
`test_docs_match_code.py` keeps the README honest: every default it quotes against
`settings.yaml`, the VRAM table and all four card recommendations against
`model_setup.recommend()`, the tool count and window cost against the registry,
the confirmation classes and unattended limits against SafetyGuard, every file it
tells you to open against the disk or against the code that writes it, every
`--mode` against `main.py`'s parser, and the connection list against
`settings_editor`. Plus two promises the guide makes about itself: no credential,
and plain ASCII, because it is a `.txt` somebody opens in Notepad.

Writing it is what surfaced the wrong 16 GB model advice in §5 — the guide had to
state which card runs which model, and the two authorities disagreed.

---

## 14. Tests

**2694 passing, 0 failing, 0 skipped-as-broken.** No test was weakened. 84 tests
were added. Four existing tests were changed, and each change made them stricter
or more faithful:

| Test | Why |
|---|---|
| `test_expressions_cannot_reach_the_interpreter` | only tried the attack that was already refused; now tries the ones that worked |
| `test_a_new_answer_clears_a_stop_left_over_from_the_last_one` | drove the mechanism at the utterance level, which is the bug; now goes through the turn boundary |
| `test_main_py_was_not_changed_to_suit_the_launchers` | forbade the *word* "launcher" anywhere in `main.py`, including a comment explaining a launcher's behaviour; now checks for an actual dependency |
| `test_the_hook_is_still_only_reached_for_successful_results` | matched the visual hook's condition across at most two lines; now matches however many it wraps onto |

`test_docs_match_code.py`'s per-document claim check was tightened: a total of
one-claim-per-document was also satisfied by one document claiming it twice and
another not at all — which is the case it exists to catch, and which my own first
edit to `settings.yaml` produced.

Named sub-suites, each run on its own: tool routing 70, streaming voice 51, voice
output 37, Windows launch 163, full diagnostics 29, architecture 80, engineering
41, docs-match-code 9, user guide 48, business mode 83, coding mode 89.
