# Leti

A local-first, open-source desktop AI assistant powered by [Ollama](https://ollama.com).
Leti can see your screen, hear and speak to you, remember things about you across
sessions, and take action on your computer — launching apps, running commands,
managing files, and browsing the web — all gated behind an explicit permission
and confirmation system.

Everything runs on your machine. No data leaves your computer except the web
searches and pages you explicitly ask it to visit.

---

## Features

- **Reasoning & tool use** via Ollama's native function calling (Qwen 2.5,
  Llama 3.3, or Mistral Nemo).
- **Screen vision** via `llama3.2-vision` — ask Leti what's on your screen.
- **Voice pipeline** — wake word (openWakeWord), speech-to-text
  (OpenAI Whisper), text-to-speech (pyttsx3, using your OS's native voice
  engine), with interruptible playback.
- **Memory** — short-term conversation buffer + long-term semantic recall
  (ChromaDB) so Leti remembers preferences and facts across sessions.
- **System control** — launch/close apps, focus windows, mouse/keyboard
  automation, sandboxed shell commands, file read/write/search/delete/move.
- **Browser automation** via Playwright.
- **Live web search** via DuckDuckGo — no API key required.
- **Safety layer** — every tool call is classified `safe` / `risky` /
  `destructive`, with forbidden-pattern and protected-path hard blocks (these
  can never be bypassed, including by pre-approval below), and an append-only
  audit log of every action attempted. Confirmation prompts are plain,
  human-readable sentences (e.g. "Leti wants to send an email to
  stavros@example.com with the subject '...'. Should I go ahead?") rather than
  raw tool/argument dumps. In **text mode**, reply `yes`/`y` or the shorthand
  `-y` to approve (`no`/`n`/`-n` to decline). In **voice mode**, if what you
  said already reads as clear approval ("yes, go ahead and send it") the
  confirmation step is skipped entirely since there's no natural way to type
  `-y` while talking; otherwise Leti asks out loud and listens for your
  spoken yes/no. This reads phrasing, not actual vocal tone/prosody — there's
  no audio sentiment model involved, just keyword-level intent matching (see
  `core/intent_signals.py`) — so it's a convenience, not a hard guarantee of
  vocal certainty.
- **Cybersecurity** — scans listening ports/services on *this* machine and
  flags commonly-exploited ones, checks OS firewall status (and can enable
  it), detects brute-force login attempts in local auth logs, lists
  startup/persistence entries and processes with live network connections
  so unwanted software is easy to spot, and can hash-baseline any file or
  folder to detect tampering/corruption later and restore it. It never
  scans, probes, or connects to other people's devices — only your own
  machine and what's passively visible in your local ARP table.
- **Email** — reads unread mail over IMAP (peek mode, doesn't mark as read)
  and sorts it into `crucial` / `financial` / `promotions` / `general` with
  an importance score, using local rule-based heuristics (sender/keyword/
  header signals) — nothing is sent to a third party to classify. Sending
  mail goes through SMTP and requires confirmation (`risky` tier). Needs an
  app password from your provider in `config/settings.yaml`; see the
  commented `email:` block there.
- **Trading (paper/sandbox only)** — monitors a configurable watchlist of
  stocks/crypto via Alpaca's market data API and evaluates simple, well-known
  technical signals (SMA crossover, RSI threshold) against them. Order
  placement is hardcoded to Alpaca's **paper trading** endpoint in code (not
  just config), so it's structurally incapable of touching a live account or
  real funds. Needs a free Alpaca paper API key in `config/settings.yaml`;
  see the commented `trading:` block there. Not financial advice.
- **Contacts** — a saved contact book (`data/contacts.json`) so Leti can turn "email Stavros
  about the mechanic news" into the right address automatically. When a name is shared by
  multiple contacts, it disambiguates using the message's topic against each contact's tags/
  notes/company (e.g. a contact tagged `mechanic` vs one tagged `plumber`); if the topic
  doesn't clearly point to one person, Leti asks which one you meant instead of guessing.
- **Venture Scout / Red-Team Strategist / COO** — a business-analysis persona run as a
  dedicated sub-call to the local reasoning model (same pattern as the screen-vision tool):
  `scout_find_trends` surfaces 3 non-obvious opportunities in a niche using weak-signal
  analysis (regulatory shifts, newly open-sourced infra, supply chain gaps) rather than
  saturated trends; `scout_stress_test` runs a brutal adversarial critique of a business idea
  (failure modes, hidden assumptions, a 1-10 moat/defensibility score) immediately followed by
  a zero-friction execution blueprint for the upgraded version. No sycophancy by design - ask
  Leti to "find trends in X" or "stress-test my idea: ...".
- **Meeting scheduling** — `schedule_meeting` creates a real calendar event via CalDAV (Google
  Calendar/iCloud/Nextcloud/Fastmail all support this with an app password, same pattern as
  email - no OAuth browser flow needed) and, if you ask for `zoom` or `teams`, generates a real
  join link (Zoom Server-to-Server OAuth / Microsoft Graph app-only auth) embedded right in the
  event. `create_video_meeting_link` does just the link generation on its own.
  `send_meeting_invite_email` sends participants a proper invite with the details and join link
  in the body plus a one-click `.ics` attachment - kept as a separate, separately-confirmed step
  from scheduling rather than bundled automatically. All three are `risky` tier. Needs the
  commented `calendar:`, `zoom:`, and/or `teams:` blocks in `config/settings.yaml`.
- **System health & updates** — a sibling to the cybersecurity tools above rather than an
  addition to them (that's specifically about attack surface; this is about hardware/OS health).
  `get_system_specs` reports CPU/RAM/disk/GPU/battery/uptime. `run_health_check` samples CPU,
  memory, swap, disk, temperature (where the platform exposes it), and battery against
  configurable thresholds, and returns each problem with a plain-English suggestion - some
  reference an existing tool by name (e.g. a runaway process points at `kill_process` with its
  PID already filled in) rather than duplicating that logic. `check_for_updates` lists pending
  OS updates (apt/dnf/pacman, macOS `softwareupdate`, or Windows Update) read-only;
  `apply_system_updates` actually installs them and is `risky` tier, so it always asks first -
  that's what satisfies "awaits user agreement" for anything it wants to change. Disk checks
  automatically skip pseudo-mounts like snap's squashfs images, which are always reported
  100%-full by design and would otherwise be constant false alarms on most Linux desktops.
- **Personality & user profile** — two separate, hand-editable JSON files (`data/personality.json`,
  `data/user_profile.json`) that give Leti a persistent tone and a memory of who you are, on top
  of (not instead of) the semantic long-term memory below. **Personality** is six 0-10 dials -
  humor, sarcasm, formality, warmth, directness, verbosity - translated into plain instructions
  injected into every turn's context; `set_personality` changes individual dials ("be funnier",
  "more sarcastic", "blunter"), `apply_personality_preset` jumps to a named combination
  (`witty_friend`, `professional`, `dry_and_sarcastic`, `warm_and_supportive`, `blunt_and_brief`),
  and `reset_personality` restores the default balance. These dials only ever shape tone - they
  never change what Leti will or won't help with, and that boundary is stated directly in the
  instruction text injected alongside them, not left implicit. **User profile** is a structured,
  always-in-context alternative to pure semantic recall: Leti proactively calls
  `remember_about_user` when you share something durable (preferences, interests, goals, how you
  like to be talked to) and `set_user_name` when you introduce yourself, so your name and key
  facts are present in every conversation rather than only surfacing when a similarity search
  happens to match the current topic. `view_user_profile` shows everything saved (with ids),
  `forget_user_fact` deletes one entry, and `clear_user_profile` (destructive) wipes it all.
- **Social media** — split across two files by how each platform is actually accessible:
  - **YouTube & Reddit** (`social_media.py`) use free, official, no-login APIs. `get_youtube_channel_latest`,
    `search_youtube_trending`, `get_subreddit_posts`, and `search_reddit` all filter for
    significance rather than dumping everything - Reddit results take a `min_score` threshold,
    and YouTube's trend search flags videos with view counts high relative to their channel's
    subscriber count (a concrete "before it gets big" signal, not just "recently uploaded").
    Needs a free YouTube Data API v3 key in `config/settings.yaml`; Reddit works out of the box.
  - **Instagram, TikTok, Facebook** (`social_login.py`) have no free official API for reading an
    arbitrary public account, so `login_to_social_platform` opens a real, visible browser window
    for you to log in yourself - password, 2FA, CAPTCHA, all handled by you directly on the real
    site. Leti never sees the password; only the resulting session (cookies) is saved to
    `data/social_sessions/`, the same mechanism "stay signed in" already uses. This is against
    those platforms' terms of service even for your own account - realistic risk is a security
    challenge or temporary lock if their bot-detection notices non-human patterns, not something
    more severe, but worth knowing. Scraping the rendered page also means these are the most
    likely tools in the project to break when a platform redesigns its site (Facebook's markup
    is the most heavily obfuscated of the three).
  - **Watches** (`add_social_watch`, `list_social_watches`, `remove_social_watch`,
    `check_social_watches`) work across every platform above. "Notify me when X uploads" adds a
    persistent watch; each one tracks the last item it saw, not a time window, so checking after
    a 5-minute gap or a 3-month gap behaves identically - whatever's genuinely new gets reported,
    however large that backlog is. The first check on a new watch just establishes the baseline
    silently, so you don't get "new!" spam for a video that was already there when you asked to
    watch it. Leti also calls `check_social_watches` once automatically at the start of each new
    session to catch you up without being asked.
  - **Trend tool integration**: `scout_find_trends` (from the Venture Scout) now pulls live
    Reddit/YouTube signals for the niche before generating ideas, grounding at least one idea in
    real current traction where relevant - it degrades gracefully to pure model knowledge if
    nothing's configured or reachable.

---

## Prerequisites

1. **Ollama installed and running.**
   ```bash
   ollama serve
   ```
2. **Pull the models referenced in `config/settings.yaml`:**
   ```bash
   ollama pull qwen2.5:14b
   ollama pull mistral-nemo
   ollama pull llama3.2-vision
   ollama pull nomic-embed-text
   ```
   Swap these for smaller/larger variants depending on your hardware — any
   Ollama model that supports tool calling works for the reasoning model.

3. **Python 3.11+**

4. **TTS engine dependency (pyttsx3 uses your OS's native voice engine):**
   - Debian/Ubuntu/Linux: `sudo apt install espeak` (pyttsx3 drives espeak)
   - macOS: no extra install — uses the built-in `NSSpeechSynthesizer`
   - Windows: no extra install — uses the built-in SAPI5 engine

   To see available voices on your machine and pick one for
   `config/settings.yaml` (`tts.voice_id`):
   ```python
   from audio.tts import Pyttsx3TTS
   for v in Pyttsx3TTS().list_voices():
       print(v["id"], v["name"])
   ```

5. **Whisper model download** — the first run of `--mode voice` or
   `--mode continuous` will auto-download the model set in
   `config/settings.yaml` (`stt.model_size`, default `base.en`) via the
   `openai-whisper` package. Larger models (`small.en`, `medium.en`) are more
   accurate but slower on CPU.

6. **Playwright browsers** (for browser automation):
   ```bash
   playwright install chromium
   ```

7. **Platform audio libraries** for `pyaudio`:
   - Debian/Ubuntu: `sudo apt install portaudio19-dev`
   - macOS: `brew install portaudio`
   - Windows: prebuilt wheels usually work out of the box.

---

## Installation

The double-click launchers below handle this automatically on first run - creating
`leti_env/`, installing `requirements.txt`, and fetching the Playwright browser - so most
people can skip straight to **Running Leti**. For a fully manual setup instead (e.g. CI, a
dev container, or if you just prefer doing it yourself):

```bash
git clone <this-repo> leti
cd leti
python -m venv leti_env
source leti_env/bin/activate   # Windows: leti_env\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

---

## Running Leti

**Double-click launchers** (first run sets everything up automatically - no manual `pip
install` needed - then opens Leti's interface in its own window, not a terminal prompt):
- **Windows**: double-click `Launch Leti (Windows).bat`.
- **macOS**: double-click `Launch Leti (macOS).command` (first time, right-click → Open
  once to clear Gatekeeper's "unidentified developer" warning).
- **Linux**: `.desktop` files can't reliably find their own folder without a one-time
  install step, so run `./scripts/install_linux_launcher.sh` once - it puts a working,
  double-clickable "Leti" icon on your Desktop and in your app menu.

All three run GUI mode (`--mode gui`, see below) by default now - a console window briefly
shows setup progress and then stays open as Leti's log, while the actual interface (dashboard,
chat, voice) opens in its own window on top. If you specifically want the old terminal-only
text mode instead, run `python main.py --mode text` yourself from an activated `leti_env`
(see **Or from a terminal you already have open**, further down).

**What "automatically" means, specifically:**
- First run creates `leti_env/` in the project folder if it isn't there yet, then installs
  every package in `requirements.txt` into it - no manual venv/pip steps needed.
- The Playwright browser (a separate download from the pip package) is fetched the first
  time too.
- On Linux, the launcher also attempts to install the system GTK/WebKit package
  `pywebview`'s native window needs (`pip install pywebview` alone can't pull this in - it's
  not a Python package). This is best-effort for Debian/Ubuntu; if it can't be installed
  automatically (a different distro, no `apt-get`, etc.), Leti still opens fully working, just
  as a browser tab you'll need to open yourself at the printed URL instead of an automatic
  window - not a failure, just a smaller convenience lost. macOS and Windows use their OS's
  built-in webview, so this extra step doesn't apply there.
- Every run after that just launches, in well under a second of overhead - it skips
  reinstalling by comparing a hash of `requirements.txt` against what was last installed,
  and only re-syncs the venv if that file actually changed (e.g. after a `git pull`).
- If setup itself fails partway (no internet, a missing system library like `portaudio`,
  no `python3-venv` package, etc.), the launcher stops with a specific, actionable message
  instead of limping forward into a confusing crash inside Leti itself - fix what it says,
  re-run the same launcher, and it picks up exactly where it left off rather than starting
  over.
- **Ollama is fully self-managed, not just checked**: if it's not installed, the launcher
  installs it (`curl ... | sh` on Linux, `brew install ollama` on macOS, `winget install
  Ollama.Ollama` on Windows); if it's installed but not running, it starts `ollama serve` as
  a detached background process that keeps running after the launcher window closes, polling
  until it responds rather than guessing with a fixed delay; either way, it then pulls every
  model listed under `ollama:` in `config/settings.yaml` (`ollama pull` is a fast no-op if a
  model's already present, so this runs on every launch harmlessly, not just the first). If
  any of these steps genuinely can't succeed (no `winget`/`brew` available, no network, a
  failed pull), it stops with a specific, actionable message instead of the old behavior of
  warning and pressing on into a raw connection-error crash later.
- The window stays open after Leti exits (success or failure) so you can actually read the
  output, instead of the classic "window flashes and closes."

**Or from a terminal you already have open** (after the manual Installation above, or once
a launcher has bootstrapped `leti_env/` for you):

**Text mode** (no microphone/speaker required — great for first-time setup and testing):
```bash
python main.py --mode text
```

**Push-to-talk voice mode:**
```bash
python main.py --mode voice
```

**Always-listening wake-word mode** (say "Hey Leti"):
```bash
python main.py --mode continuous
```

**GUI mode** (the HUD interface — voice stays active underneath, text is an additional
channel, not a replacement — and it's reachable from your phone too, not just this machine):
```bash
python main.py --mode gui
```
Needs `pip install aiohttp` (this is what serves the interface over HTTP + WebSocket to
any device — required for GUI mode to work at all, not optional). `pywebview` is genuinely
optional on top of that: with it, the desktop gets a native app window; without it, GUI mode
still works fine — you just open the printed `http://127.0.0.1:8420/` URL in a regular
browser tab instead. If you do want the native window, Linux additionally needs a system
webview package, e.g. `sudo apt install libwebkit2gtk-4.1-dev`; Windows/macOS use their
built-in webview, nothing extra needed there.

**Launching from Android (or any other device on your Wi-Fi):** GUI mode prints two URLs on
startup — one for this machine, and one like `http://192.168.1.50:8420/?token=<token>` for
your phone. Open that second URL in Chrome on Android, then use the browser menu → "Add to
Home Screen" for an app-like icon that launches full-screen. The desktop window and your
phone are looking at the **same live session** — the same conversation, the same to-do list,
image pop-ups pushed to both at once.

Before enabling this on a network you don't fully trust, know the security model: the token
is generated once into `data/gui_remote_token.txt` and printed to the console — anyone with
that URL on your Wi-Fi can drive Leti, including its risky/destructive tools (still gated by
the normal confirmation flow, but that confirmation currently happens via *this machine's*
voice/mic, not your phone — worth knowing if you're not standing next to the desktop when you
approve something). This is LAN-trust-level security, like a home router's admin page — it is
**not** meant to be exposed to the internet; never port-forward it. Set
`gui.enable_remote_access: false` in `config/settings.yaml` to disable non-loopback access
entirely and restrict the GUI to this machine only.

What's real vs. not yet: the dashboard's **time**, **weather** (Open-Meteo, no key needed —
IP-based location by default, or set `weather.latitude`/`longitude` in `config/settings.yaml`
to override), **system stats** (via `run_health_check`), and **to-do list** are all live,
not mocked. The center ring reacts to real audio: your own voice via the browser's mic
(if permitted) while the backend is listening, and a synthetic-but-correctly-timed pattern
while Leti's TTS is speaking (pyttsx3 plays directly to the OS audio device, not through the
page, so the page can't analyze its actual waveform — the animation is triggered by the
real speaking start/stop, just not literally amplitude-matched to it). The session panel
shows the current session only, and the text input auto-focuses after Leti finishes
speaking or after you finish a spoken turn, so you can seamlessly switch to typing.
Image search results and diagrams (`search_images`, `create_sketch`) pop up as visual
cards in the session panel on every connected device at once, independent of whatever
Leti says in text.

---

## Configuration

- `config/settings.yaml` — model names, voice pipeline settings, memory
  paths, and safety toggles (including a `dry_run` mode that reports what a
  state-changing tool *would* do instead of executing it — see Safety below).
  This file is the documented reference/template — most of its integration
  sections (email, calendar, weather, etc.) ship commented out with examples.
### Controlling the computer

Leti opens things the way you'd ask a person to: `launch_app` takes an app name
(`firefox`), a path to a program or document, or a URL, plus `arguments` to open
something *in* an app — "open YouTube in Firefox" is `firefox` with
`['https://youtube.com']`. It resolves apps by PATH lookup, then the platform's
own launcher (`open -a` on macOS, `start` on Windows, `gtk-launch` on Linux), and
reports honestly when nothing started rather than claiming success. It's `risky`,
so you confirm each launch — and the prompt names the arguments, so what you
approve is what happens.

`open_url` hands a page to your own default browser (http/https only). For
browsing Leti does itself: `web_search` finds pages, `browser_read_page` reads
one so it can answer from what the page says rather than a search snippet, and
`browser_navigate`/`browser_click`/`browser_fill_form` drive a dedicated
Playwright browser.

Window control (`close_app`, `focus_window`) works on Windows and macOS;
pygetwindow doesn't implement it on Linux, where the tools now say so plainly
instead of surfacing a bare exception.

- `config/permissions.yaml` — per-tool risk tiers, forbidden shell patterns,
  and protected filesystem paths. Review and tighten this before giving
  Leti broad system access. `protected_paths` is worth extending: `read_file`
  runs without confirmation and `send_email` needs only one, so anything
  readable is one approval away from leaving the machine.

Changes made through `/settings` apply immediately. The exceptions are values
bound to a resource when Leti starts — the SQLite and Chroma paths, the loaded
Whisper model, and the Ollama base URL and HTTP timeout — which need a restart.

## What Leti can do

The tools are one set, meant to be combined - a question that needs research,
the numbers behind it, and a chart is one task, not three modes.

**Research the web.** `web_search` takes several queries at once so a question
gets approached from more than one angle. `research_topic` goes further: it runs
those searches, opens the most relevant pages, spreads its reading across
domains rather than taking six pages from one site, and returns their actual
content attributed to source. Pages it couldn't read are reported rather than
dropped, so a failed fetch never passes for a source. Pass `verify` with
specific claims and it reports how many independent domains carry them - which
shows consensus, not correctness, and says so. `add_social_watch` with platform
`webpage` watches any URL for changes.

**Write and run code.** `run_code` executes Python, JavaScript, TypeScript,
Bash, MATLAB (falling back to Octave) or SQL and returns the real exit code,
stdout and stderr - a failing run is a failure, never a success with a caveat.
`run_tests` works out how a project is tested from its files. `inspect_project`
reads a codebase's shape before you change it. Editing source is the file tools;
git, builds and deploys are `run_shell_command`.

**Analyse data.** CSV, TSV, TXT, Excel, JSON, JSON Lines, Parquet and MATLAB
`.mat` all load the same way. `inspect_dataset` names what's wrong before
anything is decided - missing values, duplicates, outliers, columns stored as
text that hold numbers. `clean_dataset` does only what it's told and reports what
each operation changed. `analyze_dataset` covers statistics, correlations, group
comparisons, trends and regression; `visualize_dataset` draws it.

**Engineering.** `engineering_calculate` evaluates expressions with units
attached, so unit errors surface as errors instead of plausible wrong numbers.
`check_dimensions` tests an equation before you trust it and distinguishes "the
dimensions are wrong" from "the dimensions are right but the arithmetic isn't".
`solve_symbolic` rearranges, differentiates and integrates. Nothing here is
arithmetic done by the language model.

**Business.** Record leads, income and expenses; every metric - conversion rate,
weighted pipeline, margin, revenue by service, performance by channel - is
computed from those records rather than stored. `business_next_actions` answers
"who should I contact?", ranking open leads by stage-weighted value with the ones
that have gone quiet first. People aren't duplicated here: a record links to the
contact book by id.

**Projects.** A project is a folder plus standing instructions that follow it
into every conversation about it. They nest (`University/MATLAB`,
`Business/Clients`), and the active project's instructions and file list go to
the model each turn, so "continue the MATLAB project" resolves to something
concrete. Work from the other tools lands in the active project's folder.

**Scheduling.** A scheduled task is an instruction plus a schedule; when it
fires it runs through the same tools, so "every Monday, research 20 potential
clients and write a report" is one task. Every run is recorded, failures retry
with backoff, and a task that keeps failing is disabled and reported rather than
failing quietly forever.

By default tasks only fire while Leti is open. To have them run anyway, ask Leti
to enable system scheduling (or run `python core/system_scheduler.py install`).
That registers one periodic check with the machine's own scheduler — cron on
Linux, launchd on macOS, Task Scheduler on Windows — which starts Leti with
`--mode run-scheduled`, runs whatever is due, and exits.

Two things follow from nobody being present for those runs:

- **They can't do everything.** `scheduler.unattended_allows` decides what an
  unattended run may do; the default lets it read, compute and write files —
  which covers research, analysis and reports — but not send email, post, deploy
  or delete. Anything else is refused with a reason recorded in the task's
  history. `critical` is refused even if you add it to the list.
- **Nothing runs twice.** The in-app loop and the scheduled process both look for
  due work, so they take an OS file lock first; whichever gets it runs, the other
  skips. Downtime produces one catch-up run per task, not one per missed
  occurrence — a week with the machine off doesn't yield seven reports.

Output goes to `logs/scheduled_runs.log`, and the run exits non-zero if a task
failed, so the OS scheduler's own logs show it.

## The icon

`gui/icon.svg` is the source of truth. Three variants exist because one drawing
can't serve every size: the full mark has a HUD ring that turns to a smudge at
taskbar size, so `gui/icon-small.svg` drops the ring and enlarges the "L" for
the 16-32px entries, and `gui/icon-maskable.svg` is full-bleed with the mark
pulled into Android's safe zone, since launchers crop home-screen icons to the
device's own shape.

Edit an SVG, then regenerate the PNG/`.ico`/`.icns` files in `gui/icons/`:

```
python scripts/build_icons.py
```

They're committed so that double-clicking a launcher never needs a build step.

Where the icon actually gets used:

| Surface | Wiring |
| --- | --- |
| App window (GTK/Qt) | automatic — `gui/api.py` passes it to pywebview |
| Browser tab / phone home screen | automatic — served by `gui/server.py` |
| Linux desktop + app menu | `./scripts/install_linux_launcher.sh` |
| Windows desktop + Start Menu | `powershell -ExecutionPolicy Bypass -File scripts\install_windows_launcher.ps1` |
| macOS `.command` file in Finder | `./scripts/install_macos_icon.sh` |

The three install scripts are one-time setup. Windows needs one because a `.bat`
file can't carry an icon at all — Windows always draws the generic script icon —
so the icon has to live on a shortcut pointing at it. macOS needs one because a
custom file icon is an extended attribute applied on the machine, not something
that can ship in a repo.

## Tests

```
pip install pytest pytest-asyncio
pytest
```

The suite covers the authorization layer specifically: protected-path
canonicalization, shell-command path checks, `dry_run`, voice pre-approval
scoping, audit redaction, atomic state writes, and the subprocess runner.
- **The `/settings` command** — the easier way to actually configure something
  like email or calendar, instead of hand-editing `settings.yaml` and hunting
  for the right line. Type `/settings` in text mode or the GUI's chat box (not
  in voice mode — filling in a password by voice is a bad idea) to get an
  interactive list of every configurable integration, its current status
  (configured/not set), and a field-by-field form. Values are saved to
  `config/settings.local.yaml` — a separate, plain (uncommented) file created
  on first use — so `settings.yaml` itself is never rewritten and never loses
  its comments. Secrets (app passwords, API keys) are write-only through this
  flow: once set, they're never echoed back, in the terminal or the GUI, only
  shown as "currently set." Changes take effect immediately, no restart
  needed. Type `clear <section>` (text mode) or use the Clear button (GUI) to
  remove a section's settings entirely.

---

## Project Structure

```text
leti/
├── config/
│   ├── settings.yaml          # Model names, thresholds, paths, modes
│   └── permissions.yaml       # Safe vs risky commands/tools
├── core/
│   ├── config_loader.py       # Shared YAML config loader + canonical path resolution
│   ├── orchestrator.py        # Async event loop, state machine, tool-calling loop
│   ├── llm_client.py          # Ollama client with native function calling
│   ├── safety_guard.py        # Permissions, confirmation barrier, audit logger
│   ├── intent_signals.py      # Approval/denial phrase detection (voice pre-approval, -y shorthand)
│   ├── confirmation.py        # CLI + voice confirmation callbacks (shared by CLI and GUI modes)
│   ├── console_input.py       # Single shared stdin reader (cancellable prompts)
│   ├── system_scheduler.py    # Registers the periodic check with cron/launchd/schtasks
│   ├── atomic_write.py        # Crash-safe state-file writes
│   └── settings_editor.py     # The /settings command - schema-driven, no LLM involved
├── gui/
│   ├── hud.html                # The HUD interface (audio-reactive ring, dashboard, chat)
│   ├── icon.svg                # Icon source: full mark (48px and up)
│   ├── icon-small.svg          # Icon source: simplified, for 16-32px
│   ├── icon-maskable.svg       # Icon source: full-bleed, for Android launchers
│   ├── icons/                  # Generated PNG/.ico/.icns - see scripts/build_icons.py
│   ├── server.py               # HTTP+WebSocket server (aiohttp) - what phones/browsers connect to
│   └── api.py                  # Orchestrator-facing backend, voice loop, TTS wiring
├── audio/
│   ├── wake_word.py           # OpenWakeWord engine
│   ├── stt.py                 # openai-whisper stream/PTT handler
│   └── tts.py                 # pyttsx3 speech synthesis with interruptibility
├── memory/
│   ├── vector_store.py        # ChromaDB long-term facts/preferences
│   └── session_memory.py      # Working memory & conversation buffer (+ SQLite log)
├── tools/
│   ├── base.py                # BaseTool abstract class & JSON schema generator
│   ├── command_runner.py      # Async subprocess helper that keeps exit status/stderr
│   ├── os_control.py          # App launcher, window manager, mouse/keyboard
│   ├── shell_runner.py        # Sandboxed terminal executor
│   ├── file_manager.py        # Safe file read, write, search, organize
│   ├── browser.py             # Playwright automation
│   ├── vision.py              # Screen grab & Ollama Vision multimodal analyzer
│   ├── web_search.py          # DuckDuckGo live search
│   ├── network_security.py    # Port/service scan, firewall status, LAN ARP read (read-only)
│   ├── system_defense.py      # Firewall enable, brute-force detection, persistence/process checks
│   ├── backup_restore.py      # Hash-baseline snapshot, integrity diff, restore
│   ├── email_client.py        # IMAP read + categorize (crucial/financial/promotions/general), SMTP send
│   ├── trading_platform.py    # Alpaca PAPER-only quotes, SMA/RSI signals, simulated orders
│   ├── contacts.py            # Contact book with name+context disambiguation
│   ├── venture_scout.py       # Venture Scout / Red-Team Strategist / COO persona
│   ├── meeting_scheduler.py   # CalDAV calendar events, Zoom/Teams links, invite emails
│   ├── system_health.py       # Specs, health/performance checks, OS update management
│   ├── personality.py         # Tone dials (humor/sarcasm/formality/warmth/directness/verbosity)
│   ├── user_profile.py        # Structured, editable memory of who the user is
│   ├── social_media.py        # YouTube/Reddit (official APIs), watch list, trend signals
│   ├── social_login.py        # Instagram/TikTok/Facebook via session-cookie login
│   ├── weather.py             # Current weather via Open-Meteo (no API key)
│   └── todo_list.py           # Simple to-do list, shared between GUI and conversation
├── logs/
│   └── audit.log              # Append-only JSON log of all tool calls
├── requirements.txt
├── main.py
└── README.md
```

---

## Safety Model

Every tool call passes through `SafetyGuard.authorize()` before it runs:

1. **Hard blocks** — forbidden shell patterns (e.g. `rm -rf /`), protected
   paths (e.g. `~/.ssh`), and blocked domains are rejected unconditionally,
   regardless of tier or confirmation. Protected paths are enforced on
   canonical paths (`..` normalized, symlinks followed) and against the paths
   named inside a shell command, not just path-shaped tool arguments. The
   forbidden-pattern list is a speed bump for catastrophic one-liners, not a
   security boundary — a blocklist can't enumerate every spelling of `rm`,
   which is why shell commands also always require confirmation.
2. **Action classes** (`config/permissions.yaml`) — each tool is classified by
   what it does, and `safety.require_confirmation_for` in `settings.yaml`
   decides which classes stop and ask you first (default: modify, external,
   critical):
   - `read` — reads information and changes nothing.
   - `execute` — runs something reversible: a search, a calculation, code.
   - `modify` — changes files, data or projects on this machine.
   - `external` — affects something outside it: sends mail, posts, deploys.
   - `critical` — irreversible or potentially damaging.

   Two rules aren't configurable, since a setting that switched them off would
   defeat the point: `critical` always asks even if you remove it from the list,
   and `critical` never accepts approval inferred from a spoken request's
   wording.
3. **Audit log** — every authorization decision and execution result is
   appended to `logs/audit.log` as a JSON line, regardless of outcome. Argument
   values that are secrets or bulk content (a password typed into a form, a
   file's contents, an email body) are recorded as a length marker rather than
   verbatim — the log is permanent and unrotated, so what it keeps matters.

You can flip `safety.dry_run: true` in `settings.yaml` to have every
risky/destructive tool report what it *would* do instead of doing it — useful
when testing new prompts or tool wiring. Read-only `safe` tools still run, so
Leti can still search, read the screen and plan normally; nothing that changes
your system executes.

**On voice pre-approval.** In voice mode there's no way to type `-y`, so a
request that already states approval ("go ahead and move that file") can stand
in for the confirmation prompt. That approval covers exactly **one** risky
action and is then spent — every further tool call in the same turn prompts
normally, because the model, not the user, chose those. `destructive` tools
never accept it at all and always ask.

---

## Extending Leti

To add a new tool:

1. Subclass `BaseTool` in a new or existing file under `tools/`.
2. Define `name`, `description`, and `parameters` (a list of `ToolParameter`).
3. Implement `async def run(self, **kwargs) -> ToolResult`.
4. Register it in `build_tool_registry()` in `main.py`.
5. Add a risk tier for it in `config/permissions.yaml` (defaults to
   `destructive` if omitted, as a safe default).

---

## Known Limitations

- Local models are weaker at multi-step tool planning than large hosted
  models — complex chained tasks may need more explicit instructions or a
  higher tool-iteration cap.
- Continuous wake-word mode requires a trained openWakeWord model for your
  chosen wake word; the default "hey_leti" placeholder in
  `config/settings.yaml` should be swapped for a model you've trained or
  downloaded (see the openWakeWord docs for custom wake word training).
- Push-to-talk in `main.py` currently uses silence detection rather than a
  true key-hold listener; wire in a hotkey library (e.g. `pynput`) for a more
  traditional press-and-hold experience if desired.
