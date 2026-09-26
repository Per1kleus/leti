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
  engine), with interruptible playback. Leti asks for the microphone and speakers
  **once, on the first launch**, checks they actually work, and remembers the
  answer (see below). Optional in practice: on a machine with no microphone, no
  speaker or no espeak, GUI mode says so once and runs text-only rather than
  refusing to start.
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
- **Cybersecurity** — `inspect_network_connections` reports what *this* machine's
  sockets are doing: the ports listening on it (flagging commonly-exploited ones)
  and the processes holding live outbound connections, either view or both from
  one pass. Alongside it: OS firewall status (and enabling it), brute-force login
  attempts in local auth logs, startup/persistence entries, and hash-baselining
  any file or folder to detect tampering later and restore it. It never scans,
  probes, or connects to other people's devices — only your own machine and what's
  passively visible in your local ARP table.
- **Email** — reads unread mail over IMAP (peek mode, doesn't mark as read)
  and sorts it into `crucial` / `financial` / `promotions` / `general` with
  an importance score, using local rule-based heuristics (sender/keyword/
  header signals) — nothing is sent to a third party to classify. Sending
  mail goes through SMTP and requires confirmation (`risky` tier). Needs an
  app password from your provider in `config/settings.yaml`; see the
  commented `email:` block there.
- **Market data and trading (paper/sandbox only)** — one Alpaca client reads
  prices, OHLCV bars and volume for stocks and crypto, and is what the market
  watches are evaluated against. Figures name the feed they came from (a free
  key reads IEX, not the consolidated tape). It evaluates simple, well-known
  technical signals (SMA crossover, RSI threshold) too. Order placement is
  hardcoded to Alpaca's **paper trading** endpoint in code (not just config),
  so it's structurally incapable of touching a live account or real funds.
  Needs a free Alpaca paper API key in `config/settings.yaml`; see the
  commented `trading:` block there. Not financial advice.
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
  `system_report` answers all three questions about the machine from one call, by section:
  `specs` is what it *is* (CPU/RAM/disk/GPU/battery/uptime), `health` is how it's *doing*
  (CPU, memory, swap, disk, temperature where the platform exposes it, and battery against
  configurable thresholds), and `updates` lists pending OS updates (apt/dnf/pacman, macOS
  `softwareupdate`, or Windows Update). It defaults to `health`, the usual question, and
  collects only the sections asked for. Each health problem comes back with a plain-English
  suggestion - some reference an existing tool by name (e.g. a runaway process points at
  `kill_process` with its PID already filled in) rather than duplicating that logic.
  Reporting is read-only;
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
- **Social media** — two tools across every platform, not one tool per platform.
  `get_social_content` fetches the latest from a YouTube channel, a subreddit, a Reddit user,
  an Instagram/TikTok/Facebook account, or any web page — you name the `platform` and the
  `identifier` as that platform writes it. `search_social` searches a platform for a topic.
  How each platform is reached differs, and that's split across two files:
  - **YouTube & Reddit** (`social_media.py`) use free, official, no-login APIs. Both filter for
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
- **Files Leti can actually read** - PDF, Word, Excel, CSV, JSON, Markdown and text. "Read
  these PDFs and compare the offers" works: Leti finds which files the request is about,
  looks at their structure, and reads only the parts that answer the question - with the
  page, sheet or row range attached to every quote. Nothing loads a whole file into the
  prompt. See [Working with files](#working-with-files).
- **Long tasks you can watch and stop** - a task that runs on its own shows its real steps,
  which are done and which is running, and can be paused, resumed, retried or cancelled
  from the interface. When one stops to ask permission it says exactly what for.
  A task that said what "done" means is checked against it before anyone is told
  it worked - a file existing is not the same as a file with the right contents.
- **It reads the request before it answers it** - deterministically, with no extra
  model call, so "what time is it" stays a one-line question and "find five
  laptops, compare them and write a file" is understood as four things in order.
  A small request is shown fewer tools; an ambiguous one gets a question rather
  than a guess. See [Reading the request](#reading-the-request).
- **The interface** - three columns with a calligraphic capital L at the centre, the machine
  and the day down the left, five controls and a live activity log down the right, and
  floating panels that appear only when a sentence genuinely cannot carry the answer. See
  [The interface](#the-interface).

---

## Starting Leti on Windows

Two ways in, and they are the same Leti. Neither needs Python, pip, a terminal,
an environment to activate or a PATH to configure.

### Leti.exe

Put it in the Leti folder - the one holding `main.py` - and double-click it.

    Leti.exe
       -> checks Python            -> fetches one if there is none
       -> checks the packages      -> installs only what is missing
       -> verifies                 -> starts Leti

About 10 MB, and the same on every release: it carries a Python of its own for
one purpose, which is to prepare a real environment in the folder and start
`main.py` with it. It deliberately does **not** contain Leti - an executable with
Whisper's tensor library inside it is several gigabytes, has to be rebuilt for
every dependency bump, and could not install anything anyway, because a frozen
interpreter has no `venv` and no `ensurepip`.

The window shows what it is doing, because the first run downloads a few hundred
megabytes and somebody waiting deserves to know that is what is happening. It
hides itself once Leti's own window is up, and stays - with the reason on it - if
setup fails.

**Building it** (on Windows, from the project folder):

```
python launcher\build_exe.py
```

which installs PyInstaller if needed, builds `launcher/Leti.spec`, and checks the
result. Or directly:

```
python -m PyInstaller --clean --noconfirm launcher/Leti.spec
```

PyInstaller is not in `requirements.txt`: it is needed to make a release, not to
run Leti.

### Launch Leti (Windows).bat

Double-click it. Same steps, no executable to build - useful if you would rather
not run a downloaded binary, or you already have the repository.

Batch, because it is the one language guaranteed to be on a Windows machine that
has nothing installed. It does exactly two things itself: finds *some* Python -
fetching one with PowerShell if there is none - and then hands over to
`launcher/bootstrap.py`, which is the same module `Leti.exe` runs. One
description of what a prepared machine looks like, reached two ways.

It also makes sure Ollama is installed and running and has the models
`config/settings.yaml` names. *Which* model this machine should run is not its
business: `core/model_setup.py` asks that on first launch, inside Leti, where it
can see the hardware.

### Finding Leti afterwards

Both launchers put Leti where you look for things, the first time they set the
folder up:

    Desktop\Leti.lnk
    Start Menu\Programs\Leti\Leti.lnk

One name, one icon - `gui/icons/leti.ico`, the same mark the executable is built
with. Both point at the same thing: `Leti.exe` when the folder has one, the `.bat`
launcher when it does not. Build the executable later and the existing shortcut is
repaired to use it rather than leaving you with two.

What is already correct is left alone, so this is not work a launch repeats:
running setup five times writes two shortcuts, not ten, and never a
`Leti (1).lnk`. A shortcut that has been deleted or retargeted is put back by

```
powershell -ExecutionPolicy Bypass -File scripts\install_windows_launcher.ps1
```

which decides nothing itself - it finds an interpreter and asks
`launcher/shortcuts.py`, so that one answer to "where does Leti's shortcut go"
serves the installer, the `.bat` and `Leti.exe` alike.

Per-user throughout: your own Desktop and your own Start Menu, no administrator
prompt, no registry, nothing in Program Files.

### What it puts where, and what it never touches

| | |
|---|---|
| `leti_env\` | the environment, built from a system Python when there is a good one |
| `leti_runtime\` | a Python fetched because the machine had none |
| `data\launch_setup.json` | what has finished, so the next launch does not redo it |
| `logs\setup_pip.log` | what a failed install printed, if one did |

All four are inside the Leti folder. Nothing is installed system-wide, no
administrator prompt appears, PATH is not changed, and your own Python - if you
have one - is never written to. Uninstalling is deleting the folder.

### After the first launch

    double-click  ->  quick check  ->  Leti

The quick check is a few `stat` calls: `requirements.txt` is the same file it
was, the last run finished, and every package's metadata folder is still on disk.
Under half a second, and it downloads nothing.

That last clause is what makes a broken install repair itself. `pip uninstall`
removes a metadata folder, so a package taken away is noticed on the very next
launch and put back - a record saying all is well is only believed while the
folders it points at are still there.

Setup is safe to interrupt. Every step records that it *finished*, never that it
started, so closing the window halfway leaves the same state as never having run
it, and the next launch carries on from the last step that completed.

### Developers

Nothing above is required. `python main.py --mode gui` from a prepared checkout
works exactly as it always did, `main.py` knows nothing about any of this, and no
module in `core/`, `tools/`, `gui/`, `audio/` or `memory/` imports `launcher/`.

## Prerequisites

1. **Ollama installed and running.**
   ```bash
   ollama serve
   ```
2. **Pull the models referenced in `config/settings.yaml`:**
   ```bash
   ollama pull qwen2.5:7b
   ollama pull mistral-nemo
   ollama pull llama3.2-vision
   ollama pull nomic-embed-text
   ```
   Swap these for smaller/larger variants depending on your hardware — any
   Ollama model that supports tool calling works for the reasoning model. **Which
   one you can afford is set by VRAM, and the binding constraint is the context
   window rather than the weights** — see *Fitting your GPU* below before
   assuming a bigger model is better here.

3. **Python 3.11+**

4. **TTS engine dependency (pyttsx3 uses your OS's native voice engine):**
   - Debian/Ubuntu/Linux: `sudo apt install espeak` (pyttsx3 drives espeak)
   - macOS: no extra install — uses the built-in `NSSpeechSynthesizer`
   - Windows: no extra install — uses the built-in SAPI5 engine

   You won't need to find any of this yourself on a normal machine: the first
   launch asks (see **Microphone and speakers** below) and tells you which piece
   is missing if one is.

   None of this is required to *run* Leti. `--mode text` never touches audio;
   `--mode gui` prints why voice is unavailable and continues text-only, and
   asks for confirmations in the chat window instead of out loud. Only
   `--mode voice` and `--mode continuous` need a working mic and speaker, and
   they now say which piece is missing instead of raising a traceback.

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

## Fitting your GPU

Leti routes tools per request — `core/tool_router.py` picks a relevant subset of
the registry for each message — so a typical turn is shown 15–30 schemas rather
than all of them. But the full set is still what has to fit when routing falls
back: **129 tools serialise to 89,345 characters, roughly 22,300 tokens**,
before the system prompt, personality, user profile, recalled memories,
conversation buffer, or any tool results.
`ollama.num_ctx` is 28672 to leave room for the rest.

This matters because Ollama does not error when a request exceeds `num_ctx` — it
**truncates**. Tools fall off the end and the model behaves as if they were never
registered, which is indistinguishable from a model that is simply bad at its
job. Leti logs a warning at startup if `num_ctx` stops being big enough, so the
next time this happens it says so.

The consequence is that a big context window costs VRAM whether or not a given
turn uses it, and that is what decides your model:

These are the figures Leti's own first-launch check computes, at the current
`num_ctx` of 28672 and after reserving ~1.5 GiB for the desktop and ~0.8 GiB of
headroom:

| VRAM | Reasoning model | Needs | Why |
|---|---|---|---|
| 8 GB | `qwen2.5:3b` | ~3.6 GiB (1.9 weights + 0.98 KV) | 7B needs 6.63 GiB against a 5.7 GiB budget |
| 12 GB | `qwen2.5:7b` (the default) | ~6.6 GiB (4.4 + 1.53 KV) | fits with ~3 GiB spare |
| 16 GB | `qwen2.5:7b` | ~6.6 GiB | 14B needs 14.35 GiB against a 13.7 GiB budget |
| 24 GB | `qwen2.5:14b` | ~14.4 GiB (8.4 + 5.25 KV) | the better tool-picker, and it fits here |

The 14B's KV cache is what moves it up a card: at 28672 tokens it alone is 5.25
GiB, more than a 3B model's entire weights. It does not crash on a smaller card —
Ollama moves layers to the CPU — but an eight-iteration tool turn then takes
minutes, so 7B is the honest default on 12 GB.

Two Ollama environment variables roughly halve the KV cache cost, which is worth
setting on any card:

```
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
```

They are read by the Ollama **server**, not by Leti, so set them where `ollama
serve` starts and restart it. On Windows: *Settings → System → About → Advanced
system settings → Environment Variables*, then restart Ollama from the tray.

**Vision costs a model swap.** `llama3.2-vision` is ~7.9 GB and will not share a
12 GB card with the reasoning model, so each screen-vision call evicts one and
reloads it afterwards — a few seconds each way. That is expected, not a fault.

### Windows notes

- **TTS and audio need nothing extra.** pyttsx3 uses the built-in SAPI5 engine,
  and pyaudio ships prebuilt wheels. `espeak` is a Linux-only prerequisite.
- **Whisper on the GPU.** `stt.device` defaults to `auto`: it uses CUDA when
  torch reports a working CUDA build and the CPU when it doesn't. Plain
  `pip install openai-whisper` installs the **CPU-only** torch on Windows, so
  until you reinstall torch from PyTorch's CUDA index this will correctly, and
  quietly, stay on the CPU. Setting `device: "cuda"` by hand no longer crashes
  when it can't be honoured — it warns and falls back.
- **The interface window is Edge WebView2**, which is Chromium. The page detects
  that and enables a compositing hint worth about a third of its CPU; the same
  hint is withheld on Linux and macOS, where the native window is WebKit and it
  measured *worse*. Nothing about how it looks changes either way.

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

**The layout.** A top bar (system status, clock), three columns, and a terminal line
across the bottom. Left: local conditions, resource monitor, task queue. Centre: the
radar, with the audio-reactive core at its middle, a level meter, and the command
button. Right: the voice pipeline's state with four controls, and the session log.
Everything on it is wired to something real — the panels are readouts, not scenery.

- **System status** (top bar) is derived from the same CPU/memory/disk figures the
  monitor shows, against the thresholds `system_report` itself flags on: it reads
  OPTIMAL until one of them is genuinely under pressure.
- **Weather** is Open-Meteo, no key needed — IP-based location by default, or set
  `weather.latitude`/`longitude` in `config/settings.yaml` to override. With no
  location configured it says so rather than showing a placeholder number.
  **It keeps working offline**: the last successful reading is cached to
  `data/weather_cache.json` and served when the network is gone, labelled
  `OFFLINE · 40 MIN AGO`. A local-first assistant that blanks its dashboard the
  moment the wi-fi drops is the wrong shape — but a cached reading is only useful
  if it admits to being one, so the age travels with it everywhere, including into
  the `get_weather` tool's output (`stale`, `age_minutes`) so Leti doesn't report
  an hour-old temperature as the weather right now. The resolved coordinates are
  cached too, since IP geolocation needs the network as much as the forecast does.
  Two things it deliberately won't do: invent a reading when none has ever been
  taken, and answer "you never told me where you are" with a reading from wherever
  you used to be. The clock beside it is local and never depended on anything.
- **Resource monitor** and the **task queue** are live (`system_report`, and the same
  to-do storage a spoken "add a task" writes to).
- **The radar** runs off a single clock. Everything that moves — the core, the
  sweep, both rings, the contact blips, the level meter — is advanced by one
  animation loop in seconds of elapsed time, not in frames. That is what fixed the
  jitter: motion used to step by a fixed amount per frame (so it ran at double
  speed on a 120Hz screen), each mode had its own loop with its own counter (so
  switching to speaking made the core jump at the moment it should have been
  smoothest), the speaking animation re-rolled `Math.random()` per bin per frame
  (sixty unrelated shapes a second rather than a wobble), and the rings were three
  separate SMIL timelines beside the script's own that came back out of step after
  a backgrounded tab. Now the shape eases toward its target with frame-rate
  independent smoothing, so 60Hz and 144Hz look the same rather than merely running
  at the same speed, and the blips light up as the beam passes them instead of on
  timers of their own. `prefers-reduced-motion` stops the rotation; the core still
  tracks audio, because that part is information rather than decoration.
- **What it costs.** The radar redraws at whatever rate the moment deserves: full
  rate while the core is following live audio, ~30fps when idle and you are looking
  at it, and 8fps while the window is behind something else — an interface that
  sits open all day should not redraw a decorative sweep for an audience that isn't
  there. Hidden entirely, it stops. There are deliberately **no CSS animations
  anywhere in the page**: one perpetual keyframe animation on a 7px status dot
  measured at roughly 1.7 cores on its own, because a running CSS animation keeps
  the browser's whole frame pipeline going for as long as the page is open.
  Anything that pulses is driven from the same animation clock instead. The two
  idle rates are constants at the top of that loop if you want to trade smoothness
  against CPU differently. Two compositing hints do the rest, both measured rather
  than assumed: the status dot has a layer of its own, so the opacity the loop
  writes every frame composites instead of repainting the panel under it; and the
  page's two full-viewport overlays get one only in a browser, because the same
  hint that takes a third off the page's CPU in Blink adds a seventh to it in the
  WebKitGTK build behind the native window. Neither changes how anything looks —
  the windows say which they are in their URL so the choice is made before the
  first paint.
- **The core** reacts to real audio: your own voice via the browser's mic (if
  permitted) while the backend is listening, and a synthetic-but-correctly-timed
  pattern while Leti's TTS is speaking (pyttsx3 plays directly to the OS audio device,
  not through the page, so the page can't analyse its actual waveform — the animation
  is triggered by the real speaking start/stop, just not literally amplitude-matched to
  it). The level meter under it reads from the same amplitudes, so the two always agree.
- **Voice pipeline** reads the answers saved by first-run audio setup, so it shows what
  the backend will actually use — including "no signal", which is what a microphone that
  opened but heard nothing gets, rather than being reported as on.
- **The session log** is the current session only, timestamped as things happen. Lines
  replayed from the buffer when you open the page carry a blank stamp instead of a
  fabricated one: that buffer stores no times, and dating them all to the moment the
  page loaded would read as one conversation in one second. The log button expands it
  to a reading size (Escape closes it); image results and diagrams (`search_images`,
  `create_sketch`) appear there on every connected device at once, independent of
  whatever Leti says in text.
- The text input auto-focuses after Leti finishes speaking or after you finish a spoken
  turn, so you can switch to typing without reaching for the mouse.

Below 1200px the three columns fold into two with the radar across the top, and below
820px into a single stack — nothing is hidden at any width, since a phone is a
first-class client here rather than a fallback.

**Minimising.** The control in the top bar shrinks the interface to a radar puck
with LETI at its centre. Click it to come back, or press Escape. Drag it anywhere.

In the desktop app that is a **second native window** (`gui/desktop.py`): 190×190,
frameless, and **always on top**, so it floats over whatever else you are working
in — which is the point, and something a page can only pretend to do from inside
its own window. Dragging moves the real window across the whole desktop, not just
within the app, and it reappears wherever you left it. The two windows load the
*same* page (the puck adds `?puck=1`), so there is one interface and one renderer,
not a second miniature app to keep in step; swapping between them is a hide and a
show, which is why the puck keeps its position, its socket and its unread count.

In a browser tab or on a phone there is no window to float, so the same control
collapses the layout in place and the puck is dragged around the page instead, its
corner remembered in `localStorage`. A browser client deliberately cannot swap the
desktop app's windows: every client shares one session, and minimising on your
phone should not hide the window on your desk.

Minimised is not paused, in either form. Voice keeps running, replies keep arriving
into the log, the state ring still says whether Leti is listening or answering, and
confirmation prompts still reach you. What a puck can't show is the log, so replies
arriving while it is collapsed are counted on a badge, and the first-run audio card
is never opened in the small window — 190px is no place for a dialogue.

Transparency behind the puck is honoured on Linux and macOS and ignored on Windows,
where the puck's own circular backing keeps it looking deliberate.

---

## Configuration

- `config/settings.yaml` — model names, voice pipeline settings, memory
  paths, and safety toggles (including a `dry_run` mode that reports what a
  state-changing tool *would* do instead of executing it — see Safety below).
  This file is the documented reference/template — most of its integration
  sections (email, calendar, weather, etc.) ship commented out with examples.
### Microphone and speakers

The first time you launch Leti it asks whether it may use this computer's
microphone and speakers, and then checks that they actually work. The answer is
saved to `data/audio_setup.json` and never asked again — `/audio` in text mode,
or the **Audio** button in the interface, re-opens the same flow to change it.

Asking deliberately, at a known moment, is the point. On macOS and Windows the
*first attempt to open the microphone* is what triggers the operating system's
own permission dialog; left to itself that appears at some random later moment —
mid-sentence, behind the window, the first time you happen to say the wake word —
and if it's missed, voice silently never works. So in GUI mode Leti does not
touch the microphone at all until the card has been answered: it starts
text-only, asks, and starts listening once you accept, in the same session.

The check is a real one, not a checkbox:

- **Microphone** — it records for three seconds and reports the peak level. A
  microphone that opens but delivers silence (muted input, the wrong device, or
  permission granted to your terminal but not to Python) is indistinguishable
  from a working one unless the level is measured, so the answer is a number and
  a verdict rather than "enabled". If the machine has more than one input, you
  pick which; that choice is what `record_until_silence`, push-to-talk, *and* the
  wake-word listener all open afterwards.
- **Speakers** — it plays a test tone through the system's default output and
  asks whether you heard it. A tone rather than only a spoken sentence, because
  it separates the two failures: nothing at all means the speakers, while hearing
  the beep but not the words means espeak or a voice setting.

There's deliberately no speaker *picker*: pyttsx3 hands audio to whatever output
your OS has set as default and offers no way to choose another, so a picker here
would be a control that silently does nothing. The card names the device the
system will actually use; change it in your OS to change Leti's.

Declining is remembered too, and turns voice off rather than being re-asked every
launch. Three states are tracked, not two — "allowed", "declined", and "never
asked" — so a launch that has nobody at the keyboard to answer (a service, a
piped script, cron's `--mode run-scheduled`) saves nothing and leaves the question
open for the next real launch, instead of recording a decline on your behalf.

### Controlling the computer

Leti opens things the way you'd ask a person to: `launch_app` takes an app name
(`firefox`), a path to a program or document, or a web address (`youtube.com`),
plus `arguments` to open something *in* an app — "open YouTube in Firefox" is
`firefox` with `['https://youtube.com']`. It resolves apps by PATH lookup, then
the platform's own launcher (`open -a` on macOS, `start` on Windows, `gtk-launch`
on Linux), and reports honestly when nothing started rather than claiming success.

A web address on its own goes to your own default browser — the one with your
logins and extensions — which is what "open YouTube" means. That's the one case
`launch_app` doesn't stop to confirm: it runs nothing on the machine, and a prompt
you see twenty times a day is a prompt you stop reading. Starting a program still
asks every time, and the prompt names the arguments, so what you approve is what
happens. The two cases are classified separately in `config/permissions.yaml`
(`action_by_case`), so you can change either without touching the other.

For browsing Leti does itself: `web_search` finds pages, `browser_read_page` reads
one (and is how you navigate) so it can answer from what the page says rather than
a search snippet, and `browser_click`/`browser_fill_form` drive a dedicated
Playwright browser from there.

**Clicking is the last resort, not the first.** Before Leti drives the screen it
asks `choose_computer_approach`, which answers *tool*, *browser* or *gui* and
names what to use instead when something else fits. Hand it the `steps` of a
longer errand and the session it opens keeps that plan. "Send an email" is
`send_email`. "Create a calendar event" is `schedule_meeting`. "Read the
contract PDF" is the document tools.
"Download the invoice from their billing page" is the browser. Only when nothing
covers the job does the mouse come out, and then inside a bounded session:

- it will not act before it has looked at the screen, and a look more than 90
  seconds old is not a reason to click;
- every action invalidates that look, so the next one has to look again;
- what it expected must match what it saw, or the session stops — it does not
  click on into a window it did not predict;
- the same action twice with no change is a loop, and it refuses a third;
- twenty steps is the ceiling, and a plan written up front says how far through
  the errand is — a session that ends early marks its unfinished steps abandoned,
  so nothing is left looking like it is still running;
- an action that changes something — Send, Submit, Delete, Publish — is tracked
  until it has been looked at afterwards, and a session that ends with one
  unchecked says so rather than reporting it as done.

None of that executes anything. Every click is still `mouse_click`, called by the
orchestrator and authorised by SafetyGuard exactly as if you had asked for it —
which is what stops clicking Send being cheaper than sending an email.

Window control (`close_app`, `focus_window`) works on Windows and macOS;
pygetwindow doesn't implement it on Linux, where the tools now say so plainly
instead of surfacing a bare exception.

None of the desktop-control tools are loaded at import time, because on Linux
`import pyautogui` opens an X11 connection and raises `KeyError: 'DISPLAY'` where
there's no display. That used to take the whole application down — including
`--mode text` over SSH, and the cron entry that runs `--mode run-scheduled`,
which cron starts with no `DISPLAY` no matter how many screens the machine has.
Now only the four tools that actually need a screen fail, with an explanation.

- `config/permissions.yaml` — per-tool action classes, forbidden shell patterns,
  and protected filesystem paths. A few tools cover acts of different weight and
  carry an `action_by_case` block instead of one class (see `launch_app`): the tool
  reports which case a call is, this file still decides what each case costs.
  Review and tighten it before giving Leti broad system access. `protected_paths` is worth extending: `read_file`
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

## Watch and act

"Watch TTWO and tell me if it moves more than 5%" is a watch: a condition Leti
checks on its own and reports the moment it *becomes* true. Four tools cover all
of it — `create_watch`, `list_watches`, `manage_watch`, `check_watches` — and
one row in the scheduler evaluates every watch that is due, so ten watches are
one wake-up and no watches cost nothing at all.

What can be watched:

- **Markets.** A symbol, a metric and a threshold. `change_percent` with
  comparison `abs_above` is "moves more than 5% either way"; `volume_ratio`
  above 2 is a volume spike measured against that symbol's own recent average;
  and there are `price`, `gap_percent`, high and low breakouts, the gap to a
  moving average, momentum and volatility. Stocks and crypto, through the
  Alpaca key the trading tools already use.
- **News and events.** A topic, and how strong the reporting has to be before it
  counts. Ten outlets covering one arrest is **one** event: articles are grouped
  by what they say, each group gets a fingerprint, and a fingerprint that has
  been reported is never reported again — so the same story tomorrow is silence,
  while a genuinely new development gets through.
- **Attention.** Whether interest in something is accelerating, measured from
  how much is being published and how heavily the symbol is trading, compared
  against the same watch's own earlier observations.

`also_watch` combines two of these in one watch: "tell me if TTWO moves
unusually **and** there is major GTA VI news" is a single watch with two signals
and one set of rules, not two watches and a coincidence.

**Nothing here predicts.** A trend watch says what attention is doing now, with
the confidence and the number of observations behind it, and it says nothing at
all until it has seen enough to compare one period against another. "Attention
has accelerated sharply over the last 48 hours, on 9 observations, confidence
moderate" is the strongest thing it can honestly produce; a date for a peak is
not something these signals can know.

**Source quality is part of the answer.** Sources are sorted into official,
regulator, major outlet, secondary and social. An event is "confirmed" only when
independent major outlets or an official source carried it; otherwise it is
"reported" or "unverified", and one post on a social platform is never enough.
For anything where being wrong matters — an arrest, a charge, an official
decision — `require_confirmation` refuses to report it at all until it clears
that bar. Every notification names the sources it rests on.

**A failure is never a "no".** A rate limit, an outage, an unknown symbol, a
missing API key, a search that times out: each is recorded as what it is, with
the watch left running and the problem visible. Nothing is ever read as "the
condition is false", because a monitor that confuses those two is worse than no
monitor. A watch that keeps failing is disabled with the reason attached.

**Detection is not permission.** "If it drops 10%, sell it" — the watch can see
the drop. Selling is a task, run by the orchestrator and authorised by
SafetyGuard call by call, exactly as if you had asked for it in conversation. A
watch firing at 3am runs unattended, where external actions are refused outright
rather than confirmed by nobody.

Watches are shown in the Diagnostics panel with their condition, where their
numbers come from, when each was last checked, when it next will be, and buttons
to pause, resume or remove one. **Why** shows what the watch is measuring right
now and why it fired the last time it did.

Market figures say which feed they came from. A free Alpaca key reads **IEX** —
one exchange's view of the tape, not the whole US market — so prices and volume
can differ from a broker's; the consolidated SIP feed needs a paid subscription,
and options data needs an OPRA one. Leti reports what it actually has rather
than implying full coverage, and when a feed is refused it says that instead of
guessing. Set `trading.data_feed` in the Connections settings.

## Modes

Leti has three modes and starts in the general one every time.

| Mode | What it is | How to enter |
|---|---|---|
| **Default** | The assistant this README describes everywhere else | where you start |
| **Coding** | A software-development workspace | "enter coding mode" |
| **Business** | A business operations workspace | "enter business mode" |

**A coding task is not Coding Mode. A business task is not Business Mode.**
"Check my customer emails", "analyse this invoice", "review my leads", "debug
this Python file" are all ordinary Default Mode work and stay there. Only an
explicit command changes mode — "enter business mode", "switch to coding mode",
"exit", or the selector in the Diagnostics panel.

That rule is structural rather than a matter of good behaviour. Explicit mode
commands are matched deterministically before any model call (`core/intent.py`,
about half a microsecond), and **Default Mode has no mode tool at all** — so the
model there has nothing to call however a request is phrased. A regex cannot be
talked into Business Mode by a sentence full of invoices.

Switching changes three things: which tools the turn is shown, one line of system
prompt, and how deliberate the workflow is. Nothing else — the model stays
loaded, Ollama is not touched, and memory, projects, running tasks and
permissions carry straight across. A switch takes well under a millisecond and
involves no model call at all.

Both specialised modes are *smaller* than Default Mode, not larger:

| | Tools shown | Schemas |
|---|---|---|
| Default | 129 | ~22,300 tokens |
| Coding | 62 | ~11,600 tokens |
| Business | 77 | ~14,300 tokens |

**Business Mode costs Default Mode nothing.** Its tools are mode-only, and the
mode switch lives in the specialised modes rather than the general one — so
Default Leti's tool list, schemas and prompt are byte-identical to what they were
before either specialised mode existed.

## Business Mode

A business operations workspace: pipeline, follow-ups, documents, reporting.

Almost everything it uses, Leti already had — the CRM (`business_dashboard`,
`business_next_actions`, `update_lead`), the contact book, mail, the calendar
writer, File Intelligence for proposals and invoices, the data tools for
spreadsheets, the task manager and the scheduler. **All of those stay available
in Default Mode.** Business Mode is not a prerequisite for checking email or
reading an invoice.

What it adds is the things that read those together.

**The briefing.** `business_briefing brief` answers "what should I be doing
today" from what is actually there: what needs your attention, today's meetings,
leads that have gone quiet, active goals, work in flight, what is scheduled next,
what is blocked — and **which sources it cannot see**. A source that fails to
read becomes a named gap, because a section silently missing reads as "nothing to
report there".

**The calendar, now readable.** `business_calendar` reads the same CalDAV account
`schedule_meeting` already writes to — today, tomorrow, a week, or a date range,
with titles, times, durations, attendees and overlap detection. The rule it is
built around: *a calendar Leti cannot reach is never an empty calendar*. If the
server refuses, it says it could not look. Only a calendar that was actually read
and had nothing in it is reported as empty.

**Goals.** Goal → project → task → result, where the projects are Project
Memory's and the tasks are the task manager's — a goal holds references, not
copies, and deleting one never touches them. Progress is reported three ways and
only three: measured (a recorded result against a target you set), by tasks (how
many linked tasks are done, which it says is *work completed, not the goal
achieved*), or **not at all**. A goal with no target and no linked tasks reports
that its progress cannot be determined. It will not produce a percentage from
nothing.

**Batches, previewed before they happen.** "Follow up with every lead nobody has
contacted for a month" is eight emails. `business_batch propose` builds the list
and shows it — eight found, six with addresses, two without, *nothing sent* —
and then only the items you approve come back to be performed. This is
orchestration, not permission: approving a batch means "yes, these are the right
eight", and each email is still sent by `send_email` and still asks SafetyGuard
exactly as it would on its own. The queue cannot send, write or authorise
anything, and a test asserts all three.

**Entity resolution.** "The customer", "that lead", "the meeting tomorrow"
resolve against the data that already exists — leads, contacts, goals, projects,
tasks, calendar events. Three outcomes and no fourth: one match resolves, several
matches *ask which*, and nothing matching says so. Acting on the wrong customer
is worse than asking.

**Numbers with their provenance.** `business_briefing metrics` labels every
figure `observed` (counted from the records), `calculated` (arithmetic on them)
or `assumed` (configuration, not fact) — and anything concluded on top is
interpretation, offered as that. A rate with no denominator comes back as `null`,
never as zero.

The **Business Workspace** (which business, which project, what the objectives
are) lives for the session and is released when you leave, along with any batch
nobody acted on. It is deliberately not a second memory system: anything that
should outlive the session becomes a task, a lead, a goal, a project note or a
scheduled workflow, all of which already have somewhere to live. Goals are stored
as a record type in the business file Leti already keeps — same path, same atomic
write, no second database.

**More capable, not less safe.** Being in Business Mode changes nothing about
permissions. Sending an email is external and asks first, exactly as it does in
Default Mode; updating a lead is a write; reading a briefing is a read. Business
Mode is not a financial, legal, tax or employment adviser and says so — it
analyses, organises, prepares and explains, and leaves regulated decisions to you.

## Coding Mode

**Coding Mode** is a software-development workspace — repository and symbol
search, git checkpoints, targeted test runs, GitHub — and it is only on when you
put it on.

**Default Leti does not become a coding agent.** That is measurable rather than a
promise: `code_map`, `git_workspace` and `github` are not in Default Mode's
routing, not in its fallback and not in its prompt.

What Coding Mode adds:

- **Codebase and symbol intelligence.** `code_map` ranks the files a request is
  about, lists what a file defines (Python is parsed; other languages are matched
  with patterns and say so), finds where a name is used, and works out which
  tests relate to a change. Nothing is indexed, scanned in the background or
  watched — it walks once, per request, and forgets.
- **Git checkpoints that cannot eat your work.** Before a substantial change,
  `git_workspace checkpoint` records where the repository was and, crucially,
  *which files you were already editing*. A rollback then undoes only what Leti
  changed after that point, and refuses a file that was already dirty even if
  asked for it by name. Nothing is stashed, reset or moved.
- **Test → diagnose → fix.** Tests are chosen from what changed rather than run
  wholesale, and a failure is classified before anything is edited: caused by the
  change, pre-existing, unrelated, environment, dependency, or ambiguous. A test
  that was already failing is reported, not adopted.
- **GitHub.** Browse a repository, read files, inspect branches, commits and pull
  requests, create a branch, open a pull request. Reading is staged the same way
  File Intelligence reads a document — discover, identify, read only what matters
  — so reviewing a repository never pulls it into the context window.
- **Coding-specific verification.** Syntax, secrets, debug leftovers and scope,
  each answered `VERIFIED`, `NOT VERIFIED`, `FAILED` or `NOT APPLICABLE`. Not
  applicable means Leti could not check it, which is not the same as fine — and a
  test that was not run has not passed.

**More capable, not less safe.** Every coding tool goes through the same
SafetyGuard as everything else, and reading a repository is classified separately
from writing to one: `git_workspace status` does not ask, `git_workspace push`
does. Pushing checks the remote first and stops if somebody else's commit is
there. Leti will not force-push, `reset --hard`, or delete a branch at all —
those are refused in code, whoever asks. It opens pull requests and never merges
them.

**GitHub credentials** go in the Connections panel (`/settings`), as a
fine-grained personal access token — no password, no classic token. A
fine-grained token is scoped to the repositories you pick and the permissions you
tick, which is narrower than any OAuth scope; `Contents: read` is enough to
review a repository. The token is stored in `config/settings.local.yaml` at 0600,
echoed back as "(currently set)", and never appears in a prompt, a log, a tool
result or an error — anything token-shaped is scrubbed on the way out.

## Reading the request

Before anything is sent to the model, Leti reads what was asked. This costs no
model call and about 70 microseconds: it is regex and word lists over the text
you typed (`core/intent.py`), not a classifier in front of every turn.

What it works out is the *shape* of the request - a question, a lookup, a file
job, a message, something on screen, a watch, or several of those in order - and
from that, how much of Leti this turn needs:

- **A simple request is shown fewer tools.** "What's the price of AAPL" does not
  need thirty schemas in front of it. The adaptive engine (`core/performance.py`)
  hands the router a tighter budget, and the router applies it with its own
  scoring rather than by counting to a number. That distinction is the whole
  feature: an earlier version counted, and dropped `list_files` from "what files
  are in this folder". Measured across ten ordinary requests, exposure falls from
  30.4 tools to 24.4 with no request losing the tool it needed.
- **A complex request is not rationed at all.** "Research five companies and write
  a report" gets everything the router chose, plus one line naming the stages it
  implies and reminding Leti that finishing includes checking.
- **Under real resource pressure** — CPU above 90%, memory above 92% — the two
  pieces of optional work go: the long-term memory search, and deriving a visual
  from a tool result. Nothing Leti can *do* is ever switched off to save CPU, and
  the log says when something was skipped.

**References back are resolved, not guessed.** Ask for five laptops, then say
"compare the first three", and Leti knows which three: the order it listed them
in is kept for the next turn. When there is nothing to point at, the instruction
is to ask which one rather than to pick — "the winner" with no winner on the
table is a question, not a coin flip. The same applies to a missing piece that
would change the answer: "watch that stock" with no symbol anywhere in the
conversation asks which symbol.

**A long task is not done until it has been checked.** Give
`start_autonomous_task` a `success_criteria` and the last thing the task does is
verify itself against it — by opening the file and reading it, not by
remembering writing it. A check that fails produces one bounded correction and
one more check; a task that still cannot satisfy its own criteria is reported as
failed with the reason, never quietly completed. The check is an ordinary step,
so it meets SafetyGuard the ordinary way: if verifying something needs
permission, the task waits for you like any other step.

## What the user is doing by saying it

`shape` (above) says what a request is *about*. The Intent Layer says what the
user is **doing** by making it, which is a different question and the one that
decides whether anything should happen at all:

    conversation - question - information request - command - single-step task
    multi-step task - watch request - correction - cancellation - continuation
    clarification

It reports a confidence, whether an external side effect was asked for, whether
a tool could be needed, whether clarification is required, and whether the
message refers to work already running. All regex over the text, tens of
microseconds, no model call.

It has exactly one power, and it is structural rather than advisory: **a request
that asked for nothing is shown no tools at all.**

- *"That's interesting."* cannot become a web search, because there is nothing to
  call.
- *"Maybe we should open the project."* is a request for an opinion. The hedge is
  the request.
- *"Do you think we should open it?"* still gets a real answer - a hedged
  question is a question.
- *"Interesting - now open Spotify."* still opens Spotify. The imperative is
  tested clause by clause, because reading from position zero loses the order.

## Stop, pause, resume, skip

Task controls are **cooperative**: a step already in flight finishes, because a
tool call cannot be un-made by changing a status. So the states say so.

    RUNNING -> PAUSING -> PAUSED
    RUNNING -> CANCELLING -> CANCELLED
    PAUSED  -> RESUMING -> RUNNING

What cancellation *can* promise is that nothing further starts - checked before
every remaining step and before the verification pass - and that the record says
which steps had already run. A cancelled task is never reported as completed,
and never as failed either: it did what it was told.

"Stop" is matched deterministically, before any model call, and it outranks
everything: with nothing named and several tasks running, it stops all of them,
because a stop that asks a question while the thing it was meant to prevent goes
ahead is a stop that did not work. Every other control resolves which task
through the same resolve / ask / nothing-found rule as everything else, and says
so when it cannot tell.

"Skip this step" marks the step `skipped`, never `done`, so a plan with a hole
in it looks like one.

## What happened to that task

The task store trims - a state file holding every task forever becomes a slow
state file - and that trim is why "what happened to the task from yesterday" had
no answer. Every status transition now also writes a compact record: the
request, the plan, which steps finished, failed or were skipped, the recovery
attempts, the verification result, the final result and the reason it stopped.
No conversation is copied there; conversation lives in the session log.

Records are bounded (200, about 120 KB at capacity) and a record that cannot
support continuing says so. Asking Leti to carry on with something whose plan
was never recorded gets what *is* known and a question, not a confident
resumption of something it cannot describe.

## Recovery that diagnoses

`core/world_state.py` already answered "why did that fail". `core/recovery.py`
answers the next question - what should the next attempt do differently, and is
there any point:

    FAILURE -> CLASSIFY -> DIAGNOSE -> RECOVERY OPTION -> ACT -> VERIFY
            -> CONTINUE or ESCALATE

It executes nothing; it returns text. A refusal is never retried - a different
route to something that was refused is the thing that was refused. An unknown
cause escalates rather than being given a plausible fix. A failure that comes
back **identical** ends recovery rather than spending the budget discovering the
same thing twice. The ceiling is three attempts and nothing may exceed it.

When recovery stops, the report names five things: what failed, how it was
classified, what was attempted, why it stopped, and whether a person is needed.
Every attempt is on the record, so diagnostics never has to guess how many times
something was tried.

## Several tasks at once

Up to three tasks run concurrently, and never two that would ruin each other's
work. A task declares what it needs - `file:report.md`, `app:blender`, `screen` -
read off its own plan, and a task wanting something another is holding stays
queued with the reason written on it:

> This task needs something another task is using: file:report.md (held by the
> downloader). It will start when that one finishes, or you can stop the other
> one.

Reading the same file is not a conflict; writing it is. The screen and the
keyboard are exclusive, because there is one of each. Nothing polls for a
conflict to clear - a task finishing is the only thing that can clear one, so
that is when it is checked.

Ask *"what are you doing?"* and the answer comes from the task store, not the
model. The interface shows one line when there is more than one task
("3 active tasks - 1 needs you, 1 waiting, 1 running") and shows whichever task
needs a person rather than whichever is first.

## Asking the interface, not the screenshot

A screenshot says the word *Settings* appears somewhere. It does not say there
is an enabled button named Settings in this window - and the difference matters
when two things are called Settings, when one is greyed out, when the word is in
a tooltip, or when the list scrolled between looking and clicking.

So `core/ui_targets.py` asks the operating system when it can, in this order:

1. native accessibility / UI automation (Windows UI Automation, macOS
   accessibility, AT-SPI)
2. structured application information (which windows exist)
3. visible text plus window context
4. screenshot / OCR text
5. coordinates

Every resolution reports **which route answered it**, so a click made on screen
text is never mistaken for one made on an element the interface named. The
screenshot path is not removed - it is the fallback it should always have been,
and on a machine with no accessibility stack it is still what happens.

Seven states, and **exactly one of them may act**:

| | |
|---|---|
| **UNIQUE_MATCH** | one element, clearly ahead, enabled and visible - act |
| **AMBIGUOUS** | two the evidence cannot separate - ask, never pick |
| **PARTIAL_MATCH** | close, not close enough to click |
| **DISABLED** | it is there and clicking it would do nothing |
| **STALE** | it was there; it is not the same thing now |
| **NOT_FOUND** | the interface was asked and it is not there |
| **UNSUPPORTED** | nothing could answer - not the same as absent |

`NOT_FOUND` and `UNSUPPORTED` are deliberately different facts: *asked and
absent* versus *nothing looked*.

A resolved element is evidence about a screen, not a fact about an application,
so it is dropped on every action, observation and mismatch, and checked against
its own identity before the click if held more than twenty seconds. Position is
not part of that identity - a window that moved still holds the same button -
but a changed name or automation id is a different button however little it
moved.

**Nothing runs when nothing is happening.** No accessibility daemon, no tree
cache that outlives a session, no desktop scan, no polling. A provider is
imported the first time a resolution needs one; availability is asked at most
once a minute. A query walks one window's descendants, filtered and bounded at
2,000 nodes, and throws the result away. At most five candidates leave the
module, each a few fields wide - resolution costs about **2.5 µs**.

Afterwards the loop closes: the action is compared against what was expected in
the existing VERIFIED / NOT VERIFIED / FAILED words. An action nobody has looked
at since is **NOT VERIFIED**, which is not the same as failed - and never
"clicked successfully" because the click API returned.

## What a task is actually holding

`core/task_conflicts.py` reads a plan and predicts what it will need. That is a
guess, and it misses everything the plan does not spell out: *"tidy up the
project"* never mentions `report.md`, so two tasks could both start and both
write it.

`core/resources.py` is the authoritative half. It watches the **tool boundary** -
the one place where what a task is about to touch is already written down, in
the arguments the tool was called with - and claims before the call, releases
after it, whatever happened. No tracing, no filesystem watcher, no scanning:
`read_file`'s `path` argument already says which file.

    PLAN DECLARATION (a prediction) + RUNTIME OBSERVATION (the truth)
    -> RESOURCE SET -> CONFLICT CHECK -> ACTION

Four modes - READ, WRITE, EXCLUSIVE, CONTROL - and only two readers coexist.
Files resolve through `realpath`, so `./report.md`, `~/proj/report.md` and a
symlink to it are one resource rather than three.

**Acquisition is atomic by construction.** `acquire()` checks and records in one
synchronous function with no `await` anywhere inside it, so two coroutines on
the same event loop cannot both succeed: the second cannot run until the first
returns. No thread, no lock. About **2.7 µs** per acquire-and-release.

A conflict is a **wait**, never a failure. The blocked task goes to
WAITING_FOR_EXTERNAL with the owner named in plain words:

> Task A is currently using project.py (write). This task needs write access to
> the same thing, so it is waiting.

and is woken when the owner releases - event-driven, on the only occasion a lock
can clear. Nothing polls. The owner *failing* releases its locks exactly as
finishing does.

The ledger is in memory and never persisted, because **a lock held by a process
that no longer exists is not a lock**. And it grants nothing: the claim happens
*after* SafetyGuard authorises, so a refused call never holds anything.

## Coming back after a crash

A task that was running when the process died leaves a status saying RUNNING and
nothing else. From that, two very different situations look identical: the step
had not started, and the step had sent an email and the process died before the
answer came back. Treating the second as the first **resends the email**.

`core/checkpoints.py` records the one thing status cannot:

| | |
|---|---|
| **NOT_STARTED** | nothing was attempted |
| **STARTED** | attempted, outcome not yet known |
| **VERIFIED** | finished and checked - the recovery boundary |
| **UNKNOWN_AFTER_CRASH** | it was STARTED when the process disappeared |

That last state is the point. It is not *failed* - failing is something that was
observed - and it is not *done*. What follows depends on whether repeating the
step could do damage.

The checkpoint lives **inside the task**, in the store that already existed,
written by the save path that already went through `atomic_write`. No second
database. It is written at the transitions that were persisting anyway:
**STARTED on disk before a step runs**, VERIFIED once its result is in hand. A
few hundred bytes - no screenshots, no UI trees, no model context, no
conversation - and about **3.2 µs** to build.

Each process gets an id at import, and a checkpoint carries the id of the
process that wrote it, so *"written by a process that is gone"* is a comparison
rather than a guess. No marker file, no heartbeat, nothing left behind when the
power cuts.

On startup each interrupted task is assessed **individually** against five
conditions - a valid checkpoint, nothing irreversible in doubt, permissions
unchanged, connections available, the next action safe to retry - and:

    A -> READY TO RESUME     B -> NEEDS YOU     C -> WAITING

**Nothing resumes itself.** READY_TO_RESUME means "starting this again repeats
nothing", not "start it": the decision stays with the user and comes back
through `resume()`, which meets SafetyGuard like everything else. Locks held by
the dead process are cleared, and the task is told what it had been holding.

## Aiming at a thing, not a pixel

A coordinate is the weakest possible description of a target: wrong the moment
anything moves, and impossible to check afterwards because a pixel has no
identity. So a step that names something - "the Save button", "the tab called
General" - is checked against what was actually read off the screen, and
**refused when that thing is not there.** Clicking where a button used to be is
the failure this prevents, and it is otherwise silent.

Coordinates still work, and are reported as the weak description they are.
Nothing here clicks or types: every action is still the existing tool,
authorised by SafetyGuard exactly as if the user had asked for it directly.

## Watches that say what they did

A watch's action is one of three things and no others: tell you, run a workflow
you already approved, or start a task - and a task's steps go through the
orchestrator, which puts every tool call in front of SafetyGuard with the
unattended rules applied. **There is no path from a watch to a tool that misses
the guard**, and that is a property of the wiring rather than a promise.

Each watch now reports what its action would need permission for, what its
action last actually did (in the same VERIFIED / NOT VERIFIED / FAILED
vocabulary as everything else), and whether it has gone stale - still switched
on and not evaluated in ten of its own intervals, which usually means nothing is
running the scheduler.

## Choosing the context, not accumulating it

Everything above decides what Leti is shown. `core/context_engine.py` decides
what it is *told*.

A turn used to assemble everything available: the personality, the open
project's instructions and its whole file list, the user profile, whatever a
vector search turned up, the full conversation buffer - on a weather question as
readily as on a code review. Now a request goes through six deterministic steps,
none of which involves a model:

    UNDERSTAND -> ENTITIES -> SOURCES -> RETRIEVE -> RANK -> PACKAGE

Only the sources that could plausibly help this request are read. A weather
question inside an open project no longer carries forty filenames. A request
that stands on its own no longer carries the whole buffer - and one that points
backwards at all ("compare the first three", "do it", "why?") carries every
message, because dropping history that mattered costs more than keeping history
that did not. The long-term memory search only happens when the request could be
about something remembered; "convert 40 psi to bar" does not get one.

Measured on a furnished install - open project, filled-in profile, ten exchanges
of history - across ten ordinary requests: **11% less context per turn, about
315 tokens**, assembled in 2.4 ms. `num_ctx` did not move.

Nothing that lives on a network is ever fetched here. A request that needs the
calendar, the web, the screen or the contact book is *named* in the package as
needing a tool, which is honest and more useful than a silent five-second delay
in front of every turn that mentions email.

What the engine leaves out, it says it left out - the diagnostics panel shows
the last turn's sources, what was skipped and why, what was dropped to fit, and
what only a tool can reach.

## Did it actually happen?

A tool call that returns without raising has told you one thing: the call ran.
Whether the email reached anybody, whether the file on disk holds what was meant
to be in it, whether the button that was clicked did anything - those are
different questions, and answering the first as if it were the second is how an
assistant reports work it did not do.

`core/verification.py` holds one vocabulary for all of it, the one Coding Mode
has used since it existed:

| | |
|---|---|
| **VERIFIED** | something was checked and it holds |
| **NOT VERIFIED** | it could not be confirmed - which is not the same as fine |
| **FAILED** | it was checked and it does not hold |
| **NOT APPLICABLE** | there is nothing here to verify |

Every verifier is free: it reads local state, or the other side's own receipt.
A file write is checked by reading the file back. A pull request is believed
because GitHub returned its number. A sent email is **never** reported as
delivered - that happens on somebody else's machine, minutes later, and the only
signal Leti could get is a bounce that has not arrived. A shell command exiting 0
means it ran, not that it did what was wanted, and that is what it says.

When something cannot be confirmed, the model is told inside the tool result,
where it cannot be missed on the way to writing the answer - and only then. A
confirmed call adds no text at all.

## Which one did you mean?

Name overlap gets "Acme" right and "the customer" wrong. `core/entities.py` adds
the signals a person actually uses: it came up two turns ago, it is the project
that is open, an active task names it, the date in the request lines up, the
company or stage matches.

The rule underneath is the one that makes it safe, and it lives in exactly one
function: **one supported candidate resolves; two comparable ones is a question;
nothing supported is "I cannot find that".** A winner inside one signal's margin
of the runner-up counts as comparable - Leti asks rather than picks.

This is deterministic matching against things already recorded, not semantics.
There is no embedding, no similarity model and no learned ranking here, and
`explain()` says so, because "context-aware resolution" reads like something the
code cannot do.

## When something does not work

`core/world_state.py` keeps one small record per piece of work in flight - the
mode, task, goal, project, where it is, the last action that worked, what the
next one expects, what was actually seen - so a failure can be answered with the
right response instead of the same response again:

    OBSERVE -> UNDERSTAND -> ACT -> OBSERVE AGAIN -> COMPARE -> CONTINUE or RECOVER

Failures are classified: a wrong action, a tool failure, an environment that
changed, a stale observation, a permission problem, missing information, a
missing dependency, somebody else's service, the user intervening, an ambiguous
state - or **unknown**, which is a real answer here and is used.

The classification decides what happens. A permission problem is never retried
differently, because retrying a refusal differently is trying to get around it.
A stale observation is re-observed before anything else is tried. An unknown
cause is reported as unknown rather than given a plausible fix. Only the kinds
where a different approach could genuinely work unlock the self-recovery budget
the task manager already had - and that budget is still two attempts and five
minutes, because the failure mode of automatic recovery is a machine that will
not admit defeat.

## Confirmations you can actually answer

"Leti wants to send an email. This reaches outside your computer. Should I go
ahead?" is a question nobody can answer. `core/autonomy.py` works out what a
call means from its arguments:

> Leti wants to send the prepared email. This means: 4 recipient(s):
> a@x.com, b@x.com, c@x.com, d@x.com; subject: "Q3 follow-up". Worth knowing:
> 2 of them have no usable email address and would be skipped. This cannot be
> undone. Should I go ahead?

It decides nothing. SafetyGuard still classifies every call, still reads
`permissions.yaml`, still blocks what is forbidden, still refuses to let a
critical action ride on an inferred approval. This is text. The other half of
asking well is not asking: a low-risk action you have already allowed is not
confirmed again out of politeness, and there is no way for this layer to add a
prompt SafetyGuard did not want.

Overwriting a file is deliberately *not* described as undoable. It is not,
without a snapshot, and reassuring somebody about the one thing they should
hesitate over is worse than saying nothing.

## Run full Leti diagnostics

Say it and Leti checks sixteen subsystems - core, model, Ollama, the tool
registry, permissions, memory, Project Memory, both specialised modes, calendar,
GitHub, voice, computer use, the scheduler, workflows and connections - and
reports each one as PASS, WARNING, FAIL, NOT CONFIGURED, NOT AVAILABLE or
**NOT TESTED**.

That last one is the point. NOT TESTED outranks PASS in the summary, so a run
where nothing could be exercised cannot come back "healthy". Nothing the check
runs writes, sends or deletes anything, and by default nothing is contacted at
all - the connection states come from settings, not from dialling out. Ask for a
*deep* check and it will actually contact the model server.

It runs when you ask and never otherwise. There is no background health monitor,
no timer, and the panel's Run full check button is a button rather than part of
its refresh.

It is a command, matched deterministically like a mode command, not a tool.
Every registered tool's schema sits in Default Mode's fallback, which has 348
tokens of headroom left, and there is nothing here for a model to decide - so it
costs zero tokens and works by voice.

## Which accounts are set up

`core/connections.py` answers "is that account even configured" in about 29
microseconds by reading settings, with no network call and no credential leaving
it. The Connections Manager - the `/settings` sections - still holds the
credentials; this holds nothing.

It remembers what happened the last time something really used a connection, so
a repeatedly failing account reads as failing rather than as configured. One
failure never stops Leti trying again: that would turn a transient error into a
permanent one.

A request that needs an account Leti has not got is told so before it is
attempted, with the section to add it under - rather than attempted, failed, and
reported as an empty result.

## When Leti brings something up first

Leti can raise things you did not ask about: a task that stopped and is waiting
for you, a watch that fired, an errand left with something unchecked, scheduled
work about to run. All four are read from the stores that already hold them —
there is no second notification system and no background process looking for
things to say.

**Noticing has never been permission.** Proactive mode produces sentences. It
cannot run a tool, and the most forward setting lets Leti *offer* to look into
something — an offer being a question. If you say yes, what happens next is an
ordinary turn, authorised call by call by SafetyGuard exactly as if you had
thought of it yourself.

Four levels, in the settings you already have (`/settings`, "Proactive
assistant"):

| Level | What may come up on its own |
|---|---|
| `off` | Nothing at all. |
| `suggestions` | Only while you are already talking. |
| `notifications` | Watches and scheduled work may come up unprompted (the default). |
| `active` | And Leti may offer the obvious next step. |

The same thing is not raised twice within ten minutes, at most three things are
raised at once, and an approval you are blocking on comes before anything else.
One thing Leti deliberately does **not** offer is "you have a meeting in 30
minutes": that needs to read a calendar, and Leti can create events but has no
tool that reads one back.

## Working with files

"Read these PDFs and compare the offers" is an ordinary thing to ask and an
expensive thing to do badly: a folder of contracts is several hundred thousand
tokens, and the answer to "which is cheaper" is two numbers. So Leti works in
stages, each cheaper than reading:

1. **Which files** — `find_documents` ranks the readable files in the open
   project (or a folder you name) against your request, by name and kind. It
   opens none of them.
2. **What they are** — `inspect_document` reports a file's type and size and
   lists its pages, sheets, rows or headings with one line from each.
3. **The part that answers** — `read_document` returns the relevant sections and
   nothing else. `compare_documents` asks the same question of several files at
   once.

Every piece of text comes back attached to the place it came from — `Page 7`,
`Sheet 'Pricing'`, `Rows 51-100`, `## Payment terms` — so Leti can tell you
*"according to Offer_Company_A.pdf, page 3"* and be held to it. Those labels come
from the file's own structure; none is invented, and Leti is told not to cite one
it was not given.

Nothing here can flood the window, because every stage has a ceiling as well as
a filter. An outline is capped at 40 sections and 3,000 characters of preview —
structure, not contents. A comparison has a total budget as well as a per-file
one, and a file that would take it past the total is reported as not read rather
than silently dropped. A 300-page PDF is scanned 120 pages at a time and says so;
naming the page you want (`Page 250`, or a range like `Page 3-7`) opens it
directly without scanning what comes before.

A file it cannot read says so. A missing file, a format with no reader, a corrupt
PDF, or a scan with no text in it each come back as a plain refusal with a reason
— never as an empty extraction that reads like an empty document. When one file
of several is unreadable, the rest are still compared and the problem is named.

Formats: PDF (via `pypdf`, in tolerant mode, page by page so one broken page does
not lose the document), Word `.docx` (no package — a .docx is a zip of XML the
standard library opens), Excel `.xlsx`, CSV/TSV, JSON, Markdown and plain text.
`read_file` is the same reader: hand it a PDF or a .docx and it extracts rather
than returning the bytes.

Word tables come back as tables — rows and cells, tab-separated, in their place
in the document and labelled with it (`Table 1 under 'Pricing table' rows 1-40`),
chunked with the header row repeated if they are long. The cell text is not also
repeated into the surrounding prose, so a table costs its own size once.

**Pictures, metadata and text are three different things,** and Leti does not
blur them. `inspect_document` on an image gives metadata — dimensions, format,
mode. `look_at_image` points the vision model at it and reports what the model
*sees*, marked as such. Text extraction is the third thing, and Leti has no OCR:
there is no Tesseract in the project and none is added here. So a scanned PDF is
refused with the reason ("the pages are images"), never paraphrased, and an
answer from the vision model is never presented as the file's text.

**Projects scope the search.** With a project open, "find the invoices" looks in
that project's folder rather than the disk. With no project and no folder named,
Leti asks which folder instead of searching everything.

Nothing is sent anywhere: every one of these tools reads from disk and returns,
and they carry the same permission class `read_file` does.

## Leti speaks; the text waits until you ask

An assistant that says everything it says and also prints it is a chat window
that makes noise. The eye wins, the voice becomes decoration, and you end up
reading an answer you already heard.

So the answer is **spoken, and the words are held rather than printed**:

    QWEN -> RESPONSE BUFFER -+-> speech -> natural chunks -> TTS -> speaker
                             |
                             +-> visual need -> image / chart / math -> panel

    the words themselves: HIDDEN, until you ask

Asking is deterministic and free. "Show me the text", "show the answer", "let me
read that", "open the transcript" - read by `core/intent.py` the way mode and
lifecycle commands already are, before anything reaches the model. The words you
are shown are **the same string that was spoken**, because showing is a panel
opening over a buffer that never went anywhere. Nothing is regenerated, nothing
is asked twice, and no second answer can disagree with the first.

Hiding is the same in reverse, and it is only about the panel. The text stays,
the speech is not stopped, no task is cancelled and memory is untouched.

Two things still appear without being asked for: the reply when there is no
voice at all, because silence is not an interface; and the next answer while the
panel is already open, because you asked to read Leti and have not asked to stop.

**Task state is not the transcript.** What Leti is DOING - the current task, its
step, waiting, retrying, needs-you - keeps its own panel and is unaffected. What
is hidden is the conversational answer, not the machine's state.

## Saying it in pieces, and saying the mathematics

`core/speech.py` sits between the answer and the engine, because text written to
be read is not text to be spoken.

A model asked about kinetic energy writes `\(E_k = \frac{1}{2}mv^2\)`, and an
engine handed that says *"backslash left paren E sub k"*. So notation is rewritten
into the words a person uses - **"one half m v squared"** - and the original is
left for the visual layer, which wants exactly the markup speech cannot use.
**Raw LaTeX never reaches the engine**, and every utterance is checked for it,
not only the answer as a whole.

The answer is then cut where a person pauses. Not token by token, which is the
stutter that makes synthetic speech sound synthetic, and not all at once, which
leaves nothing to interrupt. A full stop inside `3.14`, `Dr. Adams`, `/etc/hosts`,
`example.com/a.b`, `report.md`, `U.S.` or `9.8 m/s` is **not** the end of
anything, and a sentence too short to be said alone is joined to the next one.

**"Stop" stops the talking too.** The flag is read between utterances and the one
already playing is cut by the engine's own interrupt. Clicking the core while
Leti talks does it; so does saying so. The answer survives in full - including
the part that was never said - so "stop" then "show me the answer" shows all of
it. Work that is running is still stopped by the task controls, which remain the
authority on that.

## Saying it before it is finished

Ollama streams. Leti used to ask for a whole message and wait for it, which on a
long answer is silence for as long as the answer takes and then all of it at
once - a transcript being read out rather than somebody talking.

    Qwen writes -> NDJSON fragments -> speech.Stream -> whole sentences -> TTS
                              |
                              +-> the same response buffer, assembled once

`core/llm_client.py` gained `stream_response`; `chat()` is untouched and is still
what every other caller uses. It is the same **one** call - streaming replaces
the whole-response request rather than adding to it.

The fragments feed a `Stream` in `core/speech.py`, which hands back whole
sentences and **nothing smaller**. It adds no rules of its own: where a sentence
ends, what a full stop is not the end of, and how notation is said are the
functions that were already there. A second copy of those rules living in the
streaming path would drift from the first and only be noticed by ear.

Measured against a real socket: the first utterance is ready **84 ms into a
206 ms answer**. A sentence with no punctuation in it is handed over once it
passes the ceiling, cut at whitespace and never inside a word, a number, a path,
a url or an expression.

An expression split across fragments is **held until it closes**. `\( \frac{a`
is not sayable - `say` only rewrites a span it can see the end of - so cutting
there would hand the engine a backslash and a brace.

A connection that dies mid-answer keeps what arrived, says that it stopped, and
retries nothing: the fallback model is the one retry policy and it covers the
request, not a socket that died halfway through an answer.

## Stop, without waiting for what you are stopping

The turn that is speaking holds the turn lock, so a "stop" routed through the
ordinary path arrived *after* the answer it was meant to stop. A bare stop is now
read **before** the lock and acted on at once.

It sets one flag, which the speech loop and the model stream both read between
pieces, and calls the engine's own interrupt for the utterance already playing.
No thread is killed, nothing is signalled, and there is no second cancellation
system: work that is running is still stopped by the task controls, and a stop
that names a task ("stop the research task") goes to them on the ordinary path.

The stream stops because its body closes. That needed care - a generator
suspended at a `yield` inside its own `async with` is never resumed once the
caller stops iterating, so the socket would have stayed open until the collector
noticed. Against a real server the generation stops **seven fragments in rather
than two hundred**.

Stopping silences; it does not discard. What the model managed to write is kept,
so "stop" then "show me the answer" shows it - including the part that was never
said aloud. A stop belongs to the answer it stopped and is cleared when the next
turn begins: without that, saying it once would mute everything after.

## An equation you can look at

`core/math_render.py` reads the LaTeX a model writes and emits **MathML**, which
every browser lays out natively - the desktop window, a browser and a phone
alike. Nothing is installed, downloaded or served, and nothing is rasterised: a
picture of an equation cannot be selected, searched or scaled.

sympy is already here and already emits LaTeX (`tools/engineering.py` returns
`sympy.latex(result)` for every symbolic answer), but it cannot READ LaTeX
without antlr4 - so the reader is a small recursive descent over the subset that
appears in an answer. Fractions, integrals, derivatives, summations, roots,
sub- and superscripts, Greek letters, matrices and engineering units all render.

The payload is a **tree of named elements, never a string of markup**, and the
page builds each node with `createElementNS` and sets text with `textContent`.
That shape is the security argument rather than a filter: there is no parser
anywhere in the path, so an expression cannot become a tag, an attribute, a
handler or a script, however it was written. Both sides check the same element
list, and neither trusts the other to have done it.

A panel opens only when the mathematics was asked for - "show me the equation",
or a request like *"derive the bending equation and show me the formulas"* whose
answer actually contains notation. **"What is 2 + 2" opens nothing**: it is
answered in one spoken word, and a window for that is decoration. An expression
that cannot be read comes back as nothing at all rather than half an equation on
screen.

## The interface

Three columns, and the middle one is the point.

**Left** is the machine and the day: the time and date, the weather, CPU, memory
and disk, and — where they can be read — the graphics card, its memory and the
model Leti is configured with. Those last three are read once at startup from
the same hardware check the first-launch card runs, not on a poll: a GPU does
not change while Leti is open, and asking costs a subprocess.

**Centre** is Leti. A calligraphic capital L sits in the core, inside a ring that
follows real audio — the microphone while you are speaking, Leti's own speech
while it is answering. The readout under it says what Leti is doing: idle,
listening, thinking, executing, waiting for your approval. Those come from the
orchestrator's own state machine rather than from anything the interface decides.

**Right** is five controls and a log.

| Control | What it is a view of |
| --- | --- |
| Connections | The integrations in `core/settings_editor.py` — connect, configure, disconnect. Passwords and API keys are never shown again once saved. |
| AI settings | The models, temperature, context size, tool limit and routing switch Leti is already running with. Blank fields keep their current value. |
| Voice | The microphone, wake word, Whisper and speech settings, and the setup card. |
| Personality | The six dials in `tools/personality.py`, with presets that are just six numbers each. |
| Permissions | What Leti may do on its own, and which classes stop to ask — the same setting SafetyGuard reads. |

Under them the **RT-LOG** says what Leti is doing in the words a person would
use: *listening, planning, web search done, waiting for your approval, task
completed*. Every line comes from a transition something else already made — the
state machine, the tool router, the task manager, a watch — pushed as it
happens. An idle Leti adds nothing to it and costs nothing for it.

Between the controls and the log, **a task card appears while something is
running** — and only while something is running. It shows the task, its status,
real progress counted from its own steps (`13 / 20`, or "working..." when the
number of steps is not known rather than an invented percentage), the step it is
on, and buttons to pause, resume, retry the step that failed or cancel. Steps
opens the full list: ✓ done, → running, ○ still to come, with each step's result
or its actual error.

When a task stops to ask permission the card says exactly what for — *"Step 3
needs your approval: 'send_email' is an 'external' action"* — with Approve and
Reject. Approving does not run anything: it marks that one step as carrying your
answer and hands the task back to the runner, so the action still goes through
the orchestrator and SafetyGuard, which still refuses to let an irreversible one
ride on approval given in advance.

A GUI errand shows in the same card, with its plan and how far through it is.

Leti is spoken to first and typed to second, so the conversation sits under the
core as a short strip and expands to a reading size on request.

### When something appears on screen

Most answers are words. A floating panel opens by itself only when both of these
are true: it is something a sentence cannot carry — a picture, a chart, a
diagram — and it was either asked for ("show me…", "plot…", "what does it look
like") or produced by a tool whose whole job is visual (`search_images`,
`create_sketch`, `visualize_dataset`).

Everything else Leti *could* show — a table, a set of sources, a document — gets
one quiet line in the RT-LOG that opens it on click, and is otherwise just the
text answer. So "what's the weather in Athens" puts nothing on screen, and "show
me the last five years of Tesla" puts up a chart. The rule is in
`core/artifacts.py`, it is a word list and three tool names, and there is no
second model deciding how to present an answer.

Panels are draggable, resizable and closable, they live inside the page rather
than in new operating-system windows, at most four are open at once, and nothing
about them runs on a clock.

## The icon

`gui/icon.svg` is the source of truth, and carries the same capital L the
interface draws in its core — one pen stroke, filled rather than stroked, so the
weight falls in the downstroke the way a pen puts it there. Three variants exist
because one drawing can't serve every size: the full mark has a HUD ring that
turns to a smudge at taskbar size, so `gui/icon-small.svg` drops the ring and
enlarges the letter with a heavier pen for the 16-32px entries, and
`gui/icon-maskable.svg` is full-bleed with the mark pulled into Android's safe
zone, since launchers crop home-screen icons to the device's own shape.

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

2,006 tests. The suite covers the authorization layer specifically:
protected-path canonicalization, shell-command path checks, `dry_run`, voice
pre-approval scoping, audit redaction, atomic state writes, and the subprocess
runner.

Most of the rest are written around a guarantee rather than a function — the
context engine never drops history a request points back at, verification never
infers "it happened" from "the call returned", entity resolution never picks
between two comparable candidates, the failure classifier never retries a
refusal, the full check never calls something healthy that it did not test.
Those are checked by breaking them: each one has a mutation that removes it, and
each mutation is confirmed to fail a test before the code goes back.

Two suites are structural. One audits the architecture — that each capability
still has one home, that no module defines a second orchestrator or builds its
own store or calls the model, that mode activation is still deterministic and
still unreachable from the model. The other pins the cost: no background work,
no threads started at import, no permanent index, and Default Mode's tool
fallback still exactly 89,345 characters, so nothing was paid for by raising
`num_ctx`. `pyflakes` runs over the application on every test run — both bugs it
found were functions that looked right, were covered by nothing, and raised
`NameError` the first time a user reached them.
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
│   ├── tool_router.py         # Picks which tool schemas this request is shown (not what it may run)
│   ├── safety_guard.py        # Permissions, confirmation barrier, audit logger
│   ├── permission_center.py   # A readable VIEW of safety_guard/permissions.yaml - not a second authority
│   ├── intent_signals.py      # Approval/denial phrase detection (voice pre-approval, -y shorthand)
│   ├── confirmation.py        # CLI + voice confirmation callbacks (shared by CLI and GUI modes)
│   ├── console_input.py       # Single shared stdin reader (cancellable prompts)
│   ├── task_manager.py        # Objectives that outlive a turn: steps, pause/resume, bounded recovery
│   ├── workflows.py           # Natural-language workflows: trigger + conditions + ordered steps
│   ├── modes.py               # Default / Coding / Business: which tools exist for this turn
│   ├── business.py            # Business Mode's brain: briefing, entities, metrics, workspace
│   ├── business_goals.py      # Goal -> project -> task -> result, over existing stores
│   ├── business_approvals.py  # Preview a batch, approve items - never executes anything
│   ├── coding.py              # Coding Mode's brain: symbols, test selection, verification
│   ├── git_ops.py             # Git, carefully - checkpoints that cannot eat your work
│   ├── github_client.py       # GitHub over the API, with a token it never says
│   ├── intent.py              # What the request IS, read deterministically before it is sent
│   ├── context_engine.py      # Which context this request needs - chosen, not accumulated
│   ├── ui_targets.py          # Which element that is, asked of the OS before the screenshot
│   ├── resources.py           # What a task is actually holding, at the tool boundary
│   ├── checkpoints.py         # The last thing Leti knew, written before it could be lost
│   ├── speech.py              # How an answer is said, and where it is safe to pause
│   ├── math_render.py         # An equation you can look at, as MathML the browser lays out
│   ├── transcript.py          # What Leti just said, kept so it can be shown without being said again
│   ├── task_history.py        # What happened to a task after the store trimmed it
│   ├── task_control.py        # "Stop" reaching the right task, through the one resolver
│   ├── task_conflicts.py      # Two tasks, one file: which one waits
│   ├── recovery.py            # What to try after a failure, and when to stop trying
│   ├── verification.py        # VERIFIED / NOT VERIFIED / FAILED / NOT APPLICABLE, for every domain
│   ├── entities.py            # Which one they meant, and when to ask instead of decide
│   ├── world_state.py         # What Leti believed, and why something did not work
│   ├── objectives.py          # GOAL -> PLAN -> ACTIONS -> RESULTS -> PROGRESS -> NEXT ACTION
│   ├── connections.py         # Which accounts are set up - a view of settings, never a store
│   ├── autonomy.py            # Turns a confirmation into a question somebody can answer
│   ├── performance.py         # What a turn may spend, from what it is and what the machine has
│   ├── proactive.py           # What is worth bringing up unasked - sentences only, never actions
│   ├── watches.py             # Watch a condition, act when it changes (uses the scheduler below)
│   ├── signals.py             # What a watch can measure: market, news events, attention
│   ├── system_scheduler.py    # Registers the periodic check with cron/launchd/schtasks
│   ├── computer_use.py        # Chooses tool > browser > GUI, and bounds a GUI session
│   ├── artifacts.py           # Which tool results are worth a floating panel, by shape
│   ├── documents.py           # Reading what is in a file without putting the file in the prompt
│   ├── model_setup.py         # First-launch hardware detection and model recommendation
│   ├── diagnostics.py         # Measured timings, or an honest "Unavailable"
│   ├── atomic_write.py        # Crash-safe state-file writes
│   └── settings_editor.py     # The /settings command - schema-driven, no LLM involved
├── gui/
│   ├── hud.html                # The interface: system rail, the core, controls + RT-LOG, artifacts
│   ├── desktop.py              # The two native windows: main window + always-on-top puck
│   ├── icon.svg                # Icon source: full mark (48px and up)
│   ├── icon-small.svg          # Icon source: simplified, for 16-32px
│   ├── icon-maskable.svg       # Icon source: full-bleed, for Android launchers
│   ├── icons/                  # Generated PNG/.ico/.icns - see scripts/build_icons.py
│   ├── server.py               # HTTP+WebSocket server (aiohttp) - what phones/browsers connect to
│   └── api.py                  # Orchestrator-facing backend, voice loop, TTS wiring
├── audio/
│   ├── wake_word.py           # OpenWakeWord engine
│   ├── stt.py                 # openai-whisper stream/PTT handler
│   ├── tts.py                 # pyttsx3 speech synthesis with interruptibility
│   └── setup.py               # First-run microphone permission check
├── memory/
│   ├── vector_store.py        # ChromaDB long-term facts/preferences
│   └── session_memory.py      # Working memory & conversation buffer (+ SQLite log)
├── tools/
│   ├── base.py                # BaseTool abstract class & JSON schema generator
│   ├── command_runner.py      # Async subprocess helper that keeps exit status/stderr
│   ├── os_control.py          # App launcher, window manager, mouse/keyboard
│   ├── shell_runner.py        # Sandboxed terminal executor
│   ├── file_manager.py        # Safe file read, write, search, organize
│   ├── documents.py           # Which files a request is about, and the parts that answer it
│   ├── browser.py             # Playwright automation
│   ├── computer_use.py        # Decides whether the GUI is the right layer; bounds the session
│   ├── vision.py              # Screen grab & Ollama Vision multimodal analyzer
│   ├── web_search.py          # DuckDuckGo live search
│   ├── image_search.py        # Image search, for when the user wants to SEE something
│   ├── sketch.py              # Diagram/sketch generation
│   ├── coding.py              # Running, testing and inspecting code
│   ├── engineering.py         # Units, symbolic algebra, numerical work
│   ├── data_analysis.py       # Load, clean, analyse and plot datasets
│   ├── business.py            # Leads/prospects/clients and the metrics over them
│   ├── projects.py            # Persistent project workspaces (folder + metadata + context)
│   ├── autonomous.py          # Start/inspect/control long-running objectives
│   ├── workflow_tools.py      # Describe a workflow in words, then run it
│   ├── business_agent.py      # Business Mode's four tools (hidden from Default Mode)
│   ├── coding_agent.py        # Coding Mode's three tools (hidden from Default Mode)
│   ├── watch_tools.py         # Create/list/change/remove watches, and evaluate the due ones
│   ├── scheduler.py           # In-app scheduled tasks (the one scheduler)
│   ├── control_center.py      # Permission Center + Diagnostics panel tools (read-only views)
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

- Streaming feeds the VOICE. A turn with no speak_callback - text-only, because
  the machine has no espeak - takes the whole-response path, which is simpler and
  ends in the same answer. Nothing is lost by it; there is just nothing to start
  early for.
- A preamble the model writes before calling a tool is not spoken. The loop
  throws that prose away and answers again afterwards, so saying it would be
  telling the user something provisional as though it were the answer.
- Roughly one screenful of generation is already on the wire when a stop lands,
  so the model does a little work that is thrown away. Measured against a real
  server: seven fragments of two hundred. Cutting it closer would mean reading
  the socket less eagerly, which would slow every answer to speed up a stop.
- Mathematical rendering reads the subset of LaTeX that appears in an answer.
  Environments beyond the matrix ones, alignment, `\left`/`\right` sizing, cases
  and commutative diagrams are not implemented; an expression using them either
  renders without that structure or comes back as nothing and stays text.
- MathML needs a surface that lays it out. Every current browser does; an old
  embedded WebView might not, and there the LaTeX is shown as text instead of an
  empty box.
- Mathematics is spoken by a shallow rewrite, not by reading the parsed tree. It
  says "one half m v squared" well and a deeply nested expression clumsily - the
  guarantee is only that no markup reaches the engine, not that every expression
  is read beautifully.
- The response buffer holds one answer. "Show me what you said before that" is
  not answerable from it; the conversation is in memory, but only the current
  answer can be put on screen.
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
- Entity resolution is deterministic signal matching, not semantics. It uses the
  name, aliases, what was said earlier, the open project, active tasks, dates and
  recorded relationships — it does not understand that "the turbine people" and
  "Siemens Energy" are the same customer unless something recorded says so.
- Verification is free by design, which bounds what it can confirm. A file write
  is genuinely checked; a sent email is not, and cannot be. Where an external
  read would be the only confirmation, Leti says NOT VERIFIED and names the call
  that would settle it rather than making it on every turn.
- The full diagnostics check does not exercise the microphone, the speakers, the
  screen or any configured account. Those are reported as NOT TESTED, which is
  accurate and is not the same as working.
- Task controls are cooperative, not pre-emptive. "Stop" prevents every future
  action and cannot un-send an email that has already left, and the record says
  which steps had already run rather than pretending otherwise.
- Resource conflicts between tasks are declared from the plan's own text, so a
  task that reaches a file it never mentioned is not protected. Declaring one
  wrongly costs a needless wait rather than a corrupted file, which is the right
  direction to be wrong in.
- Computer Use checks a named target against what `read_screen` reported, which
  is text. It has no accessibility-tree access, so it cannot tell two identically
  labelled buttons apart, and a partial match is reported as partial rather than
  treated as the same thing.
- Concurrency is capped at three and is about not blocking on somebody else's
  server, not throughput: every step of every task still goes through the one
  orchestrator, which serialises turns.
- Accessibility resolution needs a provider the machine actually has. Windows UI
  Automation goes through `uiautomation` or `pywinauto`, macOS through PyObjC
  *and* an accessibility grant, Linux through AT-SPI - none of which Leti
  installs. Without one, targets fall back to the text read off the screen, and
  that is reported on every resolution rather than hidden. The macOS provider
  currently names the frontmost application but does not enumerate elements.
- Runtime resource tracking covers the tools whose arguments name what they
  touch. A tool that opens a file it was never told about is not tracked, and a
  shell command that writes something is tracked as a command rather than as the
  files it wrote.
- Resource locks coordinate Leti's own tasks with each other. They say nothing
  about another program on the machine editing the same file.
- Crash recovery can say a step was in flight; it cannot say what that step did.
  For an irreversible step that is deliberately the end of the automatic path -
  the task waits for a person rather than guessing - and there is no external
  receipt or idempotency key to consult.
- The checkpoint is written before a step and after it. A crash *between* the
  tool call and the checkpoint is exactly the case reported as
  UNKNOWN_AFTER_CRASH: named honestly rather than resolved.
