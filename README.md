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
back: **123 tools serialise to 79,676 characters, roughly 20,000 tokens**,
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
│   ├── tool_router.py         # Picks which tool schemas this request is shown (not what it may run)
│   ├── safety_guard.py        # Permissions, confirmation barrier, audit logger
│   ├── permission_center.py   # A readable VIEW of safety_guard/permissions.yaml - not a second authority
│   ├── intent_signals.py      # Approval/denial phrase detection (voice pre-approval, -y shorthand)
│   ├── confirmation.py        # CLI + voice confirmation callbacks (shared by CLI and GUI modes)
│   ├── console_input.py       # Single shared stdin reader (cancellable prompts)
│   ├── task_manager.py        # Objectives that outlive a turn: steps, pause/resume, bounded recovery
│   ├── workflows.py           # Natural-language workflows: trigger + conditions + ordered steps
│   ├── watches.py             # Watch a condition, act when it changes (uses the scheduler below)
│   ├── system_scheduler.py    # Registers the periodic check with cron/launchd/schtasks
│   ├── computer_use.py        # Chooses tool > browser > GUI, and bounds a GUI session
│   ├── model_setup.py         # First-launch hardware detection and model recommendation
│   ├── diagnostics.py         # Measured timings, or an honest "Unavailable"
│   ├── atomic_write.py        # Crash-safe state-file writes
│   └── settings_editor.py     # The /settings command - schema-driven, no LLM involved
├── gui/
│   ├── hud.html                # The HUD interface (audio-reactive ring, dashboard, chat)
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
│   ├── watch_tools.py         # Create/list/remove watches
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
