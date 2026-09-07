#!/usr/bin/env bash
# Shared launch logic for GUI mode, called by the double-clickable launchers
# ("Launch Leti (macOS).command" on macOS, the installed .desktop on Linux).
# Not meant to be double-clicked directly - it doesn't open its own terminal
# window on its own.
#
# First run: creates leti_env/ in this folder and installs everything from
# requirements.txt (plus the Playwright browser and, on Linux, the system
# package pywebview's native window needs) automatically - no manual `pip
# install` needed. Every run after that just launches, unless
# requirements.txt has changed, in which case it re-syncs the venv first.
set -uo pipefail

# Resolve the real project directory regardless of where this was launched
# from (double-clicking from a file manager often sets a surprising cwd).
# This script lives in scripts/, so the project root is one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR" || { echo "Could not find the Leti folder at $PROJECT_DIR"; read -r -p "Press Enter to close..."; exit 1; }

VENV_DIR="./leti_env"
VENV_PYTHON="$VENV_DIR/bin/python"
HASH_MARKER="$VENV_DIR/.requirements_hash"
PLAYWRIGHT_MARKER="$VENV_DIR/.playwright_installed"

fail() {
    echo
    echo "ERROR: $1"
    read -r -p "Press Enter to close..."
    exit 1
}

echo "Leti - starting the interface"
echo "(from: $PROJECT_DIR)"
echo

# A system python3 (not the venv - it doesn't exist yet on first run) is
# needed to create the venv at all.
if ! command -v python3 >/dev/null 2>&1; then
    fail "No python3 found on this system. Install Python 3.10+ first, then re-run this."
fi

# --- First run (or a deleted venv): create it -----------------------------
if [ ! -x "$VENV_PYTHON" ]; then
    echo "First run detected - setting up leti_env/ (this only happens once)..."
    if ! python3 -m venv "$VENV_DIR"; then
        fail "Could not create the virtual environment. On Debian/Ubuntu this usually means \
the venv module isn't installed - run: sudo apt install python3-venv, then try again."
    fi
    echo "Virtual environment created."
    echo
fi

# --- Install/update dependencies, but only when requirements.txt actually
# changed since the last successful install (hash-gated, not a marker
# based on plain existence, so editing requirements.txt is picked up). ---
CURRENT_HASH="$(python3 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" requirements.txt 2>/dev/null)"
STORED_HASH="$(cat "$HASH_MARKER" 2>/dev/null || echo "")"

if [ -z "$CURRENT_HASH" ]; then
    fail "Could not read requirements.txt - is this really the Leti project folder?"
fi

if [ "$CURRENT_HASH" != "$STORED_HASH" ]; then
    echo "Installing dependencies into leti_env/ (first run, or requirements.txt changed)..."
    echo "This can take a few minutes the first time - it's downloading everything Leti needs."
    echo
    if ! "$VENV_PYTHON" -m pip install --upgrade pip --quiet; then
        fail "Could not upgrade pip inside leti_env. Check your internet connection and try again."
    fi
    if ! "$VENV_PYTHON" -m pip install -r requirements.txt; then
        fail "Dependency installation failed - see the pip output above for which package \
failed and why (often a missing system library, e.g. 'sudo apt install portaudio19-dev' \
for pyaudio on Debian/Ubuntu, or 'brew install portaudio' on macOS). Fix that, then \
re-run this launcher - it will pick up where it left off."
    fi
    echo "$CURRENT_HASH" > "$HASH_MARKER"
    echo "Dependencies installed."
    echo
fi

# --- Playwright's browser binary is a separate download from the pip
# package itself, and only needs doing once per venv. ---
if [ ! -f "$PLAYWRIGHT_MARKER" ]; then
    echo "Downloading the Playwright browser (needed for web browsing/automation)..."
    if "$VENV_PYTHON" -m playwright install chromium; then
        touch "$PLAYWRIGHT_MARKER"
    else
        echo "Warning: Playwright browser install failed - browser automation won't work until"
        echo "you run: leti_env/bin/python -m playwright install chromium"
    fi
    echo
fi

# --- pywebview's native window needs a system GTK/WebKit package on Linux that
# pip can't install on its own (macOS/Windows use their OS's built-in webview,
# nothing extra needed there). Best-effort: this covers Debian/Ubuntu, which is
# what this project otherwise assumes; other distros may need the equivalent
# package for their toolkit - GUI mode still works as a browser tab either way,
# it just won't get an automatic native window without this. ---
WEBKIT_MARKER="$VENV_DIR/.webkit_checked"
if [ "$(uname)" = "Linux" ] && [ ! -f "$WEBKIT_MARKER" ]; then
    if ! "$VENV_PYTHON" -c "import gi; gi.require_version('WebKit2', '4.1')" >/dev/null 2>&1 \
       && ! "$VENV_PYTHON" -c "import gi; gi.require_version('WebKit2', '4.0')" >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            echo "Installing the system package pywebview's native window needs on Linux..."
            if sudo apt-get install -y python3-gi python3-gi-cairo gir1.2-webkit2-4.1 >/dev/null 2>&1 \
               || sudo apt-get install -y python3-gi python3-gi-cairo gir1.2-webkit2-4.0 >/dev/null 2>&1; then
                echo "Done."
            else
                echo "Warning: couldn't install the GTK/WebKit package automatically - Leti will still"
                echo "work as a browser tab at the URL printed below, just without its own window."
                echo "To get a native window, see pywebview's docs for your distro's equivalent package."
            fi
        else
            echo "Note: this isn't a Debian/Ubuntu system, so the GTK/WebKit package pywebview needs"
            echo "for a native window wasn't installed automatically - Leti will still work as a"
            echo "browser tab at the URL printed below either way."
        fi
        echo
    fi
    touch "$WEBKIT_MARKER"
fi

# --- Ensure Ollama is installed, running, and has the models Leti needs -----------
# This is what actually fixes "couldn't reach Ollama" instead of just warning about
# it: install it if missing, start it as a persistent background service if it's
# not already running (survives after this window closes), and pull whatever
# models config/settings.yaml asks for (ollama pull is cheap/no-op if already
# present, so this is safe to run on every launch, not just the first).
ensure_ollama() {
    if ! command -v ollama >/dev/null 2>&1; then
        echo "Ollama not found - installing it now (one-time)..."
        if [ "$(uname)" = "Darwin" ]; then
            if command -v brew >/dev/null 2>&1; then
                brew install ollama || fail "brew install ollama failed. Install manually from https://ollama.com/download, then re-run this launcher."
            else
                fail "Homebrew not found, so Ollama can't be auto-installed on macOS. Install it \
manually from https://ollama.com/download, then re-run this launcher."
            fi
        else
            curl -fsSL https://ollama.com/install.sh | sh || fail "The Ollama install script failed. Install manually from https://ollama.com/download, then re-run this launcher."
        fi
        echo "Ollama installed."
        echo
    fi

    OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
    if ! curl -s -m 2 "$OLLAMA_HOST" >/dev/null 2>&1; then
        echo "Starting Ollama in the background (it'll keep running after this window closes)..."
        nohup ollama serve >/tmp/leti_ollama.log 2>&1 &
        disown
        for _ in $(seq 1 30); do
            curl -s -m 2 "$OLLAMA_HOST" >/dev/null 2>&1 && break
            sleep 1
        done
        if ! curl -s -m 2 "$OLLAMA_HOST" >/dev/null 2>&1; then
            fail "Ollama still isn't responding at $OLLAMA_HOST after 30s - check /tmp/leti_ollama.log for details."
        fi
        echo "Ollama is up."
        echo
    fi

    echo "Checking Leti's configured models (skips anything already downloaded)..."
    echo "First run can take a while and needs several GB of disk space - ollama shows its own progress below."
    MODELS="$("$VENV_PYTHON" -c "
import yaml
cfg = yaml.safe_load(open('config/settings.yaml'))
o = cfg.get('ollama', {})
for m in [o.get('reasoning_model'), o.get('fallback_reasoning_model'), o.get('vision_model'), o.get('embedding_model')]:
    if m: print(m)
" 2>/dev/null)"

    if [ -z "$MODELS" ]; then
        echo "Warning: couldn't read the model list from config/settings.yaml - skipping model pull."
        return 0
    fi

    while IFS= read -r model; do
        [ -z "$model" ] && continue
        echo "  - $model"
        ollama pull "$model" || echo "    Warning: failed to pull '$model' - Leti may not work correctly until this succeeds."
    done <<< "$MODELS"
    echo
}
ensure_ollama

"$VENV_PYTHON" main.py --mode gui
STATUS=$?

echo
if [ $STATUS -ne 0 ]; then
    echo "Leti exited with an error (code $STATUS) - see the output above for details."
else
    echo "Leti closed."
fi
read -r -p "Press Enter to close this window..."
