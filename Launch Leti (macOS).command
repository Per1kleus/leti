#!/usr/bin/env bash
# Double-click this in Finder to open Leti's interface. macOS runs .command
# files in a fresh Terminal window automatically - that terminal shows setup
# progress and stays open as Leti's log, while the actual interface opens in
# its own window once startup finishes.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$DIR/scripts/_run_gui_mode.sh"
