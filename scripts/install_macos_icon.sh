#!/usr/bin/env bash
# Run this ONCE after extracting/cloning Leti on macOS:
#   ./scripts/install_macos_icon.sh
#
# Gives "Launch Leti (macOS).command" Leti's icon in Finder and the Dock. macOS
# stores a file's custom icon as an extended attribute rather than anything
# inside the file, so it can't be shipped in the repo - it has to be applied on
# this machine, once.
#
# Uses AppKit's NSWorkspace.setIcon via AppleScript-ObjC deliberately: the
# older Rez/DeRez/SetFile route needs the Xcode command line tools installed,
# which is a large download to ask for over an icon.
#
# The Linux equivalent is scripts/install_linux_launcher.sh; Windows is
# scripts/install_windows_launcher.ps1.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$PROJECT_DIR/Launch Leti (macOS).command"
ICNS="$PROJECT_DIR/gui/icons/leti.icns"

if [ ! -f "$LAUNCHER" ]; then
    echo "Can't find '$LAUNCHER' - run this from inside the Leti project folder." >&2
    exit 1
fi
if [ ! -f "$ICNS" ]; then
    echo "Can't find '$ICNS'. Regenerate it with: python scripts/build_icons.py" >&2
    exit 1
fi

chmod +x "$LAUNCHER"

osascript - "$ICNS" "$LAUNCHER" <<'APPLESCRIPT'
use framework "AppKit"
use scripting additions

on run argv
    set iconPath to item 1 of argv
    set filePath to item 2 of argv

    set iconImage to current application's NSImage's alloc()'s initWithContentsOfFile:iconPath
    if iconImage is missing value then error "Could not read the icon file."

    set workspace to current application's NSWorkspace's sharedWorkspace()
    set ok to workspace's setIcon:iconImage forFile:filePath options:0
    if ok is false then error "macOS refused to set the icon on that file."
end run
APPLESCRIPT

echo "Icon applied to: $LAUNCHER"
echo
echo "Double-click it to start Leti. If Finder still shows the old icon, that's"
echo "its thumbnail cache - it refreshes on its own, or immediately if you close"
echo "and reopen the folder."
