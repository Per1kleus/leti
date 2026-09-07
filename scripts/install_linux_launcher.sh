#!/usr/bin/env bash
# Run this ONCE after extracting/cloning Leti on Linux:
#   ./scripts/install_linux_launcher.sh
#
# .desktop files can't reliably find "their own folder" on their own - Exec
# needs an absolute path, and that path is only known once you've actually
# placed the project somewhere. This script bakes in the real path so the
# resulting launcher just works, then puts a double-clickable icon on your
# Desktop and in your applications menu.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="$PROJECT_DIR/scripts/_run_gui_mode.sh"
chmod +x "$RUN_SCRIPT"

DESKTOP_FILE_CONTENT="[Desktop Entry]
Type=Application
Name=Leti
Comment=Launch Leti's interface
Exec=$RUN_SCRIPT
Path=$PROJECT_DIR
Icon=$PROJECT_DIR/gui/icon.svg
Terminal=true
Categories=Utility;
"

write_and_trust() {
    local target="$1"
    echo "$DESKTOP_FILE_CONTENT" > "$target"
    chmod +x "$target"
    # Nautilus/GNOME Files won't run a .desktop from Desktop unless it's marked
    # "trusted" - without this, double-clicking just opens it as a text file.
    if command -v gio >/dev/null 2>&1; then
        gio set "$target" metadata::trusted true 2>/dev/null || true
    fi
}

APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"
write_and_trust "$APPS_DIR/leti.desktop"
echo "Installed to app menu: $APPS_DIR/leti.desktop"

if [ -d "$HOME/Desktop" ]; then
    write_and_trust "$HOME/Desktop/Leti.desktop"
    echo "Installed to Desktop: $HOME/Desktop/Leti.desktop"
fi

echo
echo "Done. Double-click 'Leti' on your Desktop or find it in your application"
echo "menu. A terminal window will briefly show setup progress, then Leti's"
echo "interface opens in its own window. If your file manager still asks to"
echo "trust it the first time you double-click, choose 'Trust and Launch' /"
echo "'Allow Launching'."
