"""Tests for the launcher icon assets.

These are committed binaries the launchers reference by path at runtime, so the
failure mode is a launcher with a blank icon - or one that errors - rather than
anything a Python import would catch. What's checked here is that every path
something references actually exists, and that the containers really hold what
their format promises.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = PROJECT_ROOT / "gui" / "icons"

EXPECTED_PNG_SIZES = [16, 24, 32, 48, 64, 128, 192, 256, 512, 1024]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


@pytest.mark.parametrize("size", EXPECTED_PNG_SIZES)
def test_png_exists_at_its_declared_size(size):
    from PIL import Image

    path = ICON_DIR / f"leti-{size}.png"
    assert path.is_file(), f"missing {path} - run: python scripts/build_icons.py"
    with Image.open(path) as im:
        assert im.size == (size, size)


def test_ico_holds_every_size_windows_picks_from():
    """Windows chooses per context - 16px in the title bar, 32 in the taskbar,
    256 in large-icon Explorer views. A single-size .ico gets scaled badly."""
    from PIL import Image

    with Image.open(ICON_DIR / "leti.ico") as ico:
        assert sorted(s[0] for s in ico.ico.sizes()) == ICO_SIZES


def test_small_ico_entries_use_the_simplified_mark():
    """The 16-32px entries must come from gui/icon-small.svg, not a downscale of
    the ringed mark - that's the whole reason two sources exist."""
    from PIL import Image

    with Image.open(ICON_DIR / "leti.ico") as ico:
        embedded = ico.ico.getimage((16, 16)).convert("RGB")
    with Image.open(ICON_DIR / "leti-16.png") as ref:
        assert embedded.tobytes() == ref.convert("RGB").tobytes()


def test_icns_is_a_well_formed_container():
    """Written by hand rather than by iconutil, so the structure is worth
    asserting: 'icns', a length field matching the file, then typed PNG blocks."""
    raw = (ICON_DIR / "leti.icns").read_bytes()
    assert raw[:4] == b"icns"
    assert struct.unpack(">I", raw[4:8])[0] == len(raw)

    seen = {}
    offset = 8
    while offset < len(raw):
        ostype = raw[offset:offset + 4]
        length = struct.unpack(">I", raw[offset + 4:offset + 8])[0]
        assert length >= 8, "block length must include its own header"
        payload = raw[offset + 8:offset + length]
        assert payload[:8] == b"\x89PNG\r\n\x1a\n", f"{ostype!r} is not a PNG"
        seen[ostype] = len(payload)
        offset += length

    assert offset == len(raw), "blocks must tile the file exactly"
    # The retina entries macOS actually uses on a modern display.
    for required in (b"ic09", b"ic10", b"ic13", b"ic14"):
        assert required in seen


def test_maskable_icon_keeps_the_mark_inside_the_safe_zone():
    """Android crops launcher icons to the device's own shape, guaranteeing only
    the centred circle of 80% diameter. Anything outside it can be cut off."""
    from PIL import Image

    with Image.open(ICON_DIR / "leti-maskable-512.png").convert("RGB") as _im:
        im = _im.copy()

    # Sample the corners, which a circular mask always removes: they must be
    # plain background, never part of the mark.
    background = im.getpixel((256, 8))
    for corner in [(4, 4), (507, 4), (4, 507), (507, 507)]:
        pixel = im.getpixel(corner)
        assert all(abs(a - b) < 40 for a, b in zip(pixel, background)), (
            f"corner {corner} carries artwork that a launcher mask would crop"
        )


def test_maskable_icon_is_full_bleed():
    """Unlike the standard icon it must have no rounded corners of its own - the
    launcher supplies the shape, and a second rounded corner shows as a notch."""
    from PIL import Image

    with Image.open(ICON_DIR / "leti-maskable-512.png").convert("RGBA") as im:
        assert im.getpixel((0, 0))[3] == 255, "corner is transparent - not full-bleed"


# --- Everything that points at an icon must point at something that exists ---

def test_linux_desktop_entry_icon_exists():
    script = (PROJECT_ROOT / "scripts" / "install_linux_launcher.sh").read_text()
    icon = re.search(r"^Icon=\$PROJECT_DIR/(.+)$", script, re.M)
    assert icon, "the .desktop entry has no Icon= line"
    assert (PROJECT_ROOT / icon.group(1)).is_file()


def test_windows_shortcut_script_icon_exists():
    script = (PROJECT_ROOT / "scripts" / "install_windows_launcher.ps1").read_text()
    assert "leti.ico" in script
    assert (ICON_DIR / "leti.ico").is_file()


def test_macos_icon_script_icon_exists():
    script = (PROJECT_ROOT / "scripts" / "install_macos_icon.sh").read_text()
    assert "leti.icns" in script
    assert (ICON_DIR / "leti.icns").is_file()


def test_hud_favicon_links_resolve():
    html = (PROJECT_ROOT / "gui" / "hud.html").read_text()
    hrefs = re.findall(r'<link rel="(?:apple-touch-)?icon"[^>]*href="([^"]+)"', html)
    assert hrefs, "the HUD declares no favicon"
    for href in hrefs:
        assert (PROJECT_ROOT / "gui" / href.lstrip("/")).is_file(), href


def test_manifest_icons_resolve():
    """The manifest is built in Python, so a typo'd path is a 404 at install
    time on a phone - nowhere near where it would be noticed."""
    server = (PROJECT_ROOT / "gui" / "server.py").read_text()
    for src in re.findall(r'"src": "(/icons?[^"]+)"', server):
        assert (PROJECT_ROOT / "gui" / src.lstrip("/")).is_file(), src


def test_pywebview_window_icon_exists():
    api = (PROJECT_ROOT / "gui" / "api.py").read_text()
    match = re.search(r'"icons"\s*/\s*"([^"]+)"', api)
    assert match, "gui/api.py no longer points the window at an icon"
    assert (ICON_DIR / match.group(1)).is_file()
