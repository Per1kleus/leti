"""Tests for the launcher icon assets.

These are committed binaries the launchers reference by path at runtime, so the
failure mode is a launcher with a blank icon - or one that errors - rather than
anything a Python import would catch. What's checked here is that every path
something references actually exists, and that the containers really hold what
their format promises.
"""
from __future__ import annotations

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


def test_windows_shortcut_icon_exists():
    """Which icon a Windows shortcut carries is decided in launcher/shortcuts.py -
    the .ps1 finds an interpreter and asks it, so that one answer serves the
    installer, the .bat launcher and Leti.exe alike. Checked where it is decided,
    and that the file it names is really there."""
    from launcher import shortcuts

    assert shortcuts.ICON == Path("gui") / "icons" / "leti.ico"
    assert (ICON_DIR / "leti.ico").is_file()
    # And the .ps1 still routes to that decision rather than making its own.
    script = (PROJECT_ROOT / "scripts" / "install_windows_launcher.ps1").read_text()
    assert "--install-shortcuts" in script


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
    """Every icon the window might be started with is a file that is there."""
    api = (PROJECT_ROOT / "gui" / "api.py").read_text()
    names = re.findall(r'icons / "([^"]+)"', api)
    assert names, "gui/api.py no longer points the window at an icon"
    for name in names:
        assert (ICON_DIR / name).is_file(), name


def test_windows_is_given_an_ico_and_never_a_png():
    """A PNG here is not a missing icon, it is a crash.

    pywebview's Windows backend hands this to System.Drawing.Icon, which reads
    .ico and nothing else. Given leti-512.png it threw

        ArgumentException: Argument 'picture' must be a picture that can be used
        as a Icon.

    on a .NET thread - which Python cannot catch, so the fallback to the browser
    never ran and the process died on startup. Reported from a real Windows
    machine.
    """
    import sys

    from gui.api import _window_icon

    api = (PROJECT_ROOT / "gui" / "api.py").read_text()
    windows_branch = api[api.index("def _window_icon"):api.index("def run_gui_mode")]
    assert 'icons / "leti.ico"' in windows_branch
    assert "sys.platform.startswith(\"win\")" in windows_branch

    chosen = _window_icon()
    assert chosen is not None and chosen.is_file()
    if sys.platform.startswith("win"):
        assert chosen.suffix == ".ico", f"Windows would be given {chosen.name}"


def test_the_ico_is_readable_by_system_drawing():
    """Every entry has to be a BMP/DIB, not a PNG.

    Pillow's ICO writer produces PNG entries for every size. That is legal .ico
    and Explorer reads it happily; System.Drawing.Icon does not understand it at
    all, which is what crashed the Windows build. scripts/build_icons.py writes
    the container by hand for this reason.
    """
    import struct

    data = (ICON_DIR / "leti.ico").read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind) == (0, 1), "not an icon container"
    assert count >= 4, f"only {count} sizes in the icon"

    for index in range(count):
        width, height, _c, _r, _p, bpp, size, offset = struct.unpack_from(
            "<BBBBHHII", data, 6 + index * 16)
        pixels = width or 256
        assert data[offset:offset + 8] != b"\x89PNG\r\n\x1a\x0a"[:8], \
            f"the {pixels}px entry is a PNG, which System.Drawing cannot read"
        bi_size, bi_width, bi_height, _planes, bi_bpp, compression = struct.unpack_from(
            "<IiiHHI", data, offset)
        assert bi_size == 40, f"the {pixels}px entry is not a BITMAPINFOHEADER"
        assert bi_width == pixels and bi_height == pixels * 2, \
            f"the {pixels}px entry has the wrong dimensions ({bi_width}x{bi_height})"
        assert bi_bpp == 32, f"the {pixels}px entry is {bi_bpp}-bit"
        assert compression == 0, f"the {pixels}px entry is compressed"
        expected = 40 + pixels * pixels * 4 + (((pixels + 31) // 32) * 4) * pixels
        assert size == expected, \
            f"the {pixels}px entry is {size} bytes, not the {expected} a BMP entry needs"


def test_the_icon_builder_does_not_use_pillows_ico_writer():
    """It writes PNG entries, which is the bug above. Pinned so a tidy-up that
    replaces the hand-written container reintroduces it loudly."""
    source = (PROJECT_ROOT / "scripts" / "build_icons.py").read_text()
    builder = source[source.index("def _build_ico"):source.index("def _build_icns")]
    assert 'format="ICO"' not in builder, "Pillow's ICO writer is back"
    assert "BITMAPINFOHEADER" in builder or "biBitCount" in builder

# --- The mark is one mark ----------------------------------------------------------
# The letterform appears in four files: the interface's core and three icon
# variants. They are generated from one centreline at different weights, so the
# path data legitimately differs - but the SHAPE must not, or the app and its
# launcher icon stop being the same thing.

HUD = PROJECT_ROOT / "gui" / "hud.html"
ICON_SVGS = {name: PROJECT_ROOT / "gui" / f"{name}.svg"
             for name in ("icon", "icon-small", "icon-maskable")}


# The letterform's own group, wherever it sits: translate then scale, with the
# translate's two numbers space-separated (the maskable icon's outer wrapper uses
# commas, which is what keeps this from matching it instead).
MARK_GROUP = re.compile(r'<g (?:id="letiMark" )?transform="translate\(-?[\d.]+ -?[\d.]+\) '
                        r'scale\([\d.]+\)">(.*?)</g>', re.S)


def _mark_points(text):
    """Every point of the filled letterform in one file, in its own coordinates."""
    group = MARK_GROUP.search(text)
    assert group, "no letterform group found"
    paths = re.findall(r'\sd="([^"]+)"', group.group(1))
    assert paths, "the letterform group holds no paths"
    numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", " ".join(paths))]
    return list(zip(numbers[0::2], numbers[1::2]))


def _grid(points, n=8):
    """A coarse occupancy grid of the shape, normalised into a unit box."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = (x1 - x0) or 1, (y1 - y0) or 1
    cells = set()
    for x, y in points:
        cells.add((min(n - 1, int((x - x0) / w * n)), min(n - 1, int((y - y0) / h * n))))
    return cells, w / h


def test_the_icons_carry_the_same_letterform_as_the_interface():
    """Redrawing the mark in one place and not the others is the failure this
    catches - the launcher icon and the app would stop being the same thing."""
    reference, ratio = _grid(_mark_points(HUD.read_text()))
    for name, path in ICON_SVGS.items():
        cells, icon_ratio = _grid(_mark_points(path.read_text()))
        difference = len(cells ^ reference)
        # Measured: the ringed and maskable icons differ by 2 cells of 40, and the
        # small variant by 9 because it is drawn with a heavier pen so its
        # hairlines survive 16px. The block L these replaced differs by 36, which
        # is the kind of drift this is for.
        assert difference <= 14, (
            f"{name}.svg draws a different shape from the interface "
            f"({difference} grid cells differ)")
        assert abs(icon_ratio - ratio) < 0.08, (
            f"{name}.svg has different proportions ({icon_ratio:.2f} vs {ratio:.2f})")


def test_no_icon_still_draws_the_old_block_letter():
    """The mark was a geometric block L built from h/v runs. A calligraphic one is
    an outline; there is nothing left for those to be."""
    for name, path in ICON_SVGS.items():
        text = path.read_text()
        assert "h40 v112" not in text and "h78 v218" not in text, f"{name}.svg"
        assert "<rect x=" not in text, f"{name}.svg still has the level bars"


def test_the_letterform_is_filled_not_stroked_everywhere():
    """A stroked path has one width everywhere, which is what makes a letterform
    read as a wire rather than as writing."""
    for name, path in ICON_SVGS.items():
        chunk = MARK_GROUP.search(path.read_text()).group(1)
        assert "stroke" not in chunk, f"{name}.svg strokes the letter"
        assert chunk.count("<path") == 2, f"{name}.svg: expected the letter and its halo"

