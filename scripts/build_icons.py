#!/usr/bin/env python3
"""Render Leti's icon into every format the launchers need.

    python scripts/build_icons.py

Sources are the two SVGs in gui/ - edit those, re-run this, commit the result.
The generated files are committed because the launchers reference them at
runtime: a user who double-clicks "Launch Leti" must not need a build step.

Two sources, not one, because an icon that works at 512px does not work at
16px. The full mark (gui/icon.svg) has a HUD ring around the letterform; at
taskbar size that ring collapses into a grey smudge and takes the mark with it.
gui/icon-small.svg drops the ring and enlarges the "L" to fill the tile, and is
used for the 16/24/32px entries. That is what .ico is for - one file holding a
different image per size, not one image scaled down.

Uses only what Leti already depends on: Playwright renders the SVG (a real
browser, so gradients and rounded joins come out exactly as designed) and
Pillow assembles the containers.
"""
from __future__ import annotations

import asyncio
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUI_DIR = PROJECT_ROOT / "gui"
ICON_DIR = GUI_DIR / "icons"

FULL_SVG = GUI_DIR / "icon.svg"
SMALL_SVG = GUI_DIR / "icon-small.svg"
MASKABLE_SVG = GUI_DIR / "icon-maskable.svg"

# Sizes rendered from the full mark, and the small sizes rendered from the
# simplified one. 1024 is macOS's retina 512.
FULL_SIZES = [48, 64, 128, 192, 256, 512, 1024]
SMALL_SIZES = [16, 24, 32]

# Windows .ico: the sizes Explorer, the taskbar and Alt-Tab actually pick from.
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]

# macOS .icns entries: (OSType, pixel size). Each holds a complete PNG.
ICNS_ENTRIES = [
    (b"ic11", 32),    # 16pt @2x
    (b"ic12", 64),    # 32pt @2x
    (b"ic07", 128),
    (b"ic13", 256),   # 128pt @2x
    (b"ic08", 256),
    (b"ic14", 512),   # 256pt @2x
    (b"ic09", 512),
    (b"ic10", 1024),  # 512pt @2x
]


async def _render(
    svg_path: Path, sizes: list[int], out_dir: Path, stem: str = "leti"
) -> dict[int, Path]:
    """Rasterize one SVG at each size with headless Chromium.

    `stem` names the output files - the maskable variant needs its own, or
    rendering it would overwrite the same-sized standard icon.
    """
    from playwright.async_api import async_playwright

    svg = svg_path.read_text()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        try:
            for size in sizes:
                page = await browser.new_page(
                    viewport={"width": size, "height": size}, device_scale_factor=1
                )
                await page.set_content(
                    f'<body style="margin:0">'
                    f'<div style="width:{size}px;height:{size}px">{svg}</div></body>'
                )
                # The SVG carries its own width/height; override so it fills the
                # viewport exactly and the screenshot needs no cropping.
                await page.locator("svg").evaluate(
                    "(el, s) => { el.setAttribute('width', s); el.setAttribute('height', s); }",
                    size,
                )
                path = out_dir / f"{stem}-{size}.png"
                await page.screenshot(path=str(path))
                written[size] = path
                await page.close()
        finally:
            await browser.close()
    return written


def _build_ico(pngs: dict[int, Path], out: Path) -> None:
    """Multi-resolution .ico, each size from the source drawn for it.

    Written by hand rather than with Pillow's ICO writer, for the same reason the
    .icns below is: control over what actually lands in the container.

    Pillow writes every entry as a PNG. That is legal .ico and Explorer reads it
    happily - and .NET's System.Drawing.Icon does not. It only understands
    BMP/DIB entries, so the icon Leti shipped made pywebview's Windows backend
    throw

        System.ArgumentException: Argument 'picture' must be a picture that can
        be used as a Icon.

    on a .NET thread, which Python cannot catch, which killed the application on
    startup. So every entry here is a BMP/DIB.

    Each one is a BITMAPINFOHEADER whose height is doubled (the format expects
    the colour data followed by an AND mask), then bottom-up BGRA rows, then the
    mask itself. The mask is all zeros because the alpha channel already says what
    is transparent, but it has to be there and it has to be the right size:
    1 bit per pixel, each row padded to a 4-byte boundary.
    """
    from PIL import Image

    ordered = sorted(ICO_SIZES)
    entries: list[bytes] = []
    for size in ordered:
        image = Image.open(pngs[size]).convert("RGBA")
        if image.size != (size, size):
            image = image.resize((size, size), Image.LANCZOS)

        rows = []
        pixels = image.load()
        for y in range(size - 1, -1, -1):          # bottom-up
            row = bytearray()
            for x in range(size):
                r, g, b, a = pixels[x, y]
                row += bytes((b, g, r, a))          # BGRA
            rows.append(bytes(row))
        colour = b"".join(rows)

        mask_row = ((size + 31) // 32) * 4          # padded to 4 bytes
        mask = b"\x00" * (mask_row * size)

        header = struct.pack(
            "<IiiHHIIiiII",
            40,                 # biSize
            size,               # biWidth
            size * 2,           # biHeight: colour data plus the mask
            1,                  # biPlanes
            32,                 # biBitCount
            0,                  # biCompression: BI_RGB, which is the whole point
            len(colour) + len(mask),
            0, 0, 0, 0,
        )
        entries.append(header + colour + mask)

    directory = struct.pack("<HHH", 0, 1, len(ordered))
    offset = 6 + 16 * len(ordered)
    table = bytearray()
    for size, payload in zip(ordered, entries):
        table += struct.pack(
            "<BBBBHHII",
            0 if size >= 256 else size,    # 0 means 256 in this field
            0 if size >= 256 else size,
            0,                             # no colour palette
            0,                             # reserved
            1,                             # planes
            32,                            # bits per pixel
            len(payload),
            offset,
        )
        offset += len(payload)
    out.write_bytes(directory + bytes(table) + b"".join(entries))


def _build_icns(pngs: dict[int, Path], out: Path) -> None:
    """.icns container: 'icns' + total length, then typed PNG blocks.

    Written directly rather than via iconutil, which only exists on macOS - the
    icon has to be buildable wherever the project is being worked on.
    """
    blocks = bytearray()
    for ostype, size in ICNS_ENTRIES:
        data = pngs[size].read_bytes()
        blocks += ostype + struct.pack(">I", len(data) + 8) + data
    out.write_bytes(b"icns" + struct.pack(">I", len(blocks) + 8) + bytes(blocks))


def main() -> int:
    for svg in (FULL_SVG, SMALL_SVG, MASKABLE_SVG):
        if not svg.exists():
            print(f"error: missing source {svg}", file=sys.stderr)
            return 1

    ICON_DIR.mkdir(parents=True, exist_ok=True)
    pngs: dict[int, Path] = {}
    pngs.update(asyncio.run(_render(FULL_SVG, FULL_SIZES, ICON_DIR)))
    pngs.update(asyncio.run(_render(SMALL_SVG, SMALL_SIZES, ICON_DIR)))

    # Android's home-screen icon. Written under its own stem so it doesn't
    # overwrite the same-sized standard icon, and kept out of `pngs` so it can't
    # be picked up by the .ico/.icns builders.
    maskable = asyncio.run(_render(MASKABLE_SVG, [512], ICON_DIR, stem="leti-maskable"))[512]

    for size in sorted(pngs):
        print(f"  png  {pngs[size].relative_to(PROJECT_ROOT)}")

    print(f"  png  {maskable.relative_to(PROJECT_ROOT)}")

    ico = ICON_DIR / "leti.ico"
    _build_ico(pngs, ico)
    print(f"  ico  {ico.relative_to(PROJECT_ROOT)}")

    icns = ICON_DIR / "leti.icns"
    _build_icns(pngs, icns)
    print(f"  icns {icns.relative_to(PROJECT_ROOT)}")

    print("\nDone. Commit gui/icons/ along with any change to the source SVGs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
