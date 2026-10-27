"""Give Linux Chrome the fonts the smokes were written against.

The page's stack is "Inter", "Segoe UI", system-ui. On Windows, where the
smokes are written, Inter is absent and Segoe UI draws the page; on Linux
neither is there, and the fallback (DejaVu Sans, or an installed Inter) is
wider, so text measured to fit one line or one row wraps. Selawik is
Microsoft's open replacement for Segoe UI. This downloads it (pinned by
SHA-256) and writes a fontconfig file that names it Segoe UI to the
browser and lets the browser see no other fonts but these.

Close is not enough where a smoke measures a fit to the pixel, so each copy
carries Windows' own advance widths (`tools/segoe-widths.json`, measured
with `tools/font-widths.json` on Windows): Selawik's Semibold differs from
Segoe UI's by up to a sixth of an em. Symbols Segoe UI lacks come, on
Windows, from its fallbacks; here from copies of the system's DejaVu Sans
and Noto Color Emoji, given those fallbacks' widths too. Linux lays out a
line by hhea metrics where Windows uses usWin; Selawik's differ (1.2 em
against Segoe UI's 1.33), so the Selawik copies carry the usWin pair in
hhea. Glyph shapes are untouched: this matches layout, not pixels.

    python tools/smoke_fonts.py DIR      # then FONTCONFIG_FILE=DIR/fonts.conf

Under GitHub Actions the variable goes to $GITHUB_ENV for later steps.
Standard library only. The fonts are for the smokes, never served.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/microsoft/Selawik/releases/download/1.01/Selawik_Release.zip"
SHA256 = "3f62c51e05e3b5a1e6241cf92a371f0be2ea1183aa87b30718bbd40832a8d423"
FAMILY = "Segoe UI"
WIDTHS = Path(__file__).with_name("segoe-widths.json")
# Selawik's files at the weights the page draws; Light and Semilight stay as
# they are. The fallbacks are the system's, copied: Bold serves 600 as well,
# and Windows' symbol widths at 600 are its 700 ones.
SELAWIK_WEIGHTS = {"selawk.ttf": "400", "selawksb.ttf": "600", "selawkb.ttf": "700"}
FALLBACKS = {
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf": "400",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf": "700",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf": "400",
}


def _tables(font: bytes) -> dict[bytes, tuple[int, int, int]]:
    """Each table's tag to (directory entry offset, data offset, length)."""
    (count,) = struct.unpack_from(">H", font, 4)
    tables = {}
    for i in range(count):
        entry = 12 + 16 * i
        tag, _, offset, length = struct.unpack_from(">4sIII", font, entry)
        tables[tag] = (entry, offset, length)
    return tables


def _checksum(data: bytes) -> int:
    data += b"\0" * (-len(data) % 4)
    return sum(struct.unpack(f">{len(data) // 4}I", data)) & 0xFFFFFFFF


def _seal(out: bytearray, font: bytes, changed: list[bytes]) -> bytes:
    """Recompute the changed tables' checksums and head's whole-file adjustment."""
    tables = _tables(font)
    for tag in changed:
        entry, offset, length = tables[tag]
        struct.pack_into(">I", out, entry + 4, _checksum(bytes(out[offset:offset + length])))
    # head's checkSumAdjustment is figured with itself as zero.
    _, head, _ = tables[b"head"]
    struct.pack_into(">I", out, head + 8, 0)
    struct.pack_into(">I", out, head + 8, (0xB1B0AFBA - _checksum(bytes(out))) & 0xFFFFFFFF)
    return bytes(out)


def windows_metrics(font: bytes) -> bytes:
    """The font with hhea's ascender/descender/lineGap set to its usWin pair."""
    out = bytearray(font)
    tables = _tables(font)
    _, os2, _ = tables[b"OS/2"]
    win_ascent, win_descent = struct.unpack_from(">HH", font, os2 + 74)
    _, hhea, _ = tables[b"hhea"]
    struct.pack_into(">hhh", out, hhea + 4, win_ascent, -win_descent, 0)
    return _seal(out, font, [b"hhea"])


def glyph_ids(font: bytes) -> dict[int, int]:
    """Code point to glyph id, from the Unicode cmap (format 12, else 4)."""
    tables = _tables(font)
    _, cmap, _ = tables[b"cmap"]
    (count,) = struct.unpack_from(">H", font, cmap + 2)
    subtables = {}
    for i in range(count):
        platform, encoding, offset = struct.unpack_from(">HHI", font, cmap + 4 + 8 * i)
        fmt = struct.unpack_from(">H", font, cmap + offset)[0]
        if (platform, encoding) in ((3, 10), (0, 4), (0, 6)) and fmt == 12:
            subtables.setdefault(12, cmap + offset)
        elif (platform, encoding) in ((3, 1), (0, 3)) and fmt == 4:
            subtables.setdefault(4, cmap + offset)
    ids: dict[int, int] = {}
    if 12 in subtables:
        at = subtables[12]
        (groups,) = struct.unpack_from(">I", font, at + 12)
        for g in range(groups):
            start, end, first = struct.unpack_from(">III", font, at + 16 + 12 * g)
            for point in range(start, end + 1):
                ids[point] = first + point - start
        return ids
    if 4 not in subtables:
        raise ValueError("font has no Unicode cmap")
    at = subtables[4]
    segments = struct.unpack_from(">H", font, at + 6)[0] // 2
    ends = at + 14
    starts = ends + 2 * segments + 2
    deltas = starts + 2 * segments
    ranges = deltas + 2 * segments
    for i in range(segments):
        end, start = struct.unpack_from(">H", font, ends + 2 * i)[0], struct.unpack_from(">H", font, starts + 2 * i)[0]
        delta, range_offset = struct.unpack_from(">hH", font, deltas + 2 * i)[0], struct.unpack_from(">H", font, ranges + 2 * i)[0]
        for point in range(start, end + 1):
            if point == 0xFFFF:
                continue
            if range_offset == 0:
                glyph = (point + delta) & 0xFFFF
            else:
                glyph = struct.unpack_from(">H", font, ranges + 2 * i + range_offset + 2 * (point - start))[0]
                glyph = (glyph + delta) & 0xFFFF if glyph else 0
            if glyph:
                ids[point] = glyph
    return ids


def set_advances(font: bytes, widths: dict[str, float]) -> bytes:
    """The font with each character's advance set to its width (per 1000 em)
    where the font has the character; the others keep their own."""
    out = bytearray(font)
    tables = _tables(font)
    (units,) = struct.unpack_from(">H", font, tables[b"head"][1] + 18)
    (metrics,) = struct.unpack_from(">H", font, tables[b"hhea"][1] + 34)
    _, hmtx, _ = tables[b"hmtx"]
    ids = glyph_ids(font)
    for char, width in widths.items():
        glyph = ids.get(ord(char))
        if glyph is None:
            continue
        if glyph >= metrics:
            raise ValueError(f"U+{ord(char):04X} shares the last advance; it cannot be set alone")
        struct.pack_into(">H", out, hmtx + 4 * glyph, round(width * units / 1000))
    return _seal(out, font, [b"hmtx"])


def fonts_conf(font_dir: Path) -> str:
    """The system's configuration, but only the fonts in font_dir, Selawik
    among them as Segoe UI: no installed font (Inter, another fallback)
    can stand in for one of these."""
    root = font_dir.resolve().as_posix()
    return f"""<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <include ignore_missing="yes">/etc/fonts/fonts.conf</include>
  <dir>{root}</dir>
  <cachedir>{root}/cache</cachedir>
  <match target="scan">
    <test name="family"><string>Selawik</string></test>
    <edit name="family" mode="assign_replace"><string>{FAMILY}</string></edit>
  </match>
  <selectfont>
    <acceptfont><glob>{root}/*</glob></acceptfont>
    <rejectfont><glob>*</glob></rejectfont>
  </selectfont>
</fontconfig>
"""


def install(archive: bytes, out: Path, fallbacks: dict[str, str] | None = None) -> Path:
    digest = hashlib.sha256(archive).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Selawik archive has SHA-256 {digest}, expected {SHA256}")
    fallbacks = FALLBACKS if fallbacks is None else fallbacks
    missing = [path for path in fallbacks if not Path(path).is_file()]
    if missing:
        raise ValueError(f"fallback fonts missing (fonts-dejavu-core, fonts-noto-color-emoji): {', '.join(missing)}")
    widths = json.loads(WIDTHS.read_text(encoding="utf-8"))["widths"]
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = [name for name in bundle.namelist() if name.endswith(".ttf")]
        if not names:
            raise ValueError("Selawik archive holds no .ttf")
        for name in names:
            font = windows_metrics(bundle.read(name))
            if weight := SELAWIK_WEIGHTS.get(Path(name).name):
                font = set_advances(font, widths[weight])
            (out / Path(name).name).write_bytes(font)
    for path, weight in fallbacks.items():
        (out / Path(path).name).write_bytes(set_advances(Path(path).read_bytes(), widths[weight]))
    conf = out / "fonts.conf"
    conf.write_text(fonts_conf(out), encoding="utf-8")
    return conf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Give Linux Chrome the fonts the smokes were written against.")
    parser.add_argument("dir", type=Path, help="where the fonts and fonts.conf go")
    args = parser.parse_args(argv)
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = response.read()
    conf = install(archive, args.dir)
    print(f"FONTCONFIG_FILE={conf}")
    if env := os.environ.get("GITHUB_ENV"):
        with open(env, "a", encoding="utf-8") as stream:
            stream.write(f"FONTCONFIG_FILE={conf}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
