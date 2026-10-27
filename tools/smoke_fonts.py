"""Give Linux Chrome the font the smokes were written against.

The page's stack is "Inter", "Segoe UI", system-ui. On Windows, where the
smokes are written, Inter is absent and Segoe UI draws the page; on Linux
neither is there, and the fallback (DejaVu Sans, or an installed Inter) is
wider, so text measured to fit one line or one row wraps. Selawik is
Microsoft's open replacement for Segoe UI, drawn to its advance widths.
This downloads it (pinned by SHA-256), and writes a fontconfig file that
names it Segoe UI to the browser and hides Inter, as Windows has it.

Linux lays out a line by the font's hhea metrics where Windows uses its
usWin ones; Selawik's differ (1.2 em against Segoe UI's 1.33), which made
every `line-height: normal` box shorter. The copies here carry the usWin
pair in hhea too. Nothing else in the files changes.

    python tools/smoke_fonts.py DIR      # then FONTCONFIG_FILE=DIR/fonts.conf

Under GitHub Actions the variable goes to $GITHUB_ENV for later steps.
Standard library only. The fonts are for the smokes, never served.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/microsoft/Selawik/releases/download/1.01/Selawik_Release.zip"
SHA256 = "3f62c51e05e3b5a1e6241cf92a371f0be2ea1183aa87b30718bbd40832a8d423"
FAMILY = "Segoe UI"


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


def windows_metrics(font: bytes) -> bytes:
    """The font with hhea's ascender/descender/lineGap set to its usWin pair."""
    out = bytearray(font)
    tables = _tables(font)
    _, os2, _ = tables[b"OS/2"]
    win_ascent, win_descent = struct.unpack_from(">HH", font, os2 + 74)
    hhea_entry, hhea, hhea_length = tables[b"hhea"]
    struct.pack_into(">hhh", out, hhea + 4, win_ascent, -win_descent, 0)
    struct.pack_into(">I", out, hhea_entry + 4, _checksum(bytes(out[hhea:hhea + hhea_length])))
    # head's checkSumAdjustment is figured with itself as zero.
    _, head, _ = tables[b"head"]
    struct.pack_into(">I", out, head + 8, 0)
    struct.pack_into(">I", out, head + 8, (0xB1B0AFBA - _checksum(bytes(out))) & 0xFFFFFFFF)
    return bytes(out)


def fonts_conf(font_dir: Path) -> str:
    """The system's configuration plus Selawik as Segoe UI, and no Inter."""
    return f"""<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <include ignore_missing="yes">/etc/fonts/fonts.conf</include>
  <dir>{font_dir.resolve().as_posix()}</dir>
  <cachedir>{(font_dir / "cache").resolve().as_posix()}</cachedir>
  <match target="scan">
    <test name="family"><string>Selawik</string></test>
    <edit name="family" mode="assign_replace"><string>{FAMILY}</string></edit>
  </match>
  <selectfont><rejectfont><pattern><patelt name="family"><string>Inter</string></patelt></pattern></rejectfont></selectfont>
</fontconfig>
"""


def install(archive: bytes, out: Path) -> Path:
    digest = hashlib.sha256(archive).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Selawik archive has SHA-256 {digest}, expected {SHA256}")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = [name for name in bundle.namelist() if name.endswith(".ttf")]
        if not names:
            raise ValueError("Selawik archive holds no .ttf")
        for name in names:
            (out / Path(name).name).write_bytes(windows_metrics(bundle.read(name)))
    conf = out / "fonts.conf"
    conf.write_text(fonts_conf(out), encoding="utf-8")
    return conf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
