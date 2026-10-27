"""The Linux smokes' Segoe UI stand-in: pinned, Windows line metrics and
advance widths, no other fonts.

Synthetic sfnts stand in for Selawik and the fallbacks, so nothing downloads. Every table
checksum and head's whole-file adjustment are checked as FreeType-less
readers (and fontTools' strict mode) would.
"""
from __future__ import annotations

import hashlib
import io
import struct
import zipfile

import pytest

from tools import smoke_fonts


def _sfnt(tables: dict[bytes, bytes]) -> bytes:
    count = len(tables)
    header = struct.pack(">IHHHH", 0x00010000, count, 0, 0, 0)
    offset = 12 + 16 * count
    directory, data = b"", b""
    for tag, body in sorted(tables.items()):
        padded = body + b"\0" * (-len(body) % 4)
        directory += struct.pack(">4sIII", tag, smoke_fonts._checksum(body), offset + len(data), len(body))
        data += padded
    return header + directory + data


def _cmap4(ids: dict[int, int]) -> bytes:
    """A format 4 cmap of one segment per character, plus the terminator."""
    points = sorted(ids) + [0xFFFF]
    segments = len(points)
    deltas = [(ids[p] - p) & 0xFFFF for p in points[:-1]] + [1]
    body = struct.pack(f">{segments}H", *points) + b"\0\0" + struct.pack(f">{segments}H", *points)
    body += struct.pack(f">{segments}H", *deltas) + b"\0\0" * segments
    sub = struct.pack(">HHHHHHH", 4, 14 + len(body), 0, 2 * segments, 0, 0, 0) + body
    return struct.pack(">HHHHI", 0, 1, 3, 1, 12) + sub


def _cmap12(ids: dict[int, int]) -> bytes:
    groups = b"".join(struct.pack(">III", p, p, g) for p, g in sorted(ids.items()))
    sub = struct.pack(">HHIII", 12, 0, 16 + len(groups), 0, len(ids)) + groups
    return struct.pack(">HHHHI", 0, 1, 3, 10, 12) + sub


GLYPHS = {ord("a"): 1, ord("b"): 2}


def _font(ascent=2027, descent=-431, gap=0, win=(2210, 514), cmap=_cmap4, metrics=3) -> bytes:
    os2 = bytearray(96)
    struct.pack_into(">HH", os2, 74, *win)
    hhea = bytearray(36)
    struct.pack_into(">Ihhh", hhea, 0, 0x00010000, ascent, descent, gap)
    struct.pack_into(">H", hhea, 34, metrics)
    head = bytearray(54)
    struct.pack_into(">II", head, 0, 0x00010000, 0)
    struct.pack_into(">H", head, 18, 2048)
    hmtx = b"".join(struct.pack(">Hh", 1000, 0) for _ in range(metrics)) + b"\0\0" * (3 - metrics)
    return _sfnt({b"OS/2": bytes(os2), b"hhea": bytes(hhea), b"head": bytes(head),
                  b"cmap": cmap(GLYPHS), b"hmtx": hmtx})


def _advances(font: bytes) -> list[int]:
    tables = smoke_fonts._tables(font)
    _, hmtx, _ = tables[b"hmtx"]
    (metrics,) = struct.unpack_from(">H", font, tables[b"hhea"][1] + 34)
    return [struct.unpack_from(">H", font, hmtx + 4 * g)[0] for g in range(metrics)]


def _verify(font: bytes) -> None:
    tables = smoke_fonts._tables(font)
    for tag, (entry, offset, length) in tables.items():
        (recorded,) = struct.unpack_from(">I", font, entry + 4)
        body = bytearray(font[offset:offset + length])
        if tag == b"head":
            struct.pack_into(">I", body, 8, 0)
        assert recorded == smoke_fonts._checksum(bytes(body)), tag
    assert smoke_fonts._checksum(font) == 0xB1B0AFBA


def test_hhea_takes_the_win_metrics_and_checksums_hold():
    font = smoke_fonts.windows_metrics(_font())
    _, hhea, _ = smoke_fonts._tables(font)[b"hhea"]
    assert struct.unpack_from(">hhh", font, hhea + 4) == (2210, -514, 0)
    _verify(font)


def test_nothing_but_hhea_and_the_checksums_changes():
    before = _font(gap=269)
    after = smoke_fonts.windows_metrics(before)
    assert len(after) == len(before)
    tables = smoke_fonts._tables(before)
    changed = {i for i, (a, b) in enumerate(zip(before, after)) if a != b}
    hhea_entry, hhea, _ = tables[b"hhea"]
    _, head, _ = tables[b"head"]
    allowed = (set(range(hhea + 4, hhea + 10)) | set(range(hhea_entry + 4, hhea_entry + 8))
               | set(range(head + 8, head + 12)))
    assert changed and changed <= allowed


def _archive(*names: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        for name in names:
            bundle.writestr(name, _font() if name.endswith(".ttf") else b"other")
    return stream.getvalue()


@pytest.mark.parametrize("cmap", [_cmap4, _cmap12])
def test_the_cmap_gives_each_characters_glyph(cmap):
    assert smoke_fonts.glyph_ids(_font(cmap=cmap)) == GLYPHS


def test_advances_take_the_windows_widths_in_the_fonts_units():
    font = smoke_fonts.set_advances(_font(), {"a": 508.79, "z": 400})
    assert _advances(font) == [1000, round(508.79 * 2048 / 1000), 1000]
    _verify(font)


def test_a_glyph_sharing_the_last_advance_is_refused():
    with pytest.raises(ValueError, match="U\\+0062"):
        smoke_fonts.set_advances(_font(metrics=2), {"b": 500})


def test_install_writes_patched_fonts_and_the_conf(tmp_path, monkeypatch):
    archive = _archive("selawk.ttf", "selawkl.ttf", "selawk.woff")
    monkeypatch.setattr(smoke_fonts, "SHA256", hashlib.sha256(archive).hexdigest())
    fallback = tmp_path / "Fallback.ttf"
    fallback.write_bytes(_font())
    conf = smoke_fonts.install(archive, tmp_path / "fonts", {str(fallback): "700"})
    out = tmp_path / "fonts"
    assert sorted(p.name for p in out.glob("*.ttf")) == ["Fallback.ttf", "selawk.ttf", "selawkl.ttf"]
    assert not list(out.glob("*.woff"))
    for font in out.glob("*.ttf"):
        _verify(font.read_bytes())
    widths = smoke_fonts.json.loads(smoke_fonts.WIDTHS.read_text(encoding="utf-8"))["widths"]
    assert _advances((out / "selawk.ttf").read_bytes())[1] == round(widths["400"]["a"] * 2.048)
    assert _advances((out / "Fallback.ttf").read_bytes())[1] == round(widths["700"]["a"] * 2.048)
    # A weight the page never draws keeps Selawik's own advances.
    assert _advances((out / "selawkl.ttf").read_bytes()) == [1000, 1000, 1000]
    text = conf.read_text(encoding="utf-8")
    root = out.resolve().as_posix()
    assert "<include ignore_missing=\"yes\">/etc/fonts/fonts.conf</include>" in text
    assert f"<dir>{root}</dir>" in text
    assert "<string>Segoe UI</string>" in text and "<string>Selawik</string>" in text
    assert f"<acceptfont><glob>{root}/*</glob></acceptfont>" in text
    assert "<rejectfont><glob>*</glob></rejectfont>" in text


def test_missing_fallbacks_are_named_before_anything_is_written(tmp_path, monkeypatch):
    archive = _archive("selawk.ttf")
    monkeypatch.setattr(smoke_fonts, "SHA256", hashlib.sha256(archive).hexdigest())
    with pytest.raises(ValueError, match="fonts-dejavu-core"):
        smoke_fonts.install(archive, tmp_path / "fonts", {str(tmp_path / "absent.ttf"): "400"})
    assert not (tmp_path / "fonts").exists()


def test_the_width_table_covers_each_weight_the_page_draws():
    widths = smoke_fonts.json.loads(smoke_fonts.WIDTHS.read_text(encoding="utf-8"))["widths"]
    assert set(widths) == set(smoke_fonts.SELAWIK_WEIGHTS.values()) | set(smoke_fonts.FALLBACKS.values())
    assert all(len(table) == len(widths["400"]) for table in widths.values())


def test_an_archive_off_its_pin_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        smoke_fonts.install(_archive("selawk.ttf"), tmp_path / "fonts", {})
    assert not (tmp_path / "fonts").exists()


def test_an_archive_without_fonts_is_refused(tmp_path, monkeypatch):
    archive = _archive("readme.txt")
    monkeypatch.setattr(smoke_fonts, "SHA256", hashlib.sha256(archive).hexdigest())
    with pytest.raises(ValueError, match="no .ttf"):
        smoke_fonts.install(archive, tmp_path / "fonts", {})


def test_main_exports_the_variable_to_later_steps(tmp_path, monkeypatch, capsys):
    archive = _archive("selawk.ttf")
    monkeypatch.setattr(smoke_fonts, "SHA256", hashlib.sha256(archive).hexdigest())

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = []
    monkeypatch.setattr(smoke_fonts, "FALLBACKS", {})
    monkeypatch.setattr(smoke_fonts.urllib.request, "urlopen",
                        lambda url, timeout: seen.append(url) or Response(archive))
    env = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env))
    assert smoke_fonts.main([str(tmp_path / "fonts")]) == 0
    conf = tmp_path / "fonts" / "fonts.conf"
    assert seen == [smoke_fonts.URL]
    assert env.read_text(encoding="utf-8") == f"FONTCONFIG_FILE={conf}\n"
    assert capsys.readouterr().out.strip() == f"FONTCONFIG_FILE={conf}"


def test_the_pin_is_a_sha256():
    assert len(smoke_fonts.SHA256) == 64 and int(smoke_fonts.SHA256, 16) >= 0
