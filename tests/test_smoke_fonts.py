"""The Linux smokes' Segoe UI stand-in: pinned, Windows line metrics, no Inter.

A synthetic sfnt stands in for Selawik, so nothing downloads. Every table
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


def _font(ascent=2027, descent=-431, gap=0, win=(2210, 514)) -> bytes:
    os2 = bytearray(96)
    struct.pack_into(">HH", os2, 74, *win)
    hhea = bytearray(36)
    struct.pack_into(">Ihhh", hhea, 0, 0x00010000, ascent, descent, gap)
    head = bytearray(54)
    struct.pack_into(">II", head, 0, 0x00010000, 0)
    return _sfnt({b"OS/2": bytes(os2), b"hhea": bytes(hhea), b"head": bytes(head)})


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


def test_install_writes_patched_fonts_and_the_conf(tmp_path, monkeypatch):
    archive = _archive("selawk.ttf", "selawkb.ttf", "selawk.woff")
    monkeypatch.setattr(smoke_fonts, "SHA256", hashlib.sha256(archive).hexdigest())
    conf = smoke_fonts.install(archive, tmp_path / "fonts")
    assert sorted(p.name for p in (tmp_path / "fonts").glob("*.ttf")) == ["selawk.ttf", "selawkb.ttf"]
    assert not list((tmp_path / "fonts").glob("*.woff"))
    for font in (tmp_path / "fonts").glob("*.ttf"):
        _verify(font.read_bytes())
    text = conf.read_text(encoding="utf-8")
    assert "<include ignore_missing=\"yes\">/etc/fonts/fonts.conf</include>" in text
    assert f"<dir>{(tmp_path / 'fonts').resolve().as_posix()}</dir>" in text
    assert "<string>Segoe UI</string>" in text and "<string>Selawik</string>" in text
    assert "<rejectfont><pattern><patelt name=\"family\"><string>Inter</string>" in text


def test_an_archive_off_its_pin_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        smoke_fonts.install(_archive("selawk.ttf"), tmp_path / "fonts")
    assert not (tmp_path / "fonts").exists()


def test_an_archive_without_fonts_is_refused(tmp_path, monkeypatch):
    archive = _archive("readme.txt")
    monkeypatch.setattr(smoke_fonts, "SHA256", hashlib.sha256(archive).hexdigest())
    with pytest.raises(ValueError, match="no .ttf"):
        smoke_fonts.install(archive, tmp_path / "fonts")


def test_main_exports_the_variable_to_later_steps(tmp_path, monkeypatch, capsys):
    archive = _archive("selawk.ttf")
    monkeypatch.setattr(smoke_fonts, "SHA256", hashlib.sha256(archive).hexdigest())

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = []
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
