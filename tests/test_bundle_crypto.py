"""The password layer: streamed round trips and refusal of anything altered."""
from __future__ import annotations

import io
import os
import zipfile

import pytest

from backend import bundle_crypto as crypto

CHUNK = 4096


def _seal(tmp_path, data: bytes, password="correct horse", name="sealed.ldbundle"):
    path = tmp_path / name
    with path.open("xb") as stream:
        writer = crypto.EncryptingWriter(stream, password, chunk=CHUNK)
        for start in range(0, len(data), 1000):
            writer.write(data[start:start + 1000])
        writer.finish()
    return path


@pytest.mark.parametrize("size", [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 5 * CHUNK + 17])
def test_round_trip_reads_and_seeks_the_exact_plaintext(tmp_path, size):
    data = os.urandom(size)
    path = _seal(tmp_path, data)
    assert crypto.is_encrypted(path)
    with crypto.DecryptingReader(path, "correct horse") as reader:
        assert reader.read() == data
        if size > 10:
            reader.seek(-7, io.SEEK_END)
            assert reader.read() == data[-7:]
            reader.seek(size // 2)
            assert reader.read(5) == data[size // 2:size // 2 + 5]


def test_a_zip_written_through_the_cipher_reads_back_through_it(tmp_path):
    path = tmp_path / "archive.ldbundle"
    with path.open("xb") as stream:
        writer = crypto.EncryptingWriter(stream, "pässword")
        with zipfile.ZipFile(writer, "w") as archive:
            with archive.open("a.bin", "w", force_zip64=True) as member:
                member.write(b"x" * 3_000_000)
            archive.writestr("b.txt", "hello")
        writer.finish()
    # The same password typed in a different Unicode normal form still opens it.
    with crypto.DecryptingReader(path, "pässword") as reader, zipfile.ZipFile(reader) as archive:
        assert archive.read("b.txt") == b"hello"
        assert archive.read("a.bin") == b"x" * 3_000_000
        assert archive.testzip() is None


def test_wrong_password_is_refused_on_open_and_by_the_header_check(tmp_path):
    path = _seal(tmp_path, b"secret gallery")
    with pytest.raises(crypto.WrongPassword):
        crypto.DecryptingReader(path, "wrong")
    header = path.read_bytes()[:crypto.HEADER_SIZE]
    crypto.check_password(header, "correct horse")
    with pytest.raises(crypto.WrongPassword):
        crypto.check_password(header, "wrong")
    with pytest.raises(crypto.WrongPassword):
        crypto.check_password(header, "")


def test_a_password_without_a_utf8_form_is_a_crypto_error(tmp_path):
    path = _seal(tmp_path, b"secret gallery")
    header = path.read_bytes()[:crypto.HEADER_SIZE]
    # A lone surrogate, as JSON can deliver one; the codec's text, which
    # names its position, is not echoed.
    for call in (lambda: crypto.check_password(header, "a\ud800b"),
                 lambda: crypto.EncryptingWriter(io.BytesIO(), "a\ud800b")):
        with pytest.raises(crypto.CryptoError) as caught:
            call()
        assert "position" not in str(caught.value) and "codec" not in str(caught.value)


@pytest.mark.parametrize("damage", ["flip", "truncate", "drop-final", "extend", "swap", "header",
                                    "short-header", "header-only"])
def test_any_alteration_fails_authentication(tmp_path, damage):
    data = os.urandom(4 * CHUNK + 100)
    path = _seal(tmp_path, data)
    blob = bytearray(path.read_bytes())
    head, stride = crypto.HEADER_SIZE, CHUNK + 16
    if damage == "flip":
        blob[head + stride + 10] ^= 1
    elif damage == "truncate":
        del blob[-5:]
    elif damage == "drop-final":
        del blob[head + 4 * stride:]
    elif damage == "extend":
        blob += blob[head:head + stride]
    elif damage == "swap":
        first = blob[head:head + stride]
        blob[head:head + stride] = blob[head + stride:head + 2 * stride]
        blob[head + stride:head + 2 * stride] = first
    elif damage == "short-header":
        del blob[crypto.HEADER_SIZE - 1:]
    elif damage == "header-only":
        del blob[crypto.HEADER_SIZE:]
    else:
        blob[crypto.HEADER_SIZE - 20] ^= 1  # nonce prefix, covered by every chunk's AAD
    path.write_bytes(bytes(blob))
    with pytest.raises(crypto.CryptoError):
        with crypto.DecryptingReader(path, "correct horse") as reader:
            reader.read()


def test_untrusted_parameters_are_bounded_before_any_key_derivation(tmp_path):
    path = _seal(tmp_path, b"x")
    blob = bytearray(path.read_bytes())
    blob[9] = 30  # log2(n): 128 GiB of scrypt memory
    path.write_bytes(bytes(blob))
    with pytest.raises(crypto.CryptoError, match="parameters"):
        crypto.DecryptingReader(path, "correct horse")


def test_the_shipped_parameters_are_the_documented_ones_and_within_the_readers_bounds(
        tmp_path, monkeypatch, shipped_scrypt):
    # Every other test writes the cheapest cost; a default or a bound moved
    # apart would leave every real bundle unreadable with the suite green.
    assert shipped_scrypt == (17, 8, 1) and crypto._CHUNK == 1024 * 1024
    monkeypatch.setattr(crypto, "_LOG2_N", shipped_scrypt[0])
    data = os.urandom(crypto._CHUNK + 5)
    path = tmp_path / "shipped.ldbundle"
    with path.open("xb") as stream:
        writer = crypto.EncryptingWriter(stream, "pw")
        writer.write(data)
        writer.finish()
    assert tuple(path.read_bytes()[9:12]) == shipped_scrypt
    with crypto.DecryptingReader(path, "pw") as reader:
        assert reader.read() == data


def test_an_unfinished_writer_never_yields_a_readable_bundle(tmp_path):
    path = tmp_path / "partial.ldbundle"
    with path.open("xb") as stream:
        writer = crypto.EncryptingWriter(stream, "pw", chunk=CHUNK)
        writer.write(os.urandom(3 * CHUNK))
        writer.close()  # an exception mid-archive closes without finishing
    with pytest.raises(crypto.CryptoError):
        crypto.DecryptingReader(path, "pw")


@pytest.mark.parametrize("writes", [[0], [1], [CHUNK], [CHUNK + 1], [3 * CHUNK + 5],
                                    [CHUNK - 1, 1, 1], [5, 2 * CHUNK, CHUNK - 5, 7]])
def test_writes_of_any_size_seal_the_same_stream(tmp_path, writes):
    data = os.urandom(sum(writes))
    path = tmp_path / "sealed.ldbundle"
    with path.open("xb") as stream:
        writer = crypto.EncryptingWriter(stream, "pw", chunk=CHUNK)
        start = 0
        for size in writes:
            piece = bytearray(data[start:start + size])
            assert writer.write(piece) == size
            piece[:] = bytes(size)  # the caller may reuse its buffer at once
            start += size
        writer.finish()
    with crypto.DecryptingReader(path, "pw") as reader:
        assert reader.read() == data
        if len(data) > CHUNK + 3:
            reader.seek(CHUNK - 3)
            assert reader.read(7) == data[CHUNK - 3:CHUNK + 4]
    # Whole chunks only, plus one final (possibly empty) chunk.
    body = path.stat().st_size - crypto.HEADER_SIZE
    assert body == max(1, -(-len(data) // CHUNK)) * 16 + len(data)


def test_the_writer_s_key_opens_only_its_own_header(tmp_path, monkeypatch):
    path = tmp_path / "sealed.ldbundle"
    with path.open("xb") as stream:
        writer = crypto.EncryptingWriter(stream, "pw", chunk=CHUNK)
        writer.write(b"read back")
        writer.finish()
    calls = []
    real = crypto._derive
    monkeypatch.setattr(crypto, "_derive", lambda *a: calls.append(a) or real(*a))
    with crypto.DecryptingReader(path, "pw", derived=writer.derived) as reader:
        assert reader.read() == b"read back"
    assert calls == []
    # Any other header derives from the password it was given.
    other = _seal(tmp_path, b"other", password="pw", name="other.ldbundle")
    calls.clear()
    with pytest.raises(crypto.WrongPassword):
        crypto.DecryptingReader(other, "wrong", derived=writer.derived)
    assert len(calls) == 1
    # A header altered after writing still fails every chunk with the reused key.
    blob = bytearray(path.read_bytes())
    blob[crypto.HEADER_SIZE - 20] ^= 1
    path.write_bytes(bytes(blob))
    altered = bytearray(writer.derived[0])
    altered[crypto.HEADER_SIZE - 20] ^= 1
    with pytest.raises(crypto.CryptoError):
        with crypto.DecryptingReader(path, "pw", derived=(bytes(altered), writer.derived[1])) as reader:
            reader.read()
