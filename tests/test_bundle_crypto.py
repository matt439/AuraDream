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


@pytest.mark.parametrize("damage", ["flip", "truncate", "drop-final", "extend", "swap", "header"])
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


def test_an_unfinished_writer_never_yields_a_readable_bundle(tmp_path):
    path = tmp_path / "partial.ldbundle"
    with path.open("xb") as stream:
        writer = crypto.EncryptingWriter(stream, "pw", chunk=CHUNK)
        writer.write(os.urandom(3 * CHUNK))
        writer.close()  # an exception mid-archive closes without finishing
    with pytest.raises(crypto.CryptoError):
        crypto.DecryptingReader(path, "pw")
