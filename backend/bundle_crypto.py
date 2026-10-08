"""Password-protected gallery bundles: one streamed, chunked AES-256-GCM layer.

The plaintext is an ordinary bundle ZIP. It is never written to disk: export
streams the ZIP through `EncryptingWriter`, and verification and import read
it back through the seekable `DecryptingReader`, so every chunk is
authenticated with the user's password in the one pass that already checks the
archive.

Layout: a 56-byte header, then chunks of `chunk` plaintext bytes plus a 16-byte
tag. The last chunk may be shorter or empty and is the only one marked final.

    magic "LDBUNDLE" | version | log2(n) | r | p | salt[16] | chunk (u32 BE)
    | nonce prefix[8] | password check[16]

A 64-byte scrypt output gives the AES key (first half) and the password check
(SHA-256 of the second half, so the check reveals nothing about the key).
Chunk i uses nonce prefix || i (u32 BE). Its associated data is
SHA-256(header) || i (u64 BE) || final flag, which binds the header to every
chunk and makes reordering, truncation and extension fail authentication.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import struct
import unicodedata
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"LDBUNDLE"
SUFFIX = ".ldbundle"
_VERSION = 1
_HEADER = struct.Struct(">8sBBBB16sI8s16s")
HEADER_SIZE = _HEADER.size
_TAG = 16
_CHUNK = 1024 * 1024
_LOG2_N, _R, _P = 17, 8, 1  # 128 MiB, about 0.2 s on the measured desktop


class CryptoError(ValueError):
    pass


class WrongPassword(CryptoError):
    pass


def _secret(password: str) -> bytes:
    if not isinstance(password, str) or not password:
        raise WrongPassword("enter the bundle password")
    return unicodedata.normalize("NFC", password).encode("utf-8")


def _derive(password: str, log2_n: int, r: int, p: int, salt: bytes) -> tuple[bytes, bytes]:
    out = Scrypt(salt=salt, length=64, n=1 << log2_n, r=r, p=p).derive(_secret(password))
    return out[:32], hashlib.sha256(b"local-diffusion bundle password check" + out[32:]).digest()[:16]


def _aad(header_digest: bytes, index: int, final: bool) -> bytes:
    return header_digest + index.to_bytes(8, "big") + (b"\1" if final else b"\0")


def _nonce(prefix: bytes, index: int) -> bytes:
    if index >= 1 << 32:
        raise CryptoError("the bundle is too large to encrypt")
    return prefix + index.to_bytes(4, "big")


def is_encrypted(path: Path) -> bool:
    with Path(path).open("rb") as stream:
        return stream.read(len(MAGIC)) == MAGIC


def _parse(header: bytes) -> tuple:
    if len(header) < HEADER_SIZE or not header.startswith(MAGIC):
        raise CryptoError("this is not a password-protected Local Diffusion bundle")
    magic, version, log2_n, r, p, salt, chunk, prefix, check = _HEADER.unpack(header[:HEADER_SIZE])
    # Untrusted parameters: bound scrypt's memory (128 * r * n) and the chunk size.
    if version != _VERSION:
        raise CryptoError("this password-protected bundle uses an unsupported version")
    if not (14 <= log2_n <= 18 and 1 <= r <= 8 and 1 <= p <= 4 and 4096 <= chunk <= 16 * 1024 * 1024):
        raise CryptoError("the bundle's encryption parameters are not supported")
    return log2_n, r, p, salt, chunk, prefix, check


def check_password(header: bytes, password: str) -> None:
    """Raise WrongPassword unless `password` opens a bundle with this header."""
    log2_n, r, p, salt, _, _, check = _parse(header)
    if not hmac.compare_digest(_derive(password, log2_n, r, p, salt)[1], check):
        raise WrongPassword("the bundle password is incorrect")


class EncryptingWriter(io.RawIOBase):
    """A forward-only sink: zipfile writes data descriptors because seek fails."""

    def __init__(self, stream, password: str, *, chunk: int = _CHUNK):
        super().__init__()
        salt, prefix = os.urandom(16), os.urandom(8)
        key, check = _derive(password, _LOG2_N, _R, _P, salt)
        self._header = _HEADER.pack(MAGIC, _VERSION, _LOG2_N, _R, _P, salt, chunk, prefix, check)
        self._digest = hashlib.sha256(self._header).digest()
        self._aead = AESGCM(key)
        self._stream = stream
        self._chunk = chunk
        self._prefix = prefix
        self._buffer = bytearray()
        self._index = 0
        self._position = 0
        self._finished = False
        stream.write(self._header)

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._position

    def seek(self, *args):
        raise io.UnsupportedOperation("an encrypted bundle is written forward only")

    def _emit(self, data, final: bool) -> None:
        self._stream.write(self._aead.encrypt(
            _nonce(self._prefix, self._index), bytes(data), _aad(self._digest, self._index, final)))
        self._index += 1

    def write(self, data) -> int:
        if self._finished:
            raise ValueError("write to a finished encrypted bundle")
        data = memoryview(data).cast("B")
        self._buffer += data
        self._position += len(data)
        # Hold back at least one byte so the final chunk is never empty unless
        # the whole bundle is.
        while len(self._buffer) > self._chunk:
            self._emit(memoryview(self._buffer)[:self._chunk], False)
            del self._buffer[:self._chunk]
        return len(data)

    def flush(self) -> None:
        pass

    def finish(self) -> None:
        """Write the final chunk. The caller flushes, fsyncs and closes the file."""
        if not self._finished:
            self._emit(self._buffer, True)
            self._buffer = bytearray()
            self._finished = True

    def close(self) -> None:
        # Closing never finishes implicitly: an exception mid-archive must not
        # produce a validly terminated ciphertext.
        super().close()


class DecryptingReader(io.RawIOBase):
    """Seekable plaintext view of an encrypted bundle; each chunk is authenticated before use."""

    def __init__(self, path: Path, password: str):
        super().__init__()
        self._file = Path(path).open("rb")
        try:
            header = self._file.read(HEADER_SIZE)
            log2_n, r, p, salt, chunk, prefix, check = _parse(header)
            key, expected = _derive(password, log2_n, r, p, salt)
            if not hmac.compare_digest(expected, check):
                raise WrongPassword("the bundle password is incorrect")
            self._aead = AESGCM(key)
            self._digest = hashlib.sha256(header).digest()
            self._chunk, self._prefix = chunk, prefix
            body = self._file.seek(0, io.SEEK_END) - HEADER_SIZE
            stride = chunk + _TAG
            self._count = max(1, -(-body // stride))
            last = body - (self._count - 1) * stride
            if body < _TAG or last < _TAG:
                raise CryptoError("the encrypted bundle is truncated")
            self._cache: tuple[int, bytes] | None = None
            # Authenticating the final chunk first rejects a truncated or
            # extended file before zipfile seeks to its central directory.
            final = self._block(self._count - 1)
            self._size = (self._count - 1) * chunk + len(final)
            self._position = 0
        except BaseException:
            self._file.close()
            raise

    def _block(self, index: int) -> bytes:
        if self._cache and self._cache[0] == index:
            return self._cache[1]
        stride = self._chunk + _TAG
        self._file.seek(HEADER_SIZE + index * stride)
        data = self._file.read(stride)
        try:
            plain = self._aead.decrypt(_nonce(self._prefix, index), data,
                                       _aad(self._digest, index, index == self._count - 1))
        except InvalidTag as exc:
            raise CryptoError("the encrypted bundle is damaged or was changed") from exc
        if index != self._count - 1 and len(plain) != self._chunk:
            raise CryptoError("the encrypted bundle is damaged or was changed")
        self._cache = (index, plain)
        return plain

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}[whence]
        if base + offset < 0:
            raise OSError("negative seek position")
        self._position = base + offset
        return self._position

    def readinto(self, buffer) -> int:
        view = memoryview(buffer).cast("B")
        done = 0
        while done < len(view) and self._position < self._size:
            index, offset = divmod(self._position, self._chunk)
            block = self._block(index)
            take = min(len(view) - done, len(block) - offset)
            view[done:done + take] = block[offset:offset + take]
            done += take
            self._position += take
        return done

    def close(self) -> None:
        if not self.closed:
            self._file.close()
            self._cache = None
        super().close()
