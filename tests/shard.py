"""`pytest --shard K/N`: CI splits the suite across runners (`ci.yml`'s
Windows legs), each running the K-th of N parts.

A test's part comes from a CRC of its node id, never `hash()` (salted per
process): every runner and every xdist worker computes the same split, and
each test lands in exactly one part. Standard library only.
"""
from __future__ import annotations

import re
import zlib


def parse(text: str) -> tuple[int, int]:
    found = re.fullmatch(r"(\d+)/(\d+)", text.strip())
    if not found or not 1 <= int(found.group(1)) <= int(found.group(2)):
        raise ValueError(f"--shard takes K/N with 1 <= K <= N, not {text!r}")
    return int(found.group(1)), int(found.group(2))


def part(nodeid: str, count: int) -> int:
    """The part (from 1) of `count` that runs `nodeid`."""
    return zlib.crc32(nodeid.encode("utf-8")) % count + 1
