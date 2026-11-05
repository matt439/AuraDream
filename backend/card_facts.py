"""The server's CUDA card, as two facts: its total memory and its compute
capability, or None when there is no card to read.

Cold: torch is imported only on the first ask, and only `torch.cuda`'s
device queries run (the status line's `describe()` already opens the same
context); diffusers is never imported. Each device's answer is read once
and kept. Unknown - a CPU device, a CPU build of torch, no driver, no card,
any error - is None, and every caller treats None as it treated every card
before it asked: the 5080's defaults. The tests and the fake server have no
card; `tests/conftest.py` and `tools/fake_server.py` answer None here, and a
test that wants one patches `card`.
"""
from __future__ import annotations

import functools
from typing import Any, NamedTuple


class Card(NamedTuple):
    total_memory: int                # bytes, torch's `total_memory`
    capability: tuple[int, int]      # (major, minor)


def probe(device: str, torch: Any = None) -> Card | None:
    """Read `device`'s card through torch, uncached; None when unknown."""
    try:
        if torch is None:
            import torch  # noqa: PLC0415 - heavy, and only a card's facts need it
        target = torch.device(device)
        if target.type != "cuda":
            return None
        if not getattr(getattr(torch, "version", None), "cuda", None) or not torch.cuda.is_available():
            return None
        index = 0 if target.index is None else target.index
        total = int(torch.cuda.get_device_properties(index).total_memory)
        major, minor = torch.cuda.get_device_capability(index)
        return Card(total, (int(major), int(minor)))
    except Exception:  # noqa: BLE001 - an unreadable card is an unknown one
        return None


@functools.cache
def card(device: str = "cuda") -> Card | None:
    """`device`'s card, read once per device; None when unknown."""
    return probe(device)


def total_memory(device: str = "cuda") -> int | None:
    found = card(device)
    return None if found is None else found.total_memory


def capability(device: str = "cuda") -> tuple[int, int] | None:
    found = card(device)
    return None if found is None else found.capability
