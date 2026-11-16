"""The server's CUDA card: memory, compute capability and an observed name.

Cold: torch is imported only on the first ask, and only `torch.cuda`'s
device queries run (the status line's `describe()` already opens the same
context); diffusers is never imported. Each device's answer is read once
and kept. Unknown - a CPU device, a CPU build of torch, no driver, no card,
any error - is None. Existing sizing/default readers keep their old unknown
behavior; the measured Anima Refine gate holds an unknown GPU. Its `observed`
reader imports/probes nothing, and an explicit successful load refreshes facts.
The tests and the fake server have no
card; `tests/conftest.py` and `tools/fake_server.py` answer None here (the
fake until `/api/fake/card` claims one), and a test that wants one patches
`card`.
"""
from __future__ import annotations

import functools
from typing import Any, NamedTuple

_OBSERVED: dict[str, "Card | None"] = {}


class Card(NamedTuple):
    total_memory: int                # bytes, torch's `total_memory`
    capability: tuple[int, int]      # (major, minor)
    name: str | None = None          # None until a probe actually supplied it


def probe(device: str, torch: Any = None) -> Card | None:
    """Read `device`'s card through torch, uncached; None when unknown."""
    _OBSERVED[device] = None
    try:
        if torch is None:
            import torch  # noqa: PLC0415 - heavy, and only a card's facts need it
        target = torch.device(device)
        if target.type != "cuda":
            return None
        if not getattr(getattr(torch, "version", None), "cuda", None) or not torch.cuda.is_available():
            return None
        index = 0 if target.index is None else target.index
        properties = torch.cuda.get_device_properties(index)
        total = int(properties.total_memory)
        major, minor = torch.cuda.get_device_capability(index)
        name = getattr(properties, "name", None)
        found = Card(total, (int(major), int(minor)), name if isinstance(name, str) else None)
        _OBSERVED[device] = found
        return found
    except Exception:  # noqa: BLE001 - an unreadable card is an unknown one
        return None


def observed(device: str = "cuda") -> Card | None:
    """The last supplied probe result; this reader imports/probes nothing.

    Qualification must hold an unknown card rather than assume the 5080. The
    existing status/load probe owns discovery; this lookup adds no GPU work.
    """
    return _OBSERVED.get(device)


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
