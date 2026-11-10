"""An exception as one UI-presentable paragraph.

Standard library only, so the prompt analyser's leaves (`prompt_check`,
`tag_lookup`) need not import the job queue for it; `jobs` re-exports it.
"""
from __future__ import annotations

import errno


def generation_error(exc: BaseException, *, loading: bool = False) -> str:
    """Give known memory failures a next step, keeping library imports cold.

    PyTorch's exception can sit inside a loader's wrapper. Match its type,
    not arbitrary text mentioning memory; unrelated failures keep their words.
    The failure site still logs the original exception and its traceback.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    ram = False
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(
            base.__name__ == "OutOfMemoryError" and base.__module__ in {"torch", "torch.cuda"}
            for base in type(current).__mro__
        ):
            if loading:
                return (
                    "The graphics card ran out of memory while loading the checkpoint. "
                    "In Models, try Sequential offload or choose a lighter checkpoint, then load again."
                )
            return (
                "The graphics card ran out of memory while making this image. "
                "Try a smaller image or fewer images per batch. In Models, you can also try "
                "Sequential offload or choose a lighter checkpoint."
            )
        if loading and (
            isinstance(current, MemoryError)
            or isinstance(current, OSError) and (
                current.errno == errno.ENOMEM or getattr(current, "winerror", None) == 1455
            )
        ):
            ram = True
        current = current.__cause__ or current.__context__
    if ram:
        return (
            "The computer ran out of RAM or paging-file space while loading the checkpoint. "
            "Close other apps or choose a lighter checkpoint in Models, then load again. "
            "On Windows, check that the paging file can grow and its drive has free space."
        )
    return short_error(exc)


def short_error(exc: BaseException, limit: int = 320) -> str:
    """A one-paragraph, UI-presentable form of an exception.

    HuggingFace errors in particular carry a request id and a stack of URLs that
    are noise in the browser; keep the actionable sentences and drop the rest.
    """
    lines = [
        line.strip()
        for line in str(exc).splitlines()
        if line.strip() and "Request ID" not in line
    ]
    message = " ".join(lines) or exc.__class__.__name__
    if len(message) > limit:
        # Cut from the middle: PyTorch's OOM message puts the diagnosis first
        # and the remedy (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
        # last, and a cut from the end landed inside that variable's name.
        head = limit * 2 // 3
        tail = limit - head - 3
        message = message[:head].rstrip() + " … " + message[-tail:].lstrip()
    return f"{type(exc).__name__}: {message}"
