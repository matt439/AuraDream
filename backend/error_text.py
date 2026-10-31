"""An exception as one UI-presentable paragraph.

Standard library only, so the prompt analyser's leaves (`prompt_check`,
`tag_lookup`) need not import the job queue for it; `jobs` re-exports it.
"""
from __future__ import annotations


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
