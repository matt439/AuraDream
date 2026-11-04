"""Download the tokenizers the real-tokenizer tests read, and nothing else.

`tests/test_prompt_encoding.py` (REAL, Kohaku's control tokens) and
`tests/test_critic_tokens.py` (FLUX.2 [klein]'s Qwen3-4B) skip unless the
Hub cache holds these. CI runs this before pytest. Only the tokenizer files
come down (`prompt_encoding._snapshot`'s patterns), a few MB in all, never
weights. A gated repo needs HF_TOKEN from an account that has accepted it.

Each source is tried `ATTEMPTS` times. With HF_TOKEN set, one that still
will not come down fails the step (exit 1): a revoked token or a Hub outage
would otherwise skip those tests on a green run, and CI's cache step saves
only after a passing job, so a partial cache is never kept. Without it (a
fork's or Dependabot's pull request gets no secrets) a missing one is a
warning and its tests skip as before.
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable

from backend import config  # noqa: F401 - before diffusers, as everywhere
from backend import prompt_encoding as pe
from tests.test_prompt_encoding import REAL

ATTEMPTS = 3
RETRY_WAIT = 5.0


def fetch(source: pe.TokenizerSource, *, sleep: Callable[[float], None] = time.sleep) -> Exception | None:
    """None once `source` is cached, else the last attempt's error."""
    error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            pe._snapshot(source, False)
            return None
        except Exception as exc:  # noqa: BLE001 - reported below, the rest still come down
            error = exc
            if attempt < ATTEMPTS - 1:
                sleep(RETRY_WAIT)
    return error


def main(*, sleep: Callable[[float], None] = time.sleep) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    sources: set[pe.TokenizerSource] = set()
    for model_id, _ in REAL:
        encoders = pe.encoders_for(model_id)
        if encoders is None:
            raise ValueError(f"no encoder table for real-tokenizer test checkpoint {model_id}")
        sources.update(spec.source for spec in encoders)
    token = bool(os.environ.get("HF_TOKEN"))
    level = "error" if token else "warning"
    missing = 0
    for source in sorted(sources, key=lambda s: s.label):
        error = fetch(source, sleep=sleep)
        if error is not None:
            missing += 1
            # A GitHub Actions annotation; plain text anywhere else.
            print(f"::{level}::{source.label}: {type(error).__name__}: {error}".splitlines()[0])
            continue
        print(f"{source.label}: cached")
    print(f"{len(sources) - missing} of {len(sources)} tokenizers cached")
    if missing and token:
        print("HF_TOKEN is set, so a tokenizer that will not download fails the run "
              "rather than skipping its tests")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
