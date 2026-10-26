"""Download the tokenizers the real-tokenizer tests read, and nothing else.

`tests/test_prompt_encoding.py` (REAL, Kohaku's control tokens) and
`tests/test_critic_tokens.py` (FLUX.2 [klein]'s Qwen3-4B) skip unless the
Hub cache holds these. CI runs this before pytest. Only the tokenizer files
come down (`prompt_encoding._snapshot`'s patterns), a few MB in all, never
weights. A gated repo needs HF_TOKEN from an account that has accepted it;
one that will not download is a warning, and its tests skip as before.
"""
from __future__ import annotations

import sys

from backend import config  # noqa: F401 - before diffusers, as everywhere
from backend import prompt_encoding as pe
from tests.test_prompt_encoding import REAL


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sources = {spec.source for model_id, _ in REAL for spec in pe.encoders_for(model_id)}
    missing = 0
    for source in sorted(sources, key=lambda s: s.label):
        try:
            pe._snapshot(source, False)
        except Exception as exc:  # noqa: BLE001 - its tests skip; the rest still come down
            missing += 1
            # A GitHub Actions annotation; plain text anywhere else.
            print(f"::warning::{source.label}: {type(exc).__name__}: {exc}".splitlines()[0])
            continue
        print(f"{source.label}: cached")
    print(f"{len(sources) - missing} of {len(sources)} tokenizers cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
