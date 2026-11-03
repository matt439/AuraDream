"""Whether this machine has a Hugging Face sign-in, by its presence alone.

A token counts when `HF_TOKEN` is set or huggingface_hub's token file
(`constants.HF_TOKEN_PATH`) exists. Neither is read, logged, sent or checked
with the Hub here: a present token may still be refused by a gated repo,
which `hub_access` and the load say in their own words. Stdlib and
huggingface_hub's constants only, so the doctor can use it without diffusers
and the status poll can ask it as often as it runs.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal


def default_token_path() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_TOKEN_PATH)


def token_source(
    token_path: Path | None = None, environ: Mapping[str, str] | None = None,
) -> Literal["env", "file"] | None:
    """Where a token is: "env" for a nonempty HF_TOKEN, "file" for the token
    file, None for no sign-in. An unreadable location counts as none."""
    environ = os.environ if environ is None else environ
    if environ.get("HF_TOKEN"):
        return "env"
    try:
        path = default_token_path() if token_path is None else token_path
        return "file" if path.is_file() else None
    except (OSError, ImportError, ValueError):
        return None


def token_present(token_path: Path | None = None, environ: Mapping[str, str] | None = None) -> bool:
    return token_source(token_path, environ) is not None
