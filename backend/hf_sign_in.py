"""Whether this machine has a Hugging Face sign-in, by its presence alone.

A token counts when `HF_TOKEN` (or the legacy `HUGGING_FACE_HUB_TOKEN`,
which huggingface_hub still reads after it) is set or huggingface_hub's token
file (`constants.HF_TOKEN_PATH`) exists. None is read, logged, sent or checked
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


def login_instructions(*, replace: bool = False) -> str:
    """Cold instructions for the project's installed CLI, never a sign-in.

    A refused saved token needs --force or the CLI skips its login menu.
    The installer is an alternative only for a missing sign-in: it also
    leaves an existing token alone.
    """
    folder = Path(__file__).resolve().parents[1]
    command = "uv run hf auth login" + (" --force" if replace else "")
    instructions = (
        f'run `{command}` from the AuraDream folder `{folder}` '
        '(press Enter for "Log in with your browser"; '
        '"Paste an access token" takes a Read token and keeps pasted input hidden)'
    )
    if not replace:
        instructions += (
            ", or on Windows stop AuraDream, double-click install.cmd there again and answer y "
            "at the Hugging Face step"
        )
    return instructions


def default_token_path() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_TOKEN_PATH)


def token_source(
    token_path: Path | None = None, environ: Mapping[str, str] | None = None,
) -> Literal["env", "file"] | None:
    """Where a token is: "env" for a nonempty HF_TOKEN or HUGGING_FACE_HUB_TOKEN,
    "file" for the token file, None for no sign-in. An unreadable location
    counts as none."""
    environ = os.environ if environ is None else environ
    if _environment_token(environ):
        return "env"
    try:
        path = default_token_path() if token_path is None else token_path
        return "file" if path.is_file() else None
    except (OSError, ImportError, ValueError):
        return None


def _environment_token(environ: Mapping[str, str]) -> bool:
    """huggingface_hub's own environment rule (`utils._auth._get_token_from_environment`
    then `_clean_token`): HF_TOKEN, else HUGGING_FACE_HUB_TOKEN, counting only
    what is left once outer spaces and line breaks go. The value goes nowhere."""
    value = environ.get("HF_TOKEN") or environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    return bool(value.strip())


def token_present(token_path: Path | None = None, environ: Mapping[str, str] | None = None) -> bool:
    return token_source(token_path, environ) is not None
