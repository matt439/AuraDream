"""wsl/env.sh sourced twice, as `.bashrc` then `serve.sh` source it (DEV#70):
a value still equal to its marker is the file's own default and is worked out
again, so a quantization or 4-bit format changed between the two reaches the
checkpoint; a value the user set is kept. Run by bash (Linux or WSL only)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ENV_SH = Path(__file__).resolve().parent.parent / "wsl" / "env.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(sys.platform != "linux" or BASH is None, reason="bash, as WSL runs env.sh")

NAMES = ("AURADREAM_QUANT", "AURADREAM_WSL_DEFAULT_QUANT", "AURADREAM_MODEL_ID", "AURADREAM_WSL_DEFAULT_MODEL_ID")
LITE = "lite-infer/krea-2-turbo-nunchaku-lite-{}_r32-bnb4-text-encoder"


def _source_twice(tmp_path: Path, between: str) -> dict[str, str | None]:
    """Source env.sh, run `between`, source it again; the four names after."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("AURADREAM_", "UV_", "HF_"))}
    env["HOME"] = str(tmp_path)
    script = (
        f'source "{ENV_SH}"\n{between}\nsource "{ENV_SH}"\n'
        + "".join(f'printf "%s=%s\\n" {name} "${{{name}-<unset>}}"\n' for name in NAMES)
    )
    result = subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True, timeout=30, check=True)
    found = dict(line.split("=", 1) for line in result.stdout.splitlines())
    return {name: None if found[name] == "<unset>" else found[name] for name in NAMES}


def test_nothing_changed_keeps_the_marked_defaults(tmp_path):
    assert _source_twice(tmp_path, ":") == {
        "AURADREAM_QUANT": "nunchaku", "AURADREAM_WSL_DEFAULT_QUANT": "nunchaku",
        "AURADREAM_MODEL_ID": LITE.format("nvfp4"), "AURADREAM_WSL_DEFAULT_MODEL_ID": LITE.format("nvfp4"),
    }


def test_a_changed_quantization_drops_the_marked_lite_checkpoint(tmp_path):
    # The case reproduced: fp8 kept the nvfp4 lite repo, which an fp8 load
    # refuses as pre-quantized.
    assert _source_twice(tmp_path, "export AURADREAM_QUANT=fp8") == {
        "AURADREAM_QUANT": "fp8", "AURADREAM_WSL_DEFAULT_QUANT": None,
        "AURADREAM_MODEL_ID": None, "AURADREAM_WSL_DEFAULT_MODEL_ID": None,
    }


def test_a_changed_format_reaches_the_marked_checkpoint(tmp_path):
    assert _source_twice(tmp_path, "export AURADREAM_NUNCHAKU_FORMAT=int4") == {
        "AURADREAM_QUANT": "nunchaku", "AURADREAM_WSL_DEFAULT_QUANT": "nunchaku",
        "AURADREAM_MODEL_ID": LITE.format("int4"), "AURADREAM_WSL_DEFAULT_MODEL_ID": LITE.format("int4"),
    }


def test_a_checkpoint_the_user_set_is_kept_whatever_the_quantization(tmp_path):
    assert _source_twice(tmp_path, "export AURADREAM_MODEL_ID=someone/own-model AURADREAM_QUANT=fp8") == {
        "AURADREAM_QUANT": "fp8", "AURADREAM_WSL_DEFAULT_QUANT": None,
        "AURADREAM_MODEL_ID": "someone/own-model", "AURADREAM_WSL_DEFAULT_MODEL_ID": None,
    }
