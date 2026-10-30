"""The fixed speed comparison cases every card run shares (`PERF#6`).

    uv run python tools/perf_cases.py                 # each case's command line
    uv run python tools/perf_cases.py --step-cache fbc-005-v1

`PERF#1` (regional compilation), `PERF#2` (fp8 activations), `PERF#3`
(attention) and `PERF#4` (step caching) compare two arms on these cases and
change only their recipe or quantization. The cases are fixed before any
card run, so no prompt or seed is picked after a bad pair is seen; why each
value is what it is lives in docs/plan-step-caching.md, "Fixed comparison
cases". A change here is its own commit, named before the card run.

Pure data and argument lists: this imports nothing heavy, loads nothing and
starts nothing. It prints commands; it never runs one.
"""
from __future__ import annotations

import argparse
import shlex
import sys

CHECKPOINT = "black-forest-labs/FLUX.1-Krea-dev"
QUANTIZATION = "fp8"
OFFLOAD = "model"
# The krea-dev profile's own numbers (inherited from "flux" in
# model_defaults.py), inside Krea's "28-32 steps and CFG 3.5-5". No mu: the
# schedule picks its own. No negative prompt, LoRA, reference or refine.
STEPS = 28
CFG = 3.5
BATCH = 1

# A is the 1 MP case every plan uses; B the second aspect PERF#1 and PERF#3
# add. Both are BFL's FLUX.1 reference sizes and sizes.py's 1 MP presets.
SHAPES: dict[str, tuple[int, int]] = {"A": (1024, 1024), "B": (1360, 768)}

# Verbatim author examples from guides.py, one per category, held to the
# guides by tests/test_perf_cases.py. Krea's four examples have no plain
# scene, so the scene is BFL's FLUX.1 example.
PROMPTS: dict[str, str] = {
    "scene": (
        "A tall sharp-featured man in an oversized charcoal wool coat, standing on a "
        "wet cobblestone street at night, fashion editorial, moody street lighting"
    ),
    "subjects": "Photo of two people having a tea party in a lush garden",
    "text": (
        "Moody night photography, a small store with an orange glowing sign that "
        "says 'Black Forest Labs'"
    ),
}
# Which guide each prompt is quoted from (a `backend.guides` name).
SOURCES: dict[str, str] = {"scene": "FLUX1_GUIDE", "subjects": "KREA_DEV_GUIDE", "text": "KREA_DEV_GUIDE"}

SEEDS: tuple[int, ...] = (4471, 8823)

# An execution recipe's fields (`backend.execution.ExecutionSpec`) and the
# generate.py flag that names each.
RECIPE_FLAGS: dict[str, str] = {
    "compilation": "--compilation", "attention_backend": "--attention", "step_cache": "--step-cache",
}


def generate_args(
    prompt: str, shape: str, *, recipe: dict[str, str] | None = None, quantization: str = QUANTIZATION,
) -> list[str]:
    """The `backend.generate` arguments for one prompt (a `PROMPTS` name) and
    shape, both seeds in one invocation.

    `recipe` names the comparison arm's execution fields; omitted, the arm is
    Eager/Native/Off and adds no flag. `quantization` is for `PERF#2`, whose
    arms differ by quantization instead.
    """
    width, height = SHAPES[shape]
    args = [
        PROMPTS[prompt],
        "--model", CHECKPOINT, "--quant", quantization, "--offload", OFFLOAD,
        "--width", str(width), "--height", str(height),
        "--steps", str(STEPS), "--cfg", f"{CFG:g}", "--batch", str(BATCH),
        "--seed-list", ",".join(str(seed) for seed in SEEDS),
    ]
    for field, value in (recipe or {}).items():
        args += [RECIPE_FLAGS[field], value]
    return args


def cases() -> list[tuple[str, str]]:
    """Every (prompt, shape) pair, prompts in their table's order."""
    return [(prompt, shape) for shape in SHAPES for prompt in PROMPTS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the fixed speed comparison cases' command lines.")
    parser.add_argument("--shape", choices=sorted(SHAPES), help="one shape only (default: both)")
    parser.add_argument("--quant", default=QUANTIZATION, help=f"quantization (default {QUANTIZATION})")
    for field, flag in RECIPE_FLAGS.items():
        parser.add_argument(flag, dest=field, help=f"the arm's {field.replace('_', ' ')} recipe")
    args = parser.parse_args(argv)
    recipe = {field: getattr(args, field) for field in RECIPE_FLAGS if getattr(args, field) is not None}
    for prompt, shape in cases():
        if args.shape in (None, shape):
            command = ["uv", "run", "python", "-m", "backend.generate",
                       *generate_args(prompt, shape, recipe=recipe, quantization=args.quant)]
            print(shlex.join(command))
    return 0


if __name__ == "__main__":
    sys.exit(main())
