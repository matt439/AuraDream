"""Cold execution-recipe contracts; no non-default runtime is enabled yet.

Compilation, attention and step caching share one named extension. The
candidate check describes the first compilation experiment, not permission
to render: every nonempty recipe is refused until its carry/runtime exists.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ExecutionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    compilation: Literal["regional-v1"] | None = None
    attention_backend: Literal["cudnn-v1"] | None = None
    step_cache: Literal["fbc-005-v1"] | None = None


UNAVAILABLE = (
    "Execution recipes are not available in this build; "
    "use Eager, native attention and no step cache."
)


def canonical_execution(value: object) -> ExecutionSpec | None:
    """One empty spelling and no silently ignored fields or recipe names."""
    if value is None:
        return None
    spec = ExecutionSpec.model_validate(value)
    return spec if spec.model_dump(exclude_none=True) else None


def execution_key(value: object) -> str | None:
    """Canonical named extension for later carry; no CellKey slot is used yet."""
    spec = canonical_execution(value)
    if spec is None:
        return None
    return json.dumps(spec.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":"))


def unavailable_reason(value: object) -> str | None:
    return UNAVAILABLE if canonical_execution(value) is not None else None


def require_default_execution(value: object) -> None:
    reason = unavailable_reason(value)
    if reason is not None:
        raise ValueError(reason)


@dataclass(frozen=True)
class CompilerMetadata:
    """Installed distribution facts, never a compiler/device qualification."""

    platform: str
    python_version: tuple[int, int]
    torch: str | None
    diffusers: str | None
    torchao: str | None
    triton: str | None
    triton_windows: str | None


def compiler_metadata(
    *, lookup: Callable[[str], str] = version, platform: str | None = None,
    python_version: tuple[int, int] | None = None,
) -> CompilerMetadata:
    """Metadata reads only: no torch/diffusers import, download or CUDA probe."""
    def installed(name: str) -> str | None:
        try:
            return lookup(name)
        except PackageNotFoundError:
            return None

    return CompilerMetadata(
        platform=sys.platform if platform is None else platform,
        python_version=(sys.version_info.major, sys.version_info.minor) if python_version is None else python_version,
        torch=installed("torch"), diffusers=installed("diffusers"), torchao=installed("torchao"),
        triton=installed("triton"), triton_windows=installed("triton-windows"),
    )


def _minor(value: str | None, expected: str) -> bool:
    return value is not None and value.split("+", 1)[0].split(".")[:2] == expected.split(".")


def regional_candidate_reason(
    spec: ExecutionSpec, facts: CompilerMetadata, *, model_id: str, quantization: str,
    offload: str, device: str, num_images: int = 1, lora: str | None = None,
    refine: int | None = None, mask: str | None = None, outpaint: bool = False,
) -> str | None:
    """Pure first-experiment matrix. None means candidate, never available.

    Even a matching target is refused by require_default_execution today.
    Actual CUDA/toolchain, compilation, pixels and memory remain unmeasured.
    """
    if spec.compilation != "regional-v1":
        return "The regional compilation candidate needs the regional-v1 recipe."
    if spec.attention_backend is not None or spec.step_cache is not None:
        return "Regional compilation combinations with attention or step caching need separate qualification."
    if model_id != "black-forest-labs/FLUX.1-Krea-dev":
        return "The first regional compilation candidate is FLUX.1 Krea dev only."
    if quantization != "fp8" or offload != "model":
        return "The first regional compilation candidate needs fp8 and model offload."
    kind, separator, index = device.partition(":")
    if kind != "cuda" or (separator and not (index.isascii() and index.isdecimal())):
        return "The regional compilation candidate needs a CUDA device."
    if num_images != 1:
        return "The first regional compilation candidate needs one image per call."
    if lora is not None or refine is not None or mask is not None or outpaint:
        return "The first regional compilation candidate is direct generation without a LoRA or edit pass."
    if facts.platform not in {"win32", "linux"} or facts.python_version != (3, 13):
        return "The regional compilation candidate needs Windows or Linux with Python 3.13."
    if not _minor(facts.torch, "2.13") or not (facts.torch or "").endswith("+cu130"):
        return "The regional compilation candidate needs the torch 2.13 CUDA 13.0 environment."
    if not _minor(facts.diffusers, "0.40") or not _minor(facts.torchao, "0.18"):
        return "The regional compilation candidate needs diffusers 0.40 and torchao 0.18."
    if facts.platform == "win32" and facts.triton_windows != "3.7.1.post27":
        return "The Windows regional compilation candidate needs triton-windows 3.7.1.post27."
    if facts.platform == "linux" and not _minor(facts.triton, "3.7"):
        return "The Linux regional compilation candidate needs Triton 3.7."
    return None
