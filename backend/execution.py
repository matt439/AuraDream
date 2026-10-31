"""Cold execution-recipe contracts; no non-default runtime is enabled yet.

Compilation, attention and step caching share one named extension. The
candidate checks describe the initial experiments, not permission
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
INVALID_LOADER_CONFIGURATION = (
    "The saved or startup performance configuration is invalid; choose and save "
    "valid loading options or restart with --defaults."
)
EXECUTION_OMITTED = object()
EXECUTION_MISMATCH = (
    "This request records different model loading options; select its recorded "
    "compilation and attention options before rendering it."
)


def resolve_execution(
    selected_loader: object, recorded: object = EXECUTION_OMITTED, *,
    step_cache: object = EXECUTION_OMITTED,
) -> ExecutionSpec | None:
    """Resolve identity only; this never grants permission to execute.

    Omitted recorded metadata denotes a new composer request. A supplied value,
    including null/empty legacy Eager, denotes a complete replay. The separate
    cache channel belongs only to composer input, not to portable metadata.
    """
    loader = canonical_loader_execution(selected_loader)
    if recorded is not EXECUTION_OMITTED and step_cache is not EXECUTION_OMITTED:
        raise ValueError("Choose composer step cache or a recorded execution recipe, not both.")
    if recorded is EXECUTION_OMITTED:
        values = loader.model_dump(exclude_none=True) if loader is not None else {}
        if step_cache is not EXECUTION_OMITTED:
            values["step_cache"] = step_cache
        effective = canonical_execution(values)
    else:
        effective = canonical_execution(recorded)
        recorded_loader = canonical_loader_execution({
            "compilation": effective.compilation if effective is not None else None,
            "attention_backend": effective.attention_backend if effective is not None else None,
        })
        if recorded_loader != loader:
            raise ValueError(EXECUTION_MISMATCH)
    if effective is not None and sum(value is not None for value in (
        effective.compilation, effective.attention_backend, effective.step_cache,
    )) > 1:
        raise ValueError("Execution recipe combinations require separate qualification.")
    return effective


def canonical_execution(value: object) -> ExecutionSpec | None:
    """One empty spelling and no silently ignored fields or recipe names."""
    if value is None:
        return None
    spec = ExecutionSpec.model_validate(value)
    return spec if spec.model_dump(exclude_none=True) else None


def execution_key(value: object) -> str | None:
    """Canonical compact JSON for the shared database/key extension, or NULL."""
    spec = canonical_execution(value)
    if spec is None:
        return None
    return json.dumps(spec.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":"))


def execution_from_json(value: str | None) -> ExecutionSpec | None:
    """Validate stored JSON independently of runtime availability; never drop it."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("stored execution must be JSON text or NULL")
    return canonical_execution(json.loads(value))


def execution_chunk(value: object) -> dict[str, dict[str, str]]:
    """Named PNG metadata; absent for every empty spelling, including old rows."""
    spec = canonical_execution(value)
    if spec is None:
        return {}
    recipes = (
        ("compilation", spec.compilation), ("attention_backend", spec.attention_backend),
        ("step_cache", spec.step_cache),
    )
    return {"execution": {name: recipe for name, recipe in recipes if recipe is not None}}


def unavailable_reason(value: object) -> str | None:
    return UNAVAILABLE if canonical_execution(value) is not None else None


def require_default_execution(value: object) -> None:
    reason = unavailable_reason(value)
    if reason is not None:
        raise ValueError(reason)


def canonical_loader_execution(value: object) -> ExecutionSpec | None:
    """Loader intent excludes the composer's per-render step-cache choice."""
    spec = canonical_execution(value)
    if spec is not None and spec.step_cache is not None:
        raise ValueError("Step caching is a per-render choice, not a model loading option.")
    return spec


def loader_execution_from_choices(compilation: str | None, attention: str | None) -> ExecutionSpec | None:
    """Named command/environment controls, with explicit Eager/native defaults."""
    if compilation not in (None, "eager", "regional-v1") or attention not in (None, "native", "cudnn-v1"):
        raise ValueError(INVALID_LOADER_CONFIGURATION)
    return canonical_loader_execution(ExecutionSpec(
        compilation="regional-v1" if compilation == "regional-v1" else None,
        attention_backend="cudnn-v1" if attention == "cudnn-v1" else None,
    ))


def loader_configuration_reason(
    value: object, *, model_id: str, quantization: str, offload: str, device: str,
) -> str | None:
    """Pure configurable first-slice limits, separate from execution permission.

    A structurally valid configuration may be saved without an installed
    compiler or a qualified card. Neither fact grants permission to load it.
    """
    spec = canonical_loader_execution(value)
    if spec is None:
        return None
    if spec.compilation is not None and spec.attention_backend is not None:
        return "Compilation and nonnative attention cannot be combined in the first experiment."
    return _configuration_placement_reason(model_id, quantization, offload, device)


def _configuration_placement_reason(model_id: str, quantization: str, offload: str, device: str) -> str | None:
    if model_id != "black-forest-labs/FLUX.1-Krea-dev":
        return "Experimental recipes require FLUX.1 Krea dev."
    if quantization != "fp8" or offload != "model":
        return "Experimental recipes require fp8 quantization and model offload."
    kind, separator, index = device.partition(":")
    if kind != "cuda" or (separator and not (index.isascii() and index.isdecimal())):
        return "Experimental recipes require a CUDA device."
    return None


def loader_unavailable_reason(value: object) -> str | None:
    """Saved loader intent is never evidence of executable runtime support."""
    return unavailable_reason(canonical_loader_execution(value))


def require_loader_available(value: object) -> None:
    reason = loader_unavailable_reason(value)
    if reason is not None:
        raise ValueError(reason)


def configured_loader_unavailable_reason(configuration: object) -> str | None:
    """One explicit snapshot, including an unusable saved/startup intent.

    Older synthetic/custom settings have no execution fields and mean Eager.
    An invalid intent must never become that default merely because no valid
    specification could be recovered. Diagnostic payloads are never echoed.
    """
    if getattr(configuration, "loader_execution_error", None) is not None:
        return INVALID_LOADER_CONFIGURATION
    try:
        return loader_unavailable_reason(getattr(configuration, "loader_execution", None))
    except (ValueError, TypeError):
        return INVALID_LOADER_CONFIGURATION


def require_configured_loader_available(configuration: object) -> None:
    reason = configured_loader_unavailable_reason(configuration)
    if reason is not None:
        raise ValueError(reason)


def step_cache_configuration_reason(
    value: object, *, loader_execution: object = None, model_id: str,
    quantization: str, offload: str, device: str,
) -> str | None:
    """Pure composer configuration limits; all nonempty cache calls stay closed."""
    spec = canonical_execution(value)
    if spec is None:
        return None
    if spec.compilation is not None or spec.attention_backend is not None:
        return "The per-render step-cache choice cannot replace model loading options."
    if canonical_loader_execution(loader_execution) is not None:
        return "Step caching cannot be combined with compilation or nonnative attention."
    # The initial cache and loader experiments share the same structural
    # model/placement limits. This checks configuration only, never metadata,
    # device availability, an imported attention dispatcher or real cache APIs.
    return _configuration_placement_reason(model_id, quantization, offload, device)


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


def _flux_candidate_reason(
    facts: CompilerMetadata, *, model_id: str, quantization: str, offload: str,
    device: str, num_images: int, lora: str | None, refine: int | None,
    mask: str | None, outpaint: bool, then_refine: bool,
    external_attention_backend: str | None,
) -> str | None:
    """Shared cold limits for the initial attention/cache experiments."""
    if model_id != "black-forest-labs/FLUX.1-Krea-dev":
        return "The first attention/cache candidate is FLUX.1 Krea dev only."
    if quantization != "fp8" or offload != "model":
        return "The first attention/cache candidate needs fp8 and model offload."
    kind, separator, index = device.partition(":")
    if kind != "cuda" or (separator and not (index.isascii() and index.isdecimal())):
        return "The attention/cache candidate needs a CUDA device."
    if num_images != 1:
        return "The first attention/cache candidate needs one image per call."
    if lora is not None or refine is not None or mask is not None or outpaint or then_refine:
        return "The first attention/cache candidate is direct generation without a LoRA or edit pass."
    if external_attention_backend not in (None, "native"):
        return "A nonnative external attention override needs a separate execution policy."
    if facts.platform not in {"win32", "linux"} or facts.python_version != (3, 13):
        return "The attention/cache candidate needs Windows or Linux with Python 3.13."
    if not _minor(facts.torch, "2.13") or not (facts.torch or "").endswith("+cu130"):
        return "The attention/cache candidate needs the torch 2.13 CUDA 13.0 environment."
    if not _minor(facts.diffusers, "0.40") or not _minor(facts.torchao, "0.18"):
        return "The attention/cache candidate needs diffusers 0.40 and torchao 0.18."
    return None


def attention_candidate_reason(
    spec: ExecutionSpec, facts: CompilerMetadata, *, model_id: str, quantization: str,
    offload: str, device: str, num_images: int = 1, lora: str | None = None,
    refine: int | None = None, mask: str | None = None, outpaint: bool = False,
    then_refine: bool = False, external_attention_backend: str | None = None,
) -> str | None:
    """Pure cuDNN matrix; a candidate stays unavailable to render.

    The supplied external value is an immutable startup observation. This
    function neither reads the environment nor chooses override precedence.
    Loaded processor coverage and actual cuDNN shape support need later checks.
    """
    checked = canonical_execution(spec)
    if checked is None or checked.attention_backend != "cudnn-v1":
        return "The attention candidate needs the cudnn-v1 recipe."
    if checked.compilation is not None or checked.step_cache is not None:
        return "Attention combinations with compilation or step caching need separate qualification."
    return _flux_candidate_reason(
        facts, model_id=model_id, quantization=quantization, offload=offload,
        device=device, num_images=num_images, lora=lora, refine=refine, mask=mask,
        outpaint=outpaint, then_refine=then_refine,
        external_attention_backend=external_attention_backend,
    )


def cache_candidate_reason(
    spec: ExecutionSpec, facts: CompilerMetadata, *, model_id: str, quantization: str,
    offload: str, device: str, num_images: int = 1, lora: str | None = None,
    refine: int | None = None, mask: str | None = None, outpaint: bool = False,
    then_refine: bool = False, true_cfg: float | None = None,
    external_attention_backend: str | None = None,
) -> str | None:
    """Pure FBC matrix; no framework/cache import, setup or availability grant.

    True CFG is the separate FLUX unconditioned pass, not distilled guidance.
    This does not inspect a loaded model or assume that absent app settings
    prove native attention.
    """
    checked = canonical_execution(spec)
    if checked is None or checked.step_cache != "fbc-005-v1":
        return "The step-cache candidate needs the fbc-005-v1 recipe."
    if checked.compilation is not None or checked.attention_backend is not None:
        return "Step caching combinations with compilation or nonnative attention need separate qualification."
    if true_cfg is not None and true_cfg != 1.0:
        return "The first step-cache candidate needs ordinary guidance without true CFG."
    return _flux_candidate_reason(
        facts, model_id=model_id, quantization=quantization, offload=offload,
        device=device, num_images=num_images, lora=lora, refine=refine, mask=mask,
        outpaint=outpaint, then_refine=then_refine,
        external_attention_backend=external_attention_backend,
    )
