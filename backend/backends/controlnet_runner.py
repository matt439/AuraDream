"""Use one validated ControlNet alongside the loaded, offloaded SDXL owner."""
from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import controlnet_admission, refine
from ..config import settings
from ..controlnet_contract import WEIGHTS, ControlNetSpec, ControlWeights
from ..doctor import windows_memory_status
from ..model_selection import current_settings
from . import loading
from .base import Cancelled

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..schemas_generate import GenerateRequest
    from .base import CancelFn
    from .diffusers_backend import DiffusersBackend

log = logging.getLogger(__name__)
GiB = 1024**3


def _verified_snapshot(weights: ControlWeights) -> Path:
    from huggingface_hub import hf_hub_download

    config = Path(hf_hub_download(weights.repo_id, "config.json", revision=weights.revision))
    with config.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != weights.config_sha256:
            raise ValueError("The cached ControlNet config does not match the pinned revision.")
    path = Path(hf_hub_download(weights.repo_id, weights.filename, revision=weights.revision))
    if path.parent != config.parent or path.stat().st_size != weights.size:
        raise ValueError("The cached ControlNet weights have an unexpected location or size.")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != weights.sha256:
            raise ValueError("The cached ControlNet weights do not match the pinned revision.")
    return config.parent


def _require_headroom(torch: Any) -> None:
    # Cached allocator blocks from a preceding cell are not occupied by
    # another process. Release those before reading whole-card free memory.
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(0)
    if free < 10 * GiB:
        raise ValueError("ControlNet needs at least 10 GiB of free GPU memory before running.")
    ram = windows_memory_status()
    if ram is None or ram[1] < 12 * GiB:
        raise ValueError("ControlNet needs at least 12 GiB of available RAM with Animagine loaded.")
    # Keep the trial's 4 GiB live floor beside a conservative bound on the
    # observed increment in the owned process tree while rendering pose.
    if ram[2] < 13 * GiB:
        raise ValueError("ControlNet needs at least 13 GiB of available Windows commit with Animagine loaded.")


def _cleanup(owner: Any, controlnet: Any, hook: Any, original: BaseException | None) -> None:
    # The guide model is disposable. Offload would copy the entire ControlNet
    # back to host memory just as the SDXL owner needs its own storage back.
    # Clear the auxiliary tensors first even if a traceback still holds the
    # short-lived sibling. Shared SDXL modules are never put on meta.
    actions = (hook.remove, lambda: controlnet.to("meta"), owner.maybe_free_model_hooks)
    errors: list[Exception] = []
    for action in actions:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - independent hooks still need cleanup
            errors.append(exc)
    if errors:
        if original is None:
            raise ExceptionGroup("ControlNet offload cleanup failed", errors)
        for error in errors:
            original.add_note(f"ControlNet offload cleanup also failed: {error}")


@contextmanager
def prepared(
    backend: DiffusersBackend, pipe: Any, request: GenerateRequest, should_cancel: CancelFn | None,
) -> Iterator[tuple[Any, Image]]:
    """Borrow SDXL components without moving them or sharing a ControlNet hook."""
    from ..spatial_conditioning import validate_request

    spec = ControlNetSpec.model_validate(request.spatial_conditioning)
    selected = current_settings(settings)
    controlnet_admission.require_loader(backend, selected, spec.mode)
    validate_request(request)
    if backend._tiled_refine_owner_refusal(pipe) is not None:
        raise ValueError("Reload the pinned Animagine XL 4.0 checkpoint before using ControlNet.")
    if should_cancel is not None and should_cancel():
        raise Cancelled

    import torch
    from accelerate import cpu_offload_with_hook
    from diffusers.models.controlnets.controlnet import ControlNetModel
    from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline

    weights = WEIGHTS[spec.mode]
    with controlnet_admission.open_guide(spec) as guide:
        _require_headroom(torch)
        snapshot = _verified_snapshot(weights)
        if should_cancel is not None and should_cancel():
            raise Cancelled
        _require_headroom(torch)
        with loading.weights_loading():
            controlnet = ControlNetModel.from_pretrained(
                snapshot, variant=weights.variant, dtype=torch.bfloat16, use_safetensors=True,
                local_files_only=True, low_cpu_mem_usage=True,
            )
        runner = refine.build(pipe, StableDiffusionXLControlNetPipeline, controlnet=controlnet)
        runner.set_progress_bar_config(**getattr(pipe, "_progress_bar_config", {}))
        supported = set(inspect.signature(runner.__call__).parameters)
        if not {"image", "controlnet_conditioning_scale", "generator", "callback_on_step_end"} <= supported:
            raise ValueError("The installed ControlNet pipeline lacks required guide or reproducibility inputs.")
        _, hook = cpu_offload_with_hook(controlnet, execution_device=selected.device)
        log.info("ControlNet %s at %s", spec.mode, weights.revision)
        try:
            if should_cancel is not None and should_cancel():
                raise Cancelled
            yield runner, guide
        except BaseException as exc:
            _cleanup(pipe, controlnet, hook, exc)
            raise
        else:
            _cleanup(pipe, controlnet, hook, None)
