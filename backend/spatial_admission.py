"""Shared spatial-conditioning admission before queueing and generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import card_facts
from .controlnet_contract import ControlNetSpec
from .spatial_conditioning import validate_request

if TYPE_CHECKING:
    from .backends.base import GenerationBackend
    from .config import Settings
    from .schemas_generate import GenerateRequest


def require_available(
    backend: GenerationBackend, selected: Settings, request: GenerateRequest, *, check_hardware: bool = True,
) -> None:
    conditioning = validate_request(request)
    if conditioning is None:
        return
    if isinstance(conditioning, ControlNetSpec):
        from .controlnet_admission import bind_source, loader_refusal

        reason = loader_refusal(backend, selected, mode=conditioning.mode, check_hardware=False)
        if reason is None and check_hardware:
            card_facts.card(selected.device)
            reason = loader_refusal(backend, selected, mode=conditioning.mode)
        if reason is None:
            bind_source(conditioning)
    else:
        reason = backend.regional_prompts_refusal(selected, check_hardware=False)
        if reason is None and check_hardware:
            card_facts.card(selected.device)
            reason = backend.regional_prompts_refusal(selected)
    if reason is not None:
        raise ValueError(reason)
