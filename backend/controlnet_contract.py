"""Frozen ControlNet inputs and the two published SDXL weight pins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .instruction_edit import InstructionEditSource, PositiveSource

ControlMode = Literal["pose", "depth"]
RECIPE = "sdxl-controlnet-v1"
SIZE = (1024, 1024)
STEPS = 28
CFG = 5.0
DEFAULT_STRENGTH = 0.8
UNRESOLVED = "The recorded ControlNet guide has not been resolved to a gallery image."


@dataclass(frozen=True)
class ControlWeights:
    repo_id: str
    revision: str
    filename: str
    variant: str | None
    size: int
    sha256: str
    config_sha256: str


WEIGHTS: dict[ControlMode, ControlWeights] = {
    "pose": ControlWeights(
        "xinsir/controlnet-openpose-sdxl-1.0", "23f966cd5cfdd3f7729c903e243d87152162d2b7",
        "diffusion_pytorch_model.safetensors", None, 2_502_139_104,
        "b8524e557a7df60d081f5d4a0eb109967d107df217943bf88c2d99b9ebcc06c5",
        "b5b8f9a7619c9452f19407ef73a01ce3f061fcf932de43ddbaca336d12d02801",
    ),
    "depth": ControlWeights(
        "diffusers/controlnet-depth-sdxl-1.0-small", "daf3835d036574dff7c158882e8e77b75b024ee5",
        "diffusion_pytorch_model.fp16.safetensors", "fp16", 320_237_179,
        "6f23c58fd632f52238a7b35ebdc02f9f596fd13dbaa121f9b37b9f4689c2b1e9",
        "7d11cc8c8dffe63e0a2cef41e161080fa93e769849ec18d670149ea8089147eb",
    ),
}

ControlStrength = Annotated[float, Field(strict=True, ge=0.1, le=1.0, allow_inf_nan=False)]


class _ControlContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, revalidate_instances="always")

    @model_validator(mode="before")
    @classmethod
    def _check_copied_fields(cls, value: object) -> object:
        if isinstance(value, cls):
            return {**vars(value), **(value.__pydantic_extra__ or {})}
        return value


class ControlNetSpec(_ControlContract):
    """The original RGB proof remains intact when a gallery ID can't be remapped."""

    kind: Literal["controlnet"] = "controlnet"
    recipe: Literal["sdxl-controlnet-v1"] = RECIPE
    mode: ControlMode
    source: PositiveSource | None
    guide: InstructionEditSource
    strength: ControlStrength
    weight_revision: str

    @model_validator(mode="after")
    def _check_recipe(self) -> ControlNetSpec:
        if (self.guide.width, self.guide.height) != SIZE:
            raise ValueError("ControlNet needs a 1024 × 1024 guide image.")
        if self.weight_revision != WEIGHTS[self.mode].revision:
            raise ValueError("The saved ControlNet weights do not match the pinned recipe.")
        return self

    def key(self) -> tuple[str, str, ControlMode, str, float, str, int, int]:
        return (self.kind, self.recipe, self.mode, self.weight_revision, self.strength,
                self.guide.hash, self.guide.width, self.guide.height)


class ControlNetPreview(_ControlContract):
    source: PositiveSource
    mode: ControlMode
    strength: ControlStrength = DEFAULT_STRENGTH


def weight_info() -> dict[str, dict[str, str | int]]:
    return {
        mode: {"repo_id": weights.repo_id, "revision": weights.revision, "size": weights.size}
        for mode, weights in WEIGHTS.items()
    }
