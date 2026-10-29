"""Saved composer drafts, separate from effective generation requests.

Null sampling values retain profile ownership. Nothing in these types can
carry conditioning, execution intent, a gallery identity or expanded work.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import settings
from .seeds import MAX_BATCH, MAX_GRID_SEEDS, MAX_RUNS, MAX_SEED

MAX_PRESETS = 100
MAX_PRESET_NAME = 80


class _StrictPreset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @field_validator("*", mode="after")
    @classmethod
    def _unicode_text(cls, value):
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("saved setup text must be valid Unicode") from exc
        return value


class PresetGrid(_StrictPreset):
    prompt_axis: str = Field(default="", max_length=2000)
    prompt_mode: Literal["cross", "random"] = "cross"
    steps_axis: str = Field(default="", max_length=200)
    cfg_axis: str = Field(default="", max_length=200)
    mu_axis: str = Field(default="", max_length=200)
    size_axis: str = Field(default="", max_length=2000)
    seeds: int = Field(default=1, ge=1, le=MAX_GRID_SEEDS)
    seed_axis: str = Field(default="", max_length=2000)
    budget_minutes: float | None = Field(default=None, gt=0, le=7 * 24 * 60)


class PresetPayload(_StrictPreset):
    mode: Literal["generate", "grid"] = "generate"
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    width: int = Field(default=1024, ge=256, le=settings.refine_max_side)
    height: int = Field(default=1024, ge=256, le=settings.refine_max_side)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0, le=20)
    mu: float | None = Field(default=None, ge=0, le=10)
    true_cfg: float | None = Field(default=None, ge=1, le=20)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    num_images: int = Field(default=1, ge=1, le=MAX_BATCH)
    runs: int = Field(default=1, ge=1, le=MAX_RUNS)
    grid: PresetGrid | None = None

    @model_validator(mode="after")
    def _mode_has_its_grid(self) -> PresetPayload:
        if (self.mode == "grid") != (self.grid is not None):
            raise ValueError("a Grid setup needs its grid definition; a Generate setup has none")
        return self


class PresetFields(_StrictPreset):
    name: str = Field(min_length=1, max_length=MAX_PRESET_NAME)
    model_id: str = Field(min_length=1, max_length=512)
    payload: PresetPayload

    @field_validator("name")
    @classmethod
    def _readable_name(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("give the setup a name without control characters")
        return value

    @field_validator("model_id")
    @classmethod
    def _exact_checkpoint(cls, value: str) -> str:
        if value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("a setup needs an exact checkpoint id without surrounding whitespace or control characters")
        return value


class PresetUpdate(PresetFields):
    revision: int = Field(ge=1)


class PresetCreate(PresetFields):
    """Capture transport only; these intent fields never enter the sidecar."""

    model_revision: int = Field(ge=0)
    step_cache: Literal["fbc-005-v1"] | None = None


class PresetSave(PresetUpdate):
    model_revision: int = Field(ge=0)
    step_cache: Literal["fbc-005-v1"] | None = None


class PresetCheck(_StrictPreset):
    revision: int = Field(ge=1)
    model_revision: int = Field(ge=0)
    step_cache: Literal["fbc-005-v1"] | None = None


class Preset(PresetFields):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    revision: int = Field(ge=1)


class PresetCollection(_StrictPreset):
    items: list[Preset] = Field(max_length=MAX_PRESETS)
    max_items: int = MAX_PRESETS
    max_name: int = MAX_PRESET_NAME
