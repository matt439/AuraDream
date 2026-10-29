"""Portable raw generation setups and explicit file-index merge choices."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schemas_presets import MAX_PRESET_NAME, MAX_PRESETS, Preset, PresetFields

PORTABLE_FORMAT = "auradream.generation-presets"
PORTABLE_FILENAME = "auradream-generation-setups.json"
MAX_PORTABLE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 16


class PortablePreset(PresetFields):
    """Only name, exact model id and typed raw payload; no capture/local fields."""


class PortablePresets(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    format: Literal["auradream.generation-presets"]
    version: int
    items: list[PortablePreset] = Field(max_length=MAX_PRESETS)

    @field_validator("version", mode="before")
    @classmethod
    def supported_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported portable setup version")
        return value


class ImportDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int = Field(ge=0, lt=MAX_PRESETS)
    action: Literal["add", "copy", "replace", "skip"]
    name: str | None = Field(default=None, min_length=1, max_length=MAX_PRESET_NAME)
    item_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def explicit_replacement(self) -> ImportDecision:
        if self.action == "replace":
            if self.item_id is None or self.revision is None:
                raise ValueError("replacement needs a local id and revision")
        elif self.item_id is not None or self.revision is not None:
            raise ValueError("only replacement accepts a local id and revision")
        if self.name is not None and self.action != "copy":
            raise ValueError("only Add as copy accepts an edited name")
        return self


class PresetImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    payload: PortablePresets
    decisions: list[ImportDecision] = Field(default_factory=list, max_length=MAX_PRESETS)


class PresetImportCommit(PresetImportRequest):
    collection_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    payload_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class MergeRowResponse(BaseModel):
    index: int
    action: Literal["add", "copy", "replace", "skip"]
    preset: PortablePreset
    matches: list[Preset]
    exact_ids: list[str]
    incoming_collisions: list[int]
    item_id: str | None = None
    revision: int | None = None


class PresetImportPreview(BaseModel):
    collection_revision: str
    payload_digest: str
    rows: list[MergeRowResponse]
    add_count: int
    replace_count: int
    skip_count: int
    changed: bool
    can_import: bool
    reason: str
