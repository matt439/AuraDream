"""Request/response models for the HTTP API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import settings

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    width: int = Field(default=1024, ge=256, le=settings.max_side)
    height: int = Field(default=1024, ge=256, le=settings.max_side)
    steps: int = Field(default=8, ge=1, le=100)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    # Flow-matching timestep shift. None lets the pipeline pick its own.
    mu: float | None = Field(default=1.15, ge=0.0, le=10.0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    num_images: int = Field(default=1, ge=1, le=8)
    # Sequential passes. Each one is a separate pipeline call whose base seed is
    # the previous run's plus `num_images`, so a long sweep costs no more VRAM
    # than a single image - which is what makes 2K sweeps possible at all.
    runs: int = Field(default=1, ge=1, le=64)

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_16(cls, v: int) -> int:
        # The VAE downsamples by 16; off-grid sizes get silently rounded anyway.
        if v % 16:
            raise ValueError("width/height must be a multiple of 16")
        return v

    def check_budget(self) -> None:
        if self.width * self.height > settings.max_pixels:
            raise ValueError(
                f"{self.width}x{self.height} exceeds the {settings.max_pixels:,} pixel budget"
            )


class ImageRecord(BaseModel):
    id: int
    filename: str
    url: str
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    # Persisted since the first release but never served, which left the UI
    # unable to restore it - and on Turbo a different mu is a different image.
    mu: float | None = None
    seed: int
    model_id: str
    duration_ms: int
    created_at: str
    favorite: bool = False


class GalleryPage(BaseModel):
    items: list[ImageRecord]
    total: int
    limit: int
    offset: int


class ModelStatus(BaseModel):
    state: Literal["unloaded", "loading", "ready", "error"]
    defaults: dict[str, float | int | None] = {}
    model_id: str
    quantization: str
    offload: str
    detail: str | None = None
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None
    device_name: str | None = None


class JobAccepted(BaseModel):
    job_id: str
