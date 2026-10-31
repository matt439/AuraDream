"""Gallery bundles' request bodies (`api_bundles.py`, which re-exports them)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from .config import settings


class ExportRequest(BaseModel):
    """A bundle to write: of the whole gallery, cleared once it is verified
    (`expected` the typed record count), or a backup that keeps it.

    `keep_gallery` skips only the clear. `scope` is what the bundle holds:
    every row, the picked `ids`, or the live starred images of every
    checkpoint; a picked or starred bundle is always a backup.
    """

    expected: int | None = Field(default=None, ge=0)
    password: SecretStr | None = None
    keep_gallery: bool = False
    scope: Literal["all", "picked", "starred"] = "all"
    ids: list[int] | None = Field(default=None, min_length=1, max_length=settings.max_grid_images)


class ImportRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    password: SecretStr | None = None


class PasswordCheck(BaseModel):
    header: str = Field(max_length=1024)  # base64 of the bundle's first bytes
    password: SecretStr
