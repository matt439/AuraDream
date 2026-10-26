"""Startup-backup configuration/status; no credential fields."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from .backup_policy import MAX_KEEP_LAST
from .backup_settings import MAX_AGE_HOURS


class BackupSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str = Field(min_length=1, max_length=64)
    enabled: StrictBool
    minimum_age_hours: StrictInt | None = Field(default=None, ge=1, le=MAX_AGE_HOURS)
    keep_last: StrictInt | None = Field(default=None, ge=1, le=MAX_KEEP_LAST)
    allow_unencrypted: StrictBool = False
    destination: str | None = Field(default=None, max_length=32768)


class BackupStatus(BaseModel):
    revision: str | None
    enabled: bool
    minimum_age_hours: int | None
    keep_last: int | None
    protection: str | None
    destination: str | None
    warning: str | None
    state: str
    reason: str
    archive_name: str | None = None
    verified_at: str | None = None
    retention_enabled: bool = False
    requires_confirmation: bool = False
