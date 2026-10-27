"""Startup-backup configuration/status and opaque credential metadata; no secrets."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt
from typing import Literal

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
    retention_enabled: StrictBool = False
    allow_managed_deletion: StrictBool = False
    expected_mode_revision: str | None = Field(default=None, max_length=64)
    protection: Literal["unencrypted", "encrypted"] | None = None
    allow_encrypted: StrictBool = False
    confirm_recovery: StrictBool = False


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
    mode_revision: str | None = None
    shared_mode_id: str | None = None
    retention_state: str = "disabled"
    retention_reason: str = "not_enabled"
    retention_warning: str | None = None
    retention_marked: int = 0
    retention_retained: int = 0
    retention_pending: bool = False
    retention_uncertain: bool = False
    credential_configured: bool = False
    credential_state: str = "not_configured"
    credential_provider: str | None = None
    credential_generation: str | None = None
