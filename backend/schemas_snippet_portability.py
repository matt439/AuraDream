"""Strict portable snippet documents and explicit file-index merge choices."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schemas_snippets import MAX_SNIPPET_NAME, MAX_SNIPPETS, SnippetFields

PORTABLE_FORMAT = "auradream.prompt-snippets"
MAX_PORTABLE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 16


class PortableSnippet(SnippetFields):
    # Persisted/portable data must name its destination, unlike a new CRUD draft.
    target: Literal["prompt", "negative"]  # pyright: ignore[reportGeneralTypeIssues]


class PortableSnippets(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    format: Literal["auradream.prompt-snippets"]
    version: int
    items: list[PortableSnippet] = Field(max_length=MAX_SNIPPETS)

    @field_validator("version", mode="before")
    @classmethod
    def supported_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported portable snippet version")
        return value


class ImportDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int = Field(ge=0, lt=MAX_SNIPPETS)
    action: Literal["add", "copy", "replace", "skip"]
    name: str | None = Field(default=None, min_length=1, max_length=MAX_SNIPPET_NAME)
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
