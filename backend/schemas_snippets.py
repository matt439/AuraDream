"""Personal prompt snippets, independent of checkpoint and gallery records."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_SNIPPETS = 100
MAX_SNIPPET_NAME = 80
MAX_SNIPPET_TEXT = 16_000


class SnippetFields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=MAX_SNIPPET_NAME)
    text: str = Field(min_length=1, max_length=MAX_SNIPPET_TEXT)
    target: Literal["prompt", "negative"] = "prompt"

    @field_validator("name")
    @classmethod
    def readable_name(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("give the snippet a name without control characters")
        return value

    @field_validator("name", "text")
    @classmethod
    def readable_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a snippet needs some text")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("snippet text must be valid Unicode") from exc
        return value


class SnippetUpdate(SnippetFields):
    revision: int = Field(ge=1)


class Snippet(SnippetFields):
    # Persisted items must name their destination; only a new API request
    # may use the convenient Prompt default.
    # Pydantic makes this required by removing the inherited default; Pyright's
    # dataclass override rule cannot represent that supported model contract.
    target: Literal["prompt", "negative"]  # pyright: ignore[reportGeneralTypeIssues]
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    revision: int = Field(ge=1)


class SnippetCollection(BaseModel):
    items: list[Snippet]
    max_items: int = MAX_SNIPPETS
    max_name: int = MAX_SNIPPET_NAME
    max_text: int = MAX_SNIPPET_TEXT
