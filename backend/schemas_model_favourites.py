"""Model favourites: canonical checkpoint ids, never loader state or images."""
from __future__ import annotations

import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MODEL_FAVOURITES = 200
MAX_MODEL_ID = 512


def checked_model_id(value: object) -> str:
    """A favourite's id as the page sends it: trimmed, bounded, readable."""
    if not isinstance(value, str):
        raise ValueError("a model id is text")
    value = value.strip()
    if not 1 <= len(value) <= MAX_MODEL_ID:
        raise ValueError("a model id is 1 to 512 characters")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError("a model id has no control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("a model id must be valid Unicode") from None
    return value


class ModelFavouriteChange(BaseModel):
    """Star or unstar one checkpoint. A set operation: it never overwrites
    another tab's change to a different id, so it needs no expected revision."""
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str
    favourite: bool

    @field_validator("model_id", mode="before")
    @classmethod
    def valid_id(cls, value):
        return checked_model_id(value)


class ModelFavouriteImport(BaseModel):
    """A browser's earlier per-origin set, merged in once at the user's press."""
    model_config = ConfigDict(extra="forbid", strict=True)

    model_ids: list[str] = Field(max_length=MAX_MODEL_FAVOURITES)

    @field_validator("model_ids", mode="before")
    @classmethod
    def valid_ids(cls, value):
        if not isinstance(value, list):
            raise ValueError("model_ids is a list")
        return [checked_model_id(item) for item in value]


class ModelFavourites(BaseModel):
    items: list[str]
    revision: int = Field(ge=0)
    max_items: int = MAX_MODEL_FAVOURITES
    # Only an import's answer: how many of the sent ids were new here.
    imported: int | None = None
