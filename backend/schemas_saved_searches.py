"""Named gallery filters; definitions never retain image ids or model state."""
from __future__ import annotations

import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import search

MAX_SAVED_SEARCHES = 50
MAX_SEARCH_NAME = 80
MAX_SEARCH_QUERY = 4_000


def name_key(name: str) -> str:
    """The unique-name comparison, without rewriting the displayed name."""
    return unicodedata.normalize("NFC", name).casefold()


class SavedSearchFields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=MAX_SEARCH_NAME)
    query: str = Field(max_length=MAX_SEARCH_QUERY)
    favorites: bool
    all_models: bool

    @field_validator("name", mode="before")
    @classmethod
    def trim_name(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("name")
    @classmethod
    def readable_name(cls, value: str) -> str:
        if any(unicodedata.category(char) == "Cc" for char in value):
            raise ValueError("give the saved search a name without control characters")
        return value

    @field_validator("name", "query")
    @classmethod
    def valid_unicode(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("saved search text must be valid Unicode") from None
        return value

    @field_validator("query")
    @classmethod
    def valid_query(cls, value: str) -> str:
        try:
            search.split_search(value)
        except (ValueError, OverflowError):
            raise ValueError("the saved search query has an invalid filter value") from None
        return value


class SavedSearchUpdate(SavedSearchFields):
    revision: int = Field(ge=1)


class SavedSearch(SavedSearchFields):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    revision: int = Field(ge=1)


class SavedSearchCollection(BaseModel):
    items: list[SavedSearch]
    max_items: int = MAX_SAVED_SEARCHES
    max_name: int = MAX_SEARCH_NAME
    max_query: int = MAX_SEARCH_QUERY
