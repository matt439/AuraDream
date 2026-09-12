"""Danbooru tag lookup for tag-caption checkpoints.

Animagine's maintainers do not publish a checkpoint vocabulary: they say the
model was trained from nearly all Danbooru data and point users at Danbooru.
Its tag API is therefore the honest validation boundary. Lookups are cached in
process, kept off the event loop by the routes, and failures mean "unchecked",
never "invalid" - generation remains local and the editor remains authoritative.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_API_BASE = "https://safebooru.donmai.us"
_CATEGORIES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}
_YEAR_TAG = re.compile(r"year \d{4}", re.IGNORECASE)


class TagLookupUnavailable(RuntimeError):
    """The external vocabulary could not be reached or returned bad data."""


def normalize_tag(tag: str) -> str:
    """Turn the model-card's spaced, escaped form into a Danbooru name."""
    text = tag.strip().lower().replace(r"\(", "(").replace(r"\)", ")")
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def display_tag(name: str) -> str:
    """Danbooru name to the spaced form Animagine's examples use."""
    return name.replace("_", " ").replace("(", r"\(").replace(")", r"\)")


def is_model_tag(tag: str, special_tags: set[str]) -> bool:
    text = tag.strip().lower()
    return text in special_tags or bool(_YEAR_TAG.fullmatch(text))


def unchecked_results(tags: list[str], special_tags: set[str]) -> list[dict]:
    """Preserve locally known model tags while the remote index is offline."""
    results = []
    for tag in tags:
        known = is_model_tag(tag, special_tags)
        results.append(
            {
                "tag": tag,
                "valid": True if known else None,
                "canonical": tag if known else None,
                "source": "model" if known else None,
                "category": None,
                "post_count": None,
                "reason": None if known else "Danbooru lookup unavailable",
            }
        )
    return results


class DanbooruTagVocabulary:
    """Small cached client for the official Danbooru tag index."""

    def __init__(
        self, fetch_json: Callable[[str, dict[str, str]], list[dict]] | None = None
    ):
        self._fetch_json = fetch_json or self._request_json
        self._lock = threading.Lock()
        self._suggestions: dict[tuple[str, str], list[dict]] = {}
        self._records: dict[str, dict | None] = {}

    def suggest(self, query: str, cutoff: str, limit: int = 12) -> list[dict]:
        normalized = normalize_tag(query).replace("*", "").replace("?", "")
        if not normalized:
            return []
        key = (normalized, cutoff)
        with self._lock:
            cached = self._suggestions.get(key)
        if cached is not None:
            return [dict(item) for item in cached]

        rows = self._fetch_json(
            "/tags.json",
            {
                "search[name_or_alias_matches]": f"{normalized}*",
                "search[hide_empty]": "true",
                "search[order]": "count",
                "limit": str(limit),
            },
        )
        suggestions = []
        seen = set()
        for row in rows:
            name = str(row.get("name", ""))
            if not name or name in seen or not self._eligible(row, cutoff):
                continue
            seen.add(name)
            suggestions.append(self._suggestion(row))
        with self._lock:
            self._put(self._suggestions, key, suggestions, 256)
        return [dict(item) for item in suggestions]

    def validate(
        self, tags: list[str], special_tags: set[str], cutoff: str
    ) -> list[dict]:
        normalized = {
            normalize_tag(tag) for tag in tags if not is_model_tag(tag, special_tags)
        }
        with self._lock:
            missing = [name for name in normalized if name not in self._records]
        for chunk in self._chunks(missing):
            rows = self._fetch_json(
                "/tags.json",
                {"search[name_normalize]": ",".join(chunk), "limit": "100"},
            )
            found = {str(row.get("name", "")): row for row in rows if row.get("name")}
            with self._lock:
                for name in chunk:
                    self._put(self._records, name, found.get(name), 10_000)

        results = []
        for tag in tags:
            if is_model_tag(tag, special_tags):
                results.append(
                    {
                        "tag": tag,
                        "valid": True,
                        "canonical": tag,
                        "source": "model",
                        "category": None,
                        "post_count": None,
                        "reason": None,
                    }
                )
                continue
            with self._lock:
                row = self._records.get(normalize_tag(tag))
            results.append(self._validation(tag, row, cutoff))
        return results

    @staticmethod
    def _eligible(row: dict, cutoff: str) -> bool:
        created = str(row.get("created_at", ""))[:10]
        return (
            not bool(row.get("is_deprecated"))
            and int(row.get("post_count", 0)) > 0
            and bool(created)
            and created <= cutoff
        )

    @staticmethod
    def _suggestion(row: dict) -> dict:
        return {
            "text": display_tag(str(row["name"])),
            "canonical": str(row["name"]),
            "category": _CATEGORIES.get(int(row.get("category", 0)), "general"),
            "post_count": int(row.get("post_count", 0)),
        }

    def _validation(self, tag: str, row: dict | None, cutoff: str) -> dict:
        base = {
            "tag": tag,
            "valid": False,
            "canonical": None,
            "source": None,
            "category": None,
            "post_count": None,
            "reason": "not found in Danbooru",
        }
        if row is None:
            return base
        if bool(row.get("is_deprecated")):
            return {**base, "reason": "deprecated Danbooru tag"}
        if int(row.get("post_count", 0)) <= 0:
            return {**base, "reason": "Danbooru tag has no posts"}
        created = str(row.get("created_at", ""))[:10]
        if not created:
            return {**base, "reason": "Danbooru tag has no creation date"}
        if created > cutoff:
            return {**base, "reason": f"created after Animagine's {cutoff} cutoff"}
        suggestion = self._suggestion(row)
        return {
            "tag": tag,
            "valid": True,
            "canonical": suggestion["text"],
            "source": "danbooru",
            "category": suggestion["category"],
            "post_count": suggestion["post_count"],
            "reason": None,
        }

    @staticmethod
    def _chunks(
        names: Iterable[str], maximum_encoded_length: int = 1200
    ) -> Iterable[list[str]]:
        chunk: list[str] = []
        length = 0
        for name in names:
            encoded = len(urlencode({"name": name}))
            if chunk and length + encoded + 1 > maximum_encoded_length:
                yield chunk
                chunk, length = [], 0
            chunk.append(name)
            length += encoded + 1
        if chunk:
            yield chunk

    @staticmethod
    def _put(cache: dict, key, value, maximum: int) -> None:
        if key not in cache and len(cache) >= maximum:
            cache.pop(next(iter(cache)))
        cache[key] = value

    @staticmethod
    def _request_json(path: str, params: dict[str, str]) -> list[dict]:
        url = f"{_API_BASE}{path}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "LocalDiffusion/0.1 tag-helper"})
        try:
            # The origin and path are module constants; user text is query data.
            with urlopen(request, timeout=5) as response:
                payload = response.read(1_000_001)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise TagLookupUnavailable(str(exc)) from exc
        if len(payload) > 1_000_000:
            raise TagLookupUnavailable(
                "Danbooru returned an unexpectedly large response"
            )
        try:
            data = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TagLookupUnavailable("Danbooru returned invalid JSON") from exc
        if not isinstance(data, list):
            raise TagLookupUnavailable("Danbooru returned an unexpected response")
        return data


tag_vocabulary = DanbooruTagVocabulary()
