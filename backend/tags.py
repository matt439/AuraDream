"""Danbooru tag lookup for tag-caption checkpoints.

The checkpoints do not publish complete frozen vocabularies, while their
authors document Danbooru-derived tag captions, sometimes with dataset cutoffs.
Danbooru's tag API is therefore an advisory validation boundary. Lookups are cached in
process, kept off the event loop by the routes, and failures mean "unchecked",
never "invalid" - generation remains local and the editor remains authoritative.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Iterable
from http.client import HTTPException
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
    """Danbooru name to the spaced form checkpoint prompts use."""
    return name.replace("_", " ").replace("(", r"\(").replace(")", r"\)")


def is_model_tag(tag: str, special_tags: set[str], allow_year_tag: bool = True) -> bool:
    text = tag.strip().lower()
    return text in special_tags or (allow_year_tag and bool(_YEAR_TAG.fullmatch(text)))


# Longer than any name the index holds: a comma item past this - a hybrid
# prompt's sentence - is answered unknown here, never asked about, reachable
# index or not. Asked, the whole request was refused for it (a 200-character
# bound on every item), and the tray said Danbooru was unreachable.
TAG_NAME_MAX = 200


def too_long(tag: str) -> bool:
    return len(tag.strip()) > TAG_NAME_MAX


def too_long_result(tag: str) -> dict:
    return {
        "tag": tag,
        "valid": False,
        "canonical": None,
        "source": None,
        "category": None,
        "post_count": None,
        "reason": "longer than any Danbooru tag",
        "code": "not_found",
    }


def unchecked_results(
    tags: list[str], special_tags: set[str], allow_year_tag: bool = True
) -> list[dict]:
    """Preserve locally known model tags while the remote index is offline -
    and what needs no index at all: an item longer than any tag."""
    results = []
    for tag in tags:
        known = is_model_tag(tag, special_tags, allow_year_tag)
        if not known and too_long(tag):
            results.append(too_long_result(tag))
            continue
        results.append(
            {
                "tag": tag,
                "valid": True if known else None,
                "canonical": tag if known else None,
                "source": "model" if known else None,
                "category": None,
                "post_count": None,
                "reason": None if known else "Danbooru lookup unavailable",
                "code": None,
            }
        )
    return results


def rank_suggestions(
    query: str, local: list[dict], remote: list[dict], avoid: set[str], limit: int = 12
) -> list[dict]:
    """The guide's own tags and the index's, as one list for the box.

    An exact match first; then by how the name matches - from its start
    before at a later word's (the index also answers for an alias, which
    matches neither). Within that the index's order stands, by post count,
    and a guide tag, which has no count, goes ahead of it only once the query
    covers half of the tag: "high s" is heading for `high score`, "gre" is no
    more `great score` than `green hair`. Guide tags were merged ahead of
    every remote result so they stay available offline, which they do here
    too - offline they are the whole list. A tag of the guide's recommended
    negative prompt (`avoid`, normalized) never comes first unless typed
    whole: the first item is the one Enter and Tab take, and "lo" then Tab
    used to write `low quality` into the positive prompt.
    """
    wanted = normalize_tag(query).replace("*", "").replace("?", "")
    names = {item["canonical"] for item in local}
    merged = [(item, True) for item in local] + [
        (item, False) for item in remote if item["canonical"] not in names
    ]

    def key(entry: tuple[int, tuple[dict, bool]]) -> tuple:
        index, (item, guide) = entry
        name = normalize_tag(item["canonical"])
        match = (
            0 if name == wanted
            else 1 if name.startswith(wanted)
            else 2 if f"_{wanted}" in name
            else 3
        )
        ahead = guide and 2 * len(wanted) >= len(name)
        return (match, not ahead, guide, index)

    def avoided(item: dict) -> bool:
        name = normalize_tag(item["canonical"])
        return name in avoid and name != wanted

    ranked = [item for _, (item, _) in sorted(enumerate(merged), key=key)]
    if ranked and avoided(ranked[0]):
        first = next((i for i, item in enumerate(ranked) if not avoided(item)), None)
        if first is not None:
            ranked.insert(0, ranked.pop(first))
    return ranked[:limit]


class DanbooruTagVocabulary:
    """Small cached client for the official Danbooru tag index."""

    def __init__(
        self, fetch_json: Callable[[str, dict[str, str]], list[dict]] | None = None
    ):
        self._fetch_json = fetch_json or self._request_json
        self._lock = threading.Lock()
        self._suggestions: dict[tuple[str, str | None], list[dict]] = {}
        self._records: dict[str, dict | None] = {}

    def suggest(self, query: str, cutoff: str | None, limit: int = 12) -> list[dict]:
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
        self,
        tags: list[str],
        special_tags: set[str],
        cutoff: str | None,
        allow_year_tag: bool = True,
    ) -> list[dict]:
        normalized = {
            normalize_tag(tag)
            for tag in tags
            if not is_model_tag(tag, special_tags, allow_year_tag)
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
            if is_model_tag(tag, special_tags, allow_year_tag):
                results.append(
                    {
                        "tag": tag,
                        "valid": True,
                        "canonical": tag,
                        "source": "model",
                        "category": None,
                        "post_count": None,
                        "reason": None,
                        "code": None,
                    }
                )
                continue
            with self._lock:
                row = self._records.get(normalize_tag(tag))
            results.append(self._validation(tag, row, cutoff))
        return results

    @staticmethod
    def _eligible(row: dict, cutoff: str | None) -> bool:
        created = str(row.get("created_at") or "")[:10]
        return (
            not bool(row.get("is_deprecated"))
            and int(row.get("post_count", 0)) > 0
            and (cutoff is None or (bool(created) and created <= cutoff))
        )

    @staticmethod
    def _suggestion(row: dict) -> dict:
        return {
            "text": display_tag(str(row["name"])),
            "canonical": str(row["name"]),
            "category": _CATEGORIES.get(int(row.get("category", 0)), "general"),
            "post_count": int(row.get("post_count", 0)),
        }

    def _validation(self, tag: str, row: dict | None, cutoff: str | None) -> dict:
        # `code` names the reason for the page, which marks a tag too new for
        # the checkpoint apart from one the index has never heard of: the
        # fixes differ, and matching on the sentence would break with its
        # wording.
        base = {
            "tag": tag,
            "valid": False,
            "canonical": None,
            "source": None,
            "category": None,
            "post_count": None,
            "reason": "not found in Danbooru",
            "code": "not_found",
        }
        if row is None:
            return base
        if bool(row.get("is_deprecated")):
            return {**base, "reason": "deprecated Danbooru tag", "code": "deprecated"}
        if int(row.get("post_count", 0)) <= 0:
            return {**base, "reason": "Danbooru tag has no posts", "code": "empty"}
        created = str(row.get("created_at") or "")[:10]
        if cutoff is not None and not created:
            return {**base, "reason": "Danbooru tag has no creation date", "code": "no_date"}
        if cutoff is not None and created > cutoff:
            return {
                **base,
                "reason": f"created after this checkpoint's {cutoff} cutoff",
                "code": "after_cutoff",
            }
        suggestion = self._suggestion(row)
        return {
            "tag": tag,
            "valid": True,
            "canonical": suggestion["text"],
            "source": "danbooru",
            "category": suggestion["category"],
            "post_count": suggestion["post_count"],
            "reason": None,
            "code": None,
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
        # HTTPException: a reply cut off mid-body (IncompleteRead) or a
        # malformed status line is no more an answer than a refused connection.
        except (HTTPError, URLError, TimeoutError, OSError, HTTPException) as exc:
            raise TagLookupUnavailable(str(exc) or type(exc).__name__) from exc
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
