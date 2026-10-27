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


# How many suggestions one query lists: a word as broad as "hair" names more
# tags than a dozen, and the list scrolls.
SUGGESTION_LIMIT = 20


# A wiki page's DText as the finder shows its first sentence: a link as its
# words (`[[hair flip]]`, `[[Kururu (little princess)|]]`, `[[a|shown]]`),
# a markup tag or an external link's quotes dropped, a line break a space.
_WIKI_LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")
_WIKI_URL = re.compile(r'"([^"]+)":\S+')
_WIKI_BREAK = re.compile(r"\[br\]", re.IGNORECASE)
_WIKI_MARKUP = re.compile(r"\[/?[a-z]+\]", re.IGNORECASE)
_WIKI_QUALIFIER = re.compile(r"\s*\([^)]*\)$")
# A line no prose starts: a heading, a list, a post embed, a block.
_WIKI_SKIPPED = re.compile(r"(h\d\.|\*|!post|\[expand|\[table|\[quote)", re.IGNORECASE)
# A sentence ends where the next begins with a capital or a quotation, so
# "e.g. a cap" reads on.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
DEFINITION_MAX = 240


def wiki_definition(body: str) -> str | None:
    """A Danbooru wiki body's first paragraph of prose, links as their words,
    cut after its first sentence or at `DEFINITION_MAX`; None where the page
    opens with nothing but headings, lists and post embeds."""

    def link(match: re.Match) -> str:
        target, shown = match.group(1), match.group(2)
        if shown is None:
            return target
        # `[[x (y)|]]`, the pipe trick: the name without its qualifier.
        return shown or _WIKI_QUALIFIER.sub("", target)

    for paragraph in re.split(r"\r?\n\s*\r?\n", body or ""):
        # The paragraph's prose: its lines up to one no prose starts.
        lines = []
        for line in paragraph.strip().splitlines():
            if _WIKI_SKIPPED.match(line.strip()):
                break
            lines.append(line)
        text = _WIKI_BREAK.sub(" ", " ".join(lines))
        text = _WIKI_MARKUP.sub("", _WIKI_URL.sub(r"\1", _WIKI_LINK.sub(link, text)))
        text = " ".join(text.split())
        if not text:
            continue
        end = _SENTENCE_END.search(text)
        if end:
            text = text[: end.start()]
        if len(text) > DEFINITION_MAX:
            text = text[: DEFINITION_MAX - 1].rsplit(" ", 1)[0] + "…"
        return text
    return None


class TagLookupUnavailable(RuntimeError):
    """The external vocabulary could not be reached or returned bad data."""


def normalize_tag(tag: str) -> str:
    """Turn a prompt's spaced form, its parentheses escaped or not (the
    model cards' examples escape them), into a Danbooru name."""
    text = tag.strip().lower().replace(r"\(", "(").replace(r"\)", ")")
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def display_tag(name: str) -> str:
    """Danbooru name to the spaced form autocomplete writes, its parentheses
    bare: `hatsune miku (vocaloid)`, never A1111's `\\(`. Diffusers never
    unescapes, so each backslash is a CLIP token of its own - two for every
    qualified tag, in a 77-token window - and every reader here takes the
    two spellings as one tag. Whether the image differs is not measured."""
    return name.replace("_", " ")


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


def grid_variable_result(tag: str) -> dict:
    """A chip's answer for a grid placeholder (`{hair}`): no verdict, and the
    index never asked - it stands for the grid's values, not for a tag."""
    return {
        "tag": tag,
        "valid": None,
        "canonical": None,
        "source": None,
        "category": None,
        "post_count": None,
        "reason": "a grid variable: Run grid fills it with its values",
        "code": "grid_variable",
    }


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
    query: str, local: list[dict], remote: list[dict], avoid: set[str], limit: int = SUGGESTION_LIMIT
) -> list[dict]:
    """The guide's own tags and the index's, as one list for the box.

    An exact match first; then a name with the query at the start of any of
    its words - "hair" is `short hair` as much as `hair ornament`, which is
    how one asks what hair there is; then the query inside a word (`chair`),
    or only through an alias, which the index also answers for. Within that
    the index's order stands, by post count,
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
            else 1 if name.startswith(wanted) or f"_{wanted}" in name
            else 2
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
        self._descriptions: dict[tuple[str, str | None], list[dict]] = {}
        self._records: dict[str, dict | None] = {}

    def suggest(self, query: str, cutoff: str | None, limit: int = SUGGESTION_LIMIT) -> list[dict]:
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
                # Anywhere in the name: "hair" asks for `short hair` too.
                "search[name_or_alias_matches]": f"*{normalized}*",
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

    def describe(self, query: str, cutoff: str | None, limit: int = 12) -> list[dict]:
        """General tags whose Danbooru wiki page mentions `query`, by post
        count, each with its page's first sentence: "cowlick" or "stupid
        hair" finds `ahoge`, whose name says neither. A page that only
        mentions the words is found too, which the sentence lets the reader
        see; characters and series are left out, their pages listing every
        character with the trait. Blocking; raises `TagLookupUnavailable`."""
        wanted = " ".join(query.split()).lower()
        if not wanted:
            return []
        key = (wanted, cutoff)
        with self._lock:
            cached = self._descriptions.get(key)
        if cached is not None:
            return [dict(item) for item in cached]

        # By the page's tag's post count: unordered, the index answers the
        # 100 pages edited last, mostly characters, and "smile" missed
        # `smile`. The tag named what was typed is asked for whether or not
        # its page uses the words; a name with a comma would split the list.
        exact = normalize_tag(wanted)
        pages = self._fetch_json(
            "/wiki_pages.json",
            {
                "search[body_matches]": wanted,
                "search[is_deleted]": "false",
                "search[order]": "post_count",
                "limit": "100",
                "only": "title",
            },
        )
        titles = [
            title for title in dict.fromkeys(
                [exact] + [str(page.get("title", "")) for page in pages if page.get("title")]
            )
            if "," not in title
        ]
        rows = self._fetch_json(
            "/tags.json",
            {
                "search[name_comma]": ",".join(titles),
                "search[category]": "0",
                "search[hide_empty]": "true",
                "search[order]": "count",
                "limit": "100",
            },
        ) if titles else []
        found = [
            self._suggestion(row) for row in rows
            if row.get("name") and int(row.get("category", 0)) == 0 and self._eligible(row, cutoff)
        ]
        # A tag named what was typed leads: "cowlick" is a tag of its own.
        found.sort(key=lambda item: item["canonical"] != exact)
        found = found[:limit]
        bodies = self._fetch_json(
            "/wiki_pages.json",
            {
                "search[title_comma]": ",".join(item["canonical"] for item in found),
                "limit": str(len(found)),
                "only": "title,body",
            },
        ) if found else []
        definitions = {
            str(page.get("title", "")): wiki_definition(str(page.get("body") or ""))
            for page in bodies
        }
        results = [{**item, "definition": definitions.get(item["canonical"])} for item in found]
        with self._lock:
            self._put(self._descriptions, key, results, 256)
        return [dict(item) for item in results]

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
        request = Request(url, headers={"User-Agent": "AuraDream/0.1 tag-helper"})
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
