"""The Danbooru tag index as the chips and the analyser both ask it.

The guide's special and model tags, the artist prefix, the one lookup the
chips and Check share, and the categories Check's order and artist rules
read. `tag_categories` lives here rather than in `prompt_check`, though it
reads `prompt_rules` and the guide, because it is the analyser's use of this
lookup under the chips' own ceiling (`MAX_TAG_ITEMS`) and the bare Danbooru
helper (`BARE_DANBOORU`); moved into `prompt_check`, the two would import
each other.

`tag_vocabulary` is bound here, and read when a lookup runs: the tests and
the fake server replace it on this module.
"""
from __future__ import annotations

import logging

from . import prompt_rules, prompt_text
from .jobs import short_error
from .model_defaults import guide_for
from .schemas import TagValidationRequest
from .tags import (
    TagLookupUnavailable,
    grid_variable_result,
    is_model_tag,
    normalize_tag,
    tag_vocabulary,
    too_long,
    too_long_result,
)

log = logging.getLogger("local_diffusion")


def special_tags(helper: dict) -> set[str]:
    tags = {
        tag.lower()
        for group in helper.get("groups", [])
        for tag in group.get("tags", [])
    }
    tags.update(tag.lower() for tag in helper.get("recommended_suffix", []))
    return tags


def model_tag_suggestions(helper: dict, query: str) -> list[dict]:
    wanted = normalize_tag(query)
    if not wanted:
        return []
    return [
        {
            "text": tag,
            "canonical": normalize_tag(tag),
            "category": "model",
            "post_count": None,
        }
        for tag in sorted(special_tags(helper))
        if wanted in normalize_tag(tag)
    ]


def negative_tags(helper: dict) -> set[str]:
    """The guide's recommended negative prompt as tag names: what the
    suggestion list never puts first for the positive prompt."""
    return {
        normalize_tag(part)
        for part in (helper.get("negative_prompt") or "").split(",")
        if part.strip()
    }


def artist_tag_name(helper: dict, tag: str) -> str | None:
    prefix = helper.get("artist_prefix")
    if prefix and tag.lower().startswith(prefix.lower()):
        return tag[len(prefix):].strip()
    return None


def lookup_tags(helper: dict, tags: list[str]) -> list[dict]:
    """The tag index's verdict on `tags` under this helper's cutoff, special
    tags and year form. Blocking; raises `TagLookupUnavailable`.

    One call for the chips and the analyser, so the two cannot disagree about
    what a tag is. An item longer than any tag (`tags.TAG_NAME_MAX`: a hybrid
    prompt's sentence) is answered unknown here, never asked about, and a grid
    placeholder (`{hair}`) is answered with no verdict (`grid_variable`).
    """
    special = special_tags(helper)
    year = bool(helper.get("year_tag"))

    prefix = helper.get("artist_prefix")

    def asks(tag: str) -> bool:
        if prompt_text.is_grid_variable(tag):
            return False
        return not too_long(artist_tag_name(helper, tag) or tag) or is_model_tag(tag, special, year)

    asked = [artist_tag_name(helper, tag) or tag for tag in tags if asks(tag)]
    answers = iter(
        tag_vocabulary.validate(asked, special, helper["vocabulary_cutoff"], year) if asked else []
    )
    results = []
    for tag in tags:
        if prompt_text.is_grid_variable(tag):
            results.append(grid_variable_result(tag))
            continue
        if not asks(tag):
            results.append(too_long_result(tag))
            continue
        result = {**next(answers), "tag": tag}
        if artist_tag_name(helper, tag) is not None and result.get("valid") and result.get("category") != "artist":
            assert prefix is not None, "an artist-prefixed name requires its guide's prefix"
            result.update(
                valid=False, reason=f"the name after {prefix.strip()} is not a Danbooru artist", code="not_found"
            )
        results.append(result)
    return results


# The chips' own ceiling on one validation request, so the analyser never
# asks the tag index about more items than the page may.
MAX_TAG_ITEMS = TagValidationRequest.model_json_schema()["properties"]["tags"]["maxItems"]


# What a guide with no tag helper asks the index with when one of its rules
# reads a category anyway: no cutoff, no special tags of its own. Anima's "put
# @ before an artist name" took it until its guide gained the card's Tags menu
# (2026-10-17); no catalogue guide takes it today.
BARE_DANBOORU = {"vocabulary": "danbooru", "vocabulary_cutoff": None}


def tag_categories(
    model_id: str, prompt: str
) -> tuple[dict[str, str] | None, str | None, list[str]]:
    """Each comma item's Danbooru category, for the order and artist checks,
    or None - why some item a rule reads was never looked up, or None - and
    the items the index was asked about, answered or not.

    Only on a checkpoint whose guide names the Danbooru vocabulary, or whose
    rules need to know an artist (`wants_artist_categories`), and only for
    the items a rule would read (`lookup_items`: a prompt that reads as tags,
    or - where the rules read artists - the tag-shaped items of one that
    reads as prose), the chips' 64 at most. No categories when there was
    nothing to ask or the index could not be reached or gave a reply that
    did not read: the rules then place only what the guide's own groups
    know, and the index is never an error here, as it is never one for the
    chips. What that cost is still said (`categories_detail`): offline, an
    out-of-order prompt and an in-order one otherwise got the same answer.
    """
    helper = guide_for(model_id).get("tag_helper")
    if not helper or helper.get("vocabulary") != "danbooru":
        if not prompt_rules.wants_artist_categories(model_id):
            return None, None, []
        helper = BARE_DANBOORU
    wanted = prompt_rules.lookup_items(model_id, prompt)
    names = wanted[:MAX_TAG_ITEMS]
    if not names:
        return None, None, []
    try:
        results = lookup_tags(helper, names)
    except Exception as exc:  # noqa: BLE001 - advisory: a bad reply costs the order checks
        log.info("prompt analysis: tag categories unchecked: %s", short_error(exc))
        if not isinstance(exc, TagLookupUnavailable):
            log.debug("tag category traceback", exc_info=True)
        # The chips' one error covers two things: a connection that never
        # answered, and an answer - an HTTP error, a body cut off, JSON that
        # is not the rows asked for - that could not be used.
        from urllib.error import HTTPError

        cause = exc.__cause__ if isinstance(exc, TagLookupUnavailable) else None
        reason = (
            "the tag index was unreachable"
            if isinstance(cause, OSError) and not isinstance(cause, HTTPError)
            else "the tag index gave no usable reply"
        )
        return None, f"{reason}, so no tag has its Danbooru category", names
    categories = {row["tag"]: row["category"] for row in results if row.get("category")}
    if len(wanted) > len(names):
        return categories, (
            f"the tag index was asked about the first {len(names)} of {len(wanted)} tags "
            "only, so the rest have no Danbooru category"
        ), names
    return categories, None, names
