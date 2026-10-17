"""Danbooru-backed validation without reaching the network."""

from __future__ import annotations

import http.client
from urllib.error import URLError

import pytest

from backend.tags import (
    DanbooruTagVocabulary,
    TagLookupUnavailable,
    display_tag,
    normalize_tag,
    rank_suggestions,
    unchecked_results,
)
from tests.fakes import CutReply

CUTOFF = "2025-01-07"


def record(name, count=100, category=0, created="2024-01-01", deprecated=False):
    return {
        "name": name,
        "post_count": count,
        "category": category,
        "created_at": f"{created}T00:00:00Z",
        "is_deprecated": deprecated,
    }


def test_suggestions_are_normalized_ranked_data_before_the_model_cutoff():
    calls = []

    def fetch(path, params):
        calls.append((path, params))
        return [
            record("green_hair", 591_000),
            record("green_hat", 41_000),
            record("green_future", 500, created="2025-02-01"),
            record("green_old", 400, deprecated=True),
            record("green_empty", 0),
        ]

    vocabulary = DanbooruTagVocabulary(fetch)
    suggestions = vocabulary.suggest(" Green h ", CUTOFF)
    assert [item["text"] for item in suggestions] == ["green hair", "green hat"]
    assert suggestions[0] == {
        "text": "green hair",
        "canonical": "green_hair",
        "category": "general",
        "post_count": 591_000,
    }
    assert calls[0][1]["search[name_or_alias_matches]"] == "green_h*"
    assert vocabulary.suggest("green h", CUTOFF) == suggestions
    assert len(calls) == 1, "the same prefix is served from the process cache"


def test_optional_cutoff_uses_live_index_without_weakening_dated_profiles():
    rows = [
        record("green_hair"),
        record("green_new", created="2026-04-01"),
        {**record("green_undated"), "created_at": None},
        record("green_empty", count=0),
        record("green_retired", deprecated=True),
    ]
    vocabulary = DanbooruTagVocabulary(lambda _path, _params: rows)
    assert [item["text"] for item in vocabulary.suggest("green", None)] == [
        "green hair", "green new", "green undated"
    ]
    assert [item["text"] for item in vocabulary.suggest("green", CUTOFF)] == [
        "green hair"
    ]
    names = ["green hair", "green new", "green undated", "green empty", "green retired"]
    results = vocabulary.validate(names, set(), None)
    assert [item["valid"] for item in results] == [True, True, True, False, False]
    dated = vocabulary.validate(names, set(), CUTOFF)
    assert [item["valid"] for item in dated] == [True, False, False, False, False]
    assert "cutoff" in dated[1]["reason"]


def test_validation_handles_model_tags_characters_unknowns_and_cutoff():
    calls = []
    rows = {
        "green_hair": record("green_hair", 591_000),
        "firefly_(honkai:_star_rail)": record(
            "firefly_(honkai:_star_rail)", 16_000, category=4
        ),
        "future_character": record("future_character", created="2025-03-01"),
        "retired_tag": record("retired_tag", deprecated=True),
    }

    def fetch(path, params):
        calls.append((path, params))
        names = params["search[name_normalize]"].split(",")
        return [rows[name] for name in names if name in rows]

    vocabulary = DanbooruTagVocabulary(fetch)
    tags = [
        "green hair",
        r"firefly \(honkai: star rail\)",
        "great score",
        "year 2023",
        "future character",
        "retired tag",
        "made up tag",
    ]
    results = vocabulary.validate(tags, {"great score"}, CUTOFF)
    by_tag = {item["tag"]: item for item in results}
    assert by_tag["green hair"]["valid"] is True
    assert by_tag[r"firefly \(honkai: star rail\)"] == {
        "tag": r"firefly \(honkai: star rail\)",
        "valid": True,
        # Bare, as autocomplete writes it: the typed escapes are the same tag.
        "canonical": "firefly (honkai: star rail)",
        "source": "danbooru",
        "category": "character",
        "post_count": 16_000,
        "reason": None,
        "code": None,
    }
    assert by_tag["great score"]["source"] == "model"
    assert by_tag["year 2023"]["source"] == "model"
    assert "cutoff" in by_tag["future character"]["reason"]
    assert "deprecated" in by_tag["retired tag"]["reason"]
    assert by_tag["made up tag"]["reason"] == "not found in Danbooru"
    # The reason as data too, so the page can mark a tag too new for the
    # checkpoint apart from a typo without reading the sentence.
    assert [by_tag[tag]["code"] for tag in tags] == [
        None, None, None, None, "after_cutoff", "deprecated", "not_found",
    ]

    vocabulary.validate(tags, {"great score"}, CUTOFF)
    assert len(calls) == 1, "exact hits and misses are cached"


def test_every_unknown_reason_has_its_code():
    rows = {
        "empty_tag": record("empty_tag", count=0),
        "undated_tag": {**record("undated_tag"), "created_at": None},
    }

    def fetch(path, params):
        return [rows[name] for name in params["search[name_normalize]"].split(",") if name in rows]

    results = DanbooruTagVocabulary(fetch).validate(["empty tag", "undated tag"], set(), CUTOFF)
    assert [(item["code"], item["reason"]) for item in results] == [
        ("empty", "Danbooru tag has no posts"),
        ("no_date", "Danbooru tag has no creation date"),
    ]
    assert all(item["code"] is None for item in unchecked_results(["a", "b"], {"a"}))


def suggestion(name, count=None):
    return {"text": display_tag(name), "canonical": name, "category": "general", "post_count": count}


def guide(name):
    return {"text": name.replace("_", " "), "canonical": name, "category": "model", "post_count": None}


def ranked(query, local, remote, avoid=()):
    return [item["canonical"] for item in rank_suggestions(query, local, remote, set(avoid))]


def test_guide_tags_rank_among_the_index_by_how_much_of_them_is_typed():
    remote = [suggestion("green_eyes", 900_000), suggestion("green_hair", 590_000)]
    local = [guide("great_score")]
    # "gre" is no more `great score` than `green hair`: the index's order first.
    assert ranked("gre", local, remote) == ["green_eyes", "green_hair", "great_score"]
    # Half of it typed, the guide tag leads its own match.
    school = [suggestion("high_school_uniform", 300_000)]
    assert ranked("high s", [guide("high_score")], school) == ["high_score", "high_school_uniform"]
    # Offline, the guide's tags are the whole list.
    assert ranked("great s", local, []) == ["great_score"]


def test_an_exact_match_leads_and_a_name_match_beats_an_alias():
    remote = [suggestion("abs_(muscle)", 500_000), suggestion("abs", 400_000)]
    assert ranked("abs", [guide("absurdres")], remote) == ["abs", "abs_(muscle)", "absurdres"]
    # The index answers for an alias too: a name that starts with the query
    # comes before one found through an alias.
    alias = [suggestion("long_hair", 4_000_000), suggestion("very_long_hair", 600_000),
             suggestion("hair_twirling", 10_000)]
    assert ranked("long", [], alias) == ["long_hair", "very_long_hair", "hair_twirling"]


def test_the_recommended_negative_is_never_first_unless_typed_whole():
    negative = {"low_quality", "low_score", "lowres"}
    remote = [suggestion("lowres", 900_000), suggestion("long_hair", 4_000_000)]
    local = [guide("low_quality"), guide("low_score")]
    assert ranked("lo", local, remote, negative)[0] == "long_hair"
    # Typed whole, it is what was asked for.
    assert ranked("lowres", [], [suggestion("lowres", 900_000)], negative) == ["lowres"]
    assert ranked("low quality", local, [], negative)[0] == "low_quality"
    # Nothing else matches: it stays, rather than an empty list.
    assert ranked("low q", [guide("low_quality")], [], negative) == ["low_quality"]


def test_numeric_year_is_only_special_for_profiles_that_document_it():
    calls = []

    def fetch(path, params):
        calls.append((path, params))
        return []

    vocabulary = DanbooruTagVocabulary(fetch)
    result = vocabulary.validate(["year 2023"], set(), CUTOFF, allow_year_tag=False)[0]
    assert result["valid"] is False and result["reason"] == "not found in Danbooru"
    assert calls


def test_unavailable_lookup_is_not_reported_as_invalid():
    def offline(path, params):
        raise URLError("offline")

    vocabulary = DanbooruTagVocabulary(offline)
    with pytest.raises(URLError):
        vocabulary.suggest("green", CUTOFF)

    results = unchecked_results(["great score", "ordinary tag"], {"great score"})
    assert results[0]["valid"] is True
    assert results[1]["valid"] is None


def test_tag_names_round_trip_between_danbooru_and_animagine_forms():
    canonical = "firefly_(honkai:_star_rail)"
    assert normalize_tag(display_tag(canonical)) == canonical
    # The authors' escaped spelling reads as the same name.
    assert normalize_tag(r"firefly \(honkai: star rail\)") == canonical


def test_autocomplete_writes_parentheses_bare():
    # Two CLIP tokens fewer than A1111's escapes, which diffusers never
    # unescapes (the 2026-10-17 decision, design notes' prompt analyser).
    assert display_tag("hatsune_miku_(vocaloid)") == "hatsune miku (vocaloid)"
    vocabulary = DanbooruTagVocabulary(
        lambda path, params: [
            {"name": "hatsune_miku_(vocaloid)", "category": 4, "post_count": 10, "created_at": "2020-01-01"}
        ]
    )
    assert vocabulary.suggest("hatsune", None)[0]["text"] == "hatsune miku (vocaloid)"


def test_the_real_fetcher_wraps_network_failures(monkeypatch):
    def offline(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr("backend.tags.urlopen", offline)
    vocabulary = DanbooruTagVocabulary()
    with pytest.raises(TagLookupUnavailable):
        vocabulary.suggest("green", CUTOFF)


def test_the_real_fetcher_wraps_a_broken_reply(monkeypatch):
    monkeypatch.setattr("backend.tags.urlopen", lambda request, timeout: CutReply())
    with pytest.raises(TagLookupUnavailable, match="IncompleteRead"):
        DanbooruTagVocabulary().validate(["green hair"], set(), CUTOFF)

    def garbled(request, timeout):
        raise http.client.BadStatusLine("HTTP/1.1 2x0 Garbled")

    monkeypatch.setattr("backend.tags.urlopen", garbled)
    with pytest.raises(TagLookupUnavailable):
        DanbooruTagVocabulary().suggest("green", CUTOFF)


def test_the_chips_prompt_is_what_a_check_reads_and_never_a_refusal():
    # The prompt beside the chips' tags is for their order slots, Check's
    # reading of it: as long as a Check takes, and anything else - blank,
    # longer - is no prompt rather than a 422 that would leave every chip
    # unchecked.
    from backend.schemas import TAG_SLOT_PROMPT_MAX, PromptAnalysisRequest, TagValidationRequest

    schema = PromptAnalysisRequest.model_json_schema()["properties"]["prompt"]
    assert TAG_SLOT_PROMPT_MAX == schema["maxLength"]
    assert TagValidationRequest(tags=["1girl"]).prompt is None
    fits = "1girl, " + "x" * (TAG_SLOT_PROMPT_MAX - 7)
    assert TagValidationRequest(tags=["1girl"], prompt=fits).prompt == fits
    for prompt in ("", "  \n", fits + "x"):
        assert TagValidationRequest(tags=["1girl"], prompt=prompt).prompt is None
    # A code point is one character, as pydantic counts the Check's limit.
    emoji = "😀" * TAG_SLOT_PROMPT_MAX
    assert TagValidationRequest(tags=["1girl"], prompt=emoji).prompt == emoji
