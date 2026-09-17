"""Danbooru-backed validation without reaching the network."""

from __future__ import annotations

from urllib.error import URLError

import pytest

from backend.tags import (
    DanbooruTagVocabulary,
    TagLookupUnavailable,
    display_tag,
    normalize_tag,
    unchecked_results,
)

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
        "canonical": r"firefly \(honkai: star rail\)",
        "source": "danbooru",
        "category": "character",
        "post_count": 16_000,
        "reason": None,
    }
    assert by_tag["great score"]["source"] == "model"
    assert by_tag["year 2023"]["source"] == "model"
    assert "cutoff" in by_tag["future character"]["reason"]
    assert "deprecated" in by_tag["retired tag"]["reason"]
    assert by_tag["made up tag"]["reason"] == "not found in Danbooru"

    vocabulary.validate(tags, {"great score"}, CUTOFF)
    assert len(calls) == 1, "exact hits and misses are cached"


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


def test_the_real_fetcher_wraps_network_failures(monkeypatch):
    def offline(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr("backend.tags.urlopen", offline)
    vocabulary = DanbooruTagVocabulary()
    with pytest.raises(TagLookupUnavailable):
        vocabulary.suggest("green", CUTOFF)
