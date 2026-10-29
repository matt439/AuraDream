"""Unconvertible PNG dimensions leave usable settings and an honest size note."""
from __future__ import annotations

import json
import sys

import pytest

_PROMPT = "a harbor in fog"
_NEGATIVE = "blurry"
_UNREADABLE = "Its size could not be read, so the panel keeps its own."
_ABSENT = "It names no size, so the panel keeps its own."


def _chunks(source, size, *, absent=False, refine=None):
    if source == "auradream":
        metadata = {"prompt": _PROMPT, "negative_prompt": _NEGATIVE, "steps": 9}
        if not absent:
            metadata["size"] = size
        if refine is not None:
            metadata["refine"] = refine
        return {source: json.dumps(metadata)}
    settings = "Steps: 20, Seed: 1, CFG scale: 7"
    if not absent:
        settings += f", Size: {size}"
    return {"parameters": f"{_PROMPT}\nNegative prompt: {_NEGATIVE}\n{settings}"}


def _answer(client, source, size, **options):
    response = client.post("/api/png/settings", json={
        "name": "dimensions.png", "chunks": _chunks(source, size, **options),
        "panel": {"steps": 9},
    })
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["settings"]["prompt"] == _PROMPT
    assert answer["negative_prompt"] == _NEGATIVE
    assert answer["settings"]["steps"] == 9
    return answer


def _bounded_conversion_limit():
    limit = sys.get_int_max_str_digits()
    if not limit:
        pytest.skip("decimal input conversion has no configured refusal limit")
    if limit > 50_000:
        pytest.skip("the configured limit would exceed this bounded PNG fixture")
    return limit


@pytest.mark.parametrize("source", ["auradream", "parameters"])
@pytest.mark.parametrize("side", ["width", "height"])
def test_oversized_decimal_conversion_preserves_usable_png_settings(client, source, side):
    oversized = "9" * (_bounded_conversion_limit() + 1)
    size = f"{oversized}x512" if side == "width" else f"512x{oversized}"
    answer = _answer(client, source, size)
    assert answer["size"] is False
    assert answer["notes"].count(_UNREADABLE) == 1
    assert _ABSENT not in answer["notes"]
    assert client.get("/api/gallery?all_models=true").json()["total"] == 0
    assert client.get("/api/jobs").json() == []


@pytest.mark.parametrize("source", ["auradream", "parameters"])
@pytest.mark.parametrize("size", ["000512x000768", "\u0660\u0665\u0661\u0662x\u0660\u0667\u0666\u0668"])
def test_ordinary_leading_zero_and_unicode_decimal_sizes_still_carry(client, source, size):
    answer = _answer(client, source, size)
    assert answer["size"] is True
    assert (answer["settings"]["width"], answer["settings"]["height"]) == (512, 768)
    assert _UNREADABLE not in answer["notes"]


@pytest.mark.parametrize("source", ["auradream", "parameters"])
def test_leading_zeros_at_the_existing_conversion_limit_still_carry(client, source):
    limit = _bounded_conversion_limit()
    size = "0" * (limit - 3) + "512x768"
    answer = _answer(client, source, size)
    assert answer["size"] is True
    assert (answer["settings"]["width"], answer["settings"]["height"]) == (512, 768)
    assert _UNREADABLE not in answer["notes"]


@pytest.mark.parametrize("source", ["auradream", "parameters"])
@pytest.mark.parametrize("size", ["not-a-size", ""])
def test_named_unreadable_sizes_have_an_honest_note(client, source, size):
    answer = _answer(client, source, size)
    assert answer["size"] is False
    assert answer["notes"].count(_UNREADABLE) == 1
    assert _ABSENT not in answer["notes"]


@pytest.mark.parametrize("source", ["auradream", "parameters"])
def test_absent_size_keeps_its_existing_behavior(client, source):
    answer = _answer(client, source, None, absent=True)
    assert answer["size"] is False
    assert _UNREADABLE not in answer["notes"]
    assert (_ABSENT in answer["notes"]) is (source == "parameters")


def test_named_null_app_size_is_unreadable_instead_of_absent(client):
    answer = _answer(client, "auradream", None)
    assert answer["size"] is False
    assert answer["notes"].count(_UNREADABLE) == 1


def test_unconvertible_refine_size_keeps_the_existing_source_pass_note(client):
    size = "9" * (_bounded_conversion_limit() + 1) + "x512"
    answer = _answer(client, "auradream", size, refine=17)
    assert answer["size"] is False
    assert answer["settings"]["refine"] is None
    assert _UNREADABLE not in answer["notes"]
    assert any(note.startswith("It is a refine pass,") for note in answer["notes"])
