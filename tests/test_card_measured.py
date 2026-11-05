"""The measured refine and inpaint words on a card other than the 16 GB one
they were measured on (GEN#44).

No GPU: `card_memory` is handed to `model_defaults.hints_for` directly, or
`card_facts.card` is patched for the status (tests/conftest.py makes every
other test's card unknown).
"""
from __future__ import annotations

import pytest

from backend import card_facts
from backend.launch import CHECKPOINTS
from backend.model_defaults import MEASURED_CARD_GB, card_words, defaults_for, hints_for

GIB = 1024 ** 3
# torch's total_memory: the 3070 a little under its 8 GB, the 5080 16,303 MiB.
RTX_3070 = int(7.79 * GIB)
RTX_5080 = 16_303 * 1024 ** 2
EIGHT_GB = "on a 16 GB card; this card has 8 GB"

KREA_DEV = "black-forest-labs/FLUX.1-Krea-dev"
ANIMAGINE = "cagliostrolab/animagine-xl-4.0"


def _ids():
    yield from (entry.repo for entry in CHECKPOINTS)
    yield "local/custom-checkpoint"


def test_card_words_name_the_measured_card_and_this_one():
    assert MEASURED_CARD_GB == 16
    assert card_words(None) is None
    assert card_words(RTX_5080) is None
    assert card_words(16 * GIB) is None
    assert card_words(RTX_3070) == EIGHT_GB
    assert card_words(12 * GIB) == "on a 16 GB card; this card has 12 GB"
    # Larger is another card too: its numbers were not measured either.
    assert card_words(24 * GIB) == "on a 16 GB card; this card has 24 GB"


@pytest.mark.parametrize("card", [None, RTX_5080, 16 * GIB])
def test_an_unknown_or_16_gb_card_keeps_every_hint_as_it_was(card):
    for model_id in _ids():
        for refines in (None, True, False):
            assert hints_for(
                model_id, refine_supported=refines, card_memory=card,
            ) == hints_for(model_id, refine_supported=refines), model_id


def test_an_8_gb_card_says_the_refine_run_was_the_16_gb_cards():
    before = hints_for(KREA_DEV)["refine_tight"]
    after = hints_for(KREA_DEV, card_memory=RTX_3070)
    assert after["refine_tight"] == f"{before} {EIGHT_GB}"
    assert after["refine_tight"] == (
        "past the 1536×1536 (2.36 MP) measured on FLUX.1 Krea dev at fp8 "
        "on a 16 GB card; this card has 8 GB"
    )
    assert after["tight_card"] == EIGHT_GB


def test_the_measured_line_and_its_pixels_are_kept_on_any_card():
    # Scaled nothing: the page's line stays the measured pass's pixels, and
    # only the words say whose card it was.
    for model_id in _ids():
        hints = hints_for(model_id, card_memory=RTX_3070)
        pixels = defaults_for(model_id)["refine_tight_pixels"]
        assert ("refine_tight" in hints) == ("refine_tight" in hints_for(model_id)), model_id
        if "refine_tight" in hints:
            assert pixels is not None and hints["refine_tight"].endswith(EIGHT_GB), model_id
        assert hints["tight_card"] == EIGHT_GB, model_id


def test_every_other_hint_is_unchanged_on_an_8_gb_card():
    for model_id in _ids():
        before = hints_for(model_id)
        after = hints_for(model_id, card_memory=RTX_3070)
        assert set(after) == set(before) | {"tight_card"}, model_id
        for key in set(before) - {"refine_tight", "inpaint_strength"}:
            assert after[key] == before[key], (model_id, key)


def test_the_inpaint_strength_names_the_16_gb_card():
    for model_id in _ids():
        before = hints_for(model_id).get("inpaint_strength")
        after = hints_for(model_id, card_memory=RTX_3070).get("inpaint_strength")
        if before is None:
            assert after is None, model_id
            continue
        assert "on the card" in before, model_id
        assert "on the card" not in after, model_id
        assert after == before.replace("on the card", "on a 16 GB card"), model_id
    assert hints_for(ANIMAGINE, card_memory=RTX_3070)["inpaint_strength"].startswith(
        "the library's number, measured on a 16 GB card (2026-10-17)"
    )
    assert hints_for(KREA_DEV, card_memory=RTX_3070)["inpaint_strength"].startswith(
        "measured on a 16 GB card (2026-10-19)"
    )


def test_the_status_carries_the_cards_words(client, monkeypatch):
    unknown = client.get("/api/status").json()["hints"]
    assert "tight_card" not in unknown
    monkeypatch.setattr(card_facts, "card", lambda device="cuda": card_facts.Card(RTX_3070, (8, 6)))
    hints = client.get("/api/status").json()["hints"]
    assert hints["tight_card"] == EIGHT_GB
    if "refine_tight" in unknown:
        assert hints["refine_tight"] == f"{unknown['refine_tight']} {EIGHT_GB}"
    monkeypatch.setattr(card_facts, "card", lambda device="cuda": card_facts.Card(RTX_5080, (12, 0)))
    assert client.get("/api/status").json()["hints"] == unknown
