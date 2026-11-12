"""The grid preview's budget of each resolved prompt (PA#15): Check counts
the `{name}` template, so a value could carry one resolved prompt past an
encoder's budget unseen. The fake tokenizers read one token per word."""
from __future__ import annotations

import time

from backend import grid, grid_budgets, prompt_encoding
from backend.schemas import GenerateRequest

from tests.api_analysis_helpers import ANIMAGINE, FLUX, claim

LONG = " ".join(["word"] * 600)  # past T5-XXL's 511
BASE = " ".join(["word"] * 500)  # inside it, with room for a short value
PAST_CLIP = " ".join(["word"] * 90)  # past CLIP-L's 75, inside T5's


def _preview(client, **body):
    response = client.post("/api/grid/preview", json={"width": 256, "height": 256, **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_value_that_carries_a_prompt_past_the_budget_is_named(client, fake, monkeypatch):
    claim(monkeypatch, fake, FLUX)
    fake.loaded = False
    tail = " ".join(["tail"] * 20)
    budgets = _preview(
        client, prompt=f"{BASE} {{size}} {{ground}}",
        prompt_axis=f"size = big | {tail}\nground = grass | dirt",
    )["budgets"]
    # Cold, as Check is: no load, never the gate.
    assert fake.loads == 0 and not fake.loaded
    assert budgets["distinct"] == 4 and len(budgets["prompts"]) == 4 and budgets["detail"] is None
    first = budgets["prompts"][0]
    assert first["label"] == "size=big, ground=grass" and first["cut"] is False
    assert [(e["name"], e["tokens"], e["budget"]) for e in first["encoders"]] == [
        ("CLIP-L", 502, 75), ("T5-XXL", 502, 511),
    ]
    assert [p["cut"] for p in budgets["prompts"]] == [False, False, True, True]
    t5 = budgets["prompts"][2]["encoders"][1]
    assert t5["name"] == "T5-XXL" and t5["tokens"] == 521 and t5["cut"] is not None
    # The long value is in every cut prompt and in no kept one; the grounds
    # are in both, so neither is named.
    assert budgets["cutting_values"] == {"size": [tail]}


def test_a_pooled_only_cut_is_counted_but_loses_no_text(client, fake, monkeypatch):
    # FLUX.1's CLIP-L feeds one pooled vector while T5 reads the words on.
    claim(monkeypatch, fake, FLUX)
    budgets = _preview(client, prompt="{size} dog", prompt_axis=f"size = big | {PAST_CLIP}")["budgets"]
    clip = budgets["prompts"][1]["encoders"][0]
    assert clip["name"] == "CLIP-L" and clip["pooled_only"] and clip["cut"] is not None
    assert [p["cut"] for p in budgets["prompts"]] == [False, False]
    assert budgets["cutting_values"] == {}


def test_every_prompt_cut_names_no_value(client, fake, monkeypatch):
    # Then the template is what is long; no value tips it over.
    claim(monkeypatch, fake, ANIMAGINE)
    budgets = _preview(
        client, prompt=f"{PAST_CLIP}, {{size}}", prompt_axis="size = big | small",
    )["budgets"]
    assert [p["cut"] for p in budgets["prompts"]] == [True, True]
    assert budgets["cutting_values"] == {}


def test_a_grid_with_nothing_substituted_has_no_budgets(client, fake, monkeypatch):
    # Check's own count is that prompt's.
    claim(monkeypatch, fake, FLUX)
    assert _preview(client, prompt="a dog", steps_axis="4,8")["budgets"] is None


def test_random_mode_and_variants_are_measured_by_their_resolved_text(client, fake, monkeypatch):
    claim(monkeypatch, fake, FLUX)
    random = _preview(
        client, prompt="a {size} fox", prompt_axis="size = small | large", prompt_mode="random",
        seed=42, seeds=6,
    )["budgets"]
    assert {p["label"] for p in random["prompts"]} == {"size=small", "size=large"}
    assert random["distinct"] == 2
    variants = _preview(client, prompt="a fox", prompt_variants=[
        {"label": "original", "prompt": "a fox"}, {"label": "long", "prompt": LONG},
    ])["budgets"]
    assert [(p["label"], p["cut"]) for p in variants["prompts"]] == [("original", False), ("long", True)]
    # A variant's label is not a variable's value.
    assert variants["cutting_values"] == {}


def _cells(block: str, negative: str | None = None) -> list[grid.Cell]:
    request = GenerateRequest(prompt="{size} dog", negative_prompt=negative, width=256, height=256, seed=1)
    return grid.expand(request, prompt=block, steps="4,8")


def test_distinct_prompts_are_measured_once_and_bounded(monkeypatch):
    seen = []

    def measure(model_id, prompt, negative, *, wait=None):
        seen.append((prompt, negative))
        return [], [], None

    monkeypatch.setattr(prompt_encoding, "measure", measure)
    cells = _cells("size = a | b | c")  # three prompts, two steps each
    budgets = grid_budgets.measure_cells(FLUX, cells, lambda r: r, limit=2)
    assert seen == [("a dog", None), ("b dog", None)]
    assert (budgets.distinct, len(budgets.prompts)) == (3, 2)


def test_the_negative_is_measured_as_applied(monkeypatch):
    seen = []

    def measure(model_id, prompt, negative, *, wait=None):
        seen.append(negative)
        return [], [], None

    monkeypatch.setattr(prompt_encoding, "measure", measure)
    cells = _cells("size = a | b", negative="blurry")
    grid_budgets.measure_cells(FLUX, cells, lambda r: r)
    assert seen == ["blurry", "blurry"]
    seen.clear()
    grid_budgets.measure_cells(FLUX, cells, lambda r: r.model_copy(update={"negative_prompt": None}))
    assert seen == [None, None]


def test_one_wait_bounds_the_whole_preview(monkeypatch):
    # A stalled tokenizer costs the preview its counts once, not per prompt.
    waits = []

    def measure(model_id, prompt, negative, *, wait=None):
        waits.append(wait)
        time.sleep(0.05)
        return [], [], "T5-XXL still loading"

    monkeypatch.setattr(prompt_encoding, "measure", measure)
    budgets = grid_budgets.measure_cells(FLUX, _cells("size = a | b | c"), lambda r: r, wait=0.08)
    assert waits[0] <= 0.08 and waits[-1] < 0.05 and all(w >= 0 for w in waits)
    assert budgets.detail == "T5-XXL still loading"


def test_a_checkpoint_without_an_encoder_table_entry_has_no_budgets():
    assert grid_budgets.measure_cells("someone/unlisted", _cells("size = a | b"), lambda r: r) is None
