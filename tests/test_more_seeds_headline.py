"""Server-owned More seeds headlines include every allocated seed at wrap."""
from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from backend import main, more_seeds
from backend.schemas import GenerateRequest
from backend.seeds import MAX_SEED
from tests.test_more_seeds import record


def _queued(groups):
    batches = []
    for index, seeds in enumerate(groups, 1):
        template = record(index, 0, prompt=f"headline cell {index}")
        batches.append(more_seeds.Batch(
            key=more_seeds.batch_key(template), picked=[template], seeds=list(seeds),
        ))
    return batches


@pytest.mark.parametrize(("groups", "count", "favorites", "expected"), [
    ([[MAX_SEED, 0, 1]], 3, False,
     f"3 more of its one cell = 3 images \u00b7 seeds 0\u2013{MAX_SEED}"),
    ([[MAX_SEED, 0, 1], [101, 102, 103]], 3, False,
     f"3 more of each of 2 cells = 6 images \u00b7 seeds 0\u2013{MAX_SEED}"),
    ([[MAX_SEED - 1, MAX_SEED, 0]], 3, True,
     f"3 more of its one cell = 3 images \u00b7 seeds 0\u2013{MAX_SEED} \u00b7 Starred ignored"),
    ([[10, 12], [20, 25]], 2, True,
     "2 more of each of 2 cells = 4 images \u00b7 seeds 10\u201325 \u00b7 Starred ignored"),
    ([[42]], 1, False, "1 more of its one cell = 1 image \u00b7 seed 42"),
    ([[7], [7]], 1, True,
     "1 more of each of 2 cells = 2 images \u00b7 seed 7 \u00b7 Starred ignored"),
])
def test_headline_uses_all_actual_seed_extrema_and_preserves_allocations(groups, count, favorites, expected):
    batches = _queued(groups)
    before = [list(batch.seeds) for batch in batches]
    assert more_seeds.run_headline(batches, count, favorites) == expected
    assert [batch.seeds for batch in batches] == before


def test_no_eligible_batches_keep_the_existing_absent_headline():
    assert more_seeds.run_headline([], 4, True) is None


@pytest.fixture
def cold_preview(client, monkeypatch):
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("load-generation-or-admission")
        pytest.fail("a More seeds headline preview entered loading, generation or queue admission")

    monkeypatch.setattr(main, "ensure_loaded", forbidden)
    monkeypatch.setattr(main.backend, "load", forbidden)
    monkeypatch.setattr(main.backend, "generate", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    monkeypatch.setattr(main.jobs, "submit", forbidden)
    monkeypatch.setattr(main.jobs, "submit_cells", forbidden)
    yield attempts
    assert attempts == []
    assert client.get("/api/jobs").json() == []


def _plant(store, seed, run, *, model=None, batch=1):
    with Image.new("RGB", (256, 256), (20, 40, 90)) as image:
        return store.save_image(
            image, GenerateRequest(prompt="headline harbor", width=256, height=256, steps=2, guidance_scale=1.5),
            seed=seed, duration_ms=1, model_id=model or store.settings.model_id,
            run_id=run, batch=batch,
        )


def _state(store):
    with store._session() as connection:
        rows = [tuple(row) for row in connection.execute("SELECT * FROM images ORDER BY id").fetchall()]
    pngs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in store.settings.outputs_dir.glob("*.png")}
    return rows, pngs


def test_owned_python_api_preview_wrap_has_honest_headline_and_unchanged_seed_plan(
    client, store, cold_preview,
):
    source = _plant(store, MAX_SEED - 1, "headline-wrap")
    before = _state(store)
    response = client.post("/api/more-seeds/preview", json={"run": "headline-wrap", "count": 3, "favorites": False})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["ids"] == [source.id]
    assert (answer["queued"], answer["batches"], answer["jobs"], answer["counts"]) == (3, 1, 1, {"queued": 1})
    result = answer["results"][0]
    assert (result["first"], result["last"], result["skipped"], result["queued"]) == (MAX_SEED, 1, 0, 3)
    assert result["seed_low"] == result["seed_high"] == MAX_SEED - 1
    assert answer["headline"] == f"3 more of its one cell = 3 images \u00b7 seeds 0\u2013{MAX_SEED}"
    assert answer["intent"]
    assert _state(store) == before and cold_preview == []


def test_ordinary_preview_keeps_foreign_filtering_and_unrecorded_batch_fallback(
    client, store, cold_preview,
):
    current = _plant(store, 10, "headline-mixed", batch=None)
    foreign = _plant(store, 999, "headline-mixed", model=store.settings.model_id + "/other", batch=None)
    before = _state(store)
    response = client.post("/api/more-seeds/preview", json={"run": "headline-mixed", "count": 2, "batch": 3, "favorites": True})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["ids"] == [current.id, foreign.id]
    assert (answer["queued"], answer["batches"], answer["jobs"]) == (2, 1, 1)
    assert answer["counts"] == {"queued": 1, "foreign": 1}
    results = {result["id"]: result for result in answer["results"]}
    assert (results[current.id]["first"], results[current.id]["last"], results[current.id]["skipped"]) == (11, 12, 0)
    assert (results[current.id]["batch"], results[current.id]["recorded"]) == (3, False)
    assert results[foreign.id]["queued"] == 0 and results[foreign.id]["first"] is None
    assert (answer["count_default"], answer["unrecorded"]) == (None, 1)
    assert answer["headline"] == "2 more of its one cell = 2 images \u00b7 seeds 11\u201312 \u00b7 Starred ignored"
    assert answer["intent"]
    assert _state(store) == before and cold_preview == []


def test_all_foreign_preview_keeps_headline_and_intent_absent(client, store, cold_preview):
    foreign = _plant(store, 999, "headline-none", model=store.settings.model_id + "/other")
    before = _state(store)
    response = client.post("/api/more-seeds/preview", json={"run": "headline-none", "count": 2})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["ids"] == [foreign.id] and answer["counts"] == {"foreign": 1}
    assert (answer["queued"], answer["batches"], answer["jobs"]) == (0, 0, 0)
    assert answer["headline"] is None and answer["intent"] is None
    assert _state(store) == before and cold_preview == []
