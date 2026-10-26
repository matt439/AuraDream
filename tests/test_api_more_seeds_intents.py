"""More seeds acknowledges one request intent even when its reply is lost."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend import main, model_selection, storage
from backend.jobs import RunPersistenceFailed
from backend.more_seeds_intents import IntentLedger
from tests.api_helpers import stream_job
from tests.test_api_more_seeds import plant, preview, queue


def secure(client, **body):
    answer = preview(client, **body)
    assert answer["intent"]
    return {**body, "expected": answer["ids"], "intent": answer["intent"]}


def test_replay_after_completion_and_source_deletion_returns_the_same_receipt(client, fake, store):
    source = plant(store, "a lighthouse", seed=100, run_id="source", batch=2)
    body = secure(client, ids=[source.id], count=2)
    accepted = queue(client, **body)
    final = stream_job(client, accepted["job_id"])[-1]
    assert [image["seed"] for image in final["images"]] == [101, 102]
    assert accepted["warning"] is None and accepted["not_queued"] == 0
    # Neither completion nor a disappearing source turns a retry into more work.
    response = client.delete(f"/api/images/{source.id}")
    assert response.status_code == 200
    assert queue(client, **body) == accepted
    assert len(store.retained_run_jobs()) == 1
    assert len(fake.requests) == 1


def test_different_preview_tokens_allocate_distinct_extensions(client, fake, store):
    source = plant(store, "a lighthouse", seed=100, run_id="source", batch=2)
    first = secure(client, ids=[source.id], count=2)
    second = secure(client, ids=[source.id], count=2)
    assert first["intent"] != second["intent"]
    a, b = queue(client, **first), queue(client, **second)
    assert a["job_id"] != b["job_id"]
    made = [stream_job(client, item["job_id"])[-1]["images"] for item in (a, b)]
    assert [[image["seed"] for image in group] for group in made] == [[101, 102], [103, 104]]


@pytest.mark.parametrize("changed", [
    {"count": 3}, {"batch": 3}, {"expected": []}, {"search": "lighthouse"}, {"ids": [9999]},
])
def test_an_intent_refuses_changed_choices_before_any_admission(client, fake, store, changed):
    source = plant(store, "a lighthouse", seed=100, run_id="source", batch=2)
    body = secure(client, ids=[source.id], count=2)
    refused = client.post("/api/more-seeds", json={**body, **changed})
    assert refused.status_code == 409
    assert not main.jobs.active() and store.retained_run_jobs() == [] and fake.requests == []
    # A mismatched request does not consume the original valid choice.
    accepted = queue(client, **body)
    assert stream_job(client, accepted["job_id"])[-1]["status"] == "done"


def test_unknown_expired_manager_reset_and_model_revision_never_allocate(client, fake, store, monkeypatch):
    source = plant(store, "a lighthouse", seed=100, run_id="source", batch=2)
    body = secure(client, ids=[source.id], count=2)
    unknown = client.post("/api/more-seeds", json={**body, "intent": "unknown"})
    assert unknown.status_code == 409
    original = main.jobs.more_seeds_intents
    monkeypatch.setattr(main.jobs, "more_seeds_intents", IntentLedger())
    assert client.post("/api/more-seeds", json=body).status_code == 409
    monkeypatch.setattr(main.jobs, "more_seeds_intents", original)
    monkeypatch.setattr(model_selection, "revision", lambda: 99)
    assert client.post("/api/more-seeds", json=body).status_code == 409
    monkeypatch.setattr(model_selection, "revision", lambda: 0)
    then = original._clock()
    monkeypatch.setattr(original, "_clock", lambda: then + 901)
    assert client.post("/api/more-seeds", json=body).status_code == 409
    assert not main.jobs.active() and store.retained_run_jobs() == [] and fake.requests == []


def test_concurrent_same_token_waits_through_persistence_and_queues_once(client, fake, store, monkeypatch):
    source = plant(store, "a lighthouse", seed=100, run_id="source", batch=2)
    body = secure(client, ids=[source.id], count=2)
    entered, release = threading.Event(), threading.Event()
    original = storage.retain_run_job
    calls = []

    def held(definition, cells):
        calls.append(definition["job_id"])
        entered.set()
        assert release.wait(5)
        return original(definition, cells)

    monkeypatch.setattr(storage, "retain_run_job", held)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, "/api/more-seeds", json=body)
        try:
            assert entered.wait(5), "the first intent never reached persistence"
            second = pool.submit(client.post, "/api/more-seeds", json=body)
            assert main.jobs.busy() and main.jobs.more_seeds_intents.pending
        finally:
            release.set()
        a, b = first.result(timeout=5), second.result(timeout=5)
    assert a.status_code == b.status_code == 200
    assert a.json() == b.json()
    assert len(calls) == 1 and len(store.retained_run_jobs()) == 1
    assert stream_job(client, a.json()["job_id"])[-1]["status"] == "done"


@pytest.mark.parametrize("tokenized", [True, False], ids=["receipt", "legacy"])
def test_partial_acceptance_names_the_remainder_and_keeps_its_retry_contract(
    client, fake, store, monkeypatch, tokenized,
):
    a = plant(store, "a lighthouse", seed=100, run_id="one", batch=2)
    b = plant(store, "a harbour", seed=200, run_id="two", width=256, height=384, batch=2)
    body = {"ids": [a.id, b.id], "count": 2}
    if tokenized:
        body = secure(client, **body)
    original = main.jobs.submit_cells
    calls = 0

    async def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RunPersistenceFailed("the disk is full")
        return await original(*args, **kwargs)

    monkeypatch.setattr(main.jobs, "submit_cells", fail_second)
    accepted = queue(client, **body)
    assert (accepted["queued"], accepted["batches"], accepted["more"]) == (2, 1, [])
    assert (accepted["not_queued"], accepted["not_queued_batches"]) == (2, 1)
    assert f"#{b.id} at 256x384" in accepted["warning"] and "disk is full" in accepted["warning"]
    final = stream_job(client, accepted["job_id"])[-1]
    assert [image["seed"] for image in final["images"]] == [101, 102]
    if tokenized:
        assert "will not add the remainder" in accepted["warning"]
        assert queue(client, **body) == accepted
        assert calls == 2 and len(store.retained_run_jobs()) == 1
    else:
        assert "another request adds new seeds" in accepted["warning"]
        again = queue(client, **body)
        assert (again["queued"], again["batches"]) == (4, 2)
        assert again["job_id"] != accepted["job_id"]
        for job in [again, *again["more"]]:
            assert stream_job(client, job["job_id"])[-1]["status"] == "done"
        assert calls == 4 and len(store.retained_run_jobs()) == 3


def test_zero_admission_failure_replays_its_error_instead_of_retrying_allocation(client, fake, store, monkeypatch):
    source = plant(store, "a lighthouse", seed=100, run_id="source", batch=2)
    body = secure(client, ids=[source.id], count=2)
    calls = 0

    async def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RunPersistenceFailed("the disk is full")

    monkeypatch.setattr(main.jobs, "submit_cells", fail)
    a = client.post("/api/more-seeds", json=body)
    b = client.post("/api/more-seeds", json=body)
    assert a.status_code == b.status_code == 503 and a.json() == b.json()
    assert calls == 1 and not main.jobs.active() and store.retained_run_jobs() == []


def test_normalized_id_order_replays_and_legacy_requests_still_extend(client, fake, store):
    a = plant(store, "a lighthouse", seed=100, run_id="source", batch=2)
    b = plant(store, "a lighthouse", seed=101, run_id="source", batch=2)
    body = secure(client, ids=[a.id, b.id], count=1)
    accepted = queue(client, **body)
    reordered = {**body, "ids": [b.id, a.id, a.id], "expected": list(reversed(body["expected"]))}
    assert queue(client, **reordered) == accepted
    assert stream_job(client, accepted["job_id"])[-1]["status"] == "done"
    legacy = queue(client, ids=[a.id], count=1)
    assert legacy["job_id"] != accepted["job_id"]
    assert stream_job(client, legacy["job_id"])[-1]["images"][0]["seed"] == 103
