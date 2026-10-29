"""Gallery pagination rejects SQLite-binding overflow before storage.

Actual read-only TestClient/SQLite checks on owned scratch records; no browser,
real gallery, model loading, generation, queue submission or lifecycle request.
"""
from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import main
from backend.schemas import GenerateRequest
from backend.search import SQLITE_INT

ROUTES = ("/api/gallery", "/api/gallery/bin")
PARAMETERS = ("offset", "before")
MAXIMUM = SQLITE_INT[1]


@pytest.fixture(autouse=True)
def cold_reads_only(client, monkeypatch):
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("model-or-queue")
        pytest.fail("gallery paging entered model loading, generation or queue admission")

    monkeypatch.setattr(main, "ensure_loaded", forbidden)
    monkeypatch.setattr(main.backend, "load", forbidden)
    monkeypatch.setattr(main.backend, "generate", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    monkeypatch.setattr(main.jobs, "submit", forbidden)
    monkeypatch.setattr(main.jobs, "submit_cells", forbidden)
    yield
    assert attempts == []
    assert client.get("/api/jobs").json() == []


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("parameter", PARAMETERS)
def test_paging_past_sqlite_max_is_422_before_storage(client, store, paging_rows, monkeypatch, route, parameter):
    calls, binding_errors = [], []
    original = store.list_images_at_head

    def observed_storage(**kwargs):
        calls.append(kwargs)
        try:
            return original(**kwargs)  # Real SQL on the owned fixture database.
        except OverflowError:
            binding_errors.append("OverflowError")
            raise

    before = _snapshot(store, paging_rows)
    monkeypatch.setattr(store, "list_images_at_head", observed_storage)
    # Reuse the existing fixture's running app/allowed host without entering
    # another lifespan. The original route exposes its real SQLite failure
    # as HTTP 500 rather than escaping through TestClient's exception mode.
    unraised = TestClient(main.app, base_url=str(client.base_url), raise_server_exceptions=False)
    try:
        response = unraised.get(route, params={parameter: str(MAXIMUM + 1)})
    finally:
        unraised.close()
    assert _snapshot(store, paging_rows) == before
    assert response.status_code == 422, (
        f"HTTP {response.status_code}; forwarded binding errors={binding_errors}; body={response.text}"
    )
    errors = response.json()["detail"]
    assert any(error["loc"] == ["query", parameter]
               and error["type"] == "less_than_equal"
               and error["ctx"]["le"] == MAXIMUM for error in errors)
    assert calls == [] and binding_errors == []


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(("parameter", "bad", "lower"), [("offset", -1, 0), ("before", 0, 1)])
def test_existing_paging_lower_bounds_still_refuse_before_storage(client, store, monkeypatch, route, parameter, bad, lower):
    def forbidden_storage(**_kwargs):
        pytest.fail("a paging value below its existing minimum reached storage")

    monkeypatch.setattr(store, "list_images_at_head", forbidden_storage)
    response = client.get(route, params={parameter: bad})
    assert response.status_code == 422, response.text
    assert any(error["loc"] == ["query", parameter]
               and error["type"] == "greater_than_equal"
               and error["ctx"]["ge"] == lower for error in response.json()["detail"])


@pytest.fixture
def paging_rows(store):
    rows = {}
    for name, seed, model, prompt in [
        ("mine_old", 1, store.settings.model_id, "page bounds query"),
        ("mine_new", 2, store.settings.model_id, "page bounds query"),
        ("theirs_old", 3, "paging/other", "page bounds query"),
        ("theirs_new", 4, "paging/other", "other model subject"),
        ("bin_mine", 5, store.settings.model_id, "page bounds query"),
        ("bin_theirs", 6, "paging/other", "page bounds query"),
    ]:
        with Image.new("RGB", (256, 256), (seed, 20, 30)) as image:
            rows[name] = store.save_image(
                image, GenerateRequest(prompt=prompt, width=256, height=256, steps=8),
                seed=seed, duration_ms=1, model_id=model, job_id="paging-bounds", run_id="paging-bounds",
            )
    # Fixture-only state on exactly the rows just created, without lifecycle
    # endpoints or arbitrary user-data IDs. Every test operation after this is GET.
    with store._write_lock, store._session() as connection:
        connection.executemany("UPDATE images SET favorite = 1 WHERE id = ?", [
            (rows["mine_new"].id,), (rows["theirs_new"].id,), (rows["bin_mine"].id,),
        ])
        connection.executemany("UPDATE images SET binned_at = ? WHERE id = ?", [
            ("2026-10-28T00:00:00+00:00", rows["bin_mine"].id),
            ("2026-10-28T00:00:00+00:00", rows["bin_theirs"].id),
        ])
    return rows


def _snapshot(store, rows):
    with store._session() as connection:
        records = [tuple(row) for row in connection.execute("SELECT * FROM images ORDER BY id")]
    pngs = {
        row.id: hashlib.sha256((store.settings.outputs_dir / row.filename).read_bytes()).hexdigest()
        for row in rows.values()
    }
    return records, pngs


def _page(client, route, **params):
    response = client.get(route, params=params)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("parameter", PARAMETERS)
def test_sqlite_max_paging_value_remains_valid(client, store, paging_rows, route, parameter):
    before = _snapshot(store, paging_rows)
    page = _page(client, route, **{parameter: str(MAXIMUM)})
    names = ("mine_new", "mine_old") if route == "/api/gallery" else ("bin_theirs", "bin_mine")
    expected = [paging_rows[name].id for name in names] if parameter == "before" else []
    assert [row["id"] for row in page["items"]] == expected
    assert (page["total"], page["has_more"], page["other_models"]) == (2, False, 2 if route == "/api/gallery" else 0)
    assert page["offset"] == (MAXIMUM if parameter == "offset" else 0)
    assert page["newest_image_id"] == paging_rows["bin_theirs"].id
    assert _snapshot(store, paging_rows) == before


@pytest.mark.parametrize("route", ROUTES)
def test_ordinary_pages_counts_and_scopes_are_unchanged(client, store, paging_rows, route):
    before = _snapshot(store, paging_rows)
    names = ("mine_new", "mine_old") if route == "/api/gallery" else ("bin_theirs", "bin_mine")
    newer, older = (paging_rows[name].id for name in names)
    first = _page(client, route, limit=1)
    assert [row["id"] for row in first["items"]] == [newer]
    assert (first["total"], first["has_more"], first["limit"], first["offset"]) == (2, True, 1, 0)
    cursor = _page(client, route, limit=1, before=newer)
    offset = _page(client, route, limit=1, offset=1)
    for second in (cursor, offset):
        assert [row["id"] for row in second["items"]] == [older]
        assert (second["total"], second["has_more"]) == (2, False)
    assert _page(client, route, before=1)["items"] == []
    assert client.get(route, params={"limit": 0}).status_code == 422
    if route == "/api/gallery":
        all_models = _page(client, route, all_models=True)
        assert [row["id"] for row in all_models["items"]] == [paging_rows[name].id for name in (
            "theirs_new", "theirs_old", "mine_new", "mine_old",
        )]
        assert (all_models["total"], all_models["other_models"]) == (4, 0)
        exact = _page(client, route, model="paging/other")
        assert [row["id"] for row in exact["items"]] == [paging_rows["theirs_new"].id, paging_rows["theirs_old"].id]
        assert (exact["total"], exact["other_models"]) == (2, 2)
        starred = _page(client, route, favorites=True)
        assert [row["id"] for row in starred["items"]] == [paging_rows["mine_new"].id]
        assert (starred["total"], starred["other_models"]) == (1, 1)
        searched = _page(client, route, search="seed:2", all_models=True)
        assert [row["id"] for row in searched["items"]] == [paging_rows["mine_new"].id]
        assert searched["total"] == 1
    else:
        assert all(row["binned_at"] is not None for row in first["items"] + cursor["items"])
        searched = _page(client, route, search="seed:5")
        assert [row["id"] for row in searched["items"]] == [paging_rows["bin_mine"].id]
        assert searched["total"] == 1
    assert _snapshot(store, paging_rows) == before
