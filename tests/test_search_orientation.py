"""Orientation values share the existing size filter and strict query scopes."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sqlite3
import subprocess

import pytest
from PIL import Image

from backend import gallery_reports, main, saved_searches, search
from backend.schemas import GenerateRequest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_existing_frontend_search_helpers_accept_values_without_new_keys():
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({ assert });
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext(`
  assert.equal(searchKey('SIZE:PoRtRaIt'), 'size');
  assert.equal(mergeSearch('harbor size:portrait seed:7', ['size:512x768']),
    'harbor size:512x768 seed:7');
  assert.equal(mergePromptSearch('harbor size:portrait seed:7', ['moon']),
    'size:portrait seed:7 moon');
  assert.deepEqual(promptSearchWords('night size:portrait sky'), ['night', 'sky']);
  assert.deepEqual(factSearchTerms({ width: 512, height: 768,
    created_at: '2026-10-28T12:00:00Z' }).Size, ['size:512x768']);
`, context);
'''
    done = subprocess.run(
        ["node", "-e", script, str(ROOT / "web/gallery-search.js")],
        capture_output=True, text=True, timeout=15,
    )
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize(("value", "expected"), [
    ("portrait", [1, 2, 7]), ("PoRtRaIt", [1, 2, 7]),
    ("landscape", [3, 4, 8]), ("LANDSCAPE", [3, 4, 8]),
    ("square", [5, 6, 9]), ("SQUARE", [5, 6, 9]),
])
def test_orientation_compares_stored_integer_dimensions_across_sizes(value, expected):
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, width INTEGER, height INTEGER)")
        connection.executemany("INSERT INTO images VALUES (?, ?, ?)", [
            (1, 256, 384), (2, 512, 768), (3, 384, 256), (4, 768, 512),
            (5, 256, 256), (6, 512, 512),
            # Adjacent SQLite integers must stay distinct: floating aspect
            # ratios or products could lose the distinction at this size.
            (7, 2**63 - 2, 2**63 - 1), (8, 2**63 - 1, 2**63 - 2),
            (9, 2**63 - 1, 2**63 - 1),
        ])
        clauses, parameters = search.strict_scope("SIZE:" + value)
        rows = connection.execute(
            "SELECT id FROM images WHERE " + " AND ".join(clauses) + " ORDER BY id", parameters,
        ).fetchall()
        assert rows == [(image_id,) for image_id in expected]
    finally:
        connection.close()


@pytest.mark.parametrize(("query", "expected"), [
    ("size:portrait size:256x384", [1]),
    ("size:portrait size:portrait", [1, 2]),
    ("size:portrait size:landscape", []),
    ("size:square size:256x384", []),
    ("size:landscape seed:>=3", [3]),
])
def test_orientation_terms_keep_existing_and_composition(query, expected):
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, seed INTEGER, width INTEGER, height INTEGER)")
        connection.executemany("INSERT INTO images VALUES (?, ?, ?, ?)", [
            (1, 1, 256, 384), (2, 2, 512, 768), (3, 3, 384, 256), (4, 4, 256, 256),
        ])
        clauses, parameters = search.strict_scope(query)
        assert connection.execute(
            "SELECT id FROM images WHERE " + " AND ".join(clauses) + " ORDER BY id", parameters,
        ).fetchall() == [(image_id,) for image_id in expected]
    finally:
        connection.close()


@pytest.mark.parametrize("raw", [
    "large", "portrait-ish", "vertical", "horizontal", "!portrait", ">=portrait",
    '"portrait"', "portrait-landscape", "portrait..landscape", "portraitxlandscape",
    "1024x", "x768", "1024.5x768", "-1024x768", "1024x768x1",
])
def test_other_invalid_size_values_keep_the_existing_named_refusal(raw):
    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        search.filter_clause("size", raw)


@pytest.mark.parametrize(("raw", "expected"), [
    ("1024x768", [1024, 768]), ("1024X768", [1024, 768]),
    ("1024\u00d7768", [1024, 768]), ("001024x00768", [1024, 768]),
    ("9223372036854775807x1", [2**63 - 1, 1]),
])
def test_exact_size_spelling_and_integer_bounds_are_preserved(raw, expected):
    clause, values = search.filter_clause("size", raw)
    assert values == expected
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE images (width INTEGER, height INTEGER)")
        connection.execute("INSERT INTO images VALUES (?, ?)", expected)
        assert connection.execute("SELECT COUNT(*) FROM images WHERE " + clause, values).fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.parametrize("query", ['"size:portrait"', "orientation:portrait"])
def test_unknown_or_prefixed_keys_still_remain_prompt_text(query):
    clauses, values, text = search.split_search(query)
    assert clauses == [] and values == [] and text == query
    with pytest.raises(ValueError, match="free text"):
        search.strict_scope(query)


def test_an_excluded_orientation_is_the_complement_in_strict_scopes_too():
    # Until GAL#21, `-size:portrait` was unknown-key prompt text; it is now the
    # orientation filter negated (docs/plan-search-exclusion.md).
    assert search.strict_scope("-size:portrait") == (["COALESCE((width < height), 0) = 0"], [])


def test_orientation_keeps_prompt_text_out_of_strict_scopes():
    clauses, _values, text = search.split_search("harbor size:portrait")
    assert clauses and text == "harbor"
    with pytest.raises(ValueError, match="free text"):
        search.strict_scope("harbor size:portrait")


@pytest.fixture
def cold_queries_only(client, monkeypatch):
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("model-or-queue")
        pytest.fail("an orientation query entered model loading, generation or queue submission")

    monkeypatch.setattr(main, "ensure_loaded", forbidden)
    monkeypatch.setattr(main.backend, "load", forbidden)
    monkeypatch.setattr(main.backend, "generate", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    monkeypatch.setattr(main.jobs, "submit", forbidden)
    yield attempts
    assert attempts == []
    assert client.get("/api/jobs").json() == []


def _keep(store, label, width, height, seed, *, model=None, favorite=False, binned=False, refine=None):
    fields = {"prompt": label, "width": width, "height": height}
    if refine is not None:
        fields.update(refine=refine, strength=0.5)
    with Image.new("RGB", (width, height), (seed * 3, 40, 90)) as image:
        record = store.save_image(
            image, GenerateRequest(**fields), seed=seed, duration_ms=1,
            model_id=model or store.settings.model_id, run_id=f"orientation-{seed}",
        )
    if favorite or binned:
        with store._session() as connection:
            connection.execute(
                "UPDATE images SET favorite = ?, binned_at = ? WHERE id = ?",
                (int(favorite), "2026-10-28T12:00:00+00:00" if binned else None, record.id),
            )
    return record


@pytest.fixture
def owned_rows(store):
    other = store.settings.model_id + "/other"
    return {
        "p-small": _keep(store, 'harbor" portrait source', 256, 384, 11),
        "p-large": _keep(store, "harbor portrait keeper", 512, 768, 12, favorite=True),
        "l-small": _keep(store, "harbor landscape", 384, 256, 21),
        "l-large": _keep(store, "harbor wide", 768, 512, 22),
        "s-small": _keep(store, "harbor square", 256, 256, 31),
        "s-large": _keep(store, "harbor square keeper", 512, 512, 32, favorite=True),
        "other-p": _keep(store, "harbor other portrait", 384, 512, 41, model=other),
        "bin-p": _keep(store, "harbor binned portrait", 512, 768, 51, binned=True),
        "bin-l": _keep(store, "harbor binned landscape", 768, 512, 52, model=other, binned=True),
    }


def _state(store):
    with store._session() as connection:
        rows = [tuple(row) for row in connection.execute("SELECT * FROM images ORDER BY id").fetchall()]
    pngs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in store.settings.outputs_dir.glob("*.png")}
    return rows, pngs


@pytest.mark.parametrize(("orientation", "live", "binned", "favorites"), [
    ("portrait", ["p-small", "p-large", "other-p"], ["bin-p"], 1),
    ("landscape", ["l-small", "l-large"], ["bin-l"], 0),
    ("square", ["s-small", "s-large"], [], 1),
])
def test_gallery_bin_and_strict_preview_share_orientation_without_writes(
    client, store, owned_rows, cold_queries_only, orientation, live, binned, favorites,
):
    before = _state(store)
    params = {"search": "size:" + orientation, "all_models": True}
    gallery = client.get("/api/gallery", params=params)
    assert gallery.status_code == 200, gallery.text
    assert gallery.json()["total"] == len(live)
    assert {row["id"] for row in gallery.json()["items"]} == {owned_rows[key].id for key in live}
    recycle = client.get("/api/gallery/bin", params=params)
    assert recycle.status_code == 200, recycle.text
    assert recycle.json()["total"] == len(binned)
    assert {row["id"] for row in recycle.json()["items"]} == {owned_rows[key].id for key in binned}
    preview = client.get("/api/gallery/unstarred", params=params)
    assert preview.status_code == 200, preview.text
    answer = preview.json()
    assert (answer["records"], answer["favorites"], answer["sources_kept"], answer["unstarred"]) == (
        len(live), favorites, 0, len(live) - favorites,
    )
    assert answer["job_active"] is False
    assert _state(store) == before and cold_queries_only == []


def test_gallery_scope_favorites_paging_and_prompt_fallback_retain_orientation(
    client, store, owned_rows, cold_queries_only,
):
    before = _state(store)
    current = client.get("/api/gallery", params={"search": "size:portrait"})
    assert current.status_code == 200, current.text
    assert current.json()["total"] == 2 and current.json()["other_models"] == 1
    assert {row["id"] for row in current.json()["items"]} == {owned_rows[key].id for key in ("p-small", "p-large")}
    starred = client.get("/api/gallery", params={"search": "SIZE:PoRtRaIt", "favorites": True, "all_models": True})
    assert starred.status_code == 200, starred.text
    assert [row["id"] for row in starred.json()["items"]] == [owned_rows["p-large"].id]
    first = client.get("/api/gallery", params={"search": "size:portrait", "all_models": True, "limit": 1})
    assert first.status_code == 200, first.text
    assert first.json()["total"] == 3 and first.json()["has_more"] is True
    assert first.json()["items"][0]["id"] == owned_rows["other-p"].id
    tail = client.get("/api/gallery", params={
        "search": "size:portrait", "all_models": True, "before": owned_rows["other-p"].id, "limit": 10,
    })
    assert tail.status_code == 200, tail.text
    assert [row["id"] for row in tail.json()["items"]] == [owned_rows[key].id for key in ("p-large", "p-small")]
    prompt = client.get("/api/gallery", params={"search": 'harbor" size:portrait', "all_models": True})
    assert prompt.status_code == 200, prompt.text
    assert [row["id"] for row in prompt.json()["items"]] == [owned_rows["p-small"].id]
    assert _state(store) == before and cold_queries_only == []


@pytest.mark.parametrize("route", ["/api/gallery", "/api/gallery/bin", "/api/gallery/unstarred"])
def test_invalid_size_value_keeps_existing_read_validation_status(client, cold_queries_only, route):
    answer = client.get(route, params={"search": "size:portrait-ish", "all_models": True})
    assert answer.status_code == 422, answer.text
    assert "WIDTHxHEIGHT" in answer.json()["detail"]
    assert cold_queries_only == []


def test_owned_orientation_binning_keeps_stars_sources_other_shapes_and_pngs(
    client, store, owned_rows, cold_queries_only,
):
    extra = _keep(store, "harbor new portrait", 384, 512, 61)
    _keep(store, "harbor starred landscape pass", 384, 256, 62,
          favorite=True, refine=owned_rows["p-small"].id)
    before_rows, before_pngs = _state(store)
    params = {"search": "size:portrait", "all_models": False}
    preview = client.get("/api/gallery/unstarred", params=params)
    assert preview.status_code == 200, preview.text
    answer = preview.json()
    assert (answer["records"], answer["favorites"], answer["sources_kept"], answer["unstarred"]) == (3, 1, 1, 1)
    assert answer["sources"] == [owned_rows["p-small"].id]
    stale = client.delete("/api/gallery/unstarred", params={**params, "expected": 2})
    assert stale.status_code == 409, stale.text
    assert _state(store) == (before_rows, before_pngs)
    committed = client.delete("/api/gallery/unstarred", params={**params, "expected": 1})
    assert committed.status_code == 200, committed.text
    assert committed.json() == {"binned": [extra.id], "deleted": [extra.id], "files": 0}
    after_rows, after_pngs = _state(store)
    assert after_pngs == before_pngs
    assert len(after_rows) == len(before_rows)
    # Only the known extra portrait has moved; every other record is identical.
    before = {row[0]: row for row in before_rows}
    after = {row[0]: row for row in after_rows}
    assert {key: row for key, row in before.items() if key != extra.id} == {
        key: row for key, row in after.items() if key != extra.id
    }
    with store._session() as connection:
        assert connection.execute("SELECT binned_at FROM images WHERE id = ?", (extra.id,)).fetchone()[0] is not None
    assert cold_queries_only == []


def test_saved_orientation_query_retains_raw_text_ticks_and_redacted_validation(
    client, store, owned_rows, cold_queries_only,
):
    body = {"name": "Portrait keepers", "query": "  SIZE:PoRtRaIt seed:1-50 \n", "favorites": True, "all_models": False}
    created = client.post("/api/gallery/saved-searches", json=body)
    assert created.status_code == 201, created.text
    item = created.json()["items"][0]
    assert {key: item[key] for key in body} == body
    assert client.get("/api/gallery/saved-searches").json()["items"][0] == item
    applied = client.get("/api/gallery", params={
        "search": item["query"], "favorites": item["favorites"], "all_models": item["all_models"],
    })
    assert applied.status_code == 200, applied.text
    assert [row["id"] for row in applied.json()["items"]] == [owned_rows["p-large"].id]
    path = saved_searches.path_for(store.settings.db_path)
    before = path.read_bytes()
    refused = client.post("/api/gallery/saved-searches", json={
        **body, "name": "Invalid portrait", "query": "private-orientation-query size:portrait-ish",
    })
    assert refused.status_code == 422, refused.text
    assert refused.json() == {"detail": "Saved search fields are invalid. Check the name, query, ticks and revision."}
    assert "private-orientation-query" not in refused.text and path.read_bytes() == before
    assert cold_queries_only == []


def test_keeper_rates_share_orientation_scope_without_binned_or_other_model_rows(
    store, owned_rows, cold_queries_only,
):
    before = _state(store)
    rows = gallery_reports.keep_rates(search="size:portrait model:=" + store.settings.model_id)
    assert sum(row["made"] for row in rows) == 2
    assert sum(row["starred"] for row in rows) == 1
    assert {row["size"] for row in rows} == {"256x384", "512x768"}
    assert _state(store) == before and cold_queries_only == []
