"""Exact actual-MP comparison scopes, without floating products or ambient state."""
from __future__ import annotations

import sqlite3
from decimal import FloatOperation, Inexact, InvalidOperation, Rounded, localcontext

import pytest
from PIL import Image

from backend import gallery_reports, saved_searches, search
from backend.schemas import GenerateRequest
from tests.gallery_helpers import cold_queries_only, gallery_state  # noqa: F401  (fixtures by name)

MAXIMUM = 2**63 - 1
MAXIMUM_MP = "9223372036854.775807"
POSITIVE_IDS = list(range(1, 11))
SMALL = [1, 2, 5]
THROUGH_MILLION = [1, 2, 3, 5]
FROM_MILLION = [3, 4, 6, 7, 8, 9, 10]
ABOVE_MILLION = [4, 6, 7, 8, 9, 10]


def _selected(query):
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, width INTEGER, height INTEGER)")
        connection.executemany("INSERT INTO images VALUES (?, ?, ?)", [
            (1, 1, 1), (2, 999999, 1), (3, 1000, 1000), (4, 1000001, 1),
            (5, 1536, 640), (6, 1024, 1024),
            (7, MAXIMUM - 1, 1), (8, MAXIMUM, 1),
            (9, MAXIMUM, 2), (10, MAXIMUM, MAXIMUM),
            # INTEGER affinity is not a STRICT constraint. These must not
            # change division into REAL arithmetic or match an area query.
            (11, 0, 1), (12, 1, 0), (13, -1, 100), (14, 100, -1),
            (15, -1, -1), (16, None, 100), (17, 100, None),
            (18, 1000.5, 1000), (19, 1000, 1000.5),
            (20, "not-an-integer", 1000), (21, 1000, "not-an-integer"),
            (22, b"1000", 1000),
        ])
        clauses, parameters = search.strict_scope(query)
        return [row[0] for row in connection.execute(
            "SELECT id FROM images WHERE " + " AND ".join(clauses) + " ORDER BY id", parameters,
        ).fetchall()]
    finally:
        connection.close()


@pytest.mark.parametrize(("operator", "expected"), [
    (">", ABOVE_MILLION), (">=", FROM_MILLION), ("<", SMALL), ("<=", THROUGH_MILLION),
])
def test_area_uses_actual_unrounded_million_pixel_boundaries(operator, expected):
    assert _selected("size:" + operator + "1mp") == expected


_NEAR_CASES = [
    (">", "1", ABOVE_MILLION), (">=", "1", FROM_MILLION),
    ("<", "1", SMALL), ("<=", "1", THROUGH_MILLION),
    (">", "1." + "0" * 256 + "1", ABOVE_MILLION),
    (">=", "1." + "0" * 256 + "1", ABOVE_MILLION),
    ("<", "1." + "0" * 256 + "1", THROUGH_MILLION),
    ("<=", "1." + "0" * 256 + "1", THROUGH_MILLION),
    (">", "0." + "9" * 257, FROM_MILLION),
    (">=", "0." + "9" * 257, FROM_MILLION),
    ("<", "0." + "9" * 257, SMALL),
    ("<=", "0." + "9" * 257, SMALL),
    (">", "9223372036854.775806" + "9" * 256, [8, 9, 10]),
    (">=", "9223372036854.775806" + "9" * 256, [8, 9, 10]),
    ("<", "9223372036854.775806" + "9" * 256, list(range(1, 8))),
    ("<=", "9223372036854.775806" + "9" * 256, list(range(1, 8))),
]


def _context_state(context):
    return (
        context.prec, context.rounding, context.Emin, context.Emax,
        context.capitals, context.clamp, dict(context.flags), dict(context.traps),
    )


@pytest.mark.parametrize("precision", [2, 400])
@pytest.mark.parametrize(("operator", "number", "expected"), _NEAR_CASES)
def test_long_thresholds_keep_exact_pixel_edges_and_caller_decimal_state(precision, operator, number, expected):
    with localcontext() as context:
        context.prec = precision
        context.traps[Inexact] = context.traps[Rounded] = context.traps[FloatOperation] = True
        context.flags[Inexact] = True
        before = _context_state(context)
        assert _selected("size:" + operator + number + "mp") == expected
        assert _context_state(context) == before


@pytest.mark.parametrize(("operator", "expected"), [
    (">", [9, 10]), (">=", [8, 9, 10]), ("<", list(range(1, 8))), ("<=", list(range(1, 9))),
])
def test_engine_limit_and_larger_stored_products_stay_exact(operator, expected):
    assert _selected("size:" + operator + MAXIMUM_MP + "mp") == expected


@pytest.mark.parametrize("number", ["0", "-0", "0e1000000", "-0e1000000", "1e-1000000"])
def test_zero_and_tiny_thresholds_have_correct_all_or_none_positive_geometry(number):
    for operator in (">", ">="):
        assert _selected("size:" + operator + number + "mp") == POSITIVE_IDS
    for operator in ("<", "<="):
        assert _selected("size:" + operator + number + "mp") == []


@pytest.mark.parametrize(("number", "expected"), [
    ("+1", FROM_MILLION), ("1e0", FROM_MILLION), ("01.000", FROM_MILLION),
    ("1.", FROM_MILLION), (".000001", POSITIVE_IDS), ("0.000001", POSITIVE_IDS),
])
def test_finite_numeric_forms_and_mp_case_keep_their_exact_value(number, expected):
    assert _selected("SIZE:>=" + number + "MP") == expected


_INVALID = [
    "-1", "-0.000001", "NaN", "sNaN", "NaN123", "Infinity", "-Infinity", "inf",
    "1e1000000", "9223372036854.7758071", "1e999999999999999999999",
    "not-a-number", "", "1-2", "1/2",
]


@pytest.mark.parametrize("number", _INVALID)
def test_bad_area_magnitudes_have_named_refusals_before_sql(number):
    with pytest.raises(ValueError, match="size:"):
        search.strict_scope("size:>=" + number + "mp")


@pytest.mark.parametrize("precision", [2, 400])
@pytest.mark.parametrize("number", ["bad", "sNaN", "-1", "1e1000000", "1e999999999999999999999"])
def test_refusals_leave_caller_decimal_flags_and_traps_unchanged(precision, number):
    with localcontext() as context:
        context.prec = precision
        context.traps[InvalidOperation] = context.traps[Inexact] = context.traps[Rounded] = True
        context.flags[Inexact] = True
        before = _context_state(context)
        with pytest.raises(ValueError, match="size:"):
            search.strict_scope("size:>=" + number + "mp")
        assert _context_state(context) == before


@pytest.mark.parametrize("query", [
    "size:1mp", "size:=1mp", "size:1-2mp", "size:>=1m", "size:>=1megapixel", "size:>=1px",
])
def test_bare_equality_range_and_unit_alias_forms_stay_unsupported(query):
    with pytest.raises(ValueError, match="WIDTHxHEIGHT"):
        search.strict_scope(query)


@pytest.mark.parametrize(("query", "expected"), [
    ("size:1024x1024", [6]), ("size:1024\u00d71024", [6]),
    ("size:square", [1, 3, 6, 10, 15]),
    ("size:landscape size:1536x640", [5]),
])
def test_legacy_exact_and_orientation_paths_keep_their_existing_comparisons(query, expected):
    assert _selected(query) == expected


def test_area_terms_intersect_with_each_other_orientation_and_prompt_text():
    assert _selected("size:>=0.98mp size:<1mp size:landscape") == [2, 5]
    assert _selected("size:>=1mp size:<1mp") == []
    clauses, _values, text = search.split_search("harbor size:>=1mp")
    assert clauses and text == "harbor"
    with pytest.raises(ValueError, match="free text"):
        search.strict_scope("harbor size:>=1mp")


def _save(store, label, width, height, seed, *, model=None, favorite=False, binned=False):
    with Image.new("RGB", (width, height), (seed * 20, 40, 90)) as image:
        record = store.save_image(
            image, GenerateRequest(prompt=label, width=width, height=height),
            seed=seed, duration_ms=1, model_id=model or store.settings.model_id,
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
    return {
        "low": _save(store, "harbor wide nominal one MP", 1536, 640, 1),
        "over": _save(store, "harbor square nominal one MP", 1024, 1024, 2),
        "portrait": _save(store, "harbor portrait keeper", 1056, 1584, 3, favorite=True),
        "landscape": _save(store, "harbor landscape", 1584, 1056, 4),
        "other": _save(store, "harbor other square", 1024, 1024, 5, model=store.settings.model_id + "/other"),
        "bin": _save(store, "harbor binned square", 1024, 1024, 6, binned=True),
    }


def test_gallery_area_scopes_favorites_and_paging_preserve_all_rows_and_pngs(
    client, store, owned_rows, cold_queries_only,
):
    before = gallery_state(store)
    current = client.get("/api/gallery", params={"search": "size:>=1mp"})
    assert current.status_code == 200, current.text
    assert current.json()["total"] == 3 and current.json()["other_models"] == 1
    assert {row["id"] for row in current.json()["items"]} == {owned_rows[key].id for key in ("over", "portrait", "landscape")}
    starred = client.get("/api/gallery", params={
        "search": "SIZE:>=1MP size:portrait", "favorites": True, "all_models": True,
    })
    assert starred.status_code == 200, starred.text
    assert [row["id"] for row in starred.json()["items"]] == [owned_rows["portrait"].id]
    first = client.get("/api/gallery", params={"search": "size:>=1mp", "all_models": True, "limit": 1})
    assert first.status_code == 200, first.text
    assert first.json()["total"] == 4 and first.json()["has_more"] is True
    assert first.json()["items"][0]["id"] == owned_rows["other"].id
    tail = client.get("/api/gallery", params={
        "search": "size:>=1mp", "all_models": True, "before": owned_rows["other"].id,
    })
    assert tail.status_code == 200, tail.text
    assert {row["id"] for row in tail.json()["items"]} == {owned_rows[key].id for key in ("over", "portrait", "landscape")}
    assert gallery_state(store) == before and cold_queries_only == []


def test_bin_and_strict_preview_share_exact_area_with_existing_scope_rules(
    client, store, owned_rows, cold_queries_only,
):
    before = gallery_state(store)
    binned = client.get("/api/gallery/bin", params={"search": "size:>=1mp", "favorites": True})
    assert binned.status_code == 200, binned.text
    assert binned.json()["total"] == 1
    assert [row["id"] for row in binned.json()["items"]] == [owned_rows["bin"].id]
    preview = client.get("/api/gallery/unstarred", params={"search": "size:>=1mp"})
    assert preview.status_code == 200, preview.text
    answer = preview.json()
    assert (answer["records"], answer["favorites"], answer["sources_kept"], answer["unstarred"]) == (3, 1, 0, 2)
    assert answer["job_active"] is False
    assert gallery_state(store) == before and cold_queries_only == []


@pytest.mark.parametrize("route", ["/api/gallery", "/api/gallery/bin", "/api/gallery/unstarred"])
def test_bad_area_magnitude_has_read_routes_named_validation_status(client, cold_queries_only, route):
    response = client.get(route, params={"search": "size:>=NaNmp", "all_models": True})
    assert response.status_code == 422, response.text
    assert "size:" in response.json()["detail"]
    assert cold_queries_only == []


def test_saved_area_interval_retains_raw_query_ticks_and_redacted_invalid_write(
    client, store, owned_rows, cold_queries_only,
):
    body = {"name": "Larger keepers", "query": "  SIZE:>=1MP size:<2mp \n", "favorites": True, "all_models": False}
    created = client.post("/api/gallery/saved-searches", json=body)
    assert created.status_code == 201, created.text
    item = created.json()["items"][0]
    assert {key: item[key] for key in body} == body
    assert client.get("/api/gallery/saved-searches").json()["items"][0] == item
    applied = client.get("/api/gallery", params={
        "search": item["query"], "favorites": item["favorites"], "all_models": item["all_models"],
    })
    assert applied.status_code == 200, applied.text
    assert [row["id"] for row in applied.json()["items"]] == [owned_rows["portrait"].id]
    path = saved_searches.path_for(store.settings.db_path)
    before = path.read_bytes()
    refused = client.post("/api/gallery/saved-searches", json={
        **body, "name": "Invalid area", "query": "private-area-query size:>=NaNmp",
    })
    assert refused.status_code == 422, refused.text
    assert refused.json() == {"detail": "Saved search fields are invalid. Check the name, query, ticks and revision."}
    assert "private-area-query" not in refused.text and path.read_bytes() == before
    assert cold_queries_only == []


def test_keeper_rates_use_area_scope_and_preserve_binned_and_other_model_exclusions(
    store, owned_rows, cold_queries_only,
):
    before = gallery_state(store)
    rows = gallery_reports.keep_rates(search="size:>=1mp model:=" + store.settings.model_id)
    assert sum(row["made"] for row in rows) == 3
    assert sum(row["starred"] for row in rows) == 1
    assert {row["size"] for row in rows} == {"1024x1024", "1056x1584", "1584x1056"}
    assert gallery_state(store) == before and cold_queries_only == []
