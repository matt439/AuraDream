"""Explicit calendar date ranges stay shared, bounded and read-only to query."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest
from PIL import Image

from backend import main, saved_searches, search
from backend.schemas import GenerateRequest

_RANGE = "created_at >= ? AND created_at < ?"
_EXPLICIT = "2026-10-26..2026-10-28"
_SAVED = "/api/gallery/saved-searches"
_CASES = [
    pytest.param(
        _EXPLICIT, 0, 0,
        "2026-10-26T00:00:00+00:00", "2026-10-29T00:00:00+00:00", 72,
        id="three-ordinary-calendar-days",
    ),
    pytest.param(
        "2026-10-26..2026-10-26", 120, 120,
        "2026-10-25T22:00:00+00:00", "2026-10-26T22:00:00+00:00", 24,
        id="same-day-positive-offset",
    ),
    pytest.param(
        "2026-03-07..2026-03-09", -300, -240,
        "2026-03-07T05:00:00+00:00", "2026-03-10T04:00:00+00:00", 71,
        id="three-days-across-synthetic-spring-forward",
    ),
    pytest.param(
        "2026-10-31..2026-11-02", -240, -300,
        "2026-10-31T04:00:00+00:00", "2026-11-03T05:00:00+00:00", 73,
        id="three-days-across-synthetic-fall-back",
    ),
    pytest.param(
        "2026-12-31..2027-01-01", 330, 330,
        "2026-12-30T18:30:00+00:00", "2027-01-01T18:30:00+00:00", 48,
        id="year-rollover-half-hour-offset",
    ),
    pytest.param(
        "2028-02-28..2028-03-01", -180, -180,
        "2028-02-28T03:00:00+00:00", "2028-03-02T03:00:00+00:00", 72,
        id="three-days-including-leap-day",
    ),
]
_INVALID = [
    "..", "..2026-10-28", "2026-10-26..", "2026-10-28..2026-10-26",
    "2026-02-30..2026-03-01", "2026-02-28..2026-02-30",
    "today..2026-10-28", "2026-10-26..yesterday", "today..yesterday",
    "2026-10-26..2026-10-27..2026-10-28", "2026-10-26...2026-10-28",
    "20261026..20261028", "2026-W44-1..2026-W44-3",
    ">=2026-10-26..2026-10-28", "2026-10-26..<=2026-10-28",
]


def _supply_local_midnights(monkeypatch, offsets=None):
    supplied_offsets = offsets or {}

    class LocalMidnight(datetime):
        def astimezone(self, tz=None):
            assert self.tzinfo is None
            assert tz is timezone.utc
            offset = timezone(timedelta(minutes=supplied_offsets.get(self.date(), 0)))
            local = datetime(
                self.year, self.month, self.day,
                self.hour, self.minute, self.second, self.microsecond,
                tzinfo=offset,
            )
            return local.astimezone(tz)

    monkeypatch.setattr(search, "datetime", LocalMidnight)


@pytest.mark.parametrize(("raw", "first_offset", "last_offset", "start", "end", "hours"), _CASES)
def test_calendar_range_selects_inclusive_dates_with_actual_sqlite(
    monkeypatch, raw, first_offset, last_offset, start, end, hours,
):
    first_raw, last_raw = raw.split("..")
    _supply_local_midnights(monkeypatch, {
        date.fromisoformat(first_raw): first_offset,
        date.fromisoformat(last_raw) + timedelta(days=1): last_offset,
    })
    first, stop = datetime.fromisoformat(start), datetime.fromisoformat(end)
    instants = [
        first - timedelta(seconds=1), first, first + timedelta(minutes=30),
        first + (stop - first) / 2, stop - timedelta(minutes=30),
        stop - timedelta(seconds=1), stop, stop + timedelta(minutes=30),
    ]
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO images (id, created_at) VALUES (?, ?)",
            [(index, instant.isoformat()) for index, instant in enumerate(instants, 1)],
        )
        clauses, parameters = search.strict_scope(f"date:{raw}")
        found = connection.execute(
            "SELECT id FROM images WHERE " + " AND ".join(clauses) + " ORDER BY id", parameters,
        ).fetchall()
        assert found == [(2,), (3,), (4,), (5,), (6,)]
        assert clauses == [_RANGE]
        assert parameters == [start, end]
        assert stop - first == timedelta(hours=hours)
    finally:
        connection.close()


@pytest.mark.parametrize("single", ["2026-10-26", "20261026", "2026-W44-1"])
def test_same_day_range_matches_the_existing_single_day_grammar(monkeypatch, single):
    _supply_local_midnights(monkeypatch)
    canonical = date.fromisoformat(single).isoformat()
    assert search.date_clause(f"{canonical}..{canonical}") == search.date_clause(single)


@pytest.mark.parametrize("raw", _INVALID)
def test_ranges_need_two_ordered_absolute_calendar_dates(raw):
    with pytest.raises(ValueError) as refused:
        search.date_clause(raw)
    assert str(refused.value) == f"date: {raw!r} is not an ordered YYYY-MM-DD..YYYY-MM-DD range"


def test_terminal_range_end_uses_the_existing_named_overflow_refusal(monkeypatch):
    _supply_local_midnights(monkeypatch)
    raw = "9999-12-30..9999-12-31"
    with pytest.raises(ValueError) as refused:
        search.strict_scope(f"date:{raw}")
    assert str(refused.value) == f"date: {raw!r} is out of range"


def test_multiple_date_terms_still_intersect_and_free_text_is_not_a_strict_scope(monkeypatch):
    _supply_local_midnights(monkeypatch)
    clauses, parameters, text = search.split_search(
        "harbor date:2026-10-26..2026-10-28 date:2026-10-27..2026-10-29"
    )
    assert clauses == [_RANGE, _RANGE]
    assert parameters == [
        "2026-10-26T00:00:00+00:00", "2026-10-29T00:00:00+00:00",
        "2026-10-27T00:00:00+00:00", "2026-10-30T00:00:00+00:00",
    ]
    assert text == "harbor"
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO images (id, created_at) VALUES (?, ?)",
            [(day, f"2026-10-{day}T12:00:00+00:00") for day in range(26, 30)],
        )
        assert connection.execute(
            "SELECT id FROM images WHERE " + " AND ".join(clauses) + " ORDER BY id", parameters,
        ).fetchall() == [(27,), (28,)]
    finally:
        connection.close()
    with pytest.raises(ValueError, match="free text"):
        search.strict_scope(f"date:{_EXPLICIT} harbor")


def test_a_space_before_range_punctuation_keeps_existing_day_plus_text_parsing(monkeypatch):
    _supply_local_midnights(monkeypatch)
    query = "date:2026-10-26 ..2026-10-28"
    clauses, parameters, text = search.split_search(query)
    assert clauses == [_RANGE]
    assert parameters == ["2026-10-26T00:00:00+00:00", "2026-10-27T00:00:00+00:00"]
    assert text == "..2026-10-28"
    with pytest.raises(ValueError, match="free text"):
        search.strict_scope(query)


@pytest.fixture
def cold_queries_only(client, monkeypatch):
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("model-or-queue")
        pytest.fail("a calendar filter entered model loading, generation or queue submission")

    monkeypatch.setattr(main, "ensure_loaded", forbidden)
    monkeypatch.setattr(main.backend, "load", forbidden)
    monkeypatch.setattr(main.backend, "generate", forbidden)
    monkeypatch.setattr(main.jobs, "wait_for_gate", forbidden)
    monkeypatch.setattr(main.jobs, "submit", forbidden)
    yield attempts
    assert attempts == []
    assert client.get("/api/jobs").json() == []


def _owned_rows(store):
    rows = [
        ("before", "2026-10-25T23:59:59+00:00", False, False),
        ("first", "2026-10-26T00:00:00+00:00", False, False),
        ("middle", "2026-10-27T12:00:00+00:00", False, True),
        ("last", "2026-10-28T23:59:59+00:00", False, False),
        ("after", "2026-10-29T00:00:00+00:00", False, False),
        ("bin-last", "2026-10-28T23:59:59+00:00", True, False),
        ("bin-after", "2026-10-29T00:00:00+00:00", True, False),
    ]
    owned = {}
    for seed, (label, created, binned, favorite) in enumerate(rows, 1):
        with Image.new("RGB", (256, 256), (seed * 20, 40, 90)) as image:
            record = store.save_image(
                image, GenerateRequest(prompt=f"harbor {label}", width=256, height=256),
                seed=seed, duration_ms=1, model_id=store.settings.model_id,
            )
        with store._session() as connection:
            connection.execute(
                "UPDATE images SET created_at = ?, binned_at = ?, favorite = ? WHERE id = ?",
                (created, "2026-10-30T00:00:00+00:00" if binned else None, int(favorite), record.id),
            )
        owned[label] = record.id
    return owned


def _row_state(store):
    with store._session() as connection:
        return [tuple(row) for row in connection.execute(
            "SELECT id, created_at, binned_at, favorite FROM images ORDER BY id"
        ).fetchall()]


def test_gallery_bin_and_unstarred_preview_share_the_same_calendar_range(
    client, store, monkeypatch, cold_queries_only,
):
    _supply_local_midnights(monkeypatch)
    owned = _owned_rows(store)
    before = _row_state(store)
    params = {"search": f"date:{_EXPLICIT}", "all_models": True}
    live = client.get("/api/gallery", params=params)
    assert live.status_code == 200, live.text
    assert live.json()["total"] == 3
    assert {item["id"] for item in live.json()["items"]} == {owned[key] for key in ("first", "middle", "last")}
    binned = client.get("/api/gallery/bin", params=params)
    assert binned.status_code == 200, binned.text
    assert binned.json()["total"] == 1
    assert [item["id"] for item in binned.json()["items"]] == [owned["bin-last"]]
    preview = client.get("/api/gallery/unstarred", params=params)
    assert preview.status_code == 200, preview.text
    result = preview.json()
    assert (result["records"], result["favorites"], result["unstarred"], result["sources_kept"]) == (3, 1, 2, 0)
    assert result["job_active"] is False
    assert _row_state(store) == before
    assert cold_queries_only == []


@pytest.mark.parametrize("route", ["/api/gallery", "/api/gallery/bin", "/api/gallery/unstarred"])
def test_invalid_date_range_has_the_read_routes_validation_status(client, cold_queries_only, route):
    response = client.get(route, params={"search": "date:2026-10-28..2026-10-26", "all_models": True})
    assert response.status_code == 422, response.text
    assert "YYYY-MM-DD..YYYY-MM-DD" in response.json()["detail"]
    assert cold_queries_only == []


def test_owned_date_range_binning_keeps_count_guards_stars_sources_and_other_dates(
    client, store, monkeypatch, cold_queries_only,
):
    _supply_local_midnights(monkeypatch)
    owned = _owned_rows(store)
    # A valid owned Refine record makes the range's first row a kept source.
    with Image.new("RGB", (256, 256), (10, 20, 30)) as image:
        refined = store.save_image(
            image,
            GenerateRequest(
                prompt="harbor starred refinement", width=256, height=256,
                refine=owned["first"], strength=0.5,
            ),
            seed=81, duration_ms=1, model_id=store.settings.model_id,
        )
    with store._session() as connection:
        connection.execute(
            "UPDATE images SET created_at = ?, favorite = 1 WHERE id = ?",
            ("2026-10-27T18:00:00+00:00", refined.id),
        )
    before = _row_state(store)
    pngs = {path.name for path in store.settings.outputs_dir.glob("*.png")}
    params = {"search": f"date:{_EXPLICIT}", "all_models": True}
    preview = client.get("/api/gallery/unstarred", params=params)
    assert preview.status_code == 200, preview.text
    result = preview.json()
    assert (result["records"], result["favorites"], result["sources_kept"], result["unstarred"]) == (4, 2, 1, 1)
    assert result["sources"] == [owned["first"]]
    stale = client.delete("/api/gallery/unstarred", params={**params, "expected": 2})
    assert stale.status_code == 409, stale.text
    assert _row_state(store) == before
    committed = client.delete("/api/gallery/unstarred", params={**params, "expected": result["unstarred"]})
    assert committed.status_code == 200, committed.text
    assert committed.json() == {"binned": [owned["last"]], "deleted": [owned["last"]], "files": 0}
    after = {row[0]: row[1:] for row in _row_state(store)}
    assert set(after) == {row[0] for row in before}
    for image_id, created, binned, favorite in before:
        if image_id == owned["last"]:
            assert after[image_id][0] == created
            assert after[image_id][1] is not None
            assert after[image_id][2] == favorite
        else:
            assert after[image_id] == (created, binned, favorite)
    assert {path.name for path in store.settings.outputs_dir.glob("*.png")} == pngs
    assert (store.settings.outputs_dir / refined.filename).exists()
    assert cold_queries_only == []


def test_saved_date_range_keeps_raw_query_ticks_and_existing_redacted_validation(
    client, store, monkeypatch, cold_queries_only,
):
    _supply_local_midnights(monkeypatch)
    owned = _owned_rows(store)
    query = f"  harbor date:{_EXPLICIT} \n"
    body = {"name": "October nights", "query": query, "favorites": True, "all_models": False}
    created = client.post(_SAVED, json=body)
    assert created.status_code == 201, created.text
    item = created.json()["items"][0]
    assert {key: item[key] for key in body} == body
    assert client.get(_SAVED).json()["items"][0] == item
    applied = client.get("/api/gallery", params={
        "search": item["query"], "favorites": item["favorites"], "all_models": item["all_models"],
    })
    assert applied.status_code == 200, applied.text
    assert [row["id"] for row in applied.json()["items"]] == [owned["middle"]]
    path = saved_searches.path_for(store.settings.db_path)
    before = path.read_bytes()
    invalid = client.post(_SAVED, json={
        **body, "name": "Invalid nights", "query": "private-query date:2026-10-28..2026-10-26",
    })
    assert invalid.status_code == 422, invalid.text
    assert invalid.json() == {
        "detail": "Saved search fields are invalid. Check the name, query, ticks and revision."
    }
    assert "private-query" not in invalid.text
    assert path.read_bytes() == before
    assert cold_queries_only == []
