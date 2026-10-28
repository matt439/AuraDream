"""Local-day search bounds across synthetic clock changes, using real SQLite.

The supplied offsets describe synthetic local midnights, not a host timezone.
Only search.datetime is replaced; no OS timezone, environment or clock changes.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from backend import search

_RANGE = "created_at >= ? AND created_at < ?"
_CASES = [
    pytest.param(
        "2026-07-01", 0, 0,
        "2026-07-01T00:00:00+00:00", "2026-07-02T00:00:00+00:00", 24,
        id="ordinary-utc-day",
    ),
    pytest.param(
        "2026-07-01", 120, 120,
        "2026-06-30T22:00:00+00:00", "2026-07-01T22:00:00+00:00", 24,
        id="ordinary-positive-offset-day",
    ),
    pytest.param(
        "2026-03-08", -300, -240,
        "2026-03-08T05:00:00+00:00", "2026-03-09T04:00:00+00:00", 23,
        id="synthetic-spring-forward-day",
    ),
    pytest.param(
        "2026-11-01", -240, -300,
        "2026-11-01T04:00:00+00:00", "2026-11-02T05:00:00+00:00", 25,
        id="synthetic-fall-back-day",
    ),
    pytest.param(
        "2026-12-31", 330, 330,
        "2026-12-30T18:30:00+00:00", "2026-12-31T18:30:00+00:00", 24,
        id="year-rollover-half-hour-offset",
    ),
    pytest.param(
        "2028-02-28", -180, -180,
        "2028-02-28T03:00:00+00:00", "2028-02-29T03:00:00+00:00", 24,
        id="enter-leap-day",
    ),
    pytest.param(
        "2028-02-29", -180, -180,
        "2028-02-29T03:00:00+00:00", "2028-03-01T03:00:00+00:00", 24,
        id="leave-leap-day",
    ),
]


def _clock_for(offset_minutes):
    class LocalMidnight(datetime):
        def astimezone(self, tz=None):
            assert self.tzinfo is None, "the supplied midnight must still be local"
            assert tz is timezone.utc
            offset = timezone(timedelta(minutes=offset_minutes[self.date()]))
            supplied = datetime(
                self.year, self.month, self.day,
                self.hour, self.minute, self.second, self.microsecond,
                tzinfo=offset,
            )
            return supplied.astimezone(tz)

    return LocalMidnight


def _supply_midnights(monkeypatch, raw, start_offset, end_offset):
    day = date.fromisoformat(raw)
    monkeypatch.setattr(search, "datetime", _clock_for({
        day: start_offset, day + timedelta(days=1): end_offset,
    }))


@pytest.mark.parametrize(("raw", "start_offset", "end_offset", "start", "end", "hours"), _CASES)
def test_date_bounds_use_both_local_midnights(
    monkeypatch, raw, start_offset, end_offset, start, end, hours,
):
    _supply_midnights(monkeypatch, raw, start_offset, end_offset)
    clause, parameters = search.date_clause(raw)
    assert clause == _RANGE
    assert parameters == [start, end]
    assert datetime.fromisoformat(parameters[1]) - datetime.fromisoformat(parameters[0]) == timedelta(hours=hours)


@pytest.mark.parametrize(("raw", "start_offset", "end_offset", "start", "end", "hours"), _CASES)
def test_strict_date_scope_selects_exact_calendar_day_in_sqlite(
    monkeypatch, raw, start_offset, end_offset, start, end, hours,
):
    _supply_midnights(monkeypatch, raw, start_offset, end_offset)
    first = datetime.fromisoformat(start)
    stop = datetime.fromisoformat(end)
    # The last half-hour is still the selected fall-back day. The first
    # half-hour after spring-forward's next midnight belongs to tomorrow.
    instants = [
        first - timedelta(seconds=1), first, first + timedelta(minutes=30),
        stop - timedelta(minutes=30), stop - timedelta(seconds=1),
        stop, stop + timedelta(minutes=30),
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
        assert found == [(2,), (3,), (4,), (5,)]
        assert clauses == [_RANGE]
        assert parameters == [start, end]
        assert stop - first == timedelta(hours=hours)
    finally:
        connection.close()


@pytest.mark.parametrize(("raw", "offsets"), [
    pytest.param("9999-12-31", {date.max: 0}, id="no-representable-next-calendar-day"),
    pytest.param(
        "0001-01-01", {date.min: 60, date(1, 1, 2): 60},
        id="utc-start-would-precede-datetime-minimum",
    ),
])
def test_unrepresentable_date_bounds_are_named_value_refusals(monkeypatch, raw, offsets):
    monkeypatch.setattr(search, "datetime", _clock_for(offsets))
    expected = f"date: {raw!r} is out of range"
    with pytest.raises(ValueError) as direct:
        search.date_clause(raw)
    assert str(direct.value) == expected
    with pytest.raises(ValueError) as strict:
        search.strict_scope(f"date:{raw}")
    assert str(strict.value) == expected



def test_gallery_refuses_an_unrepresentable_date_with_its_named_validation_error(client, monkeypatch):
    monkeypatch.setattr(search, "datetime", _clock_for({date.max: 0}))
    response = client.get("/api/gallery", params={"search": "date:9999-12-31", "all_models": True})
    assert response.status_code == 422
    assert response.json() == {"detail": "date: '9999-12-31' is out of range"}
