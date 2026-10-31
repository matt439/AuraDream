"""`storage._ADDED_COLUMNS` is the one list of the columns an older table gains.

`_migrate` adds them, and a bundle merge's ALTER on its staged copy takes the
same types from it, so a REAL or INTEGER column listed only in the schema
would come back as TEXT there. Each column a writer supplies is either in the
original schema the bundle layer requires or in this list, with the type
`_SCHEMA` gives it.
"""
from __future__ import annotations

import re
import sqlite3

from backend import storage
from backend.bundle_common import _DB_REQUIRED


def _schema_types() -> dict[str, str]:
    table = storage._SCHEMA[storage._SCHEMA.index("CREATE TABLE IF NOT EXISTS images"):]
    table = table[:table.index(");")]
    return {
        name: kind
        for name, kind in re.findall(r"^\s+(\w+)\s+(TEXT|INTEGER|REAL)\b", table, re.M)
    }


def test_every_added_column_has_the_schema_type():
    schema = _schema_types()
    assert {column: kind for column, kind, _ in storage._ADDED_COLUMNS} == {
        column: schema[column] for column, _, _ in storage._ADDED_COLUMNS
    }


def test_every_written_column_is_original_or_added():
    added = {column for column, _, _ in storage._ADDED_COLUMNS}
    assert [column for column in storage._COLUMNS if column not in added | _DB_REQUIRED] == []
    assert set(_schema_types()) == added | _DB_REQUIRED


def test_an_original_table_migrates_to_the_schema_types(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db")
    conn.row_factory = sqlite3.Row
    original = ", ".join(sorted(_DB_REQUIRED))
    conn.execute(f"CREATE TABLE images ({original})")
    storage._migrate(conn)
    types = {row["name"]: row["type"] for row in conn.execute("PRAGMA table_info(images)")}
    conn.close()
    schema = _schema_types()
    assert {column: types[column] for column, _, _ in storage._ADDED_COLUMNS} == {
        column: schema[column] for column, _, _ in storage._ADDED_COLUMNS
    }
