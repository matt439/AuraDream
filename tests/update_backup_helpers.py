"""SQLite fixtures for the two updater scripts' interrupted-copy tests."""
from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory


def make_update_gallery(db: Path, *, wal: bool, value: int) -> None:
    # Capture the committed WAL before SQLite checkpoints it on close. The
    # updater then sees closed files, just as it would after an app crash.
    with TemporaryDirectory(dir=db.parent) as directory:
        source = Path(directory) / "source.db"
        with closing(sqlite3.connect(source)) as conn:
            conn.execute("CREATE TABLE gallery (id INTEGER PRIMARY KEY)")
            conn.commit()
            if wal:
                assert conn.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
                conn.execute("PRAGMA wal_autocheckpoint=0")
            conn.execute("INSERT INTO gallery VALUES (?)", (value,))
            conn.commit()
            shutil.copyfile(source, db)
            if wal:
                shutil.copyfile(f"{source}-wal", f"{db}-wal")


def restored_rows(copy: Path) -> list[tuple[int]]:
    with TemporaryDirectory(dir=copy.parent) as directory:
        restored = Path(directory) / "restored.db"
        shutil.copyfile(copy, restored)
        wal = Path(f"{copy}-wal")
        if wal.is_file():
            shutil.copyfile(wal, f"{restored}-wal")
        with closing(sqlite3.connect(restored)) as conn:
            return conn.execute("SELECT id FROM gallery ORDER BY id").fetchall()
