"""Bounded, atomic gallery-search definitions in a database-scoped sidecar."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from . import gallery_lock
from .schemas_saved_searches import (
    MAX_SAVED_SEARCHES, SavedSearch, SavedSearchFields, SavedSearchUpdate, name_key,
)

MAX_FILE_BYTES = 1024 * 1024
_VERSION = 1


class SavedSearchesError(RuntimeError):
    def __init__(self, message: str, status: int = 503, *, conflict_id: str | None = None):
        super().__init__(message)
        self.status = status
        self.conflict_id = conflict_id


def path_for(database: Path) -> Path:
    return database.with_name(f"auradream.saved-searches-{database.name}.json")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def _read(path: Path) -> list[SavedSearch]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("file is too large")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or set(payload) != {"version", "items"}:
            raise ValueError("unknown file schema")
        if type(payload["version"]) is not int or payload["version"] != _VERSION:
            raise ValueError("unknown version")
        raw_items = payload["items"]
        if not isinstance(raw_items, list) or len(raw_items) > MAX_SAVED_SEARCHES:
            raise ValueError("invalid collection size")
        items = [SavedSearch.model_validate(item, context={"stored": True}) for item in raw_items]
        if len({item.id for item in items}) != len(items):
            raise ValueError("duplicate ids")
        if len({name_key(item.name) for item in items}) != len(items):
            raise ValueError("duplicate names")
        return items
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        raise SavedSearchesError(
            "Saved searches could not be read. Their file has been left unchanged; "
            "check the saved searches sidecar beside the gallery database."
        ) from None


def _write(path: Path, items: list[SavedSearch]) -> None:
    data = (json.dumps(
        {"version": _VERSION, "items": [item.model_dump() for item in items]},
        ensure_ascii=False, indent=2,
    ) + "\n").encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise SavedSearchesError("The saved searches file would exceed 1 MiB.", 422)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".auradream.saved-searches.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def read(database: Path) -> list[SavedSearch]:
    # Atomic replacement makes a read coherent without creating a lock or file.
    return _read(path_for(database))


def change(
    database: Path, values: SavedSearchFields | SavedSearchUpdate | None = None,
    *, item_id: str | None = None, revision: int | None = None,
) -> list[SavedSearch]:
    path = path_for(database)
    try:
        # This sidecar has its own reservation; no gallery maintenance or GPU
        # admission is acquired, even when both files share a directory.
        with gallery_lock.hold(path, wait=gallery_lock.SESSION_WAIT):
            items = _read(path)
            if item_id is None:
                assert values is not None
                replacement = SavedSearch(**values.model_dump(), id=uuid.uuid4().hex, revision=1)
                at = None
            else:
                at = next((index for index, item in enumerate(items) if item.id == item_id), None)
                if at is None:
                    raise SavedSearchesError("That search is no longer in the saved list. Reload the list.", 404)
                expected = values.revision if isinstance(values, SavedSearchUpdate) else revision
                if items[at].revision != expected:
                    raise SavedSearchesError(
                        "That saved search changed in another tab. Your editor is unchanged; "
                        "reload the list to read it.", 409,
                    )
                replacement = None if values is None else SavedSearch(
                    **values.model_dump(exclude={"revision"}), id=item_id, revision=items[at].revision + 1,
                )
            if replacement is not None:
                duplicate = next((item for item in items if item.id != replacement.id and name_key(item.name) == name_key(replacement.name)), None)
                if duplicate is not None:
                    raise SavedSearchesError(
                        "A saved search already has that name. Explicitly update it, or use another name.",
                        409, conflict_id=duplicate.id,
                    )
                if at is None:
                    if len(items) >= MAX_SAVED_SEARCHES:
                        raise SavedSearchesError(f"The list holds at most {MAX_SAVED_SEARCHES} saved searches.", 422)
                    items.append(replacement)
                else:
                    items[at] = replacement
            else:
                assert at is not None
                del items[at]
            _write(path, items)
            return items
    except gallery_lock.GalleryBusy:
        raise SavedSearchesError("Another AuraDream process is saving searches; try again.", 409) from None
    except OSError:
        raise SavedSearchesError("The searches could not be saved. The previous saved list has been kept.") from None
