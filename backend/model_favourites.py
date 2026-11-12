"""Bounded, atomic, revisioned model favourites in a database-scoped sidecar.

One set of canonical checkpoint ids per gallery database (the Models panel's
stars), shared by every browser and address that reaches this server. It has
its own reservation and never enters gallery maintenance, model admission or
the GPU gate.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import gallery_lock, model_info
from .schemas_model_favourites import MAX_MODEL_FAVOURITES, checked_model_id

MAX_FILE_BYTES = 256 * 1024
_VERSION = 1


class ModelFavouritesError(RuntimeError):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


def path_for(database: Path) -> Path:
    return database.with_name(f"auradream.model-favourites-{database.name}.json")


def canonical(model_id: str) -> str:
    """The id a star is kept under: a catalogue entry's own repo for any of
    its builds (the page's `favourite_id`), otherwise the id as given."""
    entry = model_info._catalogue_entry(model_id)
    return entry.repo if entry is not None else model_id


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def _read(path: Path) -> tuple[list[str], int]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("file is too large")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or set(payload) != {"version", "revision", "items"}:
            raise ValueError("unknown file schema")
        if type(payload["version"]) is not int or payload["version"] != _VERSION:
            raise ValueError("unknown version")
        revision = payload["revision"]
        if type(revision) is not int or revision < 0:
            raise ValueError("invalid revision")
        raw_items = payload["items"]
        if not isinstance(raw_items, list) or len(raw_items) > MAX_MODEL_FAVOURITES:
            raise ValueError("invalid collection size")
        items = [checked_model_id(item) for item in raw_items]
        if len(set(items)) != len(items) or items != raw_items:
            raise ValueError("duplicate or untrimmed ids")
        return items, revision
    except FileNotFoundError:
        return [], 0
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        raise ModelFavouritesError(
            "Model favourites could not be read. Their file has been left unchanged; "
            "check the model favourites sidecar beside the gallery database."
        ) from None


def _write(path: Path, items: list[str], revision: int) -> None:
    data = (json.dumps(
        {"version": _VERSION, "revision": revision, "items": items}, ensure_ascii=False, indent=2,
    ) + "\n").encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise ModelFavouritesError("The model favourites file would exceed 256 KiB.", 422)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".auradream.model-favourites.",
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


def read(database: Path) -> tuple[list[str], int]:
    # Atomic replacement makes a read coherent without creating a lock or file.
    return _read(path_for(database))


def _update(database: Path, adds: list[str], removes: list[str]) -> tuple[list[str], int, int]:
    """Apply set changes under the sidecar's reservation; the answer's
    revision moves only when the set did, so a repeated press writes nothing."""
    path = path_for(database)
    try:
        with gallery_lock.hold(path, wait=gallery_lock.SESSION_WAIT):
            items, revision = _read(path)
            gone = set(removes)
            kept = [item for item in items if item not in gone]
            added = [item for item in dict.fromkeys(adds) if item not in kept]
            result = kept + added
            if result == items:
                return items, revision, 0
            if len(result) > MAX_MODEL_FAVOURITES:
                raise ModelFavouritesError(
                    f"The favourites hold at most {MAX_MODEL_FAVOURITES} models. Unstar some first.", 422,
                )
            _write(path, result, revision + 1)
            return result, revision + 1, len(added)
    except gallery_lock.GalleryBusy:
        raise ModelFavouritesError("Another AuraDream process is saving model favourites; try again.", 409) from None
    except OSError:
        raise ModelFavouritesError("The model favourites could not be saved. The previous set has been kept.") from None


def change(database: Path, model_id: str, favourite: bool) -> tuple[list[str], int]:
    model_id = canonical(model_id)
    items, revision, _ = _update(database, [model_id] if favourite else [], [] if favourite else [model_id])
    return items, revision


def merge(database: Path, model_ids: list[str]) -> tuple[list[str], int, int]:
    """Add a browser's earlier set; never removes a star already here."""
    return _update(database, [canonical(model_id) for model_id in model_ids], [])
