"""Bounded atomic named setups, independent of gallery and generation locks."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from . import gallery_lock
from .schemas_presets import MAX_PRESETS, Preset, PresetFields, PresetUpdate

MAX_FILE_BYTES = 8 * 1024 * 1024
_VERSION = 1


class PresetsError(RuntimeError):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


def path_for(database: Path) -> Path:
    # This is a new feature: there is no pre-rename saved-setup sidecar.
    return Path(database).expanduser().parent / "auradream.presets.json"


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _read(path: Path) -> list[Preset]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("file is too large")
        payload = json.loads(raw, object_pairs_hook=_unique_fields)
        if not isinstance(payload, dict) or set(payload) != {"version", "items"}:
            raise ValueError("unknown file schema")
        if type(payload["version"]) is not int or payload["version"] != _VERSION:
            raise ValueError("unknown version")
        items = payload["items"]
        if not isinstance(items, list) or len(items) > MAX_PRESETS:
            raise ValueError("invalid collection size")
        parsed = [Preset.model_validate(item) for item in items]
        if len({item.id for item in parsed}) != len(parsed):
            raise ValueError("duplicate ids")
        return parsed
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise PresetsError(
            "Saved setups could not be read. Their file has been left unchanged; "
            f"check {path.name} beside the gallery database."
        ) from exc


def _write(path: Path, items: list[Preset]) -> None:
    data = (json.dumps(
        {"version": _VERSION, "items": [item.model_dump() for item in items]},
        ensure_ascii=False, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise PresetsError("The saved setups file would exceed 8 MiB.", 422)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".auradream.presets.",
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


def read(database: Path) -> list[Preset]:
    # Atomic replacement makes unlocked reads complete and first reads cold.
    return _read(path_for(database))


def change(
    database: Path, values: PresetFields | PresetUpdate | None,
    *, item_id: str | None = None, revision: int | None = None,
) -> list[Preset]:
    path = path_for(database)
    try:
        with gallery_lock.hold(path):
            items = _read(path)
            if item_id is None:
                if len(items) >= MAX_PRESETS:
                    raise PresetsError(f"The list holds at most {MAX_PRESETS} setups.", 422)
                assert values is not None
                items.append(Preset(**values.model_dump(), id=uuid.uuid4().hex, revision=1))
            else:
                at = next((index for index, item in enumerate(items) if item.id == item_id), None)
                if at is None:
                    raise PresetsError("That setup is no longer in the saved list. Reload the list.", 404)
                expected = values.revision if isinstance(values, PresetUpdate) else revision
                if items[at].revision != expected:
                    raise PresetsError(
                        "That setup changed in another tab. Your editor is unchanged; reload the list to read it.", 409,
                    )
                if values is None:
                    del items[at]
                else:
                    fields = values.model_dump(exclude={"revision"})
                    items[at] = Preset(**fields, id=item_id, revision=items[at].revision + 1)
            _write(path, items)
            return items
    except gallery_lock.GalleryBusy as exc:
        raise PresetsError("Another AuraDream process is saving setups; try again.", 409) from exc
    except OSError as exc:
        raise PresetsError("The setups could not be saved. The previous saved list has been kept.") from exc
