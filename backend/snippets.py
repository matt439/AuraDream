"""Bounded atomic personal snippets beside the saved startup preferences.

Each edit has an optimistic item revision; the read/modify/write also holds
a separate cross-process reservation. Snippets never reserve the gallery or
generation gate. Bad or newer files are reported, never overwritten empty.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from . import gallery_lock, snippet_portability
from .schemas_snippet_portability import SnippetImportCommit
from .legacy_names import sidecar
from .schemas_snippets import MAX_SNIPPETS, Snippet, SnippetFields, SnippetUpdate

MAX_FILE_BYTES = 8 * 1024 * 1024
_VERSION = 1


class SnippetsError(RuntimeError):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


def path_for(database: Path) -> Path:
    # An existing local-diffusion.snippets.json is kept and rewritten in place.
    return sidecar(database, "snippets.json")


def _read(path: Path) -> list[Snippet]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("file is too large")
        payload = snippet_portability.parse_json(raw, max_bytes=MAX_FILE_BYTES)
        if not isinstance(payload, dict) or set(payload) != {"version", "items"}:
            raise ValueError("unknown file schema")
        if type(payload["version"]) is not int or payload["version"] != _VERSION:
            raise ValueError("unknown version")
        items = payload["items"]
        if not isinstance(items, list) or len(items) > MAX_SNIPPETS:
            raise ValueError("invalid collection size")
        parsed = [Snippet.model_validate(item) for item in items]
        if len({item.id for item in parsed}) != len(parsed):
            raise ValueError("duplicate ids")
        return parsed
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise SnippetsError(
            "Saved snippets could not be read. Their file has been left unchanged; "
            f"check {path.name} beside the gallery database."
        ) from exc


def _write(path: Path, items: list[Snippet]) -> None:
    data = (json.dumps(
        {"version": _VERSION, "items": [item.model_dump() for item in items]},
        ensure_ascii=False, indent=2,
    ) + "\n").encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise SnippetsError("The saved snippets file would exceed 8 MiB.", 422)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".auradream.snippets.",
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


def read(database: Path) -> list[Snippet]:
    # Atomic replacement publishes only a complete file; a GET needs no lock
    # and creates neither a preference file nor a directory on first use.
    return _read(path_for(database))


def change(
    database: Path, values: SnippetFields | SnippetUpdate | None,
    *, item_id: str | None = None, revision: int | None = None,
) -> list[Snippet]:
    path = path_for(database)
    try:
        with gallery_lock.hold(path):
            items = _read(path)
            if item_id is None:
                if len(items) >= MAX_SNIPPETS:
                    raise SnippetsError(f"The list holds at most {MAX_SNIPPETS} snippets.", 422)
                assert values is not None
                items.append(Snippet(**values.model_dump(), id=uuid.uuid4().hex, revision=1))
            else:
                at = next((index for index, item in enumerate(items) if item.id == item_id), None)
                if at is None:
                    raise SnippetsError("That snippet is no longer in the saved list. Reload the list.", 404)
                expected = values.revision if isinstance(values, SnippetUpdate) else revision
                if items[at].revision != expected:
                    raise SnippetsError(
                        "That snippet changed in another tab. Your editor is unchanged; reload the list to read it.", 409,
                    )
                if values is None:
                    del items[at]
                else:
                    fields = values.model_dump(exclude={"revision"})
                    items[at] = Snippet(**fields, id=item_id, revision=items[at].revision + 1)
            _write(path, items)
            return items
    except gallery_lock.GalleryBusy as exc:
        raise SnippetsError("Another AuraDream process is saving snippets; try again.", 409) from exc
    except OSError as exc:
        raise SnippetsError("The snippets could not be saved. The previous saved list has been kept.") from exc


def import_collection(database: Path, body: SnippetImportCommit) -> list[Snippet]:
    """Re-prove the complete preview under the existing snippet reservation."""
    path = path_for(database)
    try:
        with gallery_lock.hold(path):
            try:
                body = SnippetImportCommit.model_validate(body.model_dump())
            except (ValueError, TypeError, RecursionError) as exc:
                raise snippet_portability.PortabilityError("The snippet import request is invalid.") from exc
            items = _read(path)
            if snippet_portability.collection_revision(items) != body.collection_revision:
                raise SnippetsError("The saved snippets changed. Reload the list and preview again.", 409)
            if snippet_portability.payload_digest(body.payload) != body.payload_digest:
                raise snippet_portability.PortabilityError("The selected file changed. Preview it again.")
            if len(body.decisions) != len(body.payload.items) or {
                choice.index for choice in body.decisions
            } != set(range(len(body.payload.items))):
                raise snippet_portability.PortabilityError("Preview a complete choice for every incoming snippet.")
            plan = snippet_portability.plan_merge(items, body.payload, body.decisions)
            if not plan.changed:
                raise snippet_portability.PortabilityError("The selected import changes nothing.")
            result = plan.items + [Snippet(
                **item.model_dump(), id=uuid.uuid4().hex, revision=1,
            ) for item in plan.additions]
            _write(path, result)
            return result
    except gallery_lock.GalleryBusy as exc:
        raise SnippetsError("Another AuraDream process is saving snippets; try again.", 409) from exc
    except OSError as exc:
        raise SnippetsError("The snippets could not be saved. The previous saved list has been kept.") from exc
