"""Browse one verified saved archive without writing its contents to the gallery."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import struct
import zipfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, closing, contextmanager

from PIL import Image

from . import bundle_archive, bundle_crypto, bundle_images, bundle_leases, bundles, storage
from .bundle_common import BrowseUnavailable, BundleError, _image_rows, _read_masks
from .metadata import PNG_SIGNATURE

PAGE_SIZE = 24
# Decoding a source on demand is serialized for each view. Even a reindexed
# external PNG cannot make a single thumbnail decode exceed this pixel budget.
_THUMB_SOURCE_PIXELS = 32 * 1024 * 1024


def _catalogue(data: bytes) -> tuple[list[dict], dict[str, dict]]:
    with closing(sqlite3.connect(":memory:")) as conn:
        try:
            conn.deserialize(data)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA trusted_schema=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA query_only=ON")
            tables = {row["name"]: row["type"] for row in conn.execute("PRAGMA table_list")}
            if tables.get("images") != "table" or tables.get("masks", "table") != "table":
                raise BundleError("the bundle has no usable AuraDream gallery tables")
            rows = _image_rows(conn)
            if (len({row["id"] for row in rows}) != len(rows)
                    or len({row["filename"].casefold() for row in rows}) != len(rows)):
                raise BundleError("the bundle contains duplicate image IDs or filenames")
            return rows, _read_masks(conn)
        except sqlite3.Error as exc:
            raise BundleError("cannot read the bundled gallery database") from exc


class BrowseArchive:
    """Archive-local records and an open source borrowed for the view's lifetime.

    The caller serializes reads and close. Neither image files nor thumbnails
    are retained between requests.
    """

    def __init__(self, archive: zipfile.ZipFile, rows: list[dict], files: dict, inspected: dict):
        self._archive = archive
        self._records = {}
        self._sources = {}
        for row in reversed(rows):  # The database validator returns ascending IDs.
            image_id = row["id"]
            member = "images/" + row["filename"]
            # Old database schemas lack newer columns. Record only values this
            # archive carries, including supported intent recovered from its PNG.
            values = {key: row.get(key) for key in storage._COLUMNS}
            pixels = inspected[member]
            values.update(id=image_id, execution=pixels.execution, edit=pixels.edit, content_hash=pixels.hash)
            for key in ("vram_peak_allocated", "vram_peak_reserved"):
                values[key] = storage._peak(values[key])
            record = storage._row_to_record(values, lookup_thumbnail=False)
            settings = record.model_dump(mode="json", include=set(storage._COLUMNS), exclude_none=True)
            # A browser Number cannot represent the full 63-bit seed range.
            if abs(record.seed) > 2**53 - 1:
                settings["seed"] = str(record.seed)
            self._records[image_id] = {"id": image_id, "prompt": record.prompt, "run_id": record.run_id,
                                       "favorite": record.favorite, "settings": settings}
            self._sources[image_id] = (member, files[member]["sha256"])

    def images(self, prompt: str, run: str, starred: bool, offset: int) -> dict:
        prompt, run = prompt.casefold(), run.casefold()
        total = 0
        page = []
        for record in self._records.values():
            if (starred and not record["favorite"] or prompt not in record["prompt"].casefold()
                    or run not in (record["run_id"] or "").casefold()):
                continue
            if offset <= total < offset + PAGE_SIZE:
                page.append(record)
            total += 1
        return {"total": total, "images": page}

    def thumbnail(self, image_id: int) -> bytes:
        member, expected = self._sources[image_id]
        try:
            # Use the identical open ZIP, even if the path was replaced.
            # A non-cooperating editor can still modify its bytes in place.
            data = self._archive.read(member)
            if hashlib.sha256(data).hexdigest() != expected:
                raise BundleError("the saved bundle changed since it was opened")
            with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
                if image.width * image.height > _THUMB_SOURCE_PIXELS:
                    raise BrowseUnavailable("this image is too large for a bundle thumbnail")
                image.thumbnail((storage.THUMB_SIDE, storage.THUMB_SIDE))
                with io.BytesIO() as result:
                    image.info.clear()
                    image.save(result, format="WEBP", quality=82)
                    return result.getvalue()
        except (OSError, zipfile.BadZipFile, bundle_crypto.CryptoError, Image.DecompressionBombError) as exc:
            raise BundleError("the saved bundle thumbnail could not be read; reopen the bundle") from exc


@contextmanager
def open_view(name: str, password: str | None, *,
              cancelled: Callable[[], bool] = lambda: False) -> Iterator[BrowseArchive]:
    """Pin, unlock and verify only the chosen archive. Release everything on exit."""
    def progress(_phase: str, _done: int, _total: int) -> None:
        if cancelled():
            raise BrowseUnavailable("the bundle view was closed")

    def inspect(row: dict, data: bytes):
        if data.startswith(PNG_SIGNATURE) and data[12:16] == b"IHDR" and len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            if width * height > _THUMB_SOURCE_PIXELS:
                raise BrowseUnavailable("this bundle contains an image too large to browse")
        return bundle_images._inspect_source(row, data)

    progress("", 0, 0)
    path = bundles.bundle_path(name)
    with ExitStack() as scope:
        owner = scope.enter_context(bundle_leases.reading(path))
        if not path.is_file():
            raise FileNotFoundError("Bundle file not found.")
        try:
            archive = scope.enter_context(bundle_archive._open_archive(path, password, read_owner=owner))
            del password
            manifest, rows, inspections = bundle_archive._verify_archive(
                path, opened=archive, catalogue=_catalogue, inspect=inspect, progress=progress,
            )
        except (zipfile.BadZipFile, EOFError) as exc:
            raise BundleError("the saved bundle is damaged") from exc
        except bundle_crypto.WrongPassword:
            raise BrowseUnavailable("the bundle password is incorrect") from None
        except bundle_crypto.CryptoError as exc:
            raise BundleError(str(exc)) from exc
        assert rows is not None
        progress("", 0, 0)
        view = BrowseArchive(archive, rows, manifest["files"], inspections)
        del rows, manifest, inspections
        yield view
