"""Rebuild gallery rows from the PNGs' own metadata.

    uv run python -m backend.reindex                  # outputs/ in place
    uv run python -m backend.reindex --from /mnt/d/wsl-outputs   # copy in and index
    uv run python -m backend.reindex --from D:\\backup\\outputs --database D:\\backup\\auradream.db
    uv run python -m backend.reindex --dry-run

Every image carries its whole reproduction set in an `auradream` text chunk, and
nothing could read it back: auradream.db was a single point of failure for an
otherwise self-describing gallery, the WSL and Windows galleries could never be
merged, and a row whose database went missing could never be repaired. This
walks a directory of PNGs, verifies already indexed filenames against their
retained pixels, copies new files in with the same `.part`-then-rename
discipline `save_image` uses, and inserts through `storage` so the search index
is built with the rows.

With ``--database``, rows come from that gallery's database, so favorites and
durations survive too. Gallery ids are local to a database; reference-image
and refine-source ids are remapped as the source rows are merged in ascending
id order. PNG-only imports use each source's recorded filename, pixel hash and
size to recover its new id; legacy links without this provenance are dropped
and counted, never treated as destination ids. Without a database, the
favorite cannot come back and duration is recorded as 0 so the
time estimate ignores it; nor can the recycle bin, so an image that was in
it comes back live. The timestamp is parsed from the filename prefix as
UTC, which is how it was written; failing that, the file's own mtime.

An inpaint's mask is a picture of its own, named by its hash and so never
remapped. Masks come back from `masks/*.png` beside the images (a file
whose pixels are not the hash it is named by is skipped and counted), from
the source database's `masks` table with them, and - where the file is gone -
from the copy every result carries in its `auradream-mask` chunk. A
result whose mask is found nowhere keeps its hash, which still says what
ran, and is counted.

Not `import.py`: `from backend import import` is a SyntaxError, so no test
could follow the house style. Running the file directly works too - see the
sys.path bootstrap below.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, closing, contextmanager
from datetime import datetime, timezone
from functools import cache, partial
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image

    from backend import gallery_lock, png_settings, storage, tiled_refine_contract as tiled
    from backend.bundle_common import _DB_REQUIRED
    from backend.config import settings
    from backend.controlnet_admission import guide_matches_path
    from backend.controlnet_contract import ControlNetSpec
    from backend.execution import canonical_execution, execution_chunk, execution_from_json, execution_key
    from backend.instruction_edit import InstructionEditSource, StoredInstructionEdit, descriptor_from_image, edit_key
    from backend.detailer import StoredDetailer, iter_rebuilt_masks, region_mask, stored_json as detailer_json
    from backend.metadata import (
        PNG_METADATA_KEY,  # noqa: F401 - re-exported; tests read reindex.PNG_METADATA_KEY
        PNG_METADATA_KEYS, PNG_SIGNATURE, canonical_edit, edit_chunk, edit_from_info,
        edit_from_json, edit_from_sources, embedded_metadata, execution_from_info,
        execution_from_sources, metadata_chunk, require_header_metadata, text_chunk, text_keyword,
        validate_edit_record,
        canonical_detailer, detailer_chunk, detailer_from_info, detailer_from_json, detailer_from_sources,
        validate_detailer_record,
        canonical_tiled_refine, tiled_refine_chunk, tiled_refine_from_info, tiled_refine_from_json,
        tiled_refine_from_sources, validate_tiled_refine_record,
        validate_import_record, validate_import_sources, validate_import_sources_from_info,
    )
    from backend.metadata import conditioning_from_info
    from backend.spatial_conditioning import (
        canonical_conditioning, conditioning_chunk, conditioning_from_json, conditioning_from_sources,
        conditioning_json, conditioning_key, validate_record as validate_conditioning_record,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
    sys.exit(
        f"Missing dependency: {exc.name}\n\n"
        f"Running under: {sys.executable}\n"
        "That interpreter is not the project environment. Use:\n\n"
        "    uv run python -m backend.reindex\n"
    )

# `20260907-235118-42-6a1b2c.png`: the UTC timestamp save_image writes first.
_NAME = re.compile(r"^(\d{8})-(\d{6})-")

# The keys an embedded metadata block must carry to make a row at all.
_REQUIRED = ("prompt", "steps", "guidance_scale", "seed", "model")

# The PNG chunk codec and the recipe reconciliation live in `metadata`, the
# mask decoders in `record_fields` (as `storage.<name>`), and the required
# columns in `bundle_common`, where the bundle layer reads them without
# importing this CLI; the old private names stay for this module and its
# tests.
_PNG_SIGNATURE = PNG_SIGNATURE
_text_keyword = text_keyword
_execution_from_sources = execution_from_sources
_execution_from_info = execution_from_info
_open_mask = storage.open_mask
_embedded_mask = storage.embedded_mask_picture


def _created_at(path: Path) -> datetime:
    match = _NAME.match(path.name)
    if match:
        try:
            return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _read(path: Path, *, require_metadata: bool = True) -> tuple[dict | None, tuple[int, int], Image.Image | None]:
    """The embedded record, the size and a copy of the image for its thumbnail.

    `.info`, not `.text`: the latter is a property that decodes every pixel to
    answer a question the header settles.
    """
    with path.open("rb") as stream:
        require_header_metadata(stream)
        with Image.open(stream) as image:
            raw = embedded_metadata({key: image.info.get(key) for key in PNG_METADATA_KEYS})
            size = image.size
            if not raw:
                return None, size, image.copy()
            try:
                meta = json.loads(raw)
            except ValueError:
                return None, size, image.copy()
            if not isinstance(meta, dict) or (require_metadata and any(meta.get(key) is None for key in _REQUIRED)):
                return None, size, image.copy()
            return meta, size, image.copy()


def _edit_metadata(path: Path) -> dict | None:
    """Read the app header without decoding pixels, before recovery mutation."""
    try:
        with path.open("rb") as stream:
            require_header_metadata(stream)
            with Image.open(stream) as image:
                raw = embedded_metadata({key: image.info.get(key) for key in PNG_METADATA_KEYS})
                value = json.loads(raw) if raw else None
                return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        # Ordinary unreadable images retain the existing skip/report behavior.
        # A SQL edit with no readable PNG representation is refused below.
        return None


def _preflight_passes(source: Path, rows: list[dict] | None = None) -> set[str]:
    """Check durable intent and collect retained roots needed by PNG recovery."""
    tiled_roots: set[str] = set()
    if rows is None:
        for path in source.glob("*.png"):
            meta = _edit_metadata(path)
            if meta is not None:
                validate_edit_record(canonical_edit(meta.get("edit")), meta)
                detailer = canonical_detailer(meta.get("detailer"))
                validate_detailer_record(detailer, meta)
                _verify_detailer_masks(detailer)
                tiled_refine = canonical_tiled_refine(meta.get("tiled_refine"))
                validate_tiled_refine_record(tiled_refine, meta)
                validate_conditioning_record(canonical_conditioning(meta.get("spatial_conditioning")), meta)
                if tiled_refine is not None and tiled_refine.source is not None:
                    tiled_roots.add(tiled_refine.root_source.filename)
        return tiled_roots
    for row in rows:
        stored = edit_from_json(row.get("edit"))
        detailer = detailer_from_json(row.get("detailer"))
        tiled_refine = tiled_refine_from_sources(row, None)
        spatial = conditioning_from_sources(row, None)
        validate_detailer_record(detailer, row)
        _verify_detailer_masks(detailer)
        filename = row.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            continue  # The normal row validator reports unsafe ordinary names.
        path = source / filename
        if path.is_file():
            meta = _edit_metadata(path)
            if meta is not None or stored is not None:
                edit_from_sources(row, meta or {})
            if meta is not None or detailer is not None:
                detailer_from_sources(row, meta or {})
            if meta is not None or tiled_refine is not None:
                tiled_refine_from_sources(row, meta or {})
            if meta is not None or spatial is not None:
                conditioning_from_sources(row, meta or {})
    return tiled_roots


def _preflight_retained_passes(retained: dict[str, dict], outputs: Path) -> None:
    for filename, row in retained.items():
        path = outputs / filename
        # A missing ordinary destination retains its normal conflict behavior;
        # an edit cannot claim its durable PNG representation is still present.
        if path.is_file() or any(row.get(key) is not None for key in (
            "edit", "detailer", "tiled_refine", "spatial_conditioning",
        )):
            meta = _edit_metadata(path) or {}
            edit_from_sources(row, meta)
            detailer_from_sources(row, meta)
            tiled_refine_from_sources(row, meta)
            conditioning_from_sources(row, meta)


def _verify_detailer_masks(stored: StoredDetailer | None) -> None:
    if stored is not None:
        for _digest, picture in iter_rebuilt_masks(stored.plan):
            picture.close()


def _proves_detailer(path: Path, stored: StoredDetailer) -> bool:
    try:
        if path.is_symlink() or path.is_junction():
            return False
        with Image.open(path) as image:
            return descriptor_from_image(path.name, image) == stored.plan.source
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def _remapped_detailer(stored: StoredDetailer | None, source: int | None, counts: dict[str, int]) -> StoredDetailer | None:
    if stored is None:
        return None
    if source is None:
        counts["unresolved_detailers"] += 1
    return StoredDetailer(plan=stored.plan.model_copy(update={"source_id": source}), pass_index=stored.pass_index)


def _same_detailer(left: StoredDetailer | None, right: StoredDetailer | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.model_dump(exclude={"plan": {"source_id"}}) == right.model_dump(exclude={"plan": {"source_id"}})


def _proves_edit(path: Path, stored: StoredInstructionEdit) -> bool:
    """A source's full decoded RGB identity, never an old row's short digest."""
    try:
        if path.is_symlink() or path.is_junction():
            return False
        with Image.open(path) as image:
            return descriptor_from_image(path.name, image) == stored.edit_source
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def _remapped_edit(stored: StoredInstructionEdit | None, source: int | None, counts: dict[str, int]) -> StoredInstructionEdit | None:
    if stored is None:
        return None
    if source is None:
        counts["unresolved_edits"] += 1
    return StoredInstructionEdit(source=source, recipe=stored.recipe, edit_source=stored.edit_source)


def _same_edit(left: StoredInstructionEdit | None, right: StoredInstructionEdit | None) -> bool:
    """Local ids may differ across galleries; intent and complete proof cannot."""
    if left is None or right is None:
        return left is right
    return (left.recipe, left.edit_source) == (right.recipe, right.edit_source)


def _prepared_proof(path: Path, expected_hash: str) -> str | None:
    """Prove the original RGB PNG before building its frozen prepared canvas."""
    try:
        if path.is_symlink() or path.is_junction():
            return None
        with Image.open(path) as image:
            if image.format != "PNG" or tiled.descriptor_from_image(path.name, image).hash != expected_hash:
                return None
            with closing(tiled.prepare_canvas(image)) as prepared:
                return tiled.rgb_sha256(prepared)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _remapped_tiled(
    stored: tiled.StoredTiledRefine | None, source: int | None, counts: dict[str, int],
) -> tiled.StoredTiledRefine | None:
    if stored is None:
        return None
    if source is None:
        counts["unresolved_tiled_refines"] += 1
    return stored.model_copy(update={"source": source})


def _matches_tiled_root_row(row: dict, root: InstructionEditSource) -> bool:
    """A destination row must also pass the live gallery provenance check."""
    return (row["width"], row["height"]) == (root.width, root.height) and row.get("content_hash") == root.hash[:16]


def _same_tiled(
    retained: tiled.StoredTiledRefine | None, stored: tiled.StoredTiledRefine | None,
    known: dict, incoming: dict,
) -> bool:
    if retained is None or stored is None:
        return retained is stored
    return retained.model_dump(exclude={"source"}) == stored.model_dump(exclude={"source"}) and all(
        known.get(key) == incoming.get(key) for key in ("prompt", "negative_prompt", "seed")
    )


def _verify_retained_pixels(incoming: Image.Image, kept: Image.Image, row: dict, *, in_place: bool) -> None:
    """Prove same-name adoption independently of the PNG's metadata."""
    if kept.size != (row["width"], row["height"]):
        raise ValueError("the retained PNG dimensions disagree with its gallery row")
    if in_place:
        # The input and destination are the same file. The row's saved hash
        # is the independent check, where the writer recorded one.
        digest = row.get("content_hash")
        if digest is not None and storage.content_hash(incoming) != digest:
            raise ValueError("the retained PNG pixels disagree with its gallery row")
        return
    if (
        kept.size != incoming.size
        or kept.mode != incoming.mode
        or kept.getpalette(None) != incoming.getpalette(None)
        or kept.info.get("transparency") != incoming.info.get("transparency")
        or kept.tobytes() != incoming.tobytes()
    ):
        raise ValueError("different PNG pixels already use that filename")


def _dependency_order(rows: list[dict], cycles: set[int]) -> list[dict]:
    """Make a complete id map for forward edit sources as well as old links.

    Cycles cannot prove a link: the missing map entry stays unresolved. Visit
    ordinary reference/refine dependencies too, so an edit-source reorder
    cannot break an otherwise earlier ordinary dependency.
    """
    by_id = {row["id"]: row for row in rows}
    ordered: list[dict] = []
    visited: set[int] = set()
    visiting: set[int] = set()

    def visit(row: dict) -> None:
        identifier = row["id"]
        if identifier in visiting:
            cycles.update(visiting)
            return
        if identifier in visited:
            return
        visiting.add(identifier)
        stored = edit_from_json(row.get("edit"))
        detailer = detailer_from_json(row.get("detailer"))
        tiled_refine = tiled_refine_from_json(row.get("tiled_refine"))
        spatial = conditioning_from_json(row.get("spatial_conditioning"))
        links = [*_json_list(row.get("reference")), row.get("refine"), stored.source if stored else None,
                 detailer.plan.source_id if detailer is not None else None,
                 tiled_refine.source if tiled_refine is not None else None,
                 spatial.source if isinstance(spatial, ControlNetSpec) else None]
        for source_id in links:
            if type(source_id) is int and source_id in by_id:
                visit(by_id[source_id])
        visiting.remove(identifier)
        visited.add(identifier)
        ordered.append(row)

    for row in rows:
        visit(row)
    return ordered


def _with_auradream_metadata(data: bytes, meta: dict, parameters: str | None = None) -> bytes:
    """Replace app metadata without recompressing or changing pixels; with
    `parameters`, the A1111 block too (`png_settings.PARAMETERS_KEY`)."""
    if not data.startswith(_PNG_SIGNATURE):
        raise OSError("not a PNG")
    replacement = metadata_chunk(meta)
    owned = {key.encode("ascii") for key in PNG_METADATA_KEYS}
    if parameters is not None:
        replacement += text_chunk(png_settings.PARAMETERS_KEY, parameters)
        owned.add(png_settings.PARAMETERS_KEY.encode("ascii"))
    out = bytearray(_PNG_SIGNATURE)
    offset = len(_PNG_SIGNATURE)
    inserted = False
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        end = offset + 12 + length
        if end > len(data):
            raise OSError("truncated PNG")
        kind = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        if _text_keyword(kind, chunk_data) in owned:
            offset = end
            continue
        # Pillow reads metadata preceding the image data during Image.open().
        # A text chunk immediately before IEND is legal but stays invisible
        # until load(), which would make the gallery's normal header-only read
        # miss it.
        if kind in {b"IDAT", b"IEND"} and not inserted:
            out.extend(replacement)
            inserted = True
        out.extend(data[offset:end])
        offset = end
        if kind == b"IEND":
            break
    if not inserted:
        raise OSError("PNG has no IEND chunk")
    return bytes(out)


def _copy_in(source: Path, target: Path, meta: dict | None = None, parameters: str | None = None) -> None:
    """Copy into outputs under a temporary name, optionally replacing app
    metadata (and, with it, the A1111 block)."""
    partial = target.with_name(target.name + ".part")
    data = source.read_bytes()
    partial.write_bytes(
        _with_auradream_metadata(data, meta, parameters) if meta is not None else data
    )
    partial.replace(target)


def _read_embedded_mask(path: Path, digest: str) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            # Like _read, include PNG text chunks that follow IDAT.
            image.load()
            return _embedded_mask(image, digest)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


class _Masks:
    """The masks a reindex can find, and the ones it put back.

    Keep the verified path or recipe for each restored mask. Crop proofs can
    read one back even in a dry run, which writes no mask files. The decoded
    pictures are used and closed one at a time.
    """

    def __init__(
        self, counts: dict[str, int], dry_run: bool, out,
        mask_path: Callable[[str], Path | None],
    ):
        self.counts = counts
        self.dry_run = dry_run
        self.out = out
        self.mask_path = mask_path
        self.restored: dict[str, Callable[[], Image.Image | None]] = {}

    def held(self, digest: str) -> bool:
        return digest in self.restored or self.mask_path(digest) is not None

    def _restore(
        self, digest: str, mask: Image.Image, read_again: Callable[[], Image.Image | None],
    ) -> None:
        if not self.dry_run:
            storage.store_mask(mask)
        self.restored[digest] = read_again
        self.counts["masks"] += 1

    def detailer(self, stored: StoredDetailer | None) -> None:
        if stored is None:
            return
        plan = stored.plan
        for (digest, picture), region in zip(iter_rebuilt_masks(plan), plan.accepted_regions, strict=True):
            with closing(picture):
                if not self.held(digest):
                    read_again = partial(region_mask, plan.source.width, plan.source.height, region,
                                         plan.padding, plan.feather)
                    self._restore(digest, picture, read_again)
                    self.counts["detailer_masks_rebuilt"] += 1

    def from_directory(self, directory: Path) -> None:
        """Every `masks/*.png` named by the hash of its own pixels."""
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.png")):
            digest = path.stem
            if storage._MASK_HASH.fullmatch(digest) is None or (mask := _open_mask(path, digest)) is None:
                self.counts["bad_masks"] += 1
                print(f"  skip  masks/{path.name}: not the mask its name says", file=self.out)
                continue
            with closing(mask):
                if not self.held(digest):
                    self._restore(digest, mask, partial(_open_mask, path, digest))

    def _picture(
        self, digest: str, directory: Path, image: Image.Image | None,
        outpaint: storage.OutpaintSpec | None, size: tuple[int, int],
    ) -> Image.Image | None:
        """Open a restored or otherwise available mask to prove a crop region."""
        if digest in self.restored and (mask := self.restored[digest]()) is not None:
            return mask
        path = directory / f"{digest}.png"
        mask = _open_mask(path, digest) if path.is_file() else None
        if mask is None and (held := self.mask_path(digest)) is not None:
            mask = _open_mask(held, digest)
        if mask is None:
            mask = _embedded_mask(image, digest)
        if mask is None and outpaint is not None:
            mask = storage.rebuilt_outpaint_mask(*size, outpaint, digest)
        return mask

    def resolve(
        self, digest: str | None, directory: Path, result: Path, image: Image.Image | None,
        outpaint: storage.OutpaintSpec | None, size: tuple[int, int],
    ) -> None:
        """Make sure a result's mask is held: from its file in `directory`,
        else from the result's own embedded copy, else - an outpaint's - built
        again from its spec at the result's `size` where it hashes to the
        name; counted where none of them."""
        if digest is None or self.held(digest):
            return
        path = directory / f"{digest}.png"
        read_again = partial(_open_mask, path, digest)
        mask = read_again() if path.is_file() else None
        if mask is None:
            mask = _embedded_mask(image, digest)
            read_again = partial(_read_embedded_mask, result, digest)
        if mask is None and outpaint is not None:
            read_again = partial(storage.rebuilt_outpaint_mask, *size, outpaint, digest)
            mask = read_again()
            if mask is not None:
                self.counts["outpaint_masks_rebuilt"] += 1
        if mask is None:
            self.counts["unresolved_masks"] += 1
            print(f"  note  mask {digest[:16]} was found neither in masks/ nor in its result", file=self.out)
            return
        with closing(mask):
            self._restore(digest, mask, read_again)

    def crop(
        self, digest: str | None, recorded, malformed: bool, directory: Path,
        image: Image.Image | None, outpaint: storage.OutpaintSpec | None, size: tuple[int, int],
    ):
        painted = (
            self._picture(digest, directory, image, outpaint, size)
            if digest is not None and not malformed and _names_a_mask_crop(recorded, image) else None
        )
        try:
            return _mask_crop(digest, recorded, malformed, painted, image, self.counts)
        finally:
            if painted is not None:
                painted.close()


@contextmanager
def _connect_read(database: Path) -> Iterator[sqlite3.Connection]:
    database = database.resolve()
    wal = database.with_name(database.name + "-wal")
    has_wal = wal.exists()
    with ExitStack() as resources:
        if has_wal and not database.with_name(database.name + "-shm").exists():
            # A database and WAL from a backup have no shared-memory index.
            # Even a read-only SQLite connection would build it alongside them.
            snapshot = Path(resources.enter_context(TemporaryDirectory())) / database.name
            copyfile(database, snapshot)
            copyfile(wal, snapshot.with_name(wal.name))
            database = snapshot
        uri = database.as_uri() + "?mode=ro"
        if not has_wal:
            # A checkpointed WAL-mode database needs no sidecars for this read.
            uri += "&immutable=1"
        conn = resources.enter_context(closing(sqlite3.connect(uri, uri=True, timeout=15)))
        conn.row_factory = sqlite3.Row
        yield conn


class _Destination:
    """Snapshot of the existing rows this recovery will need, without migration."""

    def __init__(self, database: Path, outputs: Path, filenames: Iterable[str]):
        self.outputs = outputs
        self.existing: dict[str, int] = {}
        self.retained: dict[str, dict] = {}
        self.mask_hashes: set[str] = set()
        if not database.exists():
            return
        wanted = set(filenames)
        with _connect_read(database) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if "images" in tables:
                for row in conn.execute("SELECT * FROM images"):
                    filename = row["filename"]
                    self.existing[filename] = row["id"]
                    if filename in wanted:
                        self.retained[filename] = dict(row)
            if "masks" in tables:
                self.mask_hashes = {row[0] for row in conn.execute("SELECT hash FROM masks")}

    def mask_path(self, digest: str) -> Path | None:
        if digest not in self.mask_hashes:
            return None
        path = self.outputs / storage.MASKS_DIR / f"{digest}.png"
        return path if path.is_file() else None


def _source_rows(database: Path) -> list[dict]:
    """Read another app database without migrating or otherwise changing it."""
    try:
        with _connect_read(database) as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
            missing = sorted(_DB_REQUIRED - columns)
            if missing:
                raise ValueError("source database has no usable images table; missing " + ", ".join(missing))
            return [dict(row) for row in conn.execute("SELECT * FROM images ORDER BY id")]
    except (OSError, sqlite3.Error) as exc:
        raise ValueError(f"cannot read source database: {exc}") from exc


def _json_list(value) -> list:
    if value is None:
        return []
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def _outpaint(mask: str | None, spec, image: Image.Image | None, width: int, height: int,
              source_size: tuple[int, int] | None, counts: dict[str, int]):
    """A result's outpaint as this reindex records it: its spec only beside
    a mask (and so a source); where none came with it, one an older build's
    import stripped, from the A1111 line, where the rebuilt mask is the
    row's (`storage.recovered_outpaint`), counted."""
    if mask is None:
        return None
    if spec is not None:
        return spec
    parameters = image.info.get("parameters") if image is not None else None
    recovered = storage.recovered_outpaint(parameters, width, height, mask, source_size)
    if recovered is not None:
        counts["outpaints_recovered"] += 1
    return recovered


def _mask_crop(mask: str | None, recorded, malformed: bool, painted, image: Image.Image | None,
               counts: dict[str, int]):
    """A result's masked area as this reindex records it, `(spec, region,
    parameters)`, `(None, None, ...)` for none: only beside a mask; its
    recorded region proved against the mask under the frozen rule
    (`storage.proven_mask_crop`), drawn again
    where none was recorded, and left off - counted malformed - where it is
    not the mask's; a spec this build cannot read is left off and counted
    too. Where none came with it, one an older build's import stripped, from
    the A1111 line, its region drawn from the mask
    (`storage.recovered_mask_crop`), counted. Without the mask to read, a
    recorded spec and region are kept as given.

    `parameters` is the A1111 block to write in place of the PNG's, or None
    to keep it: one left off takes its `Mask crop:` out of the settings
    line (`storage.without_mask_crop_parameters`), which would otherwise
    give it back - drawn again from the mask it did not match - on the next
    reindex or import."""
    parameters = image.info.get("parameters") if image is not None else None
    if mask is None:
        return None, None, None
    if malformed:
        counts["mask_crop_malformed"] += 1
        return None, None, storage.without_mask_crop_parameters(parameters)
    if recorded is not None:
        spec, region = recorded
        if painted is None:
            return spec, region, None
        proven = storage.proven_mask_crop(spec, painted, region)
        if proven is None:
            counts["mask_crop_malformed"] += 1
            return None, None, storage.without_mask_crop_parameters(parameters)
        return spec, proven, None
    recovered = storage.recovered_mask_crop(parameters, painted)
    if recovered is None:
        return None, None, None
    counts["mask_crops_recovered"] += 1
    return (*recovered, None)


def _names_a_mask_crop(recorded, image: Image.Image | None) -> bool:
    """Whether a row has a masked area to prove or recover, so its mask is
    worth reading as a picture."""
    if recorded is not None:
        return True
    parameters = image.info.get("parameters") if image is not None else None
    return storage.parameters_mask_crop(parameters) is not None


def _provenance_size(provenance) -> tuple[int, int] | None:
    if not isinstance(provenance, dict):
        return None
    width, height = provenance.get("width"), provenance.get("height")
    return (width, height) if type(width) is int and type(height) is int else None


def _database_meta(
    row: dict, width: int, height: int, digest: str, references: list[int], refine: int | None,
    mask: str | None = None, outpaint=None, mask_crop=None,
) -> dict:
    try:
        variables = json.loads(row["variables"]) if row.get("variables") else None
    except (TypeError, ValueError):
        variables = None
    upscaler = storage._upscaler(
        row.get("upscaler"), source=refine, model_id=row.get("model_id"), mask=mask, outpaint=outpaint,
    )
    crop_anchor = png_settings.crop_anchor(
        row.get("crop_anchor") if refine is not None else None,
        source=refine, mask=mask, outpaint=outpaint,
    )
    return {
        "prompt": row["prompt"],
        **({"imported_name": row["imported_name"]} if row.get("imported_name") is not None else {}),
        "negative_prompt": row.get("negative_prompt"),
        "steps": row["steps"],
        "guidance_scale": row["guidance_scale"],
        "true_cfg": row.get("true_cfg"),
        "mu": row.get("mu"),
        "seed": row["seed"],
        "model": row["model_id"],
        "quantization": row.get("quantization"),
        "job": row.get("job_id"),
        "run": row.get("run_id"),
        "size": f"{width}x{height}",
        "variables": variables if isinstance(variables, dict) else None,
        "hash": digest,
        "reference": references or None,
        "reference_sources": [storage.image_provenance(i) for i in references] or None,
        "lora": row.get("lora"),
        "lora_scale": row.get("lora_scale") if row.get("lora") else None,
        "refine": refine,
        "refine_source": storage.image_provenance(refine),
        "strength": row.get("strength") if refine is not None else None,
        "mask": mask,
        "mask_source": storage.mask_provenance(mask, width, height),
        "mask_blend": "overlay" if mask is not None else None,
        "outpaint": storage.outpaint_chunk(outpaint),
        "outpaint_fill": outpaint.fill if outpaint is not None else None,
        "batch": storage._batch(row.get("batch")),
        **({"crop_anchor": crop_anchor} if crop_anchor is not None else {}),
        **({"mask_crop": storage.mask_crop_chunk(*mask_crop)} if mask_crop is not None else {}),
        **({"upscaler": upscaler} if upscaler is not None else {}),
        **execution_chunk(execution_from_json(row.get("execution"))),
        **edit_chunk(edit_from_json(row.get("edit"))),
        **detailer_chunk(detailer_from_json(row.get("detailer"))),
        **tiled_refine_chunk(tiled_refine_from_json(row.get("tiled_refine"))),
        **conditioning_chunk(conditioning_from_json(row.get("spatial_conditioning"))),
        **storage.peak_chunk(row),
    }


@gallery_lock.protected(lambda: settings.db_path)
def merge_database(
    source: Path, database: Path, dry_run: bool = False, out=sys.stdout
) -> dict[str, int]:
    """Merge the files and rows of another gallery into the configured one."""
    counts = {
        "indexed": 0,
        "present": 0,
        "missing": 0,
        "invalid": 0,
        "conflicts": 0,
        "unresolved_references": 0,
        "unresolved_edits": 0,
        "unresolved_detailers": 0,
        "unresolved_tiled_refines": 0,
        "detailer_masks_rebuilt": 0,
        "masks": 0,
        "bad_masks": 0,
        "unresolved_masks": 0,
        "outpaint_malformed": 0,
        "outpaints_recovered": 0,
        "outpaint_masks_rebuilt": 0,
        "mask_crop_malformed": 0,
        "mask_crops_recovered": 0,
    }
    rows = _source_rows(database)
    sizes = {row["id"]: (row["width"], row["height"]) for row in rows}
    outputs = settings.outputs_dir.resolve()
    source = source.resolve()
    _preflight_passes(source, rows)
    edit_cycles: set[int] = set()
    rows = _dependency_order(rows, edit_cycles)
    source_rows = {row["id"]: row for row in rows}
    destination = _Destination(settings.db_path, outputs, (
        row["filename"] for row in rows
        if isinstance(row.get("filename"), str) and Path(row["filename"]).name == row["filename"]
    ))
    existing = destination.existing
    _preflight_retained_passes(destination.retained, outputs)
    prepared_proof = cache(_prepared_proof)
    id_map: dict[int, int] = {}
    if not dry_run:
        storage.init_db()
    # The masks first, every one the other gallery's directory holds, as
    # its `masks` table would list them: a mask is named by its pixels, so
    # its file is the record, and a result below then finds it held.
    masks = _Masks(counts, dry_run, out, destination.mask_path)
    masks.from_directory(source / storage.MASKS_DIR)

    for row in rows:
        filename = row["filename"]
        source_id = int(row["id"])
        try:
            execution_from_json(row.get("execution"))
            edit_from_json(row.get("edit"))
            detailer_from_json(row.get("detailer"))
            png_settings.crop_anchor(
                row.get("crop_anchor"), source=row.get("refine"),
                mask=row.get("mask"), outpaint=row.get("outpaint"),
            )
        except ValueError as exc:
            counts["invalid"] += 1
            print(f"  skip  source row {source_id}: {exc}", file=out)
            continue
        if row.get("upscaler") is not None and storage._upscaler(
            row["upscaler"], source=row.get("refine"), model_id=row.get("model_id"),
            mask=row.get("mask"), outpaint=row.get("outpaint"),
        ) is None:
            counts["invalid"] += 1
            print(f"  skip  source row {source_id}: invalid upscaler for its Refine pass", file=out)
            continue
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".png"
        ):
            counts["invalid"] += 1
            print(f"  skip  source row {source_id}: unsafe PNG filename {filename!r}", file=out)
            continue
        path = source / filename
        if not path.is_file():
            counts["missing"] += 1
            print(f"  skip  {filename}: source image is missing", file=out)
            continue
        target = outputs / filename
        if filename not in existing and source != outputs and target.exists():
            counts["conflicts"] += 1
            print(f"  skip  {filename}: a file of that name is already in outputs", file=out)
            continue
        try:
            png_meta, (width, height), image = _read(path, require_metadata=False)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            counts["invalid"] += 1
            print(f"  skip  {filename}: not a readable PNG: {exc}", file=out)
            continue
        if image is None:  # Defensive: database imports do not require embedded metadata.
            counts["invalid"] += 1
            print(f"  skip  {filename}: not a readable PNG", file=out)
            continue
        try:
            execution = _execution_from_sources(row, png_meta)
            stored_edit = edit_from_sources(row, png_meta or {})
            stored_detailer = detailer_from_sources(row, png_meta or {})
            stored_tiled = tiled_refine_from_sources(row, png_meta or {})
            validate_tiled_refine_record(stored_tiled, {**row, "width": width, "height": height})
            stored_spatial = conditioning_from_sources(row, png_meta or {})
            validate_conditioning_record(stored_spatial, {**row, "width": width, "height": height})
            validate_import_sources(row, png_meta or {})
        except ValueError as exc:
            counts["invalid"] += 1
            print(f"  skip  {filename}: invalid execution metadata: {exc}", file=out)
            continue
        if filename in existing:
            try:
                known = destination.retained[filename]
                with target.open("rb") as stream:
                    require_header_metadata(stream)
                    with Image.open(stream) as kept:
                        if kept.format != "PNG":
                            raise ValueError("the retained image is not a PNG")
                        info = {key: kept.info.get(key) for key in PNG_METADATA_KEYS}
                        retained = _execution_from_info({"execution": known.get("execution")}, info)
                        retained_edit = edit_from_info(known, info)
                        retained_detailer = detailer_from_info(known, info)
                        retained_tiled = tiled_refine_from_info(known, info)
                        retained_spatial = conditioning_from_info(known, info)
                        validate_import_sources_from_info(known, info)
                        _verify_retained_pixels(image, kept, known, in_place=source == outputs)
                if known.get("imported_name") != row.get("imported_name"):
                    raise ValueError("different imported image metadata already uses that filename")
                if retained != execution:
                    raise ValueError("different execution already uses that filename")
                if not _same_edit(retained_edit, stored_edit) or (
                    stored_edit is not None and known["prompt"] != row["prompt"]
                ):
                    raise ValueError("different instruction edit already uses that filename")
                if not _same_detailer(retained_detailer, stored_detailer):
                    raise ValueError("different detailer manifest already uses that filename")
                if not _same_tiled(retained_tiled, stored_tiled, known, row):
                    raise ValueError("different Tiled Refine intent already uses that filename")
                if conditioning_key(retained_spatial) != conditioning_key(stored_spatial):
                    raise ValueError("different spatial conditioning already uses that filename")
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                counts["conflicts"] += 1
                print(f"  skip  {filename}: cannot use the retained image: {exc}", file=out)
                continue
            counts["present"] += 1
            id_map[source_id] = existing[filename]
            continue
        # A recovered PNG-only recipe becomes the row's value too. The
        # source database is read-only and is never migrated or repaired.
        edit_source = None
        if stored_edit is not None and stored_edit.source is not None and source_id not in edit_cycles:
            original_source = source_rows.get(stored_edit.source)
            mapped_source = id_map.get(stored_edit.source)
            if (
                original_source is not None and original_source["filename"] == stored_edit.edit_source.filename
                and mapped_source is not None and _proves_edit(source / original_source["filename"], stored_edit)
                and (dry_run or _proves_edit(outputs / original_source["filename"], stored_edit))
            ):
                edit_source = mapped_source
        remapped_edit = _remapped_edit(stored_edit, edit_source, counts)
        detailer_source = None
        if stored_detailer is not None and stored_detailer.plan.source_id is not None and source_id not in edit_cycles:
            original_source = source_rows.get(stored_detailer.plan.source_id)
            mapped_source = id_map.get(stored_detailer.plan.source_id)
            if (
                original_source is not None and original_source["filename"] == stored_detailer.plan.source.filename
                and mapped_source is not None and _proves_detailer(source / original_source["filename"], stored_detailer)
                and (dry_run or _proves_detailer(outputs / original_source["filename"], stored_detailer))
            ):
                detailer_source = mapped_source
        remapped_detailer = _remapped_detailer(stored_detailer, detailer_source, counts)
        tiled_source = None
        if stored_tiled is not None and stored_tiled.source is not None and source_id not in edit_cycles:
            original_source = source_rows.get(stored_tiled.source)
            mapped_source = id_map.get(stored_tiled.source)
            root = stored_tiled.root_source
            retained_root = destination.retained.get(root.filename)
            origin = source / root.filename
            target_root = origin if dry_run and retained_root is None else outputs / root.filename
            if (
                original_source is not None and original_source["filename"] == root.filename
                and mapped_source is not None
                and (retained_root is None or _matches_tiled_root_row(retained_root, root))
                and prepared_proof(origin, root.hash) == stored_tiled.prepared_rgb_sha256
                and prepared_proof(target_root, root.hash) == stored_tiled.prepared_rgb_sha256
            ):
                tiled_source = mapped_source
        remapped_tiled = _remapped_tiled(stored_tiled, tiled_source, counts)
        remapped_spatial = stored_spatial
        if isinstance(stored_spatial, ControlNetSpec):
            recorded_id = stored_spatial.source
            upstream = source_rows.get(recorded_id) if recorded_id is not None else None
            guide_id = id_map.get(recorded_id) if recorded_id is not None else None
            guide = stored_spatial.guide
            origin = source / guide.filename
            target_guide = origin if dry_run and guide.filename not in destination.retained else outputs / guide.filename
            if (upstream is None or upstream["filename"] != guide.filename or source_id in edit_cycles
                    or not guide_matches_path(origin, guide) or not guide_matches_path(target_guide, guide)):
                guide_id = None
            counts["unresolved_controlnet_guides"] = counts.get("unresolved_controlnet_guides", 0) + (guide_id is None)
            remapped_spatial = stored_spatial.model_copy(update={"source": guide_id})
        row = {**row, "execution": execution_key(execution), "edit": edit_key(remapped_edit),
               "detailer": detailer_json(remapped_detailer), "tiled_refine": tiled.tiled_refine_key(remapped_tiled),
               "spatial_conditioning": conditioning_json(remapped_spatial)}
        masks.detailer(remapped_detailer)

        source_references = [
            value for value in _json_list(row.get("reference")) if isinstance(value, int)
        ]
        references = []
        for reference in source_references:
            mapped = id_map.get(reference)
            if mapped is None:
                counts["unresolved_references"] += 1
            else:
                references.append(mapped)
        # A refine source is a gallery id too, and points backwards the same
        # way: remapped with the references, and dropped - with its strength,
        # and counted - where the image it names did not come across.
        refine = row.get("refine")
        if isinstance(refine, int):
            refine = id_map.get(refine)
            if refine is None:
                counts["unresolved_references"] += 1
        else:
            refine = None
        # A mask only beside the source it covers, as a hash and never
        # anything else; one this gallery cannot find keeps its hash.
        mask = storage._mask(row.get("mask")) if refine is not None else None
        # And an outpaint only beside them both, as its key.
        spec = None
        if mask is not None and row.get("outpaint") is not None:
            try:
                spec = storage.OutpaintSpec.parse_key(row["outpaint"])
            except ValueError:
                spec = None
            # One rule with the chunk's and the bundle's: a spec no request
            # could make (no side extended, a band past the bound) is left off.
            if spec is None or not spec.wellformed():
                spec = None
                counts["outpaint_malformed"] += 1
        spec = _outpaint(mask, spec, image, width, height, sizes.get(row.get("refine")), counts)
        masks.resolve(mask, source / storage.MASKS_DIR, path, image, spec, (width, height))
        # Only the masked area beside its mask, its region proved against it.
        recorded, malformed = None, False
        if mask is not None and row.get("mask_crop") is not None:
            crop_key = storage.stored_mask_crop(row.get("mask_crop"))
            if crop_key is None:
                malformed = True
            else:
                box = storage.stored_mask_crop_box(row.get("mask_crop_box"))
                recorded = (
                    storage.MaskCropSpec.parse_key(crop_key),
                    storage.parse_mask_crop_box(box) if box is not None else None,
                )
        crop_spec, crop_region, parameters = masks.crop(
            mask, recorded, malformed, source / storage.MASKS_DIR, image, spec, (width, height),
        )
        crop = (crop_spec, crop_region) if crop_spec is not None else None
        digest = storage.content_hash(image)
        counts["indexed"] += 1
        if dry_run:
            print(f"  would import  {filename}", file=out)
            # Source references always point backwards in a gallery. Keep a
            # placeholder mapping so the dry run reports only genuinely
            # missing/skipped references, not every row it merely did not add.
            id_map[source_id] = source_id
            continue

        meta = _database_meta(row, width, height, digest, references, refine, mask, spec, crop)
        values = {
            "filename": filename,
            "imported_name": row.get("imported_name"),
            "prompt": row["prompt"],
            "negative_prompt": row.get("negative_prompt"),
            "width": width,
            "height": height,
            "steps": int(row["steps"]),
            "guidance_scale": float(row["guidance_scale"]),
            "mu": row.get("mu"),
            "seed": int(row["seed"]),
            "model_id": row["model_id"],
            "duration_ms": int(row.get("duration_ms") or 0),
            "favorite": int(bool(row.get("favorite"))),
            "created_at": row.get("created_at") or _created_at(path).isoformat(),
            "job_id": row.get("job_id"),
            "run_id": row.get("run_id"),
            "quantization": row.get("quantization"),
            "variables": row.get("variables"),
            "true_cfg": row.get("true_cfg"),
            "content_hash": digest,
            "reference": json.dumps(references) if references else None,
            "lora": row.get("lora"),
            "lora_scale": row.get("lora_scale") if row.get("lora") else None,
            "refine": refine,
            "strength": row.get("strength") if refine is not None else None,
            "crop_anchor": meta.get("crop_anchor"),
            "upscaler": meta.get("upscaler"),
            "execution": execution_key(execution),
            "edit": edit_key(remapped_edit),
            "detailer": detailer_json(remapped_detailer),
            "tiled_refine": tiled.tiled_refine_key(remapped_tiled),
            "spatial_conditioning": conditioning_json(remapped_spatial),
            "batch": storage._batch(row.get("batch")),
            # The recycle bin is SQLite's alone, as the star is: carried with
            # the database, and None (live) from one that predates it.
            "binned_at": row.get("binned_at"),
            "mask": mask,
            "outpaint": spec.key() if spec is not None else None,
            "vram_peak_allocated": storage._peak(row.get("vram_peak_allocated")),
            "vram_peak_reserved": storage._peak(row.get("vram_peak_reserved")),
            "mask_crop": crop_spec.key() if crop_spec is not None else None,
            "mask_crop_box": storage.mask_crop_box(crop_region),
        }
        copied = source != outputs
        try:
            png_execution = canonical_execution(png_meta.get("execution") if png_meta is not None else None)
            png_crop = png_meta.get("mask_crop") if png_meta is not None else None
            if (copied or row.get("refine") != refine or row.get("crop_anchor") != meta.get("crop_anchor")
                    or execution != png_execution or remapped_edit != stored_edit or remapped_detailer != stored_detailer
                    or remapped_tiled != stored_tiled
                    or remapped_spatial != stored_spatial
                    or png_crop != meta.get("mask_crop") or parameters is not None):
                # A database rebuilt beside its original PNGs can renumber or
                # lose a source too. Keep --from-image's crop and source in
                # agreement with the new row, without rewriting an unchanged
                # legacy centered PNG in place.
                _copy_in(path, target, meta, parameters)
            record = storage.import_file(values)
        except BaseException:
            if copied:
                target.unlink(missing_ok=True)
            raise
        id_map[source_id] = record.id
        existing[filename] = record.id
        storage.thumbnail_for(image, filename)
    return counts


@gallery_lock.protected(lambda: settings.db_path)
def reindex(
    source: Path, dry_run: bool = False, out=sys.stdout, database: Path | None = None
) -> dict[str, int]:
    if database is not None:
        return merge_database(source, database, dry_run=dry_run, out=out)
    counts = {
        "indexed": 0, "present": 0, "no_metadata": 0, "conflicts": 0,
        "unresolved_references": 0, "masks": 0, "bad_masks": 0, "unresolved_masks": 0,
        "unresolved_edits": 0,
        "unresolved_detailers": 0, "detailer_masks_rebuilt": 0,
        "unresolved_tiled_refines": 0,
        "outpaint_malformed": 0, "outpaints_recovered": 0, "outpaint_masks_rebuilt": 0,
        "mask_crop_malformed": 0, "mask_crops_recovered": 0,
        "invalid_upscalers": 0,
        "invalid_crop_anchors": 0,
        "invalid_execution": 0,
    }
    source = source.resolve()
    outputs = settings.outputs_dir.resolve()
    in_place = source == outputs
    paths = {path.name: path for path in sorted(source.glob("*.png"))}
    tiled_roots = _preflight_passes(source)
    destination = _Destination(settings.db_path, outputs, paths.keys() | tiled_roots)
    existing = destination.existing
    _preflight_retained_passes({name: row for name, row in destination.retained.items() if name in paths}, outputs)
    visited: set[str] = set()
    visiting: set[str] = set()
    identities: dict[str, dict] = {}
    edit_cycles: set[str] = set()
    edit_identities: dict[str, object] = {}
    prepared_proof = cache(_prepared_proof)
    if not dry_run:
        storage.init_db()
    # The `masks` rows a lost database took with it, from the files first;
    # each result's own embedded copy then covers a file that went too.
    masks = _Masks(counts, dry_run, out, destination.mask_path)
    masks.from_directory(source / storage.MASKS_DIR)

    def resolve_edit(stored: StoredInstructionEdit) -> int | None:
        # Already unresolved metadata stays unresolved; explicit repair is a
        # separate operation, never a side effect of repeating recovery.
        if stored.source is None:
            return None
        filename = stored.edit_source.filename
        if filename in visiting:
            edit_cycles.update(visiting)
            return None
        if filename in paths:
            index(paths[filename])
        if filename not in existing or filename in edit_cycles:
            return None
        actual = edit_identities.get(filename)
        if actual is None:
            path = outputs / filename
            if not _proves_edit(path, stored):
                return None
        elif actual != stored.edit_source:
            return None
        return existing[filename]

    def resolve_detailer(stored: StoredDetailer) -> int | None:
        if stored.plan.source_id is None:
            return None
        filename = stored.plan.source.filename
        if filename in visiting:
            edit_cycles.update(visiting)
            return None
        if filename in paths:
            index(paths[filename])
        if filename not in existing or filename in edit_cycles:
            return None
        actual = edit_identities.get(filename)
        if actual is None:
            if not _proves_detailer(outputs / filename, stored):
                return None
        elif actual != stored.plan.source:
            return None
        return existing[filename]

    def resolve_source(provenance) -> int | None:
        if not isinstance(provenance, dict):
            return None
        filename = provenance.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            return None
        # Visit dependencies first: names within one timestamp can sort in
        # either order, and an in-place rebuild assigns entirely new ids.
        if filename in paths and filename not in visiting:
            index(paths[filename])
        if filename not in existing or filename in visiting:
            return None
        actual = identities.get(filename)
        if actual is None and (outputs / filename).is_file():
            # An existing row's hash can be stale after a file was replaced
            # outside the app. Verify the pixels the destination id loads.
            try:
                with Image.open(outputs / filename) as image:
                    actual = {
                        "filename": filename, "hash": storage.content_hash(image),
                        "width": image.width, "height": image.height,
                    }
            except (OSError, ValueError, Image.DecompressionBombError):
                return None
            identities[filename] = actual
        if actual is None or any(
            actual.get(key) != provenance.get(key) for key in ("filename", "hash", "width", "height")
        ):
            return None
        return existing[filename]

    def resolve_tiled(stored: tiled.StoredTiledRefine) -> int | None:
        if stored.source is None:
            return None
        root = stored.root_source
        filename = root.filename
        if filename in visiting:
            edit_cycles.update(visiting)
            return None
        if filename in paths:
            index(paths[filename])
        if filename not in existing or filename in edit_cycles:
            return None
        retained_root = destination.retained.get(filename)
        if retained_root is not None and not _matches_tiled_root_row(retained_root, root):
            return None
        if retained_root is None and filename not in identities:
            return None
        origin = paths.get(filename, outputs / filename)
        target_root = origin if existing[filename] < 0 else outputs / filename
        if (
            prepared_proof(origin, root.hash) == stored.prepared_rgb_sha256
            and prepared_proof(target_root, root.hash) == stored.prepared_rgb_sha256
        ):
            return abs(existing[filename])
        return None

    def resolve_controlnet(stored: ControlNetSpec) -> int | None:
        if stored.source is None:
            return None
        guide = stored.guide
        if guide.filename in visiting:
            edit_cycles.update(visiting)
            return None
        if guide.filename in paths:
            index(paths[guide.filename])
        image_id = existing.get(guide.filename)
        if image_id is None or guide.filename in edit_cycles:
            return None
        origin = paths.get(guide.filename, outputs / guide.filename)
        target_guide = origin if image_id < 0 else outputs / guide.filename
        if guide_matches_path(origin, guide) and guide_matches_path(target_guide, guide):
            return abs(image_id)
        return None

    def index(path: Path) -> None:
        if path.name in visited:
            return
        visited.add(path.name)
        if path.name in existing:
            try:
                known = destination.retained[path.name]
                incoming_meta, _, incoming_image = _read(path, require_metadata=False)
                if incoming_image is None:
                    raise ValueError("the incoming image is not a readable PNG")
                with incoming_image, (outputs / path.name).open("rb") as stream:
                    require_header_metadata(stream)
                    with Image.open(stream) as kept:
                        _verify_retained_pixels(incoming_image, kept, known, in_place=in_place)
                        info = {key: kept.info.get(key) for key in PNG_METADATA_KEYS}
                        retained = edit_from_info(known, info)
                        retained_detailer = detailer_from_info(known, info)
                        retained_tiled = tiled_refine_from_info(known, info)
                        retained_spatial = conditioning_from_info(known, info)
                        validate_import_sources_from_info(known, info)
                incoming_meta = incoming_meta or {}
                validate_import_sources(known, incoming_meta)
                incoming = canonical_edit(incoming_meta.get("edit"))
                incoming_detailer = canonical_detailer(incoming_meta.get("detailer"))
                incoming_tiled = canonical_tiled_refine(incoming_meta.get("tiled_refine"))
                incoming_spatial = canonical_conditioning(incoming_meta.get("spatial_conditioning"))
                if not _same_edit(incoming, retained) or (
                    incoming is not None and known["prompt"] != incoming_meta.get("prompt")
                ):
                    raise ValueError("different instruction edit already uses that filename")
                if not _same_detailer(incoming_detailer, retained_detailer):
                    raise ValueError("different detailer manifest already uses that filename")
                if not _same_tiled(retained_tiled, incoming_tiled, known, incoming_meta):
                    raise ValueError("different Tiled Refine intent already uses that filename")
                if conditioning_key(retained_spatial) != conditioning_key(incoming_spatial):
                    raise ValueError("different spatial conditioning already uses that filename")
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                # A retained row can keep its id and still be unsafe as a
                # source for anything imported in this recovery.
                del existing[path.name]
                counts["conflicts"] += 1
                print(f"  skip  {path.name}: cannot use the retained image: {exc}", file=out)
                return
            counts["present"] += 1
            return
        try:
            meta, (width, height), image = _read(path)
        except ValueError as exc:
            counts["invalid_execution"] += 1
            print(f"  skip  {path.name}: invalid PNG metadata: {exc}", file=out)
            return
        except OSError:  # not a PNG at all, or truncated - UnidentifiedImageError is one
            meta, (width, height), image = None, (0, 0), None
        if meta is None:
            counts["no_metadata"] += 1
            print(f"  skip  {path.name}: no AuraDream metadata", file=out)
            return
        try:
            execution = canonical_execution(meta.get("execution"))
            stored_edit = canonical_edit(meta.get("edit"))
            stored_detailer = canonical_detailer(meta.get("detailer"))
            stored_tiled = canonical_tiled_refine(meta.get("tiled_refine"))
            validate_tiled_refine_record(stored_tiled, {**meta, "width": width, "height": height})
            stored_spatial = canonical_conditioning(meta.get("spatial_conditioning"))
            validate_conditioning_record(stored_spatial, {**meta, "width": width, "height": height})
            validate_import_record(meta)
        except ValueError as exc:
            counts["invalid_execution"] += 1
            print(f"  skip  {path.name}: invalid execution metadata: {exc}", file=out)
            return
        try:
            original_crop_anchor = png_settings.crop_anchor(
                meta.get("crop_anchor"), source=meta.get("refine"),
                mask=meta.get("mask"), outpaint=meta.get("outpaint"),
            )
        except ValueError as exc:
            counts["invalid_crop_anchors"] += 1
            print(f"  skip  {path.name}: {exc}", file=out)
            return
        if meta.get("upscaler") is not None and storage._upscaler(
            meta["upscaler"], source=meta.get("refine"), model_id=meta.get("model"),
            mask=meta.get("mask"), outpaint=meta.get("outpaint"),
        ) is None:
            counts["invalid_upscalers"] += 1
            print(f"  skip  {path.name}: invalid upscaler for its Refine pass", file=out)
            return
        target = outputs / path.name
        if not in_place and target.exists():
            # A different file under a name the gallery already uses on disk
            # but not in the index: not ours to overwrite.
            counts["conflicts"] += 1
            print(f"  skip  {path.name}: a different file of that name is already in outputs", file=out)
            return
        visiting.add(path.name)
        edit_source = resolve_edit(stored_edit) if stored_edit is not None else None
        if path.name in edit_cycles:
            edit_source = None
        # Dry-run ids are negative stand-ins for legacy provenance. Stored edit
        # contracts require positive ids: retain a positive planning placeholder
        # only in memory; no dry-run metadata or file is ever written.
        planned_source = abs(edit_source) if dry_run and edit_source is not None else edit_source
        remapped_edit = _remapped_edit(stored_edit, planned_source, counts)
        detailer_source = resolve_detailer(stored_detailer) if stored_detailer is not None else None
        if path.name in edit_cycles:
            detailer_source = None
        detailer_source = abs(detailer_source) if dry_run and detailer_source is not None else detailer_source
        remapped_detailer = _remapped_detailer(stored_detailer, detailer_source, counts)
        tiled_source = resolve_tiled(stored_tiled) if stored_tiled is not None else None
        if path.name in edit_cycles:
            tiled_source = None
        remapped_tiled = _remapped_tiled(stored_tiled, tiled_source, counts)
        remapped_spatial = stored_spatial
        if isinstance(stored_spatial, ControlNetSpec):
            guide_id = resolve_controlnet(stored_spatial)
            if path.name in edit_cycles:
                guide_id = None
            counts["unresolved_controlnet_guides"] = counts.get("unresolved_controlnet_guides", 0) + (guide_id is None)
            remapped_spatial = stored_spatial.model_copy(update={"source": guide_id})
        masks.detailer(remapped_detailer)
        references = []
        reference_sources = []
        sources = _json_list(meta.get("reference_sources"))
        for i, _reference in enumerate(_json_list(meta.get("reference"))):
            provenance = sources[i] if i < len(sources) else None
            mapped = resolve_source(provenance)
            if mapped is None:
                counts["unresolved_references"] += 1
            else:
                references.append(mapped)
                reference_sources.append(provenance)
        refine = None
        if meta.get("refine") is not None:
            refine = resolve_source(meta.get("refine_source"))
            if refine is None:
                counts["unresolved_references"] += 1
        visiting.remove(path.name)
        # A mask goes where its source went: it covers that image's pixels.
        mask = storage._mask(meta.get("mask")) if refine is not None else None
        # An outpaint goes where its mask went; a malformed one is left off.
        spec = storage.chunk_outpaint(meta) if mask is not None else None
        if mask is not None and storage.malformed_outpaint(meta):
            counts["outpaint_malformed"] += 1
        spec = _outpaint(mask, spec, image, width, height, _provenance_size(meta.get("refine_source")), counts)
        masks.resolve(mask, source / storage.MASKS_DIR, path, image, spec, (width, height))
        # Only the masked area where its mask went, its region proved.
        recorded = storage.chunk_mask_crop(meta) if mask is not None else None
        crop_spec, crop_region, parameters = masks.crop(
            mask, recorded, mask is not None and storage.malformed_mask_crop(meta),
            source / storage.MASKS_DIR, image, spec, (width, height),
        )
        crop_meta = storage.mask_crop_chunk(crop_spec, crop_region) if crop_spec is not None else None
        outpaint_meta = storage.outpaint_chunk(spec)
        outpaint_fill = spec.fill if spec is not None else None
        upscaler = storage._upscaler(
            meta.get("upscaler"), source=refine, model_id=meta.get("model"), mask=mask, outpaint=spec,
        )
        # Original metadata was validated before resolving links. Dry-run
        # source ids are negative stand-ins, so do not validate those as real
        # gallery ids; only the loss of a source changes its crop setting.
        crop_anchor = original_crop_anchor if refine is not None else None
        links_changed = (
            meta.get("reference") != (references or None)
            or meta.get("refine") != refine
            or (refine is None and meta.get("strength") is not None)
            or meta.get("mask") != mask
            or meta.get("outpaint") != outpaint_meta
            or meta.get("outpaint_fill") != outpaint_fill
            or meta.get("upscaler") != upscaler
            or meta.get("crop_anchor") != crop_anchor
            or meta.get("mask_crop") != crop_meta
            or parameters is not None
            or stored_edit != remapped_edit
            or stored_detailer != remapped_detailer
            or stored_tiled != remapped_tiled
            or stored_spatial != remapped_spatial
        )
        meta = {
            **meta,
            "reference": references or None,
            "reference_sources": reference_sources or None,
            "refine": refine,
            "refine_source": meta.get("refine_source") if refine is not None else None,
            "strength": meta.get("strength") if refine is not None else None,
            "mask": mask,
            "mask_source": storage.mask_provenance(mask, width, height),
            "mask_blend": "overlay" if mask is not None else None,
            "outpaint": outpaint_meta,
            "outpaint_fill": outpaint_fill,
        }
        if upscaler is not None:
            meta["upscaler"] = upscaler
        else:
            meta.pop("upscaler", None)
        if crop_anchor is not None:
            meta["crop_anchor"] = crop_anchor
        else:
            meta.pop("crop_anchor", None)
        if crop_meta is not None:
            meta["mask_crop"] = crop_meta
        else:
            meta.pop("mask_crop", None)
        meta.pop("execution", None)
        meta.update(execution_chunk(execution))
        meta.pop("edit", None)
        meta.update(edit_chunk(remapped_edit))
        meta.pop("detailer", None)
        meta.update(detailer_chunk(remapped_detailer))
        meta.pop("tiled_refine", None)
        meta.update(tiled_refine_chunk(remapped_tiled))
        meta.pop("spatial_conditioning", None)
        meta.update(conditioning_chunk(remapped_spatial))
        digest = storage.content_hash(image) if image is not None else None
        identities[path.name] = {
            "filename": path.name, "hash": digest, "width": width, "height": height,
        }
        if image is not None:
            edit_identities[path.name] = descriptor_from_image(path.name, image)
        counts["indexed"] += 1
        if dry_run:
            # Synthetic ids exist only in this plan, so dependent files can
            # resolve their provenance without creating a database or a copy.
            existing[path.name] = -counts["indexed"]
            print(f"  would index  {path.name}", file=out)
            return
        if not in_place or links_changed:
            # Keep --from-image safe after an in-place id remap as well. Only
            # metadata changes; _copy_in preserves the compressed pixels.
            _copy_in(path, target, meta, parameters)
        # The hash of the pixels as decoded here, not the block's copy of it:
        # the file is the truth, and a PNG from before the hash has no copy.
        record = storage.index_file(
            path.name,
            meta,
            width,
            height,
            _created_at(path),
            digest=digest,
        )
        if image is not None:
            storage.thumbnail_for(image, path.name)
        existing[path.name] = record.id

    for path in paths.values():
        index(path)
    return counts


def main(argv: list[str] | None = None) -> int:
    # Redirected or piped, stdout on Windows encodes with the locale's code
    # page, where a CJK file or folder name in --from was a UnicodeEncodeError
    # partway through the report. So does `> NUL`, which answers isatty(); the
    # Windows console is UTF-8 already. Elsewhere a terminal keeps its own
    # encoding. A StringIO has none to change, and under pythonw there is no
    # stdout.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure) and (sys.platform == "win32" or not sys.stdout.isatty()):
        reconfigure(encoding="utf-8", errors="backslashreplace")
    from backend.launch import warn_legacy_env

    warn_legacy_env()  # a pre-rename variable left in a shell: named, not read
    parser = argparse.ArgumentParser(
        prog="python -m backend.reindex",
        description="Rebuild gallery rows from the PNGs' own metadata; idempotent.",
    )
    parser.add_argument(
        "--from",
        dest="source",
        metavar="DIR",
        help="a directory of this app's PNGs to copy into the gallery and index "
        "(default: the gallery's own outputs directory, in place)",
    )
    parser.add_argument(
        "--database",
        "--db",
        metavar="FILE",
        help="the AuraDream database belonging to --from; preserves favorites, "
        "durations and remaps references",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="report what would be indexed, change nothing"
    )
    args = parser.parse_args(argv)
    source = Path(args.source).expanduser() if args.source else settings.outputs_dir
    if not source.is_dir():
        print(f"{source} is not a directory", file=sys.stderr)
        return 2
    database = Path(args.database).expanduser() if args.database else None
    if database is not None and not args.source:
        print("--database requires --from so its matching outputs folder is explicit", file=sys.stderr)
        return 2
    if database is not None and not database.is_file():
        print(f"{database} is not a file", file=sys.stderr)
        return 2

    print(f"database : {settings.db_path}")
    print(f"images   : {settings.outputs_dir}")
    print(f"source   : {source}{' (in place)' if source.resolve() == settings.outputs_dir.resolve() else ''}")
    if database is not None:
        print(f"source db: {database}")
    print()
    started = time.perf_counter()
    try:
        counts = reindex(source, dry_run=args.dry_run, database=database)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if database:
        verb = "Would import" if args.dry_run else "Imported"
    else:
        verb = "Would index" if args.dry_run else "Indexed"
    summary = f"\n{verb} {counts['indexed']} file(s); {counts['present']} already in the index"
    if database:
        summary += (
            (f", {counts['missing']} missing image(s)" if counts["missing"] else "")
            + (f", {counts['invalid']} invalid row/file(s)" if counts["invalid"] else "")
        )
    else:
        summary += f", {counts['no_metadata']} without metadata" if counts["no_metadata"] else ""
        if counts["invalid_upscalers"]:
            summary += f", {counts['invalid_upscalers']} with an invalid upscaler"
        if counts["invalid_crop_anchors"]:
            summary += f", {counts['invalid_crop_anchors']} with an invalid crop anchor"
        if counts["invalid_execution"]:
            summary += f", {counts['invalid_execution']} with invalid execution metadata"
    if counts["unresolved_references"]:
        summary += f", {counts['unresolved_references']} unresolved reference/refine source(s)"
    if counts["unresolved_edits"]:
        summary += f", {counts['unresolved_edits']} unresolved instruction-edit source(s)"
    if counts["unresolved_detailers"]:
        summary += f", {counts['unresolved_detailers']} unresolved detailer original source(s)"
    if counts["unresolved_tiled_refines"]:
        summary += f", {counts['unresolved_tiled_refines']} unresolved Tiled Refine original source(s)"
    unresolved_guides = counts.get("unresolved_controlnet_guides", 0)
    if unresolved_guides:
        summary += f", {unresolved_guides} unresolved ControlNet guide(s)"
    if counts["detailer_masks_rebuilt"]:
        summary += f", {counts['detailer_masks_rebuilt']} detailer mask(s) built again from their frozen regions"
    if counts["masks"]:
        summary += f", {counts['masks']} inpaint mask(s) {'to restore' if args.dry_run else 'restored'}"
    if counts["bad_masks"]:
        summary += f", {counts['bad_masks']} mask file(s) not named by their pixels"
    if counts["unresolved_masks"]:
        summary += f", {counts['unresolved_masks']} inpaint mask(s) found nowhere"
    if counts["outpaint_masks_rebuilt"]:
        summary += f", {counts['outpaint_masks_rebuilt']} outpaint mask(s) built again from their spec"
    if counts["outpaints_recovered"]:
        summary += f", {counts['outpaints_recovered']} outpaint(s) recovered from the A1111 line"
    if counts["outpaint_malformed"]:
        summary += f", {counts['outpaint_malformed']} malformed outpaint(s) left off"
    if counts["mask_crops_recovered"]:
        summary += f", {counts['mask_crops_recovered']} masked-area crop(s) recovered from the A1111 line"
    if counts["mask_crop_malformed"]:
        summary += (
            f", {counts['mask_crop_malformed']} masked-area crop(s) left off: malformed, "
            "or not their mask's region"
        )
    summary += f", {counts['conflicts']} name conflict(s)" if counts["conflicts"] else ""
    print(summary + f". {time.perf_counter() - started:.1f}s.")
    if args.dry_run:
        print("Dry run - nothing changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
