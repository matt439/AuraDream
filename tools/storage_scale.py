"""CPU-only bundle/bin scale check, isolated to an explicitly empty scratch directory.

Run through tests/test_storage_scale.py, or with an unused --data directory.
The JSON answer records fixture sizes and measured durations, not speed limits.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile


ROWS = 208
BINNED = 200
SIDE = 256
MIN_FREE = 256 * 1024**2
SCRATCH_LIMIT = 100 * 1024**2


def _files(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


def _size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _integrity(database: Path) -> None:
    with closing(sqlite3.connect(database)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def _run(root: Path, encrypted: bool) -> dict:
    # Set paths before importing the application. A mistyped existing directory
    # refuses in main(), before any storage method can touch its contents.
    source = root / "source"
    source.mkdir()
    os.environ["LOCAL_DIFFUSION_DB_PATH"] = str(source / "local-diffusion.db")
    os.environ["LOCAL_DIFFUSION_OUTPUTS_DIR"] = str(source / "outputs")
    from PIL import Image

    from backend import bundles, storage
    from backend.metadata import PNG_MASK_KEY, PNG_METADATA_KEY
    from backend.schemas import GenerateRequest

    storage.init_db()
    pixel_hashes: dict[str, str] = {}

    def save(seed: int, **changes):
        pixels = random.Random(seed).randbytes(SIDE * SIDE * 3)
        with Image.frombytes("RGB", (SIDE, SIDE), pixels) as image:
            record = storage.save_image(
                image,
                GenerateRequest(prompt=f"scale image {seed}", width=SIDE, height=SIDE,
                                steps=28, guidance_scale=3.5, mu=None, **changes).applied(
                                    negative_prompt_supported=True, cfg_threshold=1.0,
                                    reference_supported=True, refine_supported=True, inpaint_supported=True),
                seed=seed, duration_ms=1000 + seed, model_id="test/scale",
                quantization="fp8", job_id="scale-job", run_id="scale-run", batch=4,
                variables={"fixture": str(seed % 3)},
            )
        pixel_hashes[record.filename] = hashlib.sha256(pixels).hexdigest()
        return record

    def mask(box):
        with Image.new("L", (SIDE, SIDE), 0) as image:
            image.paste(255, box)
            return storage.store_mask(image)[0]

    def rows():
        live, live_count, _ = storage.list_images(limit=ROWS + 10)
        binned, bin_count, _ = storage.list_images(limit=ROWS + 10, binned=True)
        assert len(live) == live_count and len(binned) == bin_count
        return {record.filename: record for record in live + binned}

    shared = mask((8, 8, 160, 160))
    live_only = mask((16, 16, 180, 180))
    bin_only = mask((24, 24, 200, 200))
    unused = mask((32, 32, 220, 220))
    mask_hashes = {digest: _files(storage.masks_dir())[f"{digest}.png"]
                   for digest in (shared, live_only, bin_only, unused)}

    first, second = save(1), save(2)
    doomed = [save(3)]
    live = [
        first, second,
        save(4, reference=[second.id, doomed[0].id, first.id]),
        save(5, refine=doomed[0].id, strength=0.35),
        save(6, refine=first.id, strength=0.6, mask=shared),
        save(7, refine=second.id, strength=0.7, mask=live_only),
        save(8), save(9),
    ]
    for seed in range(10, ROWS + 1):
        changes = ({"reference": [second.id, first.id]} if seed % 4 == 0 else
                   {"refine": first.id, "strength": 0.45} if seed % 4 == 1 else
                   {"refine": second.id, "strength": 0.6,
                    "mask": shared if seed == 10 else bin_only} if seed % 4 == 2 else {})
        doomed.append(save(seed, **changes))
    assert (len(live), len(doomed)) == (ROWS - BINNED, BINNED)
    for record in live[::2] + doomed[::17]:
        storage.set_favorite(record.id, True)
    assert storage.bin_images([record.id for record in doomed])["binned"] == sorted(r.id for r in doomed)
    original = rows()
    original_ids = {record.id: name for name, record in original.items()}
    original_files = _files(storage.settings.outputs_dir)
    source_bytes = _size(storage.settings.outputs_dir)
    png_bytes = sum((storage.settings.outputs_dir / name).stat().st_size for name in original)
    # Incompressible RGB data rather than hundreds of tiny solid-colour PNGs.
    assert png_bytes > ROWS * SIDE * SIDE * 3 * 0.95
    _integrity(storage.settings.db_path)
    password = "synthetic scale fixture" if encrypted else None
    started = time.perf_counter()
    archive = bundles.export_bundle(expected=ROWS, password=password)
    export_seconds = time.perf_counter() - started
    assert (archive["records"], archive["binned"], archive["masks"]) == (ROWS, BINNED, 4)
    assert archive["cleared"] and archive["retained"] == []
    assert set(archive["deleted_ids"]) == set(original_ids)
    assert rows() == {} and _files(storage.settings.outputs_dir) == {}
    assert archive["bytes"] > 32 * 1024**2
    _integrity(storage.settings.db_path)
    # Plain archives get an independent, streaming manifest/hash read as well
    # as the application's verified read-back. Encrypted archives are checked
    # by its authenticated read-back and the two imports below.
    if not encrypted:
        with ZipFile(archive["path"]) as handle:
            manifest = json.loads(handle.read("manifest.json"))
            for name, expected in manifest["files"].items():
                with handle.open(name) as member:
                    assert hashlib.file_digest(member, "sha256").hexdigest() == expected["sha256"]
                assert handle.getinfo(name).file_size == expected["size"]

    destination = root / "destination"
    storage.settings = dataclasses.replace(
        storage.settings, db_path=destination / "local-diffusion.db", outputs_dir=destination / "outputs",
    )
    storage.init_db()
    existing = [save(seed) for seed in (1001, 1002, 1003)]
    existing_files = _files(storage.settings.outputs_dir)
    started = time.perf_counter()
    imported = bundles.import_bundle(Path(archive["path"]), password)
    import_seconds = time.perf_counter() - started
    assert (imported["imported"], imported["present"], imported["binned"], imported["masks"]) == (ROWS, 0, BINNED, 4)
    assert (imported["unresolved_references"], imported["masks_missing"]) == (0, 0)
    actual = rows()
    assert len(actual) == ROWS + len(existing)
    actual_files = _files(storage.settings.outputs_dir)
    mapped = {old_id: actual[name].id for old_id, name in original_ids.items()}
    assert all(old_id != new_id for old_id, new_id in mapped.items())

    def provenance(old_id):
        filename = original_ids[old_id]
        return {"filename": filename, "hash": pixel_hashes[filename][:16], "width": SIDE, "height": SIDE}

    for name, before in original.items():
        after = actual[name]
        ignored = {"id", "url", "thumb_url", "reference", "refine"}
        assert before.model_dump(exclude=ignored) == after.model_dump(exclude=ignored)
        assert after.reference == ([mapped[i] for i in before.reference] if before.reference else None)
        assert after.refine == (mapped[before.refine] if before.refine is not None else None)
        with Image.open(storage.settings.outputs_dir / name) as image:
            assert image.mode == "RGB" and image.size == (SIDE, SIDE)
            assert hashlib.sha256(image.tobytes()).hexdigest() == pixel_hashes[name]
            assert after.content_hash == pixel_hashes[name][:16]
            metadata = json.loads(image.info[PNG_METADATA_KEY])
            assert metadata["reference"] == after.reference and metadata["refine"] == after.refine
            assert metadata["hash"] == after.content_hash and metadata["mask"] == after.mask
            if after.reference:
                assert metadata["reference_sources"] == [provenance(i) for i in before.reference]
            if after.refine is not None:
                assert metadata["refine_source"] == provenance(before.refine)
            if after.mask:
                assert metadata["mask_blend"] == "overlay" and PNG_MASK_KEY in image.info
                assert metadata["mask_source"] == {"filename": f"{after.mask}.png", "hash": after.mask,
                                                   "width": SIDE, "height": SIDE}
        thumb = "thumbnails/" + Path(name).with_suffix(".webp").name
        assert actual_files[thumb] == original_files[thumb]
    for name, digest in existing_files.items():
        assert actual_files[name] == digest
    for digest, file_hash in mask_hashes.items():
        with Image.open(storage.mask_path(digest)) as image:
            assert image.mode == "L" and storage.mask_hash(image) == digest
        assert hashlib.sha256(storage.mask_path(digest).read_bytes()).hexdigest() == file_hash
    _integrity(storage.settings.db_path)

    # A local star and a same-count restore/bin swap must win over the archive.
    restored = actual[doomed[-1].filename]
    newly_binned = actual[live[-1].filename]
    storage.restore_images([restored.id])
    storage.bin_images([newly_binned.id])
    changed = actual[live[0].filename]
    storage.set_favorite(changed.id, not changed.favorite)
    before_repeat = {name: record.model_dump() for name, record in rows().items()}
    repeat_files = _files(storage.settings.outputs_dir)
    started = time.perf_counter()
    again = bundles.import_bundle(Path(archive["path"]), password)
    repeat_seconds = time.perf_counter() - started
    assert (again["imported"], again["present"], again["binned"], again["masks"], again["masks_missing"]) == (0, ROWS, 0, 0, 0)
    assert {name: record.model_dump() for name, record in rows().items()} == before_repeat
    assert _files(storage.settings.outputs_dir) == repeat_files
    _integrity(storage.settings.db_path)
    scratch_bytes = _size(root)
    assert scratch_bytes < SCRATCH_LIMIT

    # Age the scratch masks so retention is proved by references, not by the
    # recent-upload settle window. No wall-clock sleeps or shortened cutoff.
    with storage._session() as conn:
        conn.execute("UPDATE masks SET created_at = '2000-01-01T00:00:00+00:00'")
    before_empty = rows()
    binned_ids = sorted(record.id for record in before_empty.values() if record.binned_at)
    kept = {name: record for name, record in before_empty.items() if not record.binned_at}
    summary = storage.bin_summary()
    assert summary["records"] == BINNED
    assert summary["sources_in_use"] == [mapped[doomed[0].id]]
    started = time.perf_counter()
    emptied = storage.empty_bin(expected=BINNED, pin=summary["pin"])
    empty_seconds = time.perf_counter() - started
    assert emptied == {"records": BINNED, "files": BINNED, "ids": binned_ids, "masks": 2}
    computed = {"refine_missing", "refine_binned"}
    assert {name: record.model_dump(exclude=computed) for name, record in rows().items()} == {
        name: record.model_dump(exclude=computed) for name, record in kept.items()}
    kept_thumbnails = {"thumbnails/" + Path(filename).with_suffix(".webp").name for filename in kept}
    expected_files = {name: digest for name, digest in repeat_files.items()
                      if name in kept or name in kept_thumbnails
                      or name in {f"masks/{shared}.png", f"masks/{live_only}.png"}}
    assert _files(storage.settings.outputs_dir) == expected_files
    assert storage._tracked_masks() == {shared, live_only}
    assert storage.bin_summary()["records"] == 0
    assert storage.get_image(mapped[doomed[0].id], include_binned=True) is None
    assert storage.get_image(mapped[live[3].id]).refine_missing is True
    _integrity(storage.settings.db_path)
    return {"encrypted": encrypted, "rows": ROWS, "binned": BINNED, "size": f"{SIDE}x{SIDE}",
            "source_bytes": source_bytes, "png_bytes": png_bytes, "archive_bytes": archive["bytes"],
            "retained_scratch_bytes_before_empty": scratch_bytes,
            "live_after_empty": len(kept), "masks_after_empty": 2,
            "seconds": {"export": round(export_seconds, 3), "import": round(import_seconds, 3),
                        "repeat_import": round(repeat_seconds, 3), "empty_bin": round(empty_seconds, 3)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="empty disposable scratch directory")
    parser.add_argument("--encrypted", action="store_true", help="exercise a password bundle")
    args = parser.parse_args()
    root = args.data.resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        parser.error("--data must be empty; this check never reuses an existing gallery")
    parent = root if root.exists() else next(p for p in root.parents if p.exists())
    if shutil.disk_usage(parent).free < MIN_FREE:
        parser.error("the scratch volume needs at least 256 MiB free")
    root.mkdir(parents=True, exist_ok=True)
    # Executing tools/foo.py puts tools, rather than the repository, on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    with tempfile.TemporaryDirectory(prefix="storage-scale-", dir=root) as scratch:
        work = Path(scratch).resolve()
        assert work.is_relative_to(root)
        report = _run(work, args.encrypted)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
