"""Rebuild gallery rows from the PNGs' own metadata.

    uv run python -m backend.reindex                  # outputs/ in place
    uv run python -m backend.reindex --from /mnt/d/wsl-outputs   # copy in and index
    uv run python -m backend.reindex --dry-run

Every image carries its whole reproduction set in a `krea` text chunk, and
nothing could read it back: krea.db was a single point of failure for an
otherwise self-describing gallery, the WSL and Windows galleries could never be
merged, and a row whose database went missing could never be repaired. This
walks a directory of PNGs, skips the names the index already has (filename is
UNIQUE, which makes the whole thing idempotent), copies files from elsewhere
in with the same `.part`-then-rename discipline `save_image` uses, and inserts
through `storage` so the search index is built with the rows.

What cannot come back: the favourite flag, which is written to SQLite alone,
and the duration, recorded as 0 so the time estimate ignores these rows. The
timestamp is parsed from the filename prefix as UTC, which is how it was
written; failing that, the file's own mtime.

Not `import.py`: `from backend import import` is a SyntaxError, so no test
could follow the house style. Running the file directly works too - see the
sys.path bootstrap below.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image

    from backend import storage
    from backend.config import settings
except ModuleNotFoundError as exc:  # pragma: no cover - startup guard
    sys.exit(
        f"Missing dependency: {exc.name}\n\n"
        f"Running under: {sys.executable}\n"
        "That interpreter is not the project environment. Use:\n\n"
        "    uv run python -m backend.reindex\n"
    )

# `20260907-235118-42-6a1b2c.png`: the UTC timestamp save_image writes first.
_NAME = re.compile(r"^(\d{8})-(\d{6})-")

# The keys a `krea` block must carry to make a row at all.
_REQUIRED = ("prompt", "steps", "guidance_scale", "seed", "model")


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


def _read(path: Path) -> tuple[dict | None, tuple[int, int], Image.Image | None]:
    """The krea block, the size and a copy of the image for its thumbnail.

    `.info`, not `.text`: the latter is a property that decodes every pixel to
    answer a question the header settles.
    """
    with Image.open(path) as image:
        raw = image.info.get("krea")
        size = image.size
        if not raw:
            return None, size, None
        try:
            meta = json.loads(raw)
        except ValueError:
            return None, size, None
        if not isinstance(meta, dict) or any(meta.get(key) is None for key in _REQUIRED):
            return None, size, None
        return meta, size, image.copy()


def _copy_in(source: Path, target: Path) -> None:
    """Copy a file into outputs_dir under a temporary name, then rename."""
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(source.read_bytes())
    partial.replace(target)


def reindex(source: Path, dry_run: bool = False, out=sys.stdout) -> dict[str, int]:
    counts = {"indexed": 0, "present": 0, "no_metadata": 0, "conflicts": 0}
    source = source.resolve()
    outputs = settings.outputs_dir.resolve()
    in_place = source == outputs
    present = storage.existing_filenames() if settings.db_path.exists() else set()
    if not dry_run:
        storage.init_db()
    for path in sorted(source.glob("*.png")):
        if path.name in present:
            counts["present"] += 1
            continue
        try:
            meta, (width, height), image = _read(path)
        except OSError:  # not a PNG at all, or truncated - UnidentifiedImageError is one
            meta = None
        if meta is None:
            counts["no_metadata"] += 1
            print(f"  skip  {path.name}: no krea metadata", file=out)
            continue
        target = outputs / path.name
        if not in_place and target.exists():
            # A different file under a name the gallery already uses on disk
            # but not in the index: not ours to overwrite.
            counts["conflicts"] += 1
            print(f"  skip  {path.name}: a different file of that name is already in outputs", file=out)
            continue
        counts["indexed"] += 1
        if dry_run:
            print(f"  would index  {path.name}", file=out)
            continue
        if not in_place:
            _copy_in(path, target)
        # The hash of the pixels as decoded here, not the block's copy of it:
        # the file is the truth, and a PNG from before the hash has no copy.
        storage.index_file(
            path.name,
            meta,
            width,
            height,
            _created_at(path),
            digest=storage.content_hash(image) if image is not None else None,
        )
        if image is not None:
            storage.thumbnail_for(image, path.name)
        present.add(path.name)
    return counts


def main(argv: list[str] | None = None) -> int:
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
        "-n", "--dry-run", action="store_true", help="report what would be indexed, change nothing"
    )
    args = parser.parse_args(argv)
    source = Path(args.source).expanduser() if args.source else settings.outputs_dir
    if not source.is_dir():
        print(f"{source} is not a directory", file=sys.stderr)
        return 2

    print(f"database : {settings.db_path}")
    print(f"images   : {settings.outputs_dir}")
    print(f"source   : {source}{' (in place)' if source.resolve() == settings.outputs_dir.resolve() else ''}")
    print()
    started = time.perf_counter()
    counts = reindex(source, dry_run=args.dry_run)
    verb = "Would index" if args.dry_run else "Indexed"
    print(
        f"\n{verb} {counts['indexed']} file(s); {counts['present']} already in the index"
        + (f", {counts['no_metadata']} without metadata" if counts["no_metadata"] else "")
        + (f", {counts['conflicts']} name conflict(s)" if counts["conflicts"] else "")
        + f". {time.perf_counter() - started:.1f}s."
    )
    if args.dry_run:
        print("Dry run - nothing changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
