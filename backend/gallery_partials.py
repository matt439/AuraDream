"""The gallery's work areas on disk and what killed writes leave there: the
bundles directory, the staging directory, and settling stale `.part` files and
bundle leftovers.

Like `gallery_delete`, it reads storage's `settings` and helpers at call time,
so a test's replacement of `storage.settings` reaches it. Callers reach these
names as `storage.<name>` (storage's `_MOVED`), and a patch of
`storage.<name>` intercepts those callers. To intercept a call made inside
this module (for example `_bundle_leftovers` -> `bundle_directory`), patch
`gallery_partials.<name>`.
"""
from __future__ import annotations

import logging
import re
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from . import gallery_lock, storage
from .bundle_common import BundleError
from .record_fields import mask_hash

log = logging.getLogger(__name__)


# Bundle imports unpack, and uploads land, in this directory on the images'
# volume, so installing a staged image is a rename rather than a second copy.
# Its name matches none of the sweeps' globs or `_BUNDLE_WORK`.
BUNDLE_STAGING = ".auradream-staging"
# The same directory as Local Diffusion named it. Nothing stages there any
# more; only what a killed operation of that build left is swept from it.
LEGACY_BUNDLE_STAGING = (".local-diffusion-staging",)
# What bundles.py and api_bundles.py create beside the database: export's
# snapshot directory and import's merge directory (tempfile's eight random
# characters), and uploads from before they moved to BUNDLE_STAGING.
_BUNDLE_WORK = re.compile(r"\.bundle-(?:import-|merge-|upload-)?[a-z0-9_]{8}(?:\.part)?")


def bundle_directory() -> Path:
    """The saved archives' directory, shared by reads, writes and cleanup."""
    directory = storage.settings.bundles_dir
    if directory is None:
        directory = storage.settings.db_path.parent.resolve() / "bundles"
    # Check the configured name before resolve follows a direct junction or
    # symlink and hides it. Parent links still resolve for the containment test.
    if directory.is_symlink() or directory.is_junction():
        raise BundleError("the bundles directory must not be a symbolic link")
    directory = directory.resolve()
    outputs = storage.settings.outputs_dir.resolve()
    if directory == outputs or directory.is_relative_to(outputs):
        raise BundleError("the bundles directory must be outside the images directory")
    return directory


def bundle_staging_dir() -> Path:
    """The staging directory on the images' volume, created on demand.

    Beside the images directory, out of what `/outputs` serves, unless that
    is another volume (the images directory is a mount point); then inside
    it, where the inventory skips directories.
    """
    storage.ensure_outputs_dir()
    outputs = storage.settings.outputs_dir.resolve()
    root = outputs.parent
    try:
        if root.stat().st_dev != outputs.stat().st_dev:
            root = outputs
    except OSError:
        root = outputs
    staging = root / BUNDLE_STAGING
    staging.mkdir(exist_ok=True)
    return staging


def _bundle_leftovers() -> list:
    """Everything a bundle operation writes outside the gallery proper.

    A finished or failed operation removes its own; a killed one (power cut,
    a second Ctrl-C) leaves a gallery-sized copy that nothing else reaches.
    """
    beside = storage.settings.db_path.parent
    outputs = storage.settings.outputs_dir
    return [
        *bundle_directory().glob("*.part"),
        *(p for p in beside.glob(".bundle-*") if _BUNDLE_WORK.fullmatch(p.name)),
        *(
            path
            for name in (BUNDLE_STAGING, *LEGACY_BUNDLE_STAGING)
            for root in (outputs.parent, outputs)
            for path in (root / name).glob("*")
        ),
    ]


def _stale_partials() -> list:
    """Partials nothing can still be writing - see PARTIAL_SETTLE_SECONDS.

    Bundle leftovers too: an hour is as safe for them. Every bundle operation
    holds the gallery reservation throughout, as every sweep does, and an
    upload still arriving keeps writing its file.
    """
    cutoff = time.time() - storage.PARTIAL_SETTLE_SECONDS
    stale = []
    paths = [
        *storage.settings.outputs_dir.glob("*.part"),
        *(storage.settings.outputs_dir / "thumbnails").glob("*.part"),
        *storage.masks_dir().glob("*.part"),
        *_bundle_leftovers(),
    ]
    for path in paths:
        try:
            if path.stat().st_mtime < cutoff:
                stale.append(path)
        except OSError:
            continue
    return stale


def _discard(path) -> bool:
    """Remove a stale partial: a file, or a bundle's work directory."""
    if path.is_dir() and not path.is_symlink() and not path.is_junction():
        shutil.rmtree(path, ignore_errors=True)
        return not path.exists()
    return storage._unlink(path)


def _settle_plan(stale: Sequence, tracked: set[str] | None = None) -> list:
    """Each stale partial with the name it should be renamed to, or None.

    `save_image` commits the row and then renames, so a `<name>.png.part`
    whose row exists and whose `<name>.png` does not is the only copy of a
    committed image - the rename failed, or the process died between the two.
    That one is finished, never discarded; every other stale partial is
    debris. A thumbnail's `.webp.part` is debris too, even beside a committed
    row: `thumbnail_for` encodes for rows already tracked, so one may be cut
    short, and a placed thumbnail is never encoded again. The card falls back
    to the PNG until a reindex writes a whole one. A mask's `.png.part` whose
    `masks` row exists is finished likewise: `store_mask` commits the row
    before the rename too - but only once it decodes to the mask its name
    says (`_whole_mask`). A row can already be there while a lost file is
    encoded again, so a partial cut short by a crash is debris, never
    installed as the mask.
    """
    if tracked is None:
        tracked = {row["filename"] for row in storage._tracked_files()}
    outputs = storage.settings.outputs_dir
    masks = storage.masks_dir()
    held = storage._tracked_masks() if any(path.parent == masks for path in stale) else set()
    plan = []
    for path in stale:
        name = path.name[: -len(".part")] if path.name.endswith(".part") else None
        target = None
        if name and path.parent == outputs and name.endswith(".png") and name in tracked:
            target = path.with_name(name)
        elif (
            name and path.parent == masks and name.endswith(".png") and name[: -len(".png")] in held
            and _whole_mask(path, name[: -len(".png")])
        ):
            target = path.with_name(name)
        plan.append((path, None if target is None or target.exists() else target))
    return plan


def _whole_mask(path: Path, digest: str) -> bool:
    """Whether the file at `path` is a whole PNG of the mask `digest` names."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.format == "PNG" and image.mode == "L" and mask_hash(image) == digest
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return False


def _settle(stale: Sequence) -> tuple[int, int]:
    """Finish or discard stale partials, under the reservation the caller holds.

    Returns (recovered, discarded). A recovery that fails leaves the partial
    where it is: it is the image, and the next sweep tries again.
    """
    recovered = discarded = 0
    for path, target in _settle_plan(stale):
        if target is None:
            discarded += _discard(path)
            continue
        try:
            storage._rename_with_retry(path, target)
            recovered += 1
            log.info("recovered %s from an interrupted save", target.name)
        except OSError:
            log.warning("could not finish the interrupted save of %s", target.name, exc_info=True)
    return recovered, discarded


@gallery_lock.protected(lambda: storage.settings.db_path)
def settle_partials() -> dict[str, int]:
    """Finish or sweep what killed saves and bundle operations left.

    At startup, at every bundle preview and before each bundle operation. Only
    partials older than PARTIAL_SETTLE_SECONDS: a fresh one may be an encode
    in flight, which a bundle inventory refuses instead.
    """
    recovered, discarded = _settle(_stale_partials())
    return {"recovered": recovered, "discarded": discarded}
