"""The stable codes of the refusals the page acts on, and the errors that
carry one out of the modules below the HTTP layer. Stdlib only: `storage`
and `loras` raise these, and the CLI imports both.

A refusal's `detail` is a sentence for a person and may be reworded; its
`code` is what the page reads to decide what to do (forget an image, mark it
binned, drop an adapter, hold a count), and never changes once a page reads
it. `refusals.py` puts a code beside `detail` in the response body; a
refusal the page only shows carries none.
"""
from __future__ import annotations

from collections.abc import Iterable

# No live gallery image has this id: deleted, never there, or - where the
# route cannot or does not tell them apart - in the recycle bin.
IMAGE_UNKNOWN = "image_unknown"
# The image is in the recycle bin: gone from the gallery until a Restore.
IMAGE_BINNED = "image_binned"
# More seeds' batches x count is past what one press may queue.
MORE_SEEDS_CAP = "more_seeds_cap"
# More seeds' receipt expired, is another server's or names another request.
MORE_SEEDS_RECEIPT = "more_seeds_receipt"
# The request's adapter has no file in the LoRA directory.
LORA_MISSING = "lora_missing"
# The request's adapter is there but does not fit the running checkpoint.
LORA_MISMATCH = "lora_mismatch"
# A confirmed count no longer matches what the server counts.
SELECTION_CHANGED = "selection_changed"
# The request's stored mask has gone: swept, or its file lost.
MASK_MISSING = "mask_missing"
# A saved bundle's delete waits: the file is being downloaded or imported,
# or a bundle operation holds the gallery. The row stays; try again later.
BUNDLE_IN_USE = "bundle_in_use"
# The saved bundle a delete named is no longer in the bundles directory.
BUNDLE_GONE = "bundle_gone"
# A gallery bundle import, export, clear or backup holds the gallery: every
# counted route (`bundle_manager.GalleryAdmission`) waits until it ends.
GALLERY_MAINTENANCE = "gallery_maintenance"
# A checkpoint delete (Models > On disk, `model_cache_delete`) waits: a job,
# the card, a load, switch, unload or drain, a counted model request, Check's
# tokenizer fetch from a repository it touches, a cache scan or another
# delete. Held until the status poll clears it.
MODEL_DELETE_BUSY = "model_delete_busy"
# The checkpoint is the selected (and so any loaded) one, or the one saved to
# load automatically on the next start: choose another first.
MODEL_SELECTED = "model_selected"
# A partial download (`*.incomplete`) sits in a repository the delete touches.
MODEL_DOWNLOADING = "model_downloading"
# The cache changed between the plan the person confirmed and the delete:
# its pin differs. Read the plan again and ask again.
MODEL_CACHE_CHANGED = "model_cache_changed"
# Windows refused the first rename: a file is open; nothing changed.
MODEL_FILES_IN_USE = "model_files_in_use"
# No catalogue entry by that id, or nothing of it is cached: one answer, 404.
MODEL_NOT_ON_DISK = "model_not_on_disk"
# A part of the delete the server cannot prove safe (a shared Xet store, an
# ambiguous revision, a shared repository before it is built): the plan's
# `held`, and the row's Delete held with its detail.
MODEL_DELETE_HELD = "model_delete_held"
# The codes the page reads only once Models > On disk has its Delete control
# (`GEN#72` stage 3); the server gives them from stage 1.
MODEL_DELETE_CODES = frozenset({
    MODEL_DELETE_BUSY, MODEL_SELECTED, MODEL_DOWNLOADING, MODEL_CACHE_CHANGED,
    MODEL_FILES_IN_USE, MODEL_NOT_ON_DISK, MODEL_DELETE_HELD,
})

CODES = frozenset({
    IMAGE_UNKNOWN, IMAGE_BINNED, MORE_SEEDS_CAP, MORE_SEEDS_RECEIPT,
    LORA_MISSING, LORA_MISMATCH, SELECTION_CHANGED, MASK_MISSING,
    BUNDLE_IN_USE, BUNDLE_GONE, GALLERY_MAINTENANCE,
    *MODEL_DELETE_CODES,
})


class CodedError(ValueError):
    """A refusal in words that also carries a code, and the gallery ids it
    is about where it names images. A `ValueError`, so every caller that
    reads the sentence alone is unchanged."""

    def __init__(self, message: str, code: str, ids: Iterable[int] = ()) -> None:
        super().__init__(message)
        assert code in CODES, code
        self.code = code
        self.ids = tuple(ids)


class CodedLookupError(LookupError):
    """The same for an unknown id a caller answers as a 404."""

    def __init__(self, message: str, code: str, ids: Iterable[int] = ()) -> None:
        super().__init__(message)
        assert code in CODES, code
        self.code = code
        self.ids = tuple(ids)
