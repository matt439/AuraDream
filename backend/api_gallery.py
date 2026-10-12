"""Gallery browsing, favourites, the recycle bin and scoped deletion routes."""
from __future__ import annotations

import asyncio
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from . import gallery_delete, gallery_reports, storage
from .jobs import JobManager
from .schemas import (
    BinResult,
    BinSummary,
    GalleryPage,
    IdsBody,
    ImageRecord,
    RestoreResult,
    RunSummary,
    UnstarredPreview,
)
from .search import strict_scope


def _unstarred_scope(search: str | None) -> str | None:
    """The search box as a strict scope, or a 422 that says why it is not one.

    `strict_scope` refuses free text with the grammar's own words; here the
    reader has just watched the gallery match that text, so the reply says
    what to do instead.
    """
    try:
        strict_scope(search)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Delete unstarred takes key:value terms or an empty search - {exc}",
        ) from exc
    return search or None


async def _record_or_404(image_id: int, binned: bool) -> ImageRecord:
    """One image's record, the recycle bin's only when asked for.

    A binned image answers 404 like a deleted one - the page's many readers
    of "unknown image" mean "gone from the gallery", and are right - with
    the reason after it, for the reader that wants to offer a Restore.
    """
    record = await asyncio.to_thread(storage.get_image, image_id, True)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown image")
    if record.binned_at is not None and not binned:
        raise HTTPException(
            status_code=404, detail=f"unknown image (#{image_id} is in the recycle bin)"
        )
    return record


def create_router(
    get_jobs: Callable[[], JobManager],
    model_scope: Callable[[bool], str | None],
    *,
    max_grid_images: int,
) -> APIRouter:
    """Bind app dependencies without copying its mutable runtime state.

    The scope callback is shared with Re-render, and the manager is read at
    request time so a replacement worker is also used by these routes.
    """
    router = APIRouter()

    @router.get("/api/runs", response_model=list[RunSummary])
    async def recent_runs(
        limit: int = Query(20, ge=1, le=200), all_models: bool = False
    ) -> list[RunSummary]:
        """The gallery grouped by run, newest first - one line per night's grid.

        Scoped like the gallery it lists: this checkpoint's runs unless asked for
        every one, so picking a run never opens an empty sheet.
        """
        return await asyncio.to_thread(gallery_reports.runs, limit, model_scope(all_models))


    @router.get("/api/gallery", response_model=GalleryPage)
    async def gallery(
        # A sheet asks for a whole run at once - up to a full grid - where the grid
        # view pages. The rows are small; the thumbnails load lazily either way.
        # The ceiling follows LOCAL_DIFFUSION_MAX_GRID_IMAGES, which /api/status publishes, so
        # the sheet's one-page request cannot be refused by the server that sized it.
        limit: int = Query(48, ge=1, le=max(1024, max_grid_images)),
        offset: int = Query(0, ge=0),
        # Keyset paging: rows older than this id. What the browser sends, because
        # an offset over a gallery a sweep is writing into repeats the tail of one
        # page at the head of the next and skips as many older rows for good.
        before: int | None = Query(None, ge=1),
        search: str | None = None,
        favorites: bool = False,
        # The running checkpoint's images unless asked for every checkpoint's:
        # steps, CFG and mu are the checkpoint's, so another one's pictures are
        # settings that cannot be reused, under a Reuse button that warns. The
        # default is the server's because the page cannot know the checkpoint
        # before /api/status answers, which behind a load is minutes.
        all_models: bool = False,
        # One checkpoint's images, exactly, whichever is running: the Models
        # picker's strip of what the chosen row has made. The same exact
        # scope as the running checkpoint's (Anima@aesthetic-v1.1 is not
        # Anima), and in place of it and of `all_models`. A gallery read, not
        # model work: it answers while the server is cold or switching.
        model: str | None = Query(None, min_length=1, max_length=512),
    ) -> GalleryPage:
        # Every storage call runs off the loop: SQLite's busy handler waits up
        # to 15 s, and a write that lands
        # while `clear.py` holds the file would otherwise freeze the SSE stream,
        # the cancel endpoint and the status poll for that long.
        try:
            # One row past the page, so `has_more` is a fact rather than a guess
            # from a total that keeps growing while the page is read.
            items, total, others, newest = await asyncio.to_thread(
                storage.list_images_at_head,
                limit=limit + 1,
                offset=offset,
                search=search,
                favorites_only=favorites,
                before=before,
                model_id=model or model_scope(all_models),
            )
        except ValueError as exc:  # a filter term that is not a number
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return GalleryPage(
            items=items[:limit],
            total=total,
            limit=limit,
            offset=offset,
            has_more=len(items) > limit,
            other_models=others,
            newest_image_id=newest,
        )


    @router.get("/api/images/{image_id}", response_model=ImageRecord)
    async def image_record(image_id: int, binned: bool = False) -> ImageRecord:
        """One image's record, by id.

        The panel's refine source is an id like a reference chip's, but it needs
        more than a picture: the source's own size is what the scale buttons
        multiply, and a source restored from localStorage or from a refined
        record's `refine` may be on no gallery page the browser has loaded. A 404
        is how the panel learns the source has been deleted - or binned, which
        the detail says; `binned` answers for an image in the recycle bin too.
        """
        return await _record_or_404(image_id, binned)


    @router.get("/api/images/{image_id}/thumb")
    async def image_thumb(image_id: int, binned: bool = False):
        """The thumbnail for one image, by id.

        The panel's reference chips hold ids and nothing else - a reference
        restored from a record or from localStorage may name an image no loaded
        gallery page contains - so they need one request that turns an id into a
        picture. A redirect rather than a read: the file is already served by the
        `/outputs` mount, with its caching. 404 for a binned image unless `binned`.
        """
        record = await _record_or_404(image_id, binned)
        return RedirectResponse(record.thumb_url or record.url)


    @router.post("/api/images/{image_id}/favorite")
    async def favorite(image_id: int, value: bool = True):
        # 404 for an image in the recycle bin too: it is starred or not as
        # it was binned, until a Restore.
        record = await asyncio.to_thread(storage.set_favorite, image_id, value)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown image")
        return record


    @router.post("/api/images/bin", response_model=BinResult)
    async def bin_images(body: IdsBody) -> BinResult:
        """Move images to the recycle bin: the page's delete, which Restore undoes.

        Rows and files stay as they are; every live-facing read leaves the
        image out from now on. `already` is an id another press or tab binned
        first, `missing` one the gallery does not hold at all.
        """
        result = await asyncio.to_thread(gallery_delete.bin_images, body.ids)
        get_jobs().forget_images(result["binned"], binned=True)
        return BinResult(**result)


    @router.post("/api/images/restore", response_model=RestoreResult)
    async def restore_images(body: IdsBody) -> RestoreResult:
        """Bring images back out of the recycle bin, usable again at once."""
        result = await asyncio.to_thread(gallery_delete.restore_images, body.ids)
        get_jobs().restore_images(result["restored"])
        return RestoreResult(**result)


    @router.get("/api/gallery/bin", response_model=GalleryPage)
    async def recycle_bin(
        limit: int = Query(48, ge=1, le=max(1024, max_grid_images)),
        offset: int = Query(0, ge=0),
        before: int | None = Query(None, ge=1),
        search: str | None = None,
    ) -> GalleryPage:
        """The recycle bin, paged as the gallery is, newest id first.

        Every checkpoint's: the bin is one place, and Empty recycle bin empties
        all of it. The search box narrows it as it does the gallery; Starred
        does not apply.
        """
        try:
            items, total, _, newest = await asyncio.to_thread(
                storage.list_images_at_head,
                limit=limit + 1,
                offset=offset,
                search=search,
                before=before,
                binned=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return GalleryPage(
            items=items[:limit],
            total=total,
            limit=limit,
            offset=offset,
            has_more=len(items) > limit,
            other_models=0,
            newest_image_id=newest,
        )


    @router.get("/api/gallery/bin/summary", response_model=BinSummary)
    async def recycle_bin_summary() -> BinSummary:
        """The bin's count and size, which binned images a live one needs, and
        whether a job is queued or running: what the footer and Empty recycle
        bin's confirmation say."""
        found = await asyncio.to_thread(gallery_delete.bin_summary)
        return BinSummary(**found, job_active=get_jobs().busy())


    @router.delete("/api/gallery/bin")
    async def empty_recycle_bin(
        expected: int = Query(ge=0),
        pin: str | None = Query(default=None, max_length=64),
    ) -> dict:
        """Empty recycle bin: every binned image deleted from disk, for good.

        `expected` is the count the confirmation showed and `pin` the summary's
        `pin` for its ids, both compared under the delete's lock, so an image
        binned in another tab while the dialog was open is answered 409 rather
        than deleted on a confirmation that never covered it - even where a
        Restore there kept the count the same.
        """
        try:
            result = await asyncio.to_thread(gallery_delete.empty_bin, expected, pin)
        except gallery_delete.SelectionChanged as exc:
            now = (f"{exc.found} images are in it now, not the {exc.expected} confirmed"
                   if exc.found != exc.expected
                   else f"other images are in it now than the {exc.expected} confirmed")
            raise HTTPException(
                status_code=409,
                detail=f"The recycle bin changed: {now}. Nothing was deleted; look again.",
            ) from exc
        get_jobs().forget_images(result["ids"])
        return {"deleted": result["ids"], "files": result["files"]}


    @router.delete("/api/images/{image_id}")
    async def delete(image_id: int) -> dict:
        """Delete one image and its PNG for good, live or binned: the bin
        view's Delete permanently. The page's ordinary delete bins."""
        if not await asyncio.to_thread(gallery_delete.delete_image, image_id):
            raise HTTPException(status_code=404, detail="unknown image")
        # A page that reattaches to the job that made it is replayed the job's
        # records; a deleted one came back as a broken tile.
        get_jobs().forget_images([image_id])
        return {"deleted": image_id}


    @router.get("/api/gallery/unstarred", response_model=UnstarredPreview)
    async def unstarred(
        search: str | None = None,
        all_models: bool = False,
        newer_than: int | None = Query(default=None, ge=0),
    ) -> UnstarredPreview:
        """What "Delete unstarred" would remove under this filter, and what it would keep.

        Under the gallery's checkpoint scope too, since the delete is: the count
        the dialog shows has to be the count of what the page is showing. What
        the delete keeps besides the stars - the images a starred one was made
        from - comes off the count with them, or every confirmed number would
        meet a different one under the delete's lock and answer 409.

        `job_active` is whether a job is queued or running here - the server's
        answer, not the asking page's: a job another tab started can land an
        image in scope while the dialog is open just the same, and the dialog
        says so before the count is typed.

        `newer_than` is the newest id the page held before it brought its grid
        up to date, and `newer` how many of the count are above it: the dialog's
        "made since this page last loaded", counted as the delete counts.
        """
        found = await asyncio.to_thread(
            gallery_delete.matching_preview, _unstarred_scope(search), model_scope(all_models), newer_than,
            live_only=True,
        )
        return UnstarredPreview(
            unstarred=found["records"] - found["favorites"] - found["sources_kept"],
            job_active=get_jobs().busy(),
            **found,
        )


    @router.delete("/api/gallery/unstarred")
    async def delete_unstarred(
        search: str | None = None, expected: int = Query(ge=0), all_models: bool = False
    ) -> dict:
        """The gallery's Delete unstarred: `clear --where <search> --keep-favorites`
        over HTTP, into the recycle bin rather than off the disk.

        The one bulk delete the page offers, and the inverse of the per-card one:
        star what is worth keeping, then remove the rest of the run in one gesture
        rather than the rest one card at a time. The scope is the search box read
        strictly - `key:value` terms only, an empty box meaning the whole gallery
        - and `expected` is the count the dialog showed, so a sweep that landed
        more unstarred images while it was open is answered 409 rather than
        swept up on a number confirmed for fewer. A starred image keeps what it
        was made from - a pass its source, a render its references - since the
        star is a promise to keep it reproducible. The images move to the
        recycle bin, files untouched, so a Restore undoes it; only Empty recycle
        bin removes them from disk. Images already in the bin are not counted.
        The checkpoint scope is the page's: with All models off, another
        checkpoint's images are not on the screen, and a sweep must not reach
        what is not on the screen. `deleted` repeats `binned` for a page from
        before the bin.
        """
        scope = _unstarred_scope(search)
        try:
            result = await asyncio.to_thread(
                gallery_delete.delete_matching,
                scope,
                keep_favorites=True,
                expected=expected,
                model_id=model_scope(all_models),
                to_bin=True,
            )
        except gallery_delete.SelectionChanged as exc:
            # Whether a job may move the count again is the server's to say: the
            # job is as likely another tab's as the asking page's.
            running = (
                " A job is still running and may add more images here: press Delete "
                "unstarred again, or let it finish first."
                if get_jobs().busy() else ""
            )
            raise HTTPException(
                status_code=409,
                detail=f"The selection changed: {exc.found} unstarred images match now, "
                f"not the {exc.expected} confirmed. Nothing was moved; look again.{running}",
            ) from exc
        get_jobs().forget_images(result["ids"], binned=True)
        return {"binned": result["ids"], "deleted": result["ids"], "files": result["files"]}


    return router
