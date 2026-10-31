"""Re-render: the selection again at another size, its preview and its
refusals (`foreign_refusal`)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException

from . import grid, submission
from .app_context import AppServices
from .sizes import nearest_preset
from .schemas import (
    GenerateRequest,
    ImageRecord,
    JobAccepted,
    RerenderAccepted,
    RerenderGone,
    RerenderPreview,
    RerenderRequest,
    RerenderSize,
    RerenderUnmatched,
    error_detail,
)


def foreign_refusal(
    services: AppServices, count: int, foreign: list[str], picked: list[int] | None = None
) -> str:
    """Re-render's refusal of a selection that reaches other checkpoints, with the way out.

    Unticking All models is the exact way: the gallery's scope is the running
    id itself (`AppServices.model_scope`). The search box's `model:` is a substring, so
    it is offered only where the running id, whole, is in none of the other
    ids selected - and only then does adding it leave them out. The id's tail
    was offered, and `model:Anima` is Animagine too; nor is the whole id safe
    everywhere (`Qwen/Qwen-Image` is in `Qwen/Qwen-Image-Edit`). Picked cards
    are no filter's - neither way out reaches them - so for those (`picked`,
    the foreign ids) the way out is unpicking them, named as the page names
    images, the first six.
    """
    running = services.model_id()
    lead = (
        f"{count} of the selected images {'was' if count == 1 else 'were'} made on "
        f"{', '.join(foreign)}; the server is running {running}, whose steps, CFG and mu "
        f"differ."
    )
    if picked is not None:
        names = [f"#{image_id}" for image_id in picked[:6]]
        if len(picked) > 6:
            names.append(f"{len(picked) - 6} more")
        listed = f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]
        return f"{lead} Unpick {listed}."
    narrows = not any(ch.isspace() for ch in running) and not any(
        running.lower() in other.lower() for other in foreign
    )
    search = f", or add model:{running} to the search" if narrows else ""
    return f"{lead} Untick All models{search}."


@dataclass
class _RerenderGroup:
    """One size of a re-render: its cells, those it would queue now, and
    the run id of the job it makes."""

    width: int
    height: int
    cells: list[grid.Cell]
    kept: list[grid.Cell]
    run_id: str


@dataclass
class _RerenderPlan:
    ids: list[int]
    deleted: list[RerenderGone]
    groups: list[_RerenderGroup]
    unmatched: list[RerenderUnmatched]
    pruned: int
    planned: int


def create_router(services: AppServices) -> APIRouter:
    """The routes, reading the app's state through `services` per request."""
    router = APIRouter()

    @router.post("/api/rerender/preview", response_model=RerenderPreview)
    async def preview_rerender(request: RerenderRequest) -> RerenderPreview:
        """What Re-render would queue, size by size, without queueing it.

        No side effects, so it is not a model-dependent request in the
        middleware's sense, as Swap tags' preview is not: the queueing
        request is, and carries the revision. A selection Re-render would
        refuse is refused here in the same words.
        """
        plan = await rerender_plan(request)
        return RerenderPreview(
            ids=plan.ids,
            sizes=[
                RerenderSize(width=group.width, height=group.height, images=len(group.cells), queued=len(group.kept))
                for group in plan.groups
            ],
            unmatched=plan.unmatched, deleted=plan.deleted,
            queued=sum(len(group.kept) for group in plan.groups),
            jobs=sum(1 for group in plan.groups if group.kept),
            pruned=plan.pruned, planned=plan.planned,
        )


    @router.post("/api/rerender", response_model=RerenderAccepted)
    async def rerender(request: RerenderRequest) -> RerenderAccepted:
        """Re-render the selection, leaving out gallery and live-job cells."""
        async with services.jobs().rerun_admission:
            return await rerender_selection(request)


    async def rerender_plan(request: RerenderRequest) -> _RerenderPlan:
        """The selection as cells, one group per size, each with what it would
        queue now - or the refusal, in the words the queueing request says it.

        At one size (`width`/`height`) every record is one group at it; at a
        `tier`, each record goes to its own aspect's entry at that tier
        (`sizes.nearest_preset`), and a record whose size is no preset's
        aspect is answered in `unmatched`, never guessed or queued. Every
        refusal below holds per size: an inpaint is refused at any size but
        its own, a budget at the size that passes it.
        """
        if request.tier is None:
            assert request.width is not None and request.height is not None, "the request model holds a size"
            probe = GenerateRequest(prompt="x", width=request.width, height=request.height)
            try:
                probe.check_budget()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        records, gone, binned = await submission.resolve_selection(
            services, ids=request.ids, search=request.search, favorites=request.favorites,
            all_models=request.all_models,
        )
        deleted = [RerenderGone(id=image_id, detail=submission.gone_detail(image_id, binned)) for image_id in gone]
        if not records:
            said = "; ".join(f"#{item.id} is {item.detail}" for item in deleted[:6])
            more = f"; and {len(deleted) - 6} more" if len(deleted) > 6 else ""
            raise HTTPException(status_code=404, detail=f"none of the selected images is left: {said}{more}")
        # Steps, CFG and mu are checkpoint-specific - Turbo's 8 / 0 / 1.15 on FLUX
        # is a broken image recorded under FLUX - so a record from another
        # checkpoint is refused rather than rendered wrong. From a filter, only
        # reachable with `all_models`: the scope keeps them out of it otherwise;
        # picked cards are another checkpoint's wherever they were picked.
        foreign = sorted({r.model_id for r in records if r.model_id != services.model_id()})
        if foreign:
            others = [r.id for r in records if r.model_id != services.model_id()]
            raise HTTPException(status_code=422, detail=foreign_refusal(
                services, len(others), foreign, others if request.ids is not None else None,
            ))
        # An instruction edit's size is its source's frozen geometry: never
        # re-rendered at another, and at its own already there. Named by id
        # before any cell is built (an unresolved import would refuse in
        # the request model's words); More seeds is its way to new seeds.
        edits = [record for record in records if record.edit is not None]
        if edits:
            raise HTTPException(status_code=422, detail=submission.edit_rerender_detail(
                edits, picked=request.ids is not None, others=len(edits) < len(records),
            ))
        # The records by the size each renders at, in the resolver's order.
        sized: dict[tuple[int, int], list[ImageRecord]] = {}
        unmatched: list[RerenderUnmatched] = []
        if request.tier is None:
            assert request.width is not None and request.height is not None
            sized[(request.width, request.height)] = records
        else:
            for record in records:
                size = nearest_preset(record.width, record.height, request.tier)
                if size is None:
                    unmatched.append(RerenderUnmatched(id=record.id, width=record.width, height=record.height))
                else:
                    sized.setdefault(size, []).append(record)
            if not sized:
                said = "; ".join(f"#{item.id} is {item.width}x{item.height}" for item in unmatched[:6])
                more = f"; and {len(unmatched) - 6} more" if len(unmatched) > 6 else ""
                raise HTTPException(status_code=422, detail=(
                    f"none of the picked images is a preset's aspect, so none has a {request.tier:g} MP size: "
                    f"{said}{more}"
                ))
            # Each size its own budget: the one that passes it is refused
            # by name, and nothing is queued.
            for width, height in sized:
                try:
                    GenerateRequest(prompt="x", width=width, height=height).check_budget()
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
        matched = [record for group in sized.values() for record in group]
        # An inpaint whose mask this gallery lacks is named by its id, which
        # the per-combination check below could only say by the mask's hash.
        lost = await asyncio.to_thread(submission.missing_masks, matched)
        if lost:
            raise HTTPException(status_code=422, detail=submission.missing_mask_detail(lost))
        # And an outpaint whose spec does not build its own row's mask: run
        # again it would paint under another mask under the same key.
        mismatched = await asyncio.to_thread(submission.mismatched_outpaints, matched)
        if mismatched:
            raise HTTPException(status_code=422, detail=submission.mismatched_outpaint_detail(mismatched))
        # So is an inpaint at another size than its own - its source's, which
        # its mask covers - where the per-combination check would name the
        # source the user never picked.
        for (width, height), group in sized.items():
            resized = submission.resized_inpaints(group, width, height)
            if resized:
                raise HTTPException(status_code=422, detail=submission.resized_inpaint_detail(
                    resized, width, height, picked=request.ids is not None,
                    others=len(resized) < len(records),
                ))
        model, quantization = services.model_id(), services.quantization()
        groups: list[_RerenderGroup] = []
        for (width, height), group in sized.items():
            # Oldest first (the resolver's order), so the re-render lands in the
            # order the originals did; every record's reproduction set but its
            # size. A row the request models refuse (a spec another build or an
            # import let through) is a worded refusal, never a 500.
            try:
                cells = grid.cells_from_records(group, width=width, height=height)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
            # One check per distinct combination rather than per cell: a selection of
            # 400 images carries at most a handful of adapters and sources between them.
            # Every cell is at its group's one size, so an inpaint's mask is held
            # to it here: re-rendered elsewhere, its mask no longer covers it. An
            # outpaint (frozen, so it keys) is checked at that size per padding,
            # never sharing a plain refine's verdict over the same source.
            for spec in {submission.input_spec(c.request) for c in cells}:
                if spec != submission.NO_INPUTS:
                    await submission.refuse_unusable_inputs(services, cells[0].request.model_copy(
                        update={"lora": spec[0], "reference": list(spec[1]), "refine": spec[2], "mask": spec[3],
                                "outpaint": spec[4], "upscaler": spec[5], "edit": spec[6]}
                    ))
            try:
                for cell in cells:
                    cell.request.check_refine_steps()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            # One run id per size, each the shape every other run's has; at
            # one size, the whole selection's, as it always was.
            groups.append(_RerenderGroup(width, height, cells, cells, grid.run_id(cells, model, quantization)))
        # The queue first, then the gallery: a live cell can finish while
        # the scan runs. Admission stays held until submit_cells has added
        # these jobs, including their awaited mask writes.
        planned = services.jobs().planned_keys()
        pruned = already_planned = 0
        for group in groups:
            unheld = await submission.not_yet_rendered(services, group.cells, group.width, group.height)
            group.kept = submission.not_yet_planned(services, unheld, planned)
            pruned += len(group.cells) - len(unheld)
            already_planned += len(unheld) - len(group.kept)
        return _RerenderPlan(
            ids=[record.id for record in records], deleted=deleted, groups=groups, unmatched=unmatched,
            pruned=pruned, planned=already_planned,
        )


    async def rerender_selection(request: RerenderRequest) -> RerenderAccepted:
        """Queue the gallery's current selection again at another size: one
        job at one size, or one per size at a tier.

        The documented workflow ends at "star the winners"; getting them at full
        size then meant Reuse, retype the size, Generate, twenty times over - and
        the clicking is where a seed gets mistyped. Here every selected record is
        a cell with its own settings and seed, deduplicated against what the
        gallery already holds at the target size, so a re-render can itself be
        resumed. A sibling of `run_grid`, and it shares the run-id and resume
        machinery with it. The selection is the picked cards where `ids` names
        them, else the filter (`submission.resolve_selection`); a picked image
        gone since is left out and named in `deleted`, and with none left the
        answer is 404 naming them all. `expected` is the ids the preview
        resolved: a selection that changed since is refused (409). At a tier,
        one job per size, the first at the top level and the rest in `more`,
        as Swap tags answers.
        """
        plan = await rerender_plan(request)
        if request.expected is not None and sorted(set(request.expected)) != sorted(plan.ids):
            raise HTTPException(
                status_code=409, detail="the selection changed since the check; check Re-render again",
            )
        groups = [group for group in plan.groups if group.kept]
        if not groups:
            pruned, already_planned = plan.pruned, plan.planned
            state = "rendered or queued" if already_planned and pruned else "queued" if already_planned else "rendered"
            where = (
                f"at {request.width}x{request.height}" if request.tier is None
                else f"at its {request.tier:g} MP size"
            )
            raise HTTPException(status_code=409, detail=f"every selected image is already {state} {where}")
        accepted = []
        for group in groups:
            job = await services.jobs().submit_cells(group.kept, run_id=group.run_id)
            accepted.append(JobAccepted.model_validate({
                "job_id": job.id, "seed": job.seed, "run_id": job.run_id, "estimate_seconds": job.estimate_seconds,
            }))
        first = accepted[0]
        return RerenderAccepted.model_validate({
            **first.model_dump(),
            "deleted": plan.deleted,
            "pruned": plan.pruned, "planned": plan.planned,
            "more": accepted[1:], "unmatched": plan.unmatched,
        })

    return router
