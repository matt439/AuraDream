"""More seeds: its preview and its submission; the per-batch plan is
`more_seeds_planner.more_seeds_plan`."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from . import gallery_reports, grid, model_selection, more_seeds, submission, then_refine
from .app_context import AppServices
from .more_seeds_intents import IntentProblem
from .more_seeds_planner import more_seeds_plan
from .schemas import (
    JobAccepted,
    MoreSeedsAccepted,
    MoreSeedsPreview,
    MoreSeedsRequest,
)


# How many batches the More seeds preview lists; its counts cover all of them.
MORE_SEEDS_PREVIEW_ROWS = 200


def _more_seeds_jobs(batches: list[more_seeds.Batch]) -> dict[tuple, list[more_seeds.Batch]]:
    """The queued batches by the job they go in: one per run (else job, else
    image), size and Then refine spec, so each job's images join the one run
    they extend, the job keeps one size, as every job does, and carries one
    pass (`submit_cells(then_refine=)`), whose pass run id it derives from
    the run's - the run's own pass run."""
    jobs_of: dict[tuple, list[more_seeds.Batch]] = {}
    for batch in batches:
        template = batch.template
        jobs_of.setdefault(
            (batch.key[0], template.width, template.height, then_refine.spec_key(batch.then_refine)), [],
        ).append(batch)
    return jobs_of


def _passes(batches: list[more_seeds.Batch]) -> int:
    """The Then refine passes following these batches' new bases."""
    return sum(len(batch.seeds) for batch in batches if batch.then_refine is not None)


def _intent_context(services: AppServices) -> tuple:
    return (model_selection.revision(), services.model_id(), services.quantization())


def _intent_payload(request: MoreSeedsRequest, expected: list[int] | None = None) -> dict:
    """The submitted choice, including the preview's image pin, not its token.

    Order and repeated ids do not change a selection. All other validated
    fields stay explicit, including defaults, so a replay cannot change its
    count, scope or legacy Batch fallback.
    """
    payload = request.model_dump(mode="json", exclude={"intent"})
    if expected is not None:
        payload["expected"] = expected
    for name in ("ids", "expected"):
        if payload[name] is not None:
            payload[name] = sorted(set(payload[name]))
    return payload


def create_router(services: AppServices) -> APIRouter:
    """The routes, reading the app's state through `services` per request."""
    router = APIRouter()

    @router.post("/api/more-seeds/preview", response_model=MoreSeedsPreview)
    async def preview_more_seeds(request: MoreSeedsRequest) -> MoreSeedsPreview:
        """What More seeds would queue, batch by batch, without queueing it,
        and how long it would take.

        No gallery or queue side effects. The optional short-lived intent
        records only this preview's choices for a safely replayable submit.
        The queueing request, rather than this preview, is model-dependent.
        """
        manager = services.jobs()
        context = _intent_context(services)
        ids, results, batches = await more_seeds_plan(services, request)
        queued = [batch for batch in batches if batch.status == more_seeds.QUEUED]
        model = services.model_id()
        # Time, summed per size from that size's history; a size with none
        # leaves the whole estimate out, and how many have none is said.
        seconds = 0.0
        unpriced = 0
        # Each carried pass priced at its own size, as the grid preview
        # prices Then refine's stand-ins (`then_refine.execution_units`).
        units: dict[tuple[int, int], list[grid.Cell]] = {}
        for batch in queued:
            for unit in then_refine.execution_units(batch.cells, batch.then_refine):
                units.setdefault((unit.request.width, unit.request.height), []).append(unit)
        for (width, height), cells in units.items():
            per_step_ms = await asyncio.to_thread(gallery_reports.step_cost_ms, width, height, model)
            refine_costs = await asyncio.to_thread(gallery_reports.refine_cost_ms, width, height, model)
            inpaint_costs = (
                await asyncio.to_thread(gallery_reports.refine_cost_ms, width, height, model, inpaint=True)
                if any(cell.request.inpainting for cell in cells) else None
            )
            priced = grid.estimate_seconds(
                cells, per_step_ms, refine_costs=refine_costs, inpaint_costs=inpaint_costs
            )
            if priced is None:
                unpriced += 1
            else:
                seconds += priced
        carrying = sum(batch.run_then_refine for batch in batches)
        hold = await submission.refine_unavailable(services) if carrying else None
        if context != _intent_context(services) or manager is not services.jobs():
            raise HTTPException(409, "the checkpoint or server changed while checking; review the batches again")
        try:
            intent = manager.more_seeds_intents.issue(_intent_payload(request, ids), context) if queued else None
        except IntentProblem as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc
        return MoreSeedsPreview(
            ids=ids,
            counts=submission.status_counts(results),
            queued=sum(len(batch.seeds) for batch in queued),
            batches=len(queued),
            jobs=len(_more_seeds_jobs(queued)),
            passes=_passes(queued),
            then_refine_batches=carrying,
            then_refine_hold=hold,
            results=results[:MORE_SEEDS_PREVIEW_ROWS],
            total=len(results),
            unrecorded=sum(not batch.recorded for batch in queued),
            seconds=None if unpriced or not queued else round(seconds, 1),
            unpriced_sizes=unpriced,
            count_default=more_seeds.count_default(batches),
            headline=(
                more_seeds.run_headline(queued, request.count, request.favorites)
                if request.run is not None else None
            ),
            intent=intent,
        )


    @router.post("/api/more-seeds", response_model=MoreSeedsAccepted)
    async def more_seeds_submit(request: MoreSeedsRequest) -> MoreSeedsAccepted:
        manager = services.jobs()

        async def admitted() -> dict:
            async with manager.rerun_admission:
                return (await submit_more_seeds(request)).model_dump(mode="json")

        # Older API callers deliberately ask for another extension on every
        # POST. Browser requests use the preview's token and immutable body.
        if request.intent is None:
            return MoreSeedsAccepted.model_validate(await admitted())
        try:
            receipt = await manager.more_seeds_intents.execute(
                request.intent, _intent_payload(request), _intent_context(services), admitted,
            )
        except IntentProblem as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc
        return MoreSeedsAccepted.model_validate(receipt)


    async def submit_more_seeds(request: MoreSeedsRequest) -> MoreSeedsAccepted:
        """Queue `count` more images of every selected batch.

        The sixth producer of cells: each batch's next seeds, consecutive ones
        in a call of its Batch size, every other setting the batch's own
        (`more_seeds.plan` through `grid.cell_from_record`). One job per run the
        batches extend, stamped with that run's id, so the new images join it -
        the Runs picker, the sheet and `run:` show one set - and each job
        keeps one size. `expected` pins the ids the preview resolved. With
        nothing queueable, 422 naming every count.
        """
        ids, results, batches = await more_seeds_plan(services, request)
        if request.expected is not None and sorted(set(request.expected)) != ids:
            raise HTTPException(
                status_code=409,
                detail="the selection changed since the check; review the batches again",
            )
        queued = [batch for batch in batches if batch.status == more_seeds.QUEUED]
        if not queued:
            said = ", ".join(
                f"{n} {status.replace('_', ' ')}" for status, n in submission.status_counts(results).items()
            )
            raise HTTPException(status_code=422, detail=f"no selected batch can have more seeds ({said})")
        accepted = []
        accepted_batches = []
        groups = list(_more_seeds_jobs(queued).values())
        warning = None
        remaining: list[more_seeds.Batch] = []
        for index, group in enumerate(groups):
            try:
                job = await services.jobs().submit_cells(
                    [cell for batch in group for cell in batch.cells],
                    run_id=group[0].run_id, more_seeds=request.count, then_refine=group[0].then_refine,
                )
            except Exception as exc:
                if not accepted:
                    raise
                # Earlier groups are already admitted. Acknowledge their ids
                # rather than conceal them behind an error. This is terminal:
                # replay returns this same partial receipt, never fresh seeds.
                remaining = [batch for later in groups[index:] for batch in later]
                examples = [
                    f"#{batch.template.id} at {batch.template.width}x{batch.template.height}"
                    for batch in remaining[:6]
                ]
                if len(remaining) > len(examples):
                    examples.append(f"and {len(remaining) - len(examples)} more batches")
                detail = str(getattr(exc, "detail", None) or exc).strip().rstrip(".")[:320]
                images = sum(len(batch.seeds) for batch in remaining)
                warning = (
                    f"{images} more image{'' if images == 1 else 's'} in "
                    f"{len(remaining)} batch{'' if len(remaining) == 1 else 'es'} "
                    f"were not queued ({', '.join(examples)}): {detail}. "
                    + ("Retrying this request will not add the remainder." if request.intent is not None else
                       "Review the remaining batches before requesting more; another request adds new seeds.")
                )
                break
            accepted.append(JobAccepted.of(job))
            accepted_batches.extend(group)
        first = accepted[0]
        return MoreSeedsAccepted(
            job_id=first.job_id, seed=first.seed, run_id=first.run_id, estimate_seconds=first.estimate_seconds,
            more=accepted[1:],
            queued=sum(len(batch.seeds) for batch in accepted_batches), batches=len(accepted_batches),
            passes=_passes(accepted_batches),
            warning=warning, not_queued=sum(len(batch.seeds) for batch in remaining),
            not_queued_batches=len(remaining),
        )

    return router
