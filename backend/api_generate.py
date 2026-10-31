"""Single and grid submission: Generate, Run grid and its preview. Re-render
is `api_rerender.py`'s; run details and Finish run are `api_runs.py`'s."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from . import gallery_reports, grid, submission, then_refine
from .app_context import AppServices
from .execution import require_configured_loader_available
from .model_selection import current_settings
from .schemas import (
    GenerateRequest,
    GenerateSubmission,
    GridPreview,
    GridRequest,
    JobAccepted,
    ThenRefine,
    error_detail,
)


async def expand_grid(
    services: AppServices, request: GridRequest
) -> tuple[list[grid.Cell], dict[str, list], int, int, str]:
    """The grid as cells, its resolved axes, the numbers resume dropped as
    held by the gallery and as queued in a live job, its run id.

    The cells come back as spelled, duplicates included: `preview_grid`
    counts the cells that would render an identical image and says so, and
    the run renders what was asked. Dropping them here would leave a
    `{avoid}` axis on klein with one cell of three and a sheet that cannot
    say why - the warning is the honest half, the fix is the user's grid.

    The axes come back from here rather than being resolved again by the
    preview, so a prompt whose placeholders do not line up with its variables
    fails once, as a 422, instead of raising out of the response model. The run
    id is computed over the *whole* grid, before resume subtracts anything, so
    the job that continues a night is stamped with the same id as the job that
    started it. Prompt variants, when given, are the prompt axis; the preview
    then shows their text under `prompt` and their labels under `variant`.

    Resume drops the cells a live job already plans too, so the grid tick,
    the banner's Resume and a finish line's, pressed while the same run is
    still queued, render nothing twice; counted apart, since the preview's
    "already done" is the gallery's.
    """
    selected = current_settings(services.settings())
    try:
        require_configured_loader_available(selected)
        base = request.base(selected.loader_execution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
    if request.refine is not None:
        unavailable = (
            await submission.inpaint_unavailable(services, outpainting=request.outpaint is not None)
            if request.inpainting else await submission.refine_unavailable(services)
        )
        if unavailable is not None:
            raise HTTPException(status_code=422, detail=unavailable)
    try:
        # Listed seeds are an axis the preview shows like the others; counted
        # seeds are derived from the base and are not.
        seed_list = grid.parse_seeds(request.seed_axis)
        cells = grid.expand(
            base,
            prompt=request.prompt_axis,
            steps=request.steps_axis,
            cfg=request.cfg_axis,
            mu=request.mu_axis,
            lora_scale=request.lora_scale_axis,
            strength=request.strength_axis,
            seeds=request.seeds,
            seed_list=seed_list or None,
            variants=request.prompt_variants,
            prompt_mode=request.prompt_mode,
            size=request.size_axis,
        )
        # Random choices resolve once. A second expansion with an unset seed
        # would list another sample beside the cells whose counts we report.
        values = grid.axis_values(
            request, request.steps_axis, request.cfg_axis, request.mu_axis,
            request.prompt_axis, request.lora_scale_axis,
            variants=request.prompt_variants, strength=request.strength_axis,
            prompt_mode=request.prompt_mode, cells=cells,
            size=request.size_axis,
        )
        values["seed"] = list(seed_list)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
    # Sources and masks are held to each actual canvas; the base size may be
    # replaced completely by the paired size axis.
    for group in submission.cells_by_size(cells).values():
        await submission.refuse_unusable_inputs(services, group[0].request)
    run_id = grid.run_id(cells, services.model_id(), services.quantization())
    if not request.resume:
        return cells, values, 0, 0, run_id
    # The queue first, then the gallery: see `not_yet_planned`.
    planned = services.jobs().planned_keys()
    unheld = await submission.not_yet_rendered(services, cells, request.width, request.height)
    kept = submission.not_yet_planned(services, unheld, planned)
    return kept, values, len(cells) - len(unheld), len(unheld) - len(kept), run_id


async def then_refine_spec(
    services: AppServices, spec: ThenRefine | None, requests: list[GenerateRequest],
) -> ThenRefine | None:
    """Then refine as the job runs it, its strength resolved, or a 422 in
    words where these cells cannot each be followed by a pass."""
    if spec is None or not requests:
        return None
    spec = then_refine.resolved(spec)
    reason = then_refine.refusal(requests, spec, await submission.refine_unavailable(services))
    if reason is not None:
        raise HTTPException(status_code=422, detail=reason)
    return spec


def create_router(services: AppServices) -> APIRouter:
    """The routes, reading the app's state through `services` per request."""
    router = APIRouter()

    @router.post("/api/generate", response_model=JobAccepted)
    async def generate(submitted: GenerateSubmission) -> JobAccepted:
        try:
            selected = current_settings(services.settings())
            require_configured_loader_available(selected)
            request = submitted.base(selected.loader_execution)
            request.check_budget()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        await submission.refuse_unusable_inputs(services, request)
        try:
            request.check_refine_steps()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # The job waits on ensure_loaded() itself, so a previously failed load is
        # retried here rather than short-circuiting - the user may have just been
        # granted repo access.
        spec = await then_refine_spec(services, submitted.then_refine, [request])
        job = await services.jobs().submit(request, then_refine=spec)
        return JobAccepted.of(job)


    @router.post("/api/grid", response_model=JobAccepted)
    async def run_grid(request: GridRequest) -> JobAccepted:
        if request.resume:
            async with services.jobs().rerun_admission:
                return await submit_grid(request)
        return await submit_grid(request)


    async def submit_grid(request: GridRequest) -> JobAccepted:
        """Queue a sweep over the sampling axes as one job.

        One job rather than one per cell: a grid is a night's work, and it has to be
        cancellable, reattachable and ahead of nothing in the queue as a unit.
        """
        cells, _, _, queued, run_id = await expand_grid(services, request)
        if not cells:
            # Every cell is already in the gallery, or queued. Queueing an empty
            # job would report "done" having rendered nothing, which reads like
            # a failure.
            raise HTTPException(status_code=409, detail=(
                "every cell is already rendered or queued" if queued else "every cell is already rendered"
            ))
        spec = await then_refine_spec(services, request.then_refine, [cell.request for cell in cells])
        job = await services.jobs().submit_cells(
            cells,
            run_id=run_id,
            budget_minutes=request.budget_minutes,
            prompt_template=request.prompt,
            comparison=bool(request.prompt_variants),
            then_refine=spec,
        )
        return JobAccepted.of(job)


    @router.post("/api/grid/preview", response_model=GridPreview)
    async def preview_grid(request: GridRequest) -> GridPreview:
        """What that grid would do, without queueing it.

        Same expansion as the real thing, so the count and the error messages the
        panel shows are the ones the run would have produced.

        Without Resume, the cells the gallery already holds are counted too: Run
        grid again after a night with a failed cell renders the whole night again,
        and the count is what says Resume would not. Only where the seeds are
        fixed - an explicit base or a list - since a random base matches nothing.
        """
        cells, values, skipped, queued, _ = await expand_grid(services, request)
        in_gallery = 0
        if not request.resume and (request.seed is not None or request.seed_axis.strip()):
            in_gallery = len(cells) - len(
                await submission.not_yet_rendered(services, cells, request.width, request.height)
            )
        images = sum(cell.request.num_images for cell in cells)
        listed = len(values.get("seed", []))
        spec = await then_refine_spec(services, request.then_refine, [cell.request for cell in cells])
        priced = then_refine.execution_units(cells, spec)
        pass_sizes = []
        if spec is not None and cells:
            pass_sizes = sorted({
                (probe.width, probe.height) for probe in (then_refine.stand_in(cell.request, spec) for cell in cells)
            })
        prices, disk_bytes = await asyncio.to_thread(
            gallery_reports.execution_estimates, priced, services.model_id(),
        )
        # Under Resume the seeds left are no longer alike, and the page says
        # nothing of the budget there.
        budget_seeds = (
            grid.seeds_within_budget(
                priced, listed or request.seeds, None, request.budget_minutes * 60, by_size=prices,
            )
            if request.budget_minutes and not request.resume
            else None
        )
        # Where a refine pass starts on the schedule, for the Refine dialog: only
        # where every cell starts at one point - the same steps, mu and strength
        # as the pipeline will run them - which a sweep's do not. Schedulers
        # only, no weights, and None before a load.
        start_sigma = None
        if request.refine is not None and cells:
            applied = [services.backend().applied(cell.request) for cell in cells]
            if len({(r.width, r.height, r.steps, r.mu, r.strength) for r in applied}) == 1 and applied[0].refine is not None:
                start_sigma = services.backend().start_sigma(applied[0])
        elif spec is not None and cells and len({(cell.request.width, cell.request.height, cell.request.steps) for cell in cells}) == 1:
            # Then refine's pass, where every cell's starts at one point.
            start_sigma = services.backend().start_sigma(
                services.backend().applied(then_refine.stand_in(cells[0].request, spec))
            )
        return GridPreview(
            values=values,
            prompt_sample=(
                request.prompt_mode == "random" and request.seed is None and not request.seed_axis.strip()
            ),
            cells=len(cells),
            images=images,
            skipped=skipped,
            queued=queued,
            in_gallery=in_gallery,
            # As the pipeline would run them: the capability flags the status
            # endpoint publishes, applied cell by cell, say which cells collapse
            # into one picture before the night is spent finding out.
            duplicates=grid.duplicate_count(cells, services.backend().applied),
            summary=grid.describe(
                values, listed or request.seeds, listed=bool(listed),
                prompt_mode=request.prompt_mode, batch=request.num_images,
            ),
            seconds=grid.estimate_seconds(
                priced, None, by_size=prices
            ),
            budget_seeds=budget_seeds,
            bytes=disk_bytes,
            start_sigma=start_sigma,
            passes=images if spec is not None else 0,
            pass_width=pass_sizes[0][0] if len(pass_sizes) == 1 else None,
            pass_height=pass_sizes[0][1] if len(pass_sizes) == 1 else None,
            pass_sizes=pass_sizes,
        )

    return router
