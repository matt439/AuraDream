"""Single and grid submission: Generate, Run grid, its preview, and Re-render."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from . import gallery_reports, grid, submission
from .app_context import AppServices
from .schemas import (
    GenerateRequest,
    GridPreview,
    GridRequest,
    JobAccepted,
    RerenderRequest,
    error_detail,
)


async def expand_grid(
    services: AppServices, request: GridRequest
) -> tuple[list[grid.Cell], dict[str, list], int, str]:
    """The grid as cells, its resolved axes, the number resume dropped, its run id.

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
    """
    await submission.refuse_unusable_inputs(services, request)
    try:
        request.check_budget()
        values = grid.axis_values(
            request,
            request.steps_axis,
            request.cfg_axis,
            request.mu_axis,
            request.prompt_axis,
            request.lora_scale_axis,
            variants=request.prompt_variants,
            strength=request.strength_axis,
        )
        # Listed seeds are an axis the preview shows like the others; counted
        # seeds are derived from the base and are not.
        seed_list = grid.parse_seeds(request.seed_axis)
        values["seed"] = list(seed_list)
        cells = grid.expand(
            request,
            prompt=request.prompt_axis,
            steps=request.steps_axis,
            cfg=request.cfg_axis,
            mu=request.mu_axis,
            lora_scale=request.lora_scale_axis,
            strength=request.strength_axis,
            seeds=request.seeds,
            seed_list=seed_list or None,
            variants=request.prompt_variants,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
    run_id = grid.run_id(cells, services.model_id(), services.quantization())
    if not request.resume:
        return cells, values, 0, run_id
    kept = await submission.not_yet_rendered(services, cells, request.width, request.height)
    return kept, values, len(cells) - len(kept), run_id


def foreign_refusal(services: AppServices, count: int, foreign: list[str]) -> str:
    """Re-render's refusal of a selection that reaches other checkpoints, with the way out.

    Unticking All models is the exact way: the gallery's scope is the running
    id itself (`AppServices.model_scope`). The search box's `model:` is a substring, so
    it is offered only where the running id, whole, is in none of the other
    ids selected - and only then does adding it leave them out. The id's tail
    was offered, and `model:Anima` is Animagine too; nor is the whole id safe
    everywhere (`Qwen/Qwen-Image` is in `Qwen/Qwen-Image-Edit`).
    """
    running = services.model_id()
    narrows = not any(ch.isspace() for ch in running) and not any(
        running.lower() in other.lower() for other in foreign
    )
    search = f", or add model:{running} to the search" if narrows else ""
    return (
        f"{count} of the selected images {'was' if count == 1 else 'were'} made on "
        f"{', '.join(foreign)}; the server is running {running}, whose steps, CFG and mu "
        f"differ. Untick All models{search}."
    )


def create_router(services: AppServices) -> APIRouter:
    """The routes, reading the app's state through `services` per request."""
    router = APIRouter()

    @router.post("/api/generate", response_model=JobAccepted)
    async def generate(request: GenerateRequest) -> JobAccepted:
        try:
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
        job = await services.jobs().submit(request)
        return JobAccepted(job_id=job.id, seed=job.seed, run_id=job.run_id)


    @router.post("/api/grid", response_model=JobAccepted)
    async def run_grid(request: GridRequest) -> JobAccepted:
        """Queue a sweep over the sampling axes as one job.

        One job rather than one per cell: a grid is a night's work, and it has to be
        cancellable, reattachable and ahead of nothing in the queue as a unit.
        """
        cells, _, _, run_id = await expand_grid(services, request)
        if not cells:
            # Every cell is already in the gallery. Queueing an empty job would
            # report "done" having rendered nothing, which reads like a failure.
            raise HTTPException(status_code=409, detail="every cell is already rendered")
        job = await services.jobs().submit_cells(
            cells,
            run_id=run_id,
            budget_minutes=request.budget_minutes,
            prompt_template=request.prompt,
            comparison=bool(request.prompt_variants),
        )
        return JobAccepted(job_id=job.id, seed=job.seed, run_id=job.run_id)


    @router.post("/api/rerender", response_model=JobAccepted)
    async def rerender(request: RerenderRequest) -> JobAccepted:
        """Queue the gallery's current selection again at another size, as one job.

        The documented workflow ends at "star the winners"; getting them at full
        size then meant Reuse, retype the size, Generate, twenty times over - and
        the clicking is where a seed gets mistyped. Here every selected record is
        a cell with its own settings and seed, deduplicated against what the
        gallery already holds at the target size, so a re-render can itself be
        resumed. A sibling of `run_grid`, and it shares the run-id and resume
        machinery with it.
        """
        probe = GenerateRequest(prompt="x", width=request.width, height=request.height)
        try:
            probe.check_budget()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        records, _, _ = await submission.resolve_selection(
            services, ids=None, search=request.search, favorites=request.favorites, all_models=request.all_models,
        )
        # Steps, CFG and mu are checkpoint-specific - Turbo's 8 / 0 / 1.15 on FLUX
        # is a broken image recorded under FLUX - so a record from another
        # checkpoint is refused rather than rendered wrong. Only reachable with
        # `all_models`: the scope keeps them out of the selection otherwise.
        foreign = sorted({r.model_id for r in records if r.model_id != services.model_id()})
        if foreign:
            count = sum(r.model_id != services.model_id() for r in records)
            raise HTTPException(status_code=422, detail=foreign_refusal(services, count, foreign))
        # Oldest first (the resolver's order), so the re-render lands in the
        # order the originals did; every record's reproduction set but its size.
        cells = grid.cells_from_records(records, width=request.width, height=request.height)
        # One check per distinct combination rather than per cell: a selection of
        # 400 images carries at most a handful of adapters and sources between them.
        for spec in {(c.request.lora, tuple(c.request.reference), c.request.refine) for c in cells}:
            if spec != (None, (), None):
                await submission.refuse_unusable_inputs(services, cells[0].request.model_copy(
                    update={"lora": spec[0], "reference": list(spec[1]), "refine": spec[2]}
                ))
        try:
            for cell in cells:
                cell.request.check_refine_steps()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        run_id = grid.run_id(cells, services.model_id(), services.quantization())
        kept = await submission.not_yet_rendered(services, cells, request.width, request.height)
        if not kept:
            raise HTTPException(
                status_code=409,
                detail=f"every selected image is already rendered at {request.width}x{request.height}",
            )
        job = await services.jobs().submit_cells(kept, run_id=run_id)
        return JobAccepted(job_id=job.id, seed=job.seed, run_id=job.run_id)


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
        cells, values, skipped, _ = await expand_grid(services, request)
        in_gallery = 0
        if not request.resume and (request.seed is not None or request.seed_axis.strip()):
            in_gallery = len(cells) - len(
                await submission.not_yet_rendered(services, cells, request.width, request.height)
            )
        # Cost is per step, and a grid's cells rarely agree on how many they take.
        per_step_ms = await asyncio.to_thread(
            gallery_reports.step_cost_ms, request.width, request.height, services.model_id()
        )
        refine_costs = await asyncio.to_thread(
            gallery_reports.refine_cost_ms, request.width, request.height, services.model_id()
        )
        per_image_bytes = await asyncio.to_thread(
            gallery_reports.image_bytes, request.width, request.height, services.model_id()
        )
        images = len(cells) * request.num_images
        listed = len(values.get("seed", []))
        # Under Resume the seeds left are no longer alike, and the page says
        # nothing of the budget there.
        budget_seeds = (
            grid.seeds_within_budget(
                cells, listed or request.seeds, per_step_ms, request.budget_minutes * 60,
                refine_costs=refine_costs,
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
            if len({(r.steps, r.mu, r.strength) for r in applied}) == 1 and applied[0].refine is not None:
                start_sigma = services.backend().start_sigma(applied[0])
        return GridPreview(
            values=values,
            cells=len(cells),
            images=images,
            skipped=skipped,
            in_gallery=in_gallery,
            # As the pipeline would run them: the capability flags the status
            # endpoint publishes, applied cell by cell, say which cells collapse
            # into one picture before the night is spent finding out.
            duplicates=grid.duplicate_count(cells, services.backend().applied),
            summary=grid.describe(values, listed or request.seeds, listed=bool(listed)),
            seconds=grid.estimate_seconds(cells, per_step_ms, refine_costs=refine_costs),
            budget_seeds=budget_seeds,
            bytes=per_image_bytes * images if per_image_bytes else None,
            start_sigma=start_sigma,
        )

    return router
