"""FastAPI app: text-to-image generation plus a browsable gallery."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Run as `python -m backend.main`, settle the checkpoint before `.config`
# freezes it: the terminal menu when there are no arguments, the
# --model/--quant/--offload flags otherwise. Imported by uvicorn or the tests
# instead, this does nothing and the environment stands as it is.
if __name__ == "__main__":
    from .launch import configure

    configure()

from . import grid, loras, storage  # noqa: E402
from .backends import get_backend  # noqa: E402
from .config import settings  # noqa: E402
from .jobs import BackendBusy, JobManager, short_error  # noqa: E402
from .model_defaults import defaults_for, guide_for, hints_for  # noqa: E402
from .schemas import (  # noqa: E402
    GalleryPage,
    GenerateRequest,
    GridPreview,
    GridRequest,
    JobAccepted,
    ModelStatus,
    PromptExpansion,
    PromptExpansionRequest,
    RerenderRequest,
    RunSummary,
    error_detail,
)
from .translation import prompt_translator  # noqa: E402

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
)
log = logging.getLogger("krea")

backend = get_backend("diffusers")

# Loading ~13GB of weights takes minutes, so it runs off-request.
_load_state: dict[str, str | None] = {"state": "unloaded", "detail": None}
_load_lock = asyncio.Lock()


async def ensure_loaded() -> None:
    """Load the weights once. Idempotent, and raises if the load failed.

    The lock is held across the whole load, not just the state check, so a burst
    of requests waits on a single attempt instead of each starting its own.
    """
    async with _load_lock:
        if backend.is_ready():
            _load_state.update(state="ready", detail=None)
            return
        _load_state.update(state="loading", detail=None)
        try:
            await _detached(backend.load)
        except Exception as exc:
            # A gated-repo 403 is user-actionable, not a crash; keep the log to
            # one line and put the traceback behind DEBUG.
            log.error("model load failed: %s", short_error(exc))
            log.debug("load traceback", exc_info=True)
            _load_state.update(state="error", detail=short_error(exc))
            raise
        _load_state.update(state="ready", detail=None)
        log.info("model ready")


async def _detached(fn):
    """Run `fn` on a daemon thread and await its result.

    Not `asyncio.to_thread`: that pool's threads are joined at interpreter
    exit, so a Ctrl-C four minutes into a 32 GB download waited the download
    out before the process could end. A daemon thread dies with the process,
    and the Hub's downloader resumes the partial file on the next start.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def settle(setter, value) -> None:
        if not future.done():
            setter(value)

    def run() -> None:
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the loop
            outcome = (future.set_exception, exc)
        else:
            outcome = (future.set_result, result)
        try:
            loop.call_soon_threadsafe(settle, *outcome)
        except RuntimeError:  # the loop is gone: the process is exiting
            pass

    threading.Thread(target=run, daemon=True, name="load").start()
    return await future


async def _load_in_background() -> None:
    try:
        await ensure_loaded()
    except Exception:  # noqa: BLE001 - already recorded in _load_state
        pass


jobs = JobManager(backend, ensure_loaded)


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    if settings.load_on_startup:
        asyncio.create_task(_load_in_background())
    yield
    # Stop the worker before dropping the weights it is using. unload() takes no
    # part in the job gate, so on its own it would leave a generation running
    # against a pipeline nobody can see any more. try_unload, off the loop: a
    # load in flight holds the backend's lock, and blocking behind it to find
    # there was nothing to unload froze shutdown for the length of the download.
    await jobs.shutdown()
    await asyncio.to_thread(backend.try_unload)
    await asyncio.to_thread(prompt_translator.unload)


app = FastAPI(title="Krea Local", version="0.1.0", lifespan=lifespan)

# DNS rebinding: a hostile page served from evil.example, whose record is
# re-answered as 127.0.0.1, talks to this API as same-origin - CORS never gets a
# say. Starlette compares the Host header with the port stripped, and the UI
# fetches only relative paths, so this costs nothing to have on.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

def _origin_allowed(origin: str) -> bool:
    """Is this Origin header one of our own pages?

    Compared by hostname and *effective* port rather than by string: a browser
    serialises an origin without a default port, so on `KREA_PORT=80` the page
    sends `http://127.0.0.1` and a string match against `http://127.0.0.1:80`
    refused every one of the app's own writes. `https` is allowed for the LAN
    case behind a TLS proxy.
    """
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(origin)
        port = parts.port
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or parts.hostname is None:
        return False
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return parts.hostname in settings.allowed_hosts and port == settings.port


@app.middleware("http")
async def block_cross_site_writes(request: Request, call_next):
    """Refuse a state-changing request that announces another origin.

    /api/model/unload and /api/images/{id}/favorite carry no body - `value` is a
    query parameter - which makes them CORS-simple: a cross-origin no-cors fetch,
    or for favorite a plain form POST, reaches the handler, and the reply being
    unreadable does not undo the write. Browsers attach Origin to every
    cross-site write; curl and the CLI attach none, so both keep working.
    """
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin is not None and not _origin_allowed(origin):
            return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
    return await call_next(request)


# StaticFiles checks the directory exists at mount time; storage owns creating
# it (config no longer does at import, so a --dry-run can promise "nothing").
storage.ensure_outputs_dir()
app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")
app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.web_dir / "index.html")


# ------------------------------------------------------------------- model


@app.get("/api/status", response_model=ModelStatus)
async def status() -> ModelStatus:
    # describe() reads CUDA memory, which can queue behind a cold load for a
    # good while; off the event loop it cannot stall the SSE stream with it.
    info = await asyncio.to_thread(backend.describe)
    state = _load_state["state"]
    if state == "unloaded" and backend.is_ready():
        state = "ready"
    defaults = _merged_defaults(info)
    return ModelStatus(
        state=state,
        defaults=defaults,
        hints=hints_for(
            info["model_id"],
            defaults["cfg_threshold"],
            defaults["negative_prompt_supported"],
            defaults["true_cfg_supported"],
            defaults["reference_supported"],
            defaults["lora_supported"],
        ),
        guide=guide_for(info["model_id"]),
        model_id=info["model_id"],
        quantization=info["quantization"],
        offload=info["offload"],
        detail=_load_state["detail"],
        device_name=info.get("device_name"),
        vram_total_mb=info.get("vram_total_mb"),
        vram_used_mb=info.get("vram_used_mb"),
        disk_free_mb=_disk_free_mb(),
        prompt_expansion_supported=backend.prompt_expansion_supported(),
    )


def _merged_defaults(info: dict) -> dict:
    """The checkpoint's profile, with what a loaded pipeline actually exposes
    replacing each guess.

    The profile answers from the id alone, which is all there is before a load;
    every capability it guesses at is settled by the pipeline's parameter list
    once there is one. `lora_supported` is the exception in both directions: no
    profile guesses it, and the backend can answer it cold, because what decides
    it is peft and the quantization mode rather than the checkpoint.
    """
    defaults = defaults_for(info["model_id"])
    for key in (
        "cfg_threshold",
        "mu_supported",
        "negative_prompt_supported",
        "true_cfg_supported",
        "reference_supported",
    ):
        if info.get(key) is not None:
            defaults[key] = info[key]
    defaults["lora_supported"] = bool(info.get("lora_supported"))
    return defaults


async def _refuse_unusable_inputs(request: GenerateRequest) -> None:
    """Refuse a reference or an adapter this server cannot run - at submit.

    Both are all-or-nothing inputs, unlike a negative prompt: a night queued
    with references against a pipeline that takes none, or with an adapter no
    file backs, would render every cell as a plain text-to-image and look like
    a success. `applied()` keeps the *record* honest about that; this is what
    stops it happening at cell 1 of 64.
    """
    if request.reference:
        info = await asyncio.to_thread(backend.describe)
        if not _merged_defaults(info)["reference_supported"]:
            raise HTTPException(
                status_code=422,
                detail=f"{info['model_id']} takes no reference images - its pipeline has no "
                "`image` parameter. FLUX.2 [klein] is the checkpoint here that conditions "
                "on them.",
            )
        try:
            await asyncio.to_thread(storage.reference_paths, request.reference)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if request.lora:
        reason = backend.lora_unavailable_reason()
        if reason is not None:
            raise HTTPException(status_code=422, detail=reason)
        try:
            loras.path_for(request.lora)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/loras")
async def list_loras() -> dict:
    """The adapters on disk, and whether this server could apply one.

    Listed rather than uploaded: an adapter is a 200 MB file the user already
    has, and a directory is the thing they can drop it into without a form.
    """
    return {
        "loras": await asyncio.to_thread(loras.available),
        "supported": backend.lora_supported(),
        "detail": backend.lora_unavailable_reason(),
        "directory": str(settings.loras_dir),
    }


def _disk_free_mb() -> int | None:
    """Free space where the images land. One syscall; None if the directory is gone."""
    import shutil

    try:
        return shutil.disk_usage(settings.outputs_dir).free // (1024 * 1024)
    except OSError:
        return None


@app.post("/api/model/load", response_model=ModelStatus)
async def load_model() -> ModelStatus:
    if _load_state["state"] != "ready":
        asyncio.create_task(_load_in_background())
    return await status()


@app.post("/api/model/unload", response_model=ModelStatus)
async def unload_model() -> ModelStatus:
    # Unloading under a running job frees nothing - the worker holds its own
    # reference to the pipeline for the length of the call - while the status
    # dot says "not loaded" with 13GB still resident.
    if jobs.busy():
        raise HTTPException(status_code=409, detail="a job is still running; cancel it first")
    # Refuse rather than block: unload() would wait out a load in progress and
    # then discard what it built, with the request unanswered for minutes.
    if _load_lock.locked() or not await asyncio.to_thread(backend.try_unload):
        raise HTTPException(status_code=409, detail="the weights are still loading; wait for the load")
    await asyncio.to_thread(prompt_translator.unload)
    _load_state.update(state="unloaded", detail=None)
    return await status()


@app.post("/api/prompt/expand", response_model=PromptExpansion)
async def expand_prompt(request: PromptExpansionRequest) -> PromptExpansion:
    """Expand once, visibly, then let ordinary generation use that exact text."""
    if not backend.prompt_expansion_supported():
        raise HTTPException(
            status_code=422,
            detail="this checkpoint has no prompt expander; ERNIE-Image is the one here that does",
        )
    if jobs.busy():
        raise HTTPException(
            status_code=409,
            detail="an image job is queued or running; wait for it to finish",
        )
    try:
        await ensure_loaded()
    except Exception as exc:
        log.error("model load for prompt expansion failed: %s", short_error(exc))
        log.debug("prompt expansion load traceback", exc_info=True)
        raise HTTPException(status_code=500, detail=short_error(exc)) from exc
    # The id is only a useful cold guess. A loaded pipeline's method and
    # signature are the capability, exactly as for the generation controls.
    if not backend.prompt_expansion_supported():
        raise HTTPException(status_code=422, detail="the loaded pipeline has no prompt expander")
    try:
        expanded = await jobs.run_when_idle(
            lambda: backend.expand_prompt(request.prompt, request.width, request.height)
        )
    except BackendBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.error("prompt expansion failed: %s", short_error(exc))
        log.debug("prompt expansion traceback", exc_info=True)
        raise HTTPException(status_code=500, detail=short_error(exc)) from exc
    expanded = expanded.strip()
    if not expanded:
        raise HTTPException(status_code=500, detail="the prompt expander returned no text")
    if len(expanded) > 4000:
        raise HTTPException(
            status_code=422,
            detail="the expanded prompt is over 4000 characters; shorten the source prompt and try again",
        )
    translated: str | None = None
    translation_detail: str | None = None
    try:
        translated = await asyncio.to_thread(prompt_translator.translate, expanded)
    except Exception as exc:  # noqa: BLE001 - expansion remains useful on its own
        translation_detail = short_error(exc)
        log.error("expanded-prompt translation failed: %s", translation_detail)
        log.debug("translation traceback", exc_info=True)
    return PromptExpansion(
        original_prompt=request.prompt,
        expanded_prompt=expanded,
        translated_prompt=translated,
        translation_detail=translation_detail,
    )


# -------------------------------------------------------------- generation


@app.post("/api/generate", response_model=JobAccepted)
async def generate(request: GenerateRequest) -> JobAccepted:
    try:
        request.check_budget()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await _refuse_unusable_inputs(request)
    # The job waits on ensure_loaded() itself, so a previously failed load is
    # retried here rather than short-circuiting - the user may have just been
    # granted repo access.
    job = await jobs.submit(request)
    return JobAccepted(job_id=job.id, seed=job.seed, run_id=job.run_id)


async def _expand(request: GridRequest) -> tuple[list[grid.Cell], dict[str, list], int, str]:
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
    started it.
    """
    await _refuse_unusable_inputs(request)
    try:
        request.check_budget()
        values = grid.axis_values(
            request,
            request.steps_axis,
            request.cfg_axis,
            request.mu_axis,
            request.prompt_axis,
            request.lora_scale_axis,
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
            seeds=request.seeds,
            seed_list=seed_list or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
    run_id = grid.run_id(cells, settings.model_id, settings.quantization)
    if not request.resume:
        return cells, values, 0, run_id
    # Compare what the pipeline would *run*, not what the panel says: the rows
    # were written from `backend.applied()`, so a negative prompt a Turbo cell
    # at CFG 0 never saw is None on both sides rather than on one. Off the
    # loop: up to three chunked scans, fired by the preview on every keystroke.
    done = await asyncio.to_thread(
        storage.completed_cells,
        grid.prompts_of(cells),
        request.width,
        request.height,
        settings.model_id,
        settings.quantization,
    )
    kept = [
        cell
        for cell in cells
        if grid.prune_done([grid.Cell(request=backend.applied(cell.request), seed=cell.seed)], done)
    ]
    return kept, values, len(cells) - len(kept), run_id


@app.post("/api/grid", response_model=JobAccepted)
async def run_grid(request: GridRequest) -> JobAccepted:
    """Queue a sweep over the sampling axes as one job.

    One job rather than one per cell: a grid is a night's work, and it has to be
    cancellable, reattachable and ahead of nothing in the queue as a unit.
    """
    cells, _, _, run_id = await _expand(request)
    if not cells:
        # Every cell is already in the gallery. Queueing an empty job would
        # report "done" having rendered nothing, which reads like a failure.
        raise HTTPException(status_code=409, detail="every cell is already rendered")
    job = await jobs.submit_cells(cells, run_id=run_id, budget_minutes=request.budget_minutes)
    return JobAccepted(job_id=job.id, seed=job.seed, run_id=job.run_id)


@app.post("/api/rerender", response_model=JobAccepted)
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
    # One past the ceiling, so an oversize selection is refused by count rather
    # than silently truncated to the newest N.
    records, total = await asyncio.to_thread(
        storage.list_images,
        limit=settings.max_grid_images + 1,
        search=request.search,
        favorites_only=request.favorites,
    )
    if not records:
        raise HTTPException(status_code=404, detail="nothing in the gallery matches that selection")
    if total > settings.max_grid_images:
        raise HTTPException(
            status_code=422,
            detail=f"{total} images match; a job may queue {settings.max_grid_images}. Narrow the search.",
        )
    # Steps, CFG and mu are checkpoint-specific - Turbo's 8 / 0 / 1.15 on FLUX
    # is a broken image recorded under FLUX - so a record from another
    # checkpoint is refused rather than rendered wrong. `model:` in the search
    # narrows the selection to one.
    foreign = sorted({r.model_id for r in records if r.model_id != settings.model_id})
    if foreign:
        raise HTTPException(
            status_code=422,
            detail=f"{sum(r.model_id != settings.model_id for r in records)} of the selected images "
            f"were made on {', '.join(foreign)}; the server is running {settings.model_id}, whose "
            f"steps, CFG and mu differ. Add model:{settings.model_id.split('/')[-1]} to the search.",
        )
    # Oldest first, so the re-render lands in the order the originals did.
    cells = [
        grid.Cell(
            request=GenerateRequest(
                prompt=r.prompt,
                negative_prompt=r.negative_prompt,
                width=request.width,
                height=request.height,
                steps=r.steps,
                guidance_scale=r.guidance_scale,
                # The effective value, so a FLUX record from before the knob
                # was separate re-renders with the real CFG it actually ran.
                true_cfg=r.true_cfg,
                mu=r.mu,
                # The rest of the reproduction set: an adapter or a reference
                # that made the original is as much a part of it as the seed,
                # and the selection is already pinned to this checkpoint.
                reference=r.reference or [],
                lora=r.lora,
                lora_scale=r.lora_scale if r.lora_scale is not None else 1.0,
                seed=r.seed,
            ),
            seed=r.seed,
        )
        for r in reversed(records)
    ]
    # One check per distinct pairing rather than per cell: a selection of 400
    # images carries at most a handful of adapters between them.
    for spec in {(c.request.lora, tuple(c.request.reference)) for c in cells}:
        if spec != (None, ()):
            await _refuse_unusable_inputs(cells[0].request.model_copy(
                update={"lora": spec[0], "reference": list(spec[1])}
            ))
    run_id = grid.run_id(cells, settings.model_id, settings.quantization)
    done = await asyncio.to_thread(
        storage.completed_cells,
        grid.prompts_of(cells),
        request.width,
        request.height,
        settings.model_id,
        settings.quantization,
    )
    kept = [
        cell
        for cell in cells
        if grid.prune_done([grid.Cell(request=backend.applied(cell.request), seed=cell.seed)], done)
    ]
    if not kept:
        raise HTTPException(
            status_code=409,
            detail=f"every selected image is already rendered at {request.width}x{request.height}",
        )
    job = await jobs.submit_cells(kept, run_id=run_id)
    return JobAccepted(job_id=job.id, seed=job.seed, run_id=job.run_id)


@app.post("/api/grid/preview", response_model=GridPreview)
async def preview_grid(request: GridRequest) -> GridPreview:
    """What that grid would do, without queueing it.

    Same expansion as the real thing, so the count and the error messages the
    panel shows are the ones the run would have produced.
    """
    cells, values, skipped, _ = await _expand(request)
    # Cost is per step, and a grid's cells rarely agree on how many they take.
    per_step_ms = await asyncio.to_thread(
        storage.step_cost_ms, request.width, request.height, settings.model_id
    )
    per_image_bytes = await asyncio.to_thread(
        storage.image_bytes, request.width, request.height, settings.model_id
    )
    images = len(cells) * request.num_images
    listed = len(values.get("seed", []))
    return GridPreview(
        values=values,
        cells=len(cells),
        images=images,
        skipped=skipped,
        # As the pipeline would run them: the capability flags the status
        # endpoint publishes, applied cell by cell, say which cells collapse
        # into one picture before the night is spent finding out.
        duplicates=grid.duplicate_count(cells, backend.applied),
        summary=grid.describe(values, listed or request.seeds, listed=bool(listed)),
        seconds=grid.estimate_seconds(cells, per_step_ms),
        bytes=per_image_bytes * images if per_image_bytes else None,
    )


@app.get("/api/jobs")
async def active_jobs(include_finished: bool = False) -> list[dict]:
    """Jobs that have not finished yet - and, on request, the ones that have.

    A job id is otherwise reachable only from the response that minted it, so a
    reloaded page had no way back to the sweep it left running - and no way to
    tell that the Generate it just fired is sitting behind one. The finished
    ones are what the morning after reads: a 480-cell grid that stopped at
    cell 300 used to leave an idle Generate button and nothing anywhere saying
    when or why. They live in memory, so the answer lasts as long as the
    process does.
    """
    # Without the images: a long sweep accumulates hundreds of records, and a
    # caller looking for something to reattach to needs the id and the counters.
    listed = jobs.active() + (jobs.finished() if include_finished else [])
    return [job.snapshot(images=False) for job in listed]


@app.get("/api/runs", response_model=list[RunSummary])
async def recent_runs(limit: int = Query(20, ge=1, le=200)) -> list[RunSummary]:
    """The gallery grouped by run, newest first - one line per night's grid."""
    return await asyncio.to_thread(storage.runs, limit)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.snapshot()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """Stop a job. Already-finished jobs answer with their final state, not 404."""
    job = jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.snapshot()


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str, value: bool = True) -> dict:
    """Hold a running job at its next cell boundary (`value=false` resumes).

    Between cells the card holds little but allocator reserve, which the
    worker hands back once it has stopped - so the GPU is usable for something
    else mid-night, and the weights stay loaded for a seconds-long resume.
    `paused` in the snapshot is the worker's acknowledgement; until it is set
    a denoise may still be running.
    """
    job = jobs.pause(job_id, value)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.snapshot(images=False)


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str) -> StreamingResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")

    async def event_source():
        async for event in jobs.events(job):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ----------------------------------------------------------------- gallery


@app.get("/api/gallery", response_model=GalleryPage)
async def gallery(
    # A sheet asks for a whole run at once - up to a full grid - where the grid
    # view pages. The rows are small; the thumbnails load lazily either way.
    # The ceiling follows KREA_MAX_GRID_IMAGES, which /api/status publishes, so
    # the sheet's one-page request cannot be refused by the server that sized it.
    limit: int = Query(48, ge=1, le=max(1024, settings.max_grid_images)),
    offset: int = Query(0, ge=0),
    # Keyset paging: rows older than this id. What the browser sends, because
    # an offset over a gallery a sweep is writing into repeats the tail of one
    # page at the head of the next and skips as many older rows for good.
    before: int | None = Query(None, ge=1),
    search: str | None = None,
    favorites: bool = False,
) -> GalleryPage:
    # Every storage call runs off the loop, like describe() and step_cost_ms
    # above: SQLite's busy handler waits up to 15 s, and a write that lands
    # while `clear.py` holds the file would otherwise freeze the SSE stream,
    # the cancel endpoint and the status poll for that long.
    try:
        # One row past the page, so `has_more` is a fact rather than a guess
        # from a total that keeps growing while the page is read.
        items, total = await asyncio.to_thread(
            storage.list_images,
            limit=limit + 1,
            offset=offset,
            search=search,
            favorites_only=favorites,
            before=before,
        )
    except ValueError as exc:  # a filter term that is not a number
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GalleryPage(
        items=items[:limit], total=total, limit=limit, offset=offset, has_more=len(items) > limit
    )


@app.get("/api/images/{image_id}/thumb")
async def image_thumb(image_id: int):
    """The thumbnail for one image, by id.

    The panel's reference chips hold ids and nothing else - a reference
    restored from a record or from localStorage may name an image no loaded
    gallery page contains - so they need one request that turns an id into a
    picture. A redirect rather than a read: the file is already served by the
    `/outputs` mount, with its caching.
    """
    record = await asyncio.to_thread(storage.get_image, image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown image")
    return RedirectResponse(record.thumb_url or record.url)


@app.post("/api/images/{image_id}/favorite")
async def favorite(image_id: int, value: bool = True):
    record = await asyncio.to_thread(storage.set_favorite, image_id, value)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown image")
    return record


@app.delete("/api/images/{image_id}")
async def delete(image_id: int) -> dict:
    if not await asyncio.to_thread(storage.delete_image, image_id):
        raise HTTPException(status_code=404, detail="unknown image")
    return {"deleted": image_id}


def run() -> None:
    import uvicorn

    class Server(uvicorn.Server):
        """uvicorn's server, with Ctrl-C reaching the running sweep.

        uvicorn drains open connections *before* it runs the lifespan shutdown,
        and the browser's EventSource stays open until the job emits `done` -
        so `jobs.shutdown()`, the thing that cancels the job, sat behind the
        very stream that was waiting for it, and Ctrl-C printed "Waiting for
        connections to close" for the rest of the night. Cancelling on the
        signal itself lands within a step, `done` goes out, the stream closes
        and the ordinary drain completes. A second Ctrl-C (uvicorn's forced
        exit) skips the lifespan entirely; this still runs on it.
        """

        def handle_exit(self, sig, frame) -> None:
            for job in jobs.active():
                job.cancel_requested = True
            super().handle_exit(sig, frame)

    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        # A client that stops reading its stream must not hold shutdown open.
        timeout_graceful_shutdown=30,
    )
    Server(config).run()


if __name__ == "__main__":
    run()
