"""FastAPI app: text-to-image generation plus a browsable gallery."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Run as `python -m backend.main`, settle the checkpoint before `.config`
# freezes it: the terminal menu when there are no arguments, the
# --model/--quant/--offload flags otherwise. Imported by uvicorn or the tests
# instead, this does nothing and the environment stands as it is.
if __name__ == "__main__":
    from .launch import configure

    configure()

from . import grid, storage  # noqa: E402
from .backends import get_backend  # noqa: E402
from .config import settings  # noqa: E402
from .jobs import JobManager, short_error  # noqa: E402
from .model_defaults import defaults_for, hints_for  # noqa: E402
from .schemas import (  # noqa: E402
    GalleryPage,
    GenerateRequest,
    GridPreview,
    GridRequest,
    JobAccepted,
    ModelStatus,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
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
            await asyncio.to_thread(backend.load)
        except Exception as exc:
            # A gated-repo 403 is user-actionable, not a crash; keep the log to
            # one line and put the traceback behind DEBUG.
            log.error("model load failed: %s", short_error(exc))
            log.debug("load traceback", exc_info=True)
            _load_state.update(state="error", detail=short_error(exc))
            raise
        _load_state.update(state="ready", detail=None)
        log.info("model ready")


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
    # against a pipeline nobody can see any more.
    await jobs.shutdown()
    backend.unload()


app = FastAPI(title="Krea Local", version="0.1.0", lifespan=lifespan)

# DNS rebinding: a hostile page served from evil.example, whose record is
# re-answered as 127.0.0.1, talks to this API as same-origin - CORS never gets a
# say. Starlette compares the Host header with the port stripped, and the UI
# fetches only relative paths, so this costs nothing to have on.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

_ALLOWED_ORIGINS = {f"http://{host}:{settings.port}" for host in settings.allowed_hosts}


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
        if origin is not None and origin not in _ALLOWED_ORIGINS:
            return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
    return await call_next(request)


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
    defaults = defaults_for(info["model_id"])
    # The profile's CFG threshold is a guess from the checkpoint id; once a
    # pipeline is loaded its parameter list is the real answer.
    if info.get("cfg_threshold") is not None:
        defaults["cfg_threshold"] = info["cfg_threshold"]
    if info.get("negative_prompt_supported") is not None:
        defaults["negative_prompt_supported"] = info["negative_prompt_supported"]
    return ModelStatus(
        state=state,
        defaults=defaults,
        hints=hints_for(info["model_id"], defaults["cfg_threshold"]),
        model_id=info["model_id"],
        quantization=info["quantization"],
        offload=info["offload"],
        detail=_load_state["detail"],
        device_name=info.get("device_name"),
        vram_total_mb=info.get("vram_total_mb"),
        vram_used_mb=info.get("vram_used_mb"),
    )


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
    await asyncio.to_thread(backend.unload)
    _load_state.update(state="unloaded", detail=None)
    return await status()


# -------------------------------------------------------------- generation


@app.post("/api/generate", response_model=JobAccepted)
async def generate(request: GenerateRequest) -> JobAccepted:
    try:
        request.check_budget()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # The job waits on ensure_loaded() itself, so a previously failed load is
    # retried here rather than short-circuiting - the user may have just been
    # granted repo access.
    job = await jobs.submit(request)
    return JobAccepted(job_id=job.id)


def _detail(exc: ValueError) -> str:
    """One line out of an axis parser or out of pydantic, for the 422 body.

    A cell that violates the schema - `--grid-steps 90-120` against the 100 cap -
    raises a ValidationError, whose own str() is several lines of type and url
    noise. The UI puts this straight under the field.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'request'}: {error['msg']}"
        for error in errors()
    )


def _expand(request: GridRequest) -> tuple[list[grid.Cell], dict[str, list], int]:
    """The grid as cells, its resolved axes, and the number resume dropped.

    The axes come back from here rather than being resolved again by the
    preview, so a prompt whose placeholders do not line up with its variables
    fails once, as a 422, instead of raising out of the response model.
    """
    try:
        request.check_budget()
        values = grid.axis_values(
            request,
            request.steps_axis,
            request.cfg_axis,
            request.mu_axis,
            request.prompt_axis,
        )
        cells = grid.expand(
            request,
            prompt=request.prompt_axis,
            steps=request.steps_axis,
            cfg=request.cfg_axis,
            mu=request.mu_axis,
            seeds=request.seeds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_detail(exc)) from exc
    if not request.resume:
        return cells, values, 0
    done = storage.completed_cells(
        grid.prompts_of(cells),
        request.width,
        request.height,
        settings.model_id,
    )
    kept = grid.prune_done(cells, done)
    return kept, values, len(cells) - len(kept)


@app.post("/api/grid", response_model=JobAccepted)
async def run_grid(request: GridRequest) -> JobAccepted:
    """Queue a sweep over the sampling axes as one job.

    One job rather than one per cell: a grid is a night's work, and it has to be
    cancellable, reattachable and ahead of nothing in the queue as a unit.
    """
    cells, _, _ = _expand(request)
    if not cells:
        # Every cell is already in the gallery. Queueing an empty job would
        # report "done" having rendered nothing, which reads like a failure.
        raise HTTPException(status_code=409, detail="every cell is already rendered")
    job = await jobs.submit_cells(cells)
    return JobAccepted(job_id=job.id)


@app.post("/api/grid/preview", response_model=GridPreview)
async def preview_grid(request: GridRequest) -> GridPreview:
    """What that grid would do, without queueing it.

    Same expansion as the real thing, so the count and the error messages the
    panel shows are the ones the run would have produced.
    """
    cells, values, skipped = _expand(request)
    # Cost is per step, and a grid's cells rarely agree on how many they take.
    per_step_ms = await asyncio.to_thread(storage.step_cost_ms, request.width, request.height)
    total_steps = sum(cell.request.steps for cell in cells) * request.num_images
    return GridPreview(
        values=values,
        cells=len(cells),
        images=len(cells) * request.num_images,
        skipped=skipped,
        summary=grid.describe(values, request.seeds),
        seconds=None if per_step_ms is None else round(per_step_ms * total_steps / 1000, 1),
    )


@app.get("/api/jobs")
async def active_jobs() -> list[dict]:
    """Jobs that have not finished yet.

    A job id is otherwise reachable only from the response that minted it, so a
    reloaded page had no way back to the sweep it left running - and no way to
    tell that the Generate it just fired is sitting behind one.
    """
    # Without the images: a long sweep accumulates hundreds of records, and a
    # caller looking for something to reattach to needs the id and the counters.
    return [
        {key: value for key, value in job.snapshot().items() if key != "images"}
        for job in jobs.active()
    ]


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
    limit: int = Query(48, ge=1, le=1024),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    favorites: bool = False,
) -> GalleryPage:
    try:
        items, total = storage.list_images(
            limit=limit, offset=offset, search=search, favorites_only=favorites
        )
    except ValueError as exc:  # a filter term that is not a number
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GalleryPage(items=items, total=total, limit=limit, offset=offset)


@app.post("/api/images/{image_id}/favorite")
async def favorite(image_id: int, value: bool = True):
    record = storage.set_favorite(image_id, value)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown image")
    return record


@app.delete("/api/images/{image_id}")
async def delete(image_id: int) -> dict:
    if not storage.delete_image(image_id):
        raise HTTPException(status_code=404, detail="unknown image")
    return {"deleted": image_id}


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()
