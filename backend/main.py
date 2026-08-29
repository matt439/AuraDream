"""FastAPI app: text-to-image generation plus a browsable gallery."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import storage
from .backends import get_backend
from .config import settings
from .jobs import JobManager, short_error
from .model_defaults import defaults_for
from .schemas import GalleryPage, GenerateRequest, JobAccepted, ModelStatus

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
    backend.unload()


app = FastAPI(title="Krea Local", version="0.1.0", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")
app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.web_dir / "index.html")


# ------------------------------------------------------------------- model


@app.get("/api/status", response_model=ModelStatus)
async def status() -> ModelStatus:
    info = backend.describe()
    state = _load_state["state"]
    if state == "unloaded" and backend.is_ready():
        state = "ready"
    return ModelStatus(
        state=state,
        defaults=defaults_for(info["model_id"]),
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


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
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
    limit: int = Query(48, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: str | None = None,
    favorites: bool = False,
) -> GalleryPage:
    items, total = storage.list_images(
        limit=limit, offset=offset, search=search, favorites_only=favorites
    )
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
