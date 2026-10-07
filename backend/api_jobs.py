"""Job inspection, cancellation, pause and SSE routes.

The manager is supplied by the app at request time: these routes neither create
workers nor own their lifecycle.
"""
from __future__ import annotations

import json
from collections.abc import Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .jobs import JobManager


def create_router(get_jobs: Callable[[], JobManager]) -> APIRouter:
    router = APIRouter()

    @router.get("/api/jobs")
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
        listed = get_jobs().active() + (get_jobs().finished() if include_finished else [])
        return [job.snapshot(images=False) for job in listed]


    @router.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict:
        job = get_jobs().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.snapshot()


    @router.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict:
        """Stop a job. Already-finished jobs answer with their final state, not 404."""
        job = get_jobs().cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.snapshot()


    @router.post("/api/jobs/{job_id}/pause")
    async def pause_job(job_id: str, value: bool = True) -> dict:
        """Hold a running job at its next cell boundary (`value=false` resumes).

        Between cells the card holds little but allocator reserve, which the
        worker hands back once it has stopped - so the GPU is usable for something
        else mid-night, and the weights stay loaded for a seconds-long resume.
        `paused` in the snapshot is the worker's acknowledgement; until it is set
        a denoise may still be running.
        """
        job = get_jobs().pause(job_id, value)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.snapshot(images=False)


    @router.get("/api/jobs/{job_id}/stream")
    async def job_stream(job_id: str) -> StreamingResponse:
        job = get_jobs().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")

        async def event_source():
            async for event in get_jobs().events(job):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


    return router
