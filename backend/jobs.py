"""Single-worker job queue.

The GPU can only usefully run one generation at a time, so every request is
funnelled through one worker thread. Progress hops back to the event loop via
`call_soon_threadsafe` and leaves as SSE.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import storage
from .backends import GenerationBackend
from .config import settings
from .schemas import GenerateRequest

log = logging.getLogger(__name__)

_MAX_SEED = 2**63 - 1


def short_error(exc: BaseException, limit: int = 320) -> str:
    """A one-paragraph, UI-presentable form of an exception.

    HuggingFace errors in particular carry a request id and a stack of URLs that
    are noise in the browser; keep the actionable sentences and drop the rest.
    """
    lines = [
        line.strip()
        for line in str(exc).splitlines()
        if line.strip() and "Request ID" not in line
    ]
    message = " ".join(lines) or exc.__class__.__name__
    if len(message) > limit:
        message = message[: limit - 1].rstrip() + "…"
    return f"{type(exc).__name__}: {message}"


@dataclass
class Job:
    id: str
    request: GenerateRequest
    status: str = "queued"  # queued | running | done | error
    step: int = 0
    total_steps: int = 0
    images: list[dict] = field(default_factory=list)
    error: str | None = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "step": self.step,
            "total_steps": self.total_steps,
            "images": self.images,
            "error": self.error,
        }


class JobManager:
    def __init__(
        self,
        backend: GenerationBackend,
        ensure_loaded: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.backend = backend
        self._ensure_loaded = ensure_loaded
        self._jobs: dict[str, Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")
        self._gate = asyncio.Lock()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def submit(self, request: GenerateRequest) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], request=request, total_steps=request.steps)
        self._jobs[job.id] = job
        # Keep the dict from growing without bound over a long session.
        if len(self._jobs) > 200:
            for stale in list(self._jobs)[:-100]:
                if self._jobs[stale].status in {"done", "error"}:
                    self._jobs.pop(stale, None)
        asyncio.create_task(self._run(job))
        return job

    def _emit(self, loop: asyncio.AbstractEventLoop, job: Job, event: dict) -> None:
        loop.call_soon_threadsafe(job.queue.put_nowait, event)

    async def _run(self, job: Job) -> None:
        loop = asyncio.get_running_loop()
        # Wait for the shared load rather than letting the worker thread start a
        # second one. Failing here keeps the error to a single report.
        if self._ensure_loaded is not None:
            try:
                await self._ensure_loaded()
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                job.status = "error"
                job.error = short_error(exc)
                self._emit(loop, job, {"type": "done", **job.snapshot()})
                return
        # One generation at a time; queued jobs simply wait their turn here.
        async with self._gate:
            job.status = "running"
            self._emit(loop, job, {"type": "status", **job.snapshot()})
            try:
                records = await loop.run_in_executor(self._pool, self._generate, job, loop)
                job.images = [r.model_dump() for r in records]
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                log.error("job %s failed: %s", job.id, short_error(exc))
                log.debug("job traceback", exc_info=True)
                job.status = "error"
                job.error = short_error(exc)
            self._emit(loop, job, {"type": "done", **job.snapshot()})

    def _generate(self, job: Job, loop: asyncio.AbstractEventLoop) -> list:
        req = job.request
        seed = req.seed if req.seed is not None else random.randint(0, _MAX_SEED)

        def on_progress(step: int, total: int) -> None:
            job.step, job.total_steps = step, total
            self._emit(loop, job, {"type": "progress", "step": step, "total_steps": total})

        started = time.perf_counter()
        images = self.backend.generate(req, seed=seed, on_progress=on_progress)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        records = []
        for offset, image in enumerate(images):
            # Each image in a batch gets its own reproducible seed.
            records.append(
                storage.save_image(
                    image,
                    req,
                    seed=seed + offset,
                    duration_ms=elapsed_ms // max(len(images), 1),
                    model_id=settings.model_id,
                )
            )
        return records

    async def events(self, job: Job):
        """Yield SSE payloads until the job finishes."""
        yield {"type": "status", **job.snapshot()}
        if job.status in {"done", "error"}:
            return
        while True:
            event = await job.queue.get()
            yield event
            if event.get("type") == "done":
                return
