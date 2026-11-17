"""On-demand prompt writing through the critic's transient local runtime.

The image model is never loaded. Downloads and authoritative tag lookups run
outside the image gate; the keyed child process owns it until fully stopped.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Path

from . import assistant_models, critic_runtime, keep_awake, prompt_assistant
from .api_prompt import critic_unsupported, prompt_failure, refuse_critique_now
from .app_context import AppServices
from .backends.base import Cancelled
from .detached import detached
from .jobs import BackendBusy
from .schemas_assistant import PromptAssistantRequest, PromptAssistantResponse


@dataclass
class _AssistantRun:
    stopped: threading.Event = field(default_factory=threading.Event)
    phase: str = "preparing"
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        if self.stopped.is_set():
            raise Cancelled("prompt assistant stopped")

    def progress(self, phase: str, done: int | None = None, total: int | None = None) -> None:
        with self._lock:
            if self.phase in ("complete", "stopped", "failed"):
                return
            if self.stopped.is_set() and phase != "stopped":
                return
            self.phase = phase
            self.downloaded_bytes = max(0, done) if done is not None else None
            self.total_bytes = max(0, total) if total is not None else None

    def status(self, running: bool) -> dict:
        with self._lock:
            return {
                "phase": self.phase, "running": running,
                "downloaded_bytes": self.downloaded_bytes, "total_bytes": self.total_bytes,
            }


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter()
    running: dict[str, tuple[_AssistantRun, asyncio.Task]] = {}
    stopped: dict[str, float] = {}
    finished: dict[str, tuple[_AssistantRun, float]] = {}

    @router.get("/api/prompt/assist/models")
    async def list_assistant_models() -> dict:
        return {"models": services.critic().assistant_models()}

    def prune_stops() -> None:
        cutoff = time.monotonic() - 300
        for ticket, when in list(stopped.items()):
            if when < cutoff:
                del stopped[ticket]
        while len(stopped) > 256:
            del stopped[next(iter(stopped))]
        for ticket, (_, when) in list(finished.items()):
            if when < cutoff:
                del finished[ticket]
        while len(finished) > 256:
            del finished[next(iter(finished))]
        # A stopped run may have both a stop-before-POST tombstone and a final
        # status. Bound their union, rather than keeping 256 of each forever.
        while len(stopped.keys() | finished.keys()) > 256:
            times = {ticket: when for ticket, (_, when) in finished.items()}
            times.update(stopped)
            oldest = min(times, key=times.__getitem__)
            stopped.pop(oldest, None)
            finished.pop(oldest, None)

    @router.get("/api/prompt/assist/{ticket}/status")
    async def assistant_status(ticket: str = Path(min_length=1, max_length=64)) -> dict:
        prune_stops()
        current = running.get(ticket)
        if current is not None:
            return current[0].status(True)
        previous = finished.get(ticket)
        if previous is not None:
            return previous[0].status(False)
        if ticket in stopped:
            return {"phase": "stopped", "running": False, "downloaded_bytes": None, "total_bytes": None}
        raise HTTPException(status_code=404, detail="this assistant ticket has no recent status")

    @router.post("/api/prompt/assist/{ticket}/stop")
    async def stop_assistant(ticket: str = Path(min_length=1, max_length=64)) -> dict:
        # A Stop can arrive before the matching POST. A new ticket is needed
        # for every ask, so remembering it never stops a later conversation.
        prune_stops()
        stopped[ticket] = time.monotonic()
        prune_stops()
        current = running.get(ticket)
        if current is not None:
            current[0].stopped.set()
            current[0].progress("stopped")
            # Cancelling a detached prepare forgets only its waiter; the
            # daemon download may finish, but cannot proceed to inference.
            # An admitted run_when_idle retains its gate until assist unwinds.
            current[1].cancel()
        return {"stopped": True, "running": current is not None}

    async def run_assistant(
        request: PromptAssistantRequest, run: _AssistantRun, model_id: str,
    ) -> PromptAssistantResponse:
        run.check()
        critic = services.critic()
        if not critic.supported():
            raise HTTPException(status_code=422, detail=critic_unsupported(services.settings()))
        refuse_critique_now(services)
        profile = (None if request.assistant_model == "default"
                   else assistant_models.get_model(request.assistant_model))
        try:
            prompt_assistant.check_fits(
                prompt_assistant.build_context(model_id, request), estimated=True, profile=profile,
            )
        except Exception as exc:
            raise prompt_failure(exc, "prompt assistant question") from exc

        def prepare():
            run.check()
            # A dropped request never lets a still-running download sleep the
            # machine. The daemon thread is never joined at interpreter exit.
            with keep_awake.busy("preparing the local prompt assistant"), critic_runtime.progress(run.progress):
                return critic.prepare_assistant(request.assistant_model)

        try:
            prepared = await detached(prepare, "assistant-prepare")
            run.check()
        except Cancelled:
            raise
        except Exception as exc:
            run.check()
            raise prompt_failure(exc, "preparing the local prompt assistant") from exc
        # Downloads can take minutes, during which an image load or queue may
        # begin. Never place the LLM beside weights arriving outside the gate.
        refuse_critique_now(services)

        def infer() -> PromptAssistantResponse:
            with critic_runtime.progress(run.progress):
                return critic.assist(model_id, request, run.stopped.is_set, prepared)

        try:
            run.progress("waiting")
            result = await services.jobs().run_when_idle(
                infer,
                holder="critic", ticket=request.ticket,
            )
        except BackendBusy as exc:
            run.check()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Cancelled:
            raise
        except Exception as exc:
            run.check()
            raise prompt_failure(exc, "local prompt assistant") from exc
        run.check()
        # Verify suggestions with the same vocabulary used by the chips. This
        # is online CPU work, after llama-server has released the accelerator.
        def verify_tags() -> PromptAssistantResponse:
            # A stopped HTTP waiter can leave this daemon lookup finishing;
            # its keep-awake ownership lasts until the worker itself exits.
            with keep_awake.busy("verifying prompt assistant tag candidates"):
                if result.tags:
                    run.progress("verifying")
                return prompt_assistant.validate_tags(model_id, result)

        try:
            result = await detached(verify_tags, "assistant-tags")
        except Exception as exc:
            run.check()
            raise prompt_failure(exc, "prompt assistant tag verification") from exc
        run.check()
        return result

    @router.post("/api/prompt/assist", response_model=PromptAssistantResponse)
    async def assist_prompt(request: PromptAssistantRequest) -> PromptAssistantResponse:
        prune_stops()
        if request.ticket in stopped:
            raise HTTPException(status_code=409, detail="prompt assistant stopped")
        if request.ticket in running:
            raise HTTPException(status_code=409, detail="this assistant ticket is already running")
        if request.ticket in finished:
            raise HTTPException(status_code=409, detail="this assistant ticket has already been used; send with a new ticket")
        run = _AssistantRun()
        # Capture the selected id before any await; the model-admission layer
        # holds that checkpoint throughout validation and this owned worker.
        worker = asyncio.create_task(run_assistant(request, run, services.model_id()))
        running[request.ticket] = (run, worker)
        try:
            try:
                result = await asyncio.shield(worker)
                run.progress("complete")
                return result
            except asyncio.CancelledError:
                requested_stop = run.stopped.is_set()
                run.stopped.set()
                run.progress("stopped")
                worker.cancel()
                # The worker keeps both the model read and the GPU reservation
                # until its own process is gone, including repeated disconnects.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if worker.done() and not worker.cancelled():
                    worker.exception()
                if requested_stop:
                    raise Cancelled("prompt assistant stopped") from None
                raise
        except Cancelled as exc:
            run.progress("stopped")
            raise HTTPException(status_code=409, detail="prompt assistant stopped") from exc
        except Exception:
            run.progress("failed")
            raise
        finally:
            if running.get(request.ticket) == (run, worker):
                del running[request.ticket]
                finished[request.ticket] = (run, time.monotonic())
                prune_stops()

    return router
