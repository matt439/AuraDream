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

from . import keep_awake, prompt_assistant
from .api_prompt import critic_unsupported, prompt_failure, refuse_critique_now
from .app_context import AppServices
from .backends.base import Cancelled
from .detached import detached
from .jobs import BackendBusy
from .schemas_assistant import PromptAssistantRequest, PromptAssistantResponse


@dataclass
class _AssistantRun:
    stopped: threading.Event = field(default_factory=threading.Event)

    def check(self) -> None:
        if self.stopped.is_set():
            raise Cancelled("prompt assistant stopped")


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter()
    running: dict[str, tuple[_AssistantRun, asyncio.Task]] = {}
    stopped: dict[str, float] = {}

    def prune_stops() -> None:
        cutoff = time.monotonic() - 300
        for ticket, when in list(stopped.items()):
            if when < cutoff:
                del stopped[ticket]
        while len(stopped) > 256:
            del stopped[next(iter(stopped))]

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
        try:
            prompt_assistant.check_fits(prompt_assistant.build_context(model_id, request), estimated=True)
        except Exception as exc:
            raise prompt_failure(exc, "prompt assistant question") from exc

        def prepare() -> None:
            run.check()
            # A dropped request never lets a still-running download sleep the
            # machine. The daemon thread is never joined at interpreter exit.
            with keep_awake.busy("preparing the local prompt assistant"):
                critic.prepare()

        try:
            await detached(prepare, "assistant-prepare")
            run.check()
        except Cancelled:
            raise
        except Exception as exc:
            run.check()
            raise prompt_failure(exc, "preparing the local prompt assistant") from exc
        # Downloads can take minutes, during which an image load or queue may
        # begin. Never place the LLM beside weights arriving outside the gate.
        refuse_critique_now(services)
        try:
            result = await services.jobs().run_when_idle(
                lambda: critic.assist(model_id, request, run.stopped.is_set),
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
        run = _AssistantRun()
        # Capture the selected id before any await; the model-admission layer
        # holds that checkpoint throughout validation and this owned worker.
        worker = asyncio.create_task(run_assistant(request, run, services.model_id()))
        running[request.ticket] = (run, worker)
        try:
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                requested_stop = run.stopped.is_set()
                run.stopped.set()
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
            raise HTTPException(status_code=409, detail="prompt assistant stopped") from exc
        finally:
            if running.get(request.ticket) == (run, worker):
                del running[request.ticket]

    return router
