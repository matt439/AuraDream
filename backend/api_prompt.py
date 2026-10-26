"""The tag helper and prompt routes: tag suggestions and chip verdicts,
Expand, Check, the token measure and the critic.

Check, measure and the tag lookups stay cold - a daemon thread, never
`ensure_loaded`, never the gate. Expand loads the model before it waits
for the card, so nothing holding the gate awaits the load lock.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Path, Query

from . import prompt_check, prompt_encoding, prompt_parts, prompt_text, tag_lookup
from .app_context import AppServices
from .backends.base import Cancelled
from .critic import CriticError, CriticUnavailable
from .detached import detached
from .jobs import BackendBusy, LoadFailed, short_error
from .model_defaults import guide_for
from .schemas import (
    PromptAnalysis,
    PromptAnalysisRequest,
    PromptCritique,
    PromptExpansion,
    PromptExpansionRequest,
    PromptMeasure,
    PromptMeasureRequest,
    TagDescriptionResponse,
    TagSuggestionResponse,
    TagValidationRequest,
    TagValidationResponse,
)
from .tags import TagLookupUnavailable, grid_variable_result, rank_suggestions, unchecked_results

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger("local_diffusion")


@dataclass
class _Expansion:
    stopped: threading.Event = field(default_factory=threading.Event)
    abandoned: asyncio.Event = field(default_factory=asyncio.Event)

    def stop(self) -> None:
        self.stopped.set()
        self.abandoned.set()

    def check(self) -> None:
        if self.stopped.is_set():
            raise Cancelled("prompt expansion stopped")


def tag_helper_config(model_id: str) -> dict:
    helper = guide_for(model_id).get("tag_helper")
    if not helper or helper.get("vocabulary") != "danbooru":
        raise HTTPException(status_code=404, detail="this checkpoint has no tag vocabulary")
    return helper


def chip_verdicts(model_id: str, helper: dict, request: TagValidationRequest) -> TagValidationResponse:
    """The tag index's verdict on the chips and, given the prompt, their
    order slots. Blocking: the lookup reaches the network."""
    try:
        results = tag_lookup.lookup_tags(helper, request.tags)
    except TagLookupUnavailable as exc:
        special = tag_lookup.special_tags(helper)
        year = bool(helper.get("year_tag"))
        results = []
        for tag in request.tags:
            # The grid's `{name}` needs no index to be told apart.
            if prompt_text.is_grid_variable(tag):
                results.append(grid_variable_result(tag))
                continue
            name = tag_lookup.artist_tag_name(helper, tag)
            result = unchecked_results([name or tag], set() if name is not None else special,
                                       False if name is not None else year)[0]
            results.append({**result, "tag": tag})
        return TagValidationResponse(
            available=False,
            results=results,
            detail=short_error(exc),
            parts=prompt_parts.contextual_parts(helper, request.prompt, request.tags, results),
            **prompt_check.order_slots(model_id, request.prompt, reachable=False),
        )
    return TagValidationResponse(
        available=True, results=results,
        parts=prompt_parts.contextual_parts(helper, request.prompt, request.tags, results),
        **prompt_check.order_slots(model_id, request.prompt, reachable=True)
    )


def expand_detail(exc: Exception) -> str:
    """An Expand failure as the page shows it: `short_error`'s message without
    the class name in front - "RuntimeError: CUDA out of memory..." - which the
    log line keeps. The page says whose failure it is ("Expand did not
    finish: ...") as it does the critic's. A message of one word or none
    keeps its class, which is then most of what says what went wrong:
    "KeyError: 'pe'", not "'pe'".
    """
    detail = short_error(exc)
    bare = detail.removeprefix(f"{type(exc).__name__}: ")
    return bare if len(bare.split()) > 1 else detail


def prompt_failure(exc: Exception, doing: str) -> HTTPException:
    """A failure of the check or the critic as the reply the page shows,
    logged the way Expand's are: one line, the traceback behind DEBUG.

    503 when the critic cannot run as configured (`CriticUnavailable`), 502
    when it ran and failed - a download that did not verify, a server that did
    not start or an answer that did not parse (`CriticError`) - and 500 for
    anything else, the analyser's own failures included.

    The critic's own two classes reach the page without their names: their
    messages are written to be read there, the page says whose failure it is
    ("The critic did not answer: ..."), and "CriticError:" led every one of
    them. The log line keeps the class; anything else keeps it in the reply
    too, since there the class may be all that says what went wrong.
    """
    log.error("%s failed: %s", doing, short_error(exc))
    log.debug("%s traceback", doing, exc_info=exc)
    detail = short_error(exc)
    if isinstance(exc, CriticUnavailable):
        status = 503
    elif isinstance(exc, CriticError):
        status = 502
    else:
        status = 500
    if isinstance(exc, CriticError):
        detail = detail.removeprefix(f"{type(exc).__name__}: ")
    return HTTPException(status_code=status, detail=detail)


def critic_unsupported(settings: Settings) -> str:
    from .critic import pinned_build_refusal

    if not settings.critic:
        return "the prompt critic is switched off (LOCAL_DIFFUSION_CRITIC)"
    # A build published for the platform that this system cannot run (glibc
    # too old) says so, rather than that there is none.
    return pinned_build_refusal() or (
        "llama.cpp publishes no pinned llama-server build for this platform; set "
        "LOCAL_DIFFUSION_CRITIC_SERVER to a llama-server of your own"
    )


def refuse_critique_now(services: AppServices) -> None:
    """409 while the card is promised to the image model.

    A job queued or running, as for Expand; and a load in progress, which is
    moving the weights onto the card outside the job gate - a critic started
    beside it would find the memory it planned on gone.
    """
    if services.jobs().busy():
        raise HTTPException(
            status_code=409, detail="an image job is queued or running; wait for it to finish"
        )
    if services.loading():
        raise HTTPException(
            status_code=409, detail="the image model is loading; wait for the load"
        )


def create_router(services: AppServices) -> APIRouter:
    """The routes, reading the app's state through `services` per request."""
    router = APIRouter()
    expansions: dict[str, tuple[_Expansion, asyncio.Task]] = {}
    # A Stop can overtake its POST on the network. Remember a bounded set of
    # abandoned tickets so that late arrival cannot start another expansion.
    stopped_tickets: dict[str, float] = {}

    def prune_stops() -> None:
        cutoff = time.monotonic() - 300
        for ticket, when in list(stopped_tickets.items()):
            if when < cutoff:
                del stopped_tickets[ticket]
        while len(stopped_tickets) > 256:
            del stopped_tickets[next(iter(stopped_tickets))]

    @router.post("/api/prompt/expand/{ticket}/stop")
    async def stop_expansion(ticket: str = Path(min_length=1, max_length=64)) -> dict:
        prune_stops()
        stopped_tickets[ticket] = time.monotonic()
        prune_stops()
        current = expansions.get(ticket)
        if current is not None:
            current[0].stop()
        return {"stopped": True, "running": current is not None}

    @router.get("/api/tags/suggest", response_model=TagSuggestionResponse)
    async def suggest_tags(q: str = Query(min_length=1, max_length=100)) -> TagSuggestionResponse:
        helper = tag_helper_config(services.model_id())
        prefix = helper.get("artist_prefix")
        artist_only = bool(prefix and q.lower().startswith(prefix.lower()))
        query = q[len(prefix):].strip() if artist_only else q
        if not query:
            return TagSuggestionResponse(available=True)
        local = [] if artist_only else tag_lookup.model_tag_suggestions(helper, query)
        avoid = tag_lookup.negative_tags(helper)
        try:
            remote = await asyncio.to_thread(
                tag_lookup.tag_vocabulary.suggest, query, helper["vocabulary_cutoff"]
            )
        except TagLookupUnavailable as exc:
            return TagSuggestionResponse(
                available=False,
                suggestions=rank_suggestions(query, local, [], avoid),
                detail=short_error(exc),
            )
        if artist_only:
            remote = [item for item in remote if item["category"] == "artist"]
        suggestions = rank_suggestions(query, local, remote, avoid)
        if prefix:
            suggestions = [
                {**item, "text": prefix + item["text"]} if item["category"] == "artist" else item
                for item in suggestions
            ]
        return TagSuggestionResponse(available=True, suggestions=suggestions)


    @router.get("/api/tags/describe", response_model=TagDescriptionResponse)
    async def describe_tags(q: str = Query(min_length=1, max_length=100)) -> TagDescriptionResponse:
        """The Tags helper's finder: general tags whose Danbooru wiki mentions
        `q`, for a checkpoint whose guide names the Danbooru vocabulary."""
        helper = tag_helper_config(services.model_id())
        if not q.strip():
            return TagDescriptionResponse(available=True)
        try:
            results = await asyncio.to_thread(
                tag_lookup.tag_vocabulary.describe, q, helper["vocabulary_cutoff"]
            )
        except TagLookupUnavailable as exc:
            return TagDescriptionResponse(available=False, detail=short_error(exc))
        return TagDescriptionResponse(available=True, results=results)

    @router.post("/api/tags/validate", response_model=TagValidationResponse)
    async def validate_tags(request: TagValidationRequest) -> TagValidationResponse:
        model_id = services.model_id()
        helper = tag_helper_config(model_id)
        return await asyncio.to_thread(chip_verdicts, model_id, helper, request)


    async def run_expansion(request: PromptExpansionRequest, run: _Expansion) -> PromptExpansion:
        run.check()
        if not services.backend().prompt_expansion_supported():
            raise HTTPException(
                status_code=422,
                detail="this checkpoint has no prompt expander; ERNIE-Image is the one here that does",
            )
        if services.jobs().busy():
            raise HTTPException(
                status_code=409,
                detail="an image job is queued or running; wait for it to finish",
            )
        try:
            await services.ensure_loaded()
        except LoadFailed as exc:
            run.check()
            # Logged where it failed; the status's sentence, steps and all. 503,
            # the one answer of this route that means "the model could not load":
            # the page says so in a line of its own and leaves the sentence to
            # the viewer's card, which says it already.
            raise HTTPException(status_code=503, detail=exc.detail) from exc
        except Exception as exc:
            run.check()
            log.error("model load for prompt expansion failed: %s", short_error(exc))
            log.debug("prompt expansion load traceback", exc_info=True)
            raise HTTPException(status_code=500, detail=expand_detail(exc)) from exc
        run.check()
        # The id is only a useful cold guess. A loaded pipeline's method and
        # signature are the capability, exactly as for the generation controls.
        if not services.backend().prompt_expansion_supported():
            raise HTTPException(status_code=422, detail="the loaded pipeline has no prompt expander")
        def expand() -> str:
            run.check()
            return services.backend().expand_prompt(
                request.prompt, request.width, request.height, run.stopped.is_set,
            )

        try:
            expanded = await services.jobs().run_when_idle(
                expand,
                holder="expand",
                ticket=request.ticket,
            )
        except BackendBusy as exc:
            run.check()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Cancelled:
            raise
        except Exception as exc:
            run.check()
            log.error("prompt expansion failed: %s", short_error(exc))
            log.debug("prompt expansion traceback", exc_info=True)
            raise HTTPException(status_code=500, detail=expand_detail(exc)) from exc
        run.check()
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
            run.check()
            translated = await asyncio.to_thread(services.translator().translate, expanded, run.stopped.is_set)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - expansion remains useful on its own
            run.check()
            log.error("expanded-prompt translation failed: %s", short_error(exc))
            log.debug("translation traceback", exc_info=True)
            translation_detail = expand_detail(exc)
        run.check()
        return PromptExpansion(
            original_prompt=request.prompt,
            expanded_prompt=expanded,
            translated_prompt=translated,
            translation_detail=translation_detail,
        )

    @router.post("/api/prompt/expand", response_model=PromptExpansion)
    async def expand_prompt(request: PromptExpansionRequest) -> PromptExpansion:
        """A ticket owns the worker; disconnect/Stop abandon only its answer."""
        prune_stops()
        ticket = request.ticket or uuid.uuid4().hex
        if ticket in stopped_tickets:
            raise HTTPException(status_code=409, detail="prompt expansion stopped")
        if ticket in expansions:
            raise HTTPException(status_code=409, detail="this expansion ticket is already running")
        run = _Expansion()
        worker = asyncio.create_task(run_expansion(request, run))
        expansions[ticket] = (run, worker)

        def finished(task: asyncio.Task) -> None:
            if expansions.get(ticket) == (run, task):
                del expansions[ticket]
            if not task.cancelled():
                task.exception()  # an abandoned request still owns its worker

        worker.add_done_callback(finished)
        abandoned = asyncio.create_task(run.abandoned.wait())
        try:
            await asyncio.wait((worker, abandoned), return_when=asyncio.FIRST_COMPLETED)
            run.check()
            return await asyncio.shield(worker)
        except Cancelled as exc:
            raise HTTPException(status_code=409, detail="prompt expansion stopped") from exc
        except asyncio.CancelledError:
            # The worker keeps ensure_loaded's lock / run_when_idle's gate.
            # In particular, never cancel a synchronous download or GPU call.
            run.stop()
            raise
        finally:
            abandoned.cancel()


    @router.post("/api/prompt/analyze", response_model=PromptAnalysis)
    async def analyze_prompt(request: PromptAnalysisRequest) -> PromptAnalysis:
        """Check the prompt Generate would send: what the checkpoint's encoders
        read of it, and what its guide says about it.

        CPU only and never the image model - no `ensure_loaded`, no gate - so it
        answers in a moment on a cold server and beside a running night. On a
        daemon thread, like the model load, which interpreter exit never joins;
        a first use's tokenizer fetch holds the check itself for
        `prompt_encoding.LOAD_WAIT` at most.
        """
        try:
            return await detached(
                lambda: prompt_check.analyse(
                    services.model_id(), request, services.backend().applied(request)
                ),
                "prompt-check",
            )
        except Exception as exc:
            raise prompt_failure(exc, "prompt analysis") from exc


    @router.post("/api/prompt/measure", response_model=PromptMeasure)
    async def measure_prompt(request: PromptMeasureRequest) -> PromptMeasure:
        """What the running checkpoint's encoders read of the prompt: Check's
        encoder layer alone, for the chip tray's count and the tags it marks
        past the cut while the prompt is typed, and, where no chips show, for
        the token line under the prompt box - on every checkpoint, so a first
        typing pause on a cold server may fetch any checkpoint's tokenizer files.

        No rules and no tag lookup - the tray asks at every pause in typing, and
        a Check's 64 Danbooru lookups each time would be a second tag lookup
        beside the chips' own - and like Check never the image model, never the
        gate, and off the loop on a daemon thread: a first measure on a cold
        server can fetch the checkpoint's tokenizer files before anyone presses
        Check, and waits `prompt_encoding.LOAD_WAIT` for them at most, after
        which `encoders_detail` says they are still loading.
        """
        model_id = services.model_id()

        def measure() -> PromptMeasure:
            reports, _findings, detail = prompt_encoding.measure(model_id, request.prompt, None)
            return PromptMeasure(
                model_id=model_id,
                encoders=reports,
                encoders_detail=detail,
                encoders_known=prompt_encoding.encoders_for(model_id) is not None,
            )

        try:
            return await detached(measure, "prompt-measure")
        except Exception as exc:
            raise prompt_failure(exc, "prompt measure") from exc


    @router.post("/api/prompt/critique", response_model=PromptCritique)
    async def critique_prompt(request: PromptAnalysisRequest) -> PromptCritique:
        """Ask the local critic about the prompt, grounded in the guide's sources.

        Refused before anything is fetched while the image model has the card.
        The first use downloads llama.cpp and the model - minutes, outside the
        job gate, so a download never holds an image job back - and only the
        critique itself, a server started and stopped on the card, runs inside
        it. Layer 1 runs first: the critic is told what the rules already found,
        and its rewrite goes back through the same checks once the gate is open.
        """
        if not services.critic().supported():
            raise HTTPException(status_code=422, detail=critic_unsupported(services.settings()))
        refuse_critique_now(services)
        try:
            # On a daemon thread, like the model load and for the same reason: a
            # Ctrl-C during the first use's 5 GB must not wait the download out.
            await detached(services.critic().prepare, "critic-prepare")
        except Exception as exc:
            raise prompt_failure(exc, "preparing the prompt critic") from exc
        model_id = services.model_id()
        applied = services.backend().applied(request)
        try:
            analysis = await detached(
                lambda: prompt_check.analyse(services.model_id(), request, applied), "prompt-check"
            )
        except Exception as exc:
            raise prompt_failure(exc, "prompt analysis") from exc
        # Again: a first prepare takes minutes, and a job or a load may have
        # started while it ran.
        refuse_critique_now(services)
        # The rewrite is linted here, not inside the gate. A tokenizer the
        # analysis could not load is tried again by the lint, which waits up to
        # `prompt_encoding.LOAD_WAIT` for it - a first use's fetch, or another
        # check's still in flight; neither wait may hold image jobs back any more
        # than the critic's own download may.
        try:
            critique = await services.jobs().run_when_idle(
                lambda: services.critic().critique(model_id, request, applied, analysis),
                holder="critic",
            )
        except BackendBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise prompt_failure(exc, "prompt critique") from exc
        if critique.rewrite is None:
            return critique
        try:
            findings, artists_checked = await detached(
                lambda: prompt_check.lint(model_id, request, applied, critique.rewrite, analysis), "prompt-check"
            )
        except Exception as exc:
            raise prompt_failure(exc, "prompt analysis") from exc
        # The critic's own checks on the rewrite first (quoted words the prompt
        # does not ask to render), then layer 1's, in the layers' one order.
        merged = sorted(critique.rewrite_findings + findings, key=prompt_check.finding_order)
        return critique.model_copy(
            update={"rewrite_findings": merged, "rewrite_artists_checked": artists_checked}
        )

    return router
