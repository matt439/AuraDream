"""FastAPI app: text-to-image generation plus a browsable gallery."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
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

from . import grid, loras, prompt_encoding, prompt_rules, storage  # noqa: E402
from .backends import get_backend  # noqa: E402
from .config import settings  # noqa: E402
from .critic import CriticError, CriticUnavailable, prompt_critic  # noqa: E402
from .jobs import BackendBusy, JobManager, LoadFailed, short_error  # noqa: E402
from .launch import CHECKPOINTS  # noqa: E402
from .model_defaults import defaults_for, guide_for, hints_for  # noqa: E402
from .prompt_sources import sources_for  # noqa: E402
from .search import strict_scope  # noqa: E402
from .schemas import (  # noqa: E402
    AuthorOrder,
    EncoderReport,
    GalleryPage,
    GenerateRequest,
    GridPreview,
    GridRequest,
    ImageRecord,
    JobAccepted,
    ModelStatus,
    PromptAnalysis,
    PromptAnalysisRequest,
    PromptCritique,
    PromptExpansion,
    PromptExpansionRequest,
    PromptFinding,
    PromptMeasure,
    PromptMeasureRequest,
    RerenderRequest,
    RunSummary,
    TagSlot,
    TagSuggestionResponse,
    TagValidationRequest,
    TagValidationResponse,
    UnstarredPreview,
    error_detail,
)
from .tags import (  # noqa: E402
    TagLookupUnavailable,
    normalize_tag,
    rank_suggestions,
    tag_vocabulary,
    unchecked_results,
)
from .translation import prompt_translator  # noqa: E402

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
)
log = logging.getLogger("local_diffusion")

backend = get_backend("diffusers")

# Loading ~13GB of weights takes minutes, so it runs off-request.
_load_state: dict[str, str | None] = {"state": "unloaded", "detail": None}
_load_lock = asyncio.Lock()

# When this process started. A page that sees it change knows the server it
# was talking to went away - and every job with it - which nothing else in a
# status that reads "not loaded" again says.
STARTED_AT = time.time()

# huggingface_hub's errors a load can end in, by class name: the user can do
# something about each, and their own text is a status line, a request id and
# a URL to a JSON file rather than to the page with the Agree button.
_HUB_FAILURES = (
    "GatedRepoError", "RepositoryNotFoundError", "OfflineModeIsEnabled", "LocalEntryNotFoundError",
)


def _hub_failure(exc: BaseException) -> str | None:
    """The first of `_HUB_FAILURES` this exception is, or was raised from or
    during - a loader re-raising the Hub's error as an OSError of its own still
    has the gate behind it. By class name over each one's MRO, as
    prompt_encoding._reason reads a tokenizer's: GatedRepoError is a
    RepositoryNotFoundError, and the narrower name wins."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        for kind in type(exc).__mro__:
            if kind.__name__ in _HUB_FAILURES:
                return kind.__name__
        exc = exc.__cause__ or exc.__context__
    return None


def _load_failure(exc: BaseException) -> tuple[str, str | None, str | None]:
    """What the status says about a failed load: the sentence, with the steps
    that fix it; its kind ("gated", "not_found", "offline"); and the model page
    where the fix is, for the page to link. The gate is the catalogue's, so the
    sentence says whether it is a click or the authors' approval. Anything that
    is not the Hub's is the exception as `short_error` puts it, and no kind."""
    repo = settings.model_id
    entry = next((checkpoint for checkpoint in CHECKPOINTS if checkpoint.repo == repo), None)
    name = entry.name if entry else repo
    page = f"https://huggingface.co/{repo}"
    login = "run `uv run hf auth login` once with a read token"
    kind = _hub_failure(exc)
    if kind == "GatedRepoError":
        gate = entry.gate if entry else ""
        if gate.startswith("click-through"):
            how = "accept its terms on the model page (a click-through)"
        elif gate.startswith("manual"):
            how = "ask for access on the model page and wait for its authors to approve it"
        else:
            how = "accept its terms, or ask for access, on the model page"
        return f"{name} is gated on Hugging Face: {how}, {login}, then press Retry load.", "gated", page
    if kind == "RepositoryNotFoundError":
        return (
            f"{repo} is not on Hugging Face, or is private: check the checkpoint id "
            "(LOCAL_DIFFUSION_MODEL_ID, or the start-up menu); for a private repo, "
            f"{login} that can read it, then press Retry load.",
            "not_found",
            page,
        )
    if kind == "OfflineModeIsEnabled":
        return (
            f"{name} is not all downloaded yet, and HF_HUB_OFFLINE is set: unset it for one "
            "load, then press Retry load.",
            "offline",
            None,
        )
    if kind == "LocalEntryNotFoundError":
        return (
            f"{name} is not all downloaded yet, and Hugging Face cannot be reached: connect "
            "to the network, then press Retry load.",
            "offline",
            None,
        )
    return short_error(exc), None, None


async def ensure_loaded() -> None:
    """Load the weights once. Idempotent, and raises if the load failed.

    The lock is held across the whole load, not just the state check, so a burst
    of requests waits on a single attempt instead of each starting its own.
    """
    async with _load_lock:
        if backend.is_ready():
            _load_state.update(state="ready", detail=None)
            return
        # An interactive operation holding the generation gate - the prompt
        # critic's server - may have the card, and a load puts the weights on
        # it: wait that out first. Inside the lock, so the critique route's
        # refusal during a load covers the wait too.
        await jobs.wait_for_gate()
        _load_state.update(state="loading", detail=None)
        try:
            await _detached(backend.load)
        except Exception as exc:
            # A gated-repo 403 is user-actionable, not a crash; keep the log to
            # one line and put the traceback behind DEBUG. The log keeps the
            # library's words; the status and the job say what to do.
            log.error("model load failed: %s", short_error(exc))
            log.debug("load traceback", exc_info=True)
            detail, kind, url = _load_failure(exc)
            _load_state.update(state="error", detail=detail, detail_kind=kind, detail_url=url)
            raise LoadFailed(detail) from exc
        _load_state.update(state="ready", detail=None)
        log.info("model ready")


async def _detached(fn, name: str = "load"):
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

    threading.Thread(target=run, daemon=True, name=name).start()
    return await future


async def _load_in_background() -> None:
    try:
        await ensure_loaded()
    except Exception:  # noqa: BLE001 - already recorded in _load_state
        pass


jobs = JobManager(backend, ensure_loaded)


def _stop_critic() -> None:
    """Stop the critic's server, if one runs: on Unload and on the way out.
    A kill not waited out (`CriticError`: the card may still be taken) is
    logged, and never keeps what follows from running - on Unload the
    tokenizers, on the way out the worker, the weights and the tokenizers."""
    try:
        prompt_critic.unload()
    except Exception as exc:  # noqa: BLE001 - Unload and shutdown go on without it
        log.error("stopping the prompt critic failed: %s", short_error(exc))
        log.debug("critic stop traceback", exc_info=True)


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
    #
    # The critic's server goes first: a critique in flight holds the worker
    # thread in a request to that server, which jobs.shutdown() would then
    # wait out. Stopped under it, the request fails at once.
    await asyncio.to_thread(_stop_critic)
    await jobs.shutdown()
    await asyncio.to_thread(backend.try_unload)
    await asyncio.to_thread(prompt_translator.unload)
    await asyncio.to_thread(prompt_encoding.unload)


app = FastAPI(title="Local Diffusion", version="0.1.0", lifespan=lifespan)

# DNS rebinding: a hostile page served from evil.example, whose record is
# re-answered as 127.0.0.1, talks to this API as same-origin - CORS never gets a
# say. Starlette compares the Host header with the port stripped, and the UI
# fetches only relative paths, so this costs nothing to have on.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

def _origin_allowed(origin: str) -> bool:
    """Is this Origin header one of our own pages?

    Compared by hostname and *effective* port rather than by string: a browser
    serialises an origin without a default port, so on `LOCAL_DIFFUSION_PORT=80` the page
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
            defaults["refine_supported"],
        ),
        guide=guide_for(info["model_id"]),
        model_id=info["model_id"],
        quantization=info["quantization"],
        offload=info["offload"],
        detail=_load_state["detail"],
        # Only beside the detail they describe: every other update of the
        # state clears the detail and leaves these behind.
        detail_kind=_load_state.get("detail_kind") if _load_state["detail"] else None,
        detail_url=_load_state.get("detail_url") if _load_state["detail"] else None,
        started_at=STARTED_AT,
        device_name=info.get("device_name"),
        vram_total_mb=info.get("vram_total_mb"),
        vram_used_mb=info.get("vram_used_mb"),
        disk_free_mb=_disk_free_mb(),
        prompt_expansion_supported=backend.prompt_expansion_supported(),
        # All three a few file checks at most - no network, no hashing - which
        # is what lets a route polled every three seconds ask them.
        prompt_critique_supported=prompt_critic.supported(),
        prompt_critique_ready=prompt_critic.ready(),
        prompt_critic=prompt_critic.describe(),
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
        "refine_supported",
    ):
        if info.get(key) is not None:
            defaults[key] = info[key]
    defaults["lora_supported"] = bool(info.get("lora_supported"))
    if defaults["refine_supported"] and defaults.get("refine_strength") is None:
        from .refine import DEFAULT_STRENGTH

        defaults["refine_strength"] = DEFAULT_STRENGTH
    return defaults


def _tag_helper_config() -> dict:
    helper = guide_for(settings.model_id).get("tag_helper")
    if not helper or helper.get("vocabulary") != "danbooru":
        raise HTTPException(status_code=404, detail="this checkpoint has no tag vocabulary")
    return helper


def _special_tags(helper: dict) -> set[str]:
    tags = {
        tag.lower()
        for group in helper.get("groups", [])
        for tag in group.get("tags", [])
    }
    tags.update(tag.lower() for tag in helper.get("recommended_suffix", []))
    return tags


def _model_tag_suggestions(helper: dict, query: str) -> list[dict]:
    prefix = normalize_tag(query)
    if not prefix:
        return []
    return [
        {
            "text": tag,
            "canonical": normalize_tag(tag),
            "category": "model",
            "post_count": None,
        }
        for tag in sorted(_special_tags(helper))
        if normalize_tag(tag).startswith(prefix)
    ]


def _negative_tags(helper: dict) -> set[str]:
    """The guide's recommended negative prompt as tag names: what the
    suggestion list never puts first for the positive prompt."""
    return {
        normalize_tag(part)
        for part in (helper.get("negative_prompt") or "").split(",")
        if part.strip()
    }


@app.get("/api/tags/suggest", response_model=TagSuggestionResponse)
async def suggest_tags(q: str = Query(min_length=1, max_length=100)) -> TagSuggestionResponse:
    helper = _tag_helper_config()
    local = _model_tag_suggestions(helper, q)
    avoid = _negative_tags(helper)
    try:
        remote = await asyncio.to_thread(
            tag_vocabulary.suggest, q, helper["vocabulary_cutoff"]
        )
    except TagLookupUnavailable as exc:
        return TagSuggestionResponse(
            available=False,
            suggestions=rank_suggestions(q, local, [], avoid),
            detail=short_error(exc),
        )
    return TagSuggestionResponse(
        available=True, suggestions=rank_suggestions(q, local, remote, avoid)
    )


def _lookup_tags(helper: dict, tags: list[str]) -> list[dict]:
    """The tag index's verdict on `tags` under this helper's cutoff, special
    tags and year form. Blocking; raises `TagLookupUnavailable`.

    One call for the chips and the analyser, so the two cannot disagree about
    what a tag is.
    """
    return tag_vocabulary.validate(
        tags, _special_tags(helper), helper["vocabulary_cutoff"], bool(helper.get("year_tag"))
    )


def _order_slots(prompt: str | None, reachable: bool) -> dict:
    """The chips' order slots: each comma item of `prompt` in the step of the
    documented order Check's order rule places it in, with the categories a
    Check of this prompt places with (`_tag_categories`) - Check's reading,
    not one of the page's. Empty where no prompt was sent or the rules read
    no order in it (`reads_order`, asked first, so such a prompt costs the
    tag index nothing).

    The categories come from the cache the chips' lookup has just filled,
    except a name past the chips' 64 that Check's own 64 reach (guide tags
    are not asked about): that one is fetched here, once, as a Check would.
    `reachable` False: the chips' own lookup could not reach the index, so a
    Check's would not either, and its categories are None
    (`_tag_categories`' failure path) - asked again here, it would wait out
    the same failure a second time.

    Advisory, like the lookup: a failure here costs the slots, never the
    chips' verdicts it is answered beside.
    """
    if prompt is None:
        return {}
    model_id = settings.model_id
    try:
        order = prompt_rules.documented_order(model_id)
        if order is None or not prompt_rules.reads_order(model_id, prompt):
            return {}
        categories = (
            _tag_categories(model_id, prompt, prompt_rules.detect_style(prompt))[0]
            if reachable else None
        )
        slots = prompt_rules.order_slots(model_id, prompt, categories)
        if slots is None:
            return {}
        return {
            "order_steps": order.steps,
            "slots": [TagSlot(tag=tag, slot=slot) for tag, slot in slots],
        }
    except Exception as exc:  # noqa: BLE001 - advisory: the verdicts still go out
        log.warning("chip order slots failed: %s", short_error(exc))
        log.debug("chip order slots traceback", exc_info=True)
        return {}


def _chip_verdicts(helper: dict, request: TagValidationRequest) -> TagValidationResponse:
    """The tag index's verdict on the chips and, given the prompt, their
    order slots. Blocking: the lookup reaches the network."""
    try:
        results = _lookup_tags(helper, request.tags)
    except TagLookupUnavailable as exc:
        return TagValidationResponse(
            available=False,
            results=unchecked_results(
                request.tags, _special_tags(helper), bool(helper.get("year_tag"))
            ),
            detail=short_error(exc),
            **_order_slots(request.prompt, reachable=False),
        )
    return TagValidationResponse(
        available=True, results=results, **_order_slots(request.prompt, reachable=True)
    )


@app.post("/api/tags/validate", response_model=TagValidationResponse)
async def validate_tags(request: TagValidationRequest) -> TagValidationResponse:
    helper = _tag_helper_config()
    return await asyncio.to_thread(_chip_verdicts, helper, request)


async def _refuse_unusable_inputs(request: GenerateRequest) -> None:
    """Refuse a reference, an adapter or a refine source this server cannot
    run - at submit.

    All three are all-or-nothing inputs, unlike a negative prompt: a night
    queued with references against a pipeline that takes none, with an adapter
    no file backs, or as a refine pass on a checkpoint that has none, would
    render every cell as a plain text-to-image and look like a success.
    `applied()` keeps the *record* honest about that; this is what stops it
    happening at cell 1 of 64.
    """
    if request.refine is not None:
        info = await asyncio.to_thread(backend.describe)
        if not _merged_defaults(info)["refine_supported"]:
            raise HTTPException(
                status_code=422,
                detail=f"{info['model_id']} has no refine pass in this app - its integration "
                "has no compatible image-to-image counterpart. The FLUX.1, SDXL, Z-Image and "
                "Qwen-Image checkpoints have one.",
            )
        try:
            await asyncio.to_thread(storage.reference_paths, [request.refine], "source image")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
        # And one that is there but for another checkpoint: a FLUX.1 adapter
        # on Z-Image fails at the first cell, deep in PEFT, naming neither.
        reason = await asyncio.to_thread(backend.lora_mismatch, request.lora)
        if reason is not None:
            raise HTTPException(status_code=422, detail=reason)


def _listed_loras() -> list[dict]:
    """The directory, each adapter carrying why it does not fit, if it does not.

    A `detail` is what the page hides an entry on and shows in the tooltip;
    its absence means the adapter fits or nothing here could tell. Skipped
    where no adapter can be applied at all - the reason is the listing's.
    """
    found = loras.available()
    if backend.lora_supported():
        for entry in found:
            entry["detail"] = backend.lora_mismatch(entry["name"])
    return found


@app.get("/api/loras")
async def list_loras() -> dict:
    """The adapters on disk, and whether this server could apply one.

    Listed rather than uploaded: an adapter is a 200 MB file the user already
    has, and a directory is the thing they can drop it into without a form.
    """
    return {
        "loras": await asyncio.to_thread(_listed_loras),
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
    # Said as soon as it is true: status must not read "ready" over no
    # weights while the rest lets go, and a load asked for meanwhile must not
    # be relabelled "unloaded" when this route finishes.
    _load_state.update(state="unloaded", detail=None)
    await asyncio.to_thread(prompt_translator.unload)
    # Unload is "give the card and the memory back": a critic's server still
    # running is stopped - its critique then fails with a 502 - and the
    # analyser's cached tokenizers go with the pipeline. A server that would
    # not die is logged, as shutdown logs it (`_stop_critic`), and the rest
    # goes on: raised here it was a 500 over weights that had gone, and the
    # tokenizers stayed. The answer is still true - "unloaded" is the image
    # model's state, and the VRAM it reports is read off the image model's
    # card, device-wide, so on this app's one card memory the server still
    # holds shows there rather than being claimed free. The server is pinned
    # to no card, so on a second GPU it may not: the log line, with its pid,
    # is what says so.
    await asyncio.to_thread(_stop_critic)
    await asyncio.to_thread(prompt_encoding.unload)
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
    except LoadFailed as exc:
        # Logged where it failed; the status's sentence, steps and all.
        raise HTTPException(status_code=500, detail=exc.detail) from exc
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


# The chips' own ceiling on one validation request, so the analyser never
# asks the tag index about more items than the page may.
_MAX_TAG_ITEMS = TagValidationRequest.model_json_schema()["properties"]["tags"]["maxItems"]


# What a guide with no tag helper asks the index with when one of its rules
# reads a category anyway - Anima's "put @ before an artist name", on a card
# trained on Danbooru-style tags: no cutoff, no special tags of its own.
_BARE_DANBOORU = {"vocabulary": "danbooru", "vocabulary_cutoff": None}


def _tag_categories(
    model_id: str, prompt: str, style: str
) -> tuple[dict[str, str] | None, str | None, list[str]]:
    """Each comma item's Danbooru category, for the order and artist checks,
    or None - why some item a rule reads was never looked up, or None - and
    the items the index was asked about, answered or not.

    Only on a checkpoint whose guide names the Danbooru vocabulary, or whose
    rules need to know an artist (`wants_artist_categories`), and a prompt
    that reads as tags, and only for the items a rule would read
    (`category_items`), the chips' 64 at most. No categories when there was
    nothing to ask or the index could not be reached or gave a reply that
    did not read: the rules then place only what the guide's own groups
    know, and the index is never an error here, as it is never one for the
    chips. What that cost is still said (`categories_detail`): offline, an
    out-of-order prompt and an in-order one otherwise got the same answer.
    """
    helper = guide_for(model_id).get("tag_helper")
    if not helper or helper.get("vocabulary") != "danbooru":
        if not prompt_rules.wants_artist_categories(model_id):
            return None, None, []
        helper = _BARE_DANBOORU
    if style not in ("tags", "hybrid"):
        return None, None, []
    wanted = prompt_rules.category_items(model_id, prompt)
    names = wanted[:_MAX_TAG_ITEMS]
    if not names:
        return None, None, []
    try:
        results = _lookup_tags(helper, names)
    except Exception as exc:  # noqa: BLE001 - advisory: a bad reply costs the order checks
        log.info("prompt analysis: tag categories unchecked: %s", short_error(exc))
        if not isinstance(exc, TagLookupUnavailable):
            log.debug("tag category traceback", exc_info=True)
        # The chips' one error covers two things: a connection that never
        # answered, and an answer - an HTTP error, a body cut off, JSON that
        # is not the rows asked for - that could not be used.
        from urllib.error import HTTPError

        cause = exc.__cause__ if isinstance(exc, TagLookupUnavailable) else None
        reason = (
            "the tag index was unreachable"
            if isinstance(cause, OSError) and not isinstance(cause, HTTPError)
            else "the tag index gave no usable reply"
        )
        return None, f"{reason}, so no tag has its Danbooru category", names
    categories = {row["tag"]: row["category"] for row in results if row.get("category")}
    if len(wanted) > len(names):
        return categories, (
            f"the tag index was asked about the first {len(names)} of {len(wanted)} tags "
            "only, so the rest have no Danbooru category"
        ), names
    return categories, None, names


def _order_detail(
    model_id: str,
    prompt: str,
    categories: dict[str, str] | None,
    categories_detail: str | None,
    asked: list[str],
) -> str | None:
    """Why a placed order was only partly checked, or None where the rules
    decided it whatever category each item has (`unplaced_items` empty, with
    the categories they placed with: `places_every_item`, the predicate the
    critic reads for a reorder of its own) - a lone item, or one every slot
    of which sits in order, or out of it, beside the rest, included.

    Each unplaced item under the reason it has no category: the lookup's
    own (`categories_detail`: unreachable, no usable reply, past the cap), a
    tag the index answered for with none it could use - not found, and
    deprecated, unused or past the checkpoint's cutoff alike, the chips'
    reasons, so the words claim none of them - a prose prompt's clause -
    prose is never looked up, and only a whole sentence is placed by its
    shape - or an item never asked about at all (lettering, an @ name).
    Every such item, not only those the lookup lost: said of those alone, an
    unknown tag or a prose clause left the order drawn as checked while the
    server treated it as partly placed.
    """
    unplaced = list(dict.fromkeys(prompt_rules.unplaced_items(model_id, prompt, categories)))
    if not unplaced:
        return None
    wanted = set(prompt_rules.category_items(model_id, prompt))
    asked_set = set(asked)
    prose = prompt_rules.detect_style(prompt) not in ("tags", "hybrid")
    lost: list[str] = []
    unknown: list[str] = []
    clauses: list[str] = []
    others: list[str] = []
    for text in unplaced:
        if asked and text in wanted:
            # Asked and given no usable category - or never answered: the
            # lookup failed, or stopped at the cap before it.
            (unknown if categories is not None and text in asked_set else lost).append(text)
        elif text in wanted and prose:
            clauses.append(text)
        else:
            others.append(text)

    def say(texts: list[str], one: str, many: str) -> str:
        # The count, the words for one or many, and the first three items.
        shown = ", ".join(f'"{text}"' for text in texts[:3])
        more = f" and {len(texts) - 3} more" if len(texts) > 3 else ""
        return (one if len(texts) == 1 else many).replace("{n}", str(len(texts))) + (
            f": {shown}{more}"
        )

    reasons = [
        say(texts, one, many)
        for texts, one, many in (
            (lost, f"{categories_detail}, and {{n}} tag is not placed",
             f"{categories_detail}, and {{n}} tags are not placed"),
            (unknown,
             "the tag index has no usable Danbooru category for {n} tag, so it is not placed",
             "the tag index has no usable Danbooru category for {n} tags, so they are not "
             "placed"),
            (clauses,
             "the tag index is not asked about a prose prompt, so {n} clause that is not a "
             "whole sentence is not placed",
             "the tag index is not asked about a prose prompt, so {n} clauses that are not "
             "whole sentences are not placed"),
            (others, "the tag index is not asked about {n} item, so it is not placed",
             "the tag index is not asked about {n} items, so they are not placed"),
        )
        if texts
    ]
    return "; ".join(reasons)


def _finding_order(finding: PromptFinding) -> tuple:
    # Warnings first, the prompt's before the negative prompt's, then in
    # reading order - the rules' own sort, kept across the two layers.
    return (
        finding.severity != "warn",
        finding.field != "prompt",
        finding.start is None,
        finding.start or 0,
    )


def _checked(
    model_id: str,
    request: GenerateRequest,
    applied: GenerateRequest,
    categories: dict[str, str] | None,
) -> tuple[list[EncoderReport], list[PromptFinding], str | None]:
    """Layer 1 over one request: the encoder reports, every finding, and why
    any encoder went unchecked.

    The negative prompt is measured only as `applied` runs it: one the
    pipeline drops never reaches an encoder, and a budget reported for it
    would be a number about nothing.
    """
    reports, encoder_findings, detail = prompt_encoding.measure(
        model_id, request.prompt, applied.negative_prompt
    )
    findings = prompt_rules.check(model_id, request, applied, categories) + encoder_findings
    return reports, sorted(findings, key=_finding_order), detail


def _analyse(
    request: PromptAnalysisRequest, applied: GenerateRequest | None = None
) -> PromptAnalysis:
    """The prompt against the running checkpoint's guide and tokenizers.

    Blocking - a first use can fetch a tokenizer, and a tag checkpoint asks
    the tag index - so the routes run it off the loop. Never the image
    model: `applied()` answers from the profile before a load, and the
    tokenizers are read on the CPU.
    """
    model_id = settings.model_id
    applied = applied if applied is not None else backend.applied(request)
    prompt = request.prompt
    # The shape alone decides whether to ask the tag index; what is reported
    # is the style the rules applied, which the answer's categories can move
    # ("the legend of zelda" is a series, not a phrase).
    categories, categories_detail, asked = _tag_categories(
        model_id, prompt, prompt_rules.detect_style(prompt)
    )
    reports, findings, detail = _checked(model_id, request, applied, categories)
    order = prompt_rules.documented_order(model_id)
    order_checked = order is not None and prompt_rules.order_checked(model_id, prompt)
    return PromptAnalysis(
        model_id=model_id,
        style=prompt_rules.detect_style(prompt, categories),
        words=prompt_rules.word_count(prompt),
        encoders=reports,
        encoders_detail=detail,
        findings=findings,
        order=order,
        order_checked=order_checked,
        categories_detail=categories_detail,
        order_detail=_order_detail(model_id, prompt, categories, categories_detail, asked)
        if order_checked else None,
        categories=categories,
        author_orders=[] if order is not None else [
            AuthorOrder(text=excerpt.text, url=excerpt.url, note=excerpt.note)
            for excerpt in sources_for(model_id).excerpts
            if excerpt.order
        ],
        reordered=prompt_rules.reorder_tags(model_id, prompt, categories),
    )


def _lint(
    model_id: str, request: GenerateRequest, applied: GenerateRequest, text: str
) -> list[PromptFinding]:
    """Layer 1's findings for `text` in place of the prompt - the critic's
    rewrite, held to the same rules as what the user typed.

    Blocking, and like `_analyse` it can fetch a tokenizer, so the route runs
    it off the loop and after the job gate. No tag categories: the analysis
    asked the tag index once for this critique, and without them the order
    check places only what the guide's groups know.
    """
    update = {"prompt": text}
    return _checked(
        model_id, request.model_copy(update=update), applied.model_copy(update=update), None
    )[1]


def _prompt_failure(exc: Exception, doing: str) -> HTTPException:
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


@app.post("/api/prompt/analyze", response_model=PromptAnalysis)
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
        return await _detached(lambda: _analyse(request), "prompt-check")
    except Exception as exc:
        raise _prompt_failure(exc, "prompt analysis") from exc


@app.post("/api/prompt/measure", response_model=PromptMeasure)
async def measure_prompt(request: PromptMeasureRequest) -> PromptMeasure:
    """What the running checkpoint's encoders read of the prompt: Check's
    encoder layer alone, for the chip tray's count and the tags it marks
    past the cut while the prompt is typed.

    No rules and no tag lookup - the tray asks at every pause in typing, and
    a Check's 64 Danbooru lookups each time would be a second tag lookup
    beside the chips' own - and like Check never the image model, never the
    gate, and off the loop on a daemon thread: a first measure on a cold
    server can fetch the checkpoint's tokenizer files before anyone presses
    Check, and waits `prompt_encoding.LOAD_WAIT` for them at most, after
    which `encoders_detail` says they are still loading.
    """
    model_id = settings.model_id

    def measure() -> PromptMeasure:
        reports, _findings, detail = prompt_encoding.measure(model_id, request.prompt, None)
        return PromptMeasure(model_id=model_id, encoders=reports, encoders_detail=detail)

    try:
        return await _detached(measure, "prompt-measure")
    except Exception as exc:
        raise _prompt_failure(exc, "prompt measure") from exc


def _critic_unsupported() -> str:
    from .critic import pinned_build_refusal

    if not settings.critic:
        return "the prompt critic is switched off (LOCAL_DIFFUSION_CRITIC)"
    # A build published for the platform that this system cannot run (glibc
    # too old) says so, rather than that there is none.
    return pinned_build_refusal() or (
        "llama.cpp publishes no pinned llama-server build for this platform; set "
        "LOCAL_DIFFUSION_CRITIC_SERVER to a llama-server of your own"
    )


def _refuse_critique_now() -> None:
    """409 while the card is promised to the image model.

    A job queued or running, as for Expand; and a load in progress, which is
    moving the weights onto the card outside the job gate - a critic started
    beside it would find the memory it planned on gone.
    """
    if jobs.busy():
        raise HTTPException(
            status_code=409, detail="an image job is queued or running; wait for it to finish"
        )
    if _load_lock.locked():
        raise HTTPException(
            status_code=409, detail="the image model is loading; wait for the load"
        )


@app.post("/api/prompt/critique", response_model=PromptCritique)
async def critique_prompt(request: PromptAnalysisRequest) -> PromptCritique:
    """Ask the local critic about the prompt, grounded in the guide's sources.

    Refused before anything is fetched while the image model has the card.
    The first use downloads llama.cpp and the model - minutes, outside the
    job gate, so a download never holds an image job back - and only the
    critique itself, a server started and stopped on the card, runs inside
    it. Layer 1 runs first: the critic is told what the rules already found,
    and its rewrite goes back through the same checks once the gate is open.
    """
    if not prompt_critic.supported():
        raise HTTPException(status_code=422, detail=_critic_unsupported())
    _refuse_critique_now()
    try:
        # On a daemon thread, like the model load and for the same reason: a
        # Ctrl-C during the first use's 5 GB must not wait the download out.
        await _detached(prompt_critic.prepare, "critic-prepare")
    except Exception as exc:
        raise _prompt_failure(exc, "preparing the prompt critic") from exc
    model_id = settings.model_id
    applied = backend.applied(request)
    try:
        analysis = await _detached(lambda: _analyse(request, applied), "prompt-check")
    except Exception as exc:
        raise _prompt_failure(exc, "prompt analysis") from exc
    # Again: a first prepare takes minutes, and a job or a load may have
    # started while it ran.
    _refuse_critique_now()
    # The rewrite is linted here, not inside the gate. A tokenizer the
    # analysis could not load is tried again by the lint, which waits up to
    # `prompt_encoding.LOAD_WAIT` for it - a first use's fetch, or another
    # check's still in flight; neither wait may hold image jobs back any more
    # than the critic's own download may.
    try:
        critique = await jobs.run_when_idle(
            lambda: prompt_critic.critique(model_id, request, applied, analysis)
        )
    except BackendBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise _prompt_failure(exc, "prompt critique") from exc
    if critique.rewrite is None:
        return critique
    try:
        findings = await _detached(
            lambda: _lint(model_id, request, applied, critique.rewrite), "prompt-check"
        )
    except Exception as exc:
        raise _prompt_failure(exc, "prompt analysis") from exc
    # The critic's own checks on the rewrite first (quoted words the prompt
    # does not ask to render), then layer 1's, in the layers' one order.
    merged = sorted(critique.rewrite_findings + findings, key=_finding_order)
    return critique.model_copy(update={"rewrite_findings": merged})


# -------------------------------------------------------------- generation


@app.post("/api/generate", response_model=JobAccepted)
async def generate(request: GenerateRequest) -> JobAccepted:
    try:
        request.check_budget()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await _refuse_unusable_inputs(request)
    try:
        request.check_refine_steps()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
    started it. Prompt variants, when given, are the prompt axis; the preview
    then shows their text under `prompt` and their labels under `variant`.
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
    run_id = grid.run_id(cells, settings.model_id, settings.quantization)
    if not request.resume:
        return cells, values, 0, run_id
    kept = await _not_yet_rendered(cells, request.width, request.height)
    return kept, values, len(cells) - len(kept), run_id


async def _not_yet_rendered(cells: list[grid.Cell], width: int, height: int) -> list[grid.Cell]:
    """The cells the gallery does not hold yet at this size: what Resume runs.

    Compare what the pipeline would *run*, not what the panel says: the rows
    were written from `backend.applied()`, so a negative prompt a Turbo cell
    at CFG 0 never saw is None on both sides rather than on one. Off the
    loop: up to three chunked scans, fired by the preview on every keystroke.
    """
    done = await asyncio.to_thread(
        storage.completed_cells,
        grid.prompts_of(cells),
        width,
        height,
        settings.model_id,
        settings.quantization,
    )
    return [
        cell
        for cell in cells
        if grid.prune_done([grid.Cell(request=backend.applied(cell.request), seed=cell.seed)], done)
    ]


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
    job = await jobs.submit_cells(
        cells,
        run_id=run_id,
        budget_minutes=request.budget_minutes,
        prompt_template=request.prompt,
        comparison=bool(request.prompt_variants),
    )
    return JobAccepted(job_id=job.id, seed=job.seed, run_id=job.run_id)


def _foreign_refusal(count: int, foreign: list[str]) -> str:
    """Re-render's refusal of a selection that reaches other checkpoints, with the way out.

    Unticking All models is the exact way: the gallery's scope is the running
    id itself (`_model_scope`). The search box's `model:` is a substring, so
    it is offered only where the running id, whole, is in none of the other
    ids selected - and only then does adding it leave them out. The id's tail
    was offered, and `model:Anima` is Animagine too; nor is the whole id safe
    everywhere (`Qwen/Qwen-Image` is in `Qwen/Qwen-Image-Edit`).
    """
    running = settings.model_id
    narrows = not any(ch.isspace() for ch in running) and not any(
        running.lower() in other.lower() for other in foreign
    )
    search = f", or add model:{running} to the search" if narrows else ""
    return (
        f"{count} of the selected images {'was' if count == 1 else 'were'} made on "
        f"{', '.join(foreign)}; the server is running {running}, whose steps, CFG and mu "
        f"differ. Untick All models{search}."
    )


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
    records, total, _ = await asyncio.to_thread(
        storage.list_images,
        limit=settings.max_grid_images + 1,
        search=request.search,
        favorites_only=request.favorites,
        model_id=_model_scope(request.all_models),
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
    # checkpoint is refused rather than rendered wrong. Only reachable with
    # `all_models`: the scope keeps them out of the selection otherwise.
    foreign = sorted({r.model_id for r in records if r.model_id != settings.model_id})
    if foreign:
        count = sum(r.model_id != settings.model_id for r in records)
        raise HTTPException(status_code=422, detail=_foreign_refusal(count, foreign))
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
                # A refined record re-renders as a refine pass of the same
                # source at the new size: the source and the strength are its
                # reproduction set as much as the seed.
                refine=r.refine,
                strength=r.strength if r.refine is not None else None,
                seed=r.seed,
            ),
            seed=r.seed,
            # What the placeholders were set to, or which version of a
            # comparison this is: the only place a variant's label lives, so
            # the re-rendered run's sheet keeps the rows the original had.
            variables=r.variables or None,
        )
        for r in reversed(records)
    ]
    # One check per distinct combination rather than per cell: a selection of
    # 400 images carries at most a handful of adapters and sources between them.
    for spec in {(c.request.lora, tuple(c.request.reference), c.request.refine) for c in cells}:
        if spec != (None, (), None):
            await _refuse_unusable_inputs(cells[0].request.model_copy(
                update={"lora": spec[0], "reference": list(spec[1]), "refine": spec[2]}
            ))
    try:
        for cell in cells:
            cell.request.check_refine_steps()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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

    Without Resume, the cells the gallery already holds are counted too: Run
    grid again after a night with a failed cell renders the whole night again,
    and the count is what says Resume would not. Only where the seeds are
    fixed - an explicit base or a list - since a random base matches nothing.
    """
    cells, values, skipped, _ = await _expand(request)
    in_gallery = 0
    if not request.resume and (request.seed is not None or request.seed_axis.strip()):
        in_gallery = len(cells) - len(
            await _not_yet_rendered(cells, request.width, request.height)
        )
    # Cost is per step, and a grid's cells rarely agree on how many they take.
    per_step_ms = await asyncio.to_thread(
        storage.step_cost_ms, request.width, request.height, settings.model_id
    )
    refine_costs = await asyncio.to_thread(
        storage.refine_cost_ms, request.width, request.height, settings.model_id
    )
    per_image_bytes = await asyncio.to_thread(
        storage.image_bytes, request.width, request.height, settings.model_id
    )
    images = len(cells) * request.num_images
    listed = len(values.get("seed", []))
    # Where a refine pass starts on the schedule, for the Refine dialog: only
    # where every cell starts at one point - the same steps, mu and strength
    # as the pipeline will run them - which a sweep's do not. Schedulers
    # only, no weights, and None before a load.
    start_sigma = None
    if request.refine is not None and cells:
        applied = [backend.applied(cell.request) for cell in cells]
        if len({(r.steps, r.mu, r.strength) for r in applied}) == 1 and applied[0].refine is not None:
            start_sigma = backend.start_sigma(applied[0])
    return GridPreview(
        values=values,
        cells=len(cells),
        images=images,
        skipped=skipped,
        in_gallery=in_gallery,
        # As the pipeline would run them: the capability flags the status
        # endpoint publishes, applied cell by cell, say which cells collapse
        # into one picture before the night is spent finding out.
        duplicates=grid.duplicate_count(cells, backend.applied),
        summary=grid.describe(values, listed or request.seeds, listed=bool(listed)),
        seconds=grid.estimate_seconds(cells, per_step_ms, refine_costs=refine_costs),
        bytes=per_image_bytes * images if per_image_bytes else None,
        start_sigma=start_sigma,
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
async def recent_runs(
    limit: int = Query(20, ge=1, le=200), all_models: bool = False
) -> list[RunSummary]:
    """The gallery grouped by run, newest first - one line per night's grid.

    Scoped like the gallery it lists: this checkpoint's runs unless asked for
    every one, so picking a run never opens an empty sheet.
    """
    return await asyncio.to_thread(storage.runs, limit, _model_scope(all_models))


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
    # The ceiling follows LOCAL_DIFFUSION_MAX_GRID_IMAGES, which /api/status publishes, so
    # the sheet's one-page request cannot be refused by the server that sized it.
    limit: int = Query(48, ge=1, le=max(1024, settings.max_grid_images)),
    offset: int = Query(0, ge=0),
    # Keyset paging: rows older than this id. What the browser sends, because
    # an offset over a gallery a sweep is writing into repeats the tail of one
    # page at the head of the next and skips as many older rows for good.
    before: int | None = Query(None, ge=1),
    search: str | None = None,
    favorites: bool = False,
    # The running checkpoint's images unless asked for every checkpoint's:
    # steps, CFG and mu are the checkpoint's, so another one's pictures are
    # settings that cannot be reused, under a Reuse button that warns. The
    # default is the server's because the page cannot know the checkpoint
    # before /api/status answers, which behind a load is minutes.
    all_models: bool = False,
) -> GalleryPage:
    # Every storage call runs off the loop, like describe() and step_cost_ms
    # above: SQLite's busy handler waits up to 15 s, and a write that lands
    # while `clear.py` holds the file would otherwise freeze the SSE stream,
    # the cancel endpoint and the status poll for that long.
    try:
        # One row past the page, so `has_more` is a fact rather than a guess
        # from a total that keeps growing while the page is read.
        items, total, others = await asyncio.to_thread(
            storage.list_images,
            limit=limit + 1,
            offset=offset,
            search=search,
            favorites_only=favorites,
            before=before,
            model_id=_model_scope(all_models),
        )
    except ValueError as exc:  # a filter term that is not a number
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GalleryPage(
        items=items[:limit],
        total=total,
        limit=limit,
        offset=offset,
        has_more=len(items) > limit,
        other_models=others,
    )


@app.get("/api/images/{image_id}", response_model=ImageRecord)
async def image_record(image_id: int) -> ImageRecord:
    """One image's record, by id.

    The panel's refine source is an id like a reference chip's, but it needs
    more than a picture: the source's own size is what the scale buttons
    multiply, and a source restored from localStorage or from a refined
    record's `refine` may be on no gallery page the browser has loaded. A 404
    is how the panel learns the source has been deleted.
    """
    record = await asyncio.to_thread(storage.get_image, image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown image")
    return record


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
    # A page that reattaches to the job that made it is replayed the job's
    # records; a deleted one came back as a broken tile.
    jobs.forget_images([image_id])
    return {"deleted": image_id}


def _model_scope(all_models: bool) -> str | None:
    """The gallery's checkpoint scope: the configured id, or None for every one.

    `settings.model_id` rather than the loaded pipeline's, so the scope holds
    before the first load and the page's first paint is already the right
    checkpoint's. Exact, where the search box's `model:` is a substring -
    `model:flux` is both FLUX.1 checkpoints.
    """
    return None if all_models else settings.model_id


def _unstarred_scope(search: str | None) -> str | None:
    """The search box as a strict scope, or a 422 that says why it is not one.

    `strict_scope` refuses free text with the grammar's own words; here the
    reader has just watched the gallery match that text, so the reply says
    what to do instead.
    """
    try:
        strict_scope(search)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Delete unstarred takes key:value terms or an empty search - {exc}",
        ) from exc
    return search or None


@app.get("/api/gallery/unstarred", response_model=UnstarredPreview)
async def unstarred(
    search: str | None = None,
    all_models: bool = False,
    newer_than: int | None = Query(default=None, ge=0),
) -> UnstarredPreview:
    """What "Delete unstarred" would remove under this filter, and what it would keep.

    Under the gallery's checkpoint scope too, since the delete is: the count
    the dialog shows has to be the count of what the page is showing. What
    the delete keeps besides the stars - the images a starred one was made
    from - comes off the count with them, or every confirmed number would
    meet a different one under the delete's lock and answer 409.

    `job_active` is whether a job is queued or running here - the server's
    answer, not the asking page's: a job another tab started can land an
    image in scope while the dialog is open just the same, and the dialog
    says so before the count is typed.

    `newer_than` is the newest id the page held before it brought its grid
    up to date, and `newer` how many of the count are above it: the dialog's
    "made since this page last loaded", counted as the delete counts.
    """
    found = await asyncio.to_thread(
        storage.matching_preview, _unstarred_scope(search), _model_scope(all_models), newer_than
    )
    return UnstarredPreview(
        unstarred=found["records"] - found["favorites"] - found["sources_kept"],
        job_active=jobs.busy(),
        **found,
    )


@app.delete("/api/gallery/unstarred")
async def delete_unstarred(
    search: str | None = None, expected: int = Query(ge=0), all_models: bool = False
) -> dict:
    """The gallery's Delete unstarred: `clear --where <search> --keep-favorites` over HTTP.

    The one bulk delete the page offers, and the inverse of the per-card one:
    star what is worth keeping, then remove the rest of the run in one gesture
    rather than the rest one card at a time. The scope is the search box read
    strictly - `key:value` terms only, an empty box meaning the whole gallery
    - and `expected` is the count the dialog showed, so a sweep that landed
    more unstarred images while it was open is answered 409 rather than
    swept up on a number confirmed for fewer. A starred image keeps what it
    was made from - a pass its source, a render its references - since the
    star is a promise to keep it reproducible. Rows and their files only: no
    orphan sweep and no vacuum, which stay the CLI's. The checkpoint scope
    is the page's: with All models off, another checkpoint's images are not
    on the screen, and a sweep must not reach what is not on the screen.
    """
    scope = _unstarred_scope(search)
    try:
        result = await asyncio.to_thread(
            storage.delete_matching,
            scope,
            keep_favorites=True,
            expected=expected,
            model_id=_model_scope(all_models),
        )
    except storage.SelectionChanged as exc:
        # Whether a job may move the count again is the server's to say: the
        # job is as likely another tab's as the asking page's.
        running = (
            " A job is still running and may add more images here: press Delete "
            "unstarred again, or let it finish first."
            if jobs.busy() else ""
        )
        raise HTTPException(
            status_code=409,
            detail=f"The selection changed: {exc.found} unstarred images match now, "
            f"not the {exc.expected} confirmed. Nothing was deleted; look again.{running}",
        ) from exc
    jobs.forget_images(result["ids"])
    return {"deleted": result["ids"], "files": result["files"]}


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

    The critic's server is stopped here for the same reason: a critique
    in flight is an open request too, waiting on that server - typically
    10-20 s, and up to its 600 s request timeout when `--fit` has put
    layers on the CPU. On a thread of its own, since stopping a process
    waits for it and a signal handler should not.

    At module level rather than inside `run()` only so the tests can reach
    `handle_exit`.
    """

    def handle_exit(self, sig, frame) -> None:
        for job in jobs.active():
            job.cancel_requested = True
        threading.Thread(target=_stop_critic, daemon=True, name="critic-stop").start()
        super().handle_exit(sig, frame)


def run() -> None:
    """Serve until a signal. Ctrl-C exits with status 130, as the start-up
    menu's does, and without a traceback: once shutdown is complete uvicorn
    re-raises the signal it captured, which lands here as KeyboardInterrupt
    (`uvicorn.run()` swallows it, and the status with it)."""
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        # A client that stops reading its stream must not hold shutdown open.
        timeout_graceful_shutdown=30,
    )
    try:
        Server(config).run()
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    run()
