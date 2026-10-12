"""FastAPI app: text-to-image generation plus a browsable gallery."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Run as `python -m backend.main`, settle the checkpoint before `.config`
# freezes it: remembered browser choices, overridden by environment and
# --model/--quant/--offload flags. Imported by uvicorn or the tests
# instead, this does nothing and the environment stands as it is.
if __name__ == "__main__":
    from .launch import configure

    configure()

from . import api_bundles, api_gallery, api_generate, api_jobs, api_more_seeds, api_png, api_prompt, api_refine_batch, api_swap, app_context, bundle_common, bundles, gallery_lock, loras, model_selection, prompt_encoding, startup_preferences, storage  # noqa: E402
from .backends import get_backend  # noqa: E402
from .detached import detached as _detached  # noqa: E402
from .model_selection import current_model_id, current_settings  # noqa: E402
from .model_info import details_for  # noqa: E402
from .config import settings  # noqa: E402
from .critic import CriticError, prompt_critic  # noqa: E402
from .jobs import JobManager, LoadFailed, short_error  # noqa: E402
from .launch import CHECKPOINTS  # noqa: E402
from .model_defaults import defaults_for, guide_for, hints_for  # noqa: E402
from .schemas import (  # noqa: E402
    ExpanderDownload,
    ExpanderStatus,
    ModelStatus,
    ModelSwitchRequest,
)
from .translation import MODEL_BYTES as TRANSLATOR_BYTES  # noqa: E402
from .translation import prompt_translator  # noqa: E402

logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
)
log = logging.getLogger("local_diffusion")

backend = get_backend("diffusers")

# Loading ~13GB of weights takes minutes, so it runs off-request.
_load_state: dict[str, str | None] = {"state": "unloaded", "detail": None}
_load_lock = asyncio.Lock()
_switching = False
_switch_task: asyncio.Task | None = None
_model_requests = 0
# The counted requests in flight by path, for a refusal that names them.
_model_request_paths: dict[str, int] = {}
# A switch waiting for counted requests to finish: new ones are refused, as
# during the switch itself, so the wait ends.
_switch_draining = False
# How long a switch waits for counted requests already in flight before it
# refuses. The page's own cold reads - the adapter list, the token count and
# the tag check it sends when a load lands or typing pauses - take a moment,
# and a first tokenizer fetch or a slow Danbooru up to tens of seconds.
SWITCH_DRAIN_SECONDS = 10.0
_startup_preload: bool | None = None
# Why the last switch stopped before selecting its checkpoint, the running one
# still selected; cleared by the next switch, load or unload.
_switch_error: str | None = None
# Which saved choices this start's flags/environment override (launch.configure).
_startup_overrides = startup_preferences.startup_overrides()

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
    RepositoryNotFoundError, and the narrower name wins.

    Except that OfflineModeIsEnabled wins wherever it is in the chain:
    huggingface_hub raises LocalEntryNotFoundError ("cannot be reached")
    *from* it when HF_HUB_OFFLINE is set, and the outer one sent the user to
    a network that was up rather than to the variable."""
    seen: set[int] = set()
    found: list[str] = []
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        kind = next((k.__name__ for k in type(exc).__mro__ if k.__name__ in _HUB_FAILURES), None)
        if kind is not None:
            found.append(kind)
        exc = exc.__cause__ or exc.__context__
    if "OfflineModeIsEnabled" in found:
        return "OfflineModeIsEnabled"
    return found[0] if found else None


def _catalogue_label(repo: str) -> str | None:
    """The start-up menu's name for a checkpoint, by its exact id - "NetaYume
    Lumina v4", where the repository's own name is the parent model's - or
    None off the catalogue, where the page shows the id's last part."""
    return next((checkpoint.name for checkpoint in CHECKPOINTS if checkpoint.repo == repo), None)


def _load_failure(exc: BaseException) -> tuple[str, str | None, str | None]:
    """What the status says about a failed load: the sentence, with the steps
    that fix it; its kind ("gated", "not_found", "offline"); and the model page
    where the fix is, for the page to link. The gate is the catalogue's, so the
    sentence says whether it is a click or the authors' approval. Anything that
    is not the Hub's is the exception as `short_error` puts it, and no kind."""
    repo = current_model_id(settings)
    entry = next((checkpoint for checkpoint in CHECKPOINTS if checkpoint.repo == repo), None)
    name = entry.name if entry else repo
    # The Aesthetic selector records a distinct checkpoint from the same Hub
    # repository. Error links must point at its real source, not the selector.
    from .backends import anima_loading

    hub_repo = anima_loading.MODEL_ID if anima_loading.matches(repo) else repo
    page = f"https://huggingface.co/{hub_repo}"
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
            f"{hub_repo} is not on Hugging Face, or is private: check the checkpoint id "
            "(Models in the browser, or LOCAL_DIFFUSION_MODEL_ID); for a private repo, "
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


async def _load_in_background() -> None:
    try:
        await ensure_loaded()
    except Exception:  # noqa: BLE001 - already recorded in _load_state
        pass


jobs = JobManager(backend, ensure_loaded)
bundle_manager = api_bundles.BundleManager(
    lambda: jobs,
    lambda: _switching or _load_lock.locked() or _load_state["state"] == "loading",
)


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
    try:
        storage.settle_partials()
    except gallery_lock.GalleryBusy:
        pass  # another process is mid-operation; the next bundle operation sweeps
    if settings.load_on_startup:
        asyncio.create_task(_load_in_background())
    yield
    if _switch_task is not None and not _switch_task.done():
        _switch_task.cancel()
        await asyncio.gather(_switch_task, return_exceptions=True)
    await bundle_manager.finish()
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
app.add_middleware(api_bundles.GalleryAdmission, get_manager=lambda: bundle_manager)


@app.exception_handler(gallery_lock.GalleryBusy)
async def gallery_busy(request: Request, exc: gallery_lock.GalleryBusy):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(bundles.BundleError)
async def invalid_bundle(request: Request, exc: bundles.BundleError):
    body = {"detail": bundle_common.sentence(str(exc))}
    if isinstance(exc, bundles.MissingImage):
        # Which row, for the dialog's Show it.
        body.update(image_id=exc.image_id, binned=exc.binned)
    return JSONResponse(body, status_code=409 if isinstance(exc, bundles.BundleChanged) else 422)

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


@app.middleware("http")
async def hold_model_for_request(request: Request, call_next):
    """Keep one checkpoint throughout model-dependent request validation/work.

    Pydantic fills sampling defaults before the route starts, and routes can
    await gallery reads, tokenizer loads and adapter checks before submitting
    a job. Counting at the HTTP boundary makes all of those one read of the
    selected checkpoint: a switch refuses until they finish, and once a switch
    starts no new reader can enter. This does not take the GPU gate, so cold
    prompt checks continue to work beside generation.

    A page may still have the previous checkpoint's controls between polls.
    Its optional revision header refuses that stale submission before request
    defaults or side effects are evaluated. Existing API clients without the
    header continue to use the checkpoint currently selected by the server.
    """
    global _model_requests

    path = request.url.path
    depends_on_model = path in {
        "/api/generate", "/api/grid", "/api/rerender", "/api/swap-tags", "/api/refine/batch", "/api/more-seeds",
        "/api/loras", "/api/png/settings",
        "/api/model/load", "/api/model/unload",
        "/api/gallery/unstarred",
    } or path.startswith(("/api/grid/", "/api/prompt/", "/api/tags/"))
    selects_model = path == "/api/model/switch"
    if not depends_on_model and not selects_model:
        return await call_next(request)
    if _switching or _switch_draining:
        return JSONResponse(
            {"detail": "the checkpoint is switching; wait for the switch to finish"},
            status_code=409,
        )
    expected = request.headers.get("x-local-diffusion-model-revision")
    if expected is not None:
        try:
            expected_revision = int(expected)
        except ValueError:
            return JSONResponse({"detail": "invalid model revision"}, status_code=422)
        if expected_revision != model_selection.revision():
            return JSONResponse(
                {"detail": "the checkpoint changed; refresh its settings before trying again"},
                status_code=409,
            )
    if selects_model:
        # The switch performs its own exclusive admission. Counting it as a
        # model reader would make it refuse itself forever.
        return await call_next(request)
    _model_requests += 1
    _model_request_paths[path] = _model_request_paths.get(path, 0) + 1
    try:
        return await call_next(request)
    finally:
        _model_requests -= 1
        if _model_request_paths[path] > 1:
            _model_request_paths[path] -= 1
        else:
            del _model_request_paths[path]


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
    # A status poll is allowed during a switch. Never label an old threaded
    # description with a new selection/revision or with the new load's state.
    # The gallery's newest id is read in the same thread, inside this loop:
    # an await after it let a switch run to its end between the description
    # and the rest, and the answer paired the old checkpoint and revision
    # with the finished switch's `switching` false - the page took that for
    # a switch with no revision of its own to come, and the revision that
    # did come wiped the Models note's outcome (smoke-model-switch). From
    # the loop's end to the return nothing awaits.
    while True:
        revision = model_selection.revision()
        switching = _switching
        # The live jobs first, then the gallery's newest id: a job saves its
        # rows before it ends, so one missing from `active` has every image
        # counted in `newest`, and the page never sees a job gone with its
        # images not yet in. Off the event loop: the read waits a moment for
        # the reservation.
        active = [job.id for job in jobs.active()]
        info, newest = await asyncio.to_thread(_describe_with_newest)
        if revision == model_selection.revision() and switching == _switching:
            break
    state = _load_state["state"]
    if state == "unloaded" and backend.is_ready():
        state = "ready"
    defaults = _merged_defaults(info)
    card_holder, card_ticket = jobs.card()
    return ModelStatus(
        state=state,
        switching=_switching,
        model_revision=revision,
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
        format=model_selection.format_for(settings),
        load_on_startup=_load_on_startup(),
        startup_overrides=_startup_overrides,
        switch_error=_switch_error,
        detail=_load_state["detail"],
        # Only beside the detail they describe: every other update of the
        # state clears the detail and leaves these behind.
        detail_kind=_load_state.get("detail_kind") if _load_state["detail"] else None,
        detail_url=_load_state.get("detail_url") if _load_state["detail"] else None,
        started_at=STARTED_AT,
        model_label=_catalogue_label(info["model_id"]),
        # The live jobs' ids, oldest first: a page watching none of them
        # learns another tab's job holds the card, and one watching its own
        # learns of jobs queued behind it from elsewhere or gone since.
        active_jobs=active,
        # And what holds the card meanwhile - a job, or a critic or an Expand
        # - for an Expand waiting on it to say so.
        card_holder=card_holder,
        card_ticket=card_ticket,
        newest_image_id=newest,
        device_name=info.get("device_name"),
        vram_total_mb=info.get("vram_total_mb"),
        vram_used_mb=info.get("vram_used_mb"),
        disk_free_mb=_disk_free_mb(),
        prompt_expansion_supported=backend.prompt_expansion_supported(),
        prompt_expander=_expander_status(),
        # All three a few file checks at most - no network, no hashing - which
        # is what lets a route polled every three seconds ask them.
        prompt_critique_supported=prompt_critic.supported(),
        prompt_critique_ready=prompt_critic.ready(),
        prompt_critic=prompt_critic.describe(),
    )


def _describe_with_newest() -> tuple[dict, int | None]:
    """The backend's description and the gallery's newest id, for one status."""
    return backend.describe(), storage.newest_image_id()


def _expander_status() -> ExpanderStatus | None:
    """ERNIE's Expand before it is pressed: whether the rewriter is loaded,
    and what a first use downloads - the backend's rewriter before the
    expansion, then the translator after it - for the page to say so, each
    at its own step. A few file checks, as the critic's `describe()` is."""
    state = backend.prompt_expander_status() if backend.prompt_expansion_supported() else None
    if state is None:
        return None
    fetch = [
        ExpanderDownload(name=name, bytes=size, step="expand") for name, size in state.get("fetch", [])
    ]
    if not prompt_translator.cached():
        fetch.append(
            ExpanderDownload(name="the opus-mt-zh-en translator", bytes=TRANSLATOR_BYTES, step="translate")
        )
    return ExpanderStatus(
        loaded=bool(state.get("loaded")),
        fetch=fetch,
        fetch_bytes=sum(item.bytes for item in fetch),
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
    # Whether `refine_supported` is final: the loaded pipeline's class has
    # answered, or the profile's pipeline is one diffusers has no sibling for.
    # The Refine dialog's "a load answers for certain" is for the rest.
    defaults["refine_settled"] = (
        bool(defaults.get("refine_settled")) or info.get("refine_supported") is not None
    )
    # Whose the strength is: the profile's own, or the general default filled
    # in below for an unprofiled pipeline that has a pass. The Refine dialog
    # names it as the checkpoint's own only where it is.
    defaults["refine_strength_profiled"] = defaults.get("refine_strength") is not None
    if defaults["refine_supported"] and defaults.get("refine_strength") is None:
        from .refine import DEFAULT_STRENGTH

        defaults["refine_strength"] = DEFAULT_STRENGTH
    return defaults


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
    global _switch_error
    _switch_error = None
    if _load_state["state"] != "ready":
        # Reserve the pending load before its task has entered the lock.
        _load_state.update(state="loading", detail=None)
        asyncio.create_task(_load_in_background())
    return await status()


@app.get("/api/models")
async def list_models(
    quantization: str | None = None, format: str | None = None, offload: str | None = None,
) -> dict:
    options = model_selection.options(settings)
    if quantization is not None and quantization not in {o["value"] for o in options["quantizations"]}:
        raise HTTPException(status_code=422, detail="choose a quantization available on this server")
    if format is not None and format not in {o["value"] for o in options["formats"]}:
        raise HTTPException(status_code=422, detail="choose a supported Nunchaku format")
    if offload is not None and offload not in {o["value"] for o in options["offloads"]}:
        raise HTTPException(status_code=422, detail="choose a supported offload mode")
    current = current_settings(settings)
    models = model_selection.available(settings, quantization=quantization, format=format)
    return {
        "models": models,
        "details": {
            model["model_id"]: details_for(
                model["model_id"], quantization=quantization or current.quantization,
                offload=offload or current.offload,
            ) for model in models
        },
        "model_id": current.model_id,
        "quantization": current.quantization,
        "offload": current.offload,
        "format": model_selection.format_for(settings),
        "load_on_startup": _load_on_startup(),
        "startup_overrides": _startup_overrides,
        **options,
    }


def _load_on_startup() -> bool:
    return settings.load_on_startup if _startup_preload is None else _startup_preload


def _model_name(model_id: str) -> str:
    return _catalogue_label(model_id) or model_id.rsplit("/", 1)[-1]


async def _switch_model(target, *, load: bool, previous, previous_preload) -> None:
    """Own the whole transition even if the requesting browser disconnects."""
    global _switching, _switch_error
    revision = model_selection.revision()
    current = current_model_id(settings)
    try:
        async with _load_lock:
            # Helpers first: one that will not stop aborts the switch with the
            # running weights still loaded, and the new weights never go
            # beside a critic that may still hold the card.
            await asyncio.to_thread(prompt_critic.unload)
            await asyncio.to_thread(prompt_translator.unload)
            await asyncio.to_thread(prompt_encoding.unload)
            if not await asyncio.to_thread(backend.try_unload):
                raise RuntimeError("the weights are still loading; wait for the load")
            _load_state.update(state="unloaded", detail=None)
            model_selection.select(
                target.model_id, quantization=target.quantization, offload=target.offload,
            )
            _load_state.update(state="loading" if load else "unloaded", detail=None)
            # The selection is committed: from here on this is an ordinary
            # load, which Check, tags, LoRAs and a queued Generate wait out as
            # they do the Load button's. A second switch is still refused, by
            # the lock and then the loading state (`_refuse_busy_switch`).
            _switching = False
        # The only weight-loading path, retaining load-lock -> gate ordering.
        if load:
            await ensure_loaded()
    except LoadFailed:
        pass  # ensure_loaded already preserved the actionable failure.
    except Exception as exc:
        log.error("model switch failed: %s", short_error(exc))
        log.debug("switch traceback", exc_info=True)
        if model_selection.revision() == revision:
            # Nothing was selected: the running choice stays, saved and loaded.
            _restore_preferences(previous, previous_preload)
            reason = "the prompt critic would not stop" if isinstance(exc, CriticError) else str(exc) or short_error(exc)
            _switch_error = (
                f"Could not switch to {_model_name(target.model_id)}: {reason}; "
                f"{_model_name(current)} is still selected."
            )
        else:
            detail, kind, url = _load_failure(exc)
            _load_state.update(state="error", detail=detail, detail_kind=kind, detail_url=url)
    finally:
        _switching = False


def _restore_preferences(previous, previous_preload) -> None:
    """Put back the startup choice a switch saved before it failed."""
    global _startup_preload
    _startup_preload = previous_preload
    try:
        if previous is None:
            startup_preferences.path_for(settings.db_path).unlink(missing_ok=True)
        else:
            startup_preferences.save(settings.db_path, previous)
    except Exception as exc:
        log.error("could not restore startup settings: %s", short_error(exc))


def _refuse_busy_switch() -> None:
    """Refuse a switch while a load, a job or an editor action has the server."""
    if _load_lock.locked() or _load_state["state"] == "loading":
        raise HTTPException(status_code=409, detail="the weights are still loading; wait for the load")
    if jobs.busy():
        raise HTTPException(status_code=409, detail="jobs are queued or running; finish or cancel them before switching")
    holder, _ = jobs.card()
    if holder is not None:
        raise HTTPException(status_code=409, detail=f"{holder} is using the card; wait before switching")


@app.post("/api/model/switch", response_model=ModelStatus)
async def switch_model(request: ModelSwitchRequest, http_request: Request) -> ModelStatus:
    global _switching, _switch_task, _startup_preload, _switch_error, _switch_draining
    # Unlike model readers this route is not counted while its body arrives.
    # Another switch can finish after the middleware's revision check.
    expected = http_request.headers.get("x-local-diffusion-model-revision")
    if expected is not None and int(expected) != model_selection.revision():
        raise HTTPException(status_code=409, detail="the checkpoint changed; refresh its settings before trying again")
    try:
        target = model_selection.validate(
            settings, request.model_id, quantization=request.quantization,
            offload=request.offload, format=request.format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if _switching or _switch_draining:
        raise HTTPException(status_code=409, detail="the checkpoint is switching; wait for the switch to finish")
    _refuse_busy_switch()
    if _model_requests:
        # Counted requests already in flight - typically the page's own cold
        # reads after a load lands - are waited out briefly rather than
        # refused at once; new ones are refused meanwhile, so the wait ends.
        _switch_draining = True
        try:
            deadline = time.monotonic() + SWITCH_DRAIN_SECONDS
            while _model_requests and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
        finally:
            _switch_draining = False
        _refuse_busy_switch()
    if _model_requests:
        paths = ", ".join(sorted(_model_request_paths)) or "unknown"
        raise HTTPException(
            status_code=409,
            detail=f"another model request is still in progress ({paths}) after "
            f"{SWITCH_DRAIN_SECONDS:g} s; wait for it to finish before switching",
        )
    # Admission and reservation happen without an await, so no submission can
    # enter between the idle check and the background task's first instruction.
    _switching = True
    _switch_error = None
    previous, previous_preload = startup_preferences.load(settings.db_path), _startup_preload
    preload = _load_on_startup() if request.load_on_startup is None else request.load_on_startup
    try:
        # Save before releasing working weights: a disk error must be visible
        # to the browser, and must leave both current and previous settings intact.
        startup_preferences.save(settings.db_path,
            startup_preferences.StartupPreferences(
                model_id=target.model_id, quantization=target.quantization,
                offload=target.offload, load_on_startup=preload,
            ))
    except Exception as exc:
        _switching = False
        log.error("could not save startup settings: %s", short_error(exc))
        raise HTTPException(status_code=500, detail=f"Could not save startup settings: {short_error(exc)}") from exc
    _startup_preload = preload
    current = current_settings(settings)
    if (target.model_id, target.quantization, target.offload) == (
        current.model_id, current.quantization, current.offload,
    ):
        _switching = False
        return await load_model() if request.load else await status()
    _switch_task = asyncio.create_task(_switch_model(
        target, load=request.load, previous=previous, previous_preload=previous_preload,
    ))
    return await status()


@app.post("/api/model/unload", response_model=ModelStatus)
async def unload_model() -> ModelStatus:
    global _switch_error
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
    _switch_error = None
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


# The app's state as the route modules read it: every getter reads main's
# global when a request arrives, so a test or the fake server that rebinds
# one of them on this module is seen by every route.
services = app_context.AppServices(
    settings=lambda: settings,
    backend=lambda: backend,
    jobs=lambda: jobs,
    critic=lambda: prompt_critic,
    translator=lambda: prompt_translator,
    ensure_loaded=lambda: ensure_loaded(),
    loading=lambda: _load_lock.locked(),
    merged_defaults=lambda info: _merged_defaults(info),
)


# The routers read the app-owned worker and scope when a request arrives.
app.include_router(api_prompt.create_router(services))
app.include_router(api_generate.create_router(services))
app.include_router(api_swap.create_router(services))
app.include_router(api_refine_batch.create_router(services))
app.include_router(api_more_seeds.create_router(services))
app.include_router(api_png.create_router(services))
app.include_router(api_jobs.create_router(lambda: jobs))
app.include_router(api_bundles.create_router(lambda: bundle_manager, services))
app.include_router(api_gallery.create_router(
    lambda: jobs, services.model_scope, max_grid_images=settings.max_grid_images,
))


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
        jobs.stop_all("shutdown")
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
