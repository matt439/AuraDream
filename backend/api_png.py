"""Load settings from PNG: a PNG's text chunks, read by the page, as the
panel's settings (`POST /api/png/settings`).

The page reads a dropped or picked PNG's tEXt, zTXt and iTXt chunks and
sends only those; the image is never sent, stored or made a reference. This
app's record carries the whole reproduction set, as `--from-image` reads it
(`png_settings`); an A1111/Forge/Civitai 'parameters' block carries the
prompt, the negative and the size, and says which of its other settings do
not carry over; there the panel's own sampling, which the page sends, is
what runs. Either way the answer has gone through `applied()` on the
running checkpoint, and its `notes` say what did not carry.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, TypedDict

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from . import png_settings, storage
from .app_context import AppServices
from .execution import require_configured_loader_available, require_default_execution, resolve_execution
from .metadata import PNG_METADATA_KEY
from .model_selection import current_settings
from .png_settings import PARAMETERS_KEY, PngSettingsError
from .schemas import GenerateRequest, PngPanelSampling, PngSettings, PngSettingsRequest, error_detail

# The whole body, every chunk's text with its keyword. This app's record is a
# few kilobytes, an A1111 block less; a ComfyUI workflow - read only to be
# named in the refusal - is rarely past a few hundred.
MAX_BODY = 1 << 20

_SIZE = re.compile(r"^(\d+)x(\d+)$")

# The A1111 settings that make the picture with its sampler and checkpoint,
# under the names the line says them by. None of them has a meaning here:
# another sampler on another checkpoint does not make the same image from the
# same steps, CFG and seed, and a mapping from one to the other would be one
# this app invented.
_NOT_CARRIED = (
    ("Steps", "steps"),
    ("Sampler", "sampler"),
    ("Schedule type", "schedule type"),
    ("CFG scale", "CFG scale"),
    ("Seed", "seed"),
    ("Model", "checkpoint"),
)
# Read for the panel (Size) or named above together with what it identifies.
_READ = {"Size", "Model hash", *(key for key, _ in _NOT_CARRIED)}
# How many of an A1111 block's other keys the note names before "and N more".
_OTHERS_NAMED = 6


def _series(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _workflow(chunks: dict[str, str]) -> bool:
    """ComfyUI's own chunks: its graph under 'workflow' and 'prompt', as JSON."""
    for key in ("workflow", "prompt"):
        try:
            if isinstance(json.loads(chunks.get(key) or "null"), dict):
                return True
        except (ValueError, RecursionError):
            continue
    return False


def _verified_references(meta: dict) -> tuple[list[int], int]:
    """The record's references this gallery holds as the same picture.

    The ids are the gallery's that made the image - another one's, for a
    file from WSL's gallery or a backup - so an id alone could name any
    picture here. One carries only where this gallery's live row under it
    has the decoded-pixel hash and size the record's `reference_sources`
    says it had (`storage.image_provenance`); the count of the rest is for
    the note.
    """
    ids = meta.get("reference") or []
    sources = meta.get("reference_sources") or []
    if not isinstance(ids, list):
        return [], 0
    kept: list[int] = []
    for at, image_id in enumerate(ids):
        source = sources[at] if isinstance(sources, list) and at < len(sources) else None
        if not isinstance(image_id, int) or not isinstance(source, dict):
            continue
        record = storage.get_image(image_id)
        if record is not None and record.content_hash and (
            (record.content_hash, record.width, record.height)
            == (source.get("hash"), source.get("width"), source.get("height"))
        ):
            kept.append(image_id)
    return kept, len(ids) - len(kept)


class _ImageSize(TypedDict, total=False):
    width: int
    height: int


def _size(text: Any) -> _ImageSize:
    match = _SIZE.match(text) if isinstance(text, str) else None
    try:
        return {"width": int(match[1]), "height": int(match[2])} if match else {}
    except ValueError:
        # Decimal conversion can refuse even a digit-only dimension. Treat
        # it as unreadable without changing the interpreter's input limit.
        return {}


def _reason(exc: ValueError) -> str:
    """`error_detail` without pydantic's "Value error, " and the bare "request:"
    a model validator's message comes under, for a sentence of the note."""
    return "; ".join(
        part.removeprefix("request: ").replace("Value error, ", "")
        for part in error_detail(exc).split("; ")
    )


def _request(
    fields: dict[str, Any], size: _ImageSize, notes: list[str], name: str, at: int
) -> tuple[GenerateRequest, bool]:
    """The request, with the PNG's size where it is one a direct render takes.

    A size that is not - past the largest preset, off the 16-pixel grid,
    over the budget - leaves the panel's size, with the reason, as note `at`
    (beside the sentence that says the size was read); any other refusal is
    the whole answer's.
    """
    if size:
        assert "width" in size and "height" in size, "a parsed PNG size carries both dimensions"
        try:
            request = GenerateRequest(**fields, **size)
            request.check_budget()
            return request, True
        except ValueError as exc:
            notes.insert(
                at,
                f"Its size {size['width']}x{size['height']} is not carried ({_reason(exc)}); "
                "the panel keeps its own."
            )
    try:
        return GenerateRequest(**fields), False
    except ValueError as exc:
        raise PngSettingsError(f"the settings in {name} do not fit this app: {_reason(exc)}") from None


def _from_record(meta: dict, name: str) -> tuple[dict, _ImageSize, bool, list[str]]:
    """This app's record as request fields, the size, whether to write it, notes."""
    notes: list[str] = []
    if not isinstance(meta.get("prompt"), str) or not meta["prompt"].strip():
        raise PngSettingsError(f"the AuraDream metadata in {name} carries no prompt")
    fields: dict[str, Any] = {
        key: meta[field]
        for key, field in (
            ("prompt", "prompt"), ("negative_prompt", "negative_prompt"), ("steps", "steps"),
            ("guidance_scale", "guidance_scale"), ("true_cfg", "true_cfg"), ("seed", "seed"),
            ("lora", "lora"), ("lora_scale", "lora_scale"),
            ("execution", "execution"),
        )
        if meta.get(field) is not None
    }
    # An AuraDream PNG is replay metadata. Older Eager images must not
    # acquire current loader intent when the browser submits these settings.
    fields["execution"] = meta.get("execution")
    # An explicit null is a render with no shift of ours, as `--from-image`
    # reads it; absent, the profile's.
    if "mu" in meta:
        fields["mu"] = meta["mu"]
    references, left_out = _verified_references(meta)
    if references:
        fields["reference"] = references
    if left_out:
        one = left_out == 1
        notes.append(
            f"{'One' if one else left_out} of its reference images {'is' if one else 'are'} left out: "
            f"this gallery holds no copy of {'it' if one else 'them'} under the id "
            f"{'it' if one else 'they'} had where the image was made."
        )
    size: _ImageSize = _size(meta.get("size"))
    # A refine pass is the Refine dialog's, made from a source in a gallery;
    # from a file only its prompt and sampling carry, at the panel's own size
    # - as Reuse fills the panel when a pass's source has gone.
    # An inpaint likewise: its mask carries nowhere, since the panel has no
    # source for it to cover, and only the Inpaint dialog sends one.
    if meta.get("refine") is not None:
        size = {}
        what = (
            "an outpaint" if meta.get("outpaint") is not None
            else "an inpaint" if meta.get("mask") is not None else "a refine pass"
        )
        notes.append(
            f"It is {what}, which is made again from its source in the gallery, not from a file: "
            "its prompt and settings fill the panel as a direct render, at the panel's own size."
        )
    elif "size" in meta and not size:
        notes.append("Its size could not be read, so the panel keeps its own.")
    if meta.get("upscaler") is not None:
        notes.append("Its upscaler is left out: it prepares a gallery source for an ordinary Refine pass.")
    if meta.get("crop_anchor") not in (None, "center"):
        notes.append("Its crop anchor is left out: it crops a gallery source for an ordinary Refine pass.")
    return fields, size, True, notes


def _panel_sampling(fields: dict[str, Any], panel: PngPanelSampling | None) -> dict[str, Any]:
    """The panel's steps, CFG, true CFG and Mu as it sent them, or none.

    What Generate will run where another tool's sampling does not carry, so
    `applied()` judges the negative prompt against the CFG the panel holds,
    not the profile's. Values Generate itself would refuse are left to the
    profile, as a blank box is: the file is not refused for the panel.
    """
    sent = panel.model_dump(exclude_unset=True) if panel is not None else {}
    try:
        GenerateRequest(**fields, **sent)
    except ValueError:
        return {}
    return sent


def _from_parameters(
    text: str, name: str, panel: PngPanelSampling | None
) -> tuple[dict, _ImageSize, bool, list[str]]:
    """An A1111 'parameters' block as request fields, the size, notes."""
    prompt, negative, params = png_settings.parse_parameters(text)
    if not prompt:
        raise PngSettingsError(f"the 'parameters' text in {name} carries no prompt")
    fields: dict[str, Any] = {"prompt": prompt}
    if negative:
        fields["negative_prompt"] = negative
    fields.update(_panel_sampling(fields, panel))
    size = _size(params.get("Size"))
    read = ["the prompt"] + (["negative prompt"] if negative else []) + (["size"] if size else [])
    notes = [f"Read from its A1111-style 'parameters' text: {_series(read)}."]
    if not size:
        notes.append(
            "Its size could not be read, so the panel keeps its own."
            if "Size" in params else "It names no size, so the panel keeps its own."
        )
    held = [f"{label} {params[key]}" for key, label in _NOT_CARRIED if params.get(key)]
    if held:
        notes.append(
            f"Its {_series(held)} do not carry over: they ran with another tool's sampler and checkpoint, "
            "and here the same numbers would not make the same picture. The panel keeps its own steps, "
            "CFG and seed."
        )
    others = [key for key in params if key not in _READ]
    if others:
        more = len(others) - _OTHERS_NAMED
        notes.append(
            f"Not read: {', '.join(others[:_OTHERS_NAMED])}"
            + (f" and {more} more" if more > 0 else "") + "."
        )
    return fields, size, False, notes


def read_settings(services: AppServices, asked: PngSettingsRequest) -> PngSettings:
    """The answer for one PNG's chunks; `PngSettingsError` is a refusal in words."""
    name, chunks = asked.name, asked.chunks
    meta = png_settings.record(chunks, name)
    if meta is not None:
        try:
            require_default_execution(meta.get("execution"))
            selected = current_settings(services.settings())
            require_configured_loader_available(selected)
            resolve_execution(selected.loader_execution, meta.get("execution"))
        except ValueError as exc:
            raise PngSettingsError(str(exc)) from exc
        source = PNG_METADATA_KEY
        fields, size, sampling, notes = _from_record(meta, name)
        model = meta.get("model") if isinstance(meta.get("model"), str) else None
    elif chunks.get(PARAMETERS_KEY, "").strip():
        source = PARAMETERS_KEY
        fields, size, sampling, notes = _from_parameters(chunks[PARAMETERS_KEY], name, asked.panel)
        model = None
    elif _workflow(chunks):
        raise PngSettingsError(
            f"{name} carries a ComfyUI workflow, not settings this app reads: only AuraDream's "
            "own record and an A1111-style 'parameters' text are"
        )
    else:
        found = f" (its text: {', '.join(list(chunks)[:_OTHERS_NAMED])})" if chunks else ", nor any text at all"
        raise PngSettingsError(
            f"{name} carries no settings this app reads{found}: neither AuraDream's own record "
            "nor an A1111-style 'parameters' text"
        )
    at = 1 if source == PARAMETERS_KEY else len(notes)
    request, size_carried = _request(fields, size, notes, name, at)
    ran = services.backend().applied(request)
    if request.reference and not ran.reference:
        one = len(request.reference) == 1
        notes.append(
            f"Its reference image{'' if one else 's'} {'is' if one else 'are'} not used: this checkpoint "
            "takes none, so this renders from the prompt alone."
        )
    if request.lora and not ran.lora:
        # The installation's reason; one adapter that does not fit, or is not
        # in the directory, is the page's to say, from `/api/loras`, as
        # after any Reuse.
        reason = services.backend().lora_unavailable_reason() or "this installation cannot load adapters"
        notes.append(f"Its adapter {request.lora} @ {request.lora_scale:g} is not applied here - {reason}.")
    negative = request.negative_prompt
    if negative and not ran.negative_prompt:
        notes.append(
            "Its negative prompt does not run here at these settings; the box keeps it, "
            "greyed, as it keeps a typed one."
        )
    return PngSettings(
        source=source, name=name, model_id=model, settings=ran, negative_prompt=negative,
        size=size_carried, sampling=sampling, notes=notes,
    )


async def _capped_body(request: Request) -> bytes:
    """The body, refused past `MAX_BODY` before or while it arrives."""
    refusal = HTTPException(
        status_code=413,
        detail=f"The PNG's text is over {MAX_BODY // (1 << 20)} MiB; a settings block is a few kilobytes, "
        "so this one is not read.",
    )
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_BODY:
        raise refusal
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY:
            raise refusal
    return bytes(body)


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter()

    @router.post("/api/png/settings", response_model=PngSettings)
    async def png_settings_route(request: Request) -> PngSettings:
        try:
            asked = PngSettingsRequest.model_validate_json(await _capped_body(request))
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=error_detail(exc)) from exc
        try:
            # Off the loop: a carried reference is checked against the gallery.
            return await asyncio.to_thread(read_settings, services, asked)
        except PngSettingsError as exc:
            # A sentence, opening with the file's name as it is spelled.
            message = str(exc)
            if not message.startswith(asked.name):
                message = message[:1].upper() + message[1:]
            raise HTTPException(status_code=422, detail=message + ".") from exc

    return router
