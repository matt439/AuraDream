"""Cold saved-setup CRUD and explicit validation; no submission or restore."""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from . import model_selection, presets
from .app_context import AppServices
from .execution import require_configured_loader_available, require_default_execution, resolve_execution
from .preset_validation import PresetValidation, validate_payload
from .schemas_generate import error_detail
from .schemas_presets import (
    Preset, PresetCheck, PresetCollection, PresetCreate, PresetFields,
    PresetSave, PresetUpdate,
)

_ITEM = r"/api/presets/[a-f0-9]{32}"


def depends_on_model(method: str, path: str) -> bool:
    """Only capture/validation read the selection; inspect/delete stay cold."""
    return (
        method == "POST" and (path == "/api/presets" or re.fullmatch(_ITEM + r"/validate", path) is not None)
    ) or (method == "PUT" and re.fullmatch(_ITEM, path) is not None)


class PresetChecked(PresetValidation):
    item_id: str
    revision: int
    model_id: str
    model_revision: int


class _PresetRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def validated(request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # Never echo an invalid saved prompt or malformed Unicode.
                detail = "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in exc.errors()
                )
                raise HTTPException(status_code=422, detail=detail) from exc
            except presets.PresetsError as exc:
                raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

        return validated


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter(route_class=_PresetRoute)

    def read_item(item_id: str) -> Preset:
        item = next((item for item in presets.read(services.settings().db_path) if item.id == item_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="That setup is no longer in the saved list. Reload the list.")
        return item

    def check_selection(model_id: str, revision: int, step_cache: str | None) -> None:
        selected = model_selection.current_settings(services.settings())
        if model_id != selected.model_id:
            raise HTTPException(
                status_code=409,
                detail=f"This setup belongs to {model_id}. Select that checkpoint in Models before trying again.",
            )
        if revision != model_selection.revision():
            raise HTTPException(status_code=409, detail="The checkpoint changed. Refresh its settings before trying again.")
        try:
            require_configured_loader_available(selected)
            require_default_execution(resolve_execution(selected.loader_execution, step_cache=step_cache))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=error_detail(exc)) from exc

    def validate(fields: PresetFields, revision: int, step_cache: str | None) -> PresetValidation:
        check_selection(fields.model_id, revision, step_cache)
        try:
            return validate_payload(fields.payload, fields.model_id, applier=services.backend().applied)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=error_detail(exc)) from exc

    @router.get("/api/presets", response_model=PresetCollection)
    def list_presets():
        return PresetCollection(items=presets.read(services.settings().db_path))

    @router.get("/api/presets/{item_id}", response_model=Preset)
    def get_preset(item_id: str = Path(pattern=r"^[a-f0-9]{32}$")):
        return read_item(item_id)

    @router.post("/api/presets", response_model=PresetCollection, status_code=201)
    def add_preset(body: PresetCreate):
        fields = PresetFields.model_validate(body.model_dump(exclude={"model_revision", "step_cache"}))
        validate(fields, body.model_revision, body.step_cache)
        return PresetCollection(items=presets.change(services.settings().db_path, fields))

    @router.put("/api/presets/{item_id}", response_model=PresetCollection)
    def update_preset(body: PresetSave, item_id: str = Path(pattern=r"^[a-f0-9]{32}$")):
        fields = PresetUpdate.model_validate(body.model_dump(exclude={"model_revision", "step_cache"}))
        item = read_item(item_id)
        if fields.model_id != item.model_id:
            raise HTTPException(status_code=409, detail="A saved setup keeps its checkpoint. Save a new setup instead.")
        validate(fields, body.model_revision, body.step_cache)
        return PresetCollection(items=presets.change(services.settings().db_path, fields, item_id=item_id))

    @router.delete("/api/presets/{item_id}", response_model=PresetCollection)
    def delete_preset(item_id: str = Path(pattern=r"^[a-f0-9]{32}$"), revision: int = Query(ge=1)):
        return PresetCollection(items=presets.change(services.settings().db_path, None, item_id=item_id, revision=revision))

    @router.post("/api/presets/{item_id}/validate", response_model=PresetChecked)
    def validate_preset(body: PresetCheck, item_id: str = Path(pattern=r"^[a-f0-9]{32}$")):
        item = read_item(item_id)
        if item.revision != body.revision:
            raise HTTPException(status_code=409, detail="That setup changed in another tab. Reload the list before restoring.")
        result = validate(item, body.model_revision, body.step_cache)
        # Validation may take time in another thread. Never certify an item
        # that another process edited/deleted during the calculation.
        if read_item(item_id).revision != item.revision:
            raise HTTPException(status_code=409, detail="That setup changed during validation. Reload the list before restoring.")
        return PresetChecked(
            **result.model_dump(), item_id=item_id, revision=item.revision,
            model_id=item.model_id, model_revision=body.model_revision,
        )

    return router
