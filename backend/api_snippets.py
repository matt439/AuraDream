"""Model-independent personal prompt snippets; no gallery or GPU work."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import Response
from fastapi.exceptions import RequestValidationError

from . import snippet_portability, snippets
from .schemas_snippet_portability import (
    MAX_PORTABLE_BYTES, SnippetImportCommit, SnippetImportPreview, SnippetImportRequest,
)
from .app_context import AppServices
from .bounded_route import bounded_json_route
from .schemas_snippets import SnippetCollection, SnippetFields, SnippetUpdate


MAX_IMPORT_REQUEST_BYTES = MAX_PORTABLE_BYTES + 64 * 1024
_IMPORT_PATHS = {"/api/snippets/import", "/api/snippets/import/preview"}


def _translate(request, exc: Exception) -> None:
    """Snippet and portability errors in their own words; a validation
    error as its messages alone."""
    if isinstance(exc, (snippets.SnippetsError, snippet_portability.PortabilityError)):
        raise HTTPException(status_code=exc.status, detail=str(exc)) from None
    if isinstance(exc, RequestValidationError):
        if request.url.path in _IMPORT_PATHS:
            raise HTTPException(status_code=422, detail="The snippet import fields are invalid.") from None
        # Report only the validation messages, not arbitrary saved
        # prompt text or malformed Unicode echoed from the request.
        detail = "; ".join(error["msg"] for error in exc.errors())
        raise HTTPException(status_code=422, detail=detail) from exc


_SnippetRoute = bounded_json_route(
    import_paths=_IMPORT_PATHS, max_bytes=lambda: MAX_IMPORT_REQUEST_BYTES,
    too_large="The snippet import request is too large.",
    parse=lambda body, max_bytes: snippet_portability.parse_json(body, max_bytes=max_bytes),
    translate=_translate,
)


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter(route_class=_SnippetRoute)

    def read_or_change(values=None, **kwargs) -> SnippetCollection:
        try:
            database = services.settings().db_path
            items = snippets.change(database, values, **kwargs) if kwargs or values is not None else snippets.read(database)
            return SnippetCollection(items=items)
        except snippets.SnippetsError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    @router.get("/api/snippets", response_model=SnippetCollection)
    def list_snippets():
        return read_or_change()

    @router.post("/api/snippets", response_model=SnippetCollection, status_code=201)
    def add_snippet(body: SnippetFields):
        return read_or_change(body)

    @router.get("/api/snippets/export")
    def export_snippets():
        data = snippet_portability.export_portable(snippets.read(services.settings().db_path))
        return Response(
            content=data, media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="auradream-snippets.json"',
                     "Cache-Control": "no-store"},
        )

    @router.post("/api/snippets/import/preview", response_model=SnippetImportPreview)
    def preview_snippets(body: SnippetImportRequest):
        return snippet_portability.preview(
            snippets.read(services.settings().db_path), body.payload, body.decisions,
        )

    @router.post("/api/snippets/import", response_model=SnippetCollection)
    def import_snippets(body: SnippetImportCommit):
        return SnippetCollection(items=snippets.import_collection(services.settings().db_path, body))

    @router.put("/api/snippets/{item_id}", response_model=SnippetCollection)
    def update_snippet(body: SnippetUpdate, item_id: str = Path(pattern=r"^[a-f0-9]{32}$")):
        return read_or_change(body, item_id=item_id)

    @router.delete("/api/snippets/{item_id}", response_model=SnippetCollection)
    def delete_snippet(
        item_id: str = Path(pattern=r"^[a-f0-9]{32}$"), revision: int = Query(ge=1),
    ):
        return read_or_change(item_id=item_id, revision=revision)

    return router
