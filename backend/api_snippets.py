"""Model-independent personal prompt snippets; no gallery or GPU work."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from . import snippets
from .app_context import AppServices
from .schemas_snippets import SnippetCollection, SnippetFields, SnippetUpdate


class _SnippetRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def validated(request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # Report only the validation messages, not arbitrary saved
                # prompt text or malformed Unicode echoed from the request.
                detail = "; ".join(error["msg"] for error in exc.errors())
                raise HTTPException(status_code=422, detail=detail) from exc

        return validated


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

    @router.put("/api/snippets/{item_id}", response_model=SnippetCollection)
    def update_snippet(body: SnippetUpdate, item_id: str = Path(pattern=r"^[a-f0-9]{32}$")):
        return read_or_change(body, item_id=item_id)

    @router.delete("/api/snippets/{item_id}", response_model=SnippetCollection)
    def delete_snippet(
        item_id: str = Path(pattern=r"^[a-f0-9]{32}$"), revision: int = Query(ge=1),
    ):
        return read_or_change(item_id=item_id, revision=revision)

    return router
