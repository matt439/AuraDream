"""Cold CRUD for saved gallery filters; never enters gallery or model admission."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse

from . import saved_searches
from .app_context import AppServices
from .schemas_saved_searches import SavedSearchCollection, SavedSearchFields, SavedSearchUpdate


class _SavedSearchRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def validated(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # Pydantic errors can retain arbitrary query text, field names
                # and invalid Unicode. Keep this public response fixed.
                raise HTTPException(
                    status_code=422,
                    detail="Saved search fields are invalid. Check the name, query, ticks and revision.",
                ) from None
            except saved_searches.SavedSearchesError as exc:
                if exc.conflict_id is not None:
                    return JSONResponse(status_code=exc.status, content={"detail": str(exc), "conflict_id": exc.conflict_id})
                raise HTTPException(status_code=exc.status, detail=str(exc)) from None

        return validated


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter(route_class=_SavedSearchRoute)

    @router.get("/api/gallery/saved-searches", response_model=SavedSearchCollection)
    def list_searches():
        return SavedSearchCollection(items=saved_searches.read(services.settings().db_path))

    @router.post("/api/gallery/saved-searches", response_model=SavedSearchCollection, status_code=201)
    def add_search(body: SavedSearchFields):
        return SavedSearchCollection(items=saved_searches.change(services.settings().db_path, body))

    @router.put("/api/gallery/saved-searches/{item_id}", response_model=SavedSearchCollection)
    def update_search(body: SavedSearchUpdate, item_id: str = Path(pattern=r"^[a-f0-9]{32}$")):
        return SavedSearchCollection(items=saved_searches.change(services.settings().db_path, body, item_id=item_id))

    @router.delete("/api/gallery/saved-searches/{item_id}", response_model=SavedSearchCollection)
    def delete_search(item_id: str = Path(pattern=r"^[a-f0-9]{32}$"), revision: int = Query(ge=1)):
        return SavedSearchCollection(items=saved_searches.change(services.settings().db_path, item_id=item_id, revision=revision))

    return router
