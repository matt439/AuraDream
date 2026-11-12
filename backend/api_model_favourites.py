"""Model favourites' routes: model-free, independent of gallery maintenance."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from . import model_favourites
from .app_context import AppServices
from .schemas_model_favourites import ModelFavouriteChange, ModelFavouriteImport, ModelFavourites

PREFIX = "/api/models/favourites"
PATHS = frozenset({PREFIX, PREFIX + "/import"})


class _FavouritesRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def validated(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # Pydantic's errors can echo the sent ids; keep the answer fixed.
                raise HTTPException(
                    status_code=422,
                    detail="Model favourite fields are invalid. Check the model ids and the star.",
                ) from None
            except model_favourites.ModelFavouritesError as exc:
                raise HTTPException(status_code=exc.status, detail=str(exc)) from None

        return validated


def create_router(services: AppServices) -> APIRouter:
    router = APIRouter(route_class=_FavouritesRoute)

    @router.get(PREFIX, response_model=ModelFavourites)
    def list_favourites():
        items, revision = model_favourites.read(services.settings().db_path)
        return ModelFavourites(items=items, revision=revision)

    @router.post(PREFIX, response_model=ModelFavourites)
    def change_favourite(body: ModelFavouriteChange):
        items, revision = model_favourites.change(services.settings().db_path, body.model_id, body.favourite)
        return ModelFavourites(items=items, revision=revision)

    @router.post(PREFIX + "/import", response_model=ModelFavourites)
    def import_favourites(body: ModelFavouriteImport):
        items, revision, added = model_favourites.merge(services.settings().db_path, body.model_ids)
        return ModelFavourites(items=items, revision=revision, imported=added)

    return router
