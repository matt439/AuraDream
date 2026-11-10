"""A refusal with a stable `code` beside its `detail`, for the few the page
acts on (`refusal_codes.py` lists them and what each means).

`Refused` is an `HTTPException`, so a route raises it as it raises any
refusal; `main` registers `refused_response`, which answers
`{"detail": ..., "code": ...}` and, where the refusal names gallery images,
`"ids": [...]`. The sentence stays the person's; only the code and the ids
are read by the page.
"""
from __future__ import annotations

from collections.abc import Iterable

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .refusal_codes import CODES


class Refused(HTTPException):
    def __init__(self, status_code: int, detail: str, code: str, ids: Iterable[int] = ()) -> None:
        super().__init__(status_code=status_code, detail=detail)
        assert code in CODES, code
        self.code = code
        self.ids = tuple(ids)

    def body(self) -> dict:
        body: dict = {"detail": self.detail, "code": self.code}
        if self.ids:
            body["ids"] = list(self.ids)
        return body


def refusal(status_code: int, detail: str, source: object = None) -> HTTPException:
    """The refusal to raise for `detail`: coded where `source` - the error
    or finding it was made from - carries a code, plain otherwise."""
    code = getattr(source, "code", None)
    if isinstance(code, str) and code in CODES:
        return Refused(status_code, detail, code, getattr(source, "ids", ()) or ())
    return HTTPException(status_code=status_code, detail=detail)


async def refused_response(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, Refused)
    return JSONResponse(exc.body(), status_code=exc.status_code, headers=exc.headers)
