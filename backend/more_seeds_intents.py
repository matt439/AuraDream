"""Bounded, server-issued receipts for one More seeds submission intent.

An unknown token always refuses. Expiring a receipt can lose its convenience,
but can never turn a retry into a fresh seed allocation. This is deliberately
local to a JobManager lifetime; it is not persistent job recovery.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


class IntentProblem(ValueError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class _Intent:
    payload: str
    context: tuple
    expires: float
    task: asyncio.Task[dict[str, Any]] | None = None


def _canonical(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class IntentLedger:
    def __init__(
        self, *, clock: Callable[[], float] = time.monotonic, capacity: int = 512,
        issued_seconds: float = 900, receipt_seconds: float = 3600,
    ) -> None:
        if capacity < 1 or issued_seconds <= 0 or receipt_seconds <= 0:
            raise ValueError("intent limits must be positive")
        self._clock = clock
        self._capacity = capacity
        self._issued_seconds = issued_seconds
        self._receipt_seconds = receipt_seconds
        self._entries: dict[str, _Intent] = {}
        self._closed = False

    @property
    def pending(self) -> bool:
        return any(entry.task is not None and not entry.task.done() for entry in self._entries.values())

    def _prune(self) -> None:
        now = self._clock()
        for token, entry in list(self._entries.items()):
            running = entry.task is not None and not entry.task.done()
            if not running and entry.expires <= now:
                del self._entries[token]

    def issue(self, payload: dict, context: tuple) -> str:
        if self._closed:
            raise IntentProblem(503, "the server is stopping; no More seeds request was issued")
        self._prune()
        if len(self._entries) >= self._capacity:
            # An unused preview has admitted nothing. Receipts and operations
            # retain their slots; a full ledger refuses before doing any work.
            unused = next((key for key, entry in self._entries.items() if entry.task is None), None)
            if unused is None:
                raise IntentProblem(503, "too many More seeds receipts are being kept; try again later")
            del self._entries[unused]
        token = uuid.uuid4().hex
        self._entries[token] = _Intent(
            payload=_canonical(payload), context=context,
            expires=self._clock() + self._issued_seconds,
        )
        return token

    async def execute(
        self, token: str, payload: dict, context: tuple,
        operation: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        self._prune()
        entry = self._entries.get(token)
        if entry is None:
            raise IntentProblem(409, "this More seeds receipt expired or belongs to another server; review the batches again")
        if entry.payload != _canonical(payload) or entry.context != context:
            raise IntentProblem(409, "this More seeds receipt does not match the request or checkpoint; review the batches again")
        if entry.task is None:
            if self._closed:
                raise IntentProblem(503, "the server is stopping; no More seeds work was queued")
            # This lookup and task registration have no await between them:
            # concurrent requests for one token always join the same operation.
            async def run() -> dict[str, Any]:
                try:
                    return await operation()
                finally:
                    # Publish the receipt lifetime before the task becomes
                    # done. A ready replay can run before its done callbacks.
                    entry.expires = self._clock() + self._receipt_seconds

            entry.task = asyncio.create_task(run(), name="more-seeds-intent")

            def settled(task: asyncio.Task) -> None:
                if not task.cancelled():
                    task.exception()  # A disconnected caller must not orphan an exception.

            entry.task.add_done_callback(settled)
        return copy.deepcopy(await self._settled(entry.task))

    @staticmethod
    async def _settled(task: asyncio.Task) -> Any:
        """A cancelled waiter keeps its reservations until admission settles.

        Shield alone would release the request's model/gallery accounting
        while its operation was still awaiting a mask or persistence write.
        The owned task also remains visible through ``pending`` meanwhile.
        """
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
            except Exception:
                if cancelled:
                    raise asyncio.CancelledError from None
                raise
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def close(self) -> None:
        self._closed = True
        tasks = [entry.task for entry in self._entries.values() if entry.task is not None and not entry.task.done()]
        if tasks:
            await asyncio.gather(*(self._settled(task) for task in tasks), return_exceptions=True)
