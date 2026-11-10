"""Default-off idle model release on the server's activity clock; no GPU imports."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)


class IdleUnload:
    def __init__(self, *, minutes: int, activity_age: Callable[[], float],
                 busy: Callable[[], bool], ready: Callable[[], bool],
                 unload: Callable[[], Awaitable[None]],
                 monotonic: Callable[[], float] = time.monotonic,
                 interval: float = 15.0) -> None:
        self.minutes = minutes
        self.activity_age = activity_age
        self.busy = busy
        self.ready = ready
        self.unload = unload
        self.monotonic = monotonic
        self.interval = interval
        self._quiet_since = monotonic()
        self._was_busy = True
        self._was_ready = False
        self._failed = False
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def check(self) -> bool:
        """One event-loop check; a busy interval gets its own full idle grace."""
        if self.minutes <= 0 or self._stop.is_set():
            return False
        now = self.monotonic()
        ready, busy = self.ready(), self.busy()
        if busy or self._was_busy or (ready and not self._was_ready):
            self._quiet_since = now
        self._was_busy, self._was_ready = busy, ready
        if not ready or busy:
            return False
        if min(self.activity_age(), now - self._quiet_since) < self.minutes * 60:
            return False
        await self.unload()
        self._was_ready = False
        return True

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="idle-unload")

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.check()
                self._failed = False
            except Exception as exc:  # noqa: BLE001 - a later idle tick retries
                if not self._failed:
                    log.warning("Idle unload could not finish (%s). Will try again when idle.", type(exc).__name__)
                    self._failed = True
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval)
            except TimeoutError:
                pass

    async def stop(self) -> None:
        """Wake the timer and drain an admitted unload, never cancel its worker."""
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
