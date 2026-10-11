"""Run a blocking callable on a daemon thread and await it.

The model load (`main.ensure_loaded`) and the prompt routes' cold work
(`api_prompt`) use it; the thread is never joined at interpreter exit.
"""
from __future__ import annotations

import asyncio
import threading


async def detached(fn, name: str = "load"):
    """Run `fn` on a daemon thread and await its result.

    Not `asyncio.to_thread`: that pool's threads are joined at interpreter
    exit, so a Ctrl-C four minutes into a 32 GB download waited the download
    out before the process could end. A daemon thread dies with the process,
    and the Hub's downloader resumes the partial file on the next start.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def settle(setter, value) -> None:
        if not future.done():
            setter(value)

    def run() -> None:
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the loop
            outcome = (future.set_exception, exc)
        else:
            outcome = (future.set_result, result)
        try:
            loop.call_soon_threadsafe(settle, *outcome)
        except RuntimeError:  # the loop is gone: the process is exiting
            pass

    threading.Thread(target=run, daemon=True, name=name).start()
    return await future
