"""Per-process lease exclusion and complete ASGI FileResponse lifetime."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.background import BackgroundTask

from backend.bundle_leases import LeaseBusy, LeaseRegistry, LeasedFileResponse


def test_readers_share_resolved_paths_and_exclude_claims_until_every_release(tmp_path):
    registry = LeaseRegistry()
    path = tmp_path / "copy.zip"
    one = registry.reader(path)
    two = registry.reader(tmp_path / "unused" / ".." / "copy.zip")
    assert registry.leased(path) and len(registry.snapshot()) == 1
    with pytest.raises(LeaseBusy):
        registry.exclusive(path)
    one.release()
    one.release()
    assert registry.leased(path)
    with ThreadPoolExecutor(1) as pool:
        pool.submit(two.release).result(5)
    assert not registry.leased(path) and registry.snapshot() == frozenset()
    with registry.exclusive(path):
        with pytest.raises(LeaseBusy):
            registry.reader(path)
        with pytest.raises(LeaseBusy):
            registry.exclusive(path)
        with registry.reader(tmp_path / "other.zip"):
            assert registry.leased(tmp_path / "other.zip")
    with registry.reader(path):
        assert registry.leased(path)


def test_context_errors_release_and_snapshot_is_only_a_snapshot(tmp_path):
    registry = LeaseRegistry()
    path = tmp_path / "copy.zip"
    with pytest.raises(ValueError):
        with registry.exclusive(path):
            raise ValueError("stop")
    before = registry.snapshot()
    with registry.reader(path):
        assert not before
        with pytest.raises(LeaseBusy):
            registry.exclusive(path)
    assert registry.snapshot() == before


def scope(*, headers=(), method="GET", extensions=None):
    return {"type": "http", "method": method, "headers": list(headers),
            "path": "/copy.zip", "extensions": extensions or {}}


async def receive():
    return {"type": "http.disconnect"}


@pytest.mark.parametrize("headers, status", [
    ((), 200), (([b"range", b"bytes=1-3"],), 206),
    (([b"range", b"bytes=0-1,4-5"],), 206),
    (([b"range", b"invalid"],), 400), (([b"range", b"bytes=90-100"],), 416),
])
def test_response_preserves_ranges_errors_and_releases(tmp_path, headers, status):
    async def scenario():
        path = tmp_path / "copy.zip"
        path.write_bytes(b"0123456789")
        registry = LeaseRegistry()
        response = LeasedFileResponse(path, media_type="application/zip", filename=path.name)
        response.registry = registry
        messages = []

        async def send(message):
            assert registry.leased(path)
            messages.append(message)

        await response(scope(headers=headers), receive, send)
        assert messages[0]["status"] == status
        assert not registry.leased(path)
        body = b"".join(message.get("body", b"") for message in messages)
        if status == 200:
            assert body == b"0123456789"
        elif headers == ([b"range", b"bytes=1-3"],):
            assert body == b"123"

    asyncio.run(scenario())


def test_response_holds_through_stream_and_background_and_disables_handoff(tmp_path):
    async def scenario():
        path = tmp_path / "copy.zip"
        path.write_bytes(b"body")
        registry = LeaseRegistry()
        started, release = asyncio.Event(), asyncio.Event()
        background = []
        messages = []

        async def done():
            assert registry.leased(path)
            background.append(True)

        response = LeasedFileResponse(path, background=BackgroundTask(done))
        response.registry = registry

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.start":
                started.set()
                await release.wait()

        original = scope(extensions={"http.response.pathsend": {}, "other.extension": {}})
        task = asyncio.create_task(response(original, receive, send))
        await asyncio.wait_for(started.wait(), 5)
        with pytest.raises(LeaseBusy):
            registry.exclusive(path)
        release.set()
        await asyncio.wait_for(task, 5)
        assert background == [True] and not registry.leased(path)
        assert not any(message["type"] == "http.response.pathsend" for message in messages)
        assert b"".join(message.get("body", b"") for message in messages) == b"body"
        assert "http.response.pathsend" in original["extensions"]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["cancel", "send", "background", "missing"])
def test_response_releases_on_cancellation_and_original_errors(tmp_path, failure):
    async def scenario():
        path = tmp_path / "copy.zip"
        if failure != "missing":
            path.write_bytes(b"body")
        registry = LeaseRegistry()
        entered = asyncio.Event()
        background = []

        async def done():
            assert registry.leased(path)
            background.append(True)
            if failure == "background":
                raise ValueError("background failure")

        response = LeasedFileResponse(path, background=BackgroundTask(done))
        response.registry = registry

        async def send(message):
            if message["type"] == "http.response.start":
                entered.set()
                if failure == "send":
                    raise ValueError("send failure")
                if failure == "cancel":
                    await asyncio.Event().wait()

        task = asyncio.create_task(response(scope(), receive, send))
        if failure == "cancel":
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError if failure == "missing" else ValueError):
                await asyncio.wait_for(task, 5)
        assert not registry.leased(path)
        assert background == ([True] if failure == "background" else [])

    asyncio.run(scenario())


def test_head_and_exclusive_claim_keep_original_download_behavior(tmp_path):
    async def scenario():
        path = tmp_path / "copy.zip"
        path.write_bytes(b"body")
        registry = LeaseRegistry()
        response = LeasedFileResponse(path)
        response.registry = registry
        messages = []

        async def send(message):
            messages.append(message)

        with registry.exclusive(path):
            await response(scope(), receive, send)
            assert messages[0]["status"] == 409
        messages.clear()
        await response(scope(method="HEAD"), receive, send)
        assert messages[0]["status"] == 200
        assert messages[1]["body"] == b"" and not registry.leased(path)

    asyncio.run(scenario())
