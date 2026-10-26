"""Local/shared lease exclusion and complete ASGI FileResponse lifetime."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.background import BackgroundTask

from backend.bundle_leases import LeaseBusy, LeaseRegistry, LeasedFileResponse
from backend import bundle_file_lock, bundle_leases


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


def test_independent_registry_reader_hints_are_process_wide(tmp_path):
    first, second = LeaseRegistry(), LeaseRegistry()
    path = tmp_path / "copy.zip"
    with first.reader(path):
        assert second.leased(path)
        assert first.snapshot() == second.snapshot() == frozenset({first._key(path)})
    assert not second.leased(path)
    assert first.snapshot() == second.snapshot() == frozenset()


def test_reusing_one_token_cannot_pretend_to_protect_an_unleased_file(tmp_path):
    registry = LeaseRegistry()
    path = tmp_path / "copy.zip"
    token = registry.reader(path)
    with token:
        with pytest.raises(RuntimeError, match="already"):
            with token:
                pytest.fail("one token cannot enter two independent lifetimes")
        assert registry.leased(path)
    with pytest.raises(RuntimeError, match="released"):
        with token:
            pytest.fail("a released token cannot be reused")
    with registry.exclusive(path):
        assert not registry.leased(path)


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


def test_default_downloads_never_create_or_open_companions(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default recovery downloads must not require companion writes")

    monkeypatch.setattr(bundle_file_lock, "acquire", forbidden)
    path = tmp_path / "copy.zip"
    path.write_bytes(b"body")
    registry = LeaseRegistry()
    with registry.reader(path):
        with pytest.raises(LeaseBusy):
            LeaseRegistry().exclusive(path)
    response = LeasedFileResponse(path)
    messages = []

    async def send(message):
        messages.append(message)

    asyncio.run(response(scope(), receive, send))
    assert messages[0]["status"] == 200
    assert list(tmp_path.iterdir()) == [path]


class FakeLock:
    def __init__(self):
        self.closed = False
        self.releases = 0
        self.validations = 0
        self.failure = None

    def validate(self):
        self.validations += 1

    def release(self):
        self.releases += 1
        if self.failure is not None:
            raise self.failure
        self.closed = True


@pytest.mark.parametrize("shared_first", [False, True])
def test_mixed_registries_retain_native_handle_until_every_local_reader_releases(tmp_path, monkeypatch, shared_first):
    path = tmp_path / "copy.zip"
    native = FakeLock()
    calls = []

    def acquire(archive, *, exclusive):
        calls.append((archive, exclusive))
        return native

    monkeypatch.setattr(bundle_file_lock, "acquire", acquire)
    local = LeaseRegistry()
    shared = LeaseRegistry(cross_process=True)
    first = shared.reader(path) if shared_first else local.reader(path)
    second = local.reader(path) if shared_first else shared.reader(path)
    third = LeaseRegistry(cross_process=True).reader(path)
    assert calls == [(path, False)] and native.validations == 1
    with pytest.raises(LeaseBusy):
        LeaseRegistry(cross_process=True).exclusive(path)
    # Releasing both shared tokens must not remove the lock while the local
    # stream remains. This is not an installer for mixed-process safety.
    (first if shared_first else second).release()
    third.release()
    assert native.releases == 0
    (second if shared_first else first).release()
    assert native.releases == 1 and native.closed
    assert local.snapshot() == shared.snapshot() == frozenset()


@pytest.mark.parametrize("exclusive", [False, True])
def test_native_access_failure_is_preserved_and_never_reported_as_busy(tmp_path, monkeypatch, exclusive):
    original = PermissionError(13, "injected access denial", str(tmp_path / "copy.zip.lease.lock"))

    def denied(*args, **kwargs):
        raise original

    monkeypatch.setattr(bundle_file_lock, "acquire", denied)
    registry = LeaseRegistry(cross_process=True)
    with pytest.raises(PermissionError) as caught:
        (registry.exclusive if exclusive else registry.reader)(tmp_path / "copy.zip")
    assert caught.value is original
    assert registry.snapshot() == frozenset()


def test_only_native_contention_becomes_lease_busy(tmp_path, monkeypatch):
    def contended(*args, **kwargs):
        raise bundle_file_lock.LockContended("injected native contention")

    monkeypatch.setattr(bundle_file_lock, "acquire", contended)
    with pytest.raises(LeaseBusy, match="another process") as caught:
        LeaseRegistry(cross_process=True).reader(tmp_path / "copy.zip")
    assert isinstance(caught.value.__cause__, bundle_file_lock.LockContended)


def test_failed_close_keeps_local_exclusion_and_never_retries_implicitly(tmp_path, monkeypatch):
    path = tmp_path / "copy.zip"
    native = FakeLock()
    original = PermissionError("injected close denial")
    native.failure = original
    monkeypatch.setattr(bundle_file_lock, "acquire", lambda *args, **kwargs: native)
    registry = LeaseRegistry(cross_process=True)
    lease = registry.reader(path)
    try:
        with pytest.raises(PermissionError) as caught:
            lease.release()
        assert caught.value is original and not lease._released
        with pytest.raises(RuntimeError, match="no implicit retry"):
            lease.release()
        with pytest.raises(RuntimeError, match="release failed"):
            LeaseRegistry().reader(path)
        with pytest.raises(LeaseBusy):
            LeaseRegistry(cross_process=True).exclusive(path)
        assert native.releases == 1
    finally:
        # This is injected state, with no native fd or actual denied action.
        bundle_leases._states.pop(registry._key(path))


def test_unlock_error_after_successful_close_releases_local_state(tmp_path, monkeypatch):
    native = FakeLock()
    original = OSError("injected unlock error")

    def released_with_error():
        native.closed = True
        native.releases += 1
        raise original

    monkeypatch.setattr(native, "release", released_with_error)
    monkeypatch.setattr(bundle_file_lock, "acquire", lambda *args, **kwargs: native)
    path = tmp_path / "copy.zip"
    registry = LeaseRegistry(cross_process=True)
    lease = registry.reader(path)
    with pytest.raises(OSError) as caught:
        lease.release()
    assert caught.value is original and lease._released
    lease.release()
    assert native.releases == 1 and registry.snapshot() == frozenset()
    with LeaseRegistry().exclusive(path):
        pass


def test_release_failure_does_not_hide_original_response_error(tmp_path, monkeypatch):
    native = FakeLock()
    native.failure = OSError("injected cleanup failure")
    monkeypatch.setattr(bundle_file_lock, "acquire", lambda *args, **kwargs: native)
    path = tmp_path / "copy.zip"
    registry = LeaseRegistry(cross_process=True)
    original = ValueError("original body failure")
    try:
        with pytest.raises(ValueError) as caught:
            with registry.reader(path):
                raise original
        assert caught.value is original
        assert "Archive lease release failed" in " ".join(original.__notes__)
    finally:
        bundle_leases._states.pop(registry._key(path))
