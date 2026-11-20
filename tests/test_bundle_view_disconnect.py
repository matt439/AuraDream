"""An abandoned browse request must release its archive through the real HTTP stack."""
from contextlib import closing, contextmanager
from http.client import HTTPConnection
import json
import socket
import threading
import time

import uvicorn

from backend import bundle_browse_archive, bundle_leases, bundle_views, main
from backend.bundle_common import BrowseUnavailable
from backend.bundle_manager import BROWSE_PREFIX


def test_disconnect_during_bundle_verification_releases_the_reader(fake, tmp_path, monkeypatch):
    archive = tmp_path / "scratch.zip"
    archive.touch()
    entered = threading.Event()
    abandoned = threading.Event()
    stop_worker = threading.Event()

    @contextmanager
    def verify(name, password, *, cancelled):
        assert name == archive.name
        try:
            with bundle_leases.reading(archive):
                entered.set()
                while not stop_worker.is_set():
                    if cancelled():
                        raise BrowseUnavailable("the bundle view was closed")
                    time.sleep(0.05)
                yield None  # teardown unblocks the worker if the test fails
        finally:
            abandoned.set()

    monkeypatch.setattr(bundle_browse_archive, "open_view", verify)
    views = bundle_views.BundleViews()
    monkeypatch.setattr(main.app.state, "bundle_views", views, raising=False)
    server = uvicorn.Server(uvicorn.Config(main.app, lifespan="off", log_config=None))
    # A bound listening socket gives the client an address before the worker
    # starts accepting. No fixed port or sleep-based readiness check.
    with closing(socket.socket()) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        host, port = listener.getsockname()
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            # TestClient buffers a full response. Closing a real TCP connection
            # exercises http.disconnect through the app's middleware stack.
            with closing(HTTPConnection(host, port, timeout=15)) as client:
                client.request("POST", BROWSE_PREFIX, json.dumps({"name": archive.name}),
                               {"Content-Type": "application/json"})
                assert entered.wait(15), "archive verification did not start"
                assert bundle_leases.downloads.leased(archive)
            assert abandoned.wait(15), "archive verification did not observe the closed connection"
            assert not bundle_leases.downloads.leased(archive)
        finally:
            stop_worker.set()
            server.should_exit = True
            thread.join(timeout=20)
            assert not thread.is_alive()
            views.shutdown()
