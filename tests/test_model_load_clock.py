"""The server clock covers explicit loads and does not restart on another waiter."""
import threading

from backend import main
from tests.api_helpers import ORIGIN, wait_for_status


def test_load_clock_is_shared_by_waiters_and_clears_on_completion(client, fake, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = fake.load

    def load():
        entered.set()
        assert release.wait(5)
        original()

    monkeypatch.setattr(fake, "load", load)
    assert client.post("/api/model/unload", headers={"Origin": ORIGIN}).status_code == 200
    try:
        first = client.post("/api/model/load", headers={"Origin": ORIGIN})
        assert entered.wait(5)
        assert first.status_code == 200 and first.json()["state"] == "loading"
        started = first.json()["load_started_at"]
        assert isinstance(started, float)
        assert first.json()["server_time"] >= started
        again = client.post("/api/model/load", headers={"Origin": ORIGIN})
        assert again.status_code == 200 and again.json()["load_started_at"] == started
        assert main._load_lock.locked()
    finally:
        release.set()
    ready = wait_for_status(client, state="ready")
    assert ready["load_started_at"] is None


def test_load_clock_includes_gate_wait_and_resets_after_failure_and_retry(client, fake, monkeypatch):
    assert client.post("/api/model/unload", headers={"Origin": ORIGIN}).status_code == 200
    original = fake.load

    def failed():
        raise ConnectionError("offline test")

    monkeypatch.setattr(fake, "load", failed)
    client.portal.call(main.jobs._gate.acquire)
    try:
        waiting = client.post("/api/model/load", headers={"Origin": ORIGIN}).json()
        assert waiting["state"] == "loading" and isinstance(waiting["load_started_at"], float)
        assert not fake.loaded and fake.loads == 0
        assert main.jobs._gate.locked()
    finally:
        client.portal.call(main.jobs._gate.release)
    error = wait_for_status(client, state="error")
    assert error["load_started_at"] is None and error["detail_kind"] == "connection"
    monkeypatch.setattr(fake, "load", original)
    client.portal.call(main.jobs._gate.acquire)
    try:
        retry = client.post("/api/model/load", headers={"Origin": ORIGIN}).json()
        assert retry["state"] == "loading" and retry["load_started_at"] >= waiting["load_started_at"]
        assert retry["detail"] is None and not fake.loaded
    finally:
        client.portal.call(main.jobs._gate.release)
    ready = wait_for_status(client, state="ready")
    assert ready["load_started_at"] is None and ready["detail"] is None
