"""Starting the server: a busy port in plain words, a second launch that finds
AuraDream already there, and the opt-in --open (`main.run`).

No real browser: `webbrowser.open` is replaced. The ports are the system's
ephemeral ones, held by a tiny HTTP server or a plain socket for the test.
"""
from __future__ import annotations

import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading

import pytest

from backend import main


def _page(title: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            body = b"<!doctype html><html><head><title>" + title + b"</title></head></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


@pytest.fixture
def listener():
    """Start an HTTP server whose index has the given title on an ephemeral
    loopback port; return the port. Stopped after the test."""
    servers: list[ThreadingHTTPServer] = []

    def start(title: bytes) -> int:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _page(title))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server.server_address[1]

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def opened(monkeypatch):
    urls: list[str] = []
    monkeypatch.setattr(main.webbrowser, "open", lambda url, *a, **k: urls.append(url) or True)
    return urls


def _on_port(monkeypatch, port: int, host: str = "127.0.0.1") -> None:
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, host=host, port=port))


def _never_serve(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("the server started on a port it had found busy")

    monkeypatch.setattr(main, "Server", refuse)


@pytest.mark.parametrize("open_browser", [False, True])
def test_a_second_launch_finds_auradream_and_exits_cleanly(monkeypatch, listener, opened, capsys, open_browser):
    port = listener(b"AuraDream")
    _on_port(monkeypatch, port)
    _never_serve(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main.run(open_browser=open_browser)
    assert exc.value.code == 0
    url = f"http://127.0.0.1:{port}/"
    assert capsys.readouterr().out.strip() == f"AuraDream is already running at {url}"
    # The browser only when asked for: --open, never by default.
    assert opened == ([url] if open_browser else [])


def test_another_program_on_the_port_is_named_with_the_variable(monkeypatch, listener, opened, capsys):
    port = listener(b"Something else")
    _on_port(monkeypatch, port)
    _never_serve(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main.run(open_browser=True)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert f"Port {port} is in use by another program" in err
    assert "AURADREAM_PORT" in err and f"such as {port + 1}" in err
    assert opened == []


def test_a_listener_that_never_answers_is_not_auradream(monkeypatch, opened, capsys):
    # Accepts the connection and says nothing: the probe gives up quickly.
    monkeypatch.setattr(main, "PROBE_SECONDS", 0.2)
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        _on_port(monkeypatch, silent.getsockname()[1])
        _never_serve(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            main.run()
    assert exc.value.code == 1
    assert "AURADREAM_PORT" in capsys.readouterr().err
    assert opened == []


def test_a_free_port_is_not_busy():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert main._port_busy("127.0.0.1", port) is False


@pytest.mark.parametrize(("host", "url"), [
    ("127.0.0.1", "http://127.0.0.1:7860/"),
    ("0.0.0.0", "http://127.0.0.1:7860/"),
    ("::", "http://[::1]:7860/"),
    ("::1", "http://[::1]:7860/"),
    ("localhost", "http://localhost:7860/"),
])
def test_the_browser_goes_to_an_address_this_machine_reaches(host, url):
    assert main._page_url(host, 7860) == url


def test_open_waits_until_the_server_is_listening(fake, monkeypatch, opened):
    # The real path on a real ephemeral port: the browser is asked for only
    # after uvicorn has bound and started, from its own daemon thread.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    _on_port(monkeypatch, port)
    servers: list[main.Server] = []
    seen_started: list[bool] = []

    class Recorded(main.Server):
        def __init__(self, config):
            super().__init__(config)
            servers.append(self)

    def record(url, *args, **kwargs):
        seen_started.append(servers[0].started)
        # And the real page is the one a second launch recognises.
        seen_started.append(main._auradream_at(url))
        opened.append(url)
        servers[0].should_exit = True
        return True

    monkeypatch.setattr(main, "Server", Recorded)
    monkeypatch.setattr(main.webbrowser, "open", record)
    main.run(open_browser=True)
    assert opened == [f"http://127.0.0.1:{port}/"] and seen_started == [True, True]


def test_without_open_no_browser_is_asked_for(fake, monkeypatch, opened):
    started = threading.Event()

    class Stopping(main.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            started.set()
            self.should_exit = True

    monkeypatch.setattr(main, "Server", Stopping)
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, port=0))
    main.run()
    assert started.is_set() and opened == []

