"""Exact proxy origins and a real loopback TLS server with a throwaway certificate."""
from __future__ import annotations

from dataclasses import replace
import socket
import urllib.request

import pytest

from backend import main, server_transport
from tests.tls_helpers import loopback_certificate


@pytest.mark.parametrize("origin", [
    "https://dream.home.arpa", "https://dream.home.arpa:443",
])
def test_a_configured_proxy_origin_writes_through_the_existing_host_guard(client, monkeypatch, origin):
    monkeypatch.setattr(main, "settings", replace(
        main.settings, allowed_hosts=("127.0.0.1", "dream.home.arpa"),
        allowed_origins=("https://dream.home.arpa",),
    ))
    assert client.post("/api/model/unload", headers={"Origin": origin}).status_code == 200
    for bad in ("http://dream.home.arpa", "https://dream.home.arpa:444", "https://elsewhere.example"):
        assert client.post("/api/model/unload", headers={"Origin": bad}).status_code == 403
    # Listing an origin does not grant a new hostname.
    monkeypatch.setattr(main, "settings", replace(main.settings, allowed_hosts=("127.0.0.1",)))
    assert client.post("/api/model/unload", headers={"Origin": origin}).status_code == 403
    assert client.post("/api/model/unload", headers={"Host": "elsewhere.example"}).status_code == 400


@pytest.mark.parametrize("origin", [
    "null", "https://", "ftp://localhost", "https://user@localhost", "https://localhost/",
    "https://localhost?", "https://localhost#", "https://localhost/path", "https://*.example",
    "https://localhost:", "https://localhost:0", "https://localhost:65536",
    "https://localhost\\@elsewhere", " https://localhost", "https://localhost\n",
])
def test_non_origins_are_refused(origin):
    assert server_transport.origin_key(origin) is None


def test_default_ports_and_ipv6_are_compared_as_origins():
    assert server_transport.origin_key("https://[::1]") == ("https", "::1", 443)
    assert server_transport.origin_key("http://LOCALHOST:80") == ("http", "localhost", 80)


def test_tls_requires_both_existing_files(tmp_path):
    cert = tmp_path / "certificate.pem"
    cert.write_text("not a real certificate", encoding="ascii")
    assert server_transport.setting_problem(None, None) is None
    assert "both" in server_transport.setting_problem(cert, None)
    assert "SSL_KEYFILE" in server_transport.setting_problem(cert, tmp_path / "absent")


def test_direct_tls_serves_and_probes_a_verified_https_page(fake, monkeypatch, tmp_path):
    certfile, keyfile = loopback_certificate(tmp_path)
    # Only this test process trusts its temporary root; no OS trust changes.
    monkeypatch.setenv("SSL_CERT_FILE", str(certfile))
    with socket.socket() as trial:
        trial.bind(("127.0.0.1", 0))
        port = trial.getsockname()[1]
    monkeypatch.setattr(main, "settings", replace(
        main.settings, host="127.0.0.1", port=port, ssl_certfile=certfile,
        ssl_keyfile=keyfile, db_path=tmp_path / "auradream.db",
    ))
    servers, observed = [], []

    class Recorded(main.Server):
        def __init__(self, config):
            super().__init__(config)
            servers.append(self)

    def opened(url):
        try:
            observed.append(url)
            observed.append(main._auradream_at(url))
            direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request(url + "api/model/unload", data=b"", method="POST",
                                             headers={"Origin": url.removesuffix("/")})
            with direct.open(request, timeout=5) as response:
                observed.append(response.status)
        finally:
            servers[0].should_exit = True

    monkeypatch.setattr(main, "Server", Recorded)
    monkeypatch.setattr(main.webbrowser, "open", opened)
    main.run(open_browser=True)
    assert observed == [f"https://127.0.0.1:{port}/", True, 200]
