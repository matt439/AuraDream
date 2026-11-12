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


def _bad_pairs(tmp_path):
    """Throwaway pairs AuraDream must refuse, by what is wrong with each."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    certfile, keyfile = loopback_certificate(tmp_path / "a")
    other_cert, other_key = loopback_certificate(tmp_path / "b")
    key = serialization.load_pem_private_key(keyfile.read_bytes(), None)
    encrypted = tmp_path / "encrypted.pem"
    encrypted.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                            serialization.BestAvailableEncryption(b"throwaway")))
    der = tmp_path / "certificate.cer"
    der.write_bytes(x509.load_pem_x509_certificate(certfile.read_bytes()).public_bytes(serialization.Encoding.DER))
    junk = tmp_path / "junk.pem"
    junk.write_text("not a real certificate", encoding="ascii")
    return {
        "mismatched": (certfile, other_key, "is not the private key of the certificate"),
        "encrypted": (certfile, encrypted, "protected by a password"),
        "DER certificate": (der, keyfile, "AURADREAM_SSL_CERTFILE holds no PEM certificate"),
        "junk certificate": (junk, keyfile, "AURADREAM_SSL_CERTFILE holds no PEM certificate"),
        "key as certificate": (keyfile, keyfile, "AURADREAM_SSL_CERTFILE holds no PEM certificate"),
        "junk key": (certfile, junk, "AURADREAM_SSL_KEYFILE holds no PEM private key"),
        "certificate as key": (other_cert, other_cert, "AURADREAM_SSL_KEYFILE holds no PEM private key"),
    }


def test_a_pair_that_cannot_serve_is_refused_in_words(tmp_path):
    certfile, keyfile = loopback_certificate(tmp_path)
    assert server_transport.setting_problem(certfile, keyfile) is None
    for name, (cert, key, words) in _bad_pairs(tmp_path).items():
        said = server_transport.setting_problem(cert, key)
        assert said is not None and words in said, (name, said)
        assert "PEM lib" not in said and "start again" in said, (name, said)


def test_a_bad_pair_stops_the_start_before_uvicorn_with_the_settings_status(fake, monkeypatch, capsys, tmp_path):
    def never(*args, **kwargs):
        pytest.fail("the server went on with a TLS pair it cannot load")

    pairs = _bad_pairs(tmp_path)
    monkeypatch.setattr(main, "_port_busy", never)
    monkeypatch.setattr(main, "Server", never)
    for name in ("mismatched", "encrypted", "junk key"):
        cert, key, words = pairs[name]
        monkeypatch.setattr(main, "settings", replace(
            main.settings, host="127.0.0.1", port=7860, ssl_certfile=cert, ssl_keyfile=key,
            db_path=tmp_path / "auradream.db"))
        with pytest.raises(SystemExit) as stopped:
            main.run()
        assert stopped.value.code == main.SETTINGS_REFUSED_STATUS == 78
        said = capsys.readouterr()
        assert words in said.err and "Traceback" not in said.err, (name, said.err)


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
            with direct.open(request, timeout=30) as response:
                observed.append(response.status)
        finally:
            servers[0].should_exit = True

    monkeypatch.setattr(main, "Server", Recorded)
    monkeypatch.setattr(main.webbrowser, "open", opened)
    main.run(open_browser=True)
    assert observed == [f"https://127.0.0.1:{port}/", True, 200]
