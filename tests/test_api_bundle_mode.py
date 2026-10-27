"""Actual saved-download route discovers opt-in mode without backup activation."""
from __future__ import annotations

import json
import os

import pytest

from backend import bundle_file_lock, bundle_mode, bundles
from starlette.responses import FileResponse
from tests.bundle_helpers import _save
from tools import install_bundle_mode


def test_explicit_cli_needs_operator_confirmation_and_does_not_read_settings(tmp_path, capsys):
    assert install_bundle_mode.main([str(tmp_path)]) == 1
    assert "operator-supplied" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
    assert install_bundle_mode.main([str(tmp_path), "--check"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["state"] == "not_installed" and checked["read_only"]
    assert checked["machine_verified"] is False and list(tmp_path.iterdir()) == []


def test_actual_download_route_never_streams_under_an_invalid_installed_marker(client, store, monkeypatch):
    _save("synthetic invalid-mode refusal")
    exported = bundles.export_bundle(keep_gallery=True)
    (store.bundle_directory() / bundle_mode.MARKER_NAME).write_bytes(b"{}")

    async def forbidden(*_args, **_kwargs):
        pytest.fail("an invalid installed mode reached an unleased file response")

    monkeypatch.setattr(FileResponse, "__call__", forbidden)
    with pytest.raises(bundle_mode.ModeError):
        client.get(f"/api/gallery/bundles/files/{exported['name']}")


@pytest.mark.skipif(os.name != "nt", reason="Windows fixed NTFS shared-mode integration only")
def test_actual_http_download_discovers_marker_without_touching_manual_archive(client, store):
    _save("synthetic shared-mode download")
    exported = bundles.export_bundle(keep_gallery=True)
    directory = store.bundle_directory()
    archive = directory / exported["name"]
    before = archive.read_bytes()
    bundle_mode.install(directory, other_servers_stopped=True)
    result = client.get(f"/api/gallery/bundles/files/{archive.name}")
    assert result.status_code == 200 and result.content == before
    assert bundle_file_lock.companion_path(archive).is_file()
    assert archive.read_bytes() == before
    status = client.get("/api/gallery/automatic-backup").json()
    assert not status["enabled"] and not status["retention_enabled"]
    assert client.get("/api/gallery/bundles/status").json()["state"] == "idle"


@pytest.mark.skipif(os.name != "nt", reason="Windows fixed NTFS shared-mode integration only")
def test_cli_receipt_explicitly_does_not_claim_machine_verified_quiescence(tmp_path, capsys, monkeypatch):
    assert install_bundle_mode.main([str(tmp_path), "--other-servers-stopped"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["state"] == "installed" and receipt["other_servers_stopped"] == "operator_confirmed"
    assert receipt["machine_verified"] is False and receipt["retention_enabled"] is False
    assert receipt["supported_platform"] == "windows-fixed-ntfs"
    monkeypatch.setattr(bundle_mode.tempfile, "NamedTemporaryFile", lambda **_kwargs: pytest.fail("read-only check tried publication"))
    assert install_bundle_mode.main([str(tmp_path), "--check"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["state"] == "installed" and checked["read_only"]
    assert checked["installation_id"] == receipt["installation_id"] and checked["machine_verified"] is False
