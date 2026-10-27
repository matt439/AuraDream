"""Password bundles: encrypted on disk, importable only with the password, altered ones refused."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend import bundle_crypto, bundles
from tests.bundle_helpers import _save, _destination, _snapshot


def test_password_export_is_encrypted_and_imports_only_with_its_password(store, monkeypatch, tmp_path):
    first = _save("first", 1)
    _save("second", 2, reference=[first.id])
    archive = bundles.export_bundle(expected=2, password="tea kettle")
    bundle = Path(archive["path"])
    assert archive["cleared"] and bundle.suffix == bundle_crypto.SUFFIX
    blob = bundle.read_bytes()
    assert blob.startswith(bundle_crypto.MAGIC) and b"manifest.json" not in blob and first.filename.encode() not in blob
    assert [b["encrypted"] for b in bundles.list_bundles() if b["name"] == archive["name"]] == [True]

    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    for password, message in [(None, "password-protected"), ("wrong", "incorrect")]:
        with pytest.raises(bundles.BundleError, match=message):
            bundles.import_bundle(bundle, password)
        assert _snapshot(store) == snapshot
    imported = bundles.import_bundle(bundle, "tea kettle")
    assert imported["imported"] == 2 and imported["unresolved_references"] == 0
    assert bundles.import_bundle(bundle, "tea kettle")["present"] == 2


def test_a_password_bundle_that_fails_its_read_back_clears_nothing(store, monkeypatch):
    _save("kept", 1)
    snapshot = _snapshot(store)
    # Without its final chunk the ciphertext cannot authenticate.
    monkeypatch.setattr(bundle_crypto.EncryptingWriter, "finish", lambda self: None)
    with pytest.raises(bundles.BundleError):
        bundles.export_bundle(expected=1, password="pw")
    assert _snapshot(store) == snapshot
    assert not list(bundles._directory().iterdir())


def test_altered_password_bundle_is_rejected_before_destination_changes(store, monkeypatch, tmp_path):
    _save("source", 1)
    bundle = Path(bundles.export_bundle(expected=1, password="pw")["path"])
    blob = bytearray(bundle.read_bytes())
    blob[len(blob) // 2] ^= 1
    bundle.write_bytes(bytes(blob))
    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError, match="damaged or was changed"):
        bundles.import_bundle(bundle, "pw")
    assert _snapshot(store) == snapshot


def test_a_password_bundle_altered_past_its_first_chunk_is_refused_and_staging_discarded(store, monkeypatch, tmp_path):
    # Only the header and the final chunk are checked on open; any other
    # chunk is authenticated as it is read, after earlier members are staged.
    monkeypatch.setitem(bundle_crypto.EncryptingWriter.__init__.__kwdefaults__, "chunk", 4096)
    _save("first", 1)
    _save("second", 2)
    bundle = Path(bundles.export_bundle(expected=2, password="pw")["path"])
    blob = bytearray(bundle.read_bytes())
    assert len(blob) > bundle_crypto.HEADER_SIZE + 3 * (4096 + 16)
    blob[bundle_crypto.HEADER_SIZE + 4096 + 16 + 10] ^= 1  # inside the second chunk
    bundle.write_bytes(bytes(blob))
    _destination(store, monkeypatch, tmp_path)
    _save("existing", 99)
    snapshot = _snapshot(store)
    with pytest.raises(bundles.BundleError, match="damaged or was changed"):
        bundles.import_bundle(bundle, "pw")
    assert _snapshot(store) == snapshot
    staging = store.bundle_staging_dir()
    assert not staging.exists() or not list(staging.iterdir())
    assert not list(store.settings.db_path.parent.glob(".bundle-*"))


def test_a_password_export_derives_its_key_once(store, monkeypatch, tmp_path):
    _save("sealed", 1)
    calls = []
    real = bundle_crypto._derive
    monkeypatch.setattr(bundle_crypto, "_derive", lambda *a: calls.append(a) or real(*a))
    bundle = Path(bundles.export_bundle(expected=1, password="pw")["path"])
    assert len(calls) == 1  # the read-back reuses the writer's key for its own header
    _destination(store, monkeypatch, tmp_path)
    assert bundles.import_bundle(bundle, "pw")["imported"] == 1
    assert len(calls) == 2  # import derives from the password it was given


def test_the_password_check_domain_is_frozen_from_the_old_name():
    # Bundles protected before the rename to Auradream must still open: this
    # known answer pins the "local-diffusion bundle password check" domain.
    key, check = bundle_crypto._derive("pw", 4, 1, 1, bytes(range(16)))
    assert key.hex() == "d433e37c20a14a80b34bc3a97f160455359facf59598e5547595b2e4c31d7473"
    assert check.hex() == "940ff791e4f5254fedd6c7ee7b06bf51"
