"""Selective restore keeps both tiled root proofs and remaps archive IDs."""
import pytest

from backend import tiled_refine_admission
from backend.api_bundle_browse import VIEW_HEADER
from backend.bundle_manager import BROWSE_PREFIX
from tests.api_helpers import until
from tests.bundle_helpers import png_metadata, save_source, use_destination
from tests.test_tiled_refine_bundles import backup, save_tiled_refine


@pytest.fixture
def archive_in_destination(store, monkeypatch, tmp_path, password):
    root, keeper, proof = save_tiled_refine()
    unrelated = save_source("unrelated archived image", 483)
    archive = backup(password=password)
    destination = use_destination(store, monkeypatch, tmp_path)
    occupied = save_source("other gallery's image", 937)
    assert occupied.id == root.id and occupied.filename != root.filename
    return archive, root, keeper, proof, unrelated, destination, occupied


@pytest.mark.parametrize(("include_root", "password"), [(True, "archive passphrase"), (False, None)])
def test_selective_restore_preserves_tiled_proofs_without_using_foreign_ids(
    archive_in_destination, client, store, include_root, password,
):
    archive, root, keeper, proof, unrelated, destination, occupied = archive_in_destination
    opened = client.post(BROWSE_PREFIX, json={"name": archive.name, "password": password})
    assert opened.status_code == 200, opened.text
    headers = {VIEW_HEADER: opened.json()["token"]}
    preview = client.post(BROWSE_PREFIX + "/restore/preview", headers=headers, json={"image_id": keeper.id})
    assert preview.status_code == 200, preview.text
    assert [(item["kind"], item["id"], item["role"], item["status"])
            for item in preview.json()["dependencies"]] == [("image", root.id, "tiled_refine", "available")]

    def restore():
        response = client.post(BROWSE_PREFIX + "/restore", headers=headers, json={
            "image_id": keeper.id,
            "include_ids": [root.id] if include_root else [],
        })
        assert response.status_code == 202, response.text
        operation = response.json()["id"]

        def finished():
            status = client.get("/api/gallery/bundles/status").json()
            return status if status["id"] == operation and status["state"] != "running" else None

        status = until(finished)
        assert status["state"] == "done", status
        return status["result"]

    result = restore()
    rows = {row.filename: row for row in store.list_images()[0]}
    assert set(rows) == {keeper.filename, occupied.filename} | ({root.filename} if include_root else set())
    assert not (destination.outputs_dir / unrelated.filename).exists()
    assert rows[occupied.filename] == occupied
    restored = rows[keeper.filename]
    assert restored.id == result["image_id"]
    assert result["replay_unavailable"] is (not include_root)
    source = rows[root.filename].id if include_root else None
    assert source != occupied.id
    expected = proof.model_copy(update={"source": source})
    assert restored.tiled_refine == expected
    assert png_metadata(destination.outputs_dir / restored.filename)["tiled_refine"] == expected.model_dump(mode="json")
    if include_root:
        assert tiled_refine_admission.bind_source(restored.tiled_refine) == expected
    else:
        with pytest.raises(ValueError, match="not been resolved"):
            tiled_refine_admission.bind_source(restored.tiled_refine)

    repeat = restore()
    assert repeat["image_id"] == restored.id and repeat["imported"] == 0
    assert repeat["replay_unavailable"] is (not include_root)
    assert {row.filename: row for row in store.list_images()[0]} == rows
