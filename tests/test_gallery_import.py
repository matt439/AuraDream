from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from backend import bundles, gallery_import, gallery_reports, png_settings, reindex
from backend.bundle_manager import BROWSE_PREFIX, RESTORE_PATH
from tests.api_helpers import run_one
from tests.bundle_helpers import export_all, png_metadata, save_source, use_destination
from tests.test_api_bundle_restore import _open, _selected
from tests.test_api_bundles import _finish


def _encode(format="PNG", size=(256, 256), *, mode="RGB", color=(23, 41, 91), **info):
    output = BytesIO()
    with Image.new(mode, size, color) as image:
        image.save(output, format=format, **info)
    return output.getvalue()


def test_import_is_a_gallery_source_for_reference_and_refine(client, fake, store):
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "content from another tool")
    meta.add_text("auradream", "{\"prompt\": \"this must be stripped\"}")
    response = client.post(
        "/api/gallery/import", params={"name": r"C:\Photos\sketch.png"},
        content=_encode(pnginfo=meta),
    )
    assert response.status_code == 201, response.json()
    original = response.json()
    source_id = original["id"]
    assert original["imported_name"] == "sketch.png"
    assert original["model_id"] == original["prompt"] == ""
    assert original["thumb_url"] and not fake.requests
    expected = store.settings.outputs_dir / original["filename"]
    with Image.open(expected) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (23, 41, 91)
        assert set(image.info) == {"auradream"}
    assert not list(store.settings.outputs_dir.rglob("*.part"))
    assert png_metadata(expected)["imported_name"] == "sketch.png"
    with pytest.raises(png_settings.PngSettingsError, match="has no generation settings"):
        png_settings.image_settings(str(expected))
    assert client.get("/api/gallery").json()["items"] == [original]
    model = fake.describe()["model_id"]
    assert client.get("/api/gallery", params={"model": model}).json()["items"] == []
    assert client.get("/api/gallery", params={"all_models": True, "include_imported": False}).json()["items"] == []
    assert client.get("/api/runs").json() == []
    assert gallery_reports.keep_rates() == []
    assert client.get("/api/gallery/unstarred").json()["unstarred"] == 1
    rerender = client.post(
        "/api/rerender/preview", json={"ids": [source_id], "width": 256, "height": 256},
    )
    assert rerender.status_code == 422 and "Imported sources" in rerender.json()["detail"]

    run_one(client, reference=[source_id])
    run_one(client, refine=source_id, width=256, height=256, steps=10, strength=0.5)
    assert fake.requests[-2].reference == [source_id]
    assert fake.requests[-1].refine == source_id
    refined, conditioned, source = client.get("/api/gallery").json()["items"]
    assert (refined["refine"], conditioned["reference"], source["id"]) == (source_id, [source_id], source_id)
    assert refined["model_id"] and refined["imported_name"] is None
    assert png_metadata(store.settings.outputs_dir / refined["filename"])["refine_source"]["hash"] == original["content_hash"]

    assert client.post("/api/images/bin", json={"ids": [source_id]}).json()["binned"] == [source_id]
    rejected = client.post("/api/generate", json={
        "prompt": "sketch", "width": 256, "height": 256, "refine": source_id, "strength": 0.5,
    })
    assert rejected.status_code == 422 and rejected.json()["code"] == "image_binned"


def test_upload_refuses_unsupported_or_large_images_before_storage(client, fake, store, monkeypatch):
    route = "/api/gallery/import"
    unsupported = client.post(route, content=_encode("BMP", size=(12, 12)))
    assert unsupported.status_code == 422 and "PNG, JPEG or WebP" in unsupported.json()["detail"]

    monkeypatch.setattr(gallery_import, "MAX_PIXELS", 10)
    large = client.post(route, content=_encode(size=(4, 4)))
    assert large.status_code == 422 and "megapixels" in large.json()["detail"]

    monkeypatch.setattr(gallery_import, "MAX_UPLOAD_BYTES", 5)
    assert client.post(route, content=b"123456").status_code == 413
    assert client.post(route, content=iter((b"123", b"456"))).status_code == 413
    assert store.list_images()[1] == 0 and not fake.requests
    assert not list(store.settings.outputs_dir.rglob("*.*"))


def test_import_normalizes_orientation_and_transparent_sketches(store):
    orientation = Image.Exif()
    orientation[274] = 6
    jpeg = gallery_import.store(_encode("JPEG", size=(4, 6), exif=orientation), "photo.jpg")
    transparent = gallery_import.store(_encode(mode="RGBA", color=(24, 40, 96, 0)), "drawing.png")
    assert (jpeg.width, jpeg.height) == (6, 4)
    with Image.open(store.settings.outputs_dir / jpeg.filename) as image:
        assert "exif" not in image.info
    with Image.open(store.settings.outputs_dir / transparent.filename) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("recovery", ["png", "database", "bundle"])
def test_imported_source_and_replay_survive_gallery_recovery(store, tmp_path, monkeypatch, recovery):
    photo = gallery_import.store(_encode("WEBP"), "holiday photo.webp")
    conditioned = save_source("conditioned", seed=12, reference=[photo.id])
    refined = save_source("refined", seed=13, refine=photo.id, strength=0.35)
    original = store.settings
    if recovery == "bundle":
        archive = export_all()
    dest = use_destination(store, monkeypatch, tmp_path)
    existing = save_source("ID collision", seed=1)
    if recovery == "bundle":
        assert bundles.import_bundle(Path(archive["path"]))["imported"] == 3
        assert bundles.import_bundle(Path(archive["path"]))["present"] == 3
    else:
        monkeypatch.setattr(reindex, "settings", dest)
        kwargs = {"database": original.db_path} if recovery == "database" else {}
        counts = reindex.reindex(original.outputs_dir, out=StringIO(), **kwargs)
        assert counts.get("indexed", counts.get("imported")) == 3, counts

    by_filename = {record.filename: record for record in store.list_images()[0]}
    source = by_filename[photo.filename]
    assert source.id != existing.id and source.id != photo.id
    assert source.imported_name == photo.imported_name and source.content_hash == photo.content_hash
    new_reference = by_filename[conditioned.filename]
    new_refine = by_filename[refined.filename]
    assert new_reference.reference == [source.id]
    assert new_refine.refine == source.id and not new_refine.refine_missing
    assert store.reference_paths([source.id]) == [dest.outputs_dir / photo.filename]
    assert png_metadata(dest.outputs_dir / photo.filename)["imported_name"] == photo.imported_name


def test_selectively_restore_an_imported_refine_source(client, store, tmp_path, monkeypatch):
    source = gallery_import.store(_encode(), "pencil study.png")
    keeper = save_source("keeper", seed=9, refine=source.id, strength=0.4)
    unrelated = save_source("other", seed=10)
    archive = Path(bundles.export_bundle(keep_gallery=True)["path"])
    use_destination(store, monkeypatch, tmp_path)
    collision = save_source("collision", seed=1)
    headers = _open(client, archive)
    found = client.post(BROWSE_PREFIX + "/images", headers=headers, json={"prompt": "pencil study"}).json()
    archived = found["images"][0]
    assert archived["id"] == source.id
    assert archived["settings"]["imported_name"] == "pencil study.png"
    assert "seed" not in archived["settings"]
    preview = client.post(RESTORE_PATH + "/preview", headers=headers, json={"image_id": keeper.id}).json()
    assert preview["dependencies"][0]["status"] == "available"

    outcome = _finish(client, _selected(client, headers, keeper.id, [source.id]))
    assert outcome["state"] == "done", outcome
    result = outcome["result"]
    assert result["imported"] == 2 and not result["replay_unavailable"]
    restored = store.get_image(result["image_id"])
    local_source = store.get_image(restored.refine)
    assert local_source.id != collision.id
    assert local_source.imported_name == source.imported_name
    assert local_source.content_hash == source.content_hash
    assert not (store.settings.outputs_dir / unrelated.filename).exists()
