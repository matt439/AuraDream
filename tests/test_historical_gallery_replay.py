"""What the public replay actions do with results written by the old app."""
from __future__ import annotations

import hashlib

from PIL import Image, PngImagePlugin
import pytest

from backend import main, model_selection
from tests.api_helpers import stream_job, until
from tests.bundle_helpers import save_source
from tests.historical_gallery_helpers import ARCHIVES, KONTEXT, KREA, RUN_IDS, expected_edit


def upload(client, filename):
    response = client.post("/api/gallery/bundles/upload", content=(ARCHIVES / filename).read_bytes(),
                           headers={"content-type": "application/zip"})
    assert response.status_code == 202, response.text
    operation = response.json()["id"]

    def finished():
        state = client.get("/api/gallery/bundles/status").json()
        return state if state["id"] == operation and state["state"] != "running" else None

    state = until(finished, message="historical upload did not finish")
    assert state["state"] == "done", state
    return state["result"]


def preview(client, payload):
    response = client.post("/api/more-seeds/preview", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(("archive", "model", "source_seed", "child_seed"), [
    ("legacy-v1.zip", KREA, 101, 102),
    ("edit-v2.zip", KONTEXT, 201, 202),
])
def test_more_seeds_replays_the_historical_pass(client, fake, store, monkeypatch, archive, model,
                                                source_seed, child_seed):
    # Occupy both original ids before upload. The replay must use the remap.
    assert [save_source("local", seed=seed).id for seed in (71, 72)] == [1, 2]
    monkeypatch.setattr(model_selection, "_selection", (model, 1))
    fake.edit_qualified = model == KONTEXT
    result = upload(client, archive)
    assert result["unresolved_references"] == 0
    rows = {r.seed: r for r in store.list_images()[0]}
    source, child = rows[source_seed], rows[child_seed]
    assert (source.id, child.id) == (3, 4)
    payload = {"ids": [child.id], "count": 1}
    plan = preview(client, payload)
    batch, = plan["results"]
    assert (batch["status"], batch["first"], batch["last"]) == ("queued", child_seed + 1, child_seed + 1)
    accepted = client.post("/api/more-seeds", json={**payload, "expected": [child.id], "intent": plan["intent"]})
    assert accepted.status_code == 200, accepted.text
    done = stream_job(client, accepted.json()["job_id"])[-1]
    assert done["status"] == "done", done
    saved, = done["images"]
    assert (saved["seed"], saved["run_id"], saved["width"], saved["height"]) == (
        child_seed + 1, RUN_IDS[child_seed], 512, 512,
    )
    request, = fake.requests
    if model == KREA:
        assert (saved["prompt"], saved["refine"], saved["strength"], saved["edit"]) == (
            "blue refinement", 3, 0.375, None,
        )
        assert (request.refine, request.strength, request.edit) == (3, 0.375, None)
    else:
        proof = expected_edit(3)
        assert (saved["prompt"], saved["edit"], saved["refine"]) == ("paint it ochre", proof, None)
        assert request.edit.model_dump() == proof
        # The CPU fake returns the prepared source. Check every pixel it received.
        with Image.open(store.settings.outputs_dir / saved["filename"]) as png:
            assert hashlib.sha256(png.convert("RGB").tobytes()).hexdigest() == proof["edit_source"]["hash"]


@pytest.mark.parametrize("omitted", [True, False], ids=["source-omitted", "same-id-pixel-changed"])
def test_unproven_historical_source_refuses_replay(client, fake, store, monkeypatch, omitted):
    assert save_source("unrelated at the historical source id", seed=71).id == 1
    monkeypatch.setattr(model_selection, "_selection", (KONTEXT, 1))
    fake.edit_qualified = True
    result = upload(client, "edit-only-v2.zip" if omitted else "edit-v2.zip")
    rows = {r.seed: r for r in store.list_images()[0]}
    child = rows[202]
    assert result["unresolved_references"] == int(omitted)
    payload = {"ids": [child.id], "count": 1}
    submit = payload
    if omitted:
        assert child.edit.model_dump() == expected_edit(None)
        reason = "The recorded instruction-edit source has not been resolved to a gallery image."
    else:
        source = rows[201]
        assert source.id == 2 and child.edit.model_dump() == expected_edit(2)
        # A valid preview already exists. Retain the row, name, dimensions and
        # every pixel except one. Submitting that preview must re-prove the source.
        plan = preview(client, payload)
        assert plan["queued"] == 1 and plan["intent"]
        submit = {**payload, "expected": [child.id], "intent": plan["intent"]}
        path = store.settings.outputs_dir / source.filename
        chunks = PngImagePlugin.PngInfo()
        with Image.open(path) as png:
            for key, value in png.text.items():
                chunks.add_text(key, value)
            changed = png.convert("RGB")
        with changed:
            changed.putpixel((511, 511), (17, 80, 94))
            changed.save(path, pnginfo=chunks)
        reason = "Instruction-edit source #2 no longer matches its recorded pixels or provenance."
    refused = client.post("/api/more-seeds", json=submit)
    assert refused.status_code == 422, refused.text
    check = preview(client, payload)
    row, = check["results"]
    assert (check["queued"], check["intent"], row["status"], row["detail"]) == (0, None, "invalid", reason)
    assert fake.loads == 0 and fake.requests == [] and main.jobs.active() == []
    assert store.list_images()[1] == len(rows)
