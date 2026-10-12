"""Reference images and LoRA adapters over HTTP: refused at submit, named
when unknown, recorded on the row at the weight they ran at."""
from __future__ import annotations

import pytest

from tests.api_helpers import run_one, stream_job


# ------------------------------------------------------- conditioning inputs


@pytest.fixture
def lora_dir(tmp_path, monkeypatch):
    """An adapter directory the server can list and resolve names against."""
    import dataclasses

    from backend import loras

    directory = tmp_path / "loras"
    directory.mkdir()
    (directory / "style.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        loras, "settings", dataclasses.replace(loras.settings, loras_dir=directory)
    )
    return directory


def test_status_publishes_both_conditioning_capabilities(client, fake):
    body = client.get("/api/status").json()
    assert body["defaults"]["reference_supported"] is True
    assert body["defaults"]["lora_supported"] is True
    # And the tooltips cannot contradict the greyed-out controls.
    live = (body["hints"]["reference"], body["hints"]["lora"])
    assert all(live)

    fake.reference_supported = False
    fake.lora_reason = "no peft here"
    body = client.get("/api/status").json()
    assert body["defaults"]["reference_supported"] is False
    assert body["defaults"]["lora_supported"] is False
    # A dead control explains itself rather than repeating what a live one says.
    assert (body["hints"]["reference"], body["hints"]["lora"]) != live
    assert "klein" in body["hints"]["reference"]
    assert "peft" in body["hints"]["lora"]


def test_references_are_refused_at_submit_not_at_cell_one(client, fake, store):
    # A 64-cell night must not discover at the first image that the pipeline
    # ignores `image` and every cell is a plain text-to-image.
    fake.reference_supported = False
    response = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "reference": [1]}
    )
    assert response.status_code == 422 and "klein" in response.json()["detail"]
    assert fake.requests == []


def test_an_unknown_reference_id_is_named(client, fake, store):
    response = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "reference": [4242]}
    )
    assert response.status_code == 422
    assert "no gallery image with id 4242" in response.json()["detail"]


def test_a_binned_reference_is_refused_in_the_words_the_page_reads(client, fake, store):
    # The page takes the chip out on this sentence (dropRefusedReferences),
    # by its start: "reference image N", N the gallery id, then the bin.
    run_one(client)
    run_one(client)
    newest, older = (item["id"] for item in client.get("/api/gallery").json()["items"][:2])
    assert client.post("/api/images/bin", json={"ids": [older]}).json()["binned"] == [older]
    response = client.post(
        "/api/generate",
        json={"prompt": "x", "width": 256, "height": 256, "reference": [newest, older]},
    )
    assert response.status_code == 422
    assert response.json()["detail"].startswith(f"reference image {older} is in the recycle bin; ")


def test_a_reference_reaches_the_backend_and_the_row(client, fake, store):
    run_one(client)  # something to point at
    first = client.get("/api/gallery").json()["items"][0]
    run_one(client, reference=[first["id"]])
    assert fake.requests[-1].reference == [first["id"]]
    newest = client.get("/api/gallery").json()["items"][0]
    assert newest["reference"] == [first["id"]]


def test_a_reference_chip_can_turn_an_id_into_a_picture(client, fake, store):
    # The panel's chips hold ids and nothing else - one restored from a record
    # or from localStorage may name an image no loaded gallery page contains -
    # so an id has to be enough to show the thumbnail.
    run_one(client)
    record = client.get("/api/gallery").json()["items"][0]
    response = client.get(f"/api/images/{record['id']}/thumb", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] in (record["thumb_url"], record["url"])
    assert client.get("/api/images/4242/thumb").status_code == 404


def test_an_adapter_is_refused_with_the_reason(client, fake, lora_dir):
    fake.lora_reason = "LoRA adapters need the peft package"
    response = client.post(
        "/api/generate",
        json={"prompt": "x", "width": 256, "height": 256, "lora": "style.safetensors"},
    )
    assert response.status_code == 422 and "peft" in response.json()["detail"]


def test_an_adapter_that_is_not_there_is_refused_by_name(client, fake, lora_dir):
    response = client.post(
        "/api/generate",
        json={"prompt": "x", "width": 256, "height": 256, "lora": "missing.safetensors"},
    )
    assert response.status_code == 422 and "no LoRA named" in response.json()["detail"]
    # Named, which is what the page reads the list again on (dropRefusedLora).
    assert "missing.safetensors" in response.json()["detail"]
    # And a path never reaches the directory at all - the schema refuses it.
    assert client.post(
        "/api/generate",
        json={"prompt": "x", "width": 256, "height": 256, "lora": "../../keys.safetensors"},
    ).status_code == 422


def test_the_adapter_listing_is_the_directory(client, fake, lora_dir):
    body = client.get("/api/loras").json()
    assert [entry["name"] for entry in body["loras"]] == ["style.safetensors"]
    assert body["supported"] is True and body["detail"] is None
    assert body["loras"][0]["detail"] is None


def test_an_adapter_for_another_checkpoint_is_listed_with_why_and_refused(client, fake, lora_dir):
    # The directory is the directory; what does not fit the checkpoint rides
    # with the reason, which is what the page hides it on and the tooltip
    # shows. It is not offered and it is not accepted: a FLUX.1 adapter on
    # Z-Image would otherwise fail at cell 1 of 64, deep in PEFT.
    (lora_dir / "flux.safetensors").write_bytes(b"weights")
    why = "flux.safetensors attaches to 494 modules ZImageTransformer2DModel has not got"
    fake.lora_mismatches = {"flux.safetensors": why}
    body = client.get("/api/loras").json()
    assert {entry["name"]: entry["detail"] for entry in body["loras"]} == {
        "flux.safetensors": why, "style.safetensors": None,
    }
    for route, extra in (("/api/generate", {}), ("/api/grid", {"seed": 1, "lora_scale_axis": "0,1"})):
        response = client.post(
            route, json={"prompt": "x", "width": 256, "height": 256, "lora": "flux.safetensors", **extra},
        )
        assert response.status_code == 422 and response.json()["detail"] == why
    assert fake.requests == []
    # Where no adapter can be applied at all, the fit is not asked about.
    fake.lora_reason = "no peft here"
    body = client.get("/api/loras").json()
    assert body["detail"] == "no peft here"
    assert all("detail" not in entry for entry in body["loras"])


def test_a_weight_sweep_records_the_weight_it_ran_at(client, fake, lora_dir):
    response = client.post(
        "/api/grid",
        json={
            "prompt": "a fox", "width": 256, "height": 256, "seed": 1,
            "lora": "style.safetensors", "lora_scale_axis": "0,0.5,1",
        },
    )
    assert response.status_code == 200
    stream_job(client, response.json()["job_id"])
    assert [r.lora_scale for r in fake.requests] == [0.0, 0.5, 1.0]
    rows = client.get("/api/gallery").json()["items"]
    assert {row["lora_scale"] for row in rows} == {0.0, 0.5, 1.0}
    assert {row["lora"] for row in rows} == {"style.safetensors"}
