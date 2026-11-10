"""The stable `code` beside `detail` on the refusals the page acts on, and
the page reading the code rather than the sentence.

Each test pins the sentence as it is today beside its code: the sentence is
the person's and may be reworded (this file changes with it), while a code,
once the page reads it, does not.
"""
from __future__ import annotations

import dataclasses
import io
import re
from pathlib import Path

import pytest
from PIL import Image

from backend import main, refusal_codes, refusals, storage
from backend.config import settings
from backend.schemas import GenerateRequest
from tests.api_helpers import run_one

WEB = Path(__file__).resolve().parent.parent / "web"


def _plant(store, seed=1, model_id=None):
    return store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="a lighthouse in fog", width=256, height=256),
        seed=seed, duration_ms=1, model_id=model_id or settings.model_id,
    )


def _refused(response, status, code, ids=None):
    assert response.status_code == status, response.text
    body = response.json()
    assert body["code"] == code, body
    assert isinstance(body["detail"], str) and body["detail"]
    if ids is None:
        assert "ids" not in body, body
    else:
        assert body["ids"] == ids, body
    return body["detail"]


def _png(size=(256, 256)) -> bytes:
    image = Image.new("L", size, 0)
    image.paste(255, (0, 0, 128, 64))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


# ------------------------------------------------------------ one image, by id


def test_an_image_route_says_unknown_or_binned_by_code(client, store):
    live, binned = _plant(store, 1), _plant(store, 2)
    client.post("/api/images/bin", json={"ids": [binned.id]})

    assert _refused(client.get("/api/images/9999"), 404, "image_unknown", [9999]) == "unknown image"
    assert _refused(client.get(f"/api/images/{binned.id}"), 404, "image_binned", [binned.id]) == (
        f"unknown image (#{binned.id} is in the recycle bin)"
    )
    _refused(client.get(f"/api/images/{binned.id}/thumb", follow_redirects=False), 404, "image_binned", [binned.id])
    # Asked for with the bin's rows, only a row that is nowhere is unknown.
    assert client.get(f"/api/images/{binned.id}?binned=true").status_code == 200
    _refused(client.get("/api/images/9999?binned=true"), 404, "image_unknown", [9999])
    # A star cannot tell a binned image from a deleted one: gone either way.
    _refused(client.post(f"/api/images/{binned.id}/favorite?value=true"), 404, "image_unknown", [binned.id])
    _refused(client.post("/api/images/9999/favorite?value=true"), 404, "image_unknown", [9999])
    assert client.delete(f"/api/images/{live.id}").status_code == 200
    _refused(client.delete(f"/api/images/{live.id}"), 404, "image_unknown", [live.id])


def test_the_outpaint_presets_say_unknown_or_binned_by_code(client, store):
    binned = _plant(store)
    client.post("/api/images/bin", json={"ids": [binned.id]})
    assert _refused(client.get("/api/images/9999/outpaint-presets"), 404, "image_unknown", [9999]) == (
        "no gallery image with id 9999"
    )
    _refused(client.get(f"/api/images/{binned.id}/outpaint-presets"), 404, "image_binned", [binned.id])


# ------------------------------------------------- a submission's own inputs


def test_a_submission_names_its_gone_images_by_code_and_id(client, fake, store):
    body = {"prompt": "x", "width": 256, "height": 256}
    assert _refused(
        client.post("/api/generate", json={**body, "reference": [4242]}), 422, "image_unknown", [4242],
    ) == "no gallery image with id 4242"
    run_one(client)
    first = client.get("/api/gallery").json()["items"][0]
    client.post("/api/images/bin", json={"ids": [first["id"]]})
    pass_body = {**body, "steps": 10, "refine": first["id"], "strength": 0.5}
    assert _refused(client.post("/api/generate", json=pass_body), 422, "image_binned", [first["id"]]) == (
        f"source image {first['id']} is in the recycle bin; restore it to use it"
    )
    # The pass dialogs' preview is the grid's, refused the same way.
    _refused(client.post("/api/grid/preview", json=pass_body), 422, "image_binned", [first["id"]])
    assert client.delete(f"/api/images/{first['id']}").status_code == 200
    _refused(client.post("/api/grid/preview", json=pass_body), 422, "image_unknown", [first["id"]])
    assert fake.requests and len(fake.requests) == 1


def test_a_pass_through_the_upscaler_names_its_gone_source_by_code_too(client, fake, store, monkeypatch):
    # The upscaler's own look at the source comes before the pass's: the same
    # refusal, so the Refine dialog says a deleted or binned source the same.
    from backend import refine

    monkeypatch.setattr(main.model_selection, "_selection", ("LyliaEngine/waiIllustriousSDXL_v170", 0))
    rec = _plant(store, model_id="LyliaEngine/waiIllustriousSDXL_v170")
    body = {
        "prompt": "x", "width": 256, "height": 256, "steps": 20,
        "refine": rec.id, "strength": 0.4, "upscaler": refine.ANIME6B,
    }
    client.post("/api/images/bin", json={"ids": [rec.id]})
    assert _refused(client.post("/api/grid/preview", json=body), 422, "image_binned", [rec.id]) == (
        f"source image {rec.id} is in the recycle bin; restore it to use it"
    )
    assert client.delete(f"/api/images/{rec.id}").status_code == 200
    assert _refused(client.post("/api/generate", json=body), 422, "image_unknown", [rec.id]) == (
        f"no gallery image with id {rec.id}"
    )
    assert fake.requests == []


def test_an_adapter_is_refused_by_code_missing_or_not_fitting(client, fake, tmp_path, monkeypatch):
    from backend import loras

    directory = tmp_path / "loras"
    directory.mkdir()
    (directory / "style.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(loras, "settings", dataclasses.replace(loras.settings, loras_dir=directory))
    body = {"prompt": "x", "width": 256, "height": 256}

    said = _refused(
        client.post("/api/generate", json={**body, "lora": "missing.safetensors"}), 422, "lora_missing",
    )
    assert "no LoRA named 'missing.safetensors'" in said
    fake.lora_mismatches["style.safetensors"] = "style.safetensors is another checkpoint's"
    assert _refused(
        client.post("/api/generate", json={**body, "lora": "style.safetensors"}), 422, "lora_mismatch",
    ) == "style.safetensors is another checkpoint's"
    assert fake.requests == []


def test_a_mask_that_has_gone_and_its_source_are_said_by_code(client, fake, store):
    run_one(client)
    record = client.get("/api/gallery").json()["items"][0]
    inpaint = {
        "prompt": "a red scarf", "width": 256, "height": 256, "steps": 10,
        "refine": record["id"], "strength": 0.6, "mask": "0" * 64,
    }
    assert _refused(client.post("/api/generate", json=inpaint), 422, "mask_missing").startswith(
        "no stored mask 0000000000000000"
    )
    upload = {"content": _png(), "headers": {"content-type": "image/png"}}
    assert _refused(
        client.post("/api/masks", params={"source": 9999}, **upload), 404, "image_unknown", [9999],
    ) == "No gallery image with id 9999."
    client.post("/api/images/bin", json={"ids": [record["id"]]})
    assert _refused(
        client.post("/api/masks", params={"source": record["id"]}, **upload), 422, "image_binned", [record["id"]],
    ) == f"Source image {record['id']} is in the recycle bin; restore it to use it."


def test_a_mask_gone_between_the_submissions_two_looks_keeps_its_code(client, fake, store, monkeypatch):
    # Swept after `unusable_input` found it and before `mask_unusable` read
    # it: still the mask's refusal, which the Inpaint dialog answers by
    # sending the mask again.
    run_one(client)
    record = client.get("/api/gallery").json()["items"][0]
    real, looks = storage.mask_path, []

    def swept_after_the_first_look(mask):
        looks.append(mask)
        return real("0" * 64) if len(looks) > 1 else Path("held.png")

    monkeypatch.setattr(storage, "mask_path", swept_after_the_first_look)
    inpaint = {
        "prompt": "a red scarf", "width": 256, "height": 256, "steps": 10,
        "refine": record["id"], "strength": 0.6, "mask": "1" * 64,
    }
    said = _refused(client.post("/api/generate", json=inpaint), 422, "mask_missing")
    assert said.startswith("no stored mask 0000000000000000") and len(looks) >= 2
    assert len(fake.requests) == 1


def test_a_mask_gone_between_the_check_and_the_hold_keeps_its_code(client):
    from backend.jobs import MaskGone

    @main.app.get("/api/test-only/mask-gone", include_in_schema=False)
    async def gone():
        raise MaskGone("mask 0123456789abcdef has just gone from this gallery; paint it again")

    try:
        said = _refused(client.get("/api/test-only/mask-gone"), 422, "mask_missing")
        assert said == "Mask 0123456789abcdef has just gone from this gallery; paint it again."
    finally:
        main.app.router.routes[:] = [
            route for route in main.app.router.routes if getattr(route, "path", "") != "/api/test-only/mask-gone"
        ]


def test_an_edit_says_whether_its_source_is_deleted_or_binned_by_code(client, store):
    rec = _plant(store)
    sentence = "Instruction-edit source #{} is missing or in the recycle bin."
    ask = lambda image_id: client.post(  # noqa: E731
        "/api/instruction-edit/preview", json={"source": image_id, "recipe": "kontext-v1"},
    )
    assert _refused(ask(9999), 422, "image_unknown", [9999]) == sentence.format(9999)
    client.post("/api/images/bin", json={"ids": [rec.id]})
    # One sentence for both; only the code says which.
    assert _refused(ask(rec.id), 422, "image_binned", [rec.id]) == sentence.format(rec.id)


# ------------------------------------------------------------- gallery actions


def test_more_seeds_says_its_cap_and_its_receipt_by_code(client, fake, store, monkeypatch):
    run_one(client, seed=1)
    image = client.get("/api/gallery").json()["items"][0]
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, max_grid_images=3))
    assert _refused(
        client.post("/api/more-seeds/preview", json={"ids": [image["id"]], "count": 4}), 422, "more_seeds_cap",
    ) == "1 batch × 4 is 4 images; a press may queue 3. Ask for fewer, or pick fewer batches."
    assert _refused(
        client.post("/api/more-seeds", json={"ids": [image["id"]], "count": 1, "intent": "no-such-receipt"}),
        409, "more_seeds_receipt",
    ) == "this More seeds receipt expired or belongs to another server; review the batches again"
    # Another 409 of the same route is no receipt's: shown, never acted on.
    run_one(client, seed=9, prompt="a harbour at dusk")
    other = client.get("/api/gallery").json()["items"][0]
    changed = client.post("/api/more-seeds", json={"ids": [image["id"]], "count": 1, "expected": [other["id"]]})
    assert changed.status_code == 409 and "code" not in changed.json(), changed.text


def test_delete_unstarred_says_a_stale_count_by_code(client, store):
    _plant(store, 1)
    _plant(store, 2)
    said = _refused(client.delete("/api/gallery/unstarred?expected=3"), 409, "selection_changed")
    assert said.startswith("The selection changed: 2 unstarred images match now, not the 3 confirmed.")


def test_re_render_marks_a_left_out_pick_as_binned_beside_its_words(client, fake, store):
    kept, binned, deleted = (_plant(store, seed) for seed in (1, 2, 3))
    client.post("/api/images/bin", json={"ids": [binned.id]})
    assert client.delete(f"/api/images/{deleted.id}").status_code == 200
    answer = client.post("/api/rerender", json={"ids": [kept.id, binned.id, deleted.id], "width": 512, "height": 512})
    assert answer.status_code == 200, answer.text
    assert answer.json()["deleted"] == [
        {"id": binned.id, "detail": "in the recycle bin", "binned": True},
        {"id": deleted.id, "detail": "no longer in the gallery", "binned": False},
    ]


# ------------------------------------------------------------------ the shape


def test_a_refusal_the_page_only_shows_carries_no_code(client):
    response = client.post("/api/jobs/nope/cancel")
    assert response.status_code == 404 and response.json() == {"detail": "unknown job"}


def test_a_coded_refusal_is_an_ordinary_http_refusal_with_its_code():
    refused = refusals.Refused(404, "unknown image", refusal_codes.IMAGE_UNKNOWN, [7])
    assert (refused.status_code, refused.detail) == (404, "unknown image")
    assert refused.body() == {"detail": "unknown image", "code": "image_unknown", "ids": [7]}
    # From an error below the HTTP layer: its code and ids ride; a plain
    # error's sentence goes out alone.
    coded = refusals.refusal(422, "said", refusal_codes.CodedError("said", refusal_codes.IMAGE_BINNED, [3]))
    assert isinstance(coded, refusals.Refused) and (coded.code, coded.ids) == ("image_binned", (3,))
    assert not isinstance(refusals.refusal(422, "said", ValueError("said")), refusals.Refused)
    with pytest.raises(AssertionError):
        refusals.Refused(404, "unknown image", "not_a_code")


def test_storage_and_the_adapter_directory_raise_coded_value_errors(store, tmp_path):
    binned = _plant(store)
    store.bin_images([binned.id])
    with pytest.raises(ValueError, match="no gallery image with id 404, 405") as unknown:
        store.reference_paths([404, 405])
    assert (unknown.value.code, unknown.value.ids) == ("image_unknown", (404, 405))
    with pytest.raises(ValueError, match="is in the recycle bin") as gone:
        store.reference_paths([binned.id], "source image")
    assert (gone.value.code, gone.value.ids) == ("image_binned", (binned.id,))
    with pytest.raises(ValueError, match="no stored mask") as mask:
        storage.mask_path("0" * 64)
    assert mask.value.code == "mask_missing"


# ------------------------------------------------------------------- the page


def _scripts() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(WEB.glob("*.js"))}


def test_the_page_reads_every_code_the_server_gives_and_no_other():
    # A code nobody reads is dead weight; one the page reads that the server
    # never gives is a branch that cannot run.
    text = "\n".join(_scripts().values())
    read = set(re.findall(r"err\??\.code [!=]== '([a-z_]+)'", text))
    assert read == set(refusal_codes.CODES)


def test_the_page_decides_nothing_from_a_refusals_sentence():
    # What the review listed: `err.message` tested for the server's English.
    # A refusal's sentence is shown; what is done about it is its code's.
    wording = re.compile(
        r"\.(?:test|exec)\(\s*(?:err|e|error|failure|refusal)\??\.message"
        r"|(?:err|e|error|failure|refusal)\??\.message\??\.(?:includes|startsWith|endsWith|match|search|indexOf)\("
    )
    sentences = re.compile(
        r"/[^/\n]*(?:unknown image|no gallery image|recycle bin|\\d\+ batch|selection changed|stored mask|receipt)[^/\n]*/[a-z]*\.(?:test|exec)\("
    )
    found = [
        f"{name}:{number}: {line.strip()}"
        for name, source in _scripts().items()
        for number, line in enumerate(source.splitlines(), 1)
        if wording.search(line) or sentences.search(line)
    ]
    assert found == []


def test_the_pages_error_carries_the_code_and_its_readers_use_it():
    state = _scripts()["state.js"]
    assert "const code = typeof body?.code === 'string' ? body.code : null;" in state
    assert "{ status: res.status, body, code }" in state
    for reader in ("function imageGone(err)", "function imageBinned(err)", "function refusedIds(err)",
                   "function loraRefused(err)"):
        assert reader in state
