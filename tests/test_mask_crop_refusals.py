"""Only the masked area's refusals and preview (TODO.md INP#9, slice 5): the
submit check's kind `mask_crop` (a region that gains nothing), a recorded
region its mask no longer decides (Re-render, Swap tags, More seeds), the
grid preview's region, and the command line's --mask-crop, its --from-image
carry beside the mask or outpaint, and the carried source's reset."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from backend import gallery_lock, generate, mask_crop, submission, submission_rules
from backend.schemas import GenerateRequest
from backend.schemas_mask_crop import MASK_CROP_NEEDS_A_MASK, MaskCropSpec
from tests.api_helpers import run_one
from tests.cli_helpers import (  # noqa: F401  (fixtures by name)
    ANIMAGINE,
    cli,
    isolated_execution_environment,
    mask_file,
    project_claiming,
    rewrite_record,
    to_refine,
)
from tests.test_cli_via_server import via_server  # noqa: F401  (a fixture by name)

NO_GAIN = (
    "Only the masked area: the mask and its 32 px padding cover the whole 256×256 picture, which is "
    "no larger than the 1 MP working size; run it as an ordinary inpaint."
)
MISMATCH = (
    "This inpaint's recorded masked area does not match its mask under rule bbox16; it cannot be run again."
)


# ------------------------------------------------------------------ wording


def test_the_refusals_are_the_designs_words():
    spec = MaskCropSpec()
    assert submission_rules.mask_crop_no_gain(spec, 256, 256) == NO_GAIN
    assert submission_rules.mask_crop_no_gain(spec, 384, 384, outpainting=True).endswith(
        "run it as an ordinary outpaint."
    )
    assert submission_rules.mask_crop_mismatch("bbox16") == MISMATCH
    assert submission_rules.mask_crop_mismatch("bbox16", True).startswith("This outpaint's recorded")
    assert submission.UNUSABLE_STATUS["mask_crop"] == "mask_crop"


@pytest.mark.parametrize("status_type", ["SwapStatus", "MoreSeedsStatus"])
def test_the_status_types_accept_a_masked_area(status_type):
    from typing import get_args

    from backend import schemas_rerun

    assert "mask_crop" in get_args(getattr(schemas_rerun, status_type))


def test_the_submit_check_over_a_held_picture(store):
    source = to_refine(store)
    whole = Image.new("L", (256, 384), 255)
    small = Image.new("L", (256, 384), 0)
    small.paste(255, (0, 0, 64, 64))
    request = GenerateRequest(
        prompt="x", width=256, height=384, refine=source.id, strength=0.6, mask="a" * 64,
        mask_crop={"pad": 32},
    )
    assert submission_rules.mask_crop_unusable(request, painted=small) is None
    assert "cover the whole 256×384 picture" in submission_rules.mask_crop_unusable(request, painted=whole)
    # A mask that paints nothing decides no region: said in its own words.
    said = submission_rules.mask_crop_unusable(request, painted=Image.new("L", (256, 384), 0))
    assert said == "Only the masked area: the mask paints nothing"
    # No mask file to read: the region's read is refused in words, not raised.
    assert submission_rules.mask_crop_unusable(request).startswith("Only the masked area: ")


# ------------------------------------------------------------------ over HTTP


def _png(box) -> bytes:
    image = Image.new("L", (256, 256), 0)
    image.paste(255, box)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _source_and_mask(client, box=(0, 0, 128, 64)):
    run_one(client, width=256, height=256)
    record = client.get("/api/gallery").json()["items"][0]
    response = client.post("/api/masks", params={"source": record["id"]}, content=_png(box),
                           headers={"content-type": "image/png"})
    assert response.status_code == 200, response.text
    return record, response.json()["hash"]


def _body(record, digest, **extra):
    return {"prompt": "1girl, red hair, long hair", "width": 256, "height": 256, "steps": 10,
            "refine": record["id"], "strength": 0.6, "mask": digest, "mask_crop": {"pad": 32}, "seed": 9,
            **extra}


def test_a_crop_that_gains_nothing_is_refused_at_submit_and_in_the_preview(client, fake):
    record, digest = _source_and_mask(client, box=(0, 0, 256, 256))
    before = len(fake.requests)
    response = client.post("/api/generate", json=_body(record, digest))
    assert response.status_code == 422 and response.json()["detail"] == NO_GAIN
    response = client.post("/api/grid/preview", json=_body(record, digest))
    assert response.status_code == 422 and response.json()["detail"] == NO_GAIN
    # Uncropped, the same inpaint runs.
    response = client.post("/api/generate", json=_body(record, digest, mask_crop=None))
    assert response.status_code == 200, response.text
    assert len(fake.requests) >= before


def test_the_grid_preview_answers_the_region(client, fake):
    record, digest = _source_and_mask(client)
    response = client.post("/api/grid/preview", json=_body(record, digest))
    assert response.status_code == 200, response.text
    painted = Image.new("L", (256, 256), 0)
    painted.paste(255, (0, 0, 128, 64))
    crop = mask_crop.region(painted, MaskCropSpec())
    assert response.json()["mask_crop"] == {
        "box": list(crop.box), "working": list(crop.working), "scale": round(crop.scale(), 3),
    }
    # Without a crop there is none.
    response = client.post("/api/grid/preview", json=_body(record, digest, mask_crop=None))
    assert response.status_code == 200 and response.json()["mask_crop"] is None


def test_an_outpaints_preview_and_its_refusal(client, fake):
    run_one(client, width=256, height=256)
    record = client.get("/api/gallery").json()["items"][0]
    body = {"prompt": "a wider view", "width": 320, "height": 256, "steps": 10, "refine": record["id"],
            "strength": 1.0, "mask_crop": {"pad": 0}, "seed": 3,
            "outpaint": {"left": 64, "top": 0, "right": 0, "bottom": 0, "band": 16}}
    response = client.post("/api/grid/preview", json=body)
    assert response.status_code == 200, response.text
    from backend import outpaint
    from backend.schemas import Outpaint

    built = outpaint.mask(320, 256, Outpaint(left=64, top=0, right=0, bottom=0, band=16))
    crop = mask_crop.region(built, MaskCropSpec(pad=0))
    assert response.json()["mask_crop"]["box"] == list(crop.box) and crop.box != (0, 0, 320, 256)
    # Padded on every side the box is the whole canvas: nothing gained.
    body.update(width=384, height=384, outpaint={"left": 64, "top": 64, "right": 64, "bottom": 64})
    response = client.post("/api/generate", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["detail"].endswith("run it as an ordinary outpaint.")


def _mismatched(client, store):
    """A cropped inpaint whose recorded region its mask no longer decides."""
    record, digest = _source_and_mask(client)
    _, events = run_one(client, **_body(record, digest))
    assert events[-1]["status"] == "done", events[-1]
    made = events[-1]["images"][0]
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET mask_crop_box = ? WHERE id = ?", ("0,0,256,256>1024x1024", made["id"]))
    return made


def test_a_recorded_region_its_mask_no_longer_decides_cannot_run_again(client, fake, store):
    made = _mismatched(client, store)
    assert submission.mismatched_mask_crops([store.get_image(made["id"])]) == [made["id"]]
    said = f"the recorded masked area of #{made['id']} does not match its mask under rule bbox16"
    before = len(fake.requests)
    response = client.post("/api/rerender", json={"ids": [made["id"]], "width": 256, "height": 256})
    assert response.status_code == 422 and said in response.json()["detail"]
    swap = [{"find": "red hair, long hair", "replace": "silver hair"}]
    swapped = client.post("/api/swap-tags/preview", json={"swaps": swap, "ids": [made["id"]]})
    assert swapped.status_code == 200, swapped.text
    result = next(item for item in swapped.json()["images"] if item["id"] == made["id"])
    assert (result["status"], said in result["detail"]) == ("mask_crop", True)
    assert len(fake.requests) == before


def test_a_mismatched_outpaint_is_sent_to_outpaint_again(client, fake, store):
    run_one(client, width=256, height=256)
    record = client.get("/api/gallery").json()["items"][0]
    _, events = run_one(client, prompt="1girl, red hair, long hair", width=320, height=256, steps=10,
                        refine=record["id"], strength=1.0, mask_crop={"pad": 0}, seed=3,
                        outpaint={"left": 64, "top": 0, "right": 0, "bottom": 0, "band": 16})
    assert events[-1]["status"] == "done", events[-1]
    made = events[-1]["images"][0]
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET mask_crop_box = ? WHERE id = ?", ("0,0,320,256>1144x915", made["id"]))
    remedy = "it cannot be run again; outpaint its source again instead"
    response = client.post("/api/rerender", json={"ids": [made["id"]], "width": 320, "height": 256})
    assert response.status_code == 422 and response.json()["detail"].endswith(remedy), response.text
    swap = [{"find": "red hair, long hair", "replace": "silver hair"}]
    swapped = client.post("/api/swap-tags/preview", json={"swaps": swap, "ids": [made["id"]]})
    result = next(item for item in swapped.json()["images"] if item["id"] == made["id"])
    assert result["status"] == "mask_crop" and result["detail"].endswith(remedy)
    seeds = client.post("/api/more-seeds/preview", json={"ids": [made["id"]], "count": 2})
    batch = next(item for item in seeds.json()["results"] if item["id"] == made["id"])
    assert batch["status"] == "mask_crop" and batch["detail"].endswith(remedy)
    # An inpaint's mask is painted again; a selection of both is told both.
    assert submission_rules.mismatched_mask_crop_detail([4]).endswith("paint the mask on its source again instead")
    assert "or outpaint an outpaint's source again" in submission_rules.mismatched_mask_crop_detail([4, 5], {5})


def test_more_seeds_says_a_mismatched_or_gainless_crop_per_batch(client, fake, store):
    made = _mismatched(client, store)
    response = client.post("/api/more-seeds/preview", json={"ids": [made["id"]], "count": 2})
    assert response.status_code == 200, response.text
    batch = next(item for item in response.json()["results"] if item["id"] == made["id"])
    assert batch["status"] == "mask_crop" and "does not match its mask" in batch["detail"]


def test_a_gainless_row_is_said_per_image_by_the_actions(client, fake, store):
    # A row an older build ran cropped over a mask that covers the whole
    # picture: its submit check's kind becomes the per-image status.
    record, digest = _source_and_mask(client, box=(0, 0, 256, 256))
    made = store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="1girl, red hair, long hair", width=256, height=256, steps=10,
                        refine=record["id"], strength=0.6, mask=digest, mask_crop={"pad": 32}),
        seed=4, duration_ms=10, model_id=store.settings.model_id, job_id="j", run_id="r",
    )
    swap = [{"find": "red hair, long hair", "replace": "silver hair"}]
    swapped = client.post("/api/swap-tags/preview", json={"swaps": swap, "ids": [made.id]})
    assert swapped.status_code == 200, swapped.text
    result = next(item for item in swapped.json()["images"] if item["id"] == made.id)
    assert (result["status"], result["detail"]) == ("mask_crop", NO_GAIN)


# ------------------------------------------------------------- the terminal


def test_mask_crop_from_the_terminal(cli, store, monkeypatch, tmp_path, capsys):
    source = to_refine(store)
    path, image = mask_file(tmp_path)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    crop = mask_crop.region(image, MaskCropSpec())
    argv = ["a red scarf", "--refine", str(source.id), "--mask", str(path), "--mask-crop", "32", "--seed", "4"]
    assert generate.main([*argv, "--dry-run"]) == 0
    assert (
        f"masked   : only the masked area, padding 32 px - the box {','.join(map(str, crop.box))} at "
        f"{crop.working[0]}x{crop.working[1]}, pasted back"
    ) in capsys.readouterr().out
    assert cli.requests == []
    assert generate.main([*argv, "-q"]) == 0
    ran = cli.requests[-1]
    assert (ran.mask_crop, ran.width, ran.height) == (MaskCropSpec(), 256, 384)
    newest = store.list_images()[0][0]
    assert (newest.mask_crop, newest.mask_crop_box, newest.mask_crop_working) == (
        MaskCropSpec(), crop.box, crop.working,
    )


def test_mask_crop_flags_are_refused_where_they_cannot_run(cli, store, monkeypatch, tmp_path, capsys):
    source = to_refine(store)
    path, _ = mask_file(tmp_path)
    whole, _ = mask_file(tmp_path, box=(0, 0, 256, 384), name="whole.png")
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    for argv, words in (
        (["a fox", "--refine", str(source.id), "--mask-crop", "32"], MASK_CROP_NEEDS_A_MASK),
        (["a fox", "--refine", str(source.id), "--mask", str(path), "--mask-crop", "12"], "steps of 8, not 12"),
        (["a fox", "--refine", str(source.id), "--mask", str(path), "--mask-crop", "264"], "0 to 256"),
    ):
        with pytest.raises(SystemExit) as raised:
            generate.main(argv)
        assert raised.value.code == 2, argv
        assert words in capsys.readouterr().err, argv
    # The whole picture, no larger than the working size: the server's words.
    assert generate.main(["a fox", "--refine", str(source.id), "--mask", str(whole), "--mask-crop", "32"]) == 2
    assert "cover the whole 256×384 picture" in capsys.readouterr().err
    assert cli.loads == 0 and cli.requests == []


def test_from_image_carries_the_crop_beside_its_mask(cli, store, monkeypatch, tmp_path, capsys):
    source = to_refine(store)
    path, image = mask_file(tmp_path)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--mask-crop", "16",
                          "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    first = cli.requests[-1]
    assert generate.main(["--from-image", made, "-q"]) == 0
    again = cli.requests[-1]
    assert (again.mask, again.mask_crop) == (first.mask, MaskCropSpec(pad=16))
    # A padding of its own wins over the record's.
    assert generate.main(["--from-image", made, "--mask-crop", "64", "-q"]) == 0
    assert cli.requests[-1].mask_crop == MaskCropSpec(pad=64)
    # A --refine of another image takes neither the mask nor its crop.
    assert generate.main(["--from-image", made, "--refine", str(source.id), "-q"]) == 0
    assert (cli.requests[-1].mask, cli.requests[-1].mask_crop) == (None, None)
    # A recorded region the mask does not decide is refused in the design's words.
    rewrite_record(made, lambda meta: meta["mask_crop"].update(box=[0, 0, 256, 384], working=[832, 1248]))
    assert generate.main(["--from-image", made, "-q"]) == 2
    assert MISMATCH in capsys.readouterr().err
    # With a padding of its own the record's region is not the one asked for.
    assert generate.main(["--from-image", made, "--mask-crop", "32", "-q"]) == 0
    # A region this build cannot read is refused, never dropped.
    rewrite_record(made, lambda meta: meta["mask_crop"].update(box=[0, 0, 256], working=[832, 1248]))
    with pytest.raises(SystemExit):
        generate.main(["--from-image", made, "-q"])
    assert "records a masked area this build cannot read" in capsys.readouterr().err
    # A spec this build lacks is refused by name where it would be carried...
    count = len(cli.requests)
    rewrite_record(made, lambda meta: meta.update(mask_crop={"rule": "bbox32", "pad": 16, "tier": 1}))
    assert generate.main(["--from-image", made, "-q"]) == 2
    err = capsys.readouterr().err
    assert "--from-image: the PNG's masked area:" in err and "bbox16" in err
    # ...and an off-grid padding as the PNG's, not a flag's.
    rewrite_record(made, lambda meta: meta.update(mask_crop={"rule": "bbox16", "pad": 12, "tier": 1}))
    with pytest.raises(SystemExit):
        generate.main(["--from-image", made, "-q"])
    assert "--from-image: the PNG's masked area: a masked-area padding is 0 to 256 in steps of 8, not 12" in (
        capsys.readouterr().err
    )
    assert len(cli.requests) == count
    # Never refusing the file: a padding of its own, a mask of its own or a
    # --refine of another image runs without it.
    rewrite_record(made, lambda meta: meta.update(mask_crop={"rule": "bbox32", "pad": 16, "tier": 1}))
    assert generate.main(["--from-image", made, "--mask-crop", "32", "-q"]) == 0
    assert cli.requests[-1].mask_crop == MaskCropSpec(pad=32)
    assert generate.main(["--from-image", made, "--mask", str(path), "-q"]) == 0
    assert (cli.requests[-1].mask, cli.requests[-1].mask_crop) == (first.mask, None)
    assert generate.main(["--from-image", made, "--refine", str(source.id), "-q"]) == 0
    assert (cli.requests[-1].mask, cli.requests[-1].mask_crop) == (None, None)


@pytest.mark.parametrize("crop", [None, {"rule": "bbox16", "pad": 32, "tier": 1, "box": [0, 0, 64]},
                                  {"rule": "bbox32", "pad": 16, "tier": 1}])
def test_a_dropped_source_drops_the_crop_with_the_mask(cli, store, monkeypatch, tmp_path, capsys, crop):
    source = to_refine(store)
    path, _ = mask_file(tmp_path)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--mask-crop", "32",
                          "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    if crop is not None:
        # One this build cannot read never refuses the plain generation.
        rewrite_record(made, lambda meta: meta.update(mask_crop=crop))
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    assert generate.main(["--from-image", made, "-q"]) == 0
    assert "running as a plain generation instead of an inpaint" in capsys.readouterr().err
    ran = cli.requests[-1]
    assert (ran.refine, ran.mask, ran.mask_crop) == (None, None, None)
    # A padding named on the command line asks for the pass: refused instead.
    with pytest.raises(SystemExit):
        generate.main(["--from-image", made, "--mask-crop", "32", "-q"])
    assert "cannot be run again as an inpaint" in capsys.readouterr().err


def test_an_outpaint_crop_from_the_terminal_and_its_carry(cli, store, monkeypatch, capsys):
    from backend import outpaint
    from backend.schemas import Outpaint

    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "64,0,0,0", "--band", "16",
                          "--mask-crop", "0", "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    first = cli.requests[-1]
    built = outpaint.mask(320, 384, Outpaint(left=64, top=0, right=0, bottom=0, band=16))
    assert first.mask_crop == MaskCropSpec(pad=0)
    assert store.list_images()[0][0].mask_crop_box == mask_crop.region(built, MaskCropSpec(pad=0)).box
    assert generate.main(["--from-image", made, "-q"]) == 0
    assert cli.requests[-1].mask_crop == MaskCropSpec(pad=0)
    rewrite_record(made, lambda meta: meta["mask_crop"].update(box=[0, 0, 320, 384], working=[832, 992]))
    assert generate.main(["--from-image", made, "-q"]) == 2
    assert "This outpaint's recorded masked area does not match" in capsys.readouterr().err
    # Padded both ways the box is the whole canvas: nothing gained.
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0",
                          "--mask-crop", "32", "-q"]) == 2
    assert "run it as an ordinary outpaint." in capsys.readouterr().err


def test_via_server_sends_the_crop_and_carries_it(via_server, fake, store, tmp_path, capsys):
    server = via_server
    source = to_refine(store)
    path, image = mask_file(tmp_path)
    assert generate.main(["a scarf", "--via-server", "--refine", str(source.id), "--mask", str(path),
                          "--mask-crop", "16", "--strength", "0.6", "-q"]) == 0
    body = next(call[2] for call in server.calls if call[1] == "/api/generate")
    assert body["mask_crop"] == {"rule": "bbox16", "pad": 16, "tier": 1}
    assert fake.requests[-1].mask_crop == MaskCropSpec(pad=16)
    made = store.list_images()[0][0]
    assert made.mask_crop_box == mask_crop.region(image, MaskCropSpec(pad=16)).box
    png = str(store.settings.outputs_dir / made.filename)
    assert generate.main(["--via-server", "--from-image", png, "-q"]) == 0
    assert fake.requests[-1].mask_crop == MaskCropSpec(pad=16)
    # The server refuses a crop that gains nothing, in its words.
    whole, _ = mask_file(tmp_path, box=(0, 0, 256, 384), name="whole.png")
    assert generate.main(["a scarf", "--via-server", "--refine", str(source.id), "--mask", str(whole),
                          "--mask-crop", "32", "-q"]) != 0
    assert "cover the whole 256×384 picture" in capsys.readouterr().err
