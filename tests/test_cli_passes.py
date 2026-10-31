"""The command line's refine, inpaint and outpaint passes, run locally.

The sources come from the same throwaway gallery the browser would use, and
each pass is checked against the cells the server would run.
"""
from __future__ import annotations

import io

import pytest

from backend import gallery_lock, generate, storage

from tests.cli_helpers import (  # noqa: F401  (fixtures by name)
    ANIMAGINE,
    SIZE,
    mask_file,
    project_claiming,
    rewrite_record,
    to_refine,
    cli,
    isolated_execution_environment,
)


# ------------------------------------------------------------ the refine pass


def test_a_refine_renders_at_the_sources_size_times_the_scale(cli, store, capsys):
    from backend.schemas import default_strength

    source = to_refine(store)
    assert generate.main(["a fox", "--refine", str(source.id), "--steps", "20", "--seed", "7"]) == 0
    ran = cli.requests[-1]
    # No size given: the source's own, times the default 1.5. No strength
    # given: the checkpoint's own starting one, and the header says how
    # many of the twenty steps that leaves to run.
    start = default_strength()
    assert (ran.refine, ran.strength, ran.width, ran.height) == (source.id, start, 384, 576)
    out = capsys.readouterr().out
    assert (
        f"refine   : image {source.id} at strength {start:g}  "
        f"(about {int(20 * start)} of 20 steps; source 256x384)"
    ) in out
    newest = store.list_images()[0][0]
    assert (newest.refine, newest.strength, newest.width, newest.height) == (source.id, start, 384, 576)

    assert generate.main(
        ["a fox", "--refine", str(source.id), "--scale", "2", "--strength", "0.3", "--steps", "10", "-q"]
    ) == 0
    ran = cli.requests[-1]
    assert (ran.strength, ran.width, ran.height) == (0.3, 512, 768)
    # A size given outright is the size: the source is cropped to it.
    assert generate.main(["a fox", "--refine", str(source.id), "--steps", "20", "-q", *SIZE]) == 0
    assert (cli.requests[-1].width, cli.requests[-1].height) == (256, 256)


def test_refine_flags_need_a_source_and_one_way_of_sizing(cli, store):
    source = to_refine(store)
    for argv in (
        ["a fox", "--strength", "0.3", *SIZE],
        ["a fox", "--scale", "2"],
        ["a fox", "--grid-strength", "0.2,0.4", *SIZE],
        ["a fox", "--refine", str(source.id), "--scale", "2", *SIZE],
        ["a fox", "--refine", str(source.id), "--scale", "0"],
        ["a fox", "--refine", str(source.id), "--scale", "nan"],
        ["a fox", "--refine", str(source.id), "--scale", "inf"],
        ["a fox", "--refine", str(source.id), "--scale=-inf"],
        ["a fox", "--refine", str(source.id), "--scale", "1e308"],
    ):
        with pytest.raises(SystemExit) as raised:
            generate.main(argv)
        assert raised.value.code == 2, argv
    assert cli.requests == []


def test_a_refine_is_refused_before_the_load(cli, store, monkeypatch, capsys):
    source = to_refine(store)
    # A source the gallery has not got, named - with or without a size.
    assert generate.main(["a fox", "--refine", "4242"]) == 2
    assert "no gallery image with id 4242" in capsys.readouterr().err
    assert generate.main(["a fox", "--refine", "4242", *SIZE]) == 2
    assert "no gallery image with id 4242" in capsys.readouterr().err
    # A strength that leaves no step of a distilled checkpoint's few.
    assert generate.main(["a fox", "--refine", str(source.id), "--steps", "4", "--strength", "0.2"]) == 2
    assert "less than one step" in capsys.readouterr().err
    # And a checkpoint diffusers has no image-to-image pipeline for: the same
    # refusal the server makes at submit, from the profile, ahead of the load.
    monkeypatch.setattr(generate, "_project", project_claiming(generate, "krea/Krea-2-Turbo"))
    assert generate.main(["a fox", "--refine", str(source.id)]) == 2
    assert "has no refine pass" in capsys.readouterr().err
    assert cli.loads == 0 and cli.requests == []


def test_from_image_reproduces_a_refine_pass(cli, store, capsys):
    source = to_refine(store)
    assert generate.main(
        ["a fox, sharper", "--refine", str(source.id), "--strength", "0.3", "--steps", "10", "--seed", "7", "-q"]
    ) == 0
    path = capsys.readouterr().out.split("  seed ")[0]
    # The PNG carries the source, the strength and its own size, so nothing
    # is derived again: the same pass, cell for cell.
    assert generate.main(["--from-image", path, "-q"]) == 0
    first, again = cli.requests[-2:]
    assert (again.refine, again.strength, again.width, again.height, again.steps) == (
        first.refine, first.strength, first.width, first.height, first.steps
    )
    assert cli.seeds[-2:] == [7, 7]
    # A flag still wins over the file, and --scale over the file's size.
    assert generate.main(["--from-image", path, "--strength", "0.6", "--scale", "2", "-q"]) == 0
    ran = cli.requests[-1]
    assert (ran.refine, ran.strength, ran.width, ran.height) == (source.id, 0.6, 512, 768)


def test_a_strength_sweep_from_the_terminal(cli, store, capsys):
    source = to_refine(store)
    argv = ["a fox", "--refine", str(source.id), "--steps", "20", "--seed", "1",
            "--grid-strength", "0.2-0.6x0.2"]
    assert generate.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "strength: 0.2, 0.4, 0.6" in out and "3 strength" in out
    assert generate.main([*argv, "-q"]) == 0
    assert [r.strength for r in cli.requests] == [0.2, 0.4, 0.6]
    assert {r.strength for r in store.list_images(search=f"refine:{source.id}")[0]} == {0.2, 0.4, 0.6}
    # Resumed, all three are in the gallery under the keys the cells carry.
    assert generate.main([*argv, "--resume", "-q"]) == 0
    assert len(cli.requests) == 3


@pytest.mark.parametrize("axes", [
    ["--steps", "8", "--grid-strength", "0.3,0.5"],
    ["--steps", "4", "--grid-steps", "20,28"],
])
def test_refine_cli_checks_actual_cells_after_axes_replace_base(cli, store, axes):
    source = to_refine(store)
    argv = ["a fox", "--refine", str(source.id), "--strength", "0.1", *axes]
    assert generate.main([*argv, "--dry-run"]) == 0
    assert cli.loads == 0
    assert generate.main([*argv, "-q"]) == 0
    assert len(cli.requests) == 2


def test_cli_rechecks_refine_after_loading_before_any_cell(cli, store, monkeypatch, capsys):
    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, "example/flux-custom-pipeline"))
    load = cli.load

    def unsupported_load():
        load()
        cli.refine_supported = False

    monkeypatch.setattr(cli, "load", unsupported_load)
    assert generate.main(["a fox", "--refine", str(source.id),
                          "--grid-strength", "0.2,0.4", "--seed", "1", "-q"]) == 2
    assert "has no refine pass" in capsys.readouterr().err
    assert cli.loads == 1 and not cli.loaded and cli.requests == []
    assert store.list_images()[1] == 1


@pytest.mark.parametrize("model", ["LyliaEngine/waiIllustriousSDXL_v170", "Minthy/RouWei-0.8"])
def test_gan_cli_is_opt_in_and_from_image_can_carry_override_or_drop_it(cli, store, monkeypatch, capsys, model):
    from backend import model_selection, refine

    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    model_selection.select(model)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, model))
    source = to_refine(store)
    argv = ["a fox", "--refine", str(source.id), "--upscaler", refine.ANIME6B, "--seed", "7"]
    assert generate.main([*argv, "--dry-run"]) == 0
    assert cli.loads == 0 and "upscaler : R-ESRGAN 4x+ Anime6B" in capsys.readouterr().out
    assert generate.main([*argv, "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0].strip()
    assert cli.requests[-1].upscaler == refine.ANIME6B
    assert generate.main(["--from-image", made, "-q"]) == 0
    assert cli.requests[-1].upscaler == refine.ANIME6B
    capsys.readouterr()
    assert generate.main(["--from-image", made, "--upscaler", "lanczos", "-q"]) == 0
    assert cli.requests[-1].upscaler is None
    capsys.readouterr()
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    assert generate.main(["--from-image", made, "-q"]) == 0
    assert cli.requests[-1].refine is None and cli.requests[-1].upscaler is None


def test_gan_cli_refuses_wrong_model_and_oversized_source_before_load(cli, store, monkeypatch, capsys):
    from backend import model_selection, refine

    source = to_refine(store)
    assert generate.main(["a fox", "--refine", str(source.id), "--upscaler", refine.ANIME6B, "--dry-run"]) == 2
    assert "only for WAI" in capsys.readouterr().err and cli.loads == 0
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    model_selection.select("LyliaEngine/waiIllustriousSDXL_v170")
    monkeypatch.setattr(generate, "_project", project_claiming(generate, "LyliaEngine/waiIllustriousSDXL_v170"))
    monkeypatch.setattr(refine, "check_upscaler_source", lambda image_id: refine.check_upscaler_size(4097, 256))
    assert generate.main(["a fox", "--refine", str(source.id), "--upscaler", refine.ANIME6B, "--dry-run"]) == 2
    assert "4.19 MP" in capsys.readouterr().err and cli.loads == 0 and not cli.requests


# ------------------------------------------------------------- the inpaint


@pytest.mark.parametrize("record_key, mask_key", [
    ("auradream", "auradream-mask"), ("local-diffusion", "local-diffusion-mask"),
])
def test_from_image_reads_a_pre_rename_record_and_its_mask(tmp_path, record_key, mask_key):
    import argparse
    import base64
    import json

    from PIL import Image, PngImagePlugin

    _, mask = mask_file(tmp_path)
    digest = storage.mask_hash(mask)
    encoded = io.BytesIO()
    mask.save(encoded, format="PNG")
    meta = {"prompt": "a red scarf", "steps": 10, "guidance_scale": 5.0, "seed": 4,
            "model": ANIMAGINE, "refine": 1, "strength": 0.6, "mask": digest}
    info = PngImagePlugin.PngInfo()
    info.add_text(record_key, json.dumps(meta))
    info.add_text(mask_key, base64.b64encode(encoded.getvalue()).decode("ascii"))
    path = tmp_path / "old.png"
    Image.new("RGB", (256, 384)).save(path, pnginfo=info)

    assert generate._image_settings(str(path), argparse.ArgumentParser()) == meta
    found = generate._mask_from_record(str(path), digest)
    assert found is not None and storage.mask_hash(found) == digest


def test_an_inpaint_from_the_terminal_stores_its_mask_and_records_it(cli, store, monkeypatch, tmp_path, capsys):
    source = to_refine(store)
    path, image = mask_file(tmp_path)
    digest = store.mask_hash(image)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    argv = ["a red scarf", "--refine", str(source.id), "--mask", str(path), "--strength", "0.6", "--seed", "4"]
    # A dry run says what would run and keeps nothing.
    assert generate.main([*argv, "--dry-run"]) == 0
    assert f"inpaint  : under mask {digest[:16]}" in capsys.readouterr().out
    assert store.mask_size(digest) is None and cli.loads == 0
    assert generate.main([*argv, "-q"]) == 0
    ran = cli.requests[-1]
    # At the source's own size, never scaled, under the mask's hash.
    assert (ran.refine, ran.mask, ran.strength, ran.width, ran.height) == (source.id, digest, 0.6, 256, 384)
    assert store.mask_path(digest).is_file()
    newest = store.list_images()[0][0]
    assert (newest.refine, newest.mask) == (source.id, digest)


def test_mask_flags_are_refused_where_they_cannot_run(cli, store, monkeypatch, tmp_path, capsys):
    source = to_refine(store)
    path, _ = mask_file(tmp_path)
    black, _ = mask_file(tmp_path, box=(0, 0, 0, 0), name="black.png")
    wrong, _ = mask_file(tmp_path, size=(256, 256), name="square.png")
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    for argv in (
        ["a fox", "--mask", str(path), *SIZE],
        ["a fox", "--refine", str(source.id), "--mask", str(path), "--scale", "2"],
        ["a fox", "--refine", str(source.id), "--mask", str(black)],
        ["a fox", "--refine", str(source.id), "--mask", str(tmp_path / "nowhere.png")],
    ):
        with pytest.raises(SystemExit) as raised:
            generate.main(argv)
        assert raised.value.code == 2, argv
    capsys.readouterr()
    # A mask painted on another size, said in words, before the load.
    assert generate.main(["a fox", "--refine", str(source.id), "--mask", str(wrong)]) == 2
    assert "is 256x256, but source image" in capsys.readouterr().err
    # And a checkpoint whose inpaint is not enabled, from its profile.
    monkeypatch.setattr(generate, "_project", project_claiming(generate, "black-forest-labs/FLUX.2-klein-4B"))
    assert generate.main(["a fox", "--refine", str(source.id), "--mask", str(path)]) == 2
    assert "has no inpaint in this app" in capsys.readouterr().err
    assert cli.loads == 0 and cli.requests == []


def test_from_image_reproduces_an_inpaint_over_its_mask(cli, store, monkeypatch, tmp_path, capsys):
    source = to_refine(store)
    path, image = mask_file(tmp_path)
    digest = store.mask_hash(image)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    # The mask file lost since: the PNG's own copy is the same picture.
    store.mask_path(digest).unlink()
    assert generate.main(["--from-image", made, "-q"]) == 0
    first, again = cli.requests[-2:]
    assert (again.refine, again.mask, again.strength, again.width, again.height) == (
        first.refine, first.mask, first.strength, first.width, first.height
    )
    assert store.mask_path(digest).is_file()
    # A --refine of another image takes no mask from this one.
    assert generate.main(["--from-image", made, "--refine", str(source.id), "-q"]) == 0
    assert cli.requests[-1].mask is None


def test_an_inpaint_sweep_runs_the_cells_the_server_would(cli, store, monkeypatch, tmp_path):
    from backend import grid
    from backend.schemas import GenerateRequest

    source = to_refine(store)
    path, image = mask_file(tmp_path)
    digest = store.mask_hash(image)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--steps", "20",
                          "--seed", "3", "--grid-strength", "0.4,0.6", "-q"]) == 0
    # The server's expansion of the same request: one cell per strength,
    # every one over the mask, at the same seeds.
    request = GenerateRequest(prompt="a red scarf", width=256, height=384, steps=20,
                              refine=source.id, strength=0.4, mask=digest, seed=3)
    cells = grid.expand(request, strength="0.4,0.6", seeds=1)
    assert [(r.strength, r.mask) for r in cli.requests] == [(c.request.strength, c.request.mask) for c in cells]
    assert cli.seeds == [c.seed for c in cells]



def test_a_cli_cell_whose_mask_went_fails_in_words(cli, store, monkeypatch, tmp_path, capsys):
    source = to_refine(store)
    path, image = mask_file(tmp_path)
    digest = store.mask_hash(image)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    saved = store.save_image

    def save_then_lose_the_mask(*args, **kwargs):
        # Gone once the first cell's image is saved, before the second began,
        # as the server's worker test has it.
        try:
            return saved(*args, **kwargs)
        finally:
            (store.masks_dir() / f"{digest}.png").unlink(missing_ok=True)

    monkeypatch.setattr(store, "save_image", save_then_lose_the_mask)
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--steps", "20",
                          "--seed", "3", "--grid-strength", "0.4,0.6", "-q"]) == 1
    assert "missing from the masks directory" in capsys.readouterr().err
    assert [r.strength for r in cli.requests] == [0.4]


def test_a_mask_file_with_alpha_or_16_bit_levels_is_read_as_painted(cli, store, monkeypatch, tmp_path):
    import numpy as np
    from PIL import Image

    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    # White strokes on a transparent layer whose hidden colour is white too,
    # as an editor exports them: only the stroke is masked.
    layer = Image.new("RGBA", (256, 384), (255, 255, 255, 0))
    layer.paste((255, 255, 255, 255), (0, 0, 128, 64))
    layer.save(tmp_path / "layer.png")
    # And a 16-bit greyscale export: scaled to 8 bits, not clipped.
    levels = np.zeros((384, 256), dtype=np.uint16)
    levels[:64, :128] = 65535
    levels[64:, :] = 100  # a near-black "keep", which clipping would make 100/255
    Image.fromarray(levels).save(tmp_path / "deep.png")
    with Image.open(tmp_path / "deep.png") as deep:
        assert deep.mode.startswith("I")
    wanted = Image.new("L", (256, 384), 0)
    wanted.paste(255, (0, 0, 128, 64))
    for name in ("layer.png", "deep.png"):
        assert generate.main(["a fox", "--refine", str(source.id), "--mask", str(tmp_path / name),
                              "--strength", "0.6", "-q"]) == 0, name
        assert cli.requests[-1].mask == store.mask_hash(wanted), name


def test_a_mask_in_a_mode_no_mask_is_in_is_refused_by_name(cli, store, monkeypatch, tmp_path, capsys):
    from PIL import Image

    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    Image.new("CMYK", (256, 384), (0, 0, 0, 255)).save(tmp_path / "print.tif")
    with pytest.raises(SystemExit) as raised:
        generate.main(["a fox", "--refine", str(source.id), "--mask", str(tmp_path / "print.tif")])
    assert raised.value.code == 2
    assert "not one in mode CMYK" in capsys.readouterr().err


def test_a_mask_without_a_strength_starts_at_the_inpaint_classs_default(cli, store, monkeypatch, tmp_path):
    import diffusers

    from backend import model_selection, refine

    source = to_refine(store)
    path, _ = mask_file(tmp_path)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    monkeypatch.setattr(model_selection, "_selection", (None, 0))
    model_selection.select(ANIMAGINE)
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "-q"]) == 0
    wanted = refine.library_strength(diffusers.StableDiffusionXLInpaintPipeline)
    assert cli.requests[-1].strength == wanted == 0.9999


def test_from_image_drops_a_source_this_gallery_holds_another_picture_under(
    cli, store, monkeypatch, tmp_path, capsys
):
    source = to_refine(store)
    path, _ = mask_file(tmp_path)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a red scarf", "--refine", str(source.id), "--mask", str(path), "--strength", "0.6",
                          "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    # The id now names another picture of the same size - a PNG from another
    # gallery, or ids renumbered since.
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    assert generate.main(["--from-image", made, "-q"]) == 0
    ran = cli.requests[-1]
    assert (ran.refine, ran.mask) == (None, None)
    assert f"source image {source.id} is not the image the PNG was made from" in capsys.readouterr().err
    # Asked for as an inpaint all the same, it is refused rather than run over the wrong picture.
    with pytest.raises(SystemExit):
        generate.main(["--from-image", made, "--strength", "0.5", "-q"])
    # Named on the command line, the source is the user's own choice.
    assert generate.main(["--from-image", made, "--refine", str(source.id), "-q"]) == 0
    assert cli.requests[-1].refine == source.id


# ------------------------------------------------------------- the outpaint


def test_an_outpaint_from_the_terminal_builds_stores_and_records_its_mask(cli, store, monkeypatch, capsys):
    from backend.schemas import Outpaint

    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    argv = ["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "--band", "16", "--seed", "4"]
    digest = store.outpaint_mask_hash(304, 384, Outpaint(left=32, right=16, band=16))
    # A dry run says what would run and stores nothing.
    assert generate.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "outpaint : +32/+0/+16/+0, band 16 px, fill edge-box3-r16; canvas 304x384" in out
    assert "size     : 304x384" in out
    assert store.mask_size(digest) is None and cli.loads == 0
    assert generate.main([*argv, "-q"]) == 0
    ran = cli.requests[-1]
    assert (ran.refine, ran.width, ran.height, ran.mask) == (source.id, 304, 384, digest)
    assert ran.outpaint.key() == "edge-box3-r16:32,0,16,0@16"
    assert store.mask_path(digest).is_file()
    newest = store.list_images()[0][0]
    assert (newest.outpaint.key(), newest.mask) == ("edge-box3-r16:32,0,16,0@16", digest)
    # The CLI holds no mask in `mask_holds`: it runs its cells itself.
    with store._session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM mask_holds").fetchone()[0] == 0


def test_an_outpaint_preset_from_the_terminal(cli, store, monkeypatch, capsys):
    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    # 256 x 384 to 16:9: the width to ceil(384 * 16 / 9 / 16) * 16 = 688.
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "16:9", "-q"]) == 0
    ran = cli.requests[-1]
    assert ran.outpaint.key() == "edge-box3-r16:208,0,224,0@32" and (ran.width, ran.height) == (688, 384)
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "+25%", "-q"]) == 0
    assert cli.requests[-1].outpaint.key() == "edge-box3-r16:32,48,32,48@32"
    capsys.readouterr()
    # A preset that would change nothing is held in its words.
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "2:3"]) == 2
    assert f"--outpaint 2:3: #{source.id} is already 2:3" in capsys.readouterr().err


def test_outpaint_flags_are_refused_where_they_cannot_run(cli, store, monkeypatch, tmp_path, capsys):
    source = to_refine(store)
    path, _ = mask_file(tmp_path)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    refine = ["--refine", str(source.id)]
    for argv, words in (
        (["a fox", "--outpaint", "32,0,0,0", *SIZE], "--outpaint needs --refine"),
        (["a fox", *refine, "--outpaint", "50,0,0,0"], "50 is not a multiple of 16: 48 or 64"),
        (["a fox", *refine, "--outpaint", "0,0,0,0"], "an outpaint extends at least one side"),
        (["a fox", *refine, "--outpaint", "wide"], "neither four sides"),
        (["a fox", *refine, "--outpaint", "32,0,0,0", *SIZE], "--outpaint and --width/--height/--aspect/--mp"),
        (["a fox", *refine, "--outpaint", "32,0,0,0", "--scale", "2"], "--outpaint and --scale"),
        (["a fox", *refine, "--outpaint", "32,0,0,0", "--mask", str(path)], "an outpaint builds its own mask"),
        (["a fox", *refine, "--band", "16"], "--band needs --outpaint"),
    ):
        with pytest.raises(SystemExit) as raised:
            generate.main(argv)
        assert raised.value.code == 2, argv
        assert words in capsys.readouterr().err, argv
    # A band off its grid, and one leaving nothing kept.
    assert generate.main(["a fox", *refine, "--outpaint", "32,0,0,0", "--band", "12"]) == 2
    assert "12 is not a multiple of 8" in capsys.readouterr().err
    assert generate.main(["a fox", *refine, "--outpaint", "32,0,32,0", "--band", "128"]) == 2
    assert "on the left and right sides leaves 0 px of" in capsys.readouterr().err
    # And a checkpoint whose inpaint is not enabled, in the outpaint's words.
    monkeypatch.setattr(generate, "_project", project_claiming(generate, "black-forest-labs/FLUX.2-klein-4B"))
    assert generate.main(["a fox", *refine, "--outpaint", "32,0,0,0"]) == 2
    assert "which an outpaint runs through" in capsys.readouterr().err
    assert cli.loads == 0 and cli.requests == []


def test_an_off_grid_source_is_refused_by_name_before_any_request(cli, store, monkeypatch, capsys):
    from PIL import Image

    from backend.schemas import GenerateRequest

    source = store.save_image(
        Image.new("RGB", (264, 264)), GenerateRequest(prompt="x", width=256, height=256),
        seed=1, duration_ms=1, model_id=store.settings.model_id,
    )
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a fox", "--refine", str(source.id), "--outpaint", "16,0,0,0"]) == 2
    assert f"#{source.id} is 264 x 264, off the 16-px grid" in capsys.readouterr().err


def test_from_image_reproduces_an_outpaint_by_its_spec(cli, store, monkeypatch, capsys):
    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "--band", "16",
                          "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    first = cli.requests[-1]
    # The mask file lost since: built again, never carried as a hash.
    store.mask_path(first.mask).unlink()
    assert generate.main(["--from-image", made, "-q"]) == 0
    again = cli.requests[-1]
    assert (again.outpaint, again.mask, again.width, again.height, again.strength) == (
        first.outpaint, first.mask, first.width, first.height, first.strength
    )
    assert store.mask_path(first.mask).is_file()
    # The user's own size flags are refused beside the carried outpaint.
    with pytest.raises(SystemExit):
        generate.main(["--from-image", made, "--width", "512", "--height", "512"])
    assert "--outpaint and --width/--height/--aspect/--mp" in capsys.readouterr().err


def test_from_image_of_an_outpaint_takes_a_band_of_its_own(cli, store, monkeypatch, capsys):
    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0",
                          "--seed", "4", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    first = cli.requests[-1]
    # Another band builds another mask - what was asked, never refused as
    # a mask this build builds differently.
    assert generate.main(["--from-image", made, "--band", "16", "-q"]) == 0
    again = cli.requests[-1]
    assert again.outpaint.band == 16 and first.outpaint.band != 16
    assert (again.outpaint.left, again.outpaint.top, again.outpaint.right, again.outpaint.bottom) == (
        first.outpaint.left, first.outpaint.top, first.outpaint.right, first.outpaint.bottom
    )
    assert again.mask != first.mask
    assert again.mask == store.outpaint_mask_hash(again.width, again.height, again.outpaint)


def test_from_image_refuses_a_mask_this_build_would_build_differently(cli, store, monkeypatch, capsys):
    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    rewrite_record(made, lambda meta: meta.update(mask="e" * 64))
    assert generate.main(["--from-image", made, "-q"]) == 2
    assert "builds an outpaint's mask differently" in capsys.readouterr().err
    # An unknown fill is refused by name.
    rewrite_record(made, lambda meta: meta.update(outpaint_fill="edge-box5-r8"))
    assert generate.main(["--from-image", made, "-q"]) == 2
    assert "this build has no outpaint fill 'edge-box5-r8'" in capsys.readouterr().err


def test_from_image_drops_an_outpaint_with_a_source_that_is_another_picture(cli, store, monkeypatch, capsys):
    source = to_refine(store)
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "--seed", "4",
                          "-q"]) == 0
    made = capsys.readouterr().out.split("  seed ")[0]
    with gallery_lock.hold(store.settings.db_path), store._write_lock, store._session() as conn:
        conn.execute("UPDATE images SET content_hash = ? WHERE id = ?", ("0" * 16, source.id))
    assert generate.main(["--from-image", made, "-q"]) == 0
    ran = cli.requests[-1]
    # A plain render at the PNG's own size, never an outpaint over the wrong picture.
    assert (ran.refine, ran.mask, ran.outpaint, ran.width) == (None, None, None, 304)
    assert "running as a plain generation instead of an outpaint" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        generate.main(["--from-image", made, "--band", "8", "-q"])
    assert "cannot be run again as an outpaint" in capsys.readouterr().err


def test_a_cli_cell_whose_outpaint_mask_went_fails_in_words(cli, store, monkeypatch, capsys):
    from backend.schemas import Outpaint

    source = to_refine(store)
    digest = store.outpaint_mask_hash(304, 384, Outpaint(left=32, right=16))
    monkeypatch.setattr(generate, "_project", project_claiming(generate, ANIMAGINE))
    saved = store.save_image

    def save_then_lose_the_mask(*args, **kwargs):
        try:
            return saved(*args, **kwargs)
        finally:
            (store.masks_dir() / f"{digest}.png").unlink(missing_ok=True)

    monkeypatch.setattr(store, "save_image", save_then_lose_the_mask)
    assert generate.main(["a wider view", "--refine", str(source.id), "--outpaint", "32,0,16,0", "--steps", "20",
                          "--seed", "3", "--grid-strength", "0.4,0.6", "-q"]) == 1
    assert "the outpaint's mask has gone from this gallery" in capsys.readouterr().err
