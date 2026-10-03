"""The refine pass: a gallery image scaled up and re-denoised part of the way.

It runs the image-to-image class diffusers ships beside the checkpoint's
pipeline, over the components already loaded, so what is covered here is the
part no GPU is needed for: which checkpoints have such a class (asked of the
installed diffusers), that the sibling is handed the same modules and one
generator per image, that nothing the text-to-image call passes is dropped on
the way, and that the source and the strength reach the row, the PNG and the
resume key. What it does to pixels is on the card, in `docs/verified.md`.
"""
from __future__ import annotations

import inspect
import json
import re
from types import SimpleNamespace

import diffusers
import pytest
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from backend import grid, latent_preview, launch, refine, schemas
from backend.backends import Cancelled, schedule
from backend.backends.diffusers_backend import DiffusersBackend
from backend.model_defaults import defaults_for, hints_for
from backend.schemas import GenerateRequest, default_strength

# `tests/` has no __init__.py; pytest puts this directory on sys.path.
from test_klein import KleinLikePipeline
from test_profiles import CASES
from test_schedule import STEPS, StubPipeline

# The four families diffusers 0.40 ships an image-to-image class for, of the
# pipelines this app's catalogue loads.
SIBLINGS = {
    "FluxPipeline": "FluxImg2ImgPipeline",
    "StableDiffusionXLPipeline": "StableDiffusionXLImg2ImgPipeline",
    "ZImagePipeline": "ZImageImg2ImgPipeline",
    "QwenImagePipeline": "QwenImageImg2ImgPipeline",
}


# ---------------------------------------------------------------- the sibling


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_the_profile_guess_matches_the_installed_diffusers(model_id, class_name):
    # The profile answers before a load and the pipeline's class after it; the
    # two have to agree, or the Refine box is live on a checkpoint that then
    # refuses at submit - or greyed on one that could have run it.
    found = refine.sibling(getattr(diffusers, class_name))
    assert (found is not None) is defaults_for(model_id)["refine_supported"], model_id
    assert (found.__name__ if found else None) == SIBLINGS.get(class_name)
    # And the tooltip says which state the box is in.
    hint = hints_for(model_id)["refine"]
    assert ("no image-to-image counterpart" in hint) is (found is None)


def test_the_local_pipelines_have_no_sibling():
    # Anima and NewBie run this repo's own pipeline classes, which diffusers
    # exports nothing beside; their profiles say so before a load.
    from backend.backends.anima_pipeline import AnimaTurboPipeline
    from backend.backends.newbie_pipeline import NewbiePipeline

    assert refine.sibling(AnimaTurboPipeline) is None
    assert refine.sibling(NewbiePipeline) is None
    assert not defaults_for("circlestone-labs/Anima")["refine_supported"]
    assert not defaults_for("NewBie-AI/NewBie-image-Exp0.1")["refine_supported"]
    # An unrecognised checkpoint is not promised one either, until it loads.
    assert not defaults_for("nobody/whatever")["refine_supported"]


def test_a_class_that_only_shares_the_name_has_no_sibling():
    # The sibling is built over the pipeline's components unchecked, so it is
    # offered only for the class diffusers itself exports under that name.
    assert refine.sibling(type("FluxPipeline", (), {})) is None
    assert refine.sibling(KleinLikePipeline) is None
    # And never for a class that is already image-to-image.
    assert refine.sibling(diffusers.FluxImg2ImgPipeline) is None


@pytest.mark.parametrize("class_name", sorted(SIBLINGS))
def test_every_sibling_takes_what_the_backend_passes(class_name):
    # A sibling that dropped `negative_prompt` or `sigmas` would render a
    # refine the row then describes wrongly, in silence: the backend's drop
    # loop only logs at DEBUG.
    own = set(inspect.signature(getattr(diffusers, class_name).__call__).parameters)
    params = set(inspect.signature(getattr(diffusers, SIBLINGS[class_name]).__call__).parameters)
    assert refine.REQUIRED <= params
    assert (refine.CARRIED & own) <= params


def test_carried_names_every_argument_generate_passes():
    # `CARRIED` is what `sibling` checks a class against, so it has to be the
    # list `generate()` really builds - read off its source, since a new
    # argument added there and forgotten here would be dropped unnoticed.
    source = inspect.getsource(DiffusersBackend.generate)
    literal = source[source.index("kwargs: dict[str, Any] = {"):]
    literal = literal[: literal.index("}")]
    passed = set(re.findall(r'"(\w+)":', literal)) | set(re.findall(r'kwargs\["(\w+)"\]', source))
    # The size is the source image's, and `image` and `strength` are the pass.
    assert passed - {"width", "height", "image", "strength"} == set(refine.CARRIED)


# ------------------------------------------------------------- the arithmetic


def test_the_steps_a_strength_leaves():
    # A 28-step refine at 0.4 is 11 steps of work on SDXL and 12 on the flow
    # pipelines; the estimate is the lower, and never nothing.
    assert refine.steps_run(28, 0.4) == 11
    assert refine.steps_run(8, 0.5) == 4
    assert refine.steps_run(4, 0.1) == 1
    assert refine.steps_run(28, 1.0) == 28
    # Below one step SDXL's call raises, so the request is refused first.
    assert not refine.leaves_a_step(4, 0.2)
    assert refine.leaves_a_step(4, 0.25) and refine.leaves_a_step(28, 0.04)


def test_scaled_lands_on_the_grid_and_rounds_half_up():
    assert refine.scaled(1024, 1344, 1.5) == (1536, 2016)
    assert refine.scaled(1024, 1024, 2) == (2048, 2048)
    assert refine.scaled(896, 1152, 1) == (896, 1152)
    # 1328 x 1.5 is 124.5 grid cells: `round()` would go to the even 124
    # (1984) where the page's Math.round goes to 125.
    assert refine.scaled(1328, 3168, 1.5)[0] == 2000


def test_the_source_is_covered_and_cropped_never_stretched():
    # Three bands, top to bottom, in a 2:3 portrait.
    source = Image.new("RGB", (64, 96), (255, 0, 0))
    source.paste((0, 255, 0), (0, 32, 64, 64))
    source.paste((0, 0, 255), (0, 64, 64, 96))
    # Its own shape, larger: every band is still there, in proportion.
    same = refine.fitted(source, 128, 192)
    assert same.size == (128, 192)
    assert [same.getpixel((64, y))[:3] for y in (16, 96, 176)] == [
        (255, 0, 0), (0, 255, 0), (0, 0, 255)
    ]
    # A square of it: scaled to cover the width, the top and bottom cropped
    # off equally, so the middle band is centred and nothing is squeezed.
    square = refine.fitted(source, 128, 128)
    assert square.size == (128, 128)
    assert square.getpixel((64, 64))[:3] == (0, 255, 0)
    assert square.getpixel((64, 4))[:3] == (255, 0, 0) and square.getpixel((64, 124))[:3] == (0, 0, 255)
    # Already the size asked for: handed back as it is.
    assert refine.fitted(source, 64, 96) is source


def test_a_missing_source_is_named(store):
    with pytest.raises(ValueError, match="no gallery image with id 404"):
        refine.source_image(404, 256, 256)
    record = store.save_image(
        Image.new("RGB", (64, 64)), GenerateRequest(prompt="x", width=256, height=256),
        seed=1, duration_ms=10, model_id="test/model",
    )
    assert refine.source_image(record.id, 128, 128).size == (128, 128)
    (store.settings.outputs_dir / record.filename).unlink()
    with pytest.raises(ValueError, match=f"the file for source image {record.id} is missing"):
        refine.source_image(record.id, 128, 128)


# ------------------------------------------------------------------ the request


def _applied(request, **flags):
    return request.applied(negative_prompt_supported=True, cfg_threshold=1.0, **flags)


def test_a_source_is_dropped_where_there_is_no_sibling():
    request = GenerateRequest(prompt="a fox", steps=10, refine=3, strength=0.3)
    kept = _applied(request, refine_supported=True)
    assert (kept.refine, kept.strength) == (3, 0.3)
    # Dropped from the row with its strength, as a LoRA's weight goes with it:
    # the pipeline renders a plain text-to-image, and the record must say so.
    dropped = _applied(request, refine_supported=False)
    assert (dropped.refine, dropped.strength) == (None, default_strength())
    # A strength with no source is not a setting, and must not split a key.
    plain = _applied(GenerateRequest(prompt="a fox", strength=0.7), refine_supported=True)
    assert plain.strength == default_strength()


def test_the_strength_starts_at_the_checkpoints_own():
    # A strength indexes into the checkpoint's own schedule, so where it
    # starts is a profile number like the steps: not given, the request runs
    # at the running checkpoint's.
    assert GenerateRequest(prompt="a fox", refine=3).strength == default_strength()
    assert GenerateRequest(prompt="a fox").strength == default_strength()
    assert default_strength() == defaults_for(schemas.settings.model_id)["refine_strength"]
    repos = [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]
    for repo in repos:
        profile = defaults_for(repo)
        start = profile["refine_strength"]
        # Named exactly where there is a pass for it to start.
        assert (start is not None) is profile["refine_supported"], repo
        if start is None:
            continue
        assert 0 < start <= 1, repo
        # And it has to be a pass the checkpoint's own step count can run:
        # the Refine button on a 4-step model must not set up a refusal.
        assert refine.leaves_a_step(profile["steps"], start), repo
        assert f"starts at {start:g} on this checkpoint" in hints_for(repo)["refine"], repo
    # SDXL's is inside the range WAI's author documents for the hires pass;
    # the flow-matching checkpoints start lower, their schedules being shifted.
    wai = defaults_for("LyliaEngine/waiIllustriousSDXL_v170")["refine_strength"]
    assert 0.35 <= wai <= 0.5
    assert defaults_for("black-forest-labs/FLUX.1-Krea-dev")["refine_strength"] < wai
    # A checkpoint the table does not know has no number of its own: the
    # general one, should its pipeline turn out to have a pass - in the
    # tooltip too, which a None would have failed to format.
    assert defaults_for("nobody/whatever")["refine_strength"] is None
    assert f"starts at {refine.DEFAULT_STRENGTH:g}" in hints_for("nobody/whatever", refine_supported=True)["refine"]


def test_references_have_nowhere_to_go_on_a_refine_pass():
    # The sibling's `image` is the source, so a row that claimed references
    # too would describe conditioning that never ran.
    request = GenerateRequest(prompt="a fox", steps=10, refine=3, reference=[5])
    ran = _applied(request, refine_supported=True, reference_supported=True)
    assert (ran.refine, ran.reference) == (3, [])
    # Without the pass, the references are the text-to-image call's own.
    ran = _applied(request, refine_supported=False, reference_supported=True)
    assert (ran.refine, ran.reference) == (None, [5])


def test_a_strength_that_leaves_no_step_is_refused_when_the_request_is_built():
    with pytest.raises(ValueError, match="strength 0.2 of 4 steps is less than one step"):
        GenerateRequest(prompt="x", steps=4, refine=1, strength=0.2)
    # Not a setting without a source, so nothing to refuse.
    GenerateRequest(prompt="x", steps=4, strength=0.2)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            GenerateRequest(prompt="x", refine=1, strength=bad)
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", refine=0)


# ----------------------------------------------------------------- the key


def test_the_resume_key_carries_the_source_and_the_strength():
    plain = grid.Cell(request=GenerateRequest(prompt="a fox", steps=10, seed=1), seed=1)
    refined = grid.Cell(
        request=GenerateRequest(prompt="a fox", steps=10, seed=1, refine=7, strength=0.35), seed=1
    )
    softer = grid.Cell(
        request=GenerateRequest(prompt="a fox", steps=10, seed=1, refine=7, strength=0.5), seed=1
    )
    assert len({grid.cell_key(c) for c in (plain, refined, softer)}) == 3
    assert grid.cell_key(refined)[8] == "7@0.35" and grid.cell_key(plain)[8] is None
    assert grid.cell_key(refined)[-1] == 1
    # A refine pass is a different run from the direct render of its settings.
    assert grid.run_id([refined], "m", "fp8") != grid.run_id([plain], "m", "fp8")


def test_a_strength_axis_needs_a_source_to_be_an_axis():
    # Without one the strength does nothing, so the axis collapses rather
    # than multiplying the night by a number no pixel depends on.
    base = GenerateRequest(prompt="a fox", width=256, height=256, steps=10, seed=1)
    assert len(grid.expand(base, strength="0.2,0.4")) == 1
    cells = grid.expand(base.model_copy(update={"refine": 7}), strength="0.2-0.4x0.1", steps="10,20")
    assert [(c.request.steps, c.request.strength) for c in cells] == [
        (10, 0.2), (10, 0.3), (10, 0.4), (20, 0.2), (20, 0.3), (20, 0.4)
    ]
    assert cells[1].label.endswith("strength 0.3")
    values = grid.axis_values(base.model_copy(update={"refine": 7}), "", "", "", strength="0.2,0.4")
    assert values["strength"] == [0.2, 0.4]
    assert "2 strength" in grid.describe(values, 1)
    # A cell of the sweep that would run no step fails the whole grid, by name.
    with pytest.raises(ValueError, match="less than one step"):
        grid.expand(base.model_copy(update={"refine": 7}), strength="0.05,0.4")


def test_the_estimate_and_the_progress_count_the_steps_that_run():
    request = GenerateRequest(prompt="a fox", steps=28, refine=7, strength=0.5, num_images=2)
    cell = grid.Cell(request=request, seed=1)
    assert grid.denoised_steps(request) == 14
    # Per image for the time, and through every image for the bar: a refine
    # batch is one call per image.
    assert grid.estimate_seconds([cell], 100.0) == 2.8
    assert grid.progress_steps(request) == 28
    plain = GenerateRequest(prompt="a fox", steps=28, num_images=2)
    assert grid.denoised_steps(plain) == 28 and grid.progress_steps(plain) == 28
    assert grid.estimate_seconds([grid.Cell(request=plain, seed=1)], 100.0) == 5.6


# --------------------------------------------------------------- the record


def test_the_row_and_the_png_carry_the_source_and_the_strength(store):
    source = store.save_image(
        Image.new("RGB", (256, 256)), GenerateRequest(prompt="a fox", width=256, height=256, steps=10),
        seed=3, duration_ms=10, model_id="test/model",
    )
    assert (source.refine, source.strength) == (None, None)
    request = GenerateRequest(
        prompt="a fox", width=384, height=384, steps=10, refine=source.id, strength=0.35
    )
    record = store.save_image(
        Image.new("RGB", (384, 384)), request, seed=3, duration_ms=10, model_id="test/model"
    )
    assert (record.refine, record.strength) == (source.id, 0.35)
    with Image.open(store.settings.outputs_dir / record.filename) as png:
        meta = json.loads(png.info["local-diffusion"])
        assert (meta["refine"], meta["strength"]) == (source.id, 0.35)
        # A1111's own key for an image-to-image pass, so other tools read it.
        assert f"Denoising strength: 0.35, Refine: {source.id}" in png.info["parameters"]
    # The key built from the row is the key built from the cell, or a resumed
    # sweep would refine every finished cell again.
    done = store.completed_cells(["a fox"], 384, 384, "test/model")
    assert grid.cell_keys(grid.Cell(request=request, seed=3))[0] in done
    # And both are searchable: the refinements of one image, the direct renders.
    assert [r.id for r in store.list_images(search=f"refine:{source.id}")[0]] == [record.id]
    assert [r.id for r in store.list_images(search="refine:none")[0]] == [source.id]
    assert [r.id for r in store.list_images(search="strength:0.3-0.4")[0]] == [record.id]
    rates = store.keep_rates(f"refine:{source.id}", by=("strength",))
    assert [(row["strength"], row["made"]) for row in rates] == [(0.35, 1)]


def test_a_gallery_from_before_refining_gains_the_columns(tmp_path, monkeypatch):
    import dataclasses
    import sqlite3

    from backend import storage
    from test_storage import OLD_SCHEMA

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(OLD_SCHEMA)
    conn.close()
    monkeypatch.setattr(
        storage, "settings", dataclasses.replace(storage.settings, db_path=db, outputs_dir=outputs)
    )
    record = storage.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="x", width=256, height=256, steps=10, refine=1, strength=0.5),
        seed=1, duration_ms=10, model_id="test/model",
    )
    assert (record.refine, record.strength) == (1, 0.5)


def test_step_cost_divides_a_refine_row_by_the_steps_it_ran(store):
    # 1000 ms over a 10-step schedule at strength 0.5 is 5 steps of work:
    # 200 ms a step, not 100. Divided by all ten, a size reached only by
    # refining would price a direct render of it at half its cost.
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="x", width=256, height=256, steps=10, refine=1, strength=0.5),
        seed=1, duration_ms=1000, model_id="test/model",
    )
    assert store.step_cost_ms(256, 256, "test/model") == 200.0


def test_a_reindex_reads_the_source_and_strength_back(store, monkeypatch):
    from backend import reindex

    monkeypatch.setattr(reindex, "settings", store.settings)
    source = store.save_image(
        Image.new("RGB", (64, 64)), GenerateRequest(prompt="src", width=256, height=256, steps=10),
        seed=1, duration_ms=10, model_id="test/model",
    )
    refined = store.save_image(
        Image.new("RGB", (96, 96)),
        GenerateRequest(prompt="src", width=384, height=384, steps=10, refine=source.id, strength=0.3),
        seed=1, duration_ms=10, model_id="test/model",
    )
    with store._session() as conn:
        conn.execute("DELETE FROM images")
    counts = reindex.reindex(store.settings.outputs_dir)
    assert counts["indexed"] == 2
    rebuilt = {r.filename: r for r in store.list_images()[0]}
    # The chunk's own values, as a reference's ids are read back: the id is
    # the one the gallery that wrote the file knew the source by.
    assert (rebuilt[refined.filename].refine, rebuilt[refined.filename].strength) == (source.id, 0.3)
    assert (rebuilt[source.filename].refine, rebuilt[source.filename].strength) == (None, None)


def test_a_database_merge_remaps_the_source_id(store, monkeypatch, tmp_path):
    import dataclasses

    from backend import reindex, storage

    mine = store.settings
    monkeypatch.setattr(reindex, "settings", mine)
    # Occupy id 1 here so the other gallery's ids cannot stay valid by luck.
    store.save_image(
        Image.new("RGB", (64, 64)), GenerateRequest(prompt="already here", width=256, height=256),
        seed=9, duration_ms=10, model_id="test/model",
    )
    elsewhere = tmp_path / "backup" / "outputs"
    elsewhere.mkdir(parents=True)
    source_db = elsewhere.parent / "local-diffusion.db"
    monkeypatch.setattr(
        storage, "settings", dataclasses.replace(mine, db_path=source_db, outputs_dir=elsewhere)
    )
    storage.init_db()
    source = storage.save_image(
        Image.new("RGB", (64, 64)), GenerateRequest(prompt="src", width=256, height=256, steps=10),
        seed=1, duration_ms=10, model_id="test/model",
    )
    refined = storage.save_image(
        Image.new("RGB", (96, 96)),
        GenerateRequest(prompt="refined", width=384, height=384, steps=10, refine=source.id, strength=0.3),
        seed=1, duration_ms=10, model_id="test/model",
    )
    # And one whose source was deleted there: nothing to map it to.
    orphan = storage.save_image(
        Image.new("RGB", (96, 96)),
        GenerateRequest(prompt="orphan", width=384, height=384, steps=10, refine=4242, strength=0.3),
        seed=2, duration_ms=10, model_id="test/model",
    )
    monkeypatch.setattr(storage, "settings", mine)

    counts = reindex.reindex(elsewhere, database=source_db)
    assert counts["indexed"] == 3 and counts["unresolved_references"] == 1
    by_prompt = {r.prompt: r for r in store.list_images(limit=10)[0]}
    assert by_prompt["src"].id != source.id
    assert (by_prompt["refined"].refine, by_prompt["refined"].strength) == (by_prompt["src"].id, 0.3)
    # Dropped with its strength rather than left naming an unrelated image.
    assert (by_prompt["orphan"].refine, by_prompt["orphan"].strength) == (None, None)
    # The copied PNG says the same, should this gallery be rebuilt from files.
    with Image.open(mine.outputs_dir / refined.filename) as png:
        assert json.loads(png.info["local-diffusion"])["refine"] == by_prompt["src"].id
    with Image.open(mine.outputs_dir / orphan.filename) as png:
        assert json.loads(png.info["local-diffusion"])["refine"] is None


# ---------------------------------------------------------------- the backend


class RefinableMain(KleinLikePipeline):
    """A text-to-image stand-in with what `refine.build` reads off a real
    pipeline - its components and its config - and the offload reset."""

    def __init__(self):
        super().__init__()
        self.scheduler = SimpleNamespace(sigmas=[1.0, 0.5, 0.0])
        self.denoiser = object()
        self.freed = 0

    @property
    def components(self):
        return {"scheduler": self.scheduler, "denoiser": self.denoiser, "vae": self.vae}

    @property
    def config(self):
        return {"flag": True}

    def maybe_free_model_hooks(self):
        self.freed += 1

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale,
                 num_images_per_prompt, generator, callback_on_step_end, sigmas=None,
                 negative_prompt=None):
        self.calls.append(dict(prompt=prompt, negative_prompt=negative_prompt))
        image = Image.new("RGB", (width, height))
        return type("Result", (), {"images": [image] * num_images_per_prompt})()


class Img2ImgStub:
    """The sibling's shape: the source and a strength in place of a size, and
    the flow pipelines' count of the steps a strength leaves."""

    def __init__(self, scheduler, denoiser, vae, flag=False, image_encoder=None):
        self.scheduler, self.denoiser, self.vae, self.flag = scheduler, denoiser, vae, flag
        self.calls: list[dict] = []
        self.finished = 0
        self.progress_bar = None

    def set_progress_bar_config(self, **kwargs):
        self.progress_bar = kwargs

    def __call__(self, prompt, image, strength, num_inference_steps, guidance_scale,
                 num_images_per_prompt, generator, callback_on_step_end, sigmas=None,
                 negative_prompt=None):
        self.calls.append(dict(
            prompt=prompt, image=image, strength=strength, generator=generator,
            num_images_per_prompt=num_images_per_prompt, negative_prompt=negative_prompt,
        ))
        self.num_timesteps = num_inference_steps - int(num_inference_steps * (1 - strength))
        for step in range(self.num_timesteps):
            callback_on_step_end(self, step, None, {})
        self.finished += 1
        return type("Result", (), {"images": [Image.new("RGB", image.size)]})()


def _loaded(pipe) -> DiffusersBackend:
    backend = DiffusersBackend()
    backend._pipe = pipe
    backend._call_params = set(inspect.signature(pipe.__call__).parameters)
    return backend


@pytest.fixture
def refinable(store, monkeypatch):
    """A loaded backend whose pipeline has a sibling, and a source to refine."""
    monkeypatch.setattr(refine, "sibling", lambda cls: Img2ImgStub if cls is RefinableMain else None)
    backend = _loaded(RefinableMain())
    source = store.save_image(
        Image.new("RGB", (64, 96)), GenerateRequest(prompt="a fox", width=256, height=384),
        seed=1, duration_ms=10, model_id="test/model",
    )
    return backend, source


def test_a_refine_runs_the_sibling_over_the_same_components(refinable):
    backend, source = refinable
    pipe = backend._pipe
    assert backend.describe()["refine_supported"] is True
    request = GenerateRequest(
        prompt="a fox, sharper", negative_prompt="blurry", guidance_scale=4.0,
        width=256, height=384, steps=10, refine=source.id, strength=0.4, num_images=2,
    )
    assert backend.applied(request).refine == source.id
    progress: list[tuple[int, int]] = []
    images = backend.generate(request, seed=7, on_progress=lambda s, t: progress.append((s, t)))

    # The text-to-image pipeline never ran; its sibling did, built once over
    # the very modules it holds, with the config value its constructor takes.
    runner = backend._refine_pipe
    assert pipe.calls == [] and isinstance(runner, Img2ImgStub)
    assert runner.scheduler is pipe.scheduler and runner.denoiser is pipe.denoiser
    assert runner.vae is pipe.vae and runner.flag is True
    # One call per image, each with that image's own generator - not the
    # list - so image 2 is what a solo run of its recorded seed would be.
    assert [c["num_images_per_prompt"] for c in runner.calls] == [1, 1]
    assert [c["generator"].initial_seed() for c in runner.calls] == [7, 8]
    # The source at the size asked for, and the rest of the request with it.
    assert all(c["image"].size == (256, 384) for c in runner.calls)
    assert all(c["strength"] == 0.4 and c["negative_prompt"] == "blurry" for c in runner.calls)
    assert [image.size for image in images] == [(256, 384), (256, 384)]
    # Progress counts the steps the pipeline says it ran - four of ten here -
    # through both images rather than starting over.
    assert progress == [(n, 8) for n in range(1, 9)]
    # And the offload hooks were put back through the pipeline that owns them.
    assert pipe.freed == 1

    backend.generate(request.model_copy(update={"num_images": 1}), seed=7)
    assert backend._refine_pipe is runner
    # A text-to-image request still runs the pipeline itself.
    backend.generate(GenerateRequest(prompt="a fox", width=256, height=256, steps=4), seed=1)
    assert len(pipe.calls) == 1 and len(runner.calls) == 3
    # It holds the same modules, so it has to go when the pipeline goes.
    backend.unload()
    assert backend._refine_pipe is None and backend._refine_params == set()


def test_the_cli_silences_the_siblings_progress_bar_too(refinable):
    backend, source = refinable
    backend.set_console_progress(False)
    backend.generate(
        GenerateRequest(prompt="x", width=256, height=384, steps=10, refine=source.id, strength=0.5),
        seed=1,
    )
    assert backend._refine_pipe.progress_bar == {"disable": True}


def test_a_cancelled_refine_still_hands_the_card_back(refinable):
    backend, source = refinable
    request = GenerateRequest(
        prompt="a fox", width=256, height=384, steps=10, refine=source.id, strength=0.5, num_images=2
    )
    # Mid-denoise: the step callback raises, as it does for any pipeline.
    steps = []
    with pytest.raises(Cancelled):
        backend.generate(
            request, seed=1, on_progress=lambda s, t: steps.append(s), should_cancel=lambda: len(steps) >= 2
        )
    assert steps == [1, 2] and backend._pipe.freed == 1
    # Between two images of a batch - after the first one's last step, where
    # no callback is left to notice - the second image is not started.
    runner = backend._refine_pipe
    runner.calls.clear()
    with pytest.raises(Cancelled):
        backend.generate(request, seed=1, should_cancel=lambda: runner.finished >= 1)
    assert len(runner.calls) == 1 and backend._pipe.freed == 2


def test_no_refine_pass_without_a_sibling(store):
    backend = _loaded(RefinableMain())  # nothing in diffusers beside this class
    assert backend.describe()["refine_supported"] is False
    request = GenerateRequest(prompt="a fox", width=256, height=256, steps=10, refine=3, strength=0.5)
    ran = backend.applied(request)
    assert (ran.refine, ran.strength) == (None, default_strength())
    # What gets past the submit check renders as the plain image the row
    # then says it is - through the pipeline, with no source looked up.
    backend.generate(request, seed=1)
    assert len(backend._pipe.calls) == 1 and backend._refine_pipe is None
    with pytest.raises(RuntimeError, match="no image-to-image counterpart"):
        backend._refiner(backend._pipe)


def test_before_a_load_the_profile_answers(monkeypatch):
    import dataclasses

    from backend.backends import diffusers_backend as module

    request = GenerateRequest(prompt="a fox", steps=10, refine=3, strength=0.3)
    for model_id, kept in (("black-forest-labs/FLUX.1-Krea-dev", 3), ("krea/Krea-2-Turbo", None)):
        monkeypatch.setattr(module, "settings", dataclasses.replace(module.settings, model_id=model_id))
        backend = DiffusersBackend()
        assert backend.describe()["refine_supported"] is None
        assert backend.applied(request).refine == kept


def test_build_refuses_a_sibling_whose_components_are_not_there():
    class Needy:
        def __init__(self, scheduler, controlnet):
            pass

    with pytest.raises(RuntimeError, match="Needy needs a controlnet, which RefinableMain has not got"):
        refine.build(RefinableMain(), Needy)


class MuMain(StubPipeline):
    """test_schedule's pipeline, with the components `refine.build` reads."""

    @property
    def components(self):
        return {"scheduler": self.scheduler}

    config: dict = {}


class MuSibling(StubPipeline):
    """Its image-to-image shape: the same timestep half, sized by the source."""

    def __call__(self, prompt, image, strength, num_inference_steps, guidance_scale,
                 num_images_per_prompt, generator, callback_on_step_end, sigmas=None,
                 negative_prompt=None):
        return super().__call__(
            prompt, image.width, image.height, num_inference_steps, guidance_scale,
            num_images_per_prompt, generator, callback_on_step_end, sigmas, negative_prompt,
        )


@pytest.mark.parametrize("dynamic", [True, False])
def test_a_requested_mu_is_the_whole_shift_on_a_refine_pass_too(store, monkeypatch, dynamic):
    # The sibling shares the pipeline's scheduler object, which is what lets
    # `forced_mu` - patched onto the loaded pipeline's scheduler - reach it;
    # a static scheduler is handed the pre-shifted schedule instead, less its
    # own shift. Either way the schedule that lands is e^mu, as it is for a
    # text-to-image call.
    monkeypatch.setattr(refine, "sibling", lambda cls: MuSibling if cls is MuMain else None)
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=dynamic, shift=1.0 if dynamic else 3.0)
    backend = _loaded(MuMain(scheduler))
    source = store.save_image(
        Image.new("RGB", (64, 64)), GenerateRequest(prompt="x", width=256, height=256),
        seed=1, duration_ms=10, model_id="test/model",
    )
    backend.generate(
        GenerateRequest(
            prompt="x", width=256, height=256, steps=STEPS, mu=0.5, refine=source.id, strength=0.5
        ),
        seed=1,
    )
    runner = backend._refine_pipe
    assert runner.scheduler is scheduler and runner.steps_run == STEPS
    assert (runner.sigmas_arg is None) is dynamic
    landed = [float(s) for s in scheduler.sigmas[:-1]]
    assert landed == pytest.approx(schedule.shifted_sigmas(STEPS, 0.5), abs=1e-6)


def test_a_refine_preview_reads_the_sigma_where_the_pass_is(monkeypatch):
    # The pass starts part of the way down the schedule, at the index its
    # pipeline set on the scheduler; read from the top, the preview's gain
    # would be that of a far noisier step than the one just taken.
    seen: list[float] = []
    monkeypatch.setattr(latent_preview, "factors_for", lambda vae: object())
    monkeypatch.setattr(latent_preview, "unpack", lambda pipe, latents, height, width: latents)
    monkeypatch.setattr(
        latent_preview, "render", lambda latent, factors, sigma: seen.append(sigma) or b"jpeg"
    )
    scheduler = SimpleNamespace(sigmas=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0], begin_index=3)
    pipe = SimpleNamespace(scheduler=scheduler, vae=None)
    request = GenerateRequest(prompt="x", width=256, height=256, steps=5)
    backend = DiffusersBackend()
    assert backend._preview(pipe, object(), request, step=0) == b"jpeg"
    # A text-to-image call starts at the top: index 0, or none set at all.
    for begin in (0, None):
        scheduler.begin_index = begin
        backend._preview(pipe, object(), request, step=0)
    assert seen == [0.2, 0.8, 0.8]
