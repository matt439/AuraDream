"""Refine advisories held to installed pipelines' real CPU schedule slicing."""
from __future__ import annotations

import ast
import importlib
import inspect
import math
import textwrap
from types import SimpleNamespace

import diffusers
import pytest
from diffusers import EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler

from backend import grid, refine
from backend.backends import schedule
from backend.backends.diffusers_backend import DiffusersBackend
from backend.model_defaults import defaults_for
from backend.schemas import GenerateRequest

from test_profiles import CASES
from test_refine import SIBLINGS


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_profiled_start_index_matches_the_real_siblings_scheduler(model_id, class_name):
    sibling = refine.sibling(getattr(diffusers, class_name))
    if sibling is None:
        return
    sdxl = class_name == "StableDiffusionXLPipeline"
    scheduler = (
        diffusers.EulerDiscreteScheduler()
        if sdxl else diffusers.FlowMatchEulerDiscreteScheduler()
    )
    pipe = SimpleNamespace(scheduler=scheduler)
    for steps in (4, 8, 10, 28):
        scheduler.set_timesteps(steps, device="cpu")
        for strength in (0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.9, 1.0):
            if not refine.leaves_a_step(steps, strength):
                continue
            timesteps, count = sibling.get_timesteps(pipe, steps, strength, "cpu")
            assert refine.start_index(steps, strength, model_id) == scheduler.begin_index
            assert refine.steps_run(steps, strength, model_id) == count == len(timesteps)
            assert timesteps.equal(scheduler.timesteps[scheduler.begin_index:])


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_the_inpaint_start_index_matches_the_real_inpaint_classes_scheduler(model_id, class_name):
    # An inpaint rounds as its class's own get_timesteps does; only the
    # enabled families are promised an index, the deferred ones none.
    inpaint = refine.sibling(getattr(diffusers, class_name), "Inpaint")
    if not defaults_for(model_id)["inpaint_supported"]:
        assert refine.start_index(10, 0.5, model_id, inpaint=True) is None
        return
    sdxl = class_name == "StableDiffusionXLPipeline"
    scheduler = (
        diffusers.EulerDiscreteScheduler()
        if sdxl else diffusers.FlowMatchEulerDiscreteScheduler()
    )
    pipe = SimpleNamespace(scheduler=scheduler)
    for steps in (4, 8, 10, 28):
        scheduler.set_timesteps(steps, device="cpu")
        for strength in (0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.9, 0.9999, 1.0):
            if not refine.leaves_a_step(steps, strength):
                continue
            timesteps, count = inpaint.get_timesteps(pipe, steps, strength, "cpu")
            assert refine.start_index(steps, strength, model_id, inpaint=True) == scheduler.begin_index
            assert refine.steps_run(steps, strength, model_id, inpaint=True) == count == len(timesteps)
            assert timesteps.equal(scheduler.timesteps[scheduler.begin_index:])


@pytest.mark.parametrize("model_id, steps, axis, expected", [
    ("black-forest-labs/FLUX.1-Krea-dev", 28, "0.05-0.15x0.01", 7),
    ("black-forest-labs/FLUX.1-schnell", 4, "0.5-0.9x0.1", 2),
    ("Tongyi-MAI/Z-Image-Turbo", 8, "0.2-0.5x0.05", 4),
    ("KBlueLeaf/Kohaku-XL-Zeta", 10, "0.4,0.45,0.5", 1),
])
def test_duplicate_advisory_uses_integer_start_without_changing_resume_keys(
    model_id, steps, axis, expected,
):
    request = GenerateRequest(prompt="a fox", steps=steps, seed=1, refine=3,
                              strength=0.5, num_images=2)
    cells = grid.expand(request, strength=axis)
    keys = [grid.cell_key(cell) for cell in cells]
    run = grid.run_id(cells, model_id, "fp8")
    assert grid.duplicate_count(cells, lambda r: r, model_id) == expected * 2
    assert len(set(keys)) == len(cells)
    assert [grid.cell_key(cell) for cell in cells] == keys
    assert grid.run_id(cells, model_id, "fp8") == run
    # An off-catalogue family cannot promise equivalence from a guessed index.
    assert grid.duplicate_count(cells, lambda r: r, "nobody/custom") == 0


def test_duplicate_advisory_keeps_sources_seeds_and_prompts_distinct():
    request = GenerateRequest(prompt="a fox", steps=10, refine=3, strength=0.41)
    cells = [grid.Cell(request, 1)]
    for change in ({"refine": 4}, {"prompt": "a cat"}, {"steps": 11}):
        cells.append(grid.Cell(request.model_copy(update=change), 1))
    cells.append(grid.Cell(request, 2))
    assert grid.duplicate_count(cells, lambda r: r, "black-forest-labs/FLUX.1-Krea-dev") == 0


# ---------------------------------------------------------- the start sigma
#
# The Refine preview's "starts at N% noise": `schedule.start_sigma` repeats
# what the image-to-image class's call does to its scheduler - on a copy of
# the loaded one - so what it repeats is held to the installed source, and
# its answers to the schedule the card measured (docs/verified.md).

# The scheduler configs the checkpoints publish, as their repos write them.
FLUX1 = {  # FLUX.1 Krea dev and dev: a shift that grows with the size
    "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 4096, "max_shift": 1.15,
    "num_train_timesteps": 1000, "shift": 3.0, "use_dynamic_shifting": True,
}
Z_IMAGE_TURBO = {"num_train_timesteps": 1000, "use_dynamic_shifting": False, "shift": 3.0}


def _function(cls: type, name: str = "__call__") -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(inspect.getsource(getattr(cls, name)))).body[0]


def _assignments(function: ast.FunctionDef) -> list[str]:
    """Every assignment in the function, in source order, as `ast.unparse` writes it."""
    nodes = [node for node in ast.walk(function) if isinstance(node, ast.Assign)]
    return [ast.unparse(node) for node in sorted(nodes, key=lambda n: (n.lineno, n.col_offset))]


def _same(source: str) -> str:
    return ast.unparse(ast.parse(source))


@pytest.mark.parametrize("name", sorted(schedule.START_RECIPES))
def test_the_start_recipe_is_the_installed_calls_own(name):
    sibling = getattr(diffusers, name)
    recipe = schedule.START_RECIPES[name]
    call = _function(sibling)
    assigned = _assignments(call)
    # The sequence length the shift is computed from and the shift itself,
    # with the config's base and max, in that order; and the schedule handed
    # over when the caller gives none.
    wanted = [_same(line) for line in (*recipe.source, schedule.SHIFT_CALL)]
    missing = [line for line in [*wanted, _same(recipe.ramp_source)] if line not in assigned]
    assert not missing, (name, missing)
    at = [assigned.index(line) for line in wanted]
    assert at == sorted(at)
    # Both reach the scheduler through the module's own retrieve_timesteps:
    # the schedule as `sigmas=`, the mu as `mu=` or in the kwargs it unpacks.
    (retrieve,) = [
        node for node in ast.walk(call)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "retrieve_timesteps"
    ]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in retrieve.keywords}
    assert keywords.get("sigmas") == "sigmas"
    assert keywords.get("mu") == "mu" or (
        None in keywords and _same(f"{keywords[None]} = {{'mu': mu}}") in assigned
    )
    # And the pass starts at the begin index its get_timesteps sets, which
    # `refine.start_index` is held to above.
    begin = ast.unparse(_function(sibling, "get_timesteps"))
    assert "self.scheduler.set_begin_index(t_start * self.scheduler.order)" in begin
    # The module the preview calls into is the one whose names the call reads.
    module = importlib.import_module(sibling.__module__)
    assert callable(module.calculate_shift) and callable(module.retrieve_timesteps)


@pytest.mark.parametrize("name", sorted(schedule.START_RECIPES))
def test_the_recipes_arithmetic_is_its_source(name):
    # The callables do what the source lines say, run over the same values.
    sibling = getattr(diffusers, name)
    recipe = schedule.START_RECIPES[name]
    module = importlib.import_module(sibling.__module__)
    for scale in (8, 16):
        for width, height in ((1024, 1024), (1536, 1536), (1344, 1728), (2896, 2896), (256, 4448)):
            scope = {"self": SimpleNamespace(vae_scale_factor=scale), "height": height, "width": width}
            exec("\n".join(recipe.source), {}, scope)
            assert recipe.length(scale, height, width) == scope["image_seq_len"]
    for steps in (2, 8, 28, 50):
        scope = {"num_inference_steps": steps, "sigmas": None}
        exec(recipe.ramp_source, dict(vars(module)), scope)
        assert list(recipe.ramp(module, steps)) == list(scope["sigmas"])


@pytest.mark.parametrize("name", sorted(schedule.START_RECIPES))
def test_the_sibling_scales_the_vae_as_the_loaded_pipeline_does(name):
    # The preview reads `vae_scale_factor` off the loaded text-to-image
    # pipeline rather than building the sibling outside a job: the two
    # classes' constructors compute it alike.
    own = getattr(diffusers, next(key for key, value in SIBLINGS.items() if value == name))

    def scale(cls):
        return [line for line in _assignments(_function(cls, "__init__"))
                if line.startswith("self.vae_scale_factor =")]

    assert scale(own) and scale(own) == scale(getattr(diffusers, name))


@pytest.mark.parametrize("model_id, class_name", CASES)
def test_every_flow_refine_checkpoint_has_a_start_recipe(model_id, class_name):
    profile = defaults_for(model_id)
    if profile["refine_supported"] and profile["mu_supported"]:
        assert SIBLINGS[class_name] in schedule.START_RECIPES, model_id


def _start(scheduler, sibling, size, steps, strength, mu=None, model_id="black-forest-labs/FLUX.1-Krea-dev"):
    pipe = SimpleNamespace(scheduler=scheduler, vae_scale_factor=8)
    request = GenerateRequest(
        prompt="x", width=size, height=size, steps=steps, mu=mu, refine=1, strength=strength
    )
    return schedule.start_sigma(pipe, sibling, request, refine.start_index(steps, strength, model_id))


@pytest.mark.parametrize("size, strength, mu, measured, within", [
    # docs/verified.md, "What a strength keeps, FLUX.1 at 1.5x": read off the
    # card's scheduler at the pass's begin index, 1024x1024 to 1536x1536.
    (1536, 0.05, None, 0.37, 0.005), (1536, 0.1, None, 0.47, 0.005), (1536, 0.15, None, 0.62, 0.005),
    (1536, 0.2, 1.15, 0.46, 0.005), (1536, 0.3, 1.15, 0.60, 0.005),
    # "Past 4 MP": at 2x the same 0.1 starts at 0.752, and Mu 1.15 at 0.2 at 0.463.
    (2048, 0.1, None, 0.752, 0.0005), (2048, 0.2, 1.15, 0.463, 0.0005),
])
def test_flux_starts_where_the_card_measured(size, strength, mu, measured, within):
    # A dynamic shift: the mu the call derives from the size, or a requested
    # one in its place, which is the whole shift.
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(FLUX1)
    sigma = _start(scheduler, diffusers.FluxImg2ImgPipeline, size, 28, strength, mu)
    assert sigma == pytest.approx(measured, abs=within)


def test_a_static_shift_and_a_requested_mu_on_it():
    # Z-Image ships shift 3 with no dynamic shifting: 8 steps at 0.25 start
    # two from the end, at t = 0.25, shifted to 3 * 0.25 / (1 + 2 * 0.25).
    sibling, turbo = diffusers.ZImageImg2ImgPipeline, "Tongyi-MAI/Z-Image-Turbo"

    def fresh():
        return FlowMatchEulerDiscreteScheduler.from_config(Z_IMAGE_TURBO)

    assert _start(fresh(), sibling, 1024, 8, 0.25, model_id=turbo) == pytest.approx(0.5, abs=1e-6)
    # The size does not move a static shift.
    assert _start(fresh(), sibling, 2048, 8, 0.25, model_id=turbo) == pytest.approx(0.5, abs=1e-6)
    # A requested mu replaces the shift outright, as the backend's pre-shifted
    # schedule less the static shift does: the odds times e^mu, not 3 e^mu.
    expected = math.exp(0.5) / (math.exp(0.5) + (1 / 0.25 - 1))
    assert _start(fresh(), sibling, 1024, 8, 0.25, mu=0.5, model_id=turbo) == pytest.approx(expected, abs=1e-6)
    # The loaded scheduler's shift, not its config's: `set_timesteps` reads
    # the instance's, which `set_shift` can change - to Z-Anime's 6 here,
    # whose 0.15 of 40 steps the design notes put at 0.51.
    live = fresh()
    live.set_shift(6.0)
    sigma = _start(live, sibling, 1024, 40, 0.15, model_id="SeeSee21/Z-Anime")
    assert sigma == pytest.approx(6 * 0.15 / (1 + 5 * 0.15), abs=1e-6)


def test_the_loaded_scheduler_is_never_touched():
    # A job may be running on it; the preview reads a copy.
    live = FlowMatchEulerDiscreteScheduler.from_config(FLUX1)
    live.set_timesteps(4, device="cpu", mu=1.0)
    sigmas, timesteps = live.sigmas.clone(), live.timesteps.clone()
    for mu in (None, 1.15):
        _start(live, diffusers.FluxImg2ImgPipeline, 1536, 28, 0.1, mu)
    assert live.sigmas.equal(sigmas) and live.timesteps.equal(timesteps)
    assert live.num_inference_steps == 4 and "set_timesteps" not in vars(live)


def test_no_start_sigma_where_it_would_be_a_guess():
    request = GenerateRequest(prompt="x", width=1024, height=1024, steps=28, refine=1, strength=0.4)
    # SDXL's epsilon sigmas are no noise fraction.
    sdxl = SimpleNamespace(scheduler=EulerDiscreteScheduler(), vae_scale_factor=8)
    assert schedule.start_sigma(sdxl, diffusers.StableDiffusionXLImg2ImgPipeline, request, 16) is None
    # A flow class whose call is not in the table, and a pipeline with no VAE scale.
    flux = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler.from_config(FLUX1), vae_scale_factor=8)
    assert schedule.start_sigma(flux, diffusers.StableDiffusion3Img2ImgPipeline, request, 16) is None
    bare = SimpleNamespace(scheduler=flux.scheduler)
    assert schedule.start_sigma(bare, diffusers.FluxImg2ImgPipeline, request, 16) is None


def test_the_backend_reads_it_once_loaded_and_never_before(monkeypatch):
    import dataclasses

    from backend.backends import diffusers_backend as module

    backend = DiffusersBackend()
    request = GenerateRequest(prompt="x", width=1536, height=1536, steps=28, refine=1, strength=0.1)
    assert backend.start_sigma(request) is None  # before a load
    # A FluxPipeline with a scheduler and nothing else: its class is what
    # names the sibling, and the schedule is the whole of what is read.
    pipe = diffusers.FluxPipeline.__new__(diffusers.FluxPipeline)
    vars(pipe).update(scheduler=FlowMatchEulerDiscreteScheduler.from_config(FLUX1), vae_scale_factor=8)
    backend._pipe = pipe
    assert backend.start_sigma(request) == pytest.approx(0.474, abs=0.001)
    # Not a pass; a pass on a checkpoint whose start index is unknown.
    assert backend.start_sigma(request.model_copy(update={"refine": None})) is None
    known = module.settings
    monkeypatch.setattr(module, "settings", dataclasses.replace(known, model_id="nobody/unprofiled"))
    assert backend.start_sigma(request) is None
    monkeypatch.setattr(module, "settings", known)
    # A scheduler that breaks is no answer, never a refused preview.
    vars(pipe)["scheduler"] = type("FlowMatchBroken", (), {"config": None})()
    assert backend.start_sigma(request) is None
