"""Tiled Refine across its loaded backend, original gallery source and tiles."""
from __future__ import annotations

import asyncio
from contextlib import closing
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from backend import config, grid, jobs, model_selection, refine, schemas_generate, tiled_refine
from backend import tiled_refine_admission as admission
from backend import tiled_refine_contract as contract
from backend.backends import Cancelled, diffusers_backend
from backend.schemas import GenerateRequest
from backend.seeds import MAX_SEED
from tests.backend_helpers import loaded_backend


HOST = ("win32", "0.42.1", "NVIDIA GeForce RTX 5080", True)
TILE_SIZE = (1040, 1040)


class StableDiffusionXLPipeline:
    """A shared pipeline owner whose hooks can be checked after each tile."""

    def __init__(self):
        self.vae = SimpleNamespace(config=SimpleNamespace(block_out_channels=(1, 2, 3, 4)))
        self.unet = object()
        self.scheduler = object()
        self.text_encoder = object()
        self.text_encoder_2 = object()
        self.vae_scale_factor = 8
        self.calls = []
        self.cleanups = 0
        self.config = SimpleNamespace()

    @property
    def components(self):
        return {
            "scheduler": self.scheduler, "unet": self.unet, "vae": self.vae, "owner": self,
            "text_encoder": self.text_encoder, "text_encoder_2": self.text_encoder_2,
        }

    def maybe_free_model_hooks(self):
        self.cleanups += 1

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale,
                 num_images_per_prompt, generator, callback_on_step_end, negative_prompt=None):
        self.calls.append((prompt, negative_prompt))
        return SimpleNamespace(images=[Image.new("RGB", (width, height)) for _ in range(num_images_per_prompt)])


class StableDiffusionXLInpaintPipeline:
    """Record real call arguments and use seeded synthetic tile pixels."""

    def __init__(self, owner, vae, unet, scheduler, text_encoder, text_encoder_2):
        self.owner, self.vae, self.unet, self.scheduler = owner, vae, unet, scheduler
        self.text_encoder, self.text_encoder_2 = text_encoder, text_encoder_2
        self.vae_scale_factor = 8
        self.num_timesteps = 9
        self.calls = []
        self.raw = []
        self.after = None
        self.on_step = None
        self.output = None

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, prompt, negative_prompt, image, mask_image, width, height,
                 strength, num_inference_steps, guidance_scale, num_images_per_prompt,
                 generator, callback_on_step_end):
        self.calls.append(SimpleNamespace(
            prompt=prompt, negative_prompt=negative_prompt, image=image, mask=mask_image,
            mode=image.mode, mask_mode=mask_image.mode, mask_levels=mask_image.getextrema(),
            image_size=image.size, mask_size=mask_image.size, width=width, height=height,
            strength=strength, steps=num_inference_steps, cfg=guidance_scale,
            count=num_images_per_prompt, generator=generator,
            grad=torch.is_grad_enabled(), inference=torch.is_inference_mode_enabled(),
            cleanup_count=self.owner.cleanups,
        ))
        for step in range(self.num_timesteps):
            if self.on_step is not None:
                self.on_step(len(self.calls), step)
            state = {}
            assert callback_on_step_end(self, step, step, state) is state
        if self.after is not None:
            self.after(len(self.calls))
        if self.output is None:
            color = tuple(torch.randint(0, 256, (3,), generator=generator).tolist())
            images = [Image.new("RGB", (width, height), color)]
        else:
            images = self.output()
        self.raw.extend(images)
        return SimpleNamespace(images=images)


def _rgb_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def _assert_closed(image):
    with pytest.raises(ValueError, match="closed"):
        image.getpixel((0, 0))


def _source(store, *, mode="RGB", size=(1024, 1024)):
    with closing(Image.new(mode, size, "#67226a")) as image:
        image.paste("#137620", (0, 0, 180, 400))
        return store.save_image(
            image, GenerateRequest(prompt="gallery root", width=size[0], height=size[1], mu=None),
            seed=7, duration_ms=1, model_id="root/checkpoint", quantization="fp8",
        )


def _raw_spec(source):
    return contract.TiledRefineSpec(source=source.id, recipe=contract.RECIPE_NAME)


def _request(value, **changes):
    return GenerateRequest(**{
        "prompt": "a contrasting frozen scene", "negative_prompt": "soft edges, purple",
        "width": 2048, "height": 2048, "steps": 28, "guidance_scale": 5.0,
        "strength": .35, "mu": None, "tiled_refine": value, **changes,
    })


@pytest.fixture
def candidate(store, monkeypatch, tmp_path):
    selected = replace(
        store.settings, model_id=contract.MODEL_ID, quantization=contract.QUANTIZATION,
        offload=contract.OFFLOAD, device="cuda:0", loader_execution=None, loader_execution_error=None,
    )
    for module in (config, schemas_generate, diffusers_backend, jobs):
        monkeypatch.setattr(module, "settings", selected)
    monkeypatch.setattr(model_selection, "_selection", (None, 0))

    source = _source(store)
    bound = admission.bind_source(_raw_spec(source))
    owner = StableDiffusionXLPipeline()
    snapshot = tmp_path / "models--cagliostrolab--animagine-xl-4.0" / "snapshots" / contract.REVISION
    snapshot.mkdir(parents=True)
    owner.config._name_or_path = str(snapshot)
    backend = loaded_backend(owner)
    backend._tiled_refine_candidate = owner
    backend.set_console_progress(False)
    monkeypatch.setattr(backend, "refine_host", lambda: HOST)
    monkeypatch.setattr(backend, "_peak_device", lambda: None)
    monkeypatch.setattr(
        refine, "sibling",
        lambda cls, kind="Img2Img": StableDiffusionXLInpaintPipeline
        if cls is StableDiffusionXLPipeline and kind == "Inpaint" else None,
    )
    sibling = backend._inpainter(owner)
    canvas = []
    load = admission.load_bound_source

    def record_canvas(value):
        result = load(value)
        canvas.append(result)
        return result

    monkeypatch.setattr(admission, "load_bound_source", record_canvas)
    return SimpleNamespace(
        store=store, source=source, root_path=store.settings.outputs_dir / source.filename,
        bound=bound, owner=owner, backend=backend, sibling=sibling, canvas=canvas,
    )


def test_bind_distinguishes_original_pixels_from_the_exact_lanczos_canvas(store):
    source = _source(store)
    bound = admission.bind_source(_raw_spec(source))
    assert bound.source == source.id and bound.recipe == contract.RECIPE_NAME
    assert bound.root_source.filename == source.filename
    assert (bound.root_source.width, bound.root_source.height) == (1024, 1024)
    assert bound.preparation == "lanczos-2x-rgb-v1"
    with Image.open(store.settings.outputs_dir / source.filename) as root:
        with closing(root.resize((2048, 2048), Image.Resampling.LANCZOS)) as oracle:
            with closing(admission.load_bound_source(bound)) as canvas:
                assert canvas.mode == "RGB" and canvas.size == (2048, 2048)
                assert canvas.tobytes() == oracle.tobytes()
                assert bound.root_source.hash == _rgb_hash(root)
                assert bound.prepared_rgb_sha256 == _rgb_hash(oracle)
    assert bound.root_source.hash != bound.prepared_rgb_sha256
    assert contract.stored_from_json(contract.stored_json(bound)) == bound
    assert admission.bind_source(bound) == bound


def test_an_unresolved_root_or_falsified_prepared_proof_cannot_rebind(store):
    bound = admission.bind_source(_raw_spec(_source(store)))
    unresolved = bound.model_copy(update={"source": None})
    assert contract.stored_from_json(contract.stored_json(unresolved)) == unresolved
    with pytest.raises(ValueError, match="(?i)source|root"):
        admission.load_bound_source(unresolved)
    wrong_preparation = bound.model_copy(update={"prepared_rgb_sha256": "0" * 64})
    with pytest.raises(ValueError, match="(?i)canvas|prepared|proof"):
        admission.load_bound_source(wrong_preparation)
    with pytest.raises(ValueError, match="(?i)canvas|prepared|proof"):
        admission.bind_source(wrong_preparation)


@pytest.mark.parametrize("mode,size", [("RGB", (768, 1024)), ("RGBA", (1024, 1024))])
def test_bind_refuses_a_root_outside_the_actual_pixel_envelope(store, mode, size):
    source = _source(store, mode=mode, size=size)
    with pytest.raises(ValueError, match="(?i)1024|rgb"):
        admission.bind_source(_raw_spec(source))


def test_public_generate_refuses_tiled_refine_before_preparing_a_source(candidate):
    with pytest.raises(ValueError, match="Tiled Refine.*held"):
        candidate.backend.generate(_request(candidate.bound), seed=7)
    assert candidate.canvas == [] and candidate.sibling.calls == []


def test_private_load_resolves_only_the_pinned_local_snapshot(candidate, monkeypatch):
    import huggingface_hub

    events = []
    snapshot = candidate.owner.config._name_or_path

    def find(model_id, *, revision, local_files_only):
        events.append((model_id, revision, local_files_only))
        return snapshot

    monkeypatch.setattr(diffusers_backend.card_facts, "probe", lambda device: events.append(device))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", find)
    monkeypatch.setattr(candidate.backend, "_load_prepared",
                        lambda execution, **kwargs: events.append((execution, kwargs)))
    candidate.backend._load_tiled_refine_candidate()
    assert events == [
        "cuda:0",
        (contract.MODEL_ID, contract.REVISION, True),
        (None, {"tiled_refine_snapshot": snapshot}),
    ]


@pytest.mark.parametrize("missing", ["owned pipeline", "pinned revision", "measured host", "inpaint parameter"])
def test_private_backend_refuses_a_broken_envelope_before_any_tile(candidate, monkeypatch, missing):
    if missing == "owned pipeline":
        candidate.backend._tiled_refine_candidate = None
    elif missing == "pinned revision":
        candidate.owner.config._name_or_path = contract.MODEL_ID
    elif missing == "measured host":
        monkeypatch.setattr(candidate.backend, "refine_host", lambda: ("linux", *HOST[1:]))
    else:
        candidate.backend._inpaint_params.remove("negative_prompt")
    with pytest.raises((ValueError, RuntimeError), match="(?i)tiled refine|private|negative_prompt"):
        candidate.backend._generate_tiled_refine_candidate(_request(candidate.bound), 7)
    assert candidate.sibling.calls == [] and candidate.owner.cleanups == 0
    if missing != "inpaint parameter":
        assert candidate.canvas == []
    else:
        for canvas in candidate.canvas:
            _assert_closed(canvas)


def test_exact_loaded_inpaint_calls_keep_conditioning_geometry_progress_and_owned_outputs(candidate):
    progress = []
    results = []
    images = candidate.backend._generate_tiled_refine_candidate(
        _request(candidate.bound), MAX_SEED - 1,
        on_progress=lambda done, total: progress.append((done, total)),
        on_result=lambda plan, tiles: results.append((plan, tiles)),
    )
    try:
        canvas = candidate.canvas[0]
        calls = candidate.sibling.calls
        assert len(images) == 1 and images[0].mode == "RGB" and images[0].size == (2048, 2048)
        assert len(calls) == 4 and candidate.owner.calls == []
        assert candidate.sibling.scheduler is candidate.owner.scheduler
        assert candidate.sibling.unet is candidate.owner.unet and candidate.sibling.vae is candidate.owner.vae
        assert [call.cleanup_count for call in calls] == [0, 1, 2, 3]
        assert candidate.owner.cleanups == 4
        assert len({id(call.generator) for call in calls}) == 4
        assert [call.generator.initial_seed() for call in calls] == [MAX_SEED - 1, MAX_SEED, 0, 1]
        assert progress == [(step, 36) for step in range(1, 37)]
        (plan, tiles), = results
        assert plan.recipe == tiled_refine.RECIPE
        assert plan.source_rgb_sha256 == candidate.bound.prepared_rgb_sha256
        assert [tile.callback_count for tile in tiles] == [9, 9, 9, 9]
        assert [tile.source_rgb_sha256 for tile in tiles] == [
            plan.source_rgb_sha256, *(tile.result_rgb_sha256 for tile in tiles[:-1]),
        ]
        assert tiles[-1].result_rgb_sha256 == _rgb_hash(images[0])
        for call in calls:
            assert str(call.generator.device) == "cpu"
            assert (call.prompt, call.negative_prompt) == ("a contrasting frozen scene", "soft edges, purple")
            assert (call.strength, call.steps, call.cfg, call.count) == (.35, 28, 5., 1)
            assert (call.width, call.height) == call.image_size == call.mask_size == TILE_SIZE
            assert (call.mode, call.mask_mode, call.mask_levels) == ("RGB", "L", (0, 255))
            assert (call.grad, call.inference) == (False, False)
            _assert_closed(call.image)
            _assert_closed(call.mask)
        for raw in candidate.sibling.raw:
            _assert_closed(raw)
        _assert_closed(canvas)
        assert [row.id for row in candidate.store.list_images(limit=10)[0]] == [candidate.source.id]
    finally:
        for image in images:
            image.close()


def test_batch_image_seed_matches_solo_and_progress_spans_images(candidate):
    progress = []
    images = candidate.backend._generate_tiled_refine_candidate(
        _request(candidate.bound, num_images=2), MAX_SEED,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    try:
        solo, = candidate.backend._generate_tiled_refine_candidate(_request(candidate.bound), 0)
        with closing(solo):
            assert _rgb_hash(images[1]) == _rgb_hash(solo)
            assert _rgb_hash(images[0]) != _rgb_hash(solo)
        assert progress == [(step, 72) for step in range(1, 73)]
        assert [call.generator.initial_seed() for call in candidate.sibling.calls] == [
            MAX_SEED, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3,
        ]
    finally:
        for image in images:
            image.close()


@pytest.mark.parametrize("changed_during", [1, 4], ids=["first-tile", "final-tile"])
def test_changed_root_is_refused_at_the_next_tile_boundary(candidate, changed_during):
    def replace_root(call):
        if call == changed_during:
            with closing(Image.new("RGB", (1024, 1024), "#778899")) as changed:
                changed.save(candidate.root_path)
                with candidate.store._session() as conn:
                    conn.execute("UPDATE images SET content_hash = ? WHERE id = ?",
                                 (_rgb_hash(changed)[:16], candidate.source.id))

    candidate.sibling.after = replace_root
    with pytest.raises(ValueError, match="recorded original RGB proof"):
        candidate.backend._generate_tiled_refine_candidate(_request(candidate.bound), 5)
    assert len(candidate.sibling.calls) == candidate.owner.cleanups == changed_during
    for raw in candidate.sibling.raw:
        _assert_closed(raw)
    _assert_closed(candidate.canvas[0])


def test_cancel_inside_later_tile_discards_work_and_a_retry_starts_from_the_root(candidate):
    progress = []
    request = _request(candidate.bound)
    with pytest.raises(Cancelled):
        candidate.backend._generate_tiled_refine_candidate(
            request, 9, on_progress=lambda done, _total: progress.append(done),
            should_cancel=lambda: len(progress) >= 12,
        )
    assert progress == list(range(1, 13)) and candidate.owner.cleanups == 2
    assert len(candidate.sibling.calls) == 2 and len(candidate.sibling.raw) == 1
    _assert_closed(candidate.sibling.raw[0])
    _assert_closed(candidate.canvas[0])
    for call in candidate.sibling.calls:
        _assert_closed(call.image)
        _assert_closed(call.mask)

    recovered, = candidate.backend._generate_tiled_refine_candidate(request, 9)
    with closing(recovered):
        assert recovered.size == (2048, 2048)
        assert [call.generator.initial_seed() for call in candidate.sibling.calls[2:]] == [9, 10, 11, 12]
    assert candidate.owner.cleanups == 6


def test_invalid_tile_output_count_transfers_ownership_to_the_cleanup(candidate):
    candidate.sibling.output = lambda: [Image.new("RGB", TILE_SIZE), Image.new("RGB", TILE_SIZE)]
    with pytest.raises(ValueError, match="(?i)one|image|output"):
        candidate.backend._generate_tiled_refine_candidate(_request(candidate.bound), 5)
    assert len(candidate.sibling.calls) == 1 and candidate.owner.cleanups == 1
    _assert_closed(candidate.canvas[0])
    for raw in candidate.sibling.raw:
        _assert_closed(raw)


def test_private_job_saves_only_complete_composites_and_resumes_the_retained_cell(candidate):
    request = _request(candidate.bound)
    cells = [grid.Cell(request, 7), grid.Cell(request, 11)]
    store = candidate.store

    async def settled(job):
        while not job.done_sent:
            await asyncio.sleep(.01)

    async def scenario():
        manager = jobs.JobManager(candidate.backend)
        loop = asyncio.get_running_loop()
        try:
            with pytest.raises(ValueError, match="Tiled Refine.*held"):
                await manager.submit_cells(cells)
            assert store.retained_run_jobs() == []

            async def cancel_first():
                manager.cancel(first.id)

            def cancel_during_second_cell(call, step):
                if (call, step) == (6, 2):
                    # Ask through the same event-loop API as the page. Waiting
                    # here pins cancellation to this step on a loaded runner.
                    asyncio.run_coroutine_threadsafe(cancel_first(), loop).result(timeout=30)

            candidate.sibling.on_step = cancel_during_second_cell
            first = await manager._submit_tiled_refine_candidate(cells)
            await asyncio.wait_for(settled(first), 60)
            assert (first.status, first.completed_indices, first.failed_indices) == ("cancelled", [0], [])
            assert len(first.images) == 1 and first.images[0]["seed"] == 7
            assert len(candidate.sibling.calls) == 6
            assert len(store.list_images(limit=10)[0]) == 2  # The root and one result.

            (retained,) = store.retained_run_jobs(first.run_id)
            decoded = retained["cells"]
            assert [cell.seed for cell in decoded] == [7, 11]
            assert all(cell.request.tiled_refine == candidate.bound for cell in decoded)
            ran = [replace(cell, request=cell.request.applied(
                negative_prompt_supported=True, cfg_threshold=1.0,
            )) for cell in decoded]
            done = store.completed_cells(
                grid.prompts_of(ran), 2048, 2048, contract.MODEL_ID, contract.QUANTIZATION,
            )
            missing = grid.prune_done(ran, done)
            assert [cell.seed for cell in missing] == [11]

            candidate.sibling.on_step = None
            resumed = await manager._submit_tiled_refine_candidate(missing, run_id=first.run_id)
            await asyncio.wait_for(settled(resumed), 60)
            assert (resumed.status, resumed.completed_indices, resumed.failed_indices) == ("done", [0], [])
            assert len(resumed.images) == 1 and resumed.images[0]["seed"] == 11
            return first, resumed
        finally:
            await manager.shutdown()

    first, resumed = asyncio.run(scenario())
    assert len(store.list_images(limit=10)[0]) == 3
    assert [call.generator.initial_seed() for call in candidate.sibling.calls] == [
        7, 8, 9, 10, 11, 12, 11, 12, 13, 14,
    ]
    original = store.get_image(first.images[0]["id"])
    recovered = store.get_image(resumed.images[0]["id"])
    assert original is not None and recovered is not None
    assert original.run_id == recovered.run_id == first.run_id
    assert original.tiled_refine == recovered.tiled_refine == candidate.bound
    for row in (original, recovered):
        with Image.open(store.settings.outputs_dir / row.filename) as png:
            carried = json.loads(png.info["auradream"])
            assert carried["tiled_refine"] == candidate.bound.model_dump(mode="json")
            assert png.mode == "RGB" and png.size == (2048, 2048)
            assert row.content_hash == _rgb_hash(png)[:16]

    solo, = candidate.backend._generate_tiled_refine_candidate(request, 11)
    with closing(solo), Image.open(store.settings.outputs_dir / recovered.filename) as png:
        assert _rgb_hash(png) == _rgb_hash(solo)
