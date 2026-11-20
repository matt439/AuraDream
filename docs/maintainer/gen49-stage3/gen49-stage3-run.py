"""Observational child for the GEN49 guarded private one-load experiment."""
from __future__ import annotations

import asyncio
from contextlib import closing
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import weakref

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
WORK = ROOT / "work"
FIXTURES = WORK / "gen49-stage3-fixtures"
HISTORY = Path(os.environ.get("GEN49_HISTORY", WORK / "gen49-sdxl-four-region-20261117-a1"))
OLD = json.loads((ROOT / "docs/maintainer/gen49-sdxl-four-region-2026-11-17.json").read_text())
OUT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
assert OUT.is_relative_to(WORK) and OUT != WORK


def write(path, data):
    pending = path.with_suffix(".part")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def phase(name):
    write(OUT / "phase.json", {"name": name, "started": time.time()})


def wait(marker, seconds=100):
    deadline = time.monotonic() + seconds
    while not (OUT / marker).is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Supervisor did not issue {marker}.")
        time.sleep(.05)


def rgb(im):
    return hashlib.sha256(im.tobytes()).hexdigest()


def offline(event, args):
    if event == "socket.connect":
        addr = args[1]
        if isinstance(addr, tuple) and addr[0] not in {"127.0.0.1", "::1"}:
            raise RuntimeError("External network held in offline trial.")


def main():
    if not __debug__:
        raise RuntimeError("Run the measuring child with Python assertions enabled.")
    wait("guard-admitted")
    sys.addaudithook(offline)
    for key in list(os.environ):
        if key.startswith("AURADREAM_"):
            del os.environ[key]
    os.environ.update(
        AURADREAM_MODEL_ID=OLD["selection"]["model"], AURADREAM_QUANT="fp8",
        AURADREAM_OFFLOAD="model", AURADREAM_DEVICE="cuda:0", CUDA_VISIBLE_DEVICES="0",
        AURADREAM_DB_PATH=str(OUT / "gallery.db"), AURADREAM_OUTPUTS_DIR=str(OUT / "outputs"),
        AURADREAM_BUNDLES_DIR=str(OUT / "bundles"), AURADREAM_LORAS_DIR=str(OUT / "loras"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
    )
    from backend import config, grid, jobs, refine, storage, tiled_refine_admission as admission
    from backend import tiled_refine_contract as contract
    from backend.backends.diffusers_backend import DiffusersBackend
    from backend.schemas import GenerateRequest
    from PIL import Image
    import torch

    assert config.settings.db_path.resolve() == OUT / "gallery.db"
    assert config.settings.outputs_dir.resolve() == OUT / "outputs"
    assert not config.settings.db_path.exists() and not config.settings.outputs_dir.exists()
    storage.init_db()
    manifest = (FIXTURES / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == "931fd50cbfdb4a6af86cc2fa6aac675d46fbd727bce997be2a087b1d3eb362a2"
    rows = json.loads(manifest)["scenes"]
    assert [r["name"] for r in rows] == ["architecture", "organic", "lettering", "subject"]
    historic = HISTORY / "baseline.png"
    assert hashlib.sha256(historic.read_bytes()).hexdigest() == OLD["raw_manifest"]["baseline.png"]
    rows.append(dict(name="historic", filename=str(historic), seed=12346,
                     prompt=OLD["plan"]["recipe"]["prompt"]))
    requests = {}
    proofs = {}
    for scene in rows:
        name = scene["name"]
        fp = FIXTURES / scene["filename"]
        with Image.open(fp) as im:
            assert im.mode == "RGB" and im.size == (1024,1024)
            if name != "historic":
                assert rgb(im) == scene["rgb_sha256"]
            origin = "fixture/owned-procedural"
            seed = 0
            req = GenerateRequest(prompt="Owned synthetic drawing: "+name, width=1024, height=1024,
                                  steps=1, guidance_scale=0., mu=None)
            if name == "historic":
                seed = 12345
                origin = contract.MODEL_ID
                req = GenerateRequest(prompt=scene["prompt"], negative_prompt="low quality, blurry",
                                      width=1024, height=1024, steps=28, guidance_scale=5., mu=None)
            row = storage.save_image(im, req, seed, 0, origin, quantization="fp8" if name == "historic" else "none")
        bound = admission.bind_source(contract.TiledRefineSpec(source=row.id, recipe=contract.RECIPE_NAME))
        if name != "historic":
            assert bound.prepared_rgb_sha256 == scene["prepared_rgb_sha256"]
        requests[name] = GenerateRequest(prompt=scene["prompt"], negative_prompt="low quality, blurry",
            width=2048, height=2048, steps=28, guidance_scale=5., strength=.35, mu=None, tiled_refine=bound)
        proofs[name] = bound.model_dump(mode="json")
    write(OUT / "root-proofs.json", proofs)
    backend = DiffusersBackend()
    backend.set_console_progress(False)
    receipt = {"status":"started","loads":0,"sources":proofs,"calls":{},
               "versions":{p:metadata.version(p) for p in ("torch","diffusers","torchao","transformers","accelerate","pillow")},
               "old_baseline_png_sha256":OLD["raw_manifest"]["baseline.png"]}
    write(OUT / "receipt.json", receipt)
    phase("load")
    (OUT / "load-request").touch(exist_ok=False)
    wait("load-admitted")
    started = time.monotonic()
    backend._load_tiled_refine_candidate()
    receipt["loads"] += 1
    receipt["load_seconds"] = time.monotonic()-started
    owner = backend._require_tiled_refine_owner()
    receipt["owner"] = {"class":type(owner).__name__,"snapshot":str(owner.config._name_or_path),
                        "card":torch.cuda.get_device_name(0)}
    owner_ref = weakref.ref(owner)
    sibling = backend._inpainter(owner)
    sibling_ref = weakref.ref(sibling)
    receipt["sibling"] = {"class":type(sibling).__name__,
                          "shared":{n:getattr(sibling,n) is getattr(owner,n)
                                    for n in ("unet","vae","scheduler","text_encoder","text_encoder_2")}}
    assert all(receipt["sibling"]["shared"].values())
    del owner, sibling

    def residency():
        live = backend._pipe
        comp = {}
        if live is not None:
            for name in ("unet","vae","text_encoder","text_encoder_2"):
                module = getattr(live,name)
                comp[name] = dict(parameters=sorted({str(p.device) for p in module.parameters()}),
                                  buffers=sorted({str(p.device) for p in module.buffers()}))
        return dict(ready=backend.is_ready(), components=comp, allocated=torch.cuda.memory_allocated(),
                    reserved=torch.cuda.memory_reserved(),owner_alive=owner_ref() is not None,
                    sibling_alive=sibling_ref() is not None)

    receipt["load_residency"] = residency()

    # Observe inputs and outputs at their normal transaction boundaries.
    original_invoke = backend._invoke_pipeline
    original_composite = refine.composite
    original_prove = admission.prove_root
    original_candidate = backend._generate_tiled_refine_candidate
    state = {}
    cancel_at = None
    cancel = None

    def capture_call(runner, kwargs, **kw):
        if type(runner).__name__ != "StableDiffusionXLInpaintPipeline":
            return original_invoke(runner, kwargs, **kw)
        assert runner is sibling_ref() and owner_ref() is backend._pipe
        gen = kwargs["generator"]
        scheduler = runner.scheduler
        index = len(state["tiles"])+1
        stem = outdir / f"tile-{index:02d}"
        image, mask = kwargs["image"], kwargs["mask_image"]
        image.save(str(stem)+"-input.png")
        mask.save(str(stem)+"-mask.png")
        facts = {"index":index, "seed":gen.initial_seed(),"generator_device":str(gen.device),
                 "prompt":kwargs["prompt"],"negative":kwargs["negative_prompt"],
                 "strength":kwargs["strength"],"cfg":kwargs["guidance_scale"],
                 "full_steps":kwargs["num_inference_steps"],"scheduler":type(scheduler).__name__,
                 "input_sha256":rgb(image),"mask_bytes_sha256":rgb(mask),"steps":[]}
        assert facts["seed"] == state["seed"]+(index-1)//4+(index-1)%4
        assert facts["generator_device"] == "cpu" and kwargs["num_images_per_prompt"] == 1
        assert (facts["prompt"],facts["negative"],facts["strength"],facts["cfg"],facts["full_steps"]) == (
            state["prompt"],"low quality, blurry",.35,5.,28)
        state["tiles"].append(facts)
        callback = kwargs["callback_on_step_end"]

        def observed_step(active, step, timestep, cb):
            assert active is runner
            idx = scheduler.step_index - 1
            step_row = {"step":step+1,"index":idx,"timestep":float(timestep),
                        "sigma":float(scheduler.sigmas[idx]),"next_sigma":float(scheduler.sigmas[idx+1])}
            if not facts["steps"]:
                facts.update(begin_index=scheduler.begin_index, schedule=[float(x) for x in scheduler.sigmas],
                             timesteps=[float(x) for x in scheduler.timesteps])
            assert float(scheduler.timesteps[idx]) == float(timestep)
            facts["steps"].append(step_row)
            return callback(active,step,timestep,cb)

        fresh = dict(kwargs,callback_on_step_end=observed_step)
        begin = time.monotonic()
        try:
            answer = original_invoke(runner,fresh,**kw)
            im, = answer.images
            im.save(str(stem)+"-raw.png")
            facts["raw_sha256"] = rgb(im)
            return answer
        finally:
            facts["seconds_with_observer"] = time.monotonic()-begin
            write(outdir/"call.json",state)

    def capture_composite(mask, source, image):
        final = original_composite(mask,source,image)
        index = len(state["tiles"])
        final.save(outdir / f"tile-{index:02d}-composite.png")
        state["tiles"][-1].update(source_sha256=rgb(source),composite_sha256=rgb(final))
        return final

    def capture_proof(bound):
        original_prove(bound)
        state["root_checks"].append(bound.root_source.hash)

    def capture_candidate(req, seed, on_progress=None, should_cancel=None, *, on_result=None):
        assert req.tiled_refine.model_dump(mode="json") == state["proof"]
        def progress(done,total):
            state["progress"].append([done,total])
            if on_progress:
                on_progress(done,total)
            if done == cancel_at:
                cancel()
        def result(plan,tiles):
            facts = {"source_rgb_sha256":plan.source_rgb_sha256, "tiles":[asdict(t) for t in tiles]}
            state["results"].append(facts)
            offset = (len(state["results"])-1)*4
            for index,tile in enumerate(tiles):
                state["tiles"][offset+index]["plan"] = asdict(tile.plan)
            if on_result:
                on_result(plan,tiles)
        return original_candidate(req, seed, on_progress=progress,should_cancel=should_cancel,on_result=result)

    def begin(label, scene, seed, count=1):
        nonlocal state, outdir
        phase(label)
        outdir = OUT / label
        outdir.mkdir()
        bound = requests[scene].tiled_refine
        state = dict(label=label, scene=scene, proof=proofs[scene], seed=seed, count=count,
                     prompt=requests[scene].prompt, tiles=[],results=[],progress=[],root_checks=[])
        with closing(admission.load_bound_source(bound)) as im:
            im.save(outdir/"prepared.png")
        write(outdir/"call.json",state)

    def end(label, expected_cancel=False):
        steps = 12 if expected_cancel else state["count"]*36
        assert state["progress"] == [[i,state["count"]*36] for i in range(1,steps+1)]
        assert len(state["root_checks"]) == (2 if expected_cancel else 5*state["count"])
        assert len(state["results"]) == (0 if expected_cancel else state["count"])
        state.update(peak=backend.last_peak(),residency=residency())
        write(outdir/"call.json", state)
        receipt["calls"][label] = dict(seed=state["seed"],count=state["count"],progress_count=steps,
                                       peak=state["peak"], tile_calls=len(state["tiles"]),
                                       result_count=len(state["results"]),root_checks=len(state["root_checks"]))
        write(OUT/"receipt.json",receipt)

    outdir = OUT
    backend._invoke_pipeline = capture_call
    refine.composite = capture_composite
    admission.prove_root = capture_proof
    backend._generate_tiled_refine_candidate = capture_candidate
    hashes = {}
    try:
        for scene in rows:
            name, seed = scene["name"],scene["seed"]
            begin(name,name,seed)
            im, = backend._generate_tiled_refine_candidate(requests[name],seed)
            with closing(im):
                hashes[name] = rgb(im)
                im.save(outdir/"final-1.png")
            end(name)
        for label,seed,count in (("repeat",4101,1),("batch",4101,2),("solo-next",4102,1)):
            begin(label,"architecture",seed,count)
            req = requests["architecture"].model_copy(update={"num_images":count})
            images = backend._generate_tiled_refine_candidate(req,seed)
            hashes[label] = []
            for i,im in enumerate(images,1):
                with closing(im):
                    hashes[label].append(rgb(im))
                    im.save(outdir/f"final-{i}.png")
            end(label)
        receipt["repeat_equal"] = hashes["repeat"][0] == hashes["architecture"]
        receipt["batch_equals_both_solos"] = hashes["batch"] == [hashes["architecture"], hashes["solo-next"][0]]
        assert receipt["repeat_equal"] and receipt["batch_equals_both_solos"]
        assert hashes["batch"][0] != hashes["batch"][1]

        async def scenario():
            nonlocal cancel, cancel_at
            loop = asyncio.get_running_loop()
            manager = jobs.JobManager(backend)
            async def settled(job):
                while not job.done_sent:
                    await asyncio.sleep(.05)
            try:
                req = requests["architecture"]
                begin("cancel","architecture",4101)
                before = storage.newest_image_id()
                files = set(config.settings.outputs_dir.glob("*.png"))
                cancel_at = 12
                async def send_cancel():
                    assert manager.cancel(first.id) is first
                cancel = lambda: asyncio.run_coroutine_threadsafe(send_cancel(),loop).result(timeout=15)
                first = await manager._submit_tiled_refine_candidate([grid.Cell(req,4101)])
                await asyncio.wait_for(settled(first),180)
                state["job"] = dict(id=first.id,run_id=first.run_id,status=first.status,images=first.images,
                                    completed=first.completed_indices,failed=first.failed_indices)
                assert first.status == "cancelled" and not first.images and not first.failed_indices
                assert before == storage.newest_image_id() and files == set(config.settings.outputs_dir.glob("*.png"))
                assert not list(config.settings.outputs_dir.glob("*.part"))
                assert len(state["tiles"]) == 2 and "raw_sha256" not in state["tiles"][1]
                state["unchanged_gallery"] = True
                end("cancel",True)
                retained, = storage.retained_run_jobs(first.run_id)
                cell, = retained["cells"]
                assert cell.seed == 4101 and cell.request.tiled_refine == req.tiled_refine
                cancel_at = cancel = None
                begin("recovery","architecture",4101)
                recovered = await manager._submit_tiled_refine_candidate([cell],run_id=first.run_id)
                await asyncio.wait_for(settled(recovered),180)
                assert recovered.status=="done" and recovered.completed_indices==[0] and not recovered.failed_indices
                saved, = recovered.images
                row = storage.get_image(saved["id"])
                assert row.run_id == first.run_id and row.tiled_refine == req.tiled_refine and row.seed==4101
                with Image.open(config.settings.outputs_dir / row.filename) as im:
                    hashes["recovery"] = rgb(im)
                    assert hashes["recovery"] == hashes["architecture"]
                    assert json.loads(im.info["auradream"])["tiled_refine"] == proofs["architecture"]
                    im.save(outdir/"final-1.png")
                state["job"] = dict(id=recovered.id, run_id=row.run_id, row_id=row.id,
                                   filename=row.filename, seed=row.seed,status=recovered.status)
                end("recovery")
            finally:
                await manager.shutdown()
        asyncio.run(scenario())
        phase("ordinary")
        ordinary = GenerateRequest(prompt=rows[0]["prompt"],negative_prompt="low quality, blurry",
                                   width=1024,height=1024,steps=28,guidance_scale=5.,mu=None)
        progress = []
        im, = backend.generate(ordinary, seed=4511, on_progress=lambda d,t: progress.append([d,t]))
        with closing(im):
            assert (im.mode,im.size)==("RGB",(1024,1024))
            im.save(OUT/"ordinary-after.png")
            assert progress == [[i,28] for i in range(1,29)]
            receipt["ordinary"] = dict(seed=4511,rgb_sha256=rgb(im),progress=progress,peak=backend.last_peak(),
                                       residency=residency())
        receipt["hashes"] = hashes
        receipt["status"] = "render_checks_passed"
    finally:
        phase("unload")
        refine.composite = original_composite
        admission.prove_root = original_prove
        del backend._invoke_pipeline, backend._generate_tiled_refine_candidate
        backend.unload()
        receipt["post_unload"] = residency()
        write(OUT/"receipt.json",receipt)
        (OUT / "unload-done").touch(exist_ok=False)
        wait("unload-observed")
        assert not receipt["post_unload"]["ready"] and not receipt["post_unload"]["owner_alive"]
        assert not receipt["post_unload"]["sibling_alive"] and receipt["post_unload"]["allocated"] == 0
        assert receipt["post_unload"]["reserved"] == 0
    receipt["status"] = "complete"
    write(OUT/"receipt.json",receipt)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        (OUT/"child-failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
