"""Confirm the A4 allocator signature on direct and queued private adapter calls."""

from __future__ import annotations

import asyncio
from contextlib import closing
import hashlib
import json
import os
import sys
import threading
import weakref

from cublas_residency import OUT, WORK, wait, write


def offline(event, args):
    if event == "socket.connect" and isinstance(args[1], tuple) and args[1][0] not in {"127.0.0.1", "::1"}:
        raise RuntimeError("External network held in offline trial.")


def main():
    if not __debug__ or OUT == WORK or not OUT.is_relative_to(WORK):
        raise ValueError("Use the trial supervisor and an owned scratch directory.")
    wait("guard-admitted")
    sys.addaudithook(offline)
    sys.path.insert(0, str(WORK.parent))
    os.environ.update(
        AURADREAM_MODEL_ID="cagliostrolab/animagine-xl-4.0",
        AURADREAM_QUANT="fp8",
        AURADREAM_OFFLOAD="model",
        AURADREAM_DEVICE="cuda:0",
        AURADREAM_DB_PATH=str(OUT / "gallery.db"),
        AURADREAM_OUTPUTS_DIR=str(OUT / "outputs"),
        AURADREAM_BUNDLES_DIR=str(OUT / "bundles"),
        AURADREAM_LORAS_DIR=str(OUT / "loras"),
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
    )
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    from backend import config, grid, jobs, storage, tiled_refine_admission as admission
    from backend import tiled_refine_contract as contract
    from backend.backends.diffusers_backend import DiffusersBackend
    from backend.schemas import GenerateRequest
    from PIL import Image
    import torch

    assert config.settings.db_path == OUT / "gallery.db" and config.settings.outputs_dir == OUT / "outputs"
    assert not config.settings.db_path.exists() and not config.settings.outputs_dir.exists()
    storage.init_db()
    fixtures = WORK / "gen49-stage3-fixtures"
    manifest = (fixtures / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == "931fd50cbfdb4a6af86cc2fa6aac675d46fbd727bce997be2a087b1d3eb362a2"
    scene = json.loads(manifest)["scenes"][0]
    assert scene["name"] == "architecture" and scene["seed"] == 4101
    with Image.open(fixtures / scene["filename"]) as source:
        row = storage.save_image(
            source,
            GenerateRequest(
                prompt="Owned synthetic drawing: architecture",
                width=1024,
                height=1024,
                steps=1,
                guidance_scale=0.0,
                mu=None,
            ),
            0,
            0,
            "fixture/owned-procedural",
            quantization="none",
        )
    spec = admission.bind_source(contract.TiledRefineSpec(source=row.id, recipe=contract.RECIPE_NAME))
    assert (spec.root_source.hash, spec.prepared_rgb_sha256) == (scene["rgb_sha256"], scene["prepared_rgb_sha256"])
    req = GenerateRequest(
        prompt=scene["prompt"],
        negative_prompt=scene["negative_prompt"],
        width=2048,
        height=2048,
        steps=28,
        guidance_scale=5.0,
        strength=0.35,
        mu=None,
        tiled_refine=spec,
    )
    backend = DiffusersBackend()
    backend.set_console_progress(False)
    (OUT / "load-request").touch(exist_ok=False)
    wait("load-admitted")
    backend._load_tiled_refine_candidate()
    owner = weakref.ref(backend._require_tiled_refine_owner())
    sibling = weakref.ref(backend._inpainter(owner()))
    report = dict(
        status="started",
        loads=1,
        torch=torch.__version__,
        torch_commit=torch.version.git_version,
        proof=spec.model_dump(mode="json"),
        workspace_size=torch.backends.cuda.cublas_workspace_size(),
        observations={},
        thread_ids={},
    )

    def observe(label):
        # Synchronize before reading, but keep cache intact through normal unload.
        torch.cuda.synchronize()
        active = [
            {k: b[k] for k in ("address", "size", "requested_size", "state")}
            for s in torch.cuda.memory_snapshot()
            for b in s["blocks"]
            if b["state"] != "inactive"
        ]
        report["observations"][label] = dict(
            ready=backend.is_ready(),
            owner_alive=owner() is not None,
            sibling_alive=sibling() is not None,
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
            active_blocks=active,
        )
        write("loaded-control.json", report)

    observe("loaded")
    generate = backend._generate_tiled_refine_candidate
    phase = "direct"

    def observed(*args, **kwargs):
        report["thread_ids"][phase] = threading.get_native_id()
        return generate(*args, **kwargs)

    backend._generate_tiled_refine_candidate = observed
    try:
        (im,) = backend._generate_tiled_refine_candidate(req, 4101)
        with closing(im):
            report["direct_rgb"] = hashlib.sha256(im.tobytes()).hexdigest()
        observe("direct")
        phase = "queued"

        async def run():
            manager = jobs.JobManager(backend)
            try:
                job = await manager._submit_tiled_refine_candidate([grid.Cell(req, 4101)])
                while not job.done_sent:
                    await asyncio.sleep(0.05)
                assert job.status == "done" and job.completed_indices == [0] and not job.failed_indices
                (saved,) = job.images
                row = storage.get_image(saved["id"])
                assert row.tiled_refine == spec
                with Image.open(config.settings.outputs_dir / row.filename) as image:
                    report["queued_rgb"] = hashlib.sha256(image.tobytes()).hexdigest()
                    assert json.loads(image.info["auradream"])["tiled_refine"] == spec.model_dump(mode="json")
            finally:
                await manager.shutdown()

        asyncio.run(run())
        observe("queued")
    finally:
        del backend._generate_tiled_refine_candidate
        backend.unload()
    observe("normal_unload")
    (OUT / "unload-done").touch(exist_ok=False)
    wait("unload-observed")
    # No other operation releases a resource between these allocator observations.
    torch._C._cuda_clearCublasWorkspaces()
    observe("only_cublas_clear")
    torch.cuda.empty_cache()
    observe("then_empty_cache")
    obs = report["observations"]
    ws = report["workspace_size"]
    assert ws == 32 * 2**20 and report["thread_ids"]["direct"] != report["thread_ids"]["queued"]
    assert (
        report["direct_rgb"]
        == report["queued_rgb"]
        == "8c778b1471fe150fe69085c8fea5eed41643a522feced36ae4e2437a0d989893"
    )
    initial = obs["normal_unload"]
    assert not any(initial[k] for k in ("ready", "owner_alive", "sibling_alive"))
    assert initial["allocated"] == initial["reserved"] == 2 * ws
    assert [b["size"] for b in initial["active_blocks"]] == [ws, ws]
    for b in initial["active_blocks"]:
        assert any(q["address"] == b["address"] and q["size"] == ws for q in obs["queued"]["active_blocks"])
    assert any(
        b["address"] == d["address"] and d["size"] == ws
        for d in obs["direct"]["active_blocks"]
        for b in initial["active_blocks"]
    )
    assert obs["only_cublas_clear"]["allocated"] == 0 and not obs["only_cublas_clear"]["active_blocks"]
    assert obs["then_empty_cache"]["reserved"] == 0
    report["status"] = "complete"
    write("loaded-control.json", report)


if __name__ == "__main__":
    main()
