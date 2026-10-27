"""`POST /api/instruction-edit`: one edit queued behind the backend's gate.

Driven over HTTP with the fake backend, whose gate (`edit_qualified`) a test
opens; the real backend's stays the literal False. The fake's edit branch
proves and prepares the source as the real one does and draws nothing.
No weights, downloads or GPU.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest
from PIL import Image

from backend import grid, instruction_edit_admission, main, model_selection
from backend.instruction_edit import (
    INSTRUCTION_EDIT_NEEDS_MODEL,
    INSTRUCTION_EDIT_REPLAY_HELD,
    INSTRUCTION_EDIT_TEXT_TO_IMAGE,
    INSTRUCTION_EDIT_UNAVAILABLE,
    KONTEXT_V1,
    StoredInstructionEdit,
    descriptor_from_image,
)
from backend.jobs import JobManager
from backend.metadata import PNG_METADATA_KEY
from backend.schemas import GenerateRequest, InstructionEditSubmission
from tests.api_helpers import stream_job, wait_for


def source(store, *, width=256, height=256, colour=(11, 32, 47)):
    image = Image.new("RGB", (width, height), colour)
    rec = store.save_image(
        image, GenerateRequest(prompt="original", width=width, height=height, mu=None),
        seed=3, duration_ms=1, model_id="source/checkpoint", quantization="fp8",
    )
    proof = descriptor_from_image(rec.filename, image)
    return rec, StoredInstructionEdit(source=rec.id, recipe="kontext-v1", edit_source=proof)


def body(rec, **changes):
    return {"source": rec.id, "recipe": "kontext-v1", "prompt": "make it night",
            "width": 512, "height": 512, "seed": 7, **changes}


def select(monkeypatch, model_id=KONTEXT_V1.model_id):
    # A synthetic selection of a hand-entered id: no catalogue option,
    # download or load of the editor.
    monkeypatch.setattr(model_selection, "_selection", (model_id, 1))


def open_gate(fake, monkeypatch):
    select(monkeypatch)
    fake.edit_qualified = True


def images(store):
    return store.list_images()[1]


def no_work(fake):
    return fake.requests == [] and fake.edit_calls == [] and main.jobs.active() == []


# Admission


def test_closed_gate_refuses_before_any_source_is_read(client, fake, store, monkeypatch):
    rec, _ = source(store)
    select(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("a closed instruction-edit gate reached source decoding")

    monkeypatch.setattr(instruction_edit_admission, "bind_request", forbidden)
    monkeypatch.setattr(instruction_edit_admission, "bind_source", forbidden)
    response = client.post("/api/instruction-edit", json=body(rec))
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == INSTRUCTION_EDIT_UNAVAILABLE
    assert fake.loads == 0 and no_work(fake) and images(store) == 1


def test_truthy_gate_stays_closed(client, fake, store, monkeypatch):
    rec, _ = source(store)
    select(monkeypatch)
    fake.edit_qualified = 1
    response = client.post("/api/instruction-edit", json=body(rec))
    assert response.status_code == 422 and response.json()["detail"] == INSTRUCTION_EDIT_UNAVAILABLE
    assert no_work(fake)


def test_open_gate_on_another_checkpoint_says_to_select_the_editor(client, fake, store):
    rec, _ = source(store)
    fake.edit_qualified = True
    response = client.post("/api/instruction-edit", json=body(rec))
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == INSTRUCTION_EDIT_NEEDS_MODEL
    assert no_work(fake)


@pytest.mark.parametrize("case", ["missing", "binned", "two-to-one", "wrong-size", "unchecked-size"])
def test_unusable_source_or_size_is_refused_without_a_job(client, fake, store, monkeypatch, case):
    open_gate(fake, monkeypatch)
    rec, _ = source(store, width=512, height=256) if case == "two-to-one" else source(store)
    payload = body(rec)
    if case == "missing":
        payload["source"] = rec.id + 999
    elif case == "binned":
        store.bin_images([rec.id])
    elif case == "wrong-size":
        payload.update(width=576, height=384)
    elif case == "unchecked-size":
        payload.update(width=520)
    response = client.post("/api/instruction-edit", json=payload)
    assert response.status_code == 422, response.text
    detail = json.dumps(response.json()["detail"])
    expected = {
        "missing": "missing or in the recycle bin", "binned": "missing or in the recycle bin",
        "two-to-one": "exact source aspect", "wrong-size": "requires 512x512",
        "unchecked-size": "multiple of 16",
    }[case]
    assert expected in detail
    assert no_work(fake)


@pytest.mark.parametrize("extra", [
    {"negative_prompt": "blurry"}, {"num_images": 2}, {"runs": 2}, {"reference": [1]},
    {"lora": "x"}, {"refine": 1}, {"mask": "ab" * 32}, {"execution": None}, {"path": "outside.png"},
])
def test_the_submission_takes_no_composer_setting(client, fake, store, monkeypatch, extra):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    response = client.post("/api/instruction-edit", json={**body(rec), **extra})
    assert response.status_code == 422, response.text
    assert no_work(fake)


@pytest.mark.parametrize("prompt", ["", "   \n\t"])
def test_a_blank_instruction_is_refused(client, fake, store, monkeypatch, prompt):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    response = client.post("/api/instruction-edit", json=body(rec, prompt=prompt))
    assert response.status_code == 422, response.text
    assert no_work(fake)


def test_strict_fields_refuse_coerced_values(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    for change in ({"source": True}, {"source": str(1)}, {"width": "512"}, {"recipe": "kontext-v2"},
                   {"steps": 0}, {"steps": 101}, {"guidance_scale": 21.0}, {"seed": -1}):
        response = client.post("/api/instruction-edit", json=body(rec, **change))
        assert response.status_code == 422, (change, response.text)
    assert no_work(fake)


def test_stale_model_revision_is_held_before_admission(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)

    def forbidden(*args, **kwargs):
        pytest.fail("a stale instruction edit reached source decoding")

    monkeypatch.setattr(instruction_edit_admission, "bind_request", forbidden)
    response = client.post("/api/instruction-edit", json=body(rec),
                           headers={"X-AuraDream-Model-Revision": "999999"})
    assert response.status_code == 409, response.text
    assert fake.loads == 0 and no_work(fake)


def test_submission_request_is_one_unbound_image_with_recipe_defaults():
    submitted = InstructionEditSubmission.model_validate({
        "source": 4, "recipe": "kontext-v1", "prompt": " make it night ", "width": 576, "height": 384,
    })
    request = submitted.request()
    assert request.edit is not None and request.edit.model_dump() == {"source": 4, "recipe": "kontext-v1"}
    assert (request.num_images, request.runs, request.negative_prompt, request.mu) == (1, 1, None, None)
    assert (request.steps, request.guidance_scale) == (KONTEXT_V1.default_steps, KONTEXT_V1.default_guidance_scale)
    assert request.prompt == " make it night " and request.seed is None
    assert request.execution is None
    chosen = InstructionEditSubmission.model_validate({
        **submitted.model_dump(), "seed": 9, "steps": 20, "guidance_scale": 3,
    }).request()
    assert (chosen.seed, chosen.steps, chosen.guidance_scale) == (9, 20, 3.0)


# Running


def test_an_admitted_edit_runs_once_and_is_saved_with_its_identity(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, bound = source(store, width=768, height=512)
    response = client.post("/api/instruction-edit", json=body(rec, width=576, height=384))
    assert response.status_code == 200, response.text
    accepted = response.json()
    assert accepted["seed"] == 7
    done = stream_job(client, accepted["job_id"])[-1]
    assert done["status"] == "done", done
    assert len(done["images"]) == 1
    assert fake.edit_calls == [{
        "width": 576, "height": 384, "max_area": 576 * 384, "_auto_resize": False,
        "num_images_per_prompt": 1, "max_sequence_length": 512, "image_size": (576, 384),
        "seed": 7, "steps": KONTEXT_V1.default_steps, "guidance_scale": KONTEXT_V1.default_guidance_scale,
    }]
    assert len(fake.requests) == 1 and fake.requests[0].edit == bound
    saved = store.get_image(done["images"][0]["id"])
    assert saved is not None
    assert saved.edit == bound and saved.model_id == KONTEXT_V1.model_id
    assert (saved.width, saved.height, saved.batch, saved.seed) == (576, 384, 1, 7)
    assert saved.negative_prompt is None and saved.refine is None and saved.mask is None
    assert saved.prompt == "make it night" and saved.run_id == accepted["run_id"]
    with Image.open(store.settings.outputs_dir / saved.filename) as image:
        assert image.size == (576, 384)
        chunk = json.loads(image.info[PNG_METADATA_KEY])
    assert chunk["edit"] == bound.model_dump(mode="json") and chunk["model"] == KONTEXT_V1.model_id
    # The run id is the cell's as `applied()` runs it, the edit's key in it.
    assert accepted["run_id"] == grid.run_id(
        [grid.Cell(fake.applied(fake.requests[0]), seed=7)], KONTEXT_V1.model_id, main.settings.quantization,
    )


def test_a_source_binned_after_submit_fails_its_cell_in_words(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    fake.loaded = False

    def load():
        # Between the submit and the cell: the job waits on this load.
        store.bin_images([rec.id])
        fake.loaded = True

    monkeypatch.setattr(fake, "load", load)
    response = client.post("/api/instruction-edit", json=body(rec))
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["images"] == [] and done["failed_cells"], done
    words = done["failed_cells"][0]["error"]
    assert f"source #{rec.id}" in words and "recycle bin" in words
    assert no_work(fake) and store.list_images()[1] == 0


def test_a_source_rewritten_after_submit_fails_its_cell(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    fake.loaded = False

    def load():
        Image.new("RGB", (256, 256), (1, 2, 3)).save(store.settings.outputs_dir / rec.filename)
        fake.loaded = True

    monkeypatch.setattr(fake, "load", load)
    response = client.post("/api/instruction-edit", json=body(rec))
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["images"] == [] and "changed since it was queued" in done["failed_cells"][0]["error"]
    assert no_work(fake) and images(store) == 1


def test_a_gate_closed_after_submit_errors_the_job_unavailable(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    fake.loaded = False

    def load():
        fake.edit_qualified = False
        fake.loaded = True

    monkeypatch.setattr(fake, "load", load)
    response = client.post("/api/instruction-edit", json=body(rec))
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    # The whole job's class check (`check_refine_supported`) asks the gate
    # before any cell, so it ends in error, saving nothing.
    assert (done["status"], done["images"]) == ("error", [])
    assert done["error"].endswith(INSTRUCTION_EDIT_UNAVAILABLE)
    assert no_work(fake) and images(store) == 1


def test_a_gate_closed_as_the_cell_begins_fails_that_cell_unavailable(client, fake, store, monkeypatch):
    # Past the whole job's class check, the cell's own recheck (`_recheck_edit`
    # in the job loop) is what asks the gate again before the backend runs.
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    checked = fake.check_refine_supported

    def check_then_close(request):
        checked(request)
        fake.edit_qualified = False

    monkeypatch.setattr(fake, "check_refine_supported", check_then_close)
    response = client.post("/api/instruction-edit", json=body(rec))
    assert response.status_code == 200, response.text
    done = stream_job(client, response.json()["job_id"])[-1]
    assert done["images"] == [] and len(done["failed_cells"]) == 1, done
    assert done["failed_cells"][0]["error"].endswith(INSTRUCTION_EDIT_UNAVAILABLE)
    assert no_work(fake) and images(store) == 1


def test_each_edit_cell_asks_the_gate_and_its_source_again(fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    bound = instruction_edit_admission.bind_request(
        InstructionEditSubmission.model_validate(body(rec)).request(), KONTEXT_V1.model_id,
    )
    manager = JobManager(fake, None)
    assert manager._recheck_edit(bound) is None
    fake.edit_qualified = False
    with pytest.raises(ValueError, match="not available"):
        manager._recheck_edit(bound)
    fake.edit_qualified = True
    store.bin_images([rec.id])
    with pytest.raises(ValueError, match=f"source #{rec.id} is missing, in the recycle bin or changed"):
        manager._recheck_edit(bound)
    with pytest.raises(ValueError, match="Bind and verify"):
        manager._recheck_edit(InstructionEditSubmission.model_validate(body(rec)).request())
    assert no_work(fake)


def test_a_queued_edit_job_fails_closed_once_its_backend_is_unqualified(fake, store, monkeypatch):
    # A retained or queued edit job never runs past its backend's own answer:
    # closed before the job starts, it ends in error before any load.
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    bound = instruction_edit_admission.bind_request(
        InstructionEditSubmission.model_validate(body(rec)).request(), KONTEXT_V1.model_id,
    )

    async def scenario():
        loads: list[int] = []

        async def ensure_loaded():
            loads.append(1)

        manager = JobManager(fake, ensure_loaded)
        job = await manager.submit(bound)
        fake.edit_qualified = False  # before the job's task has taken a step
        while job.status not in {"done", "error", "cancelled"}:
            await asyncio.sleep(0.01)
        await manager.shutdown()
        return job, loads

    job, loads = asyncio.run(scenario())
    assert (job.status, job.error) == ("error", INSTRUCTION_EDIT_UNAVAILABLE)
    assert loads == [] and no_work(fake)


def test_submit_cells_refuses_an_unbound_or_ungated_edit(fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    unbound = InstructionEditSubmission.model_validate(body(rec)).request()
    bound = instruction_edit_admission.bind_request(unbound, KONTEXT_V1.model_id)

    async def scenario():
        manager = JobManager(fake, None)
        try:
            with pytest.raises(ValueError, match="Bind and verify"):
                await manager.submit(unbound)
            fake.edit_qualified = False
            with pytest.raises(ValueError, match="not available"):
                await manager.submit(bound)
        finally:
            await manager.shutdown()

    asyncio.run(scenario())
    assert no_work(fake)


def test_cancel_mid_edit_keeps_nothing(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    # Ten seconds of steps, cancelled after the first: margin for slow runners.
    fake.step_seconds = 0.1
    response = client.post("/api/instruction-edit", json=body(rec, steps=100))
    job_id = response.json()["job_id"]
    wait_for(client, job_id, lambda snapshot: snapshot["step"] >= 1)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    done = stream_job(client, job_id)[-1]
    assert done["status"] == "cancelled" and done["images"] == []
    assert len(fake.edit_calls) == 1 and images(store) == 1


def test_planned_keys_hold_a_live_edit_and_skip_it_once_unqualified(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    # Ten seconds of steps, cancelled after the first: margin for slow runners.
    fake.step_seconds = 0.1
    response = client.post("/api/instruction-edit", json=body(rec, steps=100))
    job_id = response.json()["job_id"]
    try:
        wait_for(client, job_id, lambda snapshot: snapshot["step"] >= 1)
        planned = main.jobs.planned_keys()
        ran = grid.Cell(fake.applied(fake.requests[0]), seed=7)
        assert planned == {(512, 512): set(grid.cell_keys(ran))}
        assert all(grid.edit_key(fake.requests[0].edit) in key for key in planned[(512, 512)])
        fake.edit_qualified = False
        assert main.jobs.planned_keys() == {}
    finally:
        fake.edit_qualified = True
        client.post(f"/api/jobs/{job_id}/cancel")
        stream_job(client, job_id)


# What stays held


@pytest.mark.parametrize("qualified", [False, True])
def test_generate_rerender_more_seeds_swap_and_finish_never_queue_an_edit(
    client, fake, store, monkeypatch, qualified,
):
    select(monkeypatch)
    fake.edit_qualified = qualified
    rec, bound = source(store)
    edited = store.save_image(
        Image.new("RGB", (512, 512), (50, 30, 10)),
        GenerateRequest(prompt="make it night", edit=bound, width=512, height=512, seed=7),
        seed=7, duration_ms=1, model_id=KONTEXT_V1.model_id, quantization="fp8",
        run_id="instruction-edit-run", batch=1,
    )
    held = INSTRUCTION_EDIT_REPLAY_HELD if qualified else INSTRUCTION_EDIT_UNAVAILABLE
    stored = GenerateRequest(prompt="make it night", edit=bound, width=512, height=512, seed=7)
    response = client.post("/api/generate", json=stored.model_dump(mode="json"))
    assert response.status_code == 422 and response.json()["detail"] == held, response.text
    unbound = {"prompt": "make it night", "edit": {"source": rec.id, "recipe": "kontext-v1"},
               "width": 512, "height": 512}
    response = client.post("/api/generate", json=unbound)
    assert response.status_code == 422 and "instruction-edit source" in response.text, response.text
    response = client.post("/api/rerender", json={"ids": [edited.id], "width": 512, "height": 512})
    assert response.status_code == 422 and response.json()["detail"] == held, response.text
    for route, payload, rows in (
        ("/api/more-seeds/preview", {"ids": [edited.id], "count": 2}, "results"),
        ("/api/swap-tags/preview", {"ids": [edited.id], "swaps": [{"find": "night", "replace": "day"}]}, "images"),
    ):
        response = client.post(route, json=payload)
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["queued"] == 0 and answer[rows][0]["status"] == "invalid", answer
    store.retain_run_job({
        "job_id": "retained-edit", "run_id": "retained-edit-run", "created_at": time.time(),
        "origin": "generate", "model_id": KONTEXT_V1.model_id, "quantization": "fp8",
        "width": 512, "height": 512, "batch": 1, "comparison": False, "metadata": {},
    }, [grid.Cell(stored, seed=7)])
    preview = client.post("/api/runs/retained-edit-run/finish/preview", json={}).json()
    assert preview["available"] is False and preview["reason"] == held
    response = client.post("/api/runs/retained-edit-run/finish", json={})
    assert response.status_code == 422, response.text
    assert no_work(fake) and fake.loads == 0
    assert images(store) == 2


@pytest.mark.parametrize("qualified", [False, True])
def test_text_to_image_on_the_editor_says_to_edit_an_image(client, fake, monkeypatch, qualified):
    select(monkeypatch)
    fake.edit_qualified = qualified
    response = client.post("/api/generate", json={"prompt": "a lighthouse", "width": 512, "height": 512})
    # Pointing at Edit by instruction… only where an edit can queue; a closed
    # gate's own reason otherwise, never a dead end.
    detail = response.json()["detail"]
    assert response.status_code == 422
    if qualified:
        assert detail == INSTRUCTION_EDIT_TEXT_TO_IMAGE
    else:
        assert detail.endswith(INSTRUCTION_EDIT_UNAVAILABLE) and "Use Edit by instruction" not in detail
    defaults = client.get("/api/status").json()["defaults"]
    assert defaults["text_to_image_supported"] is False and defaults["text_to_image_reason"] == detail
    assert no_work(fake)


# Preview and status


@pytest.mark.parametrize(("qualified", "kontext", "reason"), [
    (False, False, INSTRUCTION_EDIT_UNAVAILABLE),
    (False, True, INSTRUCTION_EDIT_UNAVAILABLE),
    (True, False, INSTRUCTION_EDIT_NEEDS_MODEL),
    (True, True, None),
])
def test_preview_and_status_follow_the_gate_and_the_selected_model(
    client, fake, store, monkeypatch, qualified, kontext, reason,
):
    if kontext:
        select(monkeypatch)
    fake.edit_qualified = qualified
    rec, bound = source(store, width=768, height=512)
    response = client.post("/api/instruction-edit/preview", json={"source": rec.id, "recipe": "kontext-v1"})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert (answer["available"], answer["reason"]) == (reason is None, reason)
    assert (answer["width"], answer["height"]) == (576, 384)
    assert (answer["steps"], answer["guidance_scale"]) == (28, 2.5)
    assert answer["edit"] == bound.model_dump(mode="json")
    defaults = client.get("/api/status").json()["defaults"]
    assert defaults["instruction_edit_supported"] is (reason is None)
    assert defaults["instruction_edit_reason"] == reason
    assert fake.loads == 0 and no_work(fake)


def test_a_case_variant_of_the_editor_id_is_not_the_editor_anywhere(client, fake, store, monkeypatch):
    # The recipe's identity is exact (`bind_request`): status, preview, the
    # load and the submit all answer the same for a case variant.
    variant = KONTEXT_V1.model_id.lower()
    assert variant != KONTEXT_V1.model_id
    select(monkeypatch, variant)
    fake.edit_qualified = True
    rec, _ = source(store)
    preview = client.post("/api/instruction-edit/preview", json={"source": rec.id, "recipe": "kontext-v1"}).json()
    assert (preview["available"], preview["reason"]) == (False, INSTRUCTION_EDIT_NEEDS_MODEL)
    defaults = client.get("/api/status").json()["defaults"]
    assert defaults["instruction_edit_supported"] is False
    assert defaults["instruction_edit_reason"] == INSTRUCTION_EDIT_NEEDS_MODEL
    response = client.post("/api/instruction-edit", json=body(rec))
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == INSTRUCTION_EDIT_NEEDS_MODEL
    response = client.post("/api/model/load")
    assert response.status_code != 200 and "exactly" in response.text, response.text
    assert fake.loads == 0 and no_work(fake)
