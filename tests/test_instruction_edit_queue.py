"""`POST /api/instruction-edit`: one edit queued behind the backend's gate.

Driven over HTTP with the fake backend, whose gate (`edit_qualified`) a test
opens (the real backend's literal True is test_instruction_edit_release's).
The fake's edit branch
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
    INSTRUCTION_EDIT_ASPECT_UNQUALIFIED,
    INSTRUCTION_EDIT_LOADER_UNQUALIFIED,
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
    # A 1:1 source of another size: the qualified 512x512 from its own pixels.
    rec, bound = source(store, width=320, height=320)
    response = client.post("/api/instruction-edit", json=body(rec))
    assert response.status_code == 200, response.text
    accepted = response.json()
    assert accepted["seed"] == 7
    done = stream_job(client, accepted["job_id"])[-1]
    assert done["status"] == "done", done
    assert len(done["images"]) == 1
    assert fake.edit_calls == [{
        "width": 512, "height": 512, "max_area": 512 * 512, "_auto_resize": False,
        "num_images_per_prompt": 1, "max_sequence_length": 512, "image_size": (512, 512),
        "seed": 7, "steps": KONTEXT_V1.default_steps, "guidance_scale": KONTEXT_V1.default_guidance_scale,
    }]
    assert len(fake.requests) == 1 and fake.requests[0].edit == bound
    saved = store.get_image(done["images"][0]["id"])
    assert saved is not None
    assert saved.edit == bound and saved.model_id == KONTEXT_V1.model_id
    assert (saved.width, saved.height, saved.batch, saved.seed) == (512, 512, 1, 7)
    assert saved.negative_prompt is None and saved.refine is None and saved.mask is None
    assert saved.prompt == "make it night" and saved.run_id == accepted["run_id"]
    with Image.open(store.settings.outputs_dir / saved.filename) as image:
        assert image.size == (512, 512)
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


# The measured envelope (stage 4): an open gate holds what the card has not measured.

INT8 = INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="int8 quantization")
THREE_TWO = INSTRUCTION_EDIT_ASPECT_UNQUALIFIED.format(a=3, b=2, w=576, h=384)


def leave_envelope(monkeypatch):
    # The editor still selected, at a quantization the card never measured.
    monkeypatch.setattr(model_selection, "_selection", (KONTEXT_V1.model_id, 1, "int8", "model"))


def test_preview_and_submit_hold_an_unmeasured_aspect_without_a_job(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    square, _ = source(store)
    wide, _ = source(store, width=768, height=512)
    # The server can edit: its status says so, and a 1:1 preview is available.
    defaults = client.get("/api/status").json()["defaults"]
    assert defaults["instruction_edit_supported"] is True and defaults["instruction_edit_reason"] is None
    answer = client.post("/api/instruction-edit/preview", json={"source": square.id, "recipe": "kontext-v1"}).json()
    assert (answer["available"], answer["reason"], answer["width"], answer["height"]) == (True, None, 512, 512)
    # A 3:2 source is said at its size and held in words.
    answer = client.post("/api/instruction-edit/preview", json={"source": wide.id, "recipe": "kontext-v1"}).json()
    assert (answer["available"], answer["reason"], answer["width"], answer["height"]) == (False, THREE_TWO, 576, 384)
    # And its submission, at the size the frozen table gives it, queues nothing.
    response = client.post("/api/instruction-edit", json=body(wide, width=576, height=384))
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == THREE_TWO
    assert fake.loads == 0 and no_work(fake) and images(store) == 2
    # An internal caller with a bound 3:2 cell is refused whole, too.
    bound = instruction_edit_admission.bind_request(
        InstructionEditSubmission.model_validate(body(wide, width=576, height=384)).request(), KONTEXT_V1.model_id,
    )

    async def scenario():
        manager = JobManager(fake, None)
        try:
            with pytest.raises(ValueError) as caught:
                await manager.submit(bound)
            return str(caught.value)
        finally:
            await manager.shutdown()

    assert asyncio.run(scenario()) == THREE_TWO
    assert no_work(fake)


def test_an_unmeasured_loader_is_held_in_words_before_any_source_is_read(client, fake, store, monkeypatch):
    fake.edit_qualified = True
    leave_envelope(monkeypatch)
    rec, _ = source(store)
    answer = client.post("/api/instruction-edit/preview", json={"source": rec.id, "recipe": "kontext-v1"}).json()
    assert (answer["available"], answer["reason"]) == (False, INT8)
    defaults = client.get("/api/status").json()["defaults"]
    assert defaults["instruction_edit_supported"] is False and defaults["instruction_edit_reason"] == INT8
    # Generate on the editor says the loader's words, not a pointer at the dialog.
    assert INT8 in defaults["text_to_image_reason"] and "Use Edit by instruction" not in defaults["text_to_image_reason"]

    def forbidden(*args, **kwargs):
        pytest.fail("an unmeasured loader reached source decoding")

    monkeypatch.setattr(instruction_edit_admission, "bind_request", forbidden)
    response = client.post("/api/instruction-edit", json=body(rec))
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == INT8
    assert fake.loads == 0 and no_work(fake)


def test_the_fake_host_decides_the_platform_and_diffusers_answers(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    fake.edit_host = ("linux", "0.40.0")
    answer = client.post("/api/instruction-edit/preview", json={"source": rec.id, "recipe": "kontext-v1"}).json()
    assert answer["reason"] == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="Linux (WSL)")
    fake.edit_host = ("win32", None)
    answer = client.post("/api/instruction-edit/preview", json={"source": rec.id, "recipe": "kontext-v1"}).json()
    assert answer["reason"] == INSTRUCTION_EDIT_LOADER_UNQUALIFIED.format(choice="a missing diffusers")
    assert no_work(fake)


def test_recheck_refuses_a_cell_whose_selection_left_the_envelope(fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, _ = source(store)
    bound = instruction_edit_admission.bind_request(
        InstructionEditSubmission.model_validate(body(rec)).request(), KONTEXT_V1.model_id,
    )
    manager = JobManager(fake, None)
    assert manager._recheck_edit(bound) is None
    leave_envelope(monkeypatch)
    with pytest.raises(ValueError) as caught:
        manager._recheck_edit(bound)
    assert str(caught.value) == INT8
    open_gate(fake, monkeypatch)
    # A bound 3:2 cell (an internal caller's) fails its recheck in words too.
    wide, _ = source(store, width=768, height=512)
    wide_bound = instruction_edit_admission.bind_request(
        InstructionEditSubmission.model_validate(body(wide, width=576, height=384)).request(), KONTEXT_V1.model_id,
    )
    with pytest.raises(ValueError) as caught:
        manager._recheck_edit(wide_bound)
    assert str(caught.value) == THREE_TWO
    assert no_work(fake)


def test_a_queued_edit_job_errors_once_its_selection_leaves_the_envelope(fake, store, monkeypatch):
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
        leave_envelope(monkeypatch)  # before the job's task has taken a step
        while job.status not in {"done", "error", "cancelled"}:
            await asyncio.sleep(0.01)
        await manager.shutdown()
        return job, loads

    job, loads = asyncio.run(scenario())
    assert (job.status, job.error) == ("error", INT8)
    assert loads == [] and no_work(fake)


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


# Replay: More seeds runs an edit inside the envelope; the rest stays held


def edited(store, bound, *, width=512, height=512, seed=7, run_id="instruction-edit-run"):
    """A saved edit of `bound`'s source, as the editor writes one."""
    return store.save_image(
        Image.new("RGB", (width, height), (50, 30, 10)),
        GenerateRequest(prompt="make it night", edit=bound, width=width, height=height, seed=seed),
        seed=seed, duration_ms=1, model_id=KONTEXT_V1.model_id, quantization="fp8",
        run_id=run_id, batch=1,
    )


def test_more_seeds_of_an_edit_queues_new_seeds_of_the_original_source(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    rec, bound = source(store)
    edit = edited(store, bound)
    answer = client.post("/api/more-seeds/preview", json={"ids": [edit.id], "count": 2}).json()
    (row,) = answer["results"]
    assert (row["status"], row["first"], row["last"], row["batch"]) == ("queued", 8, 9, 1), row
    assert answer["queued"] == 2 and answer["intent"]
    response = client.post("/api/more-seeds", json={
        "ids": [edit.id], "count": 2, "expected": answer["ids"], "intent": answer["intent"],
    })
    assert response.status_code == 200, response.text
    accepted = response.json()
    assert accepted["run_id"] == "instruction-edit-run" and accepted["queued"] == 2
    done = stream_job(client, accepted["job_id"])[-1]
    assert done["status"] == "done" and len(done["images"]) == 2, done
    # One image per call, each an edit of the original source - never of
    # the child it was asked from - its proof unchanged.
    assert [call["seed"] for call in fake.edit_calls] == [8, 9]
    assert all(call["num_images_per_prompt"] == 1 and call["image_size"] == (512, 512) for call in fake.edit_calls)
    made = [store.get_image(image["id"]) for image in done["images"]]
    assert all(saved is not None for saved in made)
    assert [(saved.seed, saved.edit, saved.run_id, saved.width, saved.height) for saved in made if saved] == [
        (8, bound, "instruction-edit-run", 512, 512), (9, bound, "instruction-edit-run", 512, 512),
    ]
    assert store.get_image(rec.id) is not None
    # Asked again, the seeds carry on past them.
    again = client.post("/api/more-seeds/preview", json={"ids": [edit.id], "count": 2}).json()
    assert (again["results"][0]["first"], again["results"][0]["last"]) == (10, 11)


@pytest.mark.parametrize("case", ["changed", "binned", "foreign", "non-square", "closed", "loader"])
def test_more_seeds_refuses_changed_source_or_unmeasured_envelope(client, fake, store, monkeypatch, case):
    open_gate(fake, monkeypatch)
    if case == "non-square":
        # An imported or older 3:2 edit: it binds and inspects, never runs.
        rec, bound = source(store, width=768, height=512)
        edit = edited(store, bound, width=576, height=384)
    else:
        rec, bound = source(store)
        edit = edited(store, bound)
    if case == "changed":
        Image.new("RGB", (256, 256), (1, 2, 3)).save(store.settings.outputs_dir / rec.filename)
    elif case == "binned":
        store.bin_images([rec.id])
    elif case == "foreign":
        select(monkeypatch, "black-forest-labs/FLUX.1-Krea-dev")
    elif case == "closed":
        fake.edit_qualified = False
    elif case == "loader":
        leave_envelope(monkeypatch)
    expected = {
        "changed": ("invalid", f"Instruction-edit source #{rec.id} no longer matches its recorded pixels or provenance."),
        "binned": ("invalid", f"Instruction-edit source #{rec.id} is missing or in the recycle bin."),
        "foreign": ("foreign", f"made on {KONTEXT_V1.model_id}; the server is running black-forest-labs/FLUX.1-Krea-dev"),
        "non-square": ("invalid", THREE_TWO),
        "closed": ("invalid", INSTRUCTION_EDIT_UNAVAILABLE),
        "loader": ("invalid", INT8),
    }[case]
    answer = client.post("/api/more-seeds/preview", json={"ids": [edit.id], "count": 2}).json()
    (row,) = answer["results"]
    assert (row["status"], row["detail"]) == expected, row
    assert answer["queued"] == 0 and answer["intent"] is None
    response = client.post("/api/more-seeds", json={"ids": [edit.id], "count": 2})
    assert response.status_code == 422, response.text
    assert "no selected batch can have more seeds" in response.json()["detail"]
    assert no_work(fake) and fake.loads == 0


def test_rerender_of_an_edit_is_refused_naming_ids(client, fake, store, monkeypatch):
    open_gate(fake, monkeypatch)
    _, bound = source(store)
    edit = edited(store, bound)
    plain = store.save_image(
        Image.new("RGB", (512, 512)), GenerateRequest(prompt="a lighthouse", width=512, height=512),
        seed=3, duration_ms=1, model_id=KONTEXT_V1.model_id, quantization="fp8",
    )
    words = (
        f"An instruction edit renders at its source's frozen size (512x512 for #{edit.id}); "
        "use More seeds for new seeds, or Reuse to edit again."
    )
    # At its own size and at another, the same words.
    for size in ((512, 512), (768, 768)):
        response = client.post("/api/rerender", json={"ids": [edit.id], "width": size[0], "height": size[1]})
        assert response.status_code == 422 and response.json()["detail"] == words, response.text
    # A mixed selection is refused whole, with the way to leave the edit out.
    response = client.post("/api/rerender", json={"ids": [plain.id, edit.id], "width": 512, "height": 512})
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == f"{words} Unpick #{edit.id} to re-render the others."
    response = client.post("/api/rerender", json={"search": "", "favorites": False, "width": 512, "height": 512})
    assert response.status_code == 422, response.text
    assert response.json()["detail"].endswith("Add edit:no to the search to re-render the others.")
    assert no_work(fake) and fake.loads == 0


@pytest.mark.parametrize("qualified", [False, True])
def test_generate_and_finish_still_hold_an_edit(client, fake, store, monkeypatch, qualified):
    select(monkeypatch)
    fake.edit_qualified = qualified
    rec, bound = source(store)
    edit = edited(store, bound)
    held = INSTRUCTION_EDIT_REPLAY_HELD if qualified else INSTRUCTION_EDIT_UNAVAILABLE
    stored = GenerateRequest(prompt="make it night", edit=bound, width=512, height=512, seed=7)
    response = client.post("/api/generate", json=stored.model_dump(mode="json"))
    assert response.status_code == 422 and response.json()["detail"] == held, response.text
    unbound = {"prompt": "make it night", "edit": {"source": rec.id, "recipe": "kontext-v1"},
               "width": 512, "height": 512}
    response = client.post("/api/generate", json=unbound)
    assert response.status_code == 422 and "instruction-edit source" in response.text, response.text
    # Swap tags is unchanged: it never rewrites an edit, in its own words.
    response = client.post("/api/swap-tags/preview", json={
        "ids": [edit.id], "swaps": [{"find": "night", "replace": "day"}],
    })
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["queued"] == 0 and answer["images"][0]["status"] == "invalid", answer
    assert answer["images"][0]["detail"] == "Swap tags cannot rewrite an instruction edit; use its instruction workflow."
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
    # The server's reason first; where it has none, this 3:2 source's own:
    # its size is said, but only 1:1 was measured.
    held = reason or INSTRUCTION_EDIT_ASPECT_UNQUALIFIED.format(a=3, b=2, w=576, h=384)
    assert (answer["available"], answer["reason"]) == (False, held)
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
