"""The job stream and cancellation over HTTP: every image reported, every
stream terminated, cancel and pause landing where they should."""
from __future__ import annotations

import asyncio
import time

import pytest

from backend.jobs import JobManager
from backend.schemas import GenerateRequest
from tests.api_helpers import run_grid, run_one, stream_job, wait_for
from tests.fakes import FakeBackend


def test_move_front_lists_the_authoritative_order_and_never_preempts(client, fake):
    fake.step_seconds = 0.01
    payload = {"width": 256, "height": 256, "steps": 35}
    first = client.post("/api/generate", json={**payload, "prompt": "first"}).json()["job_id"]
    wait_for(client, first, lambda s: s["current_call"] is not None)
    second = client.post("/api/generate", json={**payload, "prompt": "second"}).json()["job_id"]
    third = client.post("/api/generate", json={**payload, "prompt": "third"}).json()["job_id"]
    moved = client.post(f"/api/jobs/{third}/front")
    assert moved.status_code == 200
    assert moved.json()["active_jobs"] == [first, third, second]
    assert [job["job_id"] for job in client.get("/api/jobs").json()] == [first, third, second]
    assert client.post(f"/api/jobs/{first}/front").status_code == 409
    assert client.post("/api/jobs/unknown/front").status_code == 404
    for job_id in (first, third, second):
        assert stream_job(client, job_id)[-1]["status"] == "done"
    assert [request.prompt for request in fake.requests] == ["first", "third", "second"]
    assert client.post(f"/api/jobs/{third}/front").status_code == 409


def test_skip_keeps_earlier_images_continues_and_finish_recovers_the_hole(client, fake, store):
    fake.step_seconds = 0.05  # a 1 s cell: the skip must land while it runs
    job_id = client.post("/api/generate", json={
        "prompt": "a fox", "width": 256, "height": 256, "steps": 20, "runs": 3, "seed": 10,
    }).json()["job_id"]
    running = wait_for(client, job_id, lambda s: s["run"] == 2 and s["current_call"] is not None)
    response = client.post(f"/api/jobs/{job_id}/skip", params={"call": running["current_call"]})
    assert response.status_code == 200
    assert response.json()["skip_requested"]
    done = stream_job(client, job_id)[-1]
    assert done["status"] == "done" and done["failures"] == 0 and done["skipped"] == 1
    assert [image["seed"] for image in done["images"]] == [10, 12]
    assert done["skipped_cells"] == [{"label": None, "seed": 11, "error": "Skipped by user"}]
    deadline = time.monotonic() + 3
    while (saved := store.retained_run_jobs(done["run_id"])[0])["outcome"] is None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert saved["outcome"]["skipped_indices"] == [1]
    assert saved["outcome"]["completed_indices"] == [0, 2]
    answer = client.post(f"/api/runs/{done['run_id']}/finish/preview").json()
    assert answer["available"] and answer["cells"] == 1
    fake.step_seconds = 0
    accepted = client.post(f"/api/runs/{done['run_id']}/finish", json={"expected": answer["pin"]})
    assert accepted.status_code == 200
    recovered = stream_job(client, accepted.json()["job_id"])[-1]
    assert [image["seed"] for image in recovered["images"]] == [11]


def test_multiple_intentional_skips_do_not_trigger_the_failure_rule(client, fake):
    fake.step_seconds = 0.05  # a 1 s cell: the skip must land while it runs
    job_id = client.post("/api/generate", json={
        "prompt": "a fox", "width": 256, "height": 256, "steps": 20, "runs": 5,
    }).json()["job_id"]
    for run in range(1, 5):
        running = wait_for(client, job_id, lambda s, run=run: s["run"] == run and s["current_call"] is not None)
        assert client.post(f"/api/jobs/{job_id}/skip", params={"call": running["current_call"]}).status_code == 200
    done = stream_job(client, job_id)[-1]
    assert (done["status"], done["skipped"], done["failures"], len(done["images"])) == ("done", 4, 0, 1)


def test_a_stale_skip_cannot_skip_the_next_cell(client, fake):
    fake.step_seconds = 0.01
    job_id = client.post("/api/generate", json={
        "prompt": "x", "width": 256, "height": 256, "steps": 15, "runs": 2,
    }).json()["job_id"]
    old = wait_for(client, job_id, lambda s: s["current_call"] is not None)["current_call"]
    wait_for(client, job_id, lambda s: s["run"] == 2 and s["current_call"] is not None)
    response = client.post(f"/api/jobs/{job_id}/skip", params={"call": old})
    assert response.status_code == 409 and "already finished" in response.json()["detail"]
    done = stream_job(client, job_id)[-1]
    assert done["skipped"] == 0 and len(done["images"]) == 2


def test_a_skip_after_the_last_callback_keeps_the_completed_image(client, fake):
    job_id, answers = None, []
    fake.step_seconds = 0.01

    def late_skip():
        current = client.get(f"/api/jobs/{job_id}").json()["current_call"]
        answers.append(client.post(f"/api/jobs/{job_id}/skip", params={"call": current}))

    fake.after_generate = late_skip
    job_id = client.post("/api/generate", json={
        "prompt": "x", "width": 256, "height": 256, "steps": 5, "runs": 2,
    }).json()["job_id"]
    done = stream_job(client, job_id)[-1]
    assert all(answer.status_code == 200 for answer in answers)
    assert done["status"] == "done" and done["skipped"] == 0 and len(done["images"]) == 2


def test_skip_is_held_while_paused_and_during_then_refine(client, fake):
    fake.step_seconds = 0.01
    job_id = client.post("/api/generate", json={
        "prompt": "x", "width": 256, "height": 256, "steps": 20, "runs": 2,
    }).json()["job_id"]
    running = wait_for(client, job_id, lambda s: s["current_call"] is not None)
    client.post(f"/api/jobs/{job_id}/pause")
    paused = wait_for(client, job_id, lambda s: s["paused"])
    assert "Resume" in paused["skip_reason"]
    assert client.post(f"/api/jobs/{job_id}/skip", params={"call": running["current_call"]}).status_code == 409
    client.post(f"/api/jobs/{job_id}/cancel")
    assert stream_job(client, job_id)[-1]["status"] == "cancelled"
    job_id = client.post("/api/generate", json={
        "prompt": "x", "width": 256, "height": 256, "steps": 20,
        "then_refine": {"scale": 1.5, "strength": 0.5},
    }).json()["job_id"]
    stage = wait_for(client, job_id, lambda s: s["stage"] == "pass")
    assert "Finish run adds a pass a stop left missing" in stage["skip_reason"]
    assert client.post(f"/api/jobs/{job_id}/skip", params={"call": 1}).status_code == 409
    assert stream_job(client, job_id)[-1]["status"] == "done"


@pytest.mark.parametrize("call", [None, 0, -1, "bad"])
def test_skip_requires_a_positive_call_token(client, call):
    assert client.post("/api/jobs/unknown/skip", params={} if call is None else {"call": call}).status_code == 422


def test_skip_unknown_job_is_404(client):
    assert client.post("/api/jobs/unknown/skip", params={"call": 1}).status_code == 404


# ---------------------------------------------------------------- the stream


def test_a_sweep_reports_every_image_and_finishes(client, fake):
    _, events = run_one(client, runs=2, num_images=2, seed=7)
    images = [e for e in events if e["type"] == "image"]
    assert len(images) == 4
    assert events[-1]["type"] == "done" and events[-1]["status"] == "done"
    # seed + run * batch, then +1 per image in the batch.
    assert [i["image"]["seed"] for i in images] == [7, 8, 9, 10]
    assert fake.seeds == [7, 9]


def test_a_stream_opened_after_the_end_still_terminates(client):
    job_id, _ = run_one(client)
    events = stream_job(client, job_id)  # attach again, job long finished
    assert len(events) == 1 and events[0]["type"] == "done"
    assert events[0]["images"]


def test_a_deleted_image_is_counted_but_not_replayed(client, fake):
    # A page reattaching to a job is replayed what it made; an image deleted
    # meanwhile came back as a broken tile, and deleting it again answered
    # "unknown image". The count of what the job produced stays, beside how
    # many of those have gone.
    job_id, events = run_one(client, num_images=3, seed=3)
    made = [image["id"] for image in events[-1]["images"]]
    assert client.delete(f"/api/images/{made[0]}").status_code == 200

    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert [image["id"] for image in snapshot["images"]] == made[1:]
    assert (snapshot["image_count"], snapshot["deleted_count"]) == (3, 1)
    reattached = stream_job(client, job_id)
    assert [image["id"] for image in reattached[0]["images"]] == made[1:]
    listed = client.get("/api/jobs?include_finished=true").json()
    assert [(j["image_count"], j["deleted_count"]) for j in listed if j["job_id"] == job_id] == [(3, 1)]

    # Delete unstarred forgets on the jobs too, each counting only its own.
    other, _ = run_one(client, seed=40)
    client.post(f"/api/images/{made[1]}/favorite?value=true")
    assert client.delete("/api/gallery/unstarred?expected=2").status_code == 200
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert [image["id"] for image in snapshot["images"]] == [made[1]]
    assert (snapshot["image_count"], snapshot["deleted_count"]) == (3, 2)
    assert client.get(f"/api/jobs/{other}").json()["deleted_count"] == 1


def test_a_binned_image_leaves_the_replay_until_it_is_restored(client, fake):
    # The bin is gone from the gallery as a delete is, so the job forgets it
    # the same way - and a Restore puts it back, in the replay and the count.
    job_id, events = run_one(client, num_images=3, seed=3)
    made = [image["id"] for image in events[-1]["images"]]
    client.post("/api/images/bin", json={"ids": [made[0], made[2]]})
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert [image["id"] for image in snapshot["images"]] == [made[1]]
    assert (snapshot["image_count"], snapshot["deleted_count"]) == (3, 2)

    client.post("/api/images/restore", json={"ids": [made[0]]})
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert [image["id"] for image in snapshot["images"]] == made[:2]
    assert snapshot["deleted_count"] == 1
    # Emptied, it is gone for good: restoring what is no longer there is no-op.
    client.delete("/api/gallery/bin?expected=1")
    assert client.post("/api/images/restore", json={"ids": [made[2]]}).json()["missing"] == [made[2]]
    assert client.get(f"/api/jobs/{job_id}").json()["deleted_count"] == 1


def test_a_pass_replayed_says_its_source_is_binned_then_restored(client, fake):
    job_id, events = run_one(client, seed=1)
    source = events[-1]["images"][0]["id"]
    job_id, _ = run_one(client, refine=source, strength=0.5, steps=10)
    client.post("/api/images/bin", json={"ids": [source]})
    replay = client.get(f"/api/jobs/{job_id}").json()["images"]
    assert [(r["refine_missing"], r["refine_binned"]) for r in replay] == [(True, True)]
    client.post("/api/images/restore", json={"ids": [source]})
    replay = client.get(f"/api/jobs/{job_id}").json()["images"]
    assert [(r["refine_missing"], r["refine_binned"]) for r in replay] == [(False, False)]
    client.post("/api/images/bin", json={"ids": [source]})
    client.delete("/api/gallery/bin?expected=1")
    replay = client.get(f"/api/jobs/{job_id}").json()["images"]
    assert [(r["refine_missing"], r["refine_binned"]) for r in replay] == [(True, False)]


def test_active_jobs_are_listed_without_their_images(client, fake):
    fake.step_seconds = 0.02
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 20}
    ).json()["job_id"]
    live = client.get("/api/jobs").json()
    assert [job["job_id"] for job in live] == [job_id]
    assert "images" not in live[0]
    stream_job(client, job_id)
    assert client.get("/api/jobs").json() == []


def test_every_stream_sees_the_terminal_event(store):
    """One shared queue split the events between two readers."""

    async def scenario():
        manager = JobManager(FakeBackend(step_seconds=0.01))
        job = await manager.submit(
            GenerateRequest(prompt="x", width=256, height=256, steps=4, num_images=2)
        )

        async def collect(stream):
            seen = []
            async for event in stream:
                seen.append(event)
                if event["type"] == "done":
                    return seen
            return seen

        return await asyncio.gather(
            collect(manager.events(job)), collect(manager.events(job))
        )

    first, second = asyncio.run(scenario())
    for seen in (first, second):
        assert seen[-1]["type"] == "done"
        assert len(seen[-1]["images"]) == 2


# --------------------------------------------------------------- cancellation


def test_cancel_mid_sweep_keeps_what_finished(client, fake):
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 5, "runs": 8}
    ).json()["job_id"]
    # Poll for the first image rather than sleeping a guessed interval.
    wait_for(client, job_id, lambda s: len(s["images"]) >= 1)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    events = stream_job(client, job_id)
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1 and done[0]["status"] == "cancelled"
    # Landed within a step: fewer pipeline calls than runs, and everything that
    # finished before it is kept and reported.
    assert 1 <= len(done[0]["images"]) < 8
    assert len(fake.requests) < 8
    assert client.get("/api/gallery").json()["total"] == len(done[0]["images"])


def test_a_job_pauses_between_cells_and_hands_the_card_back(client, fake):
    # One person, one GPU: the card eight hours into a grid is the card they
    # want for twenty minutes. The hold lands at the cell boundary, the
    # worker acknowledges it, the allocator reserve is returned, and a resume
    # carries on from the next cell.
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/grid", json={"prompt": "x", "width": 256, "height": 256, "steps": 4, "steps_axis": "4,5,6,7,8,9", "seed": 1},
    ).json()["job_id"]
    wait_for(client, job_id, lambda s: len(s["images"]) >= 1)
    assert client.post(f"/api/jobs/{job_id}/pause").json()["pause_requested"] is True
    held = wait_for(client, job_id, lambda s: s["paused"])
    started = len(fake.requests)
    time.sleep(0.15)
    assert len(fake.requests) == started  # nothing new starts while held
    assert held["status"] == "running" and fake.cache_freed == 1
    assert client.get("/api/jobs").json()[0]["paused"] is True
    # Unload is still refused: the weights are needed for the resume.
    assert client.post("/api/model/unload").status_code == 409

    assert client.post(f"/api/jobs/{job_id}/pause?value=false").json()["pause_requested"] is False
    events = stream_job(client, job_id)
    assert events[-1]["status"] == "done" and len(events[-1]["images"]) == 6
    assert events[-1]["paused"] is False


def test_cancel_lands_on_a_paused_job(client, fake):
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/grid", json={"prompt": "x", "width": 256, "height": 256, "steps": 4, "steps_axis": "4,5,6,7", "seed": 1},
    ).json()["job_id"]
    wait_for(client, job_id, lambda s: len(s["images"]) >= 1)
    client.post(f"/api/jobs/{job_id}/pause")
    wait_for(client, job_id, lambda s: s["paused"])
    started = time.perf_counter()
    client.post(f"/api/jobs/{job_id}/cancel")
    done = stream_job(client, job_id)[-1]
    # A polled hold, not a bare Event.wait(): the cancel lands within the poll.
    assert done["status"] == "cancelled" and time.perf_counter() - started < 2.0
    assert 1 <= len(done["images"]) < 4


def test_cancelling_a_queued_job_ends_it_at_once(client, fake):
    fake.step_seconds = 0.02
    ahead = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 40}
    ).json()["job_id"]
    queued = client.post(
        "/api/generate", json={"prompt": "y", "width": 256, "height": 256}
    ).json()["job_id"]
    started = time.perf_counter()
    assert client.post(f"/api/jobs/{queued}/cancel").json()["status"] == "cancelled"
    events = stream_job(client, queued)
    # One terminal event, delivered without waiting for the job ahead.
    assert [e["type"] for e in events] == ["done"]
    assert time.perf_counter() - started < 0.5
    stream_job(client, ahead)
    assert [r.prompt for r in fake.requests] == ["x"]


def test_a_cancel_after_the_last_step_does_not_relabel_a_complete_run(client, fake):
    # The flag is polled between steps; a cancel landing during the decode or
    # the encode of the final image changes nothing and used to still flip the
    # status to "cancelled", so a script treating that as "resume it" re-queued
    # a finished grid.
    job_id = None
    fake.after_generate = lambda: client.post(f"/api/jobs/{job_id}/cancel")
    fake.step_seconds = 0.01
    job_id = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "steps": 2}
    ).json()["job_id"]
    events = stream_job(client, job_id)
    assert events[-1]["status"] == "done"
    assert len(events[-1]["images"]) == 1


def test_one_failed_cell_does_not_end_the_grid(client, fake):
    fake.fail_on = {2}
    _, events = run_grid(client, steps_axis="6,8,10", seed=1)
    done = events[-1]
    assert done["status"] == "done"
    assert done["failures"] == 1
    assert done["error"] == "1 cell failed and was skipped; last: RuntimeError: cell 2 blew up"
    # The two other cells rendered; the failed one is what resume will find missing.
    assert [r.steps for r in fake.requests] == [6, 8, 10]
    assert sorted(r["steps"] for r in done["images"]) == [6, 10]


def test_three_failures_in_a_row_end_the_job(client, fake):
    fake.fail_on = {1, 2, 3}
    _, events = run_grid(client, steps_axis="6,8,10,12", seed=1)
    assert events[-1]["status"] == "error"
    assert len(fake.requests) == 3
    # Said as the rule that ended it, where, and what was kept - the last
    # exception alone read as one bad cell - with the exception still there.
    done = events[-1]
    assert done["stopped_by"] == "failures"
    assert done["error"] == (
        "Stopped at cell 3 of 4 after 3 cells failed; 0 images kept. Last: RuntimeError: cell 3 blew up"
    )


def test_a_plain_jobs_failures_are_said_in_images_or_batches(client, fake):
    # "cell" is a grid's word: a plain job of three failing images said
    # "after 3 cells failed", and one skipped image "1 cell(s) failed".
    fake.fail_on = {1, 2, 3}
    _, events = run_one(client, runs=3, seed=1)
    assert events[-1]["error"] == (
        "Stopped at image 3 of 3 after 3 images failed; 0 images kept. Last: RuntimeError: cell 3 blew up"
    )
    fake.fail_on = {5}
    _, events = run_one(client, runs=2, seed=1)
    assert events[-1]["error"] == "1 image failed and was skipped; last: RuntimeError: cell 5 blew up"
    # Where each repeat makes several, a repeat is a batch.
    fake.fail_on = {6, 7}
    _, events = run_one(client, runs=3, num_images=2, seed=1)
    assert events[-1]["error"] == "2 batches failed and were skipped; last: RuntimeError: cell 7 blew up"


def test_a_skipped_cell_is_named_by_its_label_and_seed(client, fake):
    # The page names the hole in the sheet rather than counting it: which
    # cell, which seed, and its own error. With the records, not on every
    # per-cell event, and not in a listing.
    fake.fail_on = {2}
    job_id, events = run_grid(client, steps_axis="6,8,10", seed=5)
    done = events[-1]
    assert done["stopped_by"] is None
    [skipped] = done["failed_cells"]
    assert skipped["label"].startswith("8 steps, ")
    assert (skipped["seed"], skipped["error"]) == (5, "RuntimeError: cell 2 blew up")
    # The stream's opening snapshot carries them with the records; the
    # per-cell events after it do not.
    assert all("failed_cells" not in e for e in events[1:] if e["type"] == "status")
    listed = client.get("/api/jobs?include_finished=true").json()
    assert [j for j in listed if j["job_id"] == job_id][0].get("failed_cells") is None


def test_a_snapshot_says_what_the_job_is(client, fake):
    # A page that did not submit the job - a reload, the queue note - names
    # it from the snapshot: the prompt, the size, how many images it makes,
    # and whether it is a grid's cells or one request's repeats.
    fake.step_seconds = 0.02
    plain = client.post(
        "/api/generate",
        json={"prompt": "a whale " * 30, "width": 512, "height": 384, "num_images": 2, "runs": 3, "steps": 8},
    ).json()["job_id"]
    grid = client.post(
        "/api/grid", json={"prompt": "a fox", "width": 256, "height": 256, "steps_axis": "4,6", "seed": 1},
    ).json()["job_id"]
    # A grid with variables is named by its own prompt, the variables
    # unfilled: the first cell's prompt named it after that cell was done.
    varied = client.post(
        "/api/grid",
        json={"prompt": "a fox, {mood}", "width": 256, "height": 256, "prompt_axis": "mood = calm | happy", "seed": 1},
    ).json()["job_id"]
    listed = {j["job_id"]: j for j in client.get("/api/jobs").json()}
    first, second, third = listed[plain], listed[grid], listed[varied]
    assert first["prompt"] == ("a whale " * 30)[:120] and len(first["prompt"]) == 120
    assert (first["width"], first["height"], first["planned_images"], first["grid"]) == (512, 384, 6, False)
    assert first["prompt_template"] is None
    assert (second["prompt"], second["planned_images"], second["grid"], second["refine"]) == ("a fox", 2, True, None)
    assert (third["prompt"], third["prompt_template"]) == ("a fox, calm", "a fox, {mood}")
    # None of them a prompt comparison, whose versions the panel's grid cannot
    # resume (test_api_grid has one that is).
    assert not any(job["comparison"] for job in (first, second, third))
    client.post(f"/api/jobs/{plain}/cancel")
    stream_job(client, grid)
    stream_job(client, varied)
