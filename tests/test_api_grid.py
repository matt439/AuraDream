"""Grid search over HTTP: the cells a grid runs, the preview, resume, the
re-render, the seeds it publishes and the gallery paging behind it."""
from __future__ import annotations

import dataclasses
import json
import threading
import time

import pytest
from PIL import Image

from backend import grid, main, storage
from backend.config import settings
from backend.model_defaults import defaults_for
from backend.schemas import GenerateRequest
from tests.api_helpers import run_grid, run_one, stream_job, wait_for


# ---------------------------------------------------------------- grid search


RANDOM_BODY = {
    "prompt": "a {size} fox", "negative_prompt": "not {size}",
    "prompt_axis": "size = small | large | none", "prompt_mode": "random",
    "width": 256, "height": 256, "seed": 42, "num_images": 2,
    "steps_axis": "6,8", "seeds": 2, "guidance_scale": 2.0,
}


def test_random_prompt_preview_and_render_agree_per_image_and_save_the_choices(client, fake, store):
    # Apply real CFG normalization so recorded negative prompts and Resume
    # compare the same effective true CFG, as a real backend does.
    fake.cfg_threshold = 1.0
    preview = client.post("/api/grid/preview", json=RANDOM_BODY).json()
    assert (preview["cells"], preview["images"], preview["prompt_sample"]) == (8, 8, False)
    assert preview["summary"].startswith("one choice per image x ")
    assert "2 images per seed (run separately)" in preview["summary"]
    accepted = client.post("/api/grid", json=RANDOM_BODY).json()
    done = stream_job(client, accepted["job_id"])[-1]
    assert done["status"] == "done" and len(done["images"]) == 8
    assert fake.seeds == [42, 42, 43, 43, 44, 44, 45, 45]
    assert all(request.num_images == 1 for request in fake.requests)
    assert set(preview["values"]["prompt"]) == {request.prompt for request in fake.requests}
    assert set(preview["values"]["negative_prompt"]) == {request.negative_prompt for request in fake.requests}
    for image in done["images"]:
        expected = grid.random_assignment({"size": ["small", "large", ""]}, image["seed"])
        assert image["variables"] == expected and image["batch"] == 1
        with Image.open(store.settings.outputs_dir / image["filename"]) as png:
            assert json.loads(png.info["local-diffusion"])["variables"] == expected
        record = store.get_image(image["id"])
        assert grid.cell_from_record(record).variables == expected
    resumed = client.post("/api/grid/preview", json={**RANDOM_BODY, "resume": True}).json()
    assert (resumed["images"], resumed["cells"], resumed["skipped"]) == (0, 0, 8)
    assert resumed["values"]["prompt"] == preview["values"]["prompt"]
    assert client.post("/api/grid", json={**RANDOM_BODY, "resume": True}).status_code == 409


def test_unseeded_random_preview_is_one_expansion_and_honestly_marked_as_a_sample(client, monkeypatch):
    drawn = []

    def seed():
        drawn.append(42)
        assert len(drawn) == 1
        return 42

    monkeypatch.setattr(grid, "random_seed", seed)
    body = {**RANDOM_BODY, "seed": None}
    preview = client.post("/api/grid/preview", json=body).json()
    assert preview["prompt_sample"] and preview["images"] == 8 and drawn == [42]
    assert preview["values"]["prompt"][0] == "a small fox"
    listed = client.post("/api/grid/preview", json={**body, "seed_axis": "42,80"}).json()
    assert listed["prompt_sample"] is False


@pytest.mark.parametrize("changes,said", [
    ({"prompt_axis": ""}, "needs prompt variables"),
    ({"prompt_mode": "unknown"}, "cross"),
    ({"prompt_variants": [{"label": "one", "prompt": "a fox"},
                          {"label": "two", "prompt": "a wolf"}]}, "prompt variants"),
])
def test_random_prompt_mode_rejects_incomplete_or_conflicting_requests(client, changes, said):
    response = client.post("/api/grid/preview", json={**RANDOM_BODY, **changes})
    assert response.status_code == 422 and said in str(response.json()["detail"])


def test_random_prompt_resume_skips_images_held_in_the_live_queue(client, fake):
    body = {**RANDOM_BODY, "seeds": 1}
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    try:
        first = client.post("/api/grid", json=body).json()["job_id"]
        preview = client.post("/api/grid/preview", json={**body, "resume": True}).json()
        assert preview["cells"] == 0 and preview["skipped"] + preview["queued"] == 4
        assert client.post("/api/grid", json={**body, "resume": True}).status_code == 409
    finally:
        release.set()
    assert stream_job(client, first)[-1]["status"] == "done"
    assert len(fake.requests) == 4


def test_random_prompt_duplicates_are_counted_after_the_pipeline_drops(client, fake):
    fake.cfg_threshold = 1.0
    fake.mu_supported = False
    body = {
        **RANDOM_BODY, "num_images": 1, "seeds": 1, "steps_axis": "6",
        "mu_axis": "none,1.15", "guidance_scale": 0.5,
    }
    preview = client.post("/api/grid/preview", json=body).json()
    assert (preview["images"], preview["duplicates"]) == (2, 1)


def test_random_prompt_budget_counts_complete_original_seed_groups(client, monkeypatch):
    from backend import gallery_reports

    monkeypatch.setattr(gallery_reports, "step_cost_ms", lambda *args: 1000.0)
    # Each original seed group has two images, each at 6 and 8 steps:
    # starts at 0, 6, 14, 20 seconds. Completing the first image alone
    # must not count a complete seed group.
    for seconds, expected in [(14, 0), (20, 1), (48, 2)]:
        response = client.post(
            "/api/grid/preview", json={**RANDOM_BODY, "budget_minutes": seconds / 60}
        )
        assert response.status_code == 200, response.text
        assert response.json()["budget_seeds"] == expected


def test_random_choices_check_the_actual_single_image_batch_budget(client, monkeypatch):
    from backend import schemas_generate

    monkeypatch.setattr(
        schemas_generate, "settings", dataclasses.replace(schemas_generate.settings, max_batch_pixels=256 * 256)
    )
    random = client.post("/api/grid/preview", json=RANDOM_BODY)
    assert random.status_code == 200 and random.json()["images"] == 8
    crossed = client.post("/api/grid/preview", json={**RANDOM_BODY, "prompt_mode": "cross"})
    assert crossed.status_code == 422 and "in one batch" in str(crossed.json()["detail"])


def test_a_grid_runs_every_cell_of_every_seed(client, fake):
    _, events = run_grid(client, steps_axis="6,8", cfg_axis="0,1", seeds=2, seed=42)

    assert events[-1]["type"] == "done" and events[-1]["status"] == "done"
    # Seeds outermost, then steps, then CFG - the order a half-finished grid
    # depends on to still be a usable comparison.
    assert [(r.steps, r.guidance_scale) for r in fake.requests] == [
        (6, 0.0), (6, 1.0), (8, 0.0), (8, 1.0),
        (6, 0.0), (6, 1.0), (8, 0.0), (8, 1.0),
    ]
    assert fake.seeds == [42, 42, 42, 42, 43, 43, 43, 43]
    assert len(events[-1]["images"]) == 8


def test_a_grid_can_take_its_seeds_as_a_list(client, fake):
    body = {
        "prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "seed_axis": "4471, 8823",
    }
    preview = client.post("/api/grid/preview", json=body).json()
    assert preview["values"]["seed"] == [4471, 8823]
    assert preview["cells"] == 4 and "2 listed seeds" in preview["summary"]
    stream_job(client, client.post("/api/grid", json=body).json()["job_id"])
    assert fake.seeds == [4471, 4471, 8823, 8823]
    # Determinate on its own, so a resume needs no base seed - and finds it all done.
    assert client.post("/api/grid", json={**body, "resume": True}).status_code == 409
    # A counted axis still shows no seed values: they are derived, not listed.
    counted = client.post("/api/grid/preview", json={**body, "seed_axis": "", "seeds": 2}).json()
    assert counted["values"]["seed"] == [] and counted["summary"].endswith("2 seeds")


def test_a_grid_says_which_cell_is_running(client, fake):
    _, events = run_grid(client, steps_axis="6,8", seed=1)
    labels = [event["label"] for event in events if event.get("label")]
    # The axes the grid did not vary are filled from the default checkpoint's
    # profile, so the expected label follows whichever checkpoint that is.
    profile = defaults_for(settings.model_id)
    assert grid._label({}, 6, profile["guidance_scale"], profile["mu"]) in labels
    # A plain request varies nothing, so it labels nothing.
    _, plain = run_one(client, runs=2)
    assert not [event for event in plain if event.get("label")]


def test_the_preview_expands_the_same_grid_it_would_run(client):
    body = {
        "prompt": "a lighthouse in fog",
        "width": 256,
        "height": 256,
        "steps_axis": "6-10x2",
        "mu_axis": "none,1.15",
        "seeds": 2,
    }
    preview = client.post("/api/grid/preview", json=body).json()
    assert preview["values"]["steps"] == [6, 8, 10]
    assert preview["values"]["mu"] == [None, 1.15]
    assert (preview["cells"], preview["images"]) == (12, 12)
    # Nothing in the gallery at this size yet, so it promises no duration at all
    # rather than guessing one.
    assert preview["seconds"] is None


def test_the_preview_counts_the_cells_that_are_one_picture(client, fake):
    # A negative-prompt variable under a CFG that never clears the pipeline's
    # threshold: three cells on paper, one image in pixels. The preview says
    # so before the night is spent, and the finished job counts what it made.
    fake.cfg_threshold = 1.0
    body = {
        "prompt": "a lighthouse in fog",
        "negative_prompt": "{avoid}",
        "prompt_axis": "avoid = blurry | dark | noisy",
        "guidance_scale": 0.5,
        "width": 256,
        "height": 256,
        "seed": 1,
    }
    preview = client.post("/api/grid/preview", json=body).json()
    assert (preview["cells"], preview["duplicates"]) == (3, 2)
    body["guidance_scale"] = 2.0
    assert client.post("/api/grid/preview", json=body).json()["duplicates"] == 0
    # The fake draws the same blank canvas for every cell, so the job's own
    # count reads 1 distinct - the number an ignored axis shows up in.
    job_id = client.post("/api/grid", json=body).json()["job_id"]
    done = stream_job(client, job_id)[-1]
    assert (done["image_count"], done["distinct_count"]) == (3, 1)
    assert all(len(image["content_hash"]) == 16 for image in done["images"])
    runs = client.get("/api/runs").json()
    assert (runs[0]["images"], runs[0]["distinct"]) == (3, 1)


def test_a_grid_varies_the_prompt_through_its_placeholders(client, fake):
    _, events = run_grid(
        client,
        prompt="{size} dog sitting on {ground}",
        prompt_axis="size = big | small | none\nground = grass",
        seed=7,
    )
    assert events[-1]["type"] == "done" and events[-1]["status"] == "done"
    assert [request.prompt for request in fake.requests] == [
        "big dog sitting on grass",
        "small dog sitting on grass",
        # `none` is a value: the placeholder and the space it sat in both go.
        "dog sitting on grass",
    ]
    # One prompt per cell, but one seed for all of them - the comparison is the
    # same noise under a different word.
    assert fake.seeds == [7, 7, 7]
    # Every record carries the assignment that made it, `none` as "".
    assert [r["variables"] for r in events[-1]["images"]] == [
        {"size": "big", "ground": "grass"},
        {"size": "small", "ground": "grass"},
        {"size": "", "ground": "grass"},
    ]


def test_the_preview_keeps_an_indented_prompt_s_shape_in_its_none_cell(client):
    # The preview is what the page lists the prompts from, by line number
    # (generation.js, promptListing): every cell keeps the prompt's lines and
    # their indentation, the none cell short of its value and its space only.
    prompt = "<character_1>\n    <n>miku</n>\n    <appearance>chibi, {hair} long_hair</appearance>\n</character_1>"
    body = {
        "prompt": prompt, "width": 256, "height": 256,
        "prompt_axis": "hair = blue_hair | none",
    }
    preview = client.post("/api/grid/preview", json=body).json()
    assert preview["values"]["prompt"] == [
        prompt.replace("{hair}", "blue_hair"),
        prompt.replace("{hair} ", ""),
    ]


def test_a_grid_stops_at_its_time_budget_and_says_so(client, fake):
    # Over-provision on purpose and take what the night reaches; the rest is
    # a resume. The budget means "start no cell after": the cell in flight
    # finishes, and the stop is reported as the plan working, not a cancel.
    fake.step_seconds = 0.02
    body = {
        "prompt": "x", "width": 256, "height": 256, "steps": 5, "steps_axis": "5,6,7,8",
        "seed": 1, "budget_minutes": 0.05 / 60,  # 50 ms
    }
    _, events = run_grid(client, **body)
    done = events[-1]
    assert done["status"] == "cancelled" and done["stopped_by"] == "budget"
    assert 1 <= len(done["images"]) < 4
    assert done["deadline"] is not None
    # Resume picks up exactly the cells the clock cut off.
    preview = client.post(
        "/api/grid/preview", json={**body, "budget_minutes": None, "resume": True}
    ).json()
    assert preview["skipped"] == len(done["images"])
    assert preview["cells"] == 4 - len(done["images"])


def test_a_pause_stops_the_time_budgets_clock(client, fake):
    # The budget is working time. Its deadline was fixed at submission, so a
    # grid paused past it ended at the Resume with no cell run - "Time
    # budget reached" - however little of the budget it had used.
    fake.step_seconds = 0.01
    job_id = client.post("/api/grid", json={
        "prompt": "x", "width": 256, "height": 256, "steps": 4, "steps_axis": "4,5,6,7,8,9",
        "seed": 1, "budget_minutes": 1.5 / 60,  # 1.5 s; the six cells take about 0.4 s
    }).json()["job_id"]
    wait_for(client, job_id, lambda s: len(s["images"]) >= 1)
    client.post(f"/api/jobs/{job_id}/pause")
    held = wait_for(client, job_id, lambda s: s["paused"])
    time.sleep(2.0)
    assert time.time() > held["deadline"]
    client.post(f"/api/jobs/{job_id}/pause?value=false")
    done = stream_job(client, job_id)[-1]
    assert done["status"] == "done" and done["stopped_by"] is None and len(done["images"]) == 6
    # Moved on by the pause, which the page's "left (budget)" reads.
    assert done["deadline"] - held["deadline"] >= 2.0


def test_the_preview_shows_the_prompts_it_would_render(client):
    preview = client.post(
        "/api/grid/preview",
        json={
            "prompt": "{size} dog on {ground}",
            "width": 256,
            "height": 256,
            "prompt_axis": "size = big | small\nground = grass | dirt",
            "steps_axis": "6,8",
        },
    ).json()
    # The finished text, not the variables behind it: the substitution is the
    # half of a prompt grid that is easy to get wrong.
    assert preview["values"]["prompt"] == [
        "big dog on grass",
        "big dog on dirt",
        "small dog on grass",
        "small dog on dirt",
    ]
    assert (preview["cells"], preview["images"]) == (8, 8)
    assert preview["summary"].startswith("4 prompts x 2 steps")


def test_a_prompt_axis_that_does_not_line_up_is_a_422(client):
    body = {"prompt": "a {size} dog", "width": 256, "height": 256}
    unused = client.post("/api/grid/preview", json={**body, "prompt_axis": "colour = red"})
    assert unused.status_code == 422
    assert "{colour}" in unused.json()["detail"]
    # No variables at all with placeholders in the prompt is the easiest
    # mistake in the feature - the box forgotten, or cleared - and used to
    # render the literal braces into every image of the night.
    unfilled = client.post("/api/grid/preview", json={**body, "prompt_axis": ""})
    assert unfilled.status_code == 422
    assert "{size}" in unfilled.json()["detail"]
    # A prompt with no placeholders and no variables is the image Generate
    # would have made.
    plain = client.post("/api/grid/preview", json={"prompt": "a dog", "width": 256, "height": 256})
    assert plain.status_code == 200
    assert plain.json()["values"]["prompt"] == ["a dog"]


# ------------------------------------------------------------ prompt variants

# The analyser's comparison, in the order written - which is not alphabetical,
# so a sheet sorted by label would put `original` last.
VARIANTS = [
    {"label": "rewrite", "prompt": "A woman in a yellow raincoat holds a red umbrella."},
    {"label": "original", "prompt": "a red umbrella held by a woman in a yellow raincoat"},
    {"label": "reordered", "prompt": "a woman in a yellow raincoat holding a red umbrella"},
]
TEXTS = [variant["prompt"] for variant in VARIANTS]
LABELS = [variant["label"] for variant in VARIANTS]


def test_prompt_variants_run_as_one_grid_on_shared_seeds(client, fake):
    _, events = run_grid(client, prompt_variants=VARIANTS, seeds=2, seed=21)
    done = events[-1]
    assert done["status"] == "done"
    # Seeds outermost, the variants in the order given inside each; the base
    # prompt the request sits on is not one of them.
    assert [request.prompt for request in fake.requests] == TEXTS + TEXTS
    assert fake.seeds == [21, 21, 21, 22, 22, 22]
    # Every record carries its label as the one variable - what the sheet lays
    # out as a row - and all of them are one run.
    assert [record["variables"] for record in done["images"]] == [
        {"variant": label} for label in LABELS
    ] * 2
    assert len({record["run_id"] for record in done["images"]}) == 1
    labels = [event["label"] for event in events if event.get("label")]
    assert any(label.startswith("variant=original,") for label in labels)
    # The snapshot says it was a comparison: a stopped one is not the panel's
    # grid to resume, and the last-run banner offers no Resume for it.
    assert done["comparison"] is True
    assert all(event["comparison"] for event in events if event["type"] == "status")


@pytest.mark.parametrize("total_seeds", [4, 8])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_comparison_batches_preserve_the_total_shared_seeds(client, fake, total_seeds, batch):
    # The comparison's seed count is images per version, while the grid's
    # seed count is batches per version. Changing batch changes the grouping,
    # never the compared seeds or the number of saved images.
    base_seed = 21
    body = {
        "prompt": "x", "width": 256, "height": 256, "prompt_variants": VARIANTS,
        "seed": base_seed, "seeds": total_seeds // batch, "num_images": batch,
    }
    response = client.post("/api/grid/preview", json=body)
    assert response.status_code == 200, response.text
    preview = response.json()
    assert (preview["cells"], preview["images"], preview["duplicates"]) == (
        len(VARIANTS) * total_seeds // batch, len(VARIANTS) * total_seeds, 0,
    )

    job_id, events = run_grid(client, **body)
    done = events[-1]
    assert done["status"] == "done"
    assert done["image_count"] == preview["images"]
    assert [(seed, request.prompt, request.num_images)
            for seed, request in zip(fake.seeds, fake.requests)] == [
        (seed, prompt, batch)
        for seed in range(base_seed, base_seed + total_seeds, batch)
        for prompt in TEXTS
    ]

    # Read the persisted gallery too: each row of the comparison must contain
    # the same complete seed sequence, with its version and prompt intact.
    gallery = client.get(f"/api/gallery?search=job:{job_id}").json()
    assert gallery["total"] == len(VARIANTS) * total_seeds
    for variant in VARIANTS:
        images = [record for record in gallery["items"]
                  if record["variables"] == {"variant": variant["label"]}]
        assert sorted(record["seed"] for record in images) == list(
            range(base_seed, base_seed + total_seeds)
        )
        assert all(record["prompt"] == variant["prompt"] for record in images)


def test_the_preview_shows_each_variant_under_its_label(client):
    body = {
        "prompt": "x", "width": 256, "height": 256, "prompt_variants": VARIANTS,
        "steps_axis": "6,8", "seeds": 4,
    }
    preview = client.post("/api/grid/preview", json=body).json()
    assert preview["values"]["prompt"] == TEXTS
    assert preview["values"]["variant"] == LABELS
    assert (preview["cells"], preview["images"], preview["duplicates"]) == (24, 24, 0)
    assert preview["summary"] == "3 variants x 2 steps x 1 CFG x 1 mu x 4 seeds"


def test_a_variant_with_braces_pipes_and_newlines_runs_as_written(client, fake):
    xml = (
        "<character_1>\n<n>{name}</n>\n<appearance>red hair | green eyes</appearance>\n"
        "</character_1>\n<general_tags>\n<style>anime</style>\n</general_tags>"
    )
    given = [{"label": "xml", "prompt": xml}, {"label": "tags", "prompt": "1girl, red hair, anime"}]
    # The base prompt's own placeholder is not what runs, so it is not refused.
    body = {"prompt": "a {size} dog", "prompt_variants": given, "seed": 3}
    preview = client.post("/api/grid/preview", json={**body, "width": 256, "height": 256})
    assert preview.status_code == 200, preview.text
    assert preview.json()["values"]["prompt"] == [xml, "1girl, red hair, anime"]
    _, events = run_grid(client, **body)
    assert [request.prompt for request in fake.requests] == [xml, "1girl, red hair, anime"]
    assert events[-1]["images"][0]["prompt"] == xml


def test_variants_that_do_not_line_up_are_a_422(client, fake):
    body = {"prompt": "a {size} dog", "width": 256, "height": 256, "prompt_variants": VARIANTS}
    both = client.post("/api/grid", json={**body, "prompt_axis": "size = big | small"})
    assert both.status_code == 422
    assert "variables" in str(both.json()["detail"])
    relabelled = [VARIANTS[0], {**VARIANTS[1], "label": " rewrite "}]
    twice = client.post("/api/grid/preview", json={**body, "prompt_variants": relabelled})
    assert twice.status_code == 422
    assert "label" in str(twice.json()["detail"])
    blank = [VARIANTS[0], {**VARIANTS[1], "label": "   "}]
    nameless = client.post("/api/grid/preview", json={**body, "prompt_variants": blank})
    assert nameless.status_code == 422
    assert "needs a label" in nameless.json()["detail"]
    # One variant is not a comparison, and nine are more than one grid takes;
    # eight, the most, are all run.
    alone = client.post("/api/grid/preview", json={**body, "prompt_variants": VARIANTS[:1]})
    assert alone.status_code == 422
    nine = [{"label": f"v{n}", "prompt": f"a dog, take {n}"} for n in range(9)]
    too_many = client.post("/api/grid/preview", json={**body, "prompt_variants": nine})
    assert too_many.status_code == 422
    assert [(e["type"], e["loc"][-1]) for e in too_many.json()["detail"]] == [
        ("too_long", "prompt_variants")
    ]
    eight = client.post("/api/grid/preview", json={**body, "prompt_variants": nine[:8]})
    assert eight.status_code == 200, eight.text
    assert eight.json()["values"]["variant"] == [f"v{n}" for n in range(8)]
    # A negative-prompt placeholder has nothing to fill it beside variants.
    unfilled = client.post("/api/grid", json={**body, "negative_prompt": "{avoid}"})
    assert unfilled.status_code == 422
    assert "{avoid}" in unfilled.json()["detail"]
    assert not fake.requests


def test_a_variant_grid_resumes_on_its_prompts(client, fake):
    body = {
        "prompt": "x", "width": 256, "height": 256, "prompt_variants": VARIANTS,
        "seeds": 2, "seed": 50,
    }
    first = client.post("/api/grid", json=body).json()
    images = stream_job(client, first["job_id"])[-1]["images"]
    # The night died after the second seed's first variant.
    for record in images[4:]:
        client.delete(f"/api/images/{record['id']}")
    preview = client.post("/api/grid/preview", json={**body, "resume": True}).json()
    assert (preview["cells"], preview["skipped"]) == (2, 4)
    fake.requests.clear()
    fake.seeds.clear()
    second = client.post("/api/grid", json={**body, "resume": True}).json()
    stream_job(client, second["job_id"])
    assert [request.prompt for request in fake.requests] == TEXTS[1:]
    assert fake.seeds == [51, 51]
    # One run under two jobs, whole again.
    assert second["run_id"] == first["run_id"]
    page = client.get(f"/api/gallery?search=run:{first['run_id']}").json()
    assert page["total"] == 6
    # Keyed on the text, not the label: the same prompts renamed are done too.
    renamed = [{**variant, "label": f"v{n}"} for n, variant in enumerate(VARIANTS)]
    again = client.post("/api/grid", json={**body, "prompt_variants": renamed, "resume": True})
    assert again.status_code == 409


def test_reversing_a_comparison_can_be_read_by_submission(client, fake, store):
    runs = []
    for original, yours in (("a red fox", "a blue fox"), ("a blue fox", "a red fox")):
        body = {
            "prompt": original, "width": 256, "height": 256, "seed": 42, "seeds": 2,
            "prompt_variants": [
                {"label": "original", "prompt": original}, {"label": "yours", "prompt": yours},
            ],
        }
        response = client.post("/api/grid", json=body)
        assert response.status_code == 200, response.text
        accepted = response.json()
        records = stream_job(client, accepted["job_id"])[-1]["images"]
        for record in records:
            if record["prompt"] == "a red fox":
                client.post(f"/api/images/{record['id']}/favorite?value=true")
        runs.append((accepted["run_id"], accepted["job_id"], original))
    # Same rendered cells, same resume/run identity; the submission owns its
    # version labels. Reading that job keeps each sheet and rate meaningful.
    assert runs[0][0] == runs[1][0]
    for _run, job, original in runs:
        page = client.get("/api/gallery", params={"search": f"job:{job}"}).json()
        assert page["total"] == 4
        assert {r["prompt"] for r in page["items"] if r["variables"]["variant"] == "original"} == {original}
        rows = store.keep_rates(f"job:{job}", by=["variant"])
        assert len(rows) == 2 and all(r["made"] == 2 for r in rows)
        assert rows[0]["variant"] == ("original" if original == "a red fox" else "yours")
        assert [r["rate"] for r in rows] == [1.0, 0.0]


def test_legacy_mixed_comparison_stats_require_an_unambiguous_scope(
    client, fake, store, monkeypatch, capsys
):
    from backend import stats

    monkeypatch.setattr(stats, "settings", store.settings)
    jobs = []
    for prompts in (("a red fox", "a blue fox"), ("a blue fox", "a red fox")):
        job, events = run_grid(client, seed=42, seeds=2, prompt_variants=[
            {"label": label, "prompt": prompt}
            for label, prompt in zip(("original", "yours"), prompts)
        ])
        jobs.append(job)
        for record in events[-1]["images"]:
            if record["prompt"] == "a red fox":
                client.post(f"/api/images/{record['id']}/favorite?value=true")
    # Simulate the rows an older build wrote, without touching any real gallery.
    with store._session() as conn:
        conn.execute("UPDATE images SET run_id = 'legacy-comparison'")
    assert stats.main(["--run", "legacy-comparison", "--by", "variant", "--min", "1"]) == 2
    output = capsys.readouterr()
    assert "more than one prompt" in output.err and not output.out
    assert "--by variant,prompt" in output.err and "--job" in output.err
    assert stats.main(["--run", "legacy-comparison", "--by", "variant,prompt", "--min", "1"]) == 0
    output = capsys.readouterr().out
    assert output.count("100%") == 2 and output.count("a red fox") == 2
    assert output.count("a blue fox") == 2
    assert stats.main(["--job", jobs[0], "--by", "variant", "--min", "1"]) == 0
    assert "100%" in capsys.readouterr().out


def test_variant_stats_allow_other_variables_and_respect_grouping(client, fake, store):
    # `variant` is also a legal ordinary variable. Pooling its colours is
    # intentional; only two meanings of the complete assignment conflict.
    jobs = []
    for animal, seed, axis in (
        ("cat", 42, "variant = small | large\ncolour = red | blue"),
        ("dog", 43, "colour = red | blue\nvariant = small | large"),
    ):
        job, _ = run_grid(client, prompt=f"a {{variant}} {{colour}} {animal}",
                          prompt_axis=axis, seed=seed)
        jobs.append(job)
        rows = store.keep_rates(f"job:{job}", by=["variant"])
        assert {r["variant"] for r in rows} == {"small", "large"}
        assert all(r["made"] == 2 for r in rows)
    # Different JSON key order must not disguise the same assignment.
    with pytest.raises(ValueError, match="same variable assignment"):
        store.keep_rates(None, by=["variant"])
    # An explicit grouping can already distinguish the two meanings.
    rows = store.keep_rates(None, by=["variant", "seed"])
    assert len(rows) == 4 and all(r["made"] == 2 for r in rows)
    # Groups not emitted by the requested floor do not block the result.
    assert store.keep_rates(None, by=["variant"], min_made=10) == []


def test_the_stats_give_a_comparison_its_starred_rate_per_version(
    client, fake, store, monkeypatch, capsys
):
    # Reorders share their opening words, which is all a table cell holds, so
    # `--by prompt` printed one identical row per version; the label is what
    # tells them apart. The rewrite is written first, so written order cannot
    # pass for the labels' alphabetical one.
    from backend import stats

    monkeypatch.setattr(stats, "settings", store.settings)
    opening = "A woman in a red dress standing on a pier at dusk"
    versions = [
        {"label": "rewrite",
         "prompt": "A woman in a red dress\r\n\tstanding on a pier as gulls wheel."},
        {"label": "original", "prompt": f"{opening}, gulls overhead, film grain"},
        {"label": "reordered", "prompt": f"{opening}, film grain, gulls overhead"},
    ]
    _, events = run_grid(client, prompt_variants=versions, seeds=4, seed=30)
    images = events[-1]["images"]

    def table(by):
        assert stats.main(["--run", images[0]["run_id"], "--by", by, "--min", "4"]) == 0
        out = capsys.readouterr().out
        # A tab is no column break; the table pads with spaces.
        assert "\t" not in out
        # Past the scope, the grouping, a blank line, the header and its rule.
        return [line.split() for line in out.splitlines()[5:]]

    # Nothing starred yet: every rate ties, and the versions read as written.
    assert [row[0] for row in table("variant")] == ["rewrite", "original", "reordered"]
    wins = {"reordered": 4, "rewrite": 2, "original": 1}
    for record in images:
        if record["seed"] - 30 < wins[record["variables"]["variant"]]:
            client.post(f"/api/images/{record['id']}/favorite?value=true")
    assert [(row[0], row[-1]) for row in table("variant")] == [
        ("reordered", "100%"), ("rewrite", "50%"), ("original", "25%"),
    ]
    # By prompt, each version is still one row: the rewrite's line break and
    # indent are folded rather than breaking the table.
    rows = table("prompt")
    assert len(rows) == 3 and [row[-1] for row in rows] == ["100%", "50%", "25%"]


def test_resume_skips_only_the_prompt_it_already_rendered(client, store):
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(
            prompt="big dog on grass", width=256, height=256, steps=8, guidance_scale=0.0, mu=None
        ),
        seed=5,
        duration_ms=100,
        model_id=settings.model_id,
    )
    preview = client.post(
        "/api/grid/preview",
        json={
            "prompt": "{size} dog on grass",
            "width": 256,
            "height": 256,
            "prompt_axis": "size = big | small",
            "steps_axis": "8",
            "cfg_axis": "0",
            "mu_axis": "none",
            "seed": 5,
            "resume": True,
        },
    ).json()
    # The row matches one prompt of two; the other has never been rendered and
    # the identical settings under it must not be skipped with it.
    assert (preview["skipped"], preview["cells"]) == (1, 1)


def test_status_reports_free_disk_and_the_preview_prices_the_pngs(client, store):
    status = client.get("/api/status").json()
    assert isinstance(status["disk_free_mb"], int) and status["disk_free_mb"] > 0
    body = {"prompt": "x", "width": 256, "height": 256, "steps_axis": "8", "seeds": 4}
    # No files at this size yet: no estimate rather than a guess.
    assert client.post("/api/grid/preview", json=body).json()["bytes"] is None
    for seed in (1, 2, 3):
        store.save_image(
            Image.new("RGB", (256, 256)),
            GenerateRequest(prompt="x", width=256, height=256),
            seed=seed, duration_ms=1, model_id=settings.model_id,
        )
    sizes = sorted(p.stat().st_size for p in store.settings.outputs_dir.glob("*.png"))
    preview = client.post("/api/grid/preview", json=body).json()
    assert preview["bytes"] == sizes[1] * 4  # the median, times the image count


def test_the_preview_estimates_from_what_the_gallery_measured(client, store):
    # Half a second per step, measured at this size on this checkpoint. The
    # fake backend renders instantly, so the row has to be a real one to
    # estimate from - and a row from another checkpoint must not count.
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="x", width=256, height=256, steps=8),
        seed=2,
        duration_ms=40000,
        model_id="some/other-checkpoint",
    )
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="x", width=256, height=256, steps=8),
        seed=1,
        duration_ms=4000,
        model_id=settings.model_id,
    )
    preview = client.post(
        "/api/grid/preview",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "8", "seeds": 4},
    ).json()
    # 0.5s x 8 steps x 4 cells. Per step, not per image: a grid whose cells
    # disagree about the step count cannot be costed from an image average.
    assert preview["seconds"] == pytest.approx(16.0)
    assert preview["budget_seeds"] is None
    # A 6 s budget over 4 s seeds: the second starts at 4 s, before the
    # deadline, and finishes past it - two seeds, where 6 / 4 said one.
    body = {"prompt": "x", "width": 256, "height": 256, "steps_axis": "8", "seeds": 4, "budget_minutes": 0.1}
    assert client.post("/api/grid/preview", json=body).json()["budget_seeds"] == 2
    # Under Resume the seeds left are not alike; the page says nothing there.
    resumed = client.post("/api/grid/preview", json={**body, "seed": 1, "resume": True}).json()
    assert resumed["budget_seeds"] is None


def test_a_bad_axis_is_refused_before_anything_runs(client, fake):
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "eight"},
    )
    assert response.status_code == 422
    assert "not a number" in response.json()["detail"]
    assert not fake.requests


def test_a_grid_out_of_the_schemas_range_names_the_field(client):
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "90-120x10"},
    )
    assert response.status_code == 422
    # Under the box's name, quoting the value - not pydantic's "steps: Input
    # should be less than or equal to 100".
    assert response.json()["detail"] == "Steps: 110 is above 100, the most it takes"


def test_runs_are_not_a_grid_axis(client):
    # They would multiply the night behind the count the preview promised.
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "runs": 4},
    )
    assert response.status_code == 422


def test_every_image_of_a_run_carries_its_job(client, fake):
    job_id, events = run_grid(client, steps_axis="6,8", seed=5)
    records = events[-1]["images"]
    assert {r["job_id"] for r in records} == {job_id}
    # And that id is what the gallery filter selects on, so a night's work is
    # one search away rather than however far down the grid it has sunk.
    page = client.get(f"/api/gallery?search=job:{job_id}").json()
    assert page["total"] == len(records)


def test_a_resumed_grid_skips_what_the_gallery_already_holds(client, fake):
    body = {
        "prompt": "a lighthouse in fog",
        "width": 256,
        "height": 256,
        "steps_axis": "6,8,10",
        "seed": 11,
    }
    stream_job(client, client.post("/api/grid", json={**body, "steps_axis": "6"}).json()["job_id"])
    assert len(fake.requests) == 1

    preview = client.post("/api/grid/preview", json={**body, "resume": True}).json()
    assert (preview["cells"], preview["skipped"]) == (2, 1)

    stream_job(client, client.post("/api/grid", json={**body, "resume": True}).json()["job_id"])
    # The 6-step cell was not rendered twice, and the order of what remains is
    # the order the grid would have reached it in.
    assert [r.steps for r in fake.requests] == [6, 8, 10]


def test_a_resume_pressed_while_the_run_is_queued_plans_nothing_twice(client, fake):
    body = {
        "prompt": "a lighthouse in fog",
        "width": 256,
        "height": 256,
        "steps_axis": "6,8,10",
        "seed": 11,
        "resume": True,
    }
    stream_job(client, client.post("/api/grid", json={**body, "steps_axis": "6"}).json()["job_id"])
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    try:
        # The rest of the night, still running when Resume is pressed again.
        first = client.post("/api/grid", json=body).json()["job_id"]
        # Queued or running, the live job's cells are not done: the gallery
        # holds one.
        preview = client.post("/api/grid/preview", json=body).json()
        assert (preview["cells"], preview["skipped"], preview["queued"]) == (0, 1, 2)
        refused = client.post("/api/grid", json=body)
        assert refused.status_code == 409
        assert refused.json()["detail"] == "every cell is already rendered or queued"
        # Widened by a cell, only that cell is new.
        wider = {**body, "steps_axis": "6,8,10,12"}
        preview = client.post("/api/grid/preview", json=wider).json()
        assert (preview["cells"], preview["skipped"], preview["queued"]) == (1, 1, 2)
        second = client.post("/api/grid", json=wider).json()["job_id"]
        # Without Resume the gallery's count stays the gallery's.
        plain = client.post("/api/grid/preview", json={**wider, "resume": False}).json()
        assert (plain["in_gallery"], plain["queued"]) == (1, 0)
    finally:
        release.set()
    assert stream_job(client, first)[-1]["status"] == "done"
    assert stream_job(client, second)[-1]["status"] == "done"
    assert [r.steps for r in fake.requests] == [6, 8, 10, 12]
    # And once it has finished, the cells it wrote are the gallery's.
    preview = client.post("/api/grid/preview", json=wider).json()
    assert (preview["cells"], preview["skipped"], preview["queued"]) == (0, 4, 0)


def test_a_cell_its_job_finishes_during_the_gallery_scan_is_not_planned_again(client, fake, monkeypatch):
    # The queue is read before the gallery: a cell whose job saves it and
    # finishes while `completed_cells` scans is in the queue's snapshot even
    # though the scan, begun before its row committed, missed it.
    body = {"prompt": "a lighthouse in fog", "width": 256, "height": 256, "steps_axis": "6", "seed": 11}
    release = threading.Event()
    fake.after_generate = lambda: release.wait(5)
    first = client.post("/api/grid", json=body).json()["job_id"]
    wait_for(client, first, lambda job: job["status"] == "running")
    scan = storage.completed_cells

    def racing(*args, **kwargs):
        done = scan(*args, **kwargs)
        release.set()
        deadline = time.monotonic() + 5
        while main.jobs.get(first).status != "done" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert main.jobs.get(first).status == "done"
        return done

    monkeypatch.setattr(storage, "completed_cells", racing)
    try:
        preview = client.post("/api/grid/preview", json={**body, "resume": True}).json()
    finally:
        release.set()
    assert (preview["cells"], preview["skipped"], preview["queued"]) == (0, 0, 1)
    assert [r.steps for r in fake.requests] == [6]


def test_without_resume_the_preview_counts_what_the_gallery_holds(client, fake):
    body = {
        "prompt": "a lighthouse in fog",
        "width": 256,
        "height": 256,
        "steps_axis": "6,8,10",
        "seed": 11,
    }
    stream_job(client, client.post("/api/grid", json={**body, "steps_axis": "6"}).json()["job_id"])

    # Nothing is dropped - Run grid as it stands renders all three again - but
    # the cell already there is counted, which is what says Resume would not.
    preview = client.post("/api/grid/preview", json=body).json()
    assert (preview["cells"], preview["skipped"], preview["in_gallery"]) == (3, 0, 1)
    resumed = client.post("/api/grid/preview", json={**body, "resume": True}).json()
    assert (resumed["cells"], resumed["skipped"], resumed["in_gallery"]) == (2, 1, 0)
    # A random base matches nothing, so it is not looked up; a listed seed is
    # as fixed as an explicit base.
    drawn = client.post("/api/grid/preview", json={**body, "seed": None}).json()
    assert drawn["in_gallery"] == 0
    listed = client.post("/api/grid/preview", json={**body, "seed": None, "seed_axis": "11"}).json()
    assert listed["in_gallery"] == 1
    assert len(fake.requests) == 1


def test_resuming_a_finished_grid_refuses_rather_than_running_nothing(client, fake):
    body = {
        "prompt": "a lighthouse in fog",
        "width": 256,
        "height": 256,
        "steps_axis": "6,8",
        "seed": 3,
    }
    stream_job(client, client.post("/api/grid", json=body).json()["job_id"])
    response = client.post("/api/grid", json={**body, "resume": True})
    # An empty job would report "done" having drawn nothing, which reads as a
    # failure; 409 says the work is already there.
    assert response.status_code == 409
    assert len(fake.requests) == 2


def test_a_grid_publishes_the_seed_it_drew(client, fake):
    # A grid with a blank Seed box draws its own base and every image carries a
    # *derived* seed - so the base has to come back in the response, or resume
    # has nothing to key on.
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "seeds": 2},
    )
    accepted = response.json()
    assert isinstance(accepted["seed"], int)
    assert client.get(f"/api/jobs/{accepted['job_id']}").json()["seed"] == accepted["seed"]
    events = stream_job(client, accepted["job_id"])
    # The base is the *oldest* image's seed, not the newest's.
    assert events[-1]["images"][0]["seed"] == accepted["seed"]
    assert events[-1]["images"][-1]["seed"] == accepted["seed"] + 1
    # And resuming with it finds every cell done.
    resume = client.post(
        "/api/grid",
        json={
            "prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "seeds": 2,
            "seed": accepted["seed"], "resume": True,
        },
    )
    assert resume.status_code == 409


def test_a_plain_request_publishes_its_seed_too(client):
    accepted = client.post(
        "/api/generate", json={"prompt": "x", "width": 256, "height": 256, "seed": 77}
    ).json()
    assert accepted["seed"] == 77
    assert accepted["run_id"]


def test_a_resumed_grid_is_one_run_under_two_jobs(client, fake):
    body = {"prompt": "a lighthouse in fog", "width": 256, "height": 256, "steps_axis": "6,8,10", "seed": 11}
    first = client.post("/api/grid", json=body).json()
    images = stream_job(client, first["job_id"])[-1]["images"]
    # Pretend the night died after one cell: lose the other two, then resume.
    for record in images[1:]:
        client.delete(f"/api/images/{record['id']}")
    second = client.post("/api/grid", json={**body, "resume": True}).json()
    stream_job(client, second["job_id"])
    assert second["job_id"] != first["job_id"]
    # The run id is the whole grid's, so both halves carry it - and the sheet's
    # `run:` filter returns the complete matrix where `job:` would return a part.
    assert second["run_id"] == first["run_id"]
    page = client.get(f"/api/gallery?search=run:{second['run_id']}").json()
    assert page["total"] == 3
    assert {item["run_id"] for item in page["items"]} == {first["run_id"]}
    assert len({item["job_id"] for item in page["items"]}) == 2


def test_the_record_carries_what_ran_not_what_was_asked(client, fake):
    # Turbo at CFG 0 never sees a negative prompt; the row and the resume key
    # used to carry it anyway, so clearing the greyed-out box before a resume
    # matched nothing and re-rendered the whole grid.
    fake.cfg_threshold = 0.0
    body = {
        "prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "cfg_axis": "0,1",
        "seed": 5, "negative_prompt": "blurry",
    }
    images = stream_job(client, client.post("/api/grid", json=body).json()["job_id"])[-1]["images"]
    by_cfg = {record["guidance_scale"]: record["negative_prompt"] for record in images}
    assert by_cfg == {0.0: None, 1.0: "blurry"}
    assert {record["quantization"] for record in images} == {settings.quantization}
    # Resubmitted with the box cleared, the CFG 0 cells count as done and only
    # the guided cells - whose negative prompt did change - are outstanding.
    preview = client.post(
        "/api/grid/preview", json={**body, "negative_prompt": None, "resume": True}
    ).json()
    assert (preview["skipped"], preview["cells"]) == (2, 2)
    # And resubmitted as it was, everything is.
    assert client.post("/api/grid", json={**body, "resume": True}).status_code == 409


def test_resume_without_a_seed_is_refused(client, fake):
    # It would draw a fresh random base, match none of the night it is meant to
    # continue, and render the whole grid again while reporting a resume.
    response = client.post(
        "/api/grid",
        json={"prompt": "x", "width": 256, "height": 256, "steps_axis": "6,8", "resume": True},
    )
    assert response.status_code == 422
    # A model validator lands in FastAPI's own structured 422 body rather than
    # in one of ours, which the UI flattens back into a sentence.
    assert "seed" in str(response.json()["detail"])
    assert not fake.requests


def test_a_selection_can_be_rendered_again_at_another_size(client, fake, store):
    # Star two cells of a 256px grid and ask for them at 512: two cells, the
    # same prompt, steps, CFG, mu and seed, only the size changed, oldest
    # first - and a second ask finds them done.
    _, events = run_grid(client, steps_axis="6,8,10", cfg_axis="0,1", seed=5, mu=1.15)
    images = sorted(events[-1]["images"], key=lambda r: r["id"])
    for record in (images[1], images[4]):
        client.post(f"/api/images/{record['id']}/favorite?value=true")
    fake.requests.clear()

    accepted = client.post("/api/rerender", json={"width": 512, "height": 512}).json()
    done = stream_job(client, accepted["job_id"])[-1]
    assert done["status"] == "done" and len(done["images"]) == 2
    asked = [(r.steps, r.guidance_scale, r.mu, r.seed, r.width) for r in fake.requests]
    assert asked == [
        (images[1]["steps"], images[1]["guidance_scale"], 1.15, 5, 512),
        (images[4]["steps"], images[4]["guidance_scale"], 1.15, 5, 512),
    ]
    assert accepted["seed"] == 5
    # A run of its own, distinct from the grid it came from, and resumable.
    assert accepted["run_id"] != images[0]["run_id"]
    assert client.post("/api/rerender", json={"width": 512, "height": 512}).status_code == 409
    # The selection is the gallery's own filter, taken server-side: the two
    # 8-step cells, starred or not.
    narrowed = client.post(
        "/api/rerender", json={"width": 768, "height": 768, "favorites": False, "search": "steps:8"}
    )
    assert narrowed.status_code == 200
    assert len(stream_job(client, narrowed.json()["job_id"])[-1]["images"]) == 2
    assert client.post(
        "/api/rerender", json={"width": 512, "height": 512, "search": "seed:999"}
    ).status_code == 404
    # The pixel budget applies, as it does to both other producers.
    assert client.post("/api/rerender", json={"width": 2736, "height": 2736}).status_code == 422


def test_a_rerender_keeps_each_cells_variables(client, fake, store):
    # A comparison's version names live in its variables alone, and a `{var}`
    # grid's assignment is what its sheet lays out; a re-render that dropped
    # them left the new run's sheet a flat Prompt axis in alphabetical order.
    run_grid(client, prompt_variants=VARIANTS, seed=3)
    run_grid(client, prompt="a {size} dog", prompt_axis="size = big | none", seed=4)
    accepted = client.post("/api/rerender", json={"width": 512, "height": 512, "favorites": False})
    assert accepted.status_code == 200, accepted.text
    images = stream_job(client, accepted.json()["job_id"])[-1]["images"]
    assert [(r["prompt"], r["variables"], r["width"]) for r in images] == [
        *((text, {"variant": label}, 512) for text, label in zip(TEXTS, LABELS)),
        ("a big dog", {"size": "big"}, 512),
        # `none` is an assignment too, and survives as one.
        ("a dog", {"size": ""}, 512),
    ]
    # The PNG's own record carries it as well, for a reindex to bring back.
    with Image.open(store.settings.outputs_dir / images[0]["filename"]) as png:
        assert json.loads(png.info["local-diffusion"])["variables"] == {"variant": LABELS[0]}


def test_a_rerender_refuses_another_checkpoints_records(client, fake, store):
    # Steps, CFG and mu are checkpoint-specific; rendering FLUX's 28 / 3.5 on
    # Turbo would record a broken image under Turbo. The gallery's scope keeps
    # the other checkpoint's row out of the selection, so the refusal is only
    # reachable once All models has let it in.
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="x", width=256, height=256),
        seed=1, duration_ms=1, model_id="somebody/else",
    )
    selection = {"width": 512, "height": 512, "favorites": False}
    assert client.post("/api/rerender", json=selection).status_code == 404
    response = client.post("/api/rerender", json={**selection, "all_models": True})
    assert response.status_code == 422
    assert response.json()["detail"] == (
        f"1 of the selected images was made on somebody/else; the server is running "
        f"{settings.model_id}, whose steps, CFG and mu differ. Untick All models, or add "
        f"model:{settings.model_id} to the search."
    )
    # With one of this checkpoint's own beside it, the scoped selection is
    # that one alone, and the other row never reaches the pipeline.
    store.save_image(
        Image.new("RGB", (256, 256)),
        GenerateRequest(prompt="mine", width=256, height=256),
        seed=2, duration_ms=1, model_id=settings.model_id,
    )
    accepted = client.post("/api/rerender", json=selection)
    assert accepted.status_code == 200, accepted.text
    done = stream_job(client, accepted.json()["job_id"])[-1]
    assert [r.prompt for r in fake.requests] == ["mine"] and len(done["images"]) == 1


def test_a_rerender_refusal_offers_only_a_search_that_leaves_the_others_out(
    client, fake, store, monkeypatch
):
    # Anima's refusal said "add model:Anima to the search", and `model:` is a
    # substring: Animagine stayed in, and following the advice gave the same
    # refusal. The whole id is offered, and only where it leaves out every
    # other checkpoint selected - else unticking All models alone, the exact
    # scope.
    monkeypatch.setattr(
        main, "settings", dataclasses.replace(main.settings, model_id="circlestone-labs/Anima")
    )

    def plant(model_id, seed):
        store.save_image(
            Image.new("RGB", (256, 256)),
            GenerateRequest(prompt="a smile", width=256, height=256),
            seed=seed, duration_ms=1, model_id=model_id,
        )

    plant("cagliostrolab/animagine-xl-4.0", 1)
    plant("circlestone-labs/Anima", 2)
    plant("circlestone-labs/Anima", 3)
    selection = {"width": 512, "height": 512, "favorites": False, "all_models": True}
    response = client.post("/api/rerender", json=selection)
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "1 of the selected images was made on cagliostrolab/animagine-xl-4.0; the server is "
        "running circlestone-labs/Anima, whose steps, CFG and mu differ. Untick All models, "
        "or add model:circlestone-labs/Anima to the search."
    )
    # Following the advice works.
    narrowed = {**selection, "search": "model:circlestone-labs/Anima"}
    followed = client.post("/api/rerender", json=narrowed)
    assert followed.status_code == 200, followed.text
    assert len(stream_job(client, followed.json()["job_id"])[-1]["images"]) == 2

    # An id that holds the running one whole: no `model:` would leave it out.
    plant("circlestone-labs/Anima-Preview", 4)
    plant("circlestone-labs/Anima-Preview", 5)
    response = client.post("/api/rerender", json=narrowed)
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "2 of the selected images were made on circlestone-labs/Anima-Preview; the server is "
        "running circlestone-labs/Anima, whose steps, CFG and mu differ. Untick All models."
    )


def test_a_rerender_takes_the_picked_cards_whatever_the_filter(client, fake, store):
    # Two keepers picked to render at full size: the filter (Starred, which
    # matches none of them) used to be the only scope, so it was every match
    # or nothing. With `ids`, exactly the picks, oldest first however they
    # were picked; one binned and one deleted since are left out and named,
    # as Swap tags names them, not a refusal of the rest.
    _, events = run_grid(client, steps_axis="6,8,10", cfg_axis="0,1", seed=5)
    images = sorted(events[-1]["images"], key=lambda r: r["id"])
    keep, also, binned, deleted = (images[i]["id"] for i in (4, 1, 2, 3))
    assert client.post("/api/images/bin", json={"ids": [binned]}).status_code == 200
    assert client.delete(f"/api/images/{deleted}").status_code == 200
    fake.requests.clear()

    picked = {"ids": [keep, binned, also, deleted], "width": 512, "height": 512}
    accepted = client.post("/api/rerender", json=picked)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["deleted"] == [
        {"id": binned, "detail": "in the recycle bin"},
        {"id": deleted, "detail": "no longer in the gallery"},
    ]
    done = stream_job(client, accepted.json()["job_id"])[-1]
    assert done["status"] == "done" and len(done["images"]) == 2
    assert [(r.steps, r.guidance_scale, r.seed, r.width) for r in fake.requests] == [
        (images[1]["steps"], images[1]["guidance_scale"], 5, 512),
        (images[4]["steps"], images[4]["guidance_scale"], 5, 512),
    ]
    # Done is done, as for a filter.
    assert client.post("/api/rerender", json=picked).status_code == 409
    # With nothing left of the picks, nothing to queue, each said.
    response = client.post("/api/rerender", json={**picked, "ids": [binned, deleted]})
    assert response.status_code == 404
    assert response.json()["detail"] == (
        f"none of the selected images is left: #{binned} is in the recycle bin; "
        f"#{deleted} is no longer in the gallery"
    )
    # A filter's answer names none.
    star = client.post(f"/api/images/{keep}/favorite?value=true")
    assert star.status_code == 200
    filtered = client.post("/api/rerender", json={"width": 768, "height": 768})
    assert filtered.status_code == 200 and filtered.json()["deleted"] == []
    # As every other action's `ids`: none, or an id below 1, is refused.
    assert client.post("/api/rerender", json={**picked, "ids": []}).status_code == 422
    assert client.post("/api/rerender", json={**picked, "ids": [0]}).status_code == 422


def test_a_rerender_of_picks_refuses_another_checkpoints_and_says_which(client, fake, store):
    # Picked cards are no filter: All models does not scope them, so another
    # checkpoint's pick is refused with or without it - and the way out is
    # unpicking it, by name, not a search or a tick that cannot reach it.
    def plant(model_id, seed):
        return store.save_image(
            Image.new("RGB", (256, 256)),
            GenerateRequest(prompt="x", width=256, height=256),
            seed=seed, duration_ms=1, model_id=model_id,
        ).id

    mine = plant(settings.model_id, 1)
    theirs = [plant("somebody/else", seed) for seed in range(2, 10)]
    response = client.post("/api/rerender", json={"ids": [mine, theirs[0]], "width": 512, "height": 512})
    assert response.status_code == 422
    assert response.json()["detail"] == (
        f"1 of the selected images was made on somebody/else; the server is running "
        f"{settings.model_id}, whose steps, CFG and mu differ. Unpick #{theirs[0]}."
    )
    # Named up to six, then counted.
    response = client.post("/api/rerender", json={"ids": [mine, *theirs], "width": 512, "height": 512})
    names = ", ".join(f"#{image_id}" for image_id in theirs[:6])
    assert response.json()["detail"].endswith(f"differ. Unpick {names} and 2 more.")
    assert not fake.requests


def test_finished_jobs_are_listed_on_request_with_their_outcome(client, fake):
    # The morning after: /api/jobs alone is empty once the night is over, and
    # nothing said whether the grid finished, stopped or died - or how much it
    # made. Newest finished first, counters and error but no records.
    fake.fail_on = {2}
    job_id, events = run_grid(client, steps_axis="6,8,10", seed=1)
    assert client.get("/api/jobs").json() == []
    listed = client.get("/api/jobs?include_finished=true").json()
    assert [job["job_id"] for job in listed] == [job_id]
    job = listed[0]
    assert (job["status"], job["failures"], job["image_count"]) == ("done", 1, 2)
    assert (job["run"], job["total_runs"]) == (3, 3)
    assert job["finished_at"] >= job["started_at"] >= job["created_at"]
    assert "blew up" in job["error"] and "images" not in job
    # And the gallery's own account of the run, which outlives the process.
    runs = client.get("/api/runs").json()
    assert len(runs) == 1
    assert (runs[0]["run_id"], runs[0]["images"], runs[0]["seed"]) == (job["run_id"], 2, 1)


def test_a_run_says_what_varied_across_it(client, fake):
    # The Runs picker names a run by its axes: every line used to read
    # "6 img · A weathered...", a steps x CFG grid and a batch alike.
    run_grid(client, steps_axis="6,8,10", cfg_axis="1,2", seed=1)
    grid = client.get("/api/runs").json()[0]
    assert grid["axes"] == {"steps": 3, "cfg": 2}
    assert (grid["refine"], grid["strengths"]) == (None, None)
    run_grid(client, prompt_variants=VARIANTS, seeds=2, seed=21)
    compared = client.get("/api/runs").json()[0]
    assert compared["axes"] == {"seed": 2, "variant": len(VARIANTS)}


def test_the_gallery_pages_by_id_and_knows_when_it_is_done(client, store):
    for seed in range(5):
        store.save_image(
            Image.new("RGB", (16, 16)),
            GenerateRequest(prompt="x", width=256, height=256),
            seed=seed, duration_ms=1, model_id=settings.model_id,
        )
    first = client.get("/api/gallery?limit=2").json()
    assert [r["seed"] for r in first["items"]] == [4, 3]
    # Exact, from a row fetched past the page - not `offset < total`, which a
    # growing total keeps true after the last old row has been shown.
    assert first["has_more"] is True
    second = client.get(f"/api/gallery?limit=2&before={first['items'][-1]['id']}").json()
    assert [r["seed"] for r in second["items"]] == [2, 1]
    third = client.get(f"/api/gallery?limit=2&before={second['items'][-1]['id']}").json()
    assert [r["seed"] for r in third["items"]] == [0]
    assert third["has_more"] is False
    assert third["total"] == 5


def test_a_bad_gallery_filter_is_a_422(client):
    response = client.get("/api/gallery?search=seed:tuesday")
    assert response.status_code == 422
    assert "not a number" in response.json()["detail"]


def test_a_side_off_the_16_grid_is_refused_with_its_neighbours(client):
    # The size line's words: the value and the two sizes either side, which
    # the page reads as "width: 1000 is not…" (readableDetail).
    response = client.post("/api/generate", json={"prompt": "x", "width": 1000, "height": 1024})
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert (error["loc"][-1], error["msg"]) == ("width", "1000 is not a multiple of 16: 992 or 1008")


def test_the_pixel_budget_is_enforced(client):
    response = client.post(
        "/api/generate", json={"prompt": "x", "width": 2816, "height": 2816}
    )
    assert response.status_code == 422


def test_the_batch_budget_says_what_each_press_can_do_instead(client):
    # Generate's answer points at the panel's Repeat; a grid refuses a
    # repeat, so its preview and Run grid say lower Batch.
    big = {"prompt": "x", "width": 1440, "height": 1440, "num_images": 12}
    detail = client.post("/api/generate", json=big).json()["detail"]
    assert "use Repeat instead of Batch" in detail
    for path in ("/api/grid/preview", "/api/grid"):
        response = client.post(path, json=big)
        assert response.status_code == 422
        assert response.json()["detail"].endswith("lower Batch: a grid runs its seeds one batch at a time")


def test_a_resume_counts_a_binned_cell_as_made_until_the_bin_is_emptied(client, fake):
    # A binned cell is a render rejected, not one the night never reached:
    # Resume leaves it, and only an emptied bin makes it a hole again.
    body = {"prompt": "a fox", "width": 256, "height": 256, "steps_axis": "6,8", "seed": 9}
    images = stream_job(client, client.post("/api/grid", json=body).json()["job_id"])[-1]["images"]
    client.post("/api/images/bin", json={"ids": [images[0]["id"]]})
    preview = client.post("/api/grid/preview", json={**body, "resume": True}).json()
    assert (preview["cells"], preview["skipped"]) == (0, 2)
    assert client.post("/api/grid", json={**body, "resume": True}).status_code == 409

    client.delete("/api/gallery/bin?expected=1")
    preview = client.post("/api/grid/preview", json={**body, "resume": True}).json()
    assert (preview["cells"], preview["skipped"]) == (1, 1)
