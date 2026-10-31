"""Prompt variants over HTTP: the analyser's comparison run as one grid on
shared seeds, its preview, resume and the starred rate per version."""
from __future__ import annotations

import pytest
from PIL import Image

from backend.config import settings
from backend.schemas import GenerateRequest
from tests.api_helpers import LABELS, TEXTS, VARIANTS, run_grid, stream_job


# ------------------------------------------------------------ prompt variants


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
