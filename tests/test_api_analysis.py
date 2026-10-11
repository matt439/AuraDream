"""Cold and concurrent prompt checks, measurements, and response contracts."""
from __future__ import annotations

import threading

import pytest

from backend import main, prompt_check, prompt_encoding, prompt_rules, tag_lookup
from backend.model_defaults import guide_for

from tests.api_analysis_helpers import (
    ANIMAGINE,
    FLUX,
    Vocabulary,
    analyse,
    claim,
    on_the_loop,
    rules,
)
from tests.api_helpers import stream_job
from tests.fakes import fake_tokenizer


# ------------------------------------------------------------------ analyse


def test_the_check_answers_cold_without_the_image_model(client, fake, monkeypatch):
    claim(monkeypatch, fake, FLUX)
    fake.loaded = False
    body = analyse(client, prompt="A lighthouse on a cliff at dusk, oil painting")
    assert fake.loads == 0 and not fake.loaded
    assert client.get("/api/status").json()["state"] == "unloaded"
    assert body["model_id"] == FLUX and body["style"] == "prose" and body["words"] == 9
    assert [(e["name"], e["budget"], e["cut"]) for e in body["encoders"]] == [
        ("CLIP-L", 75, None), ("T5-XXL", 511, None),
    ]
    assert body["encoders"][0]["source"] == f"{FLUX}/tokenizer"
    assert body["encoders_detail"] is None
    assert body["order"]["steps"] == ["subject", "setting", "style", "camera", "lighting"]


def test_the_check_answers_beside_a_running_job_off_the_loop(client, fake, monkeypatch):
    # Never the gate: a Check during a night's grid answers while the job
    # still holds the gate - not refused, not queued behind it. Off the loop:
    # a first use fetches tokenizers, which on the loop would freeze every
    # stream, Cancel and Pause for the length of the fetch.
    claim(monkeypatch, fake, FLUX)
    where: dict[str, list[bool]] = {"analyse": [], "tokenizer": []}
    real_analyse = prompt_check.analyse

    def watched_analyse(*args):
        where["analyse"].append(on_the_loop())
        return real_analyse(*args)

    def watched_tokenizer(source):
        where["tokenizer"].append(on_the_loop())
        return fake_tokenizer(source)

    monkeypatch.setattr(prompt_check, "analyse", watched_analyse)
    monkeypatch.setattr(prompt_encoding, "_load_tokenizer", watched_tokenizer)
    running, release = threading.Event(), threading.Event()
    generate = fake.generate

    def held(*args, **kwargs):
        running.set()
        assert release.wait(10)
        return generate(*args, **kwargs)

    monkeypatch.setattr(fake, "generate", held)
    job_id = client.post(
        "/api/generate", json={"prompt": "a lighthouse", "width": 256, "height": 256}
    ).json()["job_id"]
    try:
        assert running.wait(5) and main.jobs._gate.locked()
        body = analyse(client, prompt="A lighthouse on a cliff at dusk, oil painting")
        assert [e["name"] for e in body["encoders"]] == ["CLIP-L", "T5-XXL"]
        # Answered while the job ran, not after it.
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "running"
    finally:
        release.set()
    assert stream_job(client, job_id)[-1]["status"] == "done"
    assert where["analyse"] == [False] and where["tokenizer"] and not any(where["tokenizer"])


@pytest.mark.parametrize("route, measures", [
    ("/api/prompt/analyze", 1),
    # The analysis, then the rewrite's lint.
    ("/api/prompt/critique", 2),
])
def test_check_runs_on_a_daemon_thread(client, fake, monkeypatch, route, measures):
    # `detached.detached`'s own thread, which dies with the process - not the
    # default executor's, which interpreter exit joins. (Its daemon flag alone
    # proves nothing here: TestClient's loop thread is a daemon, and an
    # executor's workers inherit that.)
    threads: list[tuple[str, bool]] = []
    measure = prompt_encoding.measure

    def recording(*args):
        thread = threading.current_thread()
        threads.append((thread.name, thread.daemon))
        return measure(*args)

    monkeypatch.setattr(prompt_encoding, "measure", recording)
    claim(monkeypatch, fake, "black-forest-labs/FLUX.1-Krea-dev")
    assert client.post(route, json={"prompt": "A lighthouse on a cliff"}).status_code == 200
    assert threads == [("prompt-check", True)] * measures


# ------------------------------------------------------------------ measure


def test_the_measure_is_the_encoder_layer_alone(client, fake, monkeypatch):
    # The chip tray asks at every pause in typing: the counts and the cut,
    # the same as Check's, and no rule and no tag lookup - a Check's would
    # ask Danbooru about 64 items each time beside the chips' own lookup.
    claim(monkeypatch, fake, ANIMAGINE)
    fake.loaded = False

    def never(*args, **kwargs):
        raise AssertionError("the measure ran a rule or asked the tag index")

    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary())
    prompt = ", ".join(f"tag{i}" for i in range(50))
    expected = analyse(client, prompt=prompt)["encoders"]
    vocabulary = Vocabulary()
    monkeypatch.setattr(vocabulary, "validate", never)
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", vocabulary)
    monkeypatch.setattr(prompt_rules, "check", never)
    threads: list[tuple[str, bool]] = []
    measure = prompt_encoding.measure

    def recording(*args):
        thread = threading.current_thread()
        threads.append((thread.name, thread.daemon))
        return measure(*args)

    monkeypatch.setattr(prompt_encoding, "measure", recording)
    response = client.post("/api/prompt/measure", json={"prompt": prompt})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_id"] == ANIMAGINE and body["encoders_detail"] is None
    assert body["encoders"] == expected
    assert [(e["name"], e["budget"]) for e in body["encoders"]] == [("CLIP-L", 75), ("CLIP-G", 75)]
    # A token per word or mark: 99 of them, and the cut inside the list.
    assert all(e["tokens"] == 99 and 0 < e["cut"] < len(prompt) for e in body["encoders"])
    # Off the loop on a daemon thread of its own, never the image model.
    assert threads == [("prompt-measure", True)]
    assert fake.loads == 0 and not fake.loaded


def test_the_measure_answers_beside_a_running_job(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    running, release = threading.Event(), threading.Event()
    generate = fake.generate

    def held(*args, **kwargs):
        running.set()
        assert release.wait(10)
        return generate(*args, **kwargs)

    monkeypatch.setattr(fake, "generate", held)
    job_id = client.post(
        "/api/generate", json={"prompt": "a lighthouse", "width": 256, "height": 256}
    ).json()["job_id"]
    try:
        assert running.wait(5) and main.jobs._gate.locked()
        body = client.post("/api/prompt/measure", json={"prompt": "1girl, green hair"}).json()
        assert [e["tokens"] for e in body["encoders"]] == [4, 4]
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "running"
    finally:
        release.set()
    assert stream_job(client, job_id)[-1]["status"] == "done"


def test_an_unmeasured_encoder_is_said_never_shown_as_no_cut(client, fake, monkeypatch):
    # The tray reads a reason here as "cut unknown": an empty list alone
    # would read as a prompt every encoder takes whole.
    claim(monkeypatch, fake, ANIMAGINE)

    def offline(source):
        raise OSError("no network for a first use")

    monkeypatch.setattr(prompt_encoding, "_load_tokenizer", offline)
    body = client.post("/api/prompt/measure", json={"prompt": "1girl, green hair"}).json()
    assert body["encoders"] == [] and "CLIP-L" in body["encoders_detail"]
    # Uncounted this time, but the table knows the encoders: the line under
    # the prompt says "cut unknown" rather than nothing.
    assert body["encoders_known"] is True


def test_a_checkpoint_the_encoder_table_lacks_is_said_to_count_nothing(client, fake, monkeypatch):
    # Nothing will ever count this prompt: the line under the prompt box stays
    # hidden rather than saying "cut unknown" under every prompt, and the
    # reason is still the detail Check's Encoders section shows.
    off_table = "stabilityai/stable-diffusion-xl-base-1.0"
    assert prompt_encoding.encoders_for(off_table) is None
    claim(monkeypatch, fake, off_table)
    body = client.post("/api/prompt/measure", json={"prompt": "a cat"}).json()
    assert body["model_id"] == off_table and body["encoders"] == []
    assert body["encoders_known"] is False
    assert "no encoder table entry" in body["encoders_detail"]
    claim(monkeypatch, fake, ANIMAGINE)
    body = client.post("/api/prompt/measure", json={"prompt": "1girl, green hair"}).json()
    assert body["encoders_known"] is True and [e["tokens"] for e in body["encoders"]] == [4, 4]


def test_a_pathological_prompt_is_checked_not_a_500(client, fake, monkeypatch):
    claim(monkeypatch, fake, FLUX)
    body = analyse(client, prompt="[" * 3999)
    assert body["style"] != "json"


def test_every_finding_says_what_it_rests_on_and_warnings_come_first(client, fake, monkeypatch):
    claim(monkeypatch, fake, FLUX)
    long_negative = " ".join(f"blur{i}" for i in range(90))
    body = analyse(
        client,
        prompt="masterpiece, 8k, best quality, a street with no people, ((neon:1.3))",
        negative_prompt=long_negative,
        true_cfg=3.0,
    )
    findings = body["findings"]
    assert {"guide", "encoder", "heuristic"} <= {f["source"] for f in findings}
    for finding in findings:
        if finding["source"] == "heuristic":
            assert finding["cite"] is None and finding["url"] is None
        else:
            assert finding["cite"], finding
        if finding["source"] == "guide":
            assert finding["url"], finding
    # One order across both layers: warnings, then the prompt's before the
    # negative prompt's, then by position.
    keys = [
        (f["severity"] != "warn", f["field"] != "prompt", f["start"] is None, f["start"] or 0)
        for f in findings
    ]
    assert keys == sorted(keys)
    assert rules(findings, "encoder.weighting")[0]["severity"] == "warn"
    assert findings[-1]["field"] == "negative_prompt"


def test_flux_without_true_cfg_warns_on_a_negation_and_does_not_measure_the_negative(
    client, fake, monkeypatch
):
    claim(monkeypatch, fake, FLUX)
    prompt = "A quiet street at dawn with no people, wet cobblestones, soft light"
    body = analyse(client, prompt=prompt, negative_prompt="people, crowds")
    (negation,) = rules(body["findings"], "guide.negation")
    assert negation["severity"] == "warn" and negation["source"] == "guide"
    assert prompt[negation["start"]:negation["end"]] == "no people"
    # The negative prompt never runs, so no encoder is reported reading it.
    assert {e["field"] for e in body["encoders"]} == {"prompt"}

    # With True CFG it runs: the negation is advice, and the field is measured.
    body = analyse(client, prompt=prompt, negative_prompt="people, crowds", true_cfg=3.0)
    (negation,) = rules(body["findings"], "guide.negation")
    assert negation["severity"] == "info"
    assert {e["field"] for e in body["encoders"]} == {"prompt", "negative_prompt"}


def test_a_long_prompt_shows_where_clip_stops_and_t5_reads_on(client, fake, monkeypatch):
    claim(monkeypatch, fake, FLUX)
    prompt = " ".join(f"word{i}" for i in range(90))
    body = analyse(client, prompt=prompt)
    clip, t5 = body["encoders"]
    assert (clip["name"], clip["tokens"], clip["cut"]) == ("CLIP-L", 90, prompt.index("word75"))
    assert "pooled" in clip["role"] and clip["pooled_only"] and not t5["pooled_only"]
    assert (t5["tokens"], t5["cut"]) == (90, None)
    (cut,) = rules(body["findings"], "encoder.cut")
    assert cut["severity"] == "info" and cut["source"] == "encoder"


def test_a_template_slot_is_one_finding(client, fake, monkeypatch):
    # As a whole comma item, the author's "[character]" is also A1111's
    # de-emphasis to the encoder layer: one span, reported once, as the slot.
    claim(monkeypatch, fake, "SeeSee21/Z-Anime")
    body = analyse(client, prompt="[character], soft rim lighting, expressive eyes.")
    at_slot = [f["rule"] for f in body["findings"] if (f["start"], f["end"]) == (0, 11)]
    assert at_slot == ["author.placeholder"]
    # A bracketed item no example of the author's holds is still A1111's.
    body = analyse(client, prompt="[scene], soft rim lighting, expressive eyes.")
    assert [f["start"] for f in rules(body["findings"], "encoder.weighting")] == [0]
    # The card's landscape template: each slot once, as the author's, mid
    # clause or a whole item.
    template = next(
        prompt for prompt in guide_for("SeeSee21/Z-Anime")["examples"] if "[location]" in prompt
    )
    body = analyse(client, prompt=template)
    slots = rules(body["findings"], "author.placeholder")
    assert [template[f["start"]:f["end"]] for f in slots] == [
        "[location]", "[time of day]", "[lighting]", "[atmosphere]"
    ]
    assert rules(body["findings"], "encoder.weighting") == []


def test_the_check_says_how_many_of_its_words_are_characters(client, fake, monkeypatch):
    # Each Han or kana character is one of the rules' words; the page calls
    # them characters, from the count sent beside `words`.
    claim(monkeypatch, fake, "baidu/ERNIE-Image-Turbo")
    body = analyse(client, prompt="小狗，雪地，松树林，阴天，冷色调")
    assert (body["words"], body["characters"]) == (12, 12)
    body = analyse(client, prompt="一块写着'OPEN NOW'的木牌")
    assert (body["words"], body["characters"]) == (9, 7)
    body = analyse(client, prompt="A fox in fresh snow")
    assert (body["words"], body["characters"]) == (5, 0)


def test_a_failing_check_is_a_500_with_the_short_error(client, fake, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the rules broke\nRequest ID: 1234")

    monkeypatch.setattr(prompt_rules, "check", broken)
    response = client.post("/api/prompt/analyze", json={"prompt": "a lighthouse"})
    assert response.status_code == 500
    assert response.json()["detail"] == "RuntimeError: the rules broke"
