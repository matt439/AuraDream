"""The prompt check and the prompt critic through the HTTP layer.

`POST /api/prompt/analyze` is layer 1 - the checkpoint's tokenizers (the
word tokenizers in tests/fakes.py, at each encoder's real budget) and the
guide's rules - and must answer cold, without the image model, and beside a
running job, off the event loop and outside the job gate. `POST
/api/prompt/critique` is the local critic (FakeCritic here) around it: refused
while the card is promised to an image job or a load - before prepare() and
again after it - prepared on a daemon thread outside the job gate, run inside
it, and a job submitted meanwhile must not start moving weights onto the card
until the critic is done with it. Status never loads a tokenizer, and shutdown
stops the critic's server before anything waits on it, as Unload stops it off
the event loop - a server that will not stop is logged, and the rest of Unload
or shutdown still runs.
"""
from __future__ import annotations

import asyncio
import concurrent.futures.thread
import dataclasses
import inspect
import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError

import httpx
import pytest
import uvicorn

from backend import launch, main, prompt_encoding, prompt_rules, tags
from backend.config import settings
from backend.critic import RELEASE, CriticError, CriticUnavailable, PromptCritic, build_context
from backend.jobs import BackendBusy, JobManager
from backend.model_defaults import defaults_for, guide_for
from backend.prompt_sources import sources_for
from backend.schemas import GenerateRequest, PromptAnalysis, PromptAnalysisRequest
from backend.tags import TagLookupUnavailable
from tests.api_helpers import ORIGIN, stream_job, wait_for
from tests.fakes import CutReply, FakeBackend, FakeCritic, fake_tokenizer

FLUX = "black-forest-labs/FLUX.1-dev"
ANIMAGINE = "cagliostrolab/animagine-xl-4.0"
ANIMA = "circlestone-labs/Anima"
KOHAKU = "KBlueLeaf/Kohaku-XL-Zeta"
FIREFLY = "firefly \\(honkai: star rail\\)"
# Out of Animagine's documented order only by what Danbooru knows: a
# character after a general tag. Its own groups (1girl, the quality suffix)
# are already in place.
TAG_PROMPT = f"1girl, green hair, {FIREFLY}, masterpiece, high score, great score, absurdres"
IN_ORDER = f"1girl, {FIREFLY}, green hair, masterpiece, high score, great score, absurdres"


def claim(monkeypatch, fake, model_id):
    """Serve `model_id`, with the fake applying requests as its profile says
    - on FLUX.1 a negative prompt runs only under True CFG."""
    monkeypatch.setattr(main, "settings", dataclasses.replace(settings, model_id=model_id))
    profile = defaults_for(model_id)
    fake.cfg_threshold = profile["cfg_threshold"]
    fake.true_cfg_supported = profile["true_cfg_supported"]


class Vocabulary:
    """The tag index, stubbed: known categories, and every call recorded."""

    def __init__(self, categories=None, offline=False):
        self.categories = categories or {}
        self.offline = offline
        self.calls: list[tuple] = []

    def validate(self, tags, special, cutoff, allow_year_tag=True):
        self.calls.append((list(tags), set(special), cutoff, allow_year_tag))
        if self.offline:
            # As tags.py raises it: the connection error is the cause.
            raise TagLookupUnavailable("<urlopen error [Errno 11001] getaddrinfo failed>") from (
                URLError(OSError(11001, "getaddrinfo failed"))
            )
        return [
            {"tag": tag, "valid": tag in self.categories, "category": self.categories.get(tag)}
            for tag in tags
        ]


def rules(findings, rule):
    return [finding for finding in findings if finding["rule"] == rule]


def analyse(client, **body):
    response = client.post("/api/prompt/analyze", json={"prompt": "a lighthouse", **body})
    assert response.status_code == 200, response.text
    # The response model is the contract the page reads.
    return PromptAnalysis.model_validate(response.json()).model_dump()


def on_the_loop() -> bool:
    """Whether the calling thread is running an event loop: the app's, for
    blocking work a route called inline instead of handing it off."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


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
    real_analyse = main._analyse

    def watched_analyse(*args):
        where["analyse"].append(on_the_loop())
        return real_analyse(*args)

    def watched_tokenizer(source):
        where["tokenizer"].append(on_the_loop())
        return fake_tokenizer(source)

    monkeypatch.setattr(main, "_analyse", watched_analyse)
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
    # `main._detached`'s own thread, which dies with the process - not the
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

    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary())
    prompt = ", ".join(f"tag{i}" for i in range(50))
    expected = analyse(client, prompt=prompt)["encoders"]
    vocabulary = Vocabulary()
    monkeypatch.setattr(vocabulary, "validate", never)
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
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


def test_a_tag_checkpoint_orders_by_the_categories_the_chips_see(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary({"green hair": "general", FIREFLY: "character"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    body = analyse(client, prompt=TAG_PROMPT)
    # Asked the way POST /api/tags/validate asks: the guide's cutoff, its
    # special tags and year form - and only about what a rule would read.
    helper = guide_for(ANIMAGINE)["tag_helper"]
    ((tags, special, cutoff, year),) = vocabulary.calls
    assert tags == ["green hair", FIREFLY]
    assert special == main._special_tags(helper) and "absurdres" in special
    assert (cutoff, year) == (helper["vocabulary_cutoff"], True)

    assert body["style"] == "tags"
    assert body["order"]["steps"] == helper["order"]
    assert body["reordered"] == IN_ORDER
    (order,) = rules(body["findings"], "guide.order")
    assert order["severity"] == "info" and order["suggestion"] == IN_ORDER
    assert TAG_PROMPT[order["start"]:order["end"]] == "green hair"


def test_an_unreachable_tag_index_costs_the_order_check_not_the_answer(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary(offline=True)
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    body = analyse(client, prompt=TAG_PROMPT)
    assert len(vocabulary.calls) == 1
    assert rules(body["findings"], "guide.order") == [] and body["reordered"] is None
    assert body["order"] is not None and [e["name"] for e in body["encoders"]] == ["CLIP-L", "CLIP-G"]
    # ... and says what it cost. Placed from the guide's own groups alone, a
    # character after a general tag goes unflagged: without the detail, this
    # answer was the one an in-order prompt gets with the index reachable.
    assert body["order_checked"] is True
    assert body["categories_detail"] == (
        "the tag index was unreachable, so no tag has its Danbooru category"
    )
    monkeypatch.setattr(
        main, "tag_vocabulary", Vocabulary({"green hair": "general", FIREFLY: "character"})
    )
    in_order = analyse(client, prompt=IN_ORDER)
    assert in_order["order_checked"] is True and in_order["categories_detail"] is None
    assert rules(in_order["findings"], "guide.order") == [] and in_order["reordered"] is None

    def verdict(answer):
        return (
            answer["order_checked"], answer["categories_detail"], answer["reordered"],
            sorted(finding["rule"] for finding in answer["findings"]),
        )

    assert verdict(body) != verdict(in_order)


def test_a_broken_tag_index_reply_costs_the_order_check_not_the_answer(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    # A chunked reply cut off mid-body is http.client's error, not an OSError.
    monkeypatch.setattr(main, "tag_vocabulary", tags.DanbooruTagVocabulary())
    monkeypatch.setattr(tags, "urlopen", lambda request, timeout: CutReply())
    body = analyse(client, prompt=TAG_PROMPT)
    assert rules(body["findings"], "guide.order") == [] and body["reordered"] is None
    # An answer came, and was no use: not "unreachable".
    unusable = "the tag index gave no usable reply, so no tag has its Danbooru category"
    assert body["categories_detail"] == unusable
    # The chips read it the same way: unchecked, never an error.
    chips = client.post("/api/tags/validate", json={"tags": ["green hair"]})
    assert chips.status_code == 200 and chips.json()["available"] is False
    # A reply that parses into rows the lookup cannot read costs no more.
    monkeypatch.setattr(
        main, "tag_vocabulary", tags.DanbooruTagVocabulary(lambda path, params: ["not a row"])
    )
    body = analyse(client, prompt=TAG_PROMPT)
    assert rules(body["findings"], "guide.order") == [] and body["reordered"] is None
    assert body["categories_detail"] == unusable

    # Only a connection that never answered is the index unreachable; an
    # error status is an answer.
    def failing(error):
        def urlopen(request, timeout):
            raise error
        return urlopen

    for error, detail in [
        (HTTPError("https://danbooru.donmai.us", 503, "Service Unavailable", None, None), unusable),
        (URLError(ConnectionRefusedError(10061, "refused")), "the tag index was unreachable, "
         "so no tag has its Danbooru category"),
        (TimeoutError("timed out"), "the tag index was unreachable, "
         "so no tag has its Danbooru category"),
    ]:
        monkeypatch.setattr(main, "tag_vocabulary", tags.DanbooruTagVocabulary())
        monkeypatch.setattr(tags, "urlopen", failing(error))
        assert analyse(client, prompt=TAG_PROMPT)["categories_detail"] == detail, error


def test_the_tag_lookup_is_capped_at_the_chips_limit_and_skipped_for_prose(
    client, fake, monkeypatch
):
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary()
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    body = analyse(client, prompt=", ".join(f"tag{i}" for i in range(80)))
    ((tags, *_),) = vocabulary.calls
    assert len(tags) == main._MAX_TAG_ITEMS == 64
    # The 16 past the cap are placed by the guide's groups alone, and the
    # answer says so rather than passing them.
    assert body["order_checked"] is True and body["categories_detail"] == (
        "the tag index was asked about the first 64 of 80 tags only, so the rest have "
        "no Danbooru category"
    )
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary())
    at_the_cap = analyse(client, prompt=", ".join(f"tag{i}" for i in range(64)))
    assert at_the_cap["categories_detail"] is None
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    # Prose is not a tag list to categorise, even in comma pieces the order
    # rule would ask about - the user's sentences are not sent to Danbooru -
    # and FLUX.1 has no tag index.
    prose = "A girl with green hair stands in the rain, looking at the viewer, soft light"
    assert prompt_rules.detect_style(prose) == "prose"
    assert prompt_rules.category_items(ANIMAGINE, prose)
    analyse(client, prompt=prose)
    claim(monkeypatch, fake, FLUX)
    analyse(client, prompt="1girl, green hair")
    assert len(vocabulary.calls) == 1


def test_the_order_is_partly_checked_exactly_where_an_item_went_unplaced(
    client, fake, monkeypatch
):
    # One rule for "partly checked": some item the order rule reads kept
    # every slot a category could give it, where that decides the order
    # (`prompt_rules.places_every_item` false, placing with the categories
    # the rules had) - the predicate the critic holds a reorder of its own
    # to. `order_detail` says so, with the items and why each has no
    # category. A tag the index answered for and did not know, and a prose
    # prompt's clauses, which are never looked up, left `categories_detail`
    # empty, and the order read as checked while the critic's block said
    # Check could not place every tag.
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary({"green hair": "general", FIREFLY: "character"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    unknown = (
        'the tag index has no usable Danbooru category for 1 tag, so it is not placed: "zzkx oc"'
    )
    # An OC's name, a trigger word, a typo after a general tag: out of order
    # were it a character, in it were it a general tag - nothing is flagged,
    # and that is no pass.
    body = analyse(client, prompt="green hair, zzkx oc")
    assert body["order_checked"] is True and body["categories_detail"] is None
    assert rules(body["findings"], "guide.order") == [] and body["reordered"] is None
    assert body["order_detail"] == unknown
    # Every tag placed: checked, not partly. So too where an unplaced item's
    # slot decides nothing - ahead of a general tag, between 1girl and
    # masterpiece, it is in order whatever it is; ahead of 1girl, out of it
    # whatever it is - and for one item alone, which no order can be wrong
    # about: the lane first said "Partly checked" of all of these. Nor does a
    # tag written twice make it so: its copies take one slot, whatever it is.
    for prompt in (
        TAG_PROMPT, IN_ORDER, "zzkx oc, green hair", "1girl, zzkx oc, masterpiece",
        "zzkx oc, masterpiece", "zzkx oc", "a girl with green hair standing in the rain",
        '1girl, "OPEN", masterpiece', "1girl, @wlop, masterpiece",
        "1girl, zzkx oc, zzkx oc, masterpiece", "zzkx oc, zzkx oc",
    ):
        body = analyse(client, prompt=prompt)
        assert body["order_checked"] is True and body["order_detail"] is None, prompt
    # Beside another item it still decides the order, and is named once.
    body = analyse(client, prompt="green hair, zzkx oc, zzkx oc")
    assert body["order_detail"] == unknown
    body = analyse(client, prompt="zzkx oc, 1girl")
    assert rules(body["findings"], "guide.order") != [] and body["reordered"] == "1girl, zzkx oc"
    assert body["order_checked"] is True and body["order_detail"] is None
    # Which slots an unplaced item keeps is the guide's: Kohaku's meta tags
    # come after quality, so there it may sit out of order before masterpiece.
    claim(monkeypatch, fake, KOHAKU)
    body = analyse(client, prompt="1girl, zzkx oc, masterpiece")
    assert body["order_checked"] is True and body["order_detail"] == unknown
    claim(monkeypatch, fake, ANIMAGINE)
    # The index knows a tag and withholds its category - deprecated, or
    # created past the checkpoint's cutoff: not "unknown", which the chips,
    # reading the same rows, would contradict.
    rows = {
        "green_hair": {"name": "green_hair", "category": 0, "post_count": 5,
                       "created_at": "2010-01-01"},
        "old_tag": {"name": "old_tag", "category": 0, "post_count": 5,
                    "created_at": "2010-01-01", "is_deprecated": True},
        "new_hero_(game)": {"name": "new_hero_(game)", "category": 4, "post_count": 5,
                            "created_at": "2099-01-01"},
    }
    monkeypatch.setattr(main, "tag_vocabulary", tags.DanbooruTagVocabulary(
        fetch_json=lambda path, params: [
            rows[name] for name in params["search[name_normalize]"].split(",") if name in rows
        ]
    ))
    withheld = analyse(client, prompt="green hair, old tag, new hero \\(game\\)")
    assert withheld["categories_detail"] is None
    assert withheld["order_detail"] == (
        "the tag index has no usable Danbooru category for 2 tags, so they are not placed: "
        '"old tag", "new hero \\(game\\)"'
    )
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    # Prose is never looked up, and a clause that is not a whole sentence is
    # placed by nothing; whole sentences sit in the general slot.
    vocabulary.calls.clear()
    prose = "A girl with green hair stands in the rain, looking at the viewer, soft light"
    body = analyse(client, prompt=prose)
    assert vocabulary.calls == [] and body["style"] == "prose"
    assert body["order_checked"] is True and body["categories_detail"] is None
    assert body["order_detail"] == (
        "the tag index is not asked about a prose prompt, so 3 clauses that are not whole "
        'sentences are not placed: "A girl with green hair stands in the rain", '
        '"looking at the viewer", "soft light"'
    )
    sentences = analyse(client, prompt="A girl stands in the rain. She smiles softly.")
    assert sentences["order_checked"] is True and sentences["order_detail"] is None
    # An item never asked about - lettering, an @ name - beside a tag the
    # index did not know: each under its own reason.
    body = analyse(client, prompt='1girl, "OPEN", zzkx oc, @wlop, masterpiece')
    assert body["order_detail"] == (
        f"{unknown}; the tag index is not asked about 2 items, so they are not placed: "
        '""OPEN"", "@wlop"'
    )
    # The lookup's own reasons lead, with the items they cost.
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    body = analyse(client, prompt=TAG_PROMPT)
    assert body["order_detail"] == (
        "the tag index was unreachable, so no tag has its Danbooru category, and 2 tags are "
        f'not placed: "green hair", "{FIREFLY}"'
    )
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary())
    body = analyse(client, prompt=", ".join(f"tag{i}" for i in range(70)))
    assert body["order_detail"] == (
        "the tag index was asked about the first 64 of 70 tags only, so the rest have no "
        'Danbooru category, and 6 tags are not placed: "tag64", "tag65", "tag66" and 3 more; '
        "the tag index has no usable Danbooru category for 64 tags, so they are not placed: "
        '"tag0", "tag1", "tag2" and 61 more'
    )
    # The lookup's detail alone is no partial order: past the cap sits only a
    # phrase the rules place as prose, and every tag before it is known.
    known = [f"tag{i}" for i in range(64)]
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(dict.fromkeys(known, "general")))
    body = analyse(client, prompt=", ".join([*known, "looking at the viewer"]))
    assert body["categories_detail"].startswith("the tag index was asked about the first 64 of 65")
    assert body["order_checked"] is True and body["order_detail"] is None
    # Where the guide states no order, there is none to have checked partly:
    # the artist check's note is `categories_detail` alone.
    claim(monkeypatch, fake, ANIMA)
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    body = analyse(client, prompt="1girl, wlop, smile")
    assert body["order"] is None and body["order_detail"] is None
    assert body["categories_detail"].startswith("the tag index was unreachable")
    claim(monkeypatch, fake, FLUX)
    body = analyse(client, prompt="1girl, zzkx oc, green hair")
    assert body["order_checked"] is False and body["order_detail"] is None


def test_the_page_and_the_critic_read_partial_placement_alike(client, fake, monkeypatch):
    # The critic decides whether a reorder of its own may stand from the same
    # placement (`CriticContext.placed_partly`): wherever the analysis says
    # the order was partly checked, the critic treats it as partly placed, and
    # nowhere else.
    claim(monkeypatch, fake, ANIMAGINE)
    prompts = [
        TAG_PROMPT, IN_ORDER, "zzkx oc, green hair", "green hair, masterpiece, 1girl",
        "A girl with green hair stands in the rain, looking at the viewer, soft light",
        "A girl stands in the rain. She smiles softly.", '1girl, "OPEN", masterpiece',
        ", ".join(f"tag{i}" for i in range(70)), "1girl\ngreen hair", "",
    ]
    # A lone item and ones whose slot decides nothing, beside ones it does.
    prompts += [
        "zzkx oc", "green hair, zzkx oc", "1girl, zzkx oc, masterpiece", "zzkx oc, 1girl",
        "a girl with green hair standing in the rain", "zzkx oc, yyqv oc",
        "1girl, zzkx oc, zzkx oc, masterpiece", "green hair, zzkx oc, zzkx oc",
    ]
    for model_id in (ANIMAGINE, KOHAKU):
        claim(monkeypatch, fake, model_id)
        for vocabulary in (
            Vocabulary({"green hair": "general", FIREFLY: "character"}), Vocabulary(offline=True),
        ):
            monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
            for prompt in prompts:
                request = PromptAnalysisRequest(prompt=prompt or "a lighthouse")
                analysis = main._analyse(request)
                context = build_context(model_id, request, request, analysis)
                assert context.placed_partly == (analysis.order_detail is not None), prompt
                assert analysis.order_detail is None or analysis.order_checked, prompt


def test_the_order_says_whether_it_was_checked_and_what_the_authors_give(
    client, fake, monkeypatch
):
    # FLUX.1's order is documented but no rule reads a sentence's order: the
    # page must not let "nothing to flag" pass for a checked one.
    claim(monkeypatch, fake, FLUX)
    body = analyse(client, prompt="Golden hour light over misty mountains, a lone hiker on a ridge")
    assert body["order"] is not None and body["order_checked"] is False
    assert body["author_orders"] == []
    assert body["categories_detail"] is None
    # A tag checkpoint's comma items are placed - wholly with the tag index's
    # categories, partly without them, which is said beside the flag.
    claim(monkeypatch, fake, ANIMAGINE)
    monkeypatch.setattr(
        main, "tag_vocabulary", Vocabulary({"green hair": "general", FIREFLY: "character"})
    )
    body = analyse(client, prompt=TAG_PROMPT)
    assert body["order_checked"] is True and body["author_orders"] == []
    assert body["categories_detail"] is None
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    body = analyse(client, prompt=TAG_PROMPT)
    assert body["order_checked"] is True and body["categories_detail"] is not None
    # Anima's guide states no order for the app to check, but its card gives
    # one: shown as the authors', with its source.
    claim(monkeypatch, fake, ANIMA)
    body = analyse(client, prompt="1girl, smile")
    assert body["order"] is None and body["order_checked"] is False
    (card,) = body["author_orders"]
    assert card["text"].startswith("[quality/meta/year/safety tags] [1girl/1boy/1other etc]")
    assert card["url"] == "https://huggingface.co/circlestone-labs/Anima"
    assert card["note"] == "the card's tag order"
    claim(monkeypatch, fake, "duongve/NetaYume-Lumina-Image-2.0")
    body = analyse(client, prompt="1girl, smile")
    assert [o["note"].split("; ")[-1] for o in body["author_orders"]] == [
        "its standard sequence for the tags", "its order for the natural-language part",
    ]


def test_the_style_reported_is_the_one_the_rules_applied(client, fake, monkeypatch):
    claim(monkeypatch, fake, ANIMAGINE)
    prompt = "1girl, solo, the legend of zelda, princess zelda, smile"
    vocabulary = Vocabulary({"the legend of zelda": "copyright", "princess zelda": "character"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    body = analyse(client, prompt=prompt)
    # Its shape alone reads as partly prose; with the index's categories the
    # rules read it as the tag list it is, and so does the report.
    assert body["style"] == "tags" and rules(body["findings"], "guide.style") == []


def test_anima_asks_the_tag_index_which_items_are_artists(client, fake, monkeypatch):
    # Anima's guide has no tag helper, but "put @ before an artist name" needs
    # an artist's category: the index is asked with no cutoff of its own.
    claim(monkeypatch, fake, ANIMA)
    vocabulary = Vocabulary({"wlop": "artist", "smile": "general"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    prompt = "1girl, wlop, smile, @big chungus"
    body = analyse(client, prompt=prompt)
    ((asked, special, cutoff, year),) = vocabulary.calls
    assert asked == ["1girl", "wlop", "smile"] and (special, cutoff, year) == (set(), None, False)
    (artist,) = rules(body["findings"], "guide.artist_prefix")
    assert artist["suggestion"] == "1girl, @wlop, smile, @big chungus"
    assert prompt[artist["start"]:artist["end"]] == "wlop"
    assert body["categories_detail"] is None
    # Offline, the rule stands down, and the answer says why - there is no
    # order here to say it beside; prose is not asked about.
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    offline = analyse(client, prompt="1girl, wlop, smile")
    assert rules(offline["findings"], "guide.artist_prefix") == []
    assert offline["order"] is None and offline["categories_detail"].startswith(
        "the tag index was unreachable"
    )
    quiet = Vocabulary()
    monkeypatch.setattr(main, "tag_vocabulary", quiet)
    prose = "A girl painted by wlop stands in the rain, smiling at the viewer, soft light"
    assert prompt_rules.detect_style(prose) == "prose"
    assert prompt_rules.category_items(ANIMA, prose)
    analyse(client, prompt=prose)
    assert quiet.calls == []


def test_a_failing_check_is_a_500_with_the_short_error(client, fake, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the rules broke\nRequest ID: 1234")

    monkeypatch.setattr(prompt_rules, "check", broken)
    response = client.post("/api/prompt/analyze", json={"prompt": "a lighthouse"})
    assert response.status_code == 500
    assert response.json()["detail"] == "RuntimeError: the rules broke"


# ------------------------------------------------------------------ critique


def test_the_fake_critic_takes_what_the_real_one_takes():
    # Every route test here runs on FakeCritic and tests/test_critic.py runs
    # the real one without the routes, so nothing else would notice the two
    # drifting apart. Names, kinds and defaults; the fake's are unannotated.
    def shape(method):
        return [
            (p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()
        ]

    for name in ("supported", "ready", "describe", "prepare", "critique", "unload"):
        assert shape(getattr(FakeCritic, name)) == shape(getattr(PromptCritic, name)), name


def test_the_critique_is_refused_where_there_is_no_critic(client, fake, monkeypatch):
    fake.critic.available = False
    monkeypatch.setattr(main, "settings", dataclasses.replace(settings, critic=True))
    response = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
    assert response.status_code == 422
    assert "LOCAL_DIFFUSION_CRITIC_SERVER" in response.json()["detail"]
    monkeypatch.setattr(main, "settings", dataclasses.replace(settings, critic=False))
    response = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
    assert response.status_code == 422 and "switched off" in response.json()["detail"]
    assert fake.critic.prepares == 0


def test_the_critique_does_not_cut_ahead_of_the_queue(client, fake, monkeypatch):
    monkeypatch.setattr(main.jobs, "busy", lambda: True)
    response = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
    assert response.status_code == 409 and "queued or running" in response.json()["detail"]
    # Refused before anything is fetched.
    assert fake.critic.prepares == 0


def test_the_critique_is_refused_while_the_image_model_loads(client, fake, monkeypatch):
    fake.loaded = False
    release = threading.Event()
    load = fake.load

    def slow_load():
        assert release.wait(10)
        load()

    monkeypatch.setattr(fake, "load", slow_load)
    client.post("/api/model/load")
    try:
        deadline = time.time() + 5
        while client.get("/api/status").json()["state"] != "loading":
            assert time.time() < deadline, "the load never started"
            time.sleep(0.01)
        response = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
        assert response.status_code == 409 and "loading" in response.json()["detail"]
        assert fake.critic.prepares == 0
    finally:
        release.set()
    deadline = time.time() + 5
    while client.get("/api/status").json()["state"] != "ready":
        assert time.time() < deadline, "the load never finished"
        time.sleep(0.01)


def test_a_load_started_while_the_critic_is_fetched_is_a_409(client, fake, monkeypatch):
    # The first refusal passed; a first prepare() takes minutes, and a load
    # started meanwhile - the Load button, a preload - is no job, so
    # run_when_idle would admit the critique beside it. Only the check
    # after prepare() stops a llama-server starting beside 13 GB arriving.
    fake.loaded = False
    critic = fake.critic
    fetching, fetched, loaded = threading.Event(), threading.Event(), threading.Event()
    prepare, load = critic.prepare, fake.load

    def slow_prepare():
        fetching.set()
        assert fetched.wait(10)
        prepare()

    def slow_load():
        assert loaded.wait(10)
        load()

    monkeypatch.setattr(critic, "prepare", slow_prepare)
    monkeypatch.setattr(fake, "load", slow_load)
    with ThreadPoolExecutor(1) as pool:
        try:
            pending = pool.submit(
                client.post, "/api/prompt/critique", json={"prompt": "a lighthouse"}
            )
            assert fetching.wait(5)
            client.post("/api/model/load")
            deadline = time.time() + 5
            while client.get("/api/status").json()["state"] != "loading":
                assert time.time() < deadline, "the load never started"
                time.sleep(0.01)
            fetched.set()
            response = pending.result(timeout=10)
        finally:
            fetched.set()
            loaded.set()
    assert response.status_code == 409 and "loading" in response.json()["detail"]
    assert critic.prepares == 1 and critic.critiques == []
    deadline = time.time() + 5
    while client.get("/api/status").json()["state"] != "ready":
        assert time.time() < deadline, "the load never finished"
        time.sleep(0.01)


def test_prepare_runs_outside_the_gate_and_before_the_analysis(client, fake, monkeypatch):
    critic = fake.critic
    gate: dict[str, bool] = {}
    prepare, critique, real_analyse = critic.prepare, critic.critique, main._analyse

    def watched_prepare():
        gate["prepare"] = main.jobs._gate.locked()
        prepare()

    def watched_critique(*args):
        gate["critique"] = main.jobs._gate.locked()
        return critique(*args)

    def watched_analyse(*args):
        critic.events.append("analyse")
        return real_analyse(*args)

    monkeypatch.setattr(critic, "prepare", watched_prepare)
    monkeypatch.setattr(critic, "critique", watched_critique)
    monkeypatch.setattr(main, "_analyse", watched_analyse)
    response = client.post("/api/prompt/critique", json={"prompt": "a lighthouse in fog"})
    assert response.status_code == 200, response.text
    assert critic.events == ["prepare", "analyse", "critique", "critique done"]
    assert gate == {"prepare": False, "critique": True}


def test_prepare_and_the_load_run_on_daemon_threads(fake, monkeypatch):
    # Only a daemon thread of its own lets a Ctrl-C during the critic's first
    # download, or a load's, end the process: asyncio.to_thread's workers and
    # Starlette's threadpool's are joined at interpreter exit, and on the
    # loop's own thread nothing else runs until the call returns. Not through
    # TestClient: it runs the app's loop on a daemon thread, and every thread
    # that loop starts inherits the flag. Here the loop runs on pytest's main
    # thread, as uvicorn's does. The flag is not all of it: concurrent.futures
    # joins its own workers at exit whatever their flag, from a registry it
    # adds a worker to only once the worker has started - so that is read
    # after the calls.
    threads: dict[str, threading.Thread] = {}
    prepare, load = fake.critic.prepare, fake.load

    def watched_prepare():
        threads["prepare"] = threading.current_thread()
        prepare()

    def watched_load():
        threads["load"] = threading.current_thread()
        load()

    monkeypatch.setattr(fake.critic, "prepare", watched_prepare)
    monkeypatch.setattr(fake, "load", watched_load)

    async def scenario():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url=ORIGIN) as client:
            response = await client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
            assert response.status_code == 200, response.text
            fake.loaded = False
            await client.post("/api/model/load")
            deadline = time.time() + 5
            while (await client.get("/api/status")).json()["state"] != "ready":
                assert time.time() < deadline, "the load never finished"
                await asyncio.sleep(0.01)
        # No lifespan without TestClient: the worker the critique ran on is
        # drained here.
        await main.jobs.shutdown()

    assert not threading.current_thread().daemon
    asyncio.run(scenario())
    assert set(threads) == {"prepare", "load"}
    for name, thread in threads.items():
        assert thread.daemon and thread not in concurrent.futures.thread._threads_queues, name


def test_a_critique_cites_resolved_sources_and_lints_its_rewrite(client, fake, monkeypatch):
    claim(monkeypatch, fake, FLUX)
    prompt = "A lighthouse with no people, a stormy sea at dusk"
    response = client.post(
        "/api/prompt/critique", json={"prompt": prompt, "negative_prompt": "crowds"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    (issue,) = body["issues"]
    first = sources_for(FLUX).sentences[0]
    assert (issue["cite_id"], issue["cite"], issue["url"]) == ("G1", first.text, first.url)
    assert prompt[issue["start"]:issue["end"]] == issue["quote"] == "A"
    assert body["rewrite"] == f"Rewritten: {prompt}"
    # The rewrite went back through layer 1: its negation is flagged, as the
    # prompt's was, against the same applied request.
    (negation,) = rules(body["rewrite_findings"], "guide.negation")
    assert negation["severity"] == "warn"
    # The fake's answer went through the real validation: its reorder of the
    # prompt's own clauses stands, its parts are the prompt's words.
    assert body["reordered"] == "a stormy sea at dusk, A lighthouse with no people"
    assert body["parts"]["subject"] == "A lighthouse with no people"
    assert body["set_aside"] == []
    # One clause has nothing to reorder, on a checkpoint with an order.
    single = client.post("/api/prompt/critique", json={"prompt": "A lighthouse in fog"}).json()
    assert single["reordered"] is None and single["rewrite"] == "Rewritten: A lighthouse in fog"
    # The critic was handed the analysis and the request as it will run.
    (model_id, request, applied, analysis), _ = fake.critic.critiques
    assert model_id == FLUX and request.prompt == prompt
    assert request.negative_prompt == "crowds" and applied.negative_prompt is None
    assert rules([f.model_dump() for f in analysis.findings], "guide.negation")
    assert analysis.order is not None


def test_a_cited_excerpt_says_whose_words_it_is(client, fake, monkeypatch):
    # Krea published no prompting rules: its excerpts are BFL's family guide,
    # and a cite of one says so, as the same excerpt does under Order.
    krea = "black-forest-labs/FLUX.1-Krea-dev"
    claim(monkeypatch, fake, krea)
    answer = FakeCritic.answer

    def citing_a1(text, ordered):
        raw = json.loads(answer(text, ordered))
        raw["issues"][0]["cite_id"] = "A1"
        return json.dumps(raw)

    monkeypatch.setattr(FakeCritic, "answer", staticmethod(citing_a1))
    body = client.post("/api/prompt/critique", json={"prompt": "A lighthouse in fog"}).json()
    (issue,) = body["issues"]
    excerpt = sources_for(krea).excerpts[0]
    assert (issue["source"], issue["cite"], issue["url"]) == ("author", excerpt.text, excerpt.url)
    assert issue["note"] == excerpt.note and "Krea published no prompting rules" in issue["note"]
    # The fake's own G1, the checkpoint's guide, needs no note.
    monkeypatch.setattr(FakeCritic, "answer", staticmethod(answer))
    body = client.post("/api/prompt/critique", json={"prompt": "A lighthouse in fog"}).json()
    assert body["issues"][0]["source"] == "guide" and body["issues"][0]["note"] is None


def test_characters_the_encoder_cannot_read_do_not_take_the_critics_issues(
    client, fake, monkeypatch
):
    # T5 has no token for a brace: the check reports each place as unknown,
    # and an issue quoting the prompt's first word beside them, the braced
    # word or the empty pair once went as restating that finding.
    claim(monkeypatch, fake, FLUX)
    prompt = "a cat {sitting} on a mat {}"
    unknown = rules(analyse(client, prompt=prompt)["findings"], "encoder.unknown_chars")
    assert [prompt[finding["start"]:finding["end"]] for finding in unknown] == ["{", "}", "{}"]
    answer = FakeCritic.answer

    def quoting_braces(text, ordered):
        raw = json.loads(answer(text, ordered))
        first = raw["issues"][0]
        raw["issues"] += [{**first, "quote": "{sitting}"}, {**first, "quote": "{}"}]
        return json.dumps(raw)

    monkeypatch.setattr(FakeCritic, "answer", staticmethod(quoting_braces))
    body = client.post("/api/prompt/critique", json={"prompt": prompt}).json()
    assert [issue["quote"] for issue in body["issues"]] == ["a", "{sitting}", "{}"]
    assert body["set_aside"] == [] and body["dropped"] == 0


def test_the_critics_own_rewrite_checks_come_with_the_lint(client, fake, monkeypatch):
    # validate flags the prompt's description in quotation marks - lettering
    # on FLUX - which the rules skip; the route adds layer 1's findings.
    claim(monkeypatch, fake, FLUX)
    prompt = "A lighthouse on a cliff with no people"
    answer = FakeCritic.answer

    def quoting(text, ordered):
        raw = json.loads(answer(text, ordered))
        raw["rewrite"] = 'A "lighthouse" on a cliff with no people'
        return json.dumps(raw)

    monkeypatch.setattr(FakeCritic, "answer", staticmethod(quoting))
    body = client.post("/api/prompt/critique", json={"prompt": prompt}).json()
    # Warnings first, in the two layers' one order (the word count is info).
    assert [f["rule"] for f in body["rewrite_findings"]] == [
        "heuristic.rewrite_quote", "guide.negation", "guide.words",
    ]
    quote = body["rewrite_findings"][0]
    assert body["rewrite"][quote["start"]:quote["end"]] == '"lighthouse"'
    assert quote["source"] == "heuristic" and quote["severity"] == "warn"


def test_the_rewrite_is_linted_once_the_gate_is_open(client, fake, monkeypatch):
    # A tokenizer the analysis could not load is fetched again by the lint -
    # Hub timeouts per encoder - so the lint must not run under the gate,
    # where the fetch would hold image jobs back, and neither it nor the
    # analysis on the event loop, which the fetch would freeze.
    claim(monkeypatch, fake, FLUX)
    critic = fake.critic
    held: list[tuple[bool, bool]] = []

    def offline(source):
        held.append((main.jobs._gate.locked(), on_the_loop()))
        critic.events.append("tokenizer")
        raise OSError("the Hub is unreachable")

    monkeypatch.setattr(prompt_encoding, "_load_tokenizer", offline)
    response = client.post("/api/prompt/critique", json={"prompt": "A lighthouse with no people"})
    assert response.status_code == 200, response.text
    # CLIP-L and T5-XXL for the analysis, then again for the rewrite.
    assert critic.events == [
        "prepare", "tokenizer", "tokenizer", "critique", "critique done", "tokenizer", "tokenizer",
    ]
    # (gate held, on the loop) for each fetch.
    assert held == [(False, False)] * 4
    # The rules still ran over the rewrite; only its budgets went unchecked.
    (negation,) = rules(response.json()["rewrite_findings"], "guide.negation")
    assert negation["severity"] == "warn"


def test_the_rewrite_is_linted_without_asking_the_tag_index(client, fake, monkeypatch):
    # The analysis asks the tag index once per critique; the rewrite's order
    # check places what the guide's own groups know.
    claim(monkeypatch, fake, ANIMAGINE)
    vocabulary = Vocabulary({"green hair": "general", FIREFLY: "character"})
    monkeypatch.setattr(main, "tag_vocabulary", vocabulary)
    response = client.post("/api/prompt/critique", json={"prompt": TAG_PROMPT})
    assert response.status_code == 200, response.text
    assert len(vocabulary.calls) == 1
    assert fake.critic.critiques[0][3].reordered == IN_ORDER


def test_the_critic_holds_its_reorder_to_the_categories_check_placed_with(
    client, fake, monkeypatch
):
    # The fake's reorder puts the last tag first. Offline, no tag has its
    # category, and the order rule finds nothing wrong with firefly after
    # green hair or before it: the critic's reorder stands, and the page
    # shows what the server kept.
    claim(monkeypatch, fake, ANIMAGINE)
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))

    def critique(prompt):
        response = client.post("/api/prompt/critique", json={"prompt": prompt})
        assert response.status_code == 200, response.text
        return response.json()

    body = critique("firefly, green hair")
    assert body["reordered"] == "green hair, firefly" and body["set_aside"] == []
    assert body["reordered_kind"] == "allowed"
    analysis = fake.critic.critiques[-1][3]
    assert analysis.categories is None and analysis.categories_detail.startswith(
        "the tag index was unreachable"
    )
    # Online, with every tag looked up and none known - an OC's name - no
    # lookup failed, and the tags are placed no better: the same answer.
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary({}))
    body = critique("firefly, my oc")
    assert (body["reordered"], body["reordered_kind"], body["set_aside"]) == (
        "my oc, firefly", "allowed", [],
    )
    analysis = fake.critic.critiques[-1][3]
    assert analysis.categories == {} and analysis.categories_detail is None
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(offline=True))
    # 1girl leads by the guide's own groups, and the rule flags a tag put
    # before it.
    flagged = critique("1girl, green hair, firefly")
    assert flagged["reordered"] is None
    assert flagged["set_aside"] == [
        "The critic's reordered version was left out: in it, \"firefly\" comes before "
        "\"1girl\" (Person count); the documented order is Person count → Character → "
        "Series → Rating → Details → Quality."
    ]
    # Every tag placed, Check's order is the documented one, and the same
    # reorder is some other order.
    placed = {"firefly": "character", "green hair": "general"}
    monkeypatch.setattr(main, "tag_vocabulary", Vocabulary(placed))
    full = critique("firefly, green hair")
    assert full["reordered"] is None and full["reordered_kind"] is None
    assert full["set_aside"] == [
        "The critic's reordered version was left out: Check places these tags in the "
        "documented order itself, and the critic's order is not that one."
    ]
    assert fake.critic.critiques[-1][3].categories == placed
    # And where the critic's is Check's own order, the server says so.
    own = critique("green hair, firefly")
    assert (own["reordered"], own["reordered_kind"]) == ("firefly, green hair", "check")
    # The categories go to the critic, not to the page.
    checked = client.post("/api/prompt/analyze", json={"prompt": "firefly, green hair"})
    assert checked.status_code == 200 and "categories" not in checked.json()


@pytest.mark.parametrize(
    ("failure", "status", "detail", "logged"),
    [
        (True, 502, "llama-server exited with code 1 before it was ready", "CriticError"),
        (
            CriticUnavailable("the model file is gone"), 503, "the model file is gone",
            "CriticUnavailable",
        ),
        (RuntimeError("out of memory"), 500, "RuntimeError: out of memory", "RuntimeError"),
    ],
)
def test_a_failing_critique_answers_with_the_short_error(
    client, fake, caplog, failure, status, detail, logged
):
    # The critic's own failures reach the page without the class name, which
    # the page replaces with its own "The critic did not answer"; the log
    # line keeps it. Anything else keeps it in the reply as well.
    fake.critic.fail = failure
    with caplog.at_level("ERROR", logger=main.log.name):
        response = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
    assert response.status_code == status and response.json()["detail"] == detail
    assert f"prompt critique failed: {logged}: " in caplog.text


def test_a_prompt_too_long_for_the_critics_window_is_a_502(client, fake, monkeypatch):
    # The fake refuses what the real critic refuses, before any server
    # starts. Qwen3 reads a digit as a token, so 4000 of them, which the
    # answer repeats, overrun its 8192-token window on any estimate.
    claim(monkeypatch, fake, FLUX)
    response = client.post("/api/prompt/critique", json={"prompt": "0123456789" * 400})
    assert response.status_code == 502
    assert response.json()["detail"].startswith("the prompt is too long for the critic")
    # A prompt that fits is answered, as ever.
    assert client.post("/api/prompt/critique", json={"prompt": "a lighthouse"}).status_code == 200


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (CriticError("llama.zip has sha256 00ff, not the pinned 1d77"), 502),
        (CriticUnavailable("LOCAL_DIFFUSION_CRITIC_MODEL=x.gguf is not a file"), 503),
        (OSError("the network is unreachable"), 500),
    ],
)
def test_a_failing_prepare_answers_before_the_critic_runs(client, fake, failure, status):
    fake.critic.prepare_fail = failure
    response = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
    assert response.status_code == status
    # The critic's classes without their names, as for the critique itself.
    expected = str(failure) if isinstance(failure, CriticError) else f"OSError: {failure}"
    assert response.json()["detail"] == expected
    assert fake.critic.critiques == []


def test_a_job_queued_while_the_critique_waited_for_the_gate_is_a_409(client, fake, monkeypatch):
    async def queued_first(fn):
        raise BackendBusy("an image job was queued first")

    monkeypatch.setattr(main.jobs, "run_when_idle", queued_first)
    response = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"})
    assert response.status_code == 409
    assert response.json()["detail"] == "an image job was queued first"


def test_run_when_idle_checks_the_queue_again_once_it_has_the_gate(store):
    # The in-gate check: an operation admitted while the gate was busy waits
    # its turn, and a job queued meanwhile came first. Refused, and it never
    # runs; the job does.
    ran: list[str] = []

    async def scenario():
        manager = JobManager(FakeBackend())
        release = threading.Event()
        first = asyncio.create_task(manager.run_when_idle(release.wait))
        while not manager._gate.locked():
            await asyncio.sleep(0)
        second = asyncio.create_task(manager.run_when_idle(lambda: ran.append("second")))
        for _ in range(5):
            await asyncio.sleep(0)
        job = await manager.submit(GenerateRequest(prompt="x", width=256, height=256, steps=2))
        release.set()
        assert await first is True
        with pytest.raises(BackendBusy, match="queued first"):
            await second
        while job.status != "done":
            await asyncio.sleep(0.01)
        await manager.shutdown()
        return job

    job = asyncio.run(scenario())
    assert ran == [] and len(job.images) == 1


def test_a_job_submitted_during_a_critique_loads_only_after_it(client, fake, monkeypatch):
    # A queued job awaits the load before it takes the gate, so the gate alone
    # did not stop 13 GB of weights arriving beside the critic's server.
    fake.loaded = False
    critic = fake.critic
    critic.delay = 0.5
    load = fake.load

    def logged_load():
        critic.events.append("load")
        load()

    monkeypatch.setattr(fake, "load", logged_load)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(
            client.post, "/api/prompt/critique", json={"prompt": "a lighthouse in fog"}
        )
        assert critic.started.wait(5)
        accepted = client.post(
            "/api/generate", json={"prompt": "a lighthouse", "width": 256, "height": 256}
        )
        events = stream_job(client, accepted.json()["job_id"])
        response = pending.result(timeout=10)
    assert response.status_code == 200, response.text
    assert events[-1]["status"] == "done"
    assert critic.events == ["prepare", "critique", "critique done", "load"]
    assert fake.loads == 1


# ------------------------------------------------------------------ status and unload


def test_status_says_whether_the_critic_runs_here_and_is_on_disk(client, fake):
    status = client.get("/api/status").json()
    assert (status["prompt_critique_supported"], status["prompt_critique_ready"]) == (True, False)
    assert client.post("/api/prompt/critique", json={"prompt": "a lighthouse"}).status_code == 200
    status = client.get("/api/status").json()
    assert (status["prompt_critique_supported"], status["prompt_critique_ready"]) == (True, True)
    fake.critic.available = False
    status = client.get("/api/status").json()
    assert (status["prompt_critique_supported"], status["prompt_critique_ready"]) == (False, False)


def test_status_names_the_critic_as_its_answers_do_and_what_a_first_use_fetches(client, fake):
    # The line under the critic's button is built from this, not from a
    # fixed "Qwen3-4B": the fake stands for the pinned pair, both downloads
    # until its first use.
    before = client.get("/api/status").json()["prompt_critic"]
    assert (before["label"], before["pinned"]) == ("fake critic", True)
    assert before["fetch"] == [f"llama.cpp {RELEASE}", "fake critic"] and before["fetch_bytes"] > 4e9
    answer = client.post("/api/prompt/critique", json={"prompt": "a lighthouse"}).json()
    assert answer["model"] == before["label"]
    after = client.get("/api/status").json()["prompt_critic"]
    assert (after["fetch"], after["fetch_bytes"]) == ([], 0)


def test_status_describes_a_critic_of_the_users_own_from_the_settings(
    client, fake, monkeypatch, tmp_path
):
    # The real critic, the review's two cases: a GGUF of one's own beside
    # the installed pinned server, whose line named Qwen3-4B and its time
    # while the footer named the GGUF; and a server of one's own with the
    # pinned model not yet cached, whose line promised a llama.cpp download.
    from backend import critic as pinned

    monkeypatch.setattr(main, "prompt_critic", PromptCritic())
    monkeypatch.setattr(pinned, "platform_key", lambda: "win-cuda-13.4-x64")
    monkeypatch.setattr(pinned, "installed_server", lambda key: tmp_path / "llama-server.exe")
    monkeypatch.setattr(pinned, "_model_cached", lambda: None)
    own_model = tmp_path / "gemma-3-12b-Q4.gguf"
    monkeypatch.setattr(pinned, "settings", dataclasses.replace(
        pinned.settings, critic=True, critic_server=None, critic_model=own_model
    ))
    assert client.get("/api/status").json()["prompt_critic"] == {
        "label": "gemma-3-12b-Q4", "pinned": False, "fetch": [], "fetch_bytes": 0,
    }
    monkeypatch.setattr(pinned, "installed_server", lambda key: None)
    monkeypatch.setattr(pinned, "settings", dataclasses.replace(
        pinned.settings, critic_server=tmp_path / "llama-server", critic_model=None
    ))
    assert client.get("/api/status").json()["prompt_critic"] == {
        "label": pinned.MODEL_LABEL, "pinned": False,
        "fetch": [pinned.MODEL_LABEL], "fetch_bytes": pinned.MODEL_SIZE,
    }


def test_status_never_loads_a_tokenizer(client, fake, monkeypatch):
    # Polled every three seconds: on a cold cache a tokenizer asked for here
    # would start a Hub fetch per poll, holding the lock Check and Unload
    # wait on. Every catalogue checkpoint, before a load and after one.
    loads: list = []

    def recording(source):
        loads.append(source)
        return fake_tokenizer(source)

    monkeypatch.setattr(prompt_encoding, "_load_tokenizer", recording)
    described = fake.describe

    def poll():
        for checkpoint in launch.CHECKPOINTS:
            claim(monkeypatch, fake, checkpoint.repo)
            monkeypatch.setattr(
                fake, "describe", lambda repo=checkpoint.repo: {**described(), "model_id": repo}
            )
            assert client.get("/api/status").json()["model_id"] == checkpoint.repo

    fake.loaded = False
    poll()
    client.post("/api/model/load")
    deadline = time.time() + 5
    while client.get("/api/status").json()["state"] != "ready":
        assert time.time() < deadline, "the load never finished"
        time.sleep(0.01)
    poll()
    assert loads == [] and prompt_encoding._tokenizers == {}
    # The recorder is the loader a Check reaches.
    claim(monkeypatch, fake, FLUX)
    analyse(client)
    assert loads


def test_unload_stops_the_critic_and_drops_the_tokenizers(client, fake, monkeypatch):
    unloads: list[str] = []
    monkeypatch.setattr(prompt_encoding, "unload", lambda: unloads.append("tokenizers"))
    assert client.post("/api/model/unload").status_code == 200
    assert fake.critic.unloads == 1 and unloads == ["tokenizers"]


def test_unload_answers_at_once_while_a_check_waits_on_a_tokenizer(client, fake, monkeypatch):
    # A first use's tokenizer fetch has no timeout of its own; Unload must not
    # wait on it, and says "unloaded" as soon as that is true.
    entered, release = threading.Event(), threading.Event()

    def stalled(source):
        entered.set()
        release.wait(30)
        return fake_tokenizer(source)

    monkeypatch.setattr(prompt_encoding, "_load_tokenizer", stalled)
    claim(monkeypatch, fake, "baidu/ERNIE-Image-Turbo")
    main._load_state.update(state="ready")
    with ThreadPoolExecutor(2) as pool:
        try:
            check = pool.submit(client.post, "/api/prompt/analyze", json={"prompt": "a cat"})
            assert entered.wait(5)
            unloading = pool.submit(client.post, "/api/model/unload")
            done, _ = concurrent.futures.wait([unloading], timeout=10)
            assert done, "Unload waited on the tokenizer fetch"
            reply = unloading.result()
            assert (reply.status_code, reply.json()["state"]) == (200, "unloaded")
            assert client.get("/api/status").json()["state"] == "unloaded"
        finally:
            release.set()
        check.result(timeout=5)


def test_unload_says_unloaded_before_the_analyser_and_critic_let_go(client, fake, monkeypatch):
    # The critic's is the slow one: it stops a server, waiting up to twice
    # `STOP_TIMEOUT` to kill it - so off the event loop, as each release is,
    # or every progress stream would stall behind it.
    main._load_state.update(state="ready")
    seen: dict[str, tuple[str, bool]] = {}

    def recording(name):
        return lambda: seen.setdefault(name, (main._load_state["state"], on_the_loop()))

    monkeypatch.setattr(fake.translator, "unload", recording("translator"))
    monkeypatch.setattr(fake.critic, "unload", recording("critic"))
    monkeypatch.setattr(prompt_encoding, "unload", recording("tokenizers"))
    assert client.post("/api/model/unload").status_code == 200
    assert seen == dict.fromkeys(("translator", "critic", "tokenizers"), ("unloaded", False))


def test_a_critic_that_will_not_stop_costs_unload_nothing_else(client, fake, monkeypatch, caplog):
    # The kill not waited out that shutdown logs and goes past: raised through
    # the route, it made Unload a 500 over weights already gone, and left the
    # tokenizers cached.
    def stuck():
        raise CriticError("llama-server (pid 4242) had not exited 10 s after it was killed")

    main._load_state.update(state="ready")
    unloads: list[str] = []
    monkeypatch.setattr(fake.critic, "unload", stuck)
    monkeypatch.setattr(prompt_encoding, "unload", lambda: unloads.append("tokenizers"))
    with caplog.at_level("ERROR", logger=main.log.name):
        reply = client.post("/api/model/unload")
    assert (reply.status_code, reply.json()["state"]) == (200, "unloaded")
    assert client.get("/api/status").json()["state"] == "unloaded"
    assert unloads == ["tokenizers"]
    assert "stopping the prompt critic failed: CriticError: llama-server (pid 4242)" in caplog.text


def test_shutdown_stops_the_critic_first_and_drops_the_tokenizers(fake, monkeypatch):
    from fastapi.testclient import TestClient

    from tests.api_helpers import ORIGIN

    # The critic's server before the worker: a critique in flight holds the
    # worker thread in a request to that server, which jobs.shutdown() would
    # wait out - up to its 600 s timeout. Stopped first, the request fails
    # at once. The worker, in turn, before the weights it is using.
    stops: list[str] = []
    critic_unload, jobs_shutdown = fake.critic.unload, main.jobs.shutdown
    try_unload = fake.try_unload

    def stop_critic():
        stops.append("critic")
        critic_unload()

    async def stop_jobs():
        stops.append("jobs")
        await jobs_shutdown()

    def drop_weights():
        stops.append("weights")
        return try_unload()

    monkeypatch.setattr(fake.critic, "unload", stop_critic)
    monkeypatch.setattr(main.jobs, "shutdown", stop_jobs)
    monkeypatch.setattr(fake, "try_unload", drop_weights)
    unloads: list[str] = []
    monkeypatch.setattr(prompt_encoding, "unload", lambda: unloads.append("tokenizers"))
    with TestClient(main.app, base_url=ORIGIN):
        pass
    assert stops == ["critic", "jobs", "weights"]
    assert fake.critic.unloads == 1 and unloads == ["tokenizers"]


def test_a_critic_that_will_not_stop_costs_shutdown_nothing_else(fake, monkeypatch, caplog):
    from fastapi.testclient import TestClient

    # A kill not waited out is unload()'s CriticError: raised through the
    # lifespan, it skipped the worker's drain, the weights and the tokenizers.
    def stuck():
        raise CriticError("llama-server (pid 4242) had not exited 10 s after it was killed")

    stops: list[str] = []
    jobs_shutdown, try_unload = main.jobs.shutdown, fake.try_unload

    async def stop_jobs():
        stops.append("jobs")
        await jobs_shutdown()

    def drop_weights():
        stops.append("weights")
        return try_unload()

    monkeypatch.setattr(fake.critic, "unload", stuck)
    monkeypatch.setattr(main.jobs, "shutdown", stop_jobs)
    monkeypatch.setattr(fake, "try_unload", drop_weights)
    monkeypatch.setattr(prompt_encoding, "unload", lambda: stops.append("tokenizers"))
    with caplog.at_level("ERROR", logger=main.log.name):
        with TestClient(main.app, base_url=ORIGIN):
            pass
    assert stops == ["jobs", "weights", "tokenizers"]
    assert "stopping the prompt critic failed: CriticError: llama-server (pid 4242)" in caplog.text


def test_the_exit_signals_critic_stop_logs_a_failure(client, fake, monkeypatch, caplog):
    # The signal's own stop runs on a thread of its own; a CriticError there
    # was an unhandled thread exception, a traceback outside the app's log.
    def stuck():
        raise CriticError("llama-server (pid 4242) had not exited 10 s after it was killed")

    unhandled: list[BaseException] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: unhandled.append(args.exc_value))
    monkeypatch.setattr(fake.critic, "unload", stuck)
    server = main.Server(uvicorn.Config(main.app, log_config=None))
    with caplog.at_level("ERROR", logger=main.log.name):
        server.handle_exit(signal.SIGINT, None)
        deadline = time.time() + 5
        while "stopping the prompt critic failed" not in caplog.text and not unhandled:
            assert time.time() < deadline, "the critic's stop never finished"
            time.sleep(0.01)
    assert unhandled == [] and "CriticError: llama-server (pid 4242)" in caplog.text
    assert server.should_exit


def test_the_exit_signal_cancels_the_jobs_and_stops_the_critic(client, fake, monkeypatch):
    # uvicorn drains open connections before the lifespan runs, and a job's
    # stream and a critique in flight are both open ones: Ctrl-C reaches them
    # on the signal itself. Stopping the critic's server waits for a process,
    # so it runs on a thread of its own and the handler returns at once.
    stopping, stopped, release = threading.Event(), threading.Event(), threading.Event()
    unload = fake.critic.unload

    def slow_unload():
        stopping.set()
        release.wait(5)
        unload()
        stopped.set()

    monkeypatch.setattr(fake.critic, "unload", slow_unload)
    fake.step_seconds = 0.02
    job_id = client.post(
        "/api/generate", json={"prompt": "a lighthouse", "width": 256, "height": 256, "steps": 100}
    ).json()["job_id"]
    wait_for(client, job_id, lambda snapshot: snapshot["status"] == "running")
    server = main.Server(uvicorn.Config(main.app, log_config=None))
    try:
        server.handle_exit(signal.SIGINT, None)
        assert main.jobs.get(job_id).cancel_requested
        # Returned with the critic's stop still under way.
        assert stopping.wait(5) and not stopped.is_set()
    finally:
        release.set()
    assert stopped.wait(5) and fake.critic.unloads == 1
    assert server.should_exit
    assert stream_job(client, job_id)[-1]["status"] == "cancelled"


def test_ctrl_c_ends_the_server_with_130_and_no_traceback(fake, monkeypatch):
    # uvicorn re-raises the signal it captured once shutdown is done, and
    # `run()` calls the server directly rather than through `uvicorn.run()`,
    # which would swallow it: every Ctrl-C ended in a KeyboardInterrupt
    # traceback after "Application shutdown complete". The real path, a real
    # signal: the server starts, SIGINT arrives, the lifespan shuts down.
    started = threading.Event()

    class Interrupted(main.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)
            started.set()
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(main, "Server", Interrupted)
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, port=0))
    try:
        with pytest.raises(SystemExit) as exc:
            main.run()
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C still reached the caller as KeyboardInterrupt")
    assert started.is_set() and exc.value.code == 130
    assert fake.critic.unloads >= 1
