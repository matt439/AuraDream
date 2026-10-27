"""Critic HTTP orchestration, job gates, refusal, and rewrite validation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import asyncio
import concurrent.futures.thread
import dataclasses
import inspect
import json
import threading
import time

import httpx
import pytest

from backend import main, prompt_check, prompt_encoding, tag_lookup
from backend.config import settings
from backend.critic import CriticError, CriticUnavailable, PromptCritic
from backend.jobs import BackendBusy, JobManager
from backend.prompt_sources import sources_for
from backend.schemas import GenerateRequest

from tests.api_analysis_helpers import (
    ANIMAGINE,
    FIREFLY,
    FLUX,
    IN_ORDER,
    TAG_PROMPT,
    Vocabulary,
    analyse,
    claim,
    on_the_loop,
    rules,
)
from tests.api_helpers import ORIGIN, stream_job
from tests.fakes import FakeBackend, FakeCritic


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
    assert "AURADREAM_CRITIC_SERVER" in response.json()["detail"]
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
    prepare, critique, real_analyse = critic.prepare, critic.critique, prompt_check.analyse

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
    monkeypatch.setattr(prompt_check, "analyse", watched_analyse)
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
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", vocabulary)
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
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary(offline=True))

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
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary({}))
    body = critique("firefly, my oc")
    assert (body["reordered"], body["reordered_kind"], body["set_aside"]) == (
        "my oc, firefly", "allowed", [],
    )
    analysis = fake.critic.critiques[-1][3]
    assert analysis.categories == {} and analysis.categories_detail is None
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary(offline=True))
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
    monkeypatch.setattr(tag_lookup, "tag_vocabulary", Vocabulary(placed))
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
        (CriticUnavailable("AURADREAM_CRITIC_MODEL=x.gguf is not a file"), 503),
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
    async def queued_first(fn, holder="editor"):
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
