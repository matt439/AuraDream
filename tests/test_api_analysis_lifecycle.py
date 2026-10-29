"""Prompt-analysis status, unload, shutdown, and exit signals."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures.thread
import dataclasses
import signal
import threading
import time

import pytest
import uvicorn

from backend import launch, main, prompt_encoding
from backend.critic import CriticError, PromptCritic, RELEASE

from tests.api_analysis_helpers import (
    FLUX,
    analyse,
    claim,
    on_the_loop,
)
from tests.api_helpers import (
    ORIGIN,
    stream_job,
    wait_for,
)
from tests.fakes import fake_tokenizer


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
    from backend import critic as pinned, critic_runtime

    monkeypatch.setattr(main, "prompt_critic", PromptCritic())
    monkeypatch.setattr(critic_runtime, "platform_key", lambda: "win-cuda-13.4-x64")
    monkeypatch.setattr(critic_runtime, "installed_server", lambda key: tmp_path / "llama-server.exe")
    monkeypatch.setattr(critic_runtime, "_model_cached", lambda: None)
    own_model = tmp_path / "gemma-3-12b-Q4.gguf"
    monkeypatch.setattr(pinned, "settings", dataclasses.replace(
        pinned.settings, critic=True, critic_server=None, critic_model=own_model
    ))
    assert client.get("/api/status").json()["prompt_critic"] == {
        "label": "gemma-3-12b-Q4", "pinned": False, "fetch": [], "fetch_bytes": 0,
    }
    monkeypatch.setattr(critic_runtime, "installed_server", lambda key: None)
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
    # The catalogue ends on the editor, whose load the gate holds: load a
    # checkpoint that can.
    claim(monkeypatch, fake, FLUX)
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
        # Said as the server's stop, not a Cancel pressed in some other tab.
        assert main.jobs.get(job_id).stopped_by == "shutdown"
        # Returned with the critic's stop still under way.
        assert stopping.wait(5) and not stopped.is_set()
    finally:
        release.set()
    assert stopped.wait(5) and fake.critic.unloads == 1
    assert server.should_exit
    assert stream_job(client, job_id)[-1]["status"] == "cancelled"


def test_the_exit_signal_ends_a_job_waiting_on_the_load_at_once(client, fake, monkeypatch):
    # Asked to stop, a job awaiting the load stayed queued until the load
    # ended, and its stream with it: uvicorn drains streams before the
    # lifespan, so a Ctrl-C during a first load waited out the 30 s graceful
    # timeout and the stream was cut with no done. Finished on the signal,
    # as a Cancel finishes a queued job, and said as the server's stop.
    fake.loaded = False
    release = threading.Event()
    load = fake.load

    def slow_load():
        assert release.wait(10)
        load()

    monkeypatch.setattr(fake, "load", slow_load)
    job_id = client.post(
        "/api/generate", json={"prompt": "a lighthouse", "width": 256, "height": 256}
    ).json()["job_id"]
    try:
        wait_for(client, job_id, lambda snapshot: snapshot["waiting_on"] == "load")
        server = main.Server(uvicorn.Config(main.app, log_config=None))
        server.handle_exit(signal.SIGINT, None)
        done = stream_job(client, job_id)[-1]
        assert not release.is_set()
        assert (done["type"], done["status"], done["stopped_by"]) == ("done", "cancelled", "shutdown")
    finally:
        release.set()
    # The load finishing afterwards changes nothing: no image, no second end.
    wait_for(client, job_id, lambda snapshot: client.get("/api/status").json()["state"] == "ready")
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    assert fake.requests == []


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


@pytest.mark.parametrize(
    ("host", "loopback"),
    [("127.0.0.1", True), ("localhost", True), ("LocalHost", True), ("::1", True), ("127.0.0.2", True),
     ("0.0.0.0", False), ("192.168.1.20", False), ("::", False), ("my-box", False)],
)
def test_only_a_loopback_bind_is_quiet_about_the_missing_login(host, loopback):
    assert main._is_loopback(host) is loopback
