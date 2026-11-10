"""Idle release on virtual time and the real server's fake backend; no weights."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest

from backend import keep_awake, main
from backend.idle_unload import IdleUnload
from tests.api_helpers import ORIGIN, wait_for_status


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_idle_uses_real_activity_and_gives_busy_work_a_full_grace():
    async def scenario():
        clock = Clock()
        activity = keep_awake.KeepAwake(monotonic=clock)
        state = {"busy": False, "ready": True, "unloads": 0}

        async def unload():
            state["unloads"] += 1
            state["ready"] = False

        timer = IdleUnload(minutes=1, activity_age=activity.idle_seconds,
                           busy=lambda: state["busy"], ready=lambda: state["ready"],
                           unload=unload, monotonic=clock)
        assert not await timer.check()
        clock.now = 59
        assert not await timer.check()
        # Timed polls do not refresh the clock; real requests do.
        assert not keep_awake.counts_as_activity("GET", "/api/status")
        activity.note_activity()
        clock.now = 100
        assert not await timer.check()
        state["busy"] = True
        clock.now = 200
        assert not await timer.check()
        state["busy"] = False
        clock.now = 300
        assert not await timer.check()
        clock.now = 359
        assert not await timer.check()
        clock.now = 360
        assert await timer.check()
        assert state["unloads"] == 1
        clock.now = 900
        assert not await timer.check()
        # A later load receives a new full interval.
        state["ready"] = True
        assert not await timer.check()
        clock.now = 960
        assert await timer.check()
        assert state["unloads"] == 2
    asyncio.run(scenario())


def test_stop_drains_an_admitted_unload_without_cancelling_it():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def unload():
            entered.set()
            await release.wait()

        clock = Clock()
        timer = IdleUnload(minutes=1, activity_age=lambda: 1000, busy=lambda: False,
                           ready=lambda: True, unload=unload, monotonic=clock)
        await timer.check()
        clock.now = 61
        timer.start()
        await asyncio.wait_for(entered.wait(), 5)
        stopped = asyncio.create_task(timer.stop())
        await asyncio.sleep(0)
        assert not stopped.done()
        release.set()
        await asyncio.wait_for(stopped, 5)
        assert not await timer.check()
    asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["jobs", "card", "loading", "switch", "bundle", "counted", "request"])
def test_idle_waits_for_every_live_server_owner(fake, monkeypatch, reason):
    activity = keep_awake.KeepAwake()
    monkeypatch.setattr(main, "_keep_awake", activity)
    for provider in main.KEEP_AWAKE_PROVIDERS:
        activity.add_provider(provider)
    if reason == "jobs":
        monkeypatch.setattr(main.jobs, "busy", lambda: True)
    elif reason == "card":
        monkeypatch.setattr(main.jobs, "card", lambda: ("critic", None))
    elif reason == "loading":
        monkeypatch.setitem(main._load_state, "state", "loading")
    elif reason == "switch":
        monkeypatch.setattr(main, "_switching", True)
    elif reason == "bundle":
        monkeypatch.setattr(main, "_busy_bundle", lambda: "bundle")
        activity.add_provider(main._busy_bundle)
    elif reason == "counted":
        with activity.busy("a detached worker"):
            assert main._idle_unload_busy()
        assert not main._idle_unload_busy()
        return
    else:
        monkeypatch.setattr(main, "_model_requests", 1)
    assert main._idle_unload_busy()


def test_idle_activity_tracking_works_with_the_power_inhibitor_off(fake, monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, keep_awake=False, idle_unload_minutes=1))

    async def scenario():
        activity = main._start_keep_awake()
        assert activity is not None
        monkeypatch.setattr(main, "_keep_awake", activity)
        assert isinstance(activity.inhibitor, keep_awake.NoInhibitor)
        status = main._keep_awake_status()
        assert status is not None and status.kind == "off"
        await main._stop_keep_awake()
    asyncio.run(scenario())


def test_idle_uses_the_manual_cleanup_and_explicit_load_clears_the_reason(client, fake, monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, idle_unload_minutes=1))
    fake.load()
    calls = []
    monkeypatch.setattr(main.prompt_encoding, "unload", lambda: calls.append("tokenizers"))
    monkeypatch.setattr(main.prompt_translator, "unload", lambda: calls.append("translator"))
    monkeypatch.setattr(main, "_stop_critic", lambda: calls.append("critic"))
    client.portal.call(main._unload_for_idle)
    state = client.get("/api/status").json()
    assert state["state"] == "unloaded" and state["idle_unloaded"]
    assert state["idle_unload_minutes"] == 1
    assert calls == ["translator", "critic", "tokenizers"]
    assert client.post("/api/model/load", headers={"Origin": ORIGIN}).status_code == 200
    wait_for_status(client, state="ready")
    assert not client.get("/api/status").json()["idle_unloaded"]


def test_unload_reserves_cleanup_before_a_new_load_can_enter(client, fake, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    fake.load()

    def cleanup():
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(main.prompt_translator, "unload", cleanup)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(client.post, "/api/model/unload", headers={"Origin": ORIGIN})
        try:
            assert entered.wait(5)
            assert client.post("/api/model/load", headers={"Origin": ORIGIN}).status_code == 409
            assert client.post("/api/generate", json={"prompt": "wait"}, headers={"Origin": ORIGIN}).status_code == 409
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 200


def test_a_cancelled_unload_waiter_leaves_cleanup_reserved_until_its_worker_ends(client, fake, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    fake.load()

    def cleanup():
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(main.prompt_translator, "unload", cleanup)

    async def scenario():
        waiter = asyncio.create_task(main._unload_weights(idle=True))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            owner = main._unload_task
            assert owner is not None
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            assert main._switching and not owner.done()
            response = await asyncio.to_thread(client.post, "/api/model/load", headers={"Origin": ORIGIN})
            assert response.status_code == 409
        finally:
            release.set()
        await asyncio.wait_for(owner, 5)
        assert not main._switching and main._idle_unloaded
        # The idle flag is set before admission reopens, so a later load wins.
        response = await asyncio.to_thread(client.post, "/api/model/load", headers={"Origin": ORIGIN})
        assert response.status_code == 200
        assert not main._idle_unloaded

    client.portal.call(scenario)
