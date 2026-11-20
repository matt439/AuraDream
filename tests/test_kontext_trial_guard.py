"""Synthetic guard rehearsals. No real processes, counters, browser, or weights."""
from __future__ import annotations

from dataclasses import replace
import json
import threading
import time

import pytest

from tools import kontext_trial_guard as guard


class MockReader(guard.RehearsalReader):
    def __init__(self):
        self.fields = {}
        self.private = 1.
        self.closed = False
        self.n = 0

    def read(self, pid):
        self.n += 1
        base = super().read(pid)
        return replace(base, sample=replace(base.sample, **self.fields), private_gib=self.private if pid else 0.)

    def close(self):
        self.closed = True


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def issue(g, action):
    nonce = str(time.monotonic_ns())
    guard.atomic_json(g.output / "request.json", {"pid": g.pid, "nonce": nonce, **action})
    until = time.monotonic() + 30
    while time.monotonic() < until:
        g.check()
        dest = g.output / "permit.json"
        if dest.exists() and read(dest)["nonce"] == nonce:
            return
        time.sleep(.005)
    raise AssertionError("missing permit")


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(guard, "PERIOD", .012)


@pytest.mark.parametrize("fields,reason", [
    ({"card_mib": 2000.1}, "idle"),
    ({"available_gib": 44.99}, "RAM"),
    ({"commit_headroom_gib": guard.COMMIT_GIB - .00001}, "commit"),
    ({"adapter_shared_mib": None}, "unreadable"),
])
def test_early_hold_keeps_reason_before_any_owned_process(tmp_path, fields, reason):
    reader = MockReader()
    reader.fields = fields
    with pytest.raises(guard.GuardHeld, match=reason):
        with guard.TrialGuard(tmp_path, fake=True, reader=reader):
            pytest.fail("admitted")
    summary = read(tmp_path / "summary.json")
    assert summary["status"] == "held" and summary["server_pid"] is None
    assert reason in summary["reason"] and reader.closed


def test_load_rechecks_after_browser_started_and_subtracts_only_tree_private(tmp_path, fast):
    reader = MockReader()
    stopped = threading.Event()
    with pytest.raises(guard.GuardHeld, match="commit"):
        with guard.TrialGuard(tmp_path, fake=True, reader=reader) as g:
            g.attach(101, stopped.set)
            g.start_phase("load")
            reader.fields = {"tree_gib": 20., "commit_headroom_gib": guard.COMMIT_GIB - reader.private - .001}
            issue(g, {"action": "load"})
    assert stopped.is_set() and reader.closed and not (tmp_path / "permit.json").exists()
    rows = [json.loads(s) for s in (tmp_path / "guard.jsonl").read_text().splitlines()]
    decision = next(r for r in rows if r["event"] == "admit_load")
    assert decision["required_commit_gib"] == guard.COMMIT_GIB - reader.private
    assert decision["admitted"] is False


@pytest.mark.parametrize("fields,reason", [
    ({"card_mib": 15500.1}, "card"),
    ({"adapter_shared_mib": 55. + 1024.1}, "adapter_shared"),
    ({"process_shared_mib": 512.1}, "process_shared"),
    ({"available_gib": 3.9}, "ram"),
    ({"commit_headroom_gib": 3.9}, "commit"),
    ({"tree_gib": None}, "unreadable"),
])
def test_live_bad_reading_stops_server_without_waiting_for_browser(tmp_path, fast, fields, reason):
    reader, stopped = MockReader(), threading.Event()
    with pytest.raises(guard.GuardHeld, match=reason):
        with guard.TrialGuard(tmp_path, fake=True, reader=reader) as g:
            g.attach(101, stopped.set)
            reader.fields = fields
            assert stopped.wait(30)
            g.check()
    assert read(tmp_path / "stopped.json")["server_pid"] == 101


def test_heartbeat_stops_owned_server_during_blocked_reader(tmp_path, fast, monkeypatch):
    monkeypatch.setattr(guard, "HEARTBEAT", .12)
    stopped = threading.Event()
    original = MockReader()

    class BlockingReader(MockReader):
        def read(self, pid):
            if pid:
                assert stopped.wait(30)
            return original.read(pid)

    with pytest.raises(guard.GuardHeld, match="heartbeat"):
        with guard.TrialGuard(tmp_path, fake=True, reader=BlockingReader()) as g:
            g.attach(101, stopped.set)
    assert stopped.is_set()


def test_heartbeat_can_kill_while_sampler_holds_receipt_lock(tmp_path, fast, monkeypatch):
    monkeypatch.setattr(guard, "HEARTBEAT", .12)
    stopped = threading.Event()
    g = guard.TrialGuard(tmp_path, fake=True)
    write_row = g._row

    def stalled_write(event, **fields):
        if event == "sample":
            assert stopped.wait(30)
        write_row(event, **fields)

    monkeypatch.setattr(g, "_row", stalled_write)
    with pytest.raises(guard.GuardHeld, match="heartbeat"):
        with g:
            g.attach(101, stopped.set)
    assert read(tmp_path / "stopped.json")["phase"] == "startup"


def edit(width, height, recipe="kontext-v1"):
    return {"action": "edit", "seed": 12345, "recipe": recipe, "source": 42, "width": width, "height": height,
            "prompt": "Change the red ball to a blue ball.", "steps": 28, "guidance_scale": 2.5}


def phase(g, name, action):
    g.start_phase(name, [action])
    issue(g, action)
    return g.end_phase(name)


def test_guard_binds_all_cases_and_recovers_then_admits_fresh_replay(tmp_path, fast):
    first_dir = tmp_path / "first"
    with guard.TrialGuard(first_dir, fake=True) as g:
        g.attach(101, lambda: None)
        phase(g, "load", {"action": "load"})
        assert g.project()["verdict"] == "refuse-projected"
        for name, (w, h) in guard.SMALL_CASES.items():
            assert phase(g, name, edit(w, h))["samples"] > 1
        assert g.project()["verdict"] == "ok"
        phase(g, "first-1024", edit(1024, 1024, "kontext-1024-v1"))
        g.start_phase("unload")
        g.end_phase("unload")
        assert g.observe_unload(True, False)["samples"] >= 3
    summary = read(first_dir / "summary.json")
    assert summary["status"] == "recovered" and summary["fake"]
    assert len(summary["actions"]) == 9
    assert summary["phases"]["first-1024"]["peak_tree_rss_gib"] > 0
    with guard.TrialGuard(tmp_path / "replay", fake=True) as replay:
        replay.attach(102, lambda: None)
        phase(replay, "load", {"action": "load"})
        assert replay.replay_projection(first_dir)["verdict"] == "ok"


def test_no_large_permit_before_projection_and_unrequested_actions_stop(tmp_path, fast):
    with pytest.raises(guard.GuardHeld, match="unadmitted edit"):
        with guard.TrialGuard(tmp_path / "large", fake=True) as g:
            g.attach(101, lambda: None)
            phase(g, "load", {"action": "load"})
            phase(g, "first-1024", edit(1024, 1024, "kontext-1024-v1"))
    with pytest.raises(guard.GuardHeld, match="unexpected candidate action"):
        with guard.TrialGuard(tmp_path / "stray", fake=True) as g:
            g.attach(101, lambda: None)
            issue(g, {"action": "load"})


def test_projection_preserves_earlier_non_edit_peak(tmp_path, fast):
    reader = MockReader()
    with guard.TrialGuard(tmp_path, fake=True, reader=reader) as g:
        g.attach(101, lambda: None)
        g.start_phase("load")
        issue(g, {"action": "load"})
        reader.fields = {"card_mib": 15000.}
        g.end_phase("load")
        reader.fields = {}
        for name, (w, h) in guard.SMALL_CASES.items():
            phase(g, name, edit(w, h))
        verdict = g.project()
        assert verdict["verdict"] == "refuse-projected" and "15000" in verdict["reason"]
