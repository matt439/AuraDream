"""The guard's decision and failure paths with synthetic samples and an owned CPU child."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import time

import psutil
import pytest

WORK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gen49_guard", WORK / "gen49-stage3-guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def reading(**fields):
    s = guard.instrumentation.RehearsalReader().read(None)
    return replace(s, sample=replace(s.sample, **fields))


@pytest.mark.parametrize(("field", "value"), [
    ("tree_gib", 16.001), ("available_gib", 3.999), ("commit_headroom_gib", 3.999),
    ("card_mib", 15501.), ("adapter_shared_mib", 1079.1), ("process_shared_mib", 512.1),
    ("available_gib", None), ("process_shared_mib", None), ("card_mib", float("nan")),
])
def test_live_guards(field, value):
    with pytest.raises(RuntimeError):
        guard.check_live(reading(**{field: value}), guard.card.Baseline(55., 0.))


def test_stale_and_private():
    baseline = guard.card.Baseline(55., 0.)
    for r in [replace(reading(), card_at=time.monotonic() - .6), replace(reading(), private_gib=16.01)]:
        with pytest.raises(RuntimeError):
            guard.check_live(r, baseline)


def test_owned_accounting():
    r = replace(reading(commit_headroom_gib=21.5, available_gib=24., card_mib=2500.), private_gib=2.5)
    assert not guard.admit(r, cold=True)["admitted"]
    assert guard.admit(r, cold=False)["admitted"]
    assert not guard.admit(replace(r, private_gib=2.499), cold=False)["admitted"]
    assert not guard.admit(replace(r, sample=replace(r.sample, available_gib=23.99)), cold=False)["admitted"]


@pytest.mark.parametrize("failure", ["cold", "load", "stalled"])
def test_real_child_supervision(monkeypatch, tmp_path, failure):
    class Fake(guard.instrumentation.RehearsalReader):
        def read(self, pid):
            s = super().read(pid)
            if (failure == "cold" and pid is None) or (failure == "load" and (tmp_path / "load-request").exists()):
                return replace(s, sample=replace(s.sample, commit_headroom_gib=10.))
            if failure == "stalled" and (tmp_path / "load-request").exists():
                time.sleep(3.)
            return s
    monkeypatch.setattr(guard.instrumentation, "WindowsReader", Fake)
    assert guard.run(tmp_path, WORK / "gen49-guard-rehearsal-child.py") == 5
    summary = json.loads((tmp_path / "guard-result.json").read_text())
    assert summary["status"] == "held" and not (tmp_path / "load-admitted").exists()
    if failure == "cold":
        assert summary["child_pid"] is None
        assert not (tmp_path / "child.log").exists()
    else:
        assert not psutil.pid_exists(summary["child_pid"])
        assert not psutil.pid_exists(int((tmp_path / "descendant-pid").read_text()))
        if failure == "stalled":
            assert "watchdog" in summary["reason"]
