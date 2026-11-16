"""Private card runner's admission, ownership and retained failure paths, CPU only."""
from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import errno
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from PIL import Image

from tools import anima_refine_card as card
from tools import card_guard as guard


GOOD = guard.Sample(card_mib=1300., card_total_mib=16303., adapter_shared_mib=100.,
                    process_shared_mib=0., available_gib=47., commit_headroom_gib=55., tree_gib=.05)
LIMITS = replace(guard.LIMITS, need_ram_gib=40., busy_card_mib=2000.)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Worker:
    """A deterministic worker that never imports a backend or starts a process."""
    pid = 417

    def __init__(self, path, clock, phases=None, stall=False):
        self.path, self.clock = path, clock
        self.exited = False
        self.killed = False
        self.phases = phases or [("load", "load"), *[("case", row["id"]) for row in card.pack.cases()
                                                     if not row.get("optional")], ("cleanup", "cleanup")]
        self.sequence = 1
        self.stall = stall
        self.released = []
        self.request()

    def request(self):
        kind, name = self.phases[self.sequence - 1]
        card.publish_message(self.path, "request", self.sequence,
                             {"sequence": self.sequence, "kind": kind, "id": name})

    def poll(self):
        return (1 if self.killed else 0) if self.exited else None

    def wait(self, timeout):
        if not self.exited:
            raise subprocess.TimeoutExpired("fake worker", timeout)
        return self.poll()

    def sleep(self, seconds):
        self.clock.sleep(seconds)
        ack = next((found for action in ("complete", "skip", "begin")
                    if (found := card.read_json(card.message_path(self.path, "ack", self.sequence, action))) is not None), None)
        if ack is None or ack.get("sequence") != self.sequence:
            return
        if ack["action"] == "begin":
            if self.sequence not in self.released:
                self.released.append(self.sequence)
            if self.stall:
                self.clock.sleep(601)
                return
            result = {"sequence": self.sequence, "ok": True}
            kind, name = self.phases[self.sequence - 1]
            if kind == "case":
                result["case"] = {"status": "cancelled" if name == "cancel-04" else "completed",
                                  "metrics": {"elapsed_seconds": .1, "allocated_peak_mib": 100.,
                                              "reserved_peak_mib": 110., "resident_allocated_mib": 20.}}
            if not card.message_path(self.path, "result", self.sequence).exists():
                card.publish_message(self.path, "result", self.sequence, result)
        elif ack["action"] in ("complete", "skip"):
            if self.sequence == len(self.phases):
                self.exited = True
            else:
                self.sequence += 1
                self.request()


class Host:
    sources = {"shared": "pdh", "card": "fake"}

    def __init__(self, worker, readings=None):
        self.worker = worker
        self.readings = iter(readings or [])
        self.kills = 0

    def tree(self, pid):
        # Deliberately continue returning the PID after exit: the supervisor
        # must reap a completed child using poll, not spin to the wall cap.
        return [pid]

    def sample(self, pids):
        return next(self.readings, GOOD)

    def kill(self, pid):
        self.kills += 1
        self.worker.killed = self.worker.exited = True


def supervise(path, *, phases=None, readings=None, stall=False, fallback=False):
    clock = Clock()
    worker = Worker(path, clock, phases=phases, stall=stall)
    host = Host(worker, readings)
    if fallback:
        host.sources = {"shared": "typeperf -sc 1"}
    code = card.supervise(path, LIMITS, host=host, proc=worker, clock=clock, sleep=worker.sleep)
    return code, worker, host, card.read_json(path / "receipt.json")


@pytest.mark.parametrize("name", card.REQUIRED_COUNTERS)
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1.])
def test_required_counter_gaps_never_release_load(tmp_path, name, value):
    code, worker, host, receipt = supervise(tmp_path, readings=[replace(GOOD, **{name: value})])
    assert code == 1 and worker.released == [] and host.kills == 1
    assert "required counter" in receipt["stopped_reason"]
    assert receipt["load_count"] is None
    assert (tmp_path / "after-exit.json").is_file()


@pytest.mark.parametrize("sample,reason", [(replace(GOOD, available_gib=39.), "preflight RAM"),
    (replace(GOOD, commit_headroom_gib=3.), "preflight commit"),
    (replace(GOOD, card_mib=2001.), "card busy")])
def test_preflight_resource_refusals_precede_load(tmp_path, sample, reason):
    code, worker, host, receipt = supervise(tmp_path, readings=[sample])
    assert code == 1 and worker.released == [] and host.kills == 1
    assert reason in receipt["stopped_reason"]


def test_typeperf_fallback_is_refused_before_release(tmp_path):
    code, worker, _, receipt = supervise(tmp_path, fallback=True)
    assert code == 1 and worker.released == []
    assert "PDH" in receipt["stopped_reason"]


def test_guard_trip_during_load_stops_before_later_case(tmp_path):
    code, worker, host, receipt = supervise(tmp_path, readings=[GOOD, GOOD, replace(GOOD, process_shared_mib=513.)])
    assert code == 1 and worker.released == [1] and host.kills == 1
    assert "process_shared" in receipt["stopped_reason"]
    rows = [json.loads(line) for line in (tmp_path / "guard.jsonl").read_text().splitlines()]
    assert any(row["event"] == "sample" and row["answer"]["verdict"] == "trip" for row in rows)


def test_load_timeout_retains_partial_receipt_and_kills(tmp_path):
    code, worker, host, receipt = supervise(tmp_path, stall=True)
    assert code == 1 and host.kills == 1 and worker.released == [1]
    assert "load wall-time cap" in receipt["stopped_reason"]


@pytest.mark.parametrize("kind,seconds", [("load", 600), ("case", 120), ("cleanup", 120)])
def test_phase_caps(kind, seconds):
    assert card.deadline_reason(seconds - .01, 0., {"kind": kind}, 0.) is None
    assert kind in card.deadline_reason(seconds, 0., {"kind": kind}, 0.)
    assert "session" in card.deadline_reason(1800, 0., None, 0.)


def test_exited_child_is_reaped_even_if_tree_still_lists_pid(tmp_path):
    code, worker, host, receipt = supervise(tmp_path)
    assert code == 0 and worker.exited and host.kills == 0
    assert receipt["load_count"] == 1 and receipt["stopped_reason"] is None
    assert receipt["session"]["final_residency_receipt"] == "cleanup-receipt.json"
    assert all(row["visual_review"] is None for row in receipt["cases"])


def test_optional_enlargement_without_a_measured_slope_is_skipped(tmp_path):
    phases = [("load", "load"), *[("case", row["id"]) for row in card.pack.cases()], ("cleanup", "cleanup")]
    code, worker, host, receipt = supervise(tmp_path, phases=phases)
    assert code == 0 and host.kills == 0
    assert receipt["cases"][-1]["status"] == "skipped"
    assert "no defensible enlargement projection" in receipt["cases"][-1]["reason"]
    # The optional worker phase receives skip, never begin.
    assert len(card.pack.cases()) + 1 not in worker.released


def test_environment_preserves_cache_but_clears_live_app_choices(monkeypatch, tmp_path):
    monkeypatch.setenv("AURADREAM_PRELOAD", "true")
    monkeypatch.setenv("AURADREAM_PASSCODE", "private-test-value")
    monkeypatch.setenv("AURADREAM_MODEL_ID", "other-model")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    env = card.environment(tmp_path)
    assert env["HF_HOME"] == str(tmp_path / "cache")
    assert env["AURADREAM_MODEL_ID"] == card.pack.CHECKPOINT
    assert env["AURADREAM_PRELOAD"] == "false" and "AURADREAM_PASSCODE" not in env
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["AURADREAM_DB_PATH"] == str(tmp_path / "scratch.db")


def test_import_and_help_are_cold():
    script = "from tools import anima_refine_card; import sys; assert 'torch' not in sys.modules; assert 'backend' not in sys.modules"
    done = subprocess.run([sys.executable, "-c", script], cwd=card.ROOT, capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    help_result = subprocess.run([sys.executable, str(Path(card.__file__)), "--help"],
                                 capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0 and "private" in help_result.stdout.lower()


def test_independent_watchdog_detects_stalled_sampler_and_wall_cap():
    clock = Clock()
    watchdog = card.Watchdog(lambda: None, clock=clock)
    clock.sleep(15)  # initial counter startup is before any worker release
    assert watchdog.check() is None
    watchdog.pulse({"kind": "load"})
    clock.sleep(2.01)
    assert "heartbeat" in watchdog.check()
    watchdog.pulse({"kind": "load"})
    clock.sleep(600)
    assert "load wall-time cap" in watchdog.check()


def test_independent_watchdog_actually_kills_once_when_supervisor_stalls():
    clock = Clock()
    kills = []
    watchdog = card.Watchdog(lambda: kills.append(True), clock=clock)
    watchdog.pulse({"kind": "load"})
    class TickStop:
        def wait(self, interval):
            clock.sleep(3)
            return False
    watchdog.stop = TickStop()
    watchdog._run()
    assert kills == [True]
    assert "heartbeat" in watchdog.reason


class FakeTorch:
    def __init__(self):
        self.cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 20, max_memory_reserved=lambda: 40, memory_allocated=lambda: 10)

    def Generator(self, device):
        assert device == "cpu"
        return SimpleNamespace(manual_seed=lambda seed: seed)


class Cancelled(Exception):
    pass


class Pipeline:
    def __init__(self, failure=False):
        self.scheduler = SimpleNamespace(begin_index=4, step_index=5, sigmas=[1., 1., 1., 1., .75])
        self.cleanups = 0
        self.failure = failure
        self.inputs = []

    def maybe_free_model_hooks(self):
        self.cleanups += 1

    def __call__(self, **kw):
        self.inputs.append(kw)
        if self.failure:
            raise RuntimeError("injected encode failure")
        for index in range(4):
            kw["callback_on_step_end"](self, index, 750 - index, {})
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "blue")])


@pytest.mark.parametrize("cancel", [False, True])
def test_real_case_path_uses_immutable_source_actual_callbacks_and_owner_cleanup(tmp_path, cancel):
    source = Image.new("RGB", (8, 8), "red")
    source_bytes = source.tobytes()
    text, runner = Pipeline(), Pipeline()
    spec = next(row for row in card.pack.cases() if row["id"] == ("cancel-04" if cancel else "strength-04"))
    result = card.run_case(text, runner, spec, source, torch=FakeTorch(), cancelled=Cancelled,
                           fitted=lambda image, w, h: image, output=tmp_path)
    case = result["case"]
    assert source.tobytes() == source_bytes
    assert runner.inputs[0]["image"] is not source
    assert runner.inputs[0]["generator"] == card.pack.SEED
    assert text.cleanups == 1 and runner.cleanups == 0
    assert case["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert case["start_sigma"] == .75
    if cancel:
        assert case["status"] == "cancelled" and case["completed_steps"] == 1
        assert result["image"] is None and list(tmp_path.iterdir()) == []
    else:
        assert case["completed_steps"] == 4 and case["status"] == "completed"
        assert case["pixel_sha256"] == hashlib.sha256(result["image"].tobytes()).hexdigest()
        assert case["metadata_sha256"] == hashlib.sha256((tmp_path / "strength-04.json").read_bytes()).hexdigest()


def test_loaded_owner_cleanup_after_call_exception(tmp_path):
    text, runner = Pipeline(), Pipeline(failure=True)
    spec = next(row for row in card.pack.cases() if row["id"] == "strength-04")
    with pytest.raises(RuntimeError, match="injected encode failure"):
        card.run_case(text, runner, spec, Image.new("RGB", (8, 8)), torch=FakeTorch(), cancelled=Cancelled,
                      fitted=lambda image, w, h: image, output=tmp_path)
    assert text.cleanups == 1 and list(tmp_path.iterdir()) == []


def test_uncached_conversion_refuses_before_backend_load(monkeypatch, tmp_path):
    import torch
    from backend import backends
    from backend.backends import anima_loading

    events = []
    class ImmediateHandshake:
        sequence = 1
        def __init__(self, path):
            pass
        def begin(self, kind, name):
            events.append((kind, name))
            return True
        def finish(self, **result):
            events.append(result)
    monkeypatch.setattr(card, "Handshake", ImmediateHandshake)
    @contextmanager
    def held(path):
        yield SimpleNamespace(kind="power-request", held=True)
    monkeypatch.setattr(card, "worker_hold", held)
    monkeypatch.setattr(card.sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA GeForce RTX 5080")
    monkeypatch.setattr(anima_loading, "converted_tree", lambda _: None)
    monkeypatch.setattr(backends, "get_backend", lambda _: pytest.fail("backend cannot be constructed on a cache miss"))
    assert card.worker(tmp_path, False) == 1
    assert events[0] == ("load", "load")
    assert events[-1]["ok"] is False and "not cached" in events[-1]["error"]


@pytest.mark.parametrize("name", ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"])
def test_cuda_device_remapping_refuses_before_child_or_directory(monkeypatch, tmp_path, name):
    monkeypatch.setattr(card.sys, "platform", "win32")
    monkeypatch.setenv(name, "0")
    path = tmp_path / "not-created"
    with pytest.raises(SystemExit):
        card.main(["--output", str(path), "--need-ram-gib", "40", "--busy-card-mib", "2000"])
    assert not path.exists()


class FakeHolder:
    kind = "power-request"
    held = False
    def __init__(self, fail=False):
        self.fail = fail
        self.holds = self.releases = 0
    def hold(self):
        self.holds += 1
        if self.fail:
            raise OSError("injected power request refusal")
        self.held = True
    def release(self):
        self.releases += 1
        self.held = False


@pytest.mark.parametrize("fail_work", [False, True])
def test_worker_power_request_covers_work_and_releases_even_on_exception(tmp_path, fail_work):
    holder = FakeHolder()
    def exercise():
        with card.worker_hold(tmp_path, inhibitor=holder) as active:
            assert active is holder and holder.held
            live = card.read_json(tmp_path / "keep-awake.json")
            assert live["holding"] and live["acquired"] and not live["released"]
            if fail_work:
                raise RuntimeError("injected render failure")
    if fail_work:
        with pytest.raises(RuntimeError, match="injected render failure"):
            exercise()
    else:
        exercise()
    assert holder.holds == holder.releases == 1 and not holder.held
    final = card.read_json(tmp_path / "keep-awake.json")
    assert final["acquired"] and final["released"] and not final["holding"]
    assert final["failure"] is None
    assert bool(final["scope_error"]) == fail_work


def test_unavailable_power_request_fails_closed_before_work_and_still_releases(tmp_path, caplog):
    holder = FakeHolder(fail=True)
    with pytest.raises(RuntimeError, match="required Windows power request"):
        with card.worker_hold(tmp_path, inhibitor=holder):
            pytest.fail("no model work may begin without the required hold")
    assert holder.holds == holder.releases == 1
    assert "injected power request refusal" in caplog.text
    final = card.read_json(tmp_path / "keep-awake.json")
    assert not final["acquired"] and not final["holding"] and final["released"]
    assert "not held" in final["failure"]


def test_worker_releases_power_before_reporting_failure_to_killing_parent(monkeypatch, tmp_path):
    from backend import keep_awake

    holder = FakeHolder()
    events = []
    class ImmediateHandshake:
        def __init__(self, path):
            pass
        def begin(self, kind, name):
            events.append((kind, name))
            return True
        def finish(self, **result):
            assert not holder.held and holder.releases == 1
            events.append(result)
    def fail_session(*args):
        assert holder.held
        raise RuntimeError("injected session failure")
    monkeypatch.setattr(keep_awake, "platform_inhibitor", lambda reason: holder)
    monkeypatch.setattr(card, "Handshake", ImmediateHandshake)
    monkeypatch.setattr(card, "_worker_session", fail_session)
    assert card.worker(tmp_path, False) == 1
    assert events[-1]["ok"] is False and "injected session failure" in events[-1]["error"]
    assert card.read_json(tmp_path / "keep-awake.json")["released"]


def test_ipc_messages_are_never_replaced_or_reused(tmp_path):
    card.publish_message(tmp_path, "ack", 1, {"action": "begin"}, "begin")
    before = card.message_path(tmp_path, "ack", 1, "begin").read_bytes()
    card.publish_message(tmp_path, "ack", 1, {"action": "complete"}, "complete")
    with pytest.raises(FileExistsError, match="immutable IPC"):
        card.publish_message(tmp_path, "ack", 1, {"action": "changed"}, "begin")
    assert card.message_path(tmp_path, "ack", 1, "begin").read_bytes() == before
    assert card.message_path(tmp_path, "ack", 1, "complete").is_file()


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_mutable_receipt_retries_only_transient_windows_publication_errors(monkeypatch, tmp_path, winerror):
    original = Path.replace
    attempts, delays = [], []
    def replace_file(path, target):
        attempts.append(True)
        if len(attempts) <= 2:
            exc = PermissionError("injected Windows sharing conflict")
            exc.winerror = winerror
            raise exc
        return original(path, target)
    monkeypatch.setattr(Path, "replace", replace_file)
    monkeypatch.setattr(card.time, "sleep", delays.append)
    card.atomic_json(tmp_path / "receipt.json", {"kept": True})
    assert len(attempts) == 3 and delays == [.01, .02]
    assert card.read_json(tmp_path / "receipt.json") == {"kept": True}


def test_permanent_windows_publication_conflict_has_a_short_finite_budget(monkeypatch, tmp_path):
    attempts, delays = [], []
    def replace_file(path, target):
        attempts.append(True)
        exc = PermissionError("persistent Windows conflict")
        exc.winerror = 5
        raise exc
    monkeypatch.setattr(Path, "replace", replace_file)
    monkeypatch.setattr(card.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        card.atomic_json(tmp_path / "receipt.json", {"kept": True})
    assert len(attempts) == 8 and len(delays) == 7 and sum(delays) == pytest.approx(.28)
    assert not (tmp_path / "receipt.json").exists()


def test_other_publication_errors_fail_without_retry(monkeypatch, tmp_path):
    def replace_file(path, target):
        raise PermissionError("ordinary non-Windows permissions failure")
    monkeypatch.setattr(Path, "replace", replace_file)
    monkeypatch.setattr(card.time, "sleep", lambda _: pytest.fail("unrelated I/O errors are not retried"))
    with pytest.raises(PermissionError, match="non-Windows"):
        card.atomic_json(tmp_path / "receipt.json", {})


@pytest.mark.parametrize("winerror", [5, 32, 33, None])
def test_new_message_read_retries_windows_and_crt_sharing_errors(monkeypatch, tmp_path, winerror):
    path = tmp_path / "message.json"
    path.write_text('{"kept": true}', encoding="utf-8")
    original = Path.read_text
    attempts, delays = [], []
    def read_text(file, *args, **kwargs):
        attempts.append(True)
        if len(attempts) <= 2:
            exc = PermissionError(errno.EACCES, "injected transient Windows read sharing conflict")
            if winerror is not None:
                exc.winerror = winerror
            raise exc
        return original(file, *args, **kwargs)
    monkeypatch.setattr(card.sys, "platform", "win32")
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(card.time, "sleep", delays.append)
    assert card.read_json(path) == {"kept": True}
    assert len(attempts) == 3 and delays == [.01, .02]


@pytest.mark.parametrize("winerror", [5, 32, 33, None])
def test_persistent_windows_message_read_error_stops_within_budget(monkeypatch, tmp_path, winerror):
    attempts, delays = [], []
    def read_text(file, *args, **kwargs):
        attempts.append(True)
        exc = PermissionError(errno.EACCES, "persistent Windows message read conflict")
        if winerror is not None:
            exc.winerror = winerror
        raise exc
    monkeypatch.setattr(card.sys, "platform", "win32")
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(card.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="persistent Windows"):
        card.read_json(tmp_path / "message.json")
    assert len(attempts) == 8 and len(delays) == 7 and sum(delays) == pytest.approx(.28)


@pytest.mark.parametrize("failure,platform", [(PermissionError(errno.EACCES, "POSIX access denied"), "linux"),
    (OSError(errno.EIO, "disk read failed"), "win32"), (FileNotFoundError(), "win32")])
def test_unrelated_and_missing_message_reads_are_never_retried(monkeypatch, tmp_path, failure, platform):
    attempts = []
    def read_text(file, *args, **kwargs):
        attempts.append(True)
        raise failure
    monkeypatch.setattr(card.sys, "platform", platform)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(card.time, "sleep", lambda _: pytest.fail("unrelated or missing read must not retry"))
    if isinstance(failure, FileNotFoundError):
        assert card.read_json(tmp_path / "absent.json") is None
    else:
        with pytest.raises(type(failure)):
            card.read_json(tmp_path / "message.json")
    assert len(attempts) == 1


def test_malformed_message_does_not_become_a_missing_or_retried_read(monkeypatch, tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text('{"unfinished":', encoding="utf-8")
    monkeypatch.setattr(card.time, "sleep", lambda _: pytest.fail("JSON syntax errors must stay fatal"))
    with pytest.raises(json.JSONDecodeError):
        card.read_json(path)


@pytest.mark.skipif(sys.platform != "win32", reason="real Windows delete-sharing regression")
def test_windows_real_process_handshake_survives_readers_holding_prior_acknowledgements(tmp_path):
    # An open Windows handle deliberately omits FILE_SHARE_DELETE. Replacing
    # the old mutable ack.json would fail deterministically until this child
    # saw the completion ack. Immutable per-action publication avoids the
    # cycle, without any model imports, GPU counters or OS power requests.
    script = textwrap.dedent('''
        import ctypes
        from ctypes import wintypes
        import errno
        import json
        from pathlib import Path
        import sys
        from tools import anima_refine_card as card
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        output = Path(sys.argv[1])
        original_read = Path.read_text
        injected_reads = 0
        def transient_crt_read(path, *args, **kwargs):
            global injected_reads
            if path.name == "ack-0009-complete.json" and path.exists() and injected_reads == 0:
                injected_reads += 1
                raise PermissionError(errno.EACCES, "deterministic CRT read race at the previously failed handoff")
            return original_read(path, *args, **kwargs)
        Path.read_text = transient_crt_read
        hand = card.Handshake(output)
        phases = [("load", "load"), *[("case", row["id"]) for row in card.pack.cases()
                  if not row.get("optional")], ("cleanup", "cleanup")]
        locks = 0
        for kind, name in phases:
            assert hand.begin(kind, name)
            ack = card.message_path(output, "ack", hand.sequence, "begin")
            handle = kernel.CreateFileW(str(ack), 0x80000000, 1 | 2, None, 3, 0, None)
            assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()
            try:
                try:
                    ack.replace(ack.with_suffix(".must-not-replace"))
                except PermissionError as exc:
                    assert exc.winerror in (5, 32, 33)
                else:
                    raise AssertionError("test did not establish a real Windows delete-sharing lock")
                locks += 1
                result = {}
                if kind == "case":
                    result["case"] = {"status": "cancelled" if name == "cancel-04" else "completed",
                                      "metrics": {"elapsed_seconds": .01}}
                hand.finish(ok=True, **result)
            finally:
                assert kernel.CloseHandle(handle)
        assert "torch" not in sys.modules and "backend" not in sys.modules
        print(json.dumps({"real_locked_handoffs": locks, "injected_crt_read_recovered": injected_reads}))
    ''')
    stdout_path = tmp_path / "handshake-child.stdout.log"
    stderr_path = tmp_path / "handshake-child.stderr.log"
    stdout_log = stdout_path.open("w", encoding="utf-8")
    stderr_log = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], cwd=card.ROOT,
                            stdout=stdout_log, stderr=stderr_log, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    class ProcessHost:
        sources = {"shared": "pdh", "card": "CPU fixture, no GPU access"}
        def tree(self, pid):
            return [pid] if proc.poll() is None else None
        def sample(self, pids):
            return GOOD
        def kill(self, pid):
            proc.kill()
    try:
        code = card.supervise(tmp_path, LIMITS, host=ProcessHost(), proc=proc)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        stdout_log.close()
        stderr_log.close()
    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    assert code == 0, f"retained child logs: {stdout_path}, {stderr_path}\n{stderr}"
    assert proc.returncode == 0, f"retained child stderr: {stderr_path}\n{stderr}"
    assert json.loads(stdout) == {"real_locked_handoffs": 13, "injected_crt_read_recovered": 1}
