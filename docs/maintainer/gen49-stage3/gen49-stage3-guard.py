"""Independent GEN#49 Windows resource supervisor for one offline owned process."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(ROOT))
from tools import card_guard as card  # noqa: E402
from tools import kontext_trial_guard as instrumentation  # noqa: E402
from tools.anima_refine_card import atomic_json  # noqa: E402

PYTHON = Path(sys.executable)
PERIOD, HEARTBEAT = 0.25, 2.0
IDLE, FREE, RSS, PRIVATE = 2500.0, 24.0, 16.0, 16.0
PHASE_CAP, SESSION_CAP, UNLOAD_TIMEOUT = 180.0, 1200.0, 30.0
LIMITS = dict(
    period_s=PERIOD,
    heartbeat_s=HEARTBEAT,
    cold_free_ram_gib=FREE,
    cold_free_commit_gib=FREE,
    load_free_ram_gib=FREE,
    load_commit_before_owned_gib=FREE,
    idle_card_mib=IDLE,
    owned_rss_cap_gib=RSS,
    owned_private_cap_gib=PRIVATE,
    phase_cap_s=PHASE_CAP,
    session_cap_s=SESSION_CAP,
    unload_wait_cap_s=UNLOAD_TIMEOUT,
    live=asdict(card.LIMITS),
)


def admit(reading, *, cold):
    instrumentation.validate(reading)
    sample = reading.sample
    credit = 0.0 if cold else reading.private_gib
    reasons = []
    if sample.available_gib < FREE:
        reasons.append("free RAM below 24 GiB")
    if sample.commit_headroom_gib + credit < FREE:
        reasons.append("free commit plus owned private credit below 24 GiB")
    if sample.card_mib > IDLE:
        reasons.append("whole card above 2500 MiB")
    return dict(
        admitted=not reasons,
        reasons=reasons,
        owned_private_credit_gib=credit,
        required_free_commit_gib=max(card.LIMITS.commit_floor_gib, FREE - credit),
    )


def check_live(reading, baseline):
    instrumentation.validate(reading)
    answer = card.decide(reading.sample, baseline)
    if answer.verdict != "ok":
        raise RuntimeError(f"live {answer.why}: {answer.reading} crosses {answer.limit}")
    if reading.sample.tree_gib > RSS or reading.private_gib > PRIVATE:
        raise RuntimeError("owned process tree RSS/private cap crossed")
    if not 0 <= time.monotonic() - reading.card_at <= PERIOD * 2:
        raise RuntimeError("whole card sample stale")


def spawn_bound(child, attempt, log, job):
    # Attach before Python's Windows venv redirector can launch the interpreter.
    # All descendants then inherit this private job; an abrupt supervisor exit kills it.
    env = {k: v for k, v in os.environ.items() if not k.startswith("AURADREAM_")}
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(ROOT),
    )
    process = subprocess.Popen(
        [str(PYTHON), str(child), str(attempt)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=ROOT,
        creationflags=subprocess.CREATE_NO_WINDOW | 0x4,
    )  # CREATE_SUSPENDED
    try:
        if not job.assign(process.pid):
            raise RuntimeError("owned suspended process could not enter kill-on-close job")
        resume = ctypes.WinDLL("ntdll").NtResumeProcess
        resume.argtypes, resume.restype = [wintypes.HANDLE], ctypes.c_long
        if resume(int(process._handle)) != 0:
            raise RuntimeError("owned job process could not start")
    except BaseException:
        process.kill()
        process.wait(5)
        raise
    return process


def run(attempt, child):
    attempt.mkdir(parents=True, exist_ok=True)
    if any(attempt.iterdir()):
        raise ValueError("attempt directory must be empty")
    raw = (attempt / "guard.jsonl").open("x", encoding="utf-8")
    readings = []
    result = dict(
        status="incomplete",
        guard_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        child_sha256=hashlib.sha256(child.read_bytes()).hexdigest(),
        limits=LIMITS,
        child_pid=None,
        child_exit=None,
        load_admitted=False,
        unload_observed=False,
    )
    job = reader = proc = guard_thread = child_log = None
    monitor_stop = threading.Event()
    started = beat = time.monotonic()
    phase_since = started
    phase = "cold"
    watchdog_fault = []

    def row(event, **fields):
        raw.write(
            json.dumps(
                dict(event=event, wall=time.time(), monotonic=time.monotonic(), phase=phase, **fields), allow_nan=False
            )
            + "\n"
        )
        raw.flush()

    def watchdog():
        while not monitor_stop.wait(0.1):
            now = time.monotonic()
            if now - beat > HEARTBEAT or now - started > SESSION_CAP or now - phase_since > PHASE_CAP:
                watchdog_fault.append(f"watchdog: heartbeat, phase or session cap during {phase}")
                job.close()
                atomic_json(attempt / "guard-watchdog.json", {"reason": watchdog_fault[0], "wall": time.time()})
                return

    try:
        row("frozen", **result)
        job = card.KillOnCloseJob.open()
        if job is None:
            raise RuntimeError("Windows kill-on-close job unavailable")
        reader = instrumentation.WindowsReader()
        first = reader.read(None)
        result["cold_sample"] = asdict(first)
        instrumentation.validate(first)
        baseline = card.Baseline(first.sample.adapter_shared_mib, first.sample.process_shared_mib)
        check_live(first, baseline)
        cold = admit(first, cold=True)
        result["cold"] = cold
        row("cold", reading=asdict(first), **cold)
        if not cold["admitted"]:
            raise RuntimeError("cold hold: " + ", ".join(cold["reasons"]))
        phase = "bootstrap"
        child_log = (attempt / "child.log").open("xb")
        proc = spawn_bound(child, attempt, child_log, job)
        result["child_pid"] = proc.pid
        phase_since = beat = time.monotonic()
        guard_thread = threading.Thread(target=watchdog, daemon=True)
        guard_thread.start()
        phase_token = None
        load_seen = unload_seen = None
        post_unload = []
        next_due = time.monotonic()
        longest_gap = 0.0
        while proc.poll() is None:
            if watchdog_fault:
                raise RuntimeError(watchdog_fault[0])
            now = time.monotonic()
            if load_seen is None and (attempt / "load-request").exists():
                load_seen = now
            if unload_seen is None and (attempt / "unload-done").exists():
                unload_seen = now
            phase_file = attempt / "phase.json"
            if phase_file.exists():
                stamp = json.loads(phase_file.read_text(encoding="utf-8"))
                token = stamp["name"], float(stamp["started"])
                if token != phase_token:
                    age = time.time() - token[1]
                    if not (0 <= age <= PHASE_CAP) or not isinstance(token[0], str) or not token[0]:
                        raise RuntimeError("invalid/stale phase receipt")
                    phase_token, phase, phase_since = token, token[0], now - age
                    row("phase")
            try:
                reading = reader.read(proc.pid)
            except Exception:
                if proc.poll() is not None and result["unload_observed"]:
                    break
                raise
            now = time.monotonic()
            gap = now - beat
            row("sample", gap_s=gap, **asdict(reading))
            if gap > HEARTBEAT or proc.pid not in reading.pids:
                raise RuntimeError("stale sample or owned root absent")
            check_live(reading, baseline)
            readings.append(reading)
            longest_gap = max(longest_gap, gap)
            beat = now
            if len(readings) == 1:
                atomic_json(attempt / "guard-admitted", {"guarded_pid": proc.pid})
            if load_seen is not None and not result["load_admitted"] and reading.card_at >= load_seen:
                decision = admit(reading, cold=False)
                result["load"] = dict(**decision, reading=asdict(reading))
                row("admit_load", **decision)
                if not decision["admitted"]:
                    raise RuntimeError("load hold: " + ", ".join(decision["reasons"]))
                atomic_json(attempt / "load-admitted", {"guarded_pid": proc.pid})
                result["load_admitted"] = True
            if unload_seen is not None and not result["unload_observed"]:
                if not result["load_admitted"]:
                    raise RuntimeError("premature unload marker")
                if now - unload_seen > UNLOAD_TIMEOUT:
                    raise RuntimeError("post-unload whole-card residency did not settle within timeout")
                if reading.card_at >= unload_seen and (not post_unload or reading.card_at > post_unload[-1].card_at):
                    if reading.sample.card_mib <= IDLE:
                        post_unload.append(reading)
                    else:
                        post_unload.clear()
                    if len(post_unload) == 3:
                        result["post_unload"] = [asdict(r) for r in post_unload]
                        row("unload_observed", samples=result["post_unload"])
                        atomic_json(attempt / "unload-observed", {"samples": 3})
                        result["unload_observed"] = True
            next_due = max(next_due + PERIOD, time.monotonic())
            time.sleep(max(0.0, next_due - time.monotonic()))
        if watchdog_fault:
            raise RuntimeError(watchdog_fault[0])
        if proc.returncode or not result["unload_observed"]:
            raise RuntimeError(f"child exit {proc.returncode}, unload observed={result['unload_observed']}")
        result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "held"
        result["reason"] = (watchdog_fault or [f"{type(exc).__name__}: {exc}"])[0]
    finally:
        monitor_stop.set()
        if guard_thread:
            guard_thread.join(3.0)
        if job:
            job.close()
        if proc:
            result["child_exit"] = proc.wait(timeout=15)
        if reader:
            reader.close()
        if child_log:
            child_log.close()
        if readings:
            result["metrics"] = instrumentation.metrics(readings, baseline)
            result["longest_gap_s"] = longest_gap
        result["elapsed_s"] = time.monotonic() - started
        row("end", **result)
        raw.close()
        atomic_json(attempt / "guard-result.json", result)
    print(json.dumps(result))
    return 0 if result["status"] == "complete" else 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("child", type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.attempt.resolve(), args.child.resolve(strict=True)))
