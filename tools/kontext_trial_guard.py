"""Out-of-process INP#13 controller. Only the isolated trial imports this module.

The browser server waits at each actual load/edit entry for a one-use permit.
The controller owns the guard log, permission decisions and tree stop callback.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Protocol

from tools import card_guard as card
from tools.anima_refine_card import atomic_json
from tools.kontext_trial_runtime import read_mailbox

PERIOD = .25
HEARTBEAT = 2.
RAM_GIB = 45.
IDLE_MIB = 2000.
# Full precision retained in instruction-acceptance-2026-11-18.json.
PRIVATE_PEAK_GIB = 46.54413986206055
COMMIT_GIB = PRIVATE_PEAK_GIB + card.LIMITS.commit_floor_gib + 6.
SMALL_CASES = {"control-512": (512, 512), "16x9": (512, 288), "9x16": (288, 512),
               "4x3": (512, 384), "3x4": (384, 512), "3x2": (576, 384), "2x3": (384, 576)}


class GuardHeld(RuntimeError):
    """Guard stopped or refused this trial; the receipt contains the reason."""


@dataclass(frozen=True)
class Reading:
    sample: card.Sample
    private_gib: float
    pids: list[int]
    card_at: float  # monotonic timestamp when the nvidia-smi line arrived


class Reader(Protocol):
    def read(self, pid: int | None) -> Reading: ...
    def close(self) -> None: ...


class WindowsReader:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise GuardHeld("the real guard requires Windows")
        self.host = card.LocalHost(PERIOD)
        if self.host.shared is None:
            self.host.close()
            raise GuardHeld("held Windows PDH shared memory counters unavailable")
        # Keep the actual whole-card acquisition time for phase/permit freshness.
        # Poll GPU 0 a little faster than the controller to avoid sample aliasing.
        def stamped(line: str) -> tuple[float, float, float] | None:
            pair = card.parse_nvidia_smi(line)
            return None if pair is None else (*pair, time.monotonic())
        self.stream = card.nvidia_smi_stream(PERIOD / 2)
        self.stream.parse = stamped

    def read(self, pid: int | None) -> Reading:
        import psutil
        pids = self.host.tree(pid) if pid is not None else []
        if pids is None:
            raise GuardHeld("owned server process tree unavailable")
        gpu = self.stream.latest(PERIOD * 2, wait=8. if pid is None else 0.)
        if gpu is None:
            raise GuardHeld("fresh whole-card reading unavailable")
        assert self.host.shared is not None
        counters = self.host.shared.read()
        if counters is None:
            raise GuardHeld("Windows PDH collection unreadable")
        adapter, process = card.shared_usage(counters, pids)
        rss, private = 0, 0
        for child in pids:
            memory: Any = psutil.Process(child).memory_info()
            rss += memory.rss
            private += memory.private
        sample = card.Sample(card_mib=gpu[0], card_total_mib=gpu[1], adapter_shared_mib=adapter,
                             process_shared_mib=process if pids else 0., tree_gib=rss / 2**30,
                             commit_headroom_gib=card.commit_headroom_gib(),
                             available_gib=psutil.virtual_memory().available / 2**30)
        return Reading(sample, private / 2**30, pids, gpu[2])

    def close(self) -> None:
        self.stream.close()
        self.host.close()


class RehearsalReader:
    """Synthetic guard readings, never device evidence."""
    def read(self, pid: int | None) -> Reading:
        return Reading(card.Sample(card_mib=1300., card_total_mib=16303., adapter_shared_mib=55.,
                                   process_shared_mib=0., available_gib=60., commit_headroom_gib=70.,
                                   tree_gib=.5 if pid else 0.), .5 if pid else 0.,
                       [] if pid is None else [pid], time.monotonic())

    def close(self) -> None:
        pass


def headroom(reading: Reading, *, cold: bool) -> dict[str, Any]:
    needed = COMMIT_GIB - (0. if cold else reading.private_gib)
    s = reading.sample
    reasons = []
    if s.card_mib is None or s.card_mib > IDLE_MIB:
        reasons.append("card above idle limit")
    if s.available_gib is None or s.available_gib < RAM_GIB:
        reasons.append("free RAM below load requirement")
    if s.commit_headroom_gib is None or s.commit_headroom_gib < needed:
        reasons.append("free commit below measured load requirement")
    return {"stage": "cold" if cold else "load", "required_commit_gib": needed,
            "required_free_ram_gib": RAM_GIB, "idle_limit_mib": IDLE_MIB,
            "owned_private_gib": reading.private_gib, "sample": asdict(s),
            "admitted": not reasons, "reasons": reasons}


def validate(reading: Reading) -> None:
    for key, value in {**asdict(reading.sample), "private_gib": reading.private_gib}.items():
        if value is None or not math.isfinite(value) or value < 0:
            raise GuardHeld(f"unreadable guard counter: {key}")


def metrics(readings: list[Reading], baseline: card.Baseline) -> dict[str, Any]:
    samples = [r.sample for r in readings]
    def vals(field: str) -> list[float]:
        return [float(getattr(s, field)) for s in samples]
    assert baseline.adapter_shared_mib is not None and baseline.process_shared_mib is not None
    return {"samples": len(samples), "peak_card_mib": max(vals("card_mib")),
            "final_card_mib": samples[-1].card_mib,
            "peak_adapter_shared_rise_mib": max(vals("adapter_shared_mib")) - baseline.adapter_shared_mib,
            "peak_process_shared_rise_mib": max(vals("process_shared_mib")) - baseline.process_shared_mib,
            "peak_tree_rss_gib": max(vals("tree_gib")), "peak_tree_private_gib": max(r.private_gib for r in readings),
            "min_available_gib": min(vals("available_gib")), "min_commit_headroom_gib": min(vals("commit_headroom_gib"))}


class TrialGuard:
    def __init__(self, output: Path, *, fake: bool = False, reader: Reader | None = None) -> None:
        self.output = output
        self.fake = fake
        self.reader = reader
        self.cv = threading.Condition()
        self.trip_lock = threading.Lock()
        self.stop = threading.Event()
        self.pid: int | None = None
        self.kill: Callable[[], None] | None = None
        self.reason: str | None = None
        self.threads: list[threading.Thread] = []
        self.phase = "startup"
        self.phase_at = time.monotonic()
        self.beat = self.phase_at
        self.baseline = card.Baseline()
        self.latest: Reading | None = None
        self.latest_started = 0.
        self.first_observed = 0.
        self.readings: list[Reading] = []
        self.completed: dict[str, list[Reading]] = {}
        self.phases: dict[str, Any] = {}
        self.expected: deque[dict[str, Any]] = deque()
        self.seen: set[str] = set()
        self.actions: list[dict[str, Any]] = []
        self.active_request: dict[str, Any] | None = None
        self.action_start = 0
        self.loaded = False
        self.admitted_large = False
        self.started_large = False
        self.replay_reference: tuple[float, float] | None = None
        self.recovered = False
        self.sequence = 0
        self.longest_gap = 0.
        self.session_peak = 0.
        self.raw: Any = None

    def _row(self, event: str, **fields: Any) -> None:
        self.raw.write(json.dumps({"event": event, "time": time.time(), "phase": self.phase, **fields},
                                  allow_nan=False) + "\n")
        self.raw.flush()

    def __enter__(self) -> TrialGuard:
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.iterdir()):
            raise ValueError(f"trial guard directory must be fresh: {self.output}")
        self.raw = (self.output / "guard.jsonl").open("x", encoding="utf-8")
        try:
            self._row("start", mode="fake" if self.fake else "real", period_s=PERIOD, limits=asdict(card.LIMITS),
                      measured_private_gib=PRIVATE_PEAK_GIB, cold_commit_gib=COMMIT_GIB)
            if self.reader is None:
                self.reader = RehearsalReader() if self.fake else WindowsReader()
            first = self.reader.read(None)
            self._row("cold_sample", reading=asdict(first))
            validate(first)
            self.baseline = card.Baseline(first.sample.adapter_shared_mib, first.sample.process_shared_mib)
            decision = headroom(first, cold=True)
            self._row("preflight", **decision)
            if not decision["admitted"]:
                raise GuardHeld(", ".join(decision["reasons"]))
            return self
        except BaseException as exc:
            self.reason = str(exc)
            self._close()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise GuardHeld(str(exc)) from exc

    def attach(self, pid: int, kill: Callable[[], None]) -> None:
        if self.pid is not None or pid <= 0:
            raise ValueError("expected one owned server root")
        self.pid, self.kill = pid, kill
        self.phase_at = self.beat = time.monotonic()
        self.threads = [threading.Thread(target=fn, name=name, daemon=True) for fn, name in
                        [(self._monitor, "Kontext guard heartbeat"), (self._sample, "Kontext guard sampler")]]
        for thread in self.threads:
            thread.start()
        self._fresh()

    def _trip(self, reason: str) -> None:
        # The sampler might be stalled writing a receipt while holding cv.
        # Killing the owned job must not wait on any sampler-held lock.
        with self.trip_lock:
            if self.reason is not None:
                return
            self.reason = reason
            self.stop.set()
        try:
            if self.kill is not None:
                self.kill()
        finally:
            atomic_json(self.output / "stopped.json", {"reason": reason, "phase": self.phase,
                                                       "seq": self.sequence, "server_pid": self.pid})

    def _monitor(self) -> None:
        while not self.stop.wait(PERIOD):
            if time.monotonic() - self.beat > HEARTBEAT:
                self._trip("guard sampling heartbeat exceeded two seconds")
                return

    def _sample(self) -> None:
        assert self.reader is not None
        due = time.monotonic()
        waiting_nonce: str | None = None
        requested_at = due
        try:
            while not self.stop.wait(max(0., due - time.monotonic())):
                # Read requests *before* sampling so a permit cannot use old headroom.
                request = read_mailbox(self.output / "request.json")
                started = time.monotonic()
                if request is not None and request.get("nonce") != waiting_nonce:
                    waiting_nonce = request.get("nonce")
                    requested_at = started
                r = self.reader.read(self.pid)
                now = time.monotonic()
                with self.cv:
                    if self.stop.is_set():
                        return
                    self.sequence += 1
                    self.longest_gap = max(self.longest_gap, now - self.beat)
                    self._row("sample", seq=self.sequence, sampled_at=now, started_at=started, **asdict(r))
                    validate(r)
                    if self.pid not in r.pids:
                        raise GuardHeld("sample does not cover the owned server root")
                    if now - self.beat > HEARTBEAT or now - r.card_at > PERIOD * 2:
                        raise GuardHeld("stale guard sample")
                    self.beat = now
                    answer = card.decide(r.sample, self.baseline)
                    self._row("verdict", seq=self.sequence, **asdict(answer))
                    if answer.verdict != "ok":
                        raise GuardHeld(f"{answer.why}: reading {answer.reading} crosses {answer.limit}")
                    self.latest = r
                    self.latest_started = started
                    if self.sequence == 1:
                        self.first_observed = now
                    assert r.sample.card_mib is not None
                    self.session_peak = max(self.session_peak, r.sample.card_mib)
                    if started >= self.phase_at and r.card_at >= self.phase_at:
                        self.readings.append(r)
                        if request is not None and r.card_at >= requested_at:
                            self._permit(request, r)
                    self.cv.notify_all()
                due = max(due + PERIOD, now)
        except Exception as exc:
            self._trip(str(exc))

    def _permit(self, request: dict[str, Any], r: Reading) -> None:
        nonce = request.get("nonce")
        if not isinstance(nonce, str) or not nonce or request.get("pid") != self.pid:
            raise GuardHeld("invalid candidate action identity")
        if nonce in self.seen:  # The last atomic request stays until the next one.
            return
        action = {k: v for k, v in request.items() if k not in ("pid", "nonce")}
        if not self.expected or self.expected[0] != action:
            raise GuardHeld(f"unexpected candidate action for phase {self.phase}: {action}")
        if action["action"] == "load":
            if self.loaded or self.phase != "load":
                raise GuardHeld("second or out-of-phase load")
            decision = headroom(r, cold=False)
            self._row("admit_load", seq=self.sequence, **decision)
            if not decision["admitted"]:
                raise GuardHeld(", ".join(decision["reasons"]))
            self.loaded = True
        elif action["action"] == "edit":
            if not self.loaded or (action["recipe"] == "kontext-1024-v1" and not self.admitted_large):
                raise GuardHeld("unadmitted edit")
            if action["recipe"] == "kontext-1024-v1" and not self.started_large:
                # The browser may have opened or consumed memory after project().
                if self.session_peak >= card.LIMITS.refuse_at_peak_mib:
                    raise GuardHeld(f"pre-edit whole-card peak {self.session_peak} MiB prevents larger call")
                if self.replay_reference is not None:
                    peak, prior_loaded = self.replay_reference
                    assert r.sample.card_mib is not None
                    projected = peak + max(0., r.sample.card_mib - prior_loaded)
                    self._row("replay_entry_projection", seq=self.sequence, projected_mib=projected)
                    if projected > card.LIMITS.project_mib:
                        raise GuardHeld(f"fresh replay entry projects {projected} MiB")
                self.started_large = True
        else:
            raise GuardHeld("unknown candidate action")
        self.expected.popleft()
        permit = {"pid": self.pid, "nonce": nonce, "action": action["action"], "approved": True}
        self.seen.add(nonce)
        self._finish_action()
        self.active_request = request
        self.action_start = len(self.readings) - 1
        self._row("permit", seq=self.sequence, request=request, permit=permit)
        self.check()
        atomic_json(self.output / "permit.json", permit)

    def _finish_action(self) -> None:
        if self.active_request is None:
            return
        measured = metrics(self.readings[self.action_start:], self.baseline)
        self.actions.append({"request": self.active_request, "phase": self.phase, **measured})
        self._row("action_window_end", request=self.active_request, **measured)
        self.active_request = None

    def check(self) -> None:
        if self.reason is not None:
            raise GuardHeld(self.reason)

    def _fresh(self) -> Reading:
        mark = time.monotonic()
        with self.cv:
            if self.pid is None:
                raise RuntimeError("attach the server before requesting a guard reading")
            while self.reason is None:
                if self.latest is not None and self.latest_started >= mark and self.latest.card_at >= mark:
                    return self.latest
                self.cv.wait(PERIOD)
            raise GuardHeld(self.reason)

    def start_phase(self, name: str, expected: list[dict[str, Any]] | None = None) -> None:
        if name in self.completed or self.phase not in ("startup", "idle"):
            raise ValueError(f"unexpected phase {name} after {self.phase}")
        with self.cv:
            self.check()
            self.phase, self.phase_at = name, time.monotonic()
            self.readings = []
            self.expected = deque([{"action": "load"}] if expected is None and name == "load" else expected or [])
            self._row("phase_start", expected=list(self.expected))
        self._fresh()

    def end_phase(self, name: str) -> dict[str, Any]:
        if self.phase != name:
            raise ValueError(f"expected to finish {self.phase}, got {name}")
        self._fresh()
        with self.cv:
            self.check()
            if self.expected:
                raise GuardHeld(f"{name}: candidate actions did not reach guarded entry: {list(self.expected)}")
            self._finish_action()
            self.completed[name] = self.readings
            result = metrics(self.readings, self.baseline)
            self.phases[name] = result
            self._row("phase_end", **result)
            self.phase, self.phase_at, self.readings = "idle", time.monotonic(), []
            return result

    def project(self) -> dict[str, Any]:
        current = self._fresh()
        completed = [(card.rung_x(*size), [r.sample for r in self.completed.get(name, [])])
                     for name, size in SMALL_CASES.items()]
        decision = card.plan_sampled_windows_rung(completed, card.rung_x(1024, 1024),
                                                  current=current.sample, baseline=self.baseline)
        if decision.verdict == "ok":
            for name, (width, height) in SMALL_CASES.items():
                requests = [a["request"] for a in self.actions if a["phase"] == name]
                if (len(requests) != 1 or requests[0].get("action") != "edit"
                        or requests[0].get("recipe") != "kontext-v1" or requests[0].get("seed") != 12345
                        or (requests[0].get("width"), requests[0].get("height")) != (width, height)):
                    decision = card.Answer("refuse-projected", measured=decision.measured,
                                           reason=f"missing matching completed edit for {name}")
                    break
        if decision.verdict == "ok" and self.session_peak >= card.LIMITS.refuse_at_peak_mib:
            decision = card.Answer("refuse-projected", measured=decision.measured,
                                   reason=f"earlier whole-card peak {self.session_peak} MiB")
        result = asdict(decision)
        with self.cv:
            self.check()
            self.admitted_large = decision.verdict == "ok"
            self._row("projection", **result)
            atomic_json(self.output / "projection.json", result)
        return result

    def replay_projection(self, prior_guard_dir: Path) -> dict[str, Any]:
        current = self._fresh()
        prior = json.loads((prior_guard_dir / "summary.json").read_text(encoding="utf-8"))
        previous = json.loads((prior_guard_dir / "projection.json").read_text(encoding="utf-8"))
        if (prior["status"] != "recovered" or prior["fake"] != self.fake or previous["verdict"] != "ok"
                or "first-1024" not in prior["phases"] or not self.loaded):
            raise GuardHeld("prior session does not establish completed guarded 1024-square evidence")
        peak = prior["phases"]["first-1024"]["peak_card_mib"]
        assert current.sample.card_mib is not None
        delta = current.sample.card_mib - prior["phases"]["load"]["final_card_mib"]
        projected = peak + max(0., delta)
        self.admitted_large = (max(peak, self.session_peak) < card.LIMITS.refuse_at_peak_mib
                               and projected <= card.LIMITS.project_mib)
        self.replay_reference = (peak, prior["phases"]["load"]["final_card_mib"])
        result = {"verdict": "ok" if self.admitted_large else "refuse-projected", "projected_mib": projected,
                  "prior_peak_mib": peak, "loaded_baseline_delta_mib": delta}
        with self.cv:
            self.check()
            self._row("replay_projection", **result)
            atomic_json(self.output / "projection.json", result)
        return result

    def observe_unload(self, accepted: bool, ready: bool) -> dict[str, Any]:
        if "unload" not in self.completed or accepted is not True or ready is not False:
            raise GuardHeld(f"actual try_unload was not accepted/unready: {accepted=}, {ready=}")
        self.start_phase("post-unload")
        # Three independently acquired samples after the observed try_unload.
        self._fresh()
        result = self.end_phase("post-unload")
        if result["final_card_mib"] > IDLE_MIB:
            raise GuardHeld("post-unload whole-card reading remains above idle limit")
        self.recovered = True
        return {**result, "accepted": accepted, "ready": ready}

    def __exit__(self, kind: Any, error: BaseException | None, tb: Any) -> None:
        if error is not None:
            self._trip(str(error))
        try:
            self._close()
        finally:
            if error is None:
                self.check()

    def _close(self) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=HEARTBEAT + 1.)
        if self.reader is not None and not any(t.is_alive() for t in self.threads):
            self.reader.close()
        mean_period = (self.beat - self.first_observed) / (self.sequence - 1) if self.sequence > 1 else None
        summary = {"status": "held" if self.reason is not None else "recovered" if self.recovered else "incomplete",
                   "reason": self.reason, "fake": self.fake, "server_pid": self.pid, "period_s": PERIOD,
                   "achieved_period_s": mean_period, "longest_gap_s": self.longest_gap,
                   "peak_card_mib": self.session_peak, "samples": self.sequence, "phases": self.phases,
                   "actions": self.actions}
        self._row("summary", **summary)
        atomic_json(self.output / "summary.json", summary)
        self.raw.close()
