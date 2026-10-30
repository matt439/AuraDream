"""The card session's guard (`DEV#20`): whole card, shared GPU memory and RAM.

    uv run python tools/card_guard.py watch --watch-pid PID [--out rows.jsonl] [--need-ram-gib N]
    uv run python tools/card_guard.py plan --points "[[1.05, 9815], [2.07, 10247]]" --size 2048x2048

docs/plan-card-harness.md, "Card guard". Two halves:

- **The decision** (`decide`, `plan_rung`) is pure: no I/O, no clock, no
  subprocess, nothing from `backend`, torch or diffusers. An in-process
  harness imports it and answers its own `should_cancel` from it.
- **The sampler and the watcher** read nvidia-smi, the Windows GPU Adapter
  Memory and GPU Process Memory shared counters (`typeperf`), free RAM
  (`psutil`) and commit headroom (`GetPerformanceInfo`), write JSONL rows,
  and on a trip kill the watched process tree (children in reverse, then
  the root) and exit with `card4.py`'s code. The watcher never starts,
  loads or cancels anything through the app's API.

Exit codes: 0 ok, 2 refused input, 3 card or shared trip, 4 RAM or commit
trip, 5 refused rung or preflight, 7 the card busy before the start.

`LIMITS` below is the one place the defaults live; a session overrides any
of them with a flag or `--limits FILE.json`, and the limits in force head
the first row. The guard never lowers a limit by itself. Reading the
counters starts no GPU work, and a CPU or fake run of the guard is never a
card measurement.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Protocol, TextIO


# --- Limits in one block ----------------------------------------------------
# Every default the guard uses, and only here. Comparisons are strict, as the
# past guards' were: a reading equal to a limit does not trip.


@dataclass(frozen=True)
class Limits:
    # verified.md, "Behaviour" and "Inpaint" (the ERNIE, refine, WAI and
    # inpaint watchdogs): "a stop at card 15,500 MiB".
    card_mib: float = 15500.0
    # verified.md, "Checkpoints" (ERNIE's ladder): "projection (17,354 MiB
    # against a 15,200 limit)"; `ernie_card.py`'s line through the last two.
    project_mib: float = 15200.0
    # `ernie_card.py`: refuse a rung after an earlier one peaked at 15,000 MiB.
    refuse_at_peak_mib: float = 15000.0
    # verified.md, "RAM at the load, the fixed loader (2026-10-27)": "a 1 GiB
    # rise in shared GPU memory" (the adapter, over its baseline).
    adapter_shared_rise_mib: float = 1024.0
    # verified.md, "Peak per image and Then refine, WAI v17 (2026-10-25)":
    # "the server's shared memory +512".
    process_shared_rise_mib: float = 512.0
    # verified.md, "RAM at the load (2026-10-25)": "the harnesses' 4 GiB
    # free-RAM floor".
    ram_floor_gib: float = 4.0
    # `card4.py` / `card_refine.py`: 4 GiB of CommitLimit - CommitTotal.
    commit_floor_gib: float = 4.0
    # No default: a load run states its RAM cost (`--need-ram-gib`), which
    # depends on the checkpoint ("GPU OK is not RAM OK").
    need_ram_gib: float | None = None
    # No default: the whole-card reading at the start above which the card
    # counts as busy with something else (exit 7).
    busy_card_mib: float | None = None
    # Not from a verified.md row: after the first sample, how many samples in
    # a row may come back without a whole-card reading (nvidia-smi failing
    # or timing out) before the guard trips (exit 3). Watching blind under
    # pressure is when a spill is likely; one reading back resets the count.
    card_unread_samples: float = 3.0
    # Not limits: how near one a sample is written as a row whatever the
    # row period (MiB for the card and shared rises, GiB for RAM and commit).
    margin_mib: float = 500.0
    margin_gib: float = 1.0


LIMITS = Limits()

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_SPILL = 3
EXIT_RAM = 4
EXIT_PREFLIGHT = 5
EXIT_BUSY = 7


# --- The decision (pure) ----------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One reading of every counter; None where a counter was not read."""

    card_mib: float | None = None
    adapter_shared_mib: float | None = None
    process_shared_mib: float | None = None
    available_gib: float | None = None
    commit_headroom_gib: float | None = None
    tree_gib: float | None = None
    card_total_mib: float | None = None


@dataclass(frozen=True)
class Baseline:
    """The shared counters at the start; rises are measured from these."""

    adapter_shared_mib: float | None = None
    process_shared_mib: float | None = None


Verdict = Literal["ok", "trip", "refuse-projected"]
Why = Literal["card", "card_unread", "adapter_shared", "process_shared", "ram", "commit"]


@dataclass(frozen=True)
class Answer:
    verdict: Verdict
    why: Why | None = None
    reading: float | None = None
    limit: float | None = None
    projected_mib: int | None = None
    measured: list[list[float]] = field(default_factory=list)
    reason: str | None = None

    @property
    def exit_code(self) -> int:
        if self.verdict == "trip":
            return EXIT_RAM if self.why in ("ram", "commit") else EXIT_SPILL
        if self.verdict == "refuse-projected":
            return EXIT_PREFLIGHT
        return EXIT_OK


OK = Answer("ok")


def _rise(now: float | None, base: float | None) -> float | None:
    if now is None or base is None:
        return None
    return now - base


def rises(sample: Sample, baseline: Baseline) -> tuple[float | None, float | None]:
    """The adapter and process shared rises over the baseline, in MiB."""
    return (_rise(sample.adapter_shared_mib, baseline.adapter_shared_mib),
            _rise(sample.process_shared_mib, baseline.process_shared_mib))


def decide(sample: Sample, baseline: Baseline, limits: Limits = LIMITS) -> Answer:
    """`ok`, or `trip` with the first limit crossed: RAM, commit, card, shared."""
    if sample.available_gib is not None and sample.available_gib < limits.ram_floor_gib:
        return Answer("trip", "ram", sample.available_gib, limits.ram_floor_gib)
    if sample.commit_headroom_gib is not None and sample.commit_headroom_gib < limits.commit_floor_gib:
        return Answer("trip", "commit", sample.commit_headroom_gib, limits.commit_floor_gib)
    if sample.card_mib is not None and sample.card_mib > limits.card_mib:
        return Answer("trip", "card", sample.card_mib, limits.card_mib)
    adapter, process = rises(sample, baseline)
    if adapter is not None and adapter > limits.adapter_shared_rise_mib:
        return Answer("trip", "adapter_shared", adapter, limits.adapter_shared_rise_mib)
    if process is not None and process > limits.process_shared_rise_mib:
        return Answer("trip", "process_shared", process, limits.process_shared_rise_mib)
    return OK


def near(sample: Sample, baseline: Baseline, limits: Limits = LIMITS) -> bool:
    """Whether any reading is within the margin of its limit (rows only)."""
    adapter, process = rises(sample, baseline)
    mib = [(sample.card_mib, limits.card_mib), (adapter, limits.adapter_shared_rise_mib),
           (process, limits.process_shared_rise_mib)]
    gib = [(sample.available_gib, limits.ram_floor_gib),
           (sample.commit_headroom_gib, limits.commit_floor_gib)]
    return (any(v is not None and v > lim - limits.margin_mib for v, lim in mib)
            or any(v is not None and v < lim + limits.margin_gib for v, lim in gib))


def rung_x(width: int, height: int, batch: int = 1) -> float:
    """The projection's x-axis: megapixels times batch, to two places."""
    return round(width * height * batch / 1e6, 2)


def _number(value: float) -> str:
    return f"{value:g}" if value != int(value) else str(int(value))


def plan_rung(measured: list[list[float]], x: float, limits: Limits = LIMITS,
              tripped: bool = False) -> Answer:
    """Whether a rung at `x` may run, given the session's `[x, peak MiB]` points.

    `ernie_card.py`'s rule: refuse after any trip, when an earlier rung peaked
    at the refuse-at figure, or when the line through the last two points
    projects past the projection limit. Fewer than two points project nothing.
    """
    points = [[float(px), float(py)] for px, py in measured]
    if tripped:
        return Answer("refuse-projected", measured=points, reason="the session tripped")
    if points:
        peak = max(py for _, py in points)
        if peak >= limits.refuse_at_peak_mib:
            return Answer("refuse-projected", measured=points,
                          reason=f"an earlier rung peaked at {_number(peak)} MiB >= "
                                 f"{_number(limits.refuse_at_peak_mib)}")
    if len(points) < 2:
        return Answer("ok", measured=points)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    if x1 == x2:
        raise ValueError(f"the last two points share x {_number(x1)}: no line")
    projected = round(y2 + (y2 - y1) / (x2 - x1) * (x - x2))
    if projected > limits.project_mib:
        return Answer("refuse-projected", projected_mib=projected, measured=points,
                      reason=f"projected {projected} MiB > {_number(limits.project_mib)}")
    return Answer("ok", projected_mib=projected, measured=points)


# --- Limits in force --------------------------------------------------------


_FIELDS = {f.name for f in dataclasses.fields(Limits)}


def resolve_limits(file_values: dict[str, Any] | None, flags: dict[str, float | None]) -> Limits:
    """The defaults, then a JSON file's values, then the command line's."""
    values: dict[str, Any] = {}
    for source in (file_values or {}, flags):
        for name, value in source.items():
            if name not in _FIELDS:
                raise ValueError(f"unknown limit {name!r}")
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"limit {name} must be a finite number, not {value!r}")
            values[name] = float(value)
    return dataclasses.replace(LIMITS, **values)


# --- Counter readers (stubbed in tests) -------------------------------------


def parse_nvidia_smi(text: str) -> tuple[float, float] | None:
    """`memory.used, memory.total` of the first GPU, MiB."""
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return None
    return None


_ADAPTER = re.compile(r"\\GPU Adapter Memory\((luid_[^)]*)\)\\(Shared|Dedicated) Usage$", re.I)
_PROCESS = re.compile(r"\\GPU Process Memory\(pid_(\d+)_(luid_[^)]*)\)\\Shared Usage$", re.I)
_MIB = 1024 * 1024


def parse_typeperf(text: str, pids: list[int]) -> tuple[float | None, float | None]:
    """The adapter with the largest dedicated use: its shared MiB, and the shared
    MiB of `pids` on it (None with no pids). Counter values are bytes."""
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    header = next((r for r in rows if r and r[0].startswith("(PDH-CSV")), None)
    data = [r for r in rows if r is not header and len(r) == len(header or [])]
    if header is None or not data:
        return None, None
    values = data[-1]
    shared: dict[str, float] = {}
    dedicated: dict[str, float] = {}
    process: list[tuple[int, str, float]] = []
    for name, raw in zip(header[1:], values[1:]):
        try:
            value = float(raw)
        except ValueError:
            continue
        if m := _ADAPTER.search(name):
            (shared if m.group(2).lower() == "shared" else dedicated)[m.group(1).lower()] = value
        elif m := _PROCESS.search(name):
            process.append((int(m.group(1)), m.group(2).lower(), value))
    if not dedicated:
        return None, None
    luid = max(dedicated, key=lambda k: dedicated[k])
    adapter = shared.get(luid)
    adapter_mib = None if adapter is None else adapter / _MIB
    if not pids:
        return adapter_mib, None
    wanted = set(pids)
    total = sum(v for pid, on, v in process if pid in wanted and on == luid)
    return adapter_mib, total / _MIB


TYPEPERF_COUNTERS = ["\\GPU Adapter Memory(*)\\Shared Usage", "\\GPU Adapter Memory(*)\\Dedicated Usage",
                     "\\GPU Process Memory(*)\\Shared Usage"]


def _run(argv: list[str], timeout: float) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def commit_headroom_gib() -> float | None:
    """CommitLimit - CommitTotal through `GetPerformanceInfo`; None off Windows."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t),
                    ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                    ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
                    ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_size_t), ("HandleCount", wintypes.DWORD),
                    ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]

    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    try:
        psapi = ctypes.WinDLL("psapi")
        if not psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None
    except OSError:
        return None
    return (info.CommitLimit - info.CommitTotal) * info.PageSize / 2**30


class Host(Protocol):
    """What the watcher reads and kills; tests hand it a stub."""

    def tree(self, root: int) -> list[int] | None: ...
    def sample(self, pids: list[int]) -> Sample: ...
    def kill(self, root: int) -> None: ...


class LocalHost:
    """The real counters. psutil comes with accelerate (uv.lock); not declared."""

    def tree(self, root: int) -> list[int] | None:
        import psutil
        try:
            proc = psutil.Process(root)
            return [root] + [c.pid for c in proc.children(recursive=True)]
        except psutil.Error:
            return None

    def sample(self, pids: list[int]) -> Sample:
        import psutil
        smi = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits"], 15)
        card = parse_nvidia_smi(smi) if smi else None
        adapter = process = None
        if sys.platform == "win32":
            perf = _run(["typeperf", "-sc", "1", *TYPEPERF_COUNTERS], 30)
            if perf:
                adapter, process = parse_typeperf(perf, pids)
        tree = None
        if pids:
            rss = 0
            for pid in pids:
                try:
                    rss += psutil.Process(pid).memory_info().rss
                except psutil.Error:
                    pass
            tree = rss / 2**30
        return Sample(card_mib=None if card is None else card[0], card_total_mib=None if card is None else card[1],
                      adapter_shared_mib=adapter, process_shared_mib=process,
                      available_gib=psutil.virtual_memory().available / 2**30,
                      commit_headroom_gib=commit_headroom_gib(), tree_gib=tree)

    def kill(self, root: int) -> None:
        import psutil
        try:
            proc = psutil.Process(root)
            children = proc.children(recursive=True)
        except psutil.Error:
            return
        victims = [*reversed(children), proc]
        for victim in victims:
            try:
                victim.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(victims, timeout=10)


# --- Rows and the watcher ---------------------------------------------------


def _round(value: float | None, places: int) -> float | None:
    return None if value is None else round(value, places)


def sample_fields(sample: Sample, baseline: Baseline) -> dict[str, Any]:
    adapter, process = rises(sample, baseline)
    return {"card_mib": _round(sample.card_mib, 1), "adapter_shared_mib": _round(sample.adapter_shared_mib, 1),
            "adapter_rise_mib": _round(adapter, 1), "process_shared_mib": _round(sample.process_shared_mib, 1),
            "process_rise_mib": _round(process, 1), "available_gib": _round(sample.available_gib, 2),
            "commit_headroom_gib": _round(sample.commit_headroom_gib, 2), "tree_gib": _round(sample.tree_gib, 2)}


class Rows:
    """JSONL, one object per line, each with its wall time first."""

    def __init__(self, out: TextIO, wall: Callable[[], str]) -> None:
        self.out = out
        self.wall = wall

    def write(self, event: str, **fields: Any) -> None:
        self.out.write(json.dumps({"time": self.wall(), "event": event, **fields}) + "\n")
        self.out.flush()


@dataclass
class _Peaks:
    card_mib: float | None = None
    adapter_shared_mib: float | None = None
    process_shared_mib: float | None = None
    adapter_rise_mib: float | None = None
    process_rise_mib: float | None = None
    tree_gib: float | None = None
    min_available_gib: float | None = None
    min_commit_headroom_gib: float | None = None
    samples: int = 0

    def add(self, sample: Sample, baseline: Baseline) -> None:
        adapter, process = rises(sample, baseline)
        for name, value in (("card_mib", sample.card_mib), ("adapter_shared_mib", sample.adapter_shared_mib),
                            ("process_shared_mib", sample.process_shared_mib), ("adapter_rise_mib", adapter),
                            ("process_rise_mib", process), ("tree_gib", sample.tree_gib)):
            if value is not None:
                held = getattr(self, name)
                setattr(self, name, value if held is None else max(held, value))
        for name, value in (("min_available_gib", sample.available_gib),
                            ("min_commit_headroom_gib", sample.commit_headroom_gib)):
            if value is not None:
                held = getattr(self, name)
                setattr(self, name, value if held is None else min(held, value))
        self.samples += 1

    def fields(self) -> dict[str, Any]:
        places = {"tree_gib": 2, "min_available_gib": 2, "min_commit_headroom_gib": 2}
        out: dict[str, Any] = {name: _round(getattr(self, name), places.get(name, 1))
                               for name in ("card_mib", "adapter_shared_mib", "process_shared_mib",
                                            "tree_gib", "min_available_gib", "min_commit_headroom_gib")}
        out["peaks"] = {"card_mib": out.pop("card_mib"), "adapter_shared_mib": out.pop("adapter_shared_mib"),
                        "process_shared_mib": out.pop("process_shared_mib"), "tree_gib": out.pop("tree_gib")}
        out["largest_shared_rise_mib"] = {"adapter": _round(self.adapter_rise_mib, 1),
                                          "process": _round(self.process_rise_mib, 1)}
        out["samples"] = self.samples
        return out


def watch(host: Host, limits: Limits, rows: Rows, *, sleep: Callable[[float], None],
          clock: Callable[[], float], watch_pid: int | None = None, period: float = 1.0,
          row_every: float = 5.0, duration: float | None = None) -> int:
    """Sample until the watched tree exits, `duration` passes or a limit trips."""
    pids: list[int] = []
    if watch_pid is not None:
        found = host.tree(watch_pid)
        if found is None:
            print(f"card_guard: no process {watch_pid} to watch", file=sys.stderr)
            return EXIT_REFUSED
        pids = found
    first = host.sample(pids)
    if first.card_mib is None:
        print("card_guard: nvidia-smi gave no whole-card reading; refusing to watch blind", file=sys.stderr)
        return EXIT_REFUSED
    baseline = Baseline(first.adapter_shared_mib, first.process_shared_mib)
    rows.write("baseline", limits=dataclasses.asdict(limits), watch_pid=watch_pid, pids=pids,
               period_s=period, card_total_mib=first.card_total_mib, **sample_fields(first, baseline))
    peaks = _Peaks()
    peaks.add(first, baseline)
    start = clock()

    def finish(code: int, stopped: str) -> int:
        rows.write("summary", exit=code, stopped=stopped, seconds=round(clock() - start, 1), **peaks.fields())
        return code

    def trip(answer: Answer, sample: Sample) -> int:
        reading = _round(answer.reading, 2)
        if answer.why in ("ram", "commit"):
            rows.write("ram_guard", why=answer.why, reading=reading, limit=answer.limit,
                       available_gib=_round(sample.available_gib, 2),
                       commit_headroom_gib=_round(sample.commit_headroom_gib, 2), tree_gib=_round(sample.tree_gib, 2))
        else:
            rows.write("TRIP", why=answer.why, reading=reading, limit=answer.limit, **sample_fields(sample, baseline))
        if watch_pid is not None:
            host.kill(watch_pid)
        return finish(answer.exit_code, answer.why or "trip")

    answer = decide(first, baseline, limits)
    if answer.verdict == "trip":
        return trip(answer, first)
    if limits.busy_card_mib is not None and first.card_mib > limits.busy_card_mib:
        rows.write("rung_refused", reason=f"card busy: {_number(first.card_mib)} MiB > "
                                          f"{_number(limits.busy_card_mib)}", measured=[])
        return finish(EXIT_BUSY, "busy")
    if (limits.need_ram_gib is not None and first.available_gib is not None
            and first.available_gib < limits.need_ram_gib):
        rows.write("rung_refused", reason=f"preflight: {first.available_gib:.2f} GiB available < "
                                          f"{_number(limits.need_ram_gib)} needed", measured=[])
        return finish(EXIT_PREFLIGHT, "preflight")
    last_row = start
    unread = 0
    try:
        while True:
            if duration is not None and clock() - start >= duration:
                return finish(EXIT_OK, "duration")
            sleep(period)
            if watch_pid is not None:
                found = host.tree(watch_pid)
                if found is None:
                    return finish(EXIT_OK, "watched process exited")
                pids = found
            sample = host.sample(pids)
            peaks.add(sample, baseline)
            answer = decide(sample, baseline, limits)
            if answer.verdict == "trip":
                return trip(answer, sample)
            if sample.card_mib is None:
                unread += 1
                if unread == 1:
                    rows.write("card_unread", **sample_fields(sample, baseline))
                if unread >= limits.card_unread_samples:
                    return trip(Answer("trip", "card_unread", unread, limits.card_unread_samples), sample)
            else:
                unread = 0
            now = clock()
            if now - last_row >= row_every or near(sample, baseline, limits):
                rows.write("sample", **sample_fields(sample, baseline))
                last_row = now
    except KeyboardInterrupt:
        return finish(EXIT_OK, "interrupted")


# --- Command line -----------------------------------------------------------


_LIMIT_FLAGS = {
    "card_mib": "--card-mib", "project_mib": "--project-mib", "refuse_at_peak_mib": "--refuse-at-mib",
    "adapter_shared_rise_mib": "--adapter-shared-mib", "process_shared_rise_mib": "--process-shared-mib",
    "ram_floor_gib": "--ram-floor-gib", "commit_floor_gib": "--commit-floor-gib",
    "need_ram_gib": "--need-ram-gib", "busy_card_mib": "--busy-card-mib",
    "card_unread_samples": "--card-unread-samples",
    "margin_mib": "--margin-mib", "margin_gib": "--margin-gib",
}


def _limit_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("limits (defaults in card_guard.LIMITS; one session only)")
    for name, flag in _LIMIT_FLAGS.items():
        group.add_argument(flag, dest=name, type=float, default=None,
                           help=f"default {getattr(LIMITS, name)}")
    group.add_argument("--limits", dest="limits_file", help="a JSON object of Limits field names to values")


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="card_guard", description=__doc__.splitlines()[0] if __doc__ else None)
    sub = top.add_subparsers(dest="command", required=True)
    w = sub.add_parser("watch", help="sample the counters; on a trip kill the watched tree")
    w.add_argument("--watch-pid", type=int, default=None, help="the process tree to watch and kill on a trip")
    w.add_argument("--out", help="append rows to this JSONL file (default stdout)")
    w.add_argument("--period", type=float, default=1.0, help="seconds between samples (default 1)")
    w.add_argument("--row-every", type=float, default=5.0, help="seconds between sample rows (default 5)")
    w.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    _limit_args(w)
    p = sub.add_parser("plan", help="whether the next rung may run")
    p.add_argument("--points", required=True, help="JSON [[x, peak MiB], ...] of the rungs already run")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--x", type=float, help="the rung's megapixels times batch")
    where.add_argument("--size", help="WxH, with --batch")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--tripped", action="store_true", help="the session already tripped")
    p.add_argument("--out", help="append the rung_refused row to this JSONL file (default stdout)")
    _limit_args(p)
    return top


def _wall() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _limits_from(args: argparse.Namespace) -> Limits:
    file_values = None
    if args.limits_file:
        with open(args.limits_file, encoding="utf-8") as fh:
            file_values = json.load(fh)
        if not isinstance(file_values, dict):
            raise ValueError("--limits must hold a JSON object")
    return resolve_limits(file_values, {name: getattr(args, name) for name in _LIMIT_FLAGS})


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _plan(args: argparse.Namespace, limits: Limits, rows: Rows) -> int:
    points = json.loads(args.points)
    if not isinstance(points, list) or not all(
            isinstance(p, list) and len(p) == 2 and all(_finite(v) for v in p) for p in points):
        raise ValueError("--points must be a JSON list of [x, peak MiB] pairs of finite numbers")
    if args.size:
        m = re.fullmatch(r"(\d+)x(\d+)", args.size)
        if not m:
            raise ValueError(f"--size {args.size!r} is not WxH")
        x = rung_x(int(m.group(1)), int(m.group(2)), args.batch)
    else:
        x = args.x
    if not _finite(x):
        raise ValueError(f"--x {x!r} is not a finite number")
    answer = plan_rung(points, x, limits, args.tripped)
    if answer.verdict == "ok":
        print(json.dumps({"verdict": "ok", "x": x, "projected_mib": answer.projected_mib,
                          "measured": answer.measured}))
    else:
        rows.write("rung_refused", x=x, reason=answer.reason, projected_mib=answer.projected_mib,
                   measured=answer.measured, limits=dataclasses.asdict(limits))
    return answer.exit_code


def main(argv: list[str] | None = None, host: Host | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        limits = _limits_from(args)
    except (OSError, ValueError) as exc:
        print(f"card_guard: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    out: TextIO = open(args.out, "a", encoding="utf-8") if args.out else sys.stdout
    try:
        rows = Rows(out, _wall)
        if args.command == "plan":
            try:
                return _plan(args, limits, rows)
            except ValueError as exc:
                print(f"card_guard: {exc}", file=sys.stderr)
                return EXIT_REFUSED
        return watch(host or LocalHost(), limits, rows, sleep=time.sleep, clock=time.monotonic,
                     watch_pid=args.watch_pid, period=args.period, row_every=args.row_every,
                     duration=args.duration)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    sys.exit(main())
