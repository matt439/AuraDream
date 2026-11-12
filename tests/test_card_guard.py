"""The card guard (`tools/card_guard.py`) answers the recorded sessions' readings.

docs/maintainer/plan-card-harness.md, "Card guard": the decision on fixed samples copied
from the recorded rows (the 2026-10-25 RAM stops, the fixed loader's lowest
free RAM, ERNIE's projection gates, the shared rises that never spilled), the
limits' strict boundaries and overrides, and the watcher on a stub host with
an injected sleep and clock, so nothing reads a counter or races a clock.
"""
from __future__ import annotations

import ast
import dataclasses
import io
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tools import card_guard
from tools.card_guard import LIMITS, Baseline, Limits, Sample, decide, plan_rung

BASE = Baseline(adapter_shared_mib=300.0, process_shared_mib=40.0)


def _sample(**fields: float) -> Sample:
    """A quiet desktop reading with the named counters changed."""
    values: dict[str, float] = {"card_mib": 3000.0, "adapter_shared_mib": 300.0, "process_shared_mib": 40.0,
                                "available_gib": 30.0, "commit_headroom_gib": 20.0}
    values.update(fields)
    return Sample(**values)


# --- The decision -------------------------------------------------------------


@pytest.mark.parametrize("available", [3.83, 3.93, 3.7, 3.8, 3.78, 3.7])
def test_the_2026_10_25_ram_stops_each_trip_on_free_ram(available: float) -> None:
    answer = decide(_sample(available_gib=available), BASE)
    assert (answer.verdict, answer.why, answer.reading, answer.limit) == ("trip", "ram", available, 4.0)
    assert answer.exit_code == 4


def test_the_qwen_image_stop_is_ram_not_commit() -> None:
    answer = decide(_sample(available_gib=3.7, commit_headroom_gib=10.0), BASE)
    assert answer.why == "ram"


def test_a_commit_headroom_under_the_floor_trips_with_exit_4() -> None:
    answer = decide(_sample(commit_headroom_gib=3.9), BASE)
    assert (answer.verdict, answer.why, answer.exit_code) == ("trip", "commit", 4)


def test_the_fixed_loaders_lowest_free_ram_is_ok() -> None:
    assert decide(_sample(available_gib=14.2), BASE).verdict == "ok"


def test_free_ram_at_the_floor_does_not_trip() -> None:
    assert decide(_sample(available_gib=4.0, commit_headroom_gib=4.0), BASE).verdict == "ok"


def test_the_card_trips_past_15500_and_not_at_it() -> None:
    assert decide(_sample(card_mib=15500.0), BASE).verdict == "ok"
    answer = decide(_sample(card_mib=15501.0), BASE)
    assert (answer.verdict, answer.why, answer.reading, answer.limit) == ("trip", "card", 15501.0, 15500.0)
    assert answer.exit_code == 3


@pytest.mark.parametrize("adapter_rise,process_rise", [
    (82.0, 0.0), (86.0, 0.0),    # the first render's constant 82-86 MiB
    (82.0, 8.0),                 # WAI v17: process +8, adapter +82
    (172.0, 0.0), (180.0, 0.0),  # the critic's own llama-server allocation
])
def test_shared_rises_that_never_spilled_do_not_trip(adapter_rise: float, process_rise: float) -> None:
    sample = _sample(adapter_shared_mib=300.0 + adapter_rise, process_shared_mib=40.0 + process_rise)
    assert decide(sample, BASE).verdict == "ok"


def test_a_1100_mib_adapter_rise_trips() -> None:
    answer = decide(_sample(adapter_shared_mib=300.0 + 1100.0), BASE)
    assert (answer.verdict, answer.why, answer.reading, answer.limit) == ("trip", "adapter_shared", 1100.0, 1024.0)
    assert answer.exit_code == 3


def test_a_process_rise_past_512_trips() -> None:
    assert decide(_sample(process_shared_mib=40.0 + 512.0), BASE).verdict == "ok"
    answer = decide(_sample(process_shared_mib=40.0 + 513.0), BASE)
    assert (answer.why, answer.reading) == ("process_shared", 513.0)


def test_unread_counters_are_skipped() -> None:
    assert decide(Sample(), Baseline()).verdict == "ok"
    assert decide(_sample(process_shared_mib=9999.0), Baseline(adapter_shared_mib=300.0)).verdict == "ok"


def test_the_ernie_gate_two_points_project_ok() -> None:
    answer = plan_rung([[1.05, 9815], [2.07, 10247]], card_guard.rung_x(2048, 2048))
    assert card_guard.rung_x(2048, 2048) == 4.19
    assert (answer.verdict, answer.projected_mib) == ("ok", 11145)


def test_the_ernie_gate_refuses_the_next_rung_as_recorded() -> None:
    measured = [[1.05, 9815], [2.07, 10247], [4.19, 12631]]
    answer = plan_rung(measured, 8.39)
    assert answer.verdict == "refuse-projected"
    assert answer.projected_mib == 17354
    assert answer.reason == "projected 17354 MiB > 15200"
    assert answer.measured == measured
    assert answer.exit_code == 5


def test_a_rung_is_refused_after_a_peak_at_the_refuse_at_figure_or_a_trip() -> None:
    assert plan_rung([[1.0, 14999]], 2.0).verdict == "ok"
    peaked = plan_rung([[1.0, 15000]], 2.0)
    assert (peaked.verdict, peaked.reason) == ("refuse-projected", "an earlier rung peaked at 15000 MiB >= 15000")
    tripped = plan_rung([], 1.0, tripped=True)
    assert (tripped.verdict, tripped.reason) == ("refuse-projected", "the session tripped")


def test_one_point_projects_nothing_and_a_shared_x_is_refused() -> None:
    assert plan_rung([[1.05, 9815]], 4.19) == card_guard.Answer("ok", measured=[[1.05, 9815.0]])
    with pytest.raises(ValueError, match="share x"):
        plan_rung([[1.0, 9000], [1.0, 9100]], 2.0)


# --- Limits ---------------------------------------------------------------------


def test_overrides_change_the_answer() -> None:
    raised = card_guard.resolve_limits(None, {"card_mib": 15950.0})
    assert decide(_sample(card_mib=15900.0), BASE).verdict == "trip"
    assert decide(_sample(card_mib=15900.0), BASE, raised).verdict == "ok"
    from_file = card_guard.resolve_limits({"project_mib": 18000}, {})
    assert plan_rung([[1.05, 9815], [2.07, 10247], [4.19, 12631]], 8.39, from_file).verdict == "ok"
    floor = card_guard.resolve_limits({"ram_floor_gib": 20}, {"ram_floor_gib": 40.0})
    assert floor.ram_floor_gib == 40.0  # the command line over the file
    assert decide(_sample(available_gib=30.0), BASE, floor).why == "ram"


def test_unknown_or_non_numeric_limits_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown limit"):
        card_guard.resolve_limits({"card": 1}, {})
    with pytest.raises(ValueError, match="finite number"):
        card_guard.resolve_limits({"card_mib": "15500"}, {})
    with pytest.raises(ValueError, match="finite number"):
        card_guard.resolve_limits({"card_mib": True}, {})


def test_the_defaults_block_is_the_only_source_of_the_defaults() -> None:
    assert (LIMITS.card_mib, LIMITS.project_mib, LIMITS.refuse_at_peak_mib, LIMITS.adapter_shared_rise_mib,
            LIMITS.process_shared_rise_mib, LIMITS.ram_floor_gib, LIMITS.commit_floor_gib) == (
        15500, 15200, 15000, 1024, 512, 4, 4)
    assert LIMITS.need_ram_gib is None  # a load run states its own
    assert card_guard.resolve_limits(None, {}) == Limits() == LIMITS
    # Every flag defaults to None, so nothing but the block supplies a value...
    for command in ("watch", "plan"):
        argv = [command] if command == "watch" else [command, "--points", "[]", "--x", "1"]
        args = card_guard.parser().parse_args(argv)
        assert {name: getattr(args, name) for name in card_guard._LIMIT_FLAGS} == dict.fromkeys(
            card_guard._LIMIT_FLAGS)
    assert set(card_guard._LIMIT_FLAGS) == {f.name for f in dataclasses.fields(Limits)}
    # ...and no function's own defaults name a limit's figure.
    source = Path(card_guard.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    figures = {v for v in dataclasses.astuple(LIMITS) if v is not None and v not in (1.0,)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Limits":
            continue
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if isinstance(default, ast.Constant) and not isinstance(default.value, bool):
                    assert default.value not in figures, (getattr(node, "name", "lambda"), default.value)


# --- Counter parsing ------------------------------------------------------------

TYPEPERF = (
    '\r\n"(PDH-CSV 4.0)","\\\\PC\\GPU Adapter Memory(luid_0x00000000_0x00017694_phys_0)\\Shared Usage",'
    '"\\\\PC\\GPU Adapter Memory(luid_0x00000000_0x00018F2D_phys_0)\\Shared Usage",'
    '"\\\\PC\\GPU Adapter Memory(luid_0x00000000_0x00017694_phys_0)\\Dedicated Usage",'
    '"\\\\PC\\GPU Adapter Memory(luid_0x00000000_0x00018F2D_phys_0)\\Dedicated Usage",'
    '"\\\\PC\\GPU Process Memory(pid_100_luid_0x00000000_0x00017694_phys_0)\\Shared Usage",'
    '"\\\\PC\\GPU Process Memory(pid_101_luid_0x00000000_0x00017694_phys_0)\\Shared Usage",'
    '"\\\\PC\\GPU Process Memory(pid_100_luid_0x00000000_0x00018F2D_phys_0)\\Shared Usage",'
    '"\\\\PC\\GPU Process Memory(pid_555_luid_0x00000000_0x00017694_phys_0)\\Shared Usage"\r\n'
    '"10/30/2026 22:19:08.142","104857600.000000","8192.000000","3192995840.000000","0.000000",'
    '"2097152.000000","1048576.000000","9999999.000000","52428800.000000"\r\n'
    "Exiting, please wait...\r\nThe command completed successfully.\r\n"
)


def test_typeperf_reads_the_busiest_adapter_and_the_trees_processes_on_it() -> None:
    assert card_guard.parse_typeperf(TYPEPERF, [100, 101]) == (100.0, 3.0)
    assert card_guard.parse_typeperf(TYPEPERF, []) == (100.0, None)
    assert card_guard.parse_typeperf("", [100]) == (None, None)


def test_nvidia_smi_reads_used_and_total() -> None:
    assert card_guard.parse_nvidia_smi("3029, 16303\n") == (3029.0, 16303.0)
    assert card_guard.parse_nvidia_smi("[N/A], 16303\n") is None
    assert card_guard.parse_nvidia_smi("") is None


# --- The watcher, on a stub host ------------------------------------------------


class StubHost:
    def __init__(self, samples: list[Sample], tree: list[int] | None = None, exits_after: int | None = None):
        self.samples = list(samples)
        self.tree_pids = tree
        self.exits_after = exits_after
        self.trees = 0
        self.killed: list[int] = []

    def tree(self, root: int) -> list[int] | None:
        self.trees += 1
        if self.exits_after is not None and self.trees > self.exits_after:
            return None
        return self.tree_pids

    def sample(self, pids: list[int]) -> Sample:
        return self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]

    def kill(self, root: int) -> None:
        self.killed.append(root)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _watch(host: StubHost, limits: Limits = LIMITS, **kw: object) -> tuple[int, list[dict[str, object]]]:
    out = io.StringIO()
    clock = Clock()
    code = card_guard.watch(host, limits, card_guard.Rows(out, lambda: "t"), sleep=clock.sleep, clock=clock,
                            **kw)  # type: ignore[arg-type]
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def test_a_quiet_session_writes_baseline_samples_and_summary() -> None:
    code, rows = _watch(StubHost([_sample()], tree=[100, 101]), watch_pid=100, period=1.0, row_every=5.0,
                        duration=12.0)
    assert code == 0
    events = [r["event"] for r in rows]
    assert events == ["baseline", "sample", "sample", "summary"]
    assert rows[0]["limits"] == dataclasses.asdict(LIMITS)
    assert rows[0]["pids"] == [100, 101]
    assert rows[-1]["stopped"] == "duration"
    assert rows[-1]["samples"] == 13


def test_a_sample_near_a_limit_is_written_whatever_the_period() -> None:
    near = _sample(card_mib=15100.0)
    code, rows = _watch(StubHost([_sample(), near, _sample()]), period=1.0, row_every=60.0, duration=3.0)
    assert code == 0
    samples = [r for r in rows if r["event"] == "sample"]
    assert [r["card_mib"] for r in samples] == [15100.0]


def test_a_ram_trip_writes_ram_guard_kills_the_tree_and_exits_4() -> None:
    host = StubHost([_sample(), _sample(available_gib=3.83, tree_gib=46.31)], tree=[100])
    code, rows = _watch(host, watch_pid=100, duration=60.0)
    assert code == 4
    assert host.killed == [100]
    guard = next(r for r in rows if r["event"] == "ram_guard")
    assert (guard["why"], guard["available_gib"], guard["tree_gib"], guard["limit"]) == ("ram", 3.83, 46.31, 4.0)
    assert rows[-1]["event"] == "summary" and rows[-1]["exit"] == 4


def test_a_floor_above_free_ram_trips_at_once() -> None:
    host = StubHost([_sample(available_gib=30.0)], tree=[100])
    code, rows = _watch(host, card_guard.resolve_limits(None, {"ram_floor_gib": 1000.0}), watch_pid=100)
    assert code == 4 and host.killed == [100]
    assert [r["event"] for r in rows] == ["baseline", "ram_guard", "summary"]


def test_a_shared_trip_writes_trip_and_exits_3() -> None:
    host = StubHost([_sample(), _sample(adapter_shared_mib=300.0 + 1100.0)], tree=[100])
    code, rows = _watch(host, watch_pid=100, duration=60.0)
    assert code == 3 and host.killed == [100]
    trip = next(r for r in rows if r["event"] == "TRIP")
    assert (trip["why"], trip["reading"], trip["limit"]) == ("adapter_shared", 1100.0, 1024.0)


def test_without_a_watched_pid_a_trip_kills_nothing() -> None:
    host = StubHost([_sample(card_mib=15501.0)])
    code, _ = _watch(host)
    assert code == 3 and host.killed == []


def test_the_watcher_ends_when_the_watched_tree_exits() -> None:
    code, rows = _watch(StubHost([_sample()], tree=[100], exits_after=3), watch_pid=100)
    assert code == 0
    assert rows[-1]["stopped"] == "watched process exited"


def test_preflight_and_a_busy_card_refuse_before_the_start() -> None:
    host = StubHost([_sample(available_gib=30.0)], tree=[100])
    code, rows = _watch(host, card_guard.resolve_limits(None, {"need_ram_gib": 46.0}), watch_pid=100)
    assert code == 5 and host.killed == []
    assert rows[1]["event"] == "rung_refused" and rows[1]["reason"] == "preflight: 30.00 GiB available < 46 needed"
    code, rows = _watch(StubHost([_sample(card_mib=3000.0)]), card_guard.resolve_limits(None, {"busy_card_mib": 2500.0}))
    assert code == 7
    assert rows[1]["reason"] == "card busy: 3000 MiB > 2500"


def test_a_missing_pid_or_card_reading_is_refused() -> None:
    assert _watch(StubHost([_sample()], tree=None), watch_pid=100)[0] == 2
    assert _watch(StubHost([Sample(available_gib=30.0)]))[0] == 2


def _unread() -> Sample:
    return dataclasses.replace(_sample(), card_mib=None, card_total_mib=None)


def test_the_card_unread_in_a_row_trips_kills_the_tree_and_exits_3() -> None:
    assert LIMITS.card_unread_samples == 3.0
    host = StubHost([_sample(), _sample(), _unread(), _unread(), _unread(), _sample()], tree=[100])
    code, rows = _watch(host, watch_pid=100, duration=60.0)
    assert code == 3 and host.killed == [100]
    events = [r["event"] for r in rows]
    # One card_unread row, at the first unread sample; the third trips.
    assert events.count("card_unread") == 1
    assert events[events.index("card_unread"):] == ["card_unread", "TRIP", "summary"]
    trip = next(r for r in rows if r["event"] == "TRIP")
    assert (trip["why"], trip["reading"], trip["limit"], trip["card_mib"]) == ("card_unread", 3, 3.0, None)
    assert rows[-1]["exit"] == 3 and rows[-1]["stopped"] == "card_unread"


def test_a_card_reading_back_resets_the_unread_count() -> None:
    # Two unread, a reading, two unread, a reading: never three in a row.
    host = StubHost([_sample(), _unread(), _unread(), _sample(), _unread(), _unread(), _sample()], tree=[100])
    code, rows = _watch(host, watch_pid=100, duration=10.0)
    assert code == 0 and host.killed == []
    assert [r["event"] for r in rows].count("card_unread") == 2
    assert rows[-1]["stopped"] == "duration"


def test_the_unread_count_is_a_limit_a_session_can_change() -> None:
    host = StubHost([_sample(), _unread()], tree=[100])
    limits = card_guard.resolve_limits(None, {"card_unread_samples": 1.0})
    code, rows = _watch(host, limits, watch_pid=100, duration=60.0)
    assert code == 3 and host.killed == [100]
    assert [r["event"] for r in rows] == ["baseline", "card_unread", "TRIP", "summary"]
    # A RAM trip in the same sample still wins: it is the decision's.
    host = StubHost([_sample(), dataclasses.replace(_unread(), available_gib=3.7)], tree=[100])
    code, rows = _watch(host, watch_pid=100, duration=60.0)
    assert code == 4 and [r["event"] for r in rows][1] == "ram_guard"


# --- Holding the period (DEV#20) ----------------------------------------------------


class SlowHost(StubHost):
    """A stub whose every sample takes `cost` seconds of the injected clock."""

    def __init__(self, clock: Clock, cost: float, samples: list[Sample]) -> None:
        super().__init__(samples)
        self.clock = clock
        self.cost = cost
        self.closed = 0

    def sample(self, pids: list[int]) -> Sample:
        self.clock.now += self.cost
        return super().sample(pids)

    def close(self) -> None:
        self.closed += 1


def _paced(cost: float, period: float, duration: float) -> tuple[SlowHost, list[float], list[dict[str, object]]]:
    clock = Clock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(round(seconds, 6))
        clock.sleep(seconds)

    host = SlowHost(clock, cost, [_sample()])
    out = io.StringIO()
    code = card_guard.watch(host, LIMITS, card_guard.Rows(out, lambda: "t"), sleep=sleep, clock=clock,
                            period=period, row_every=60.0, duration=duration)
    assert code == 0
    return host, sleeps, [json.loads(line) for line in out.getvalue().splitlines()]


def test_a_samples_cost_comes_out_of_the_sleep_not_on_top_of_the_period() -> None:
    host, sleeps, rows = _paced(cost=0.3, period=1.0, duration=10.0)
    # The first sleep is the whole period (the baseline's cost came before
    # the start); every later one is the period less the sample's 0.3 s.
    assert sleeps[0] == 1.0 and set(sleeps[1:]) == {0.7}
    summary = rows[-1]
    assert (summary["period_s"], summary["achieved_period_s"], summary["longest_gap_s"]) == (1.0, 1.0, 1.0)
    assert host.closed == 1


def test_a_sampler_slower_than_the_period_reports_the_period_it_achieved() -> None:
    # The 2026-11-11 session's case: 0.5 s asked, about 1.8 s per sample.
    host, sleeps, rows = _paced(cost=1.8, period=0.5, duration=9.0)
    assert sleeps == [0.5]  # behind from then on: no sleep, and no burst to catch up
    summary = rows[-1]
    assert (summary["period_s"], summary["achieved_period_s"], summary["longest_gap_s"]) == (0.5, 1.8, 1.8)
    assert rows[0]["period_s"] == 0.5 and "sources" not in rows[0]


def test_the_host_is_closed_however_the_watch_ends() -> None:
    clock = Clock()
    host = SlowHost(clock, 0.0, [_sample(card_mib=15501.0)])
    assert card_guard.watch(host, LIMITS, card_guard.Rows(io.StringIO(), lambda: "t"), sleep=clock.sleep,
                            clock=clock) == 3
    assert host.closed == 1
    blind = SlowHost(clock, 0.0, [Sample(available_gib=30.0)])
    assert card_guard.watch(blind, LIMITS, card_guard.Rows(io.StringIO(), lambda: "t"), sleep=clock.sleep,
                            clock=clock) == 2
    assert blind.closed == 1


def test_one_sample_reports_no_achieved_period() -> None:
    _, _, rows = _paced(cost=0.0, period=1.0, duration=0.0)
    assert (rows[-1]["achieved_period_s"], rows[-1]["longest_gap_s"]) == (None, None)


def test_the_pdh_counters_read_as_the_typeperf_rows_do() -> None:
    luid, other = "luid_0x00000000_0x00017694_phys_0", "luid_0x00000000_0x00018F2D_phys_0"
    counters = {
        f"\\GPU Adapter Memory({luid})\\Shared Usage": 104857600.0,
        f"\\GPU Adapter Memory({other})\\Shared Usage": 8192.0,
        f"\\GPU Adapter Memory({luid})\\Dedicated Usage": 3192995840.0,
        f"\\GPU Adapter Memory({other})\\Dedicated Usage": 0.0,
        f"\\GPU Process Memory(pid_100_{luid})\\Shared Usage": 2097152.0,
        f"\\GPU Process Memory(pid_101_{luid})\\Shared Usage": 1048576.0,
        f"\\GPU Process Memory(pid_100_{other})\\Shared Usage": 9999999.0,
        f"\\GPU Process Memory(pid_555_{luid})\\Shared Usage": 52428800.0,
    }
    assert card_guard.shared_usage(counters, [100, 101]) == card_guard.parse_typeperf(TYPEPERF, [100, 101])
    assert card_guard.shared_usage(counters, [100, 101]) == (100.0, 3.0)
    assert card_guard.shared_usage({}, [100]) == (None, None)


class FakeProc:
    """A stream process whose stdout yields `lines`, then blocks until `release`."""

    def __init__(self, lines: list[str], release: threading.Event) -> None:
        self.lines = lines
        self.release = release
        self.terminated = False
        self.drained = threading.Event()

    @property
    def stdout(self):  # type: ignore[no-untyped-def]
        yield from self.lines
        self.drained.set()
        self.release.wait(30)

    def terminate(self) -> None:
        self.terminated = True
        self.release.set()

    def poll(self) -> int | None:
        return 0 if self.release.is_set() else None

    def wait(self, timeout: float | None = None) -> int:
        return 0


class Spawner:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.argvs: list[list[str]] = []
        self.procs: list[FakeProc] = []

    def __call__(self, argv: list[str]) -> FakeProc:
        self.argvs.append(argv)
        proc = FakeProc(self.lines, threading.Event())
        self.procs.append(proc)
        return proc


def test_nvidia_smi_is_one_looping_stream_on_the_first_gpu() -> None:
    spawn = Spawner([])
    stream = card_guard.nvidia_smi_stream(0.5, spawn=spawn)
    assert stream.argv == ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits",
                           "-i", "0", "--loop-ms=500"]
    assert card_guard.nvidia_smi_stream(0.01, spawn=spawn).argv[-1] == "--loop-ms=100"
    assert spawn.argvs == []  # nothing runs until the first read


def test_the_stream_keeps_the_newest_line_and_goes_stale() -> None:
    clock = Clock()
    spawn = Spawner(["[N/A], 16303\n", "3029, 16303\n", "3100, 16303\n"])
    stream = card_guard.LineStream(["nvidia-smi"], card_guard.parse_nvidia_smi, spawn=spawn, clock=clock)
    # The first read waits for the stream's lines (the unparsable one skipped).
    assert stream.latest(5.0, wait=10.0) is not None
    assert spawn.procs[0].drained.wait(10)
    assert stream.latest(5.0) == (3100.0, 16303.0)
    assert stream.starts == 1
    # Silent past the age allowed: unread, the process kept.
    clock.now += 6.0
    assert stream.latest(5.0) is None
    assert stream.starts == 1
    stream.close()
    assert spawn.procs[0].terminated


def test_an_exited_stream_starts_again_and_a_missing_one_reads_none() -> None:
    clock = Clock()
    spawn = Spawner(["3029, 16303\n"])
    stream = card_guard.LineStream(["nvidia-smi"], card_guard.parse_nvidia_smi, spawn=spawn, clock=clock)
    assert stream.latest(5.0, wait=10.0) == (3029.0, 16303.0)
    spawn.procs[0].release.set()  # the process ends
    assert stream._thread is not None
    stream._thread.join(10)
    clock.now += 6.0
    assert stream.latest(5.0, wait=10.0) == (3029.0, 16303.0)
    assert stream.starts == 2
    stream.close()

    def missing(argv: list[str]) -> FakeProc:
        raise FileNotFoundError(argv[0])

    gone = card_guard.LineStream(["nvidia-smi"], card_guard.parse_nvidia_smi, spawn=missing, clock=clock)
    assert gone.latest(5.0, wait=1.0) is None
    gone.close()


class BadLineProc(FakeProc):
    """A process still running whose output the pump cannot read (an
    undecodable byte under text mode raises UnicodeDecodeError)."""

    @property
    def stdout(self):  # type: ignore[no-untyped-def]
        yield "3029, 16303\n"
        raise UnicodeDecodeError("cp1252", b"\x81", 0, 1, "undefined")


def test_a_stream_whose_reader_died_stops_its_process_before_starting_another() -> None:
    clock = Clock()
    procs: list[FakeProc] = []

    def spawn(argv: list[str]) -> FakeProc:
        proc = (BadLineProc if not procs else FakeProc)(["3100, 16303\n"], threading.Event())
        procs.append(proc)
        return proc

    stream = card_guard.LineStream(["nvidia-smi"], card_guard.parse_nvidia_smi, spawn=spawn, clock=clock)
    assert stream.latest(5.0, wait=10.0) == (3029.0, 16303.0)
    assert stream._thread is not None
    stream._thread.join(10)
    assert procs[0].poll() is None  # the reader is gone, the process is not
    clock.now += 6.0
    assert stream.latest(5.0, wait=10.0) == (3100.0, 16303.0)
    assert stream.starts == 2
    assert procs[0].terminated  # not left running on a pipe nobody drains
    stream.close()
    assert procs[1].terminated


class FakePdh:
    """pdh.dll's three calls, writing real PDH_FMT_COUNTERVALUE_ITEM_W records
    into the caller's buffer. `grow` adds instances between a counter's size
    probe and its fill that many times (a process starting mid-read)."""

    MORE_DATA = 0x800007D2

    def __init__(self, instances: dict[int, list[tuple[str, int, float]]], grow: int = 0) -> None:
        self.instances = instances
        self.grow = grow
        self.fills = 0
        self._keep: list[object] = []

    def PdhCollectQueryData(self, query: object) -> int:  # noqa: N802
        return 0

    def PdhGetFormattedCounterArrayW(self, handle: int, fmt: int, size_ref, count_ref, buf) -> int:  # noqa: N802,ANN001
        import ctypes
        item_type = card_guard.pdh_item_type()
        rows = self.instances[handle]
        size, count = size_ref._obj, count_ref._obj
        need = len(rows) * ctypes.sizeof(item_type)
        if buf is None:
            size.value, count.value = need, len(rows)
            return 0 if not rows else self.MORE_DATA
        self.fills += 1
        if self.grow:
            self.grow -= 1
            rows.append(("pid_999_luid_x", 0, 1.0))
            size.value, count.value = len(rows) * ctypes.sizeof(item_type), len(rows)
            return self.MORE_DATA
        array = (item_type * len(rows))()
        for i, (name, status, value) in enumerate(rows):
            array[i].szName = name
            array[i].FmtValue.CStatus = status
            array[i].FmtValue.doubleValue = value
        self._keep.append(array)
        ctypes.memmove(buf, array, need)
        count.value = len(rows)
        return 0


def _pdh_instances() -> dict[int, list[tuple[str, int, float]]]:
    luid, other = "luid_0x00000000_0x00017694_phys_0", "luid_0x00000000_0x00018F2D_phys_0"
    return {
        0: [(luid, 0, 104857600.0), (other, 0, 8192.0)],
        1: [(luid, 1, 3192995840.0), (other, 0, 0.0)],
        2: [(f"pid_100_{luid}", 0, 2097152.0), (f"pid_101_{luid}", 0, 1048576.0),
            (f"pid_100_{other}", 0, 9999999.0), (f"pid_555_{luid}", 0, 52428800.0),
            (f"pid_101_{other}", 0xC0000BBA, 7e9)],  # PDH_CSTATUS_INVALID_DATA: skipped
    }


def _pdh_counters(fake: FakePdh) -> card_guard.PdhCounters:
    return card_guard.PdhCounters(fake, object(), [(path, i) for i, path in enumerate(card_guard.TYPEPERF_COUNTERS)])


def test_the_pdh_read_names_each_instance_and_skips_invalid_ones() -> None:
    counters = _pdh_counters(FakePdh(_pdh_instances())).read()
    assert counters is not None
    assert "\\GPU Process Memory(pid_101_luid_0x00000000_0x00018F2D_phys_0)\\Shared Usage" not in counters
    assert counters["\\GPU Adapter Memory(luid_0x00000000_0x00017694_phys_0)\\Dedicated Usage"] == 3192995840.0
    assert card_guard.shared_usage(counters, [100, 101]) == card_guard.parse_typeperf(TYPEPERF, [100, 101])


def test_a_pdh_array_that_grows_mid_read_is_probed_again_not_dropped() -> None:
    fake = FakePdh(_pdh_instances(), grow=2)
    counters = _pdh_counters(fake).read()
    assert counters is not None and fake.fills == 5  # two MORE_DATA fills, then the three counters
    assert card_guard.shared_usage(counters, [100, 101]) == (100.0, 3.0)
    # One that keeps growing past the attempts fails the collection.
    endless = FakePdh(_pdh_instances(), grow=100)
    assert _pdh_counters(endless).read() is None


@pytest.mark.skipif(sys.platform != "win32", reason="a Windows job object")
def test_a_stream_process_dies_with_the_guards_job() -> None:
    job = card_guard.KillOnCloseJob.open()
    assert job is not None
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert job.assign(child.pid)
        assert child.poll() is None
        job.close()  # what the guard's exit does, however it ends
        assert child.wait(timeout=10) is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


class StubStream:
    def __init__(self, value: tuple[float, float] | None) -> None:
        self.value = value
        self.calls: list[tuple[float, float]] = []
        self.closed = False

    def latest(self, max_age: float, wait: float = 0.0) -> tuple[float, float] | None:
        self.calls.append((max_age, wait))
        return self.value

    def close(self) -> None:
        self.closed = True


class StubPdh:
    def __init__(self, counters: dict[str, float] | None) -> None:
        self.counters = counters
        self.closed = False

    def read(self) -> dict[str, float] | None:
        return self.counters

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def quiet_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real RAM or commit reading in a LocalHost test."""
    import psutil

    class Memory:
        available = 30 * 2**30

    monkeypatch.setattr(psutil, "virtual_memory", lambda: Memory())
    monkeypatch.setattr(card_guard, "commit_headroom_gib", lambda: 20.0)


def _typeperf_never() -> str | None:
    raise AssertionError("typeperf spawned with PDH open")


def test_the_local_host_reads_the_stream_and_the_held_query(quiet_memory: None) -> None:
    card = StubStream((3029.0, 16303.0))
    luid = "luid_0x0_0x1_phys_0"
    pdh = StubPdh({f"\\GPU Adapter Memory({luid})\\Shared Usage": 100 * 2**20,
                   f"\\GPU Adapter Memory({luid})\\Dedicated Usage": 3 * 2**30})
    host = card_guard.LocalHost(0.5, card=card, open_shared=lambda: pdh,  # type: ignore[arg-type,return-value]
                                typeperf=_typeperf_never, windows=True)
    assert host.sources == {"card": "nvidia-smi --loop-ms", "shared": "pdh"}
    first = host.sample([])
    assert (first.card_mib, first.card_total_mib, first.adapter_shared_mib, first.available_gib,
            first.commit_headroom_gib) == (3029.0, 16303.0, 100.0, 30.0, 20.0)
    card.value = None  # the stream went silent
    assert host.sample([]).card_mib is None
    # Only the first sample waits for nvidia-smi's first line; a line counts
    # as read for two periods plus the slack.
    assert card.calls == [(1.0 + card_guard._STALE_SLACK_S, card_guard._FIRST_READ_S),
                          (1.0 + card_guard._STALE_SLACK_S, 0.0)]
    pdh.counters = None  # a failed collection reads nothing
    assert host.sample([]).adapter_shared_mib is None
    host.close()
    assert card.closed and pdh.closed


def test_without_pdh_windows_falls_back_to_typeperf_and_linux_reads_no_shared(quiet_memory: None) -> None:
    host = card_guard.LocalHost(1.0, card=StubStream((3029.0, 16303.0)), open_shared=lambda: None,  # type: ignore[arg-type]
                                typeperf=lambda: TYPEPERF, windows=True)
    assert host.sources["shared"] == "typeperf -sc 1"
    assert host.sample([100, 101]).process_shared_mib == 3.0

    def never_opened() -> None:
        raise AssertionError("PDH opened off Windows")

    linux = card_guard.LocalHost(1.0, card=StubStream((3029.0, 16303.0)), open_shared=never_opened,  # type: ignore[arg-type]
                                 typeperf=_typeperf_never, windows=False)
    assert linux.sources == {"card": "nvidia-smi --loop-ms", "shared": None}
    sample = linux.sample([])
    assert (sample.card_mib, sample.adapter_shared_mib, sample.process_shared_mib) == (3029.0, None, None)


# --- The command line -------------------------------------------------------------


def test_plan_cli_writes_the_recorded_rung_refused_row(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "rows.jsonl"
    code = card_guard.main(["plan", "--points", "[[1.05, 9815], [2.07, 10247], [4.19, 12631]]", "--x", "8.39",
                            "--out", str(out)])
    assert code == 5
    row = json.loads(out.read_text(encoding="utf-8"))
    assert (row["event"], row["reason"], row["measured"]) == (
        "rung_refused", "projected 17354 MiB > 15200", [[1.05, 9815], [2.07, 10247], [4.19, 12631]])
    assert card_guard.main(["plan", "--points", "[[1.05, 9815], [2.07, 10247]]", "--size", "2048x2048"]) == 0
    assert json.loads(capsys.readouterr().out)["projected_mib"] == 11145


def test_cli_limits_from_a_file_and_refusals(tmp_path: Path) -> None:
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"project_mib": 18000}), encoding="utf-8")
    points = "[[1.05, 9815], [2.07, 10247], [4.19, 12631]]"
    assert card_guard.main(["plan", "--points", points, "--x", "8.39", "--limits", str(limits)]) == 0
    limits.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    assert card_guard.main(["plan", "--points", points, "--x", "8.39", "--limits", str(limits)]) == 2
    assert card_guard.main(["plan", "--points", "{}", "--x", "8.39"]) == 2
    assert card_guard.main(["plan", "--points", "[]", "--size", "big"]) == 2
    # A point or x that is not a finite number is refused, not a traceback.
    assert card_guard.main(["plan", "--points", "[[null, 1], [2, 3]]", "--x", "3"]) == 2
    assert card_guard.main(["plan", "--points", "[[1, true], [2, 3]]", "--x", "3"]) == 2
    assert card_guard.main(["plan", "--points", "[[1, 9000], [2, 9500]]", "--x", "inf"]) == 2


def test_watch_cli_on_a_stub_host(tmp_path: Path) -> None:
    out = tmp_path / "rows.jsonl"
    host = StubHost([_sample(available_gib=30.0)], tree=[100])
    code = card_guard.main(["watch", "--watch-pid", "100", "--ram-floor-gib", "1000", "--out", str(out)], host=host)
    assert code == 4 and host.killed == [100]
    events = [json.loads(line)["event"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert events == ["baseline", "ram_guard", "summary"]


def test_the_module_imports_nothing_heavy() -> None:
    tree = ast.parse(Path(card_guard.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names & {"backend", "torch", "diffusers", "numpy"}


def test_watch_cli_refuses_a_period_that_is_not_positive() -> None:
    host = StubHost([_sample()])
    assert card_guard.main(["watch", "--period", "0"], host=host) == 2
    assert card_guard.main(["watch", "--period", "-1"], host=host) == 2
    assert card_guard.main(["watch", "--period", "nan"], host=host) == 2
