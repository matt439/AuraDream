"""Private, offline one-load Turbo Refine experiment; never enables public Refine.

Run only in an authorized idle-card window. The parent guards a waiting child
before releasing its first load, retains raw samples and stops the child tree
on missing counters, a limit, an exception or a fixed wall-time budget. Image
judgements remain blank for an actual reviewer; a successful run is evidence,
not qualification. See docs/maintainer/plan-anima-refine.md.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import anima_refine_pack as pack, card_guard as guard  # noqa: E402

PERIOD = .25
BUDGETS = {"load": 600, "case": 120, "cleanup": 120}
SESSION_SECONDS = 1800
REQUIRED_COUNTERS = tuple(field.name for field in dataclasses.fields(guard.Sample))


def atomic_json(path: Path, value: Any) -> None:
    data = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    # Windows readers (and scanners) may briefly omit delete sharing. Only
    # those platform errors are retried, within .28 seconds and well below
    # the independent two-second monitor heartbeat. Other errors stay fatal.
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(.01 * (attempt + 1))


def message_path(output: Path, channel: str, sequence: int, action: str | None = None) -> Path:
    suffix = "" if action is None else "-" + action
    return output / f"{channel}-{sequence:04d}{suffix}.json"


def publish_message(output: Path, channel: str, sequence: int, value: Any,
                    action: str | None = None) -> None:
    # Each handoff/action is published once. Pollers open only its final
    # name, which is absent until publication, and it is never replaced.
    path = message_path(output, channel, sequence, action)
    if path.exists():
        raise FileExistsError(f"immutable IPC message already published: {path.name}")
    atomic_json(path, value)


def read_json(path: Path) -> dict[str, Any] | None:
    for attempt in range(8):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except OSError as exc:
            # A newly published immutable file can briefly be held by an
            # exclusive Windows reader/scanner. CRT open may report that
            # as PermissionError(EACCES) without preserving a winerror.
            sharing = getattr(exc, "winerror", None) in (5, 32, 33) or (
                sys.platform == "win32" and isinstance(exc, PermissionError)
                and getattr(exc, "winerror", None) is None and exc.errno == errno.EACCES
            )
            if not sharing or attempt == 7:
                raise
            time.sleep(.01 * (attempt + 1))
    raise AssertionError("bounded read retry must return or raise")


def strict_sample(sample: guard.Sample) -> None:
    for name in REQUIRED_COUNTERS:
        value = getattr(sample, name)
        if value is None or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"required counter unavailable or invalid: {name}")


def admission(sample: guard.Sample, limits: guard.Limits) -> guard.Baseline:
    strict_sample(sample)
    if limits.need_ram_gib is None or limits.busy_card_mib is None:
        raise ValueError("explicit load-RAM and busy-card limits are required")
    baseline = guard.Baseline(sample.adapter_shared_mib, sample.process_shared_mib)
    answer = guard.decide(sample, baseline, limits)
    if answer.verdict != "ok":
        raise RuntimeError(f"preflight {answer.why}: {answer.reading} crosses {answer.limit}")
    assert sample.available_gib is not None and sample.card_mib is not None
    if sample.available_gib < limits.need_ram_gib:
        raise RuntimeError(f"preflight RAM: {sample.available_gib:.2f} GiB < {limits.need_ram_gib}")
    if sample.card_mib > limits.busy_card_mib:
        raise RuntimeError(f"card busy: {sample.card_mib:.0f} MiB > {limits.busy_card_mib}")
    return baseline


def environment(output: Path) -> dict[str, str]:
    # No saved choices, private gallery, loader recipe or inherited preload.
    env = {name: value for name, value in os.environ.items() if not name.startswith("AURADREAM_")}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                "AURADREAM_MODEL_ID": pack.CHECKPOINT, "AURADREAM_QUANT": "fp8",
                "AURADREAM_OFFLOAD": "model", "AURADREAM_DEVICE": "cuda",
                "AURADREAM_PRELOAD": "false", "AURADREAM_COMPILATION": "eager",
                "AURADREAM_ATTENTION": "native", "AURADREAM_DB_PATH": str(output / "scratch.db"),
                "AURADREAM_OUTPUTS_DIR": str(output / "scratch-outputs")})
    return env


def host_metrics(samples: list[guard.Sample], baseline: guard.Baseline) -> dict[str, float]:
    if not samples:
        raise RuntimeError("phase has no guard samples")
    for sample in samples:
        strict_sample(sample)
    assert baseline.adapter_shared_mib is not None and baseline.process_shared_mib is not None
    def values(name: str) -> list[float]:
        return [float(getattr(sample, name)) for sample in samples]
    return {"whole_card_peak_mib": max(values("card_mib")),
            "adapter_shared_rise_mib": max(0., max(values("adapter_shared_mib")) - float(baseline.adapter_shared_mib)),
            "process_shared_rise_mib": max(0., max(values("process_shared_mib")) - float(baseline.process_shared_mib)),
            "process_ram_peak_gib": max(values("tree_gib")),
            "minimum_free_ram_gib": min(values("available_gib")),
            "minimum_commit_headroom_gib": min(values("commit_headroom_gib"))}


def deadline_reason(now: float, started: float, phase: dict[str, Any] | None,
                    phase_started: float) -> str | None:
    if now - started >= SESSION_SECONDS:
        return "session wall-time cap (1800 seconds)"
    if phase is not None and now - phase_started >= BUDGETS[phase["kind"]]:
        return f"{phase['kind']} wall-time cap ({BUDGETS[phase['kind']]} seconds)"
    return None


class Watchdog:
    """Independent wall/heartbeat backstop, including a blocked sampler."""
    def __init__(self, kill: Callable[[], None], *, clock: Callable[[], float] = time.monotonic) -> None:
        self.kill = kill
        self.clock = clock
        self.started = self.last = self.phase_started = clock()
        self.phase: dict[str, Any] | None = None
        self.armed = False
        self.reason: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="Anima card watchdog")

    def pulse(self, phase: dict[str, Any] | None) -> None:
        now = self.clock()
        if phase != self.phase:
            self.phase_started = now
            self.phase = phase
        self.last = now
        self.armed = True

    def check(self) -> str | None:
        now = self.clock()
        reason = deadline_reason(now, self.started, self.phase, self.phase_started)
        if reason is None and self.armed and now - self.last > 2:
            reason = "guard sampler heartbeat absent for more than 2 seconds"
        return reason

    def _run(self) -> None:
        while not self.stop.wait(.1):
            reason = self.check()
            if reason:
                self.reason = reason
                self.kill()
                return

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=12)


def supervise(output: Path, limits: guard.Limits, *, host: Any, proc: Any,
              clock: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep, watchdog: Watchdog | None = None) -> int:
    """Guard/handshake loop, injectable for CPU failure-path qualification."""
    receipt = pack.plan()["receipt"]
    receipt["session"].update({"guard_log": "guard.jsonl", "limits_receipt": "limits.json",
                               "preflight_receipt": "preflight.json"})
    atomic_json(output / "receipt.json", receipt)
    atomic_json(output / "limits.json", dataclasses.asdict(limits))
    started = clock()
    phase: dict[str, Any] | None = None
    phase_started = started
    sequence = 0
    samples: list[guard.Sample] = []
    periods: list[float] = []
    previous = started
    baseline: guard.Baseline | None = None
    failure: str | None = None
    final: guard.Sample | None = None
    completed = False
    with (output / "guard.jsonl").open("a", encoding="utf-8") as raw:
        def row(event: str, **fields: Any) -> None:
            raw.write(json.dumps({"event": event, "elapsed": clock() - started, **fields}, allow_nan=False) + "\n")
            raw.flush()
        try:
            if getattr(host, "sources", {}).get("shared") != "pdh":
                raise RuntimeError("quarter-second protocol requires held PDH counters; typeperf fallback refused")
            pids = host.tree(proc.pid)
            if not pids:
                raise RuntimeError("waiting worker exited before guard admission")
            first = host.sample(pids)
            baseline = admission(first, limits)
            atomic_json(output / "preflight.json", {"sample": dataclasses.asdict(first),
                         "baseline": dataclasses.asdict(baseline), "sources": getattr(host, "sources", {})})
            row("baseline", sample=dataclasses.asdict(first), limits=dataclasses.asdict(limits))
            if watchdog is not None:
                watchdog.pulse(None)
            while True:
                now = clock()
                if watchdog is not None and watchdog.reason:
                    raise RuntimeError(watchdog.reason)
                reason = deadline_reason(now, started, phase, phase_started)
                if reason:
                    raise RuntimeError(reason)
                if proc.poll() is not None:
                    if completed and proc.poll() == 0:
                        break
                    raise RuntimeError(f"worker exited unexpectedly ({proc.poll()})")
                pids = host.tree(proc.pid)
                if not pids:
                    if completed and proc.poll() == 0:
                        break
                    raise RuntimeError(f"worker exited unexpectedly ({proc.poll()})")
                sample = host.sample(pids)
                if watchdog is not None and watchdog.reason:
                    raise RuntimeError(watchdog.reason)
                final = sample
                strict_sample(sample)
                answer = guard.decide(sample, baseline, limits)
                row("sample", phase=phase, sample=dataclasses.asdict(sample), answer=dataclasses.asdict(answer))
                periods.append(now - previous)
                previous = now
                if answer.verdict != "ok":
                    raise RuntimeError(f"guard {answer.why}: {answer.reading} crosses {answer.limit}")
                if phase is not None:
                    samples.append(sample)
                    result = (read_json(message_path(output, "failure", sequence)) or
                              read_json(message_path(output, "result", sequence)))
                    if result is not None and result.get("sequence") == sequence:
                        result = {**result, "host_metrics": host_metrics(samples, baseline)}
                        atomic_json(output / f"{phase['id']}-receipt.json", result)
                        if phase["kind"] == "load":
                            receipt["load_count"] = 1
                            receipt["session"].update(result.get("session", {}))
                            receipt["session"]["load_receipt"] = "load-receipt.json"
                        elif phase["kind"] == "case":
                            target = next(item for item in receipt["cases"] if item["id"] == phase["id"])
                            target.update(result.get("case", {}))
                            target["metrics"].update(result["host_metrics"])
                            target["guard_ok"] = result.get("ok") is True
                        else:
                            receipt["session"]["final_residency_receipt"] = "cleanup-receipt.json"
                            completed = result.get("ok") is True
                        atomic_json(output / "receipt.json", receipt)
                        if result.get("ok") is not True:
                            raise RuntimeError(f"{phase['id']}: {result.get('error', 'worker failure')}")
                        publish_message(output, "ack", sequence, {"sequence": sequence, "action": "complete"}, "complete")
                        phase = None
                if phase is None and not completed:
                    request = read_json(message_path(output, "request", sequence + 1))
                    if request is not None and request.get("sequence") == sequence + 1:
                        if request.get("kind") not in BUDGETS:
                            raise RuntimeError("worker requested an unknown phase")
                        sequence += 1
                        phase = request
                        phase_started = clock()
                        samples = [sample]
                        if request["id"] == "upscale-125":
                            points = [[guard.rung_x(1024, 1024), max(
                                item["metrics"]["whole_card_peak_mib"] for item in receipt["cases"]
                                if item["status"] == "completed")]]
                            projection = guard.plan_rung(points, guard.rung_x(1280, 1280), limits)
                            row("upscale_projection", answer=dataclasses.asdict(projection))
                            # Every preceding case is the same area. One
                            # point has no slope: plan_rung's permissive
                            # single-point answer is not a projection.
                            if projection.verdict != "ok" or projection.projected_mib is None:
                                target = receipt["cases"][-1]
                                target.update(status="skipped", reason=projection.reason or
                                              "only one measured area; no defensible enlargement projection")
                                atomic_json(output / "receipt.json", receipt)
                                publish_message(output, "ack", sequence, {"sequence": sequence, "action": "skip"}, "skip")
                                phase = None
                                continue
                        publish_message(output, "ack", sequence, {"sequence": sequence, "action": "begin"}, "begin")
                if watchdog is not None:
                    watchdog.pulse(phase)
                sleep(max(0., PERIOD - (clock() - now)))
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            receipt["stopped_reason"] = failure
            row("stopped", reason=failure)
        finally:
            if failure is not None:
                host.kill(proc.pid)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                host.kill(proc.pid)
                failure = failure or "worker did not exit after cleanup"
                receipt["stopped_reason"] = failure
            # Final whole-card/shared/host reading after the child exits; it
            # cannot measure the extinct process's allocator or working set.
            try:
                final = host.sample([])
                assert final is not None
                for name in ("card_mib", "card_total_mib", "adapter_shared_mib", "available_gib", "commit_headroom_gib"):
                    value = getattr(final, name)
                    if value is None or not math.isfinite(value) or value < 0:
                        raise RuntimeError(f"final required counter unavailable: {name}")
                atomic_json(output / "after-exit.json", dataclasses.asdict(final))
            except Exception as exc:
                failure = failure or f"final counter read failed: {exc}"
                receipt["stopped_reason"] = failure
            atomic_json(output / "receipt.json", receipt)
            row("summary", stopped_reason=failure, complete=completed and failure is None,
                samples=len(periods), achieved_max_period_seconds=max(periods, default=0.),
                final_sample=None if final is None else dataclasses.asdict(final))
    return 0 if completed and failure is None else 1


class Handshake:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.sequence = 0

    def wait(self, action: str) -> str:
        start = time.monotonic()
        while time.monotonic() - start < 30:
            ack = read_json(message_path(self.output, "ack", self.sequence, action))
            if ack == {"sequence": self.sequence, "action": action}:
                return action
            skip = read_json(message_path(self.output, "ack", self.sequence, "skip")) if action == "begin" else None
            if skip == {"sequence": self.sequence, "action": "skip"}:
                return "skip"
            time.sleep(.025)
        raise RuntimeError(f"guard handshake timed out waiting for {action}")

    def begin(self, kind: str, name: str) -> bool:
        self.sequence += 1
        publish_message(self.output, "request", self.sequence, {"sequence": self.sequence, "kind": kind, "id": name})
        return self.wait("begin") == "begin"

    def finish(self, **result: Any) -> None:
        publish_message(self.output, "failure" if result.get("ok") is False else "result", self.sequence,
                        {"sequence": self.sequence, **result})
        self.wait("complete")


def run_case(text: Any, runner: Any, spec: dict[str, Any], source: Any, *,
             torch: Any, cancelled: type[Exception], fitted: Callable[..., Any], output: Path) -> dict[str, Any]:
    """One image call; measured callback facts and loaded-owner hook cleanup."""
    callbacks: list[dict[str, Any]] = []
    def callback(pipe: Any, index: int, timestep: Any, values: Any) -> Any:
        begin = pipe.scheduler.begin_index
        callbacks.append({"completed_steps": index + 1, "timestep": float(timestep),
                          "start_sigma": float(pipe.scheduler.sigmas[begin]) if spec["strength"] is not None else None,
                          "begin_index": begin,
                          "step_index": pipe.scheduler.step_index})
        if spec.get("cancel_after_completed_steps") == index + 1:
            raise cancelled()
        return values
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    image = None
    status = "completed"
    try:
        kwargs = {"prompt": pack.PROMPT, "negative_prompt": "", "width": spec["width"],
                  "height": spec["height"], "num_inference_steps": 8, "guidance_scale": 1.,
                  "num_images_per_prompt": 1, "generator": torch.Generator(device="cpu").manual_seed(spec["seed"]),
                  "callback_on_step_end": callback}
        if spec["strength"] is not None:
            kwargs.update(image=fitted(source.copy(), spec["width"], spec["height"]), strength=spec["strength"])
        image = (text if spec["strength"] is None else runner)(**kwargs).images[0].convert("RGB")
        if "cancel_after_completed_steps" in spec:
            raise RuntimeError("cancel callback returned a decoded image")
    except cancelled:
        if "cancel_after_completed_steps" not in spec:
            raise
        status = "cancelled"
    finally:
        text.maybe_free_model_hooks()
        torch.cuda.synchronize()
    metrics = {"elapsed_seconds": time.monotonic() - started,
               "allocated_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
               "reserved_peak_mib": torch.cuda.max_memory_reserved() / 2**20,
               "resident_allocated_mib": torch.cuda.memory_allocated() / 2**20}
    result: dict[str, Any] = {"status": status, "metrics": metrics, "completed_steps": len(callbacks),
                              "start_sigma": callbacks[0]["start_sigma"] if callbacks and spec["strength"] is not None else None,
                              "source_sha256": hashlib.sha256(source.tobytes()).hexdigest() if source is not None else None,
                              "pixel_sha256": None, "png_sha256": None, "metadata_sha256": None}
    if image is not None:
        png = output / (spec["id"] + ".png")
        image.save(png)
        metadata = output / (spec["id"] + ".json")
        atomic_json(metadata, {"request": spec, "settings": pack.SETTINGS, "callbacks": callbacks, "result": result})
        result.update(pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                      png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                      metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
    return {"case": result, "image": image}


@contextmanager
def worker_hold(output: Path, *, inhibitor: Any = None) -> Iterator[Any]:
    """Require the CLI power primitive, with no app/controller provider polling."""
    from backend import keep_awake

    reason = "AuraDream private Anima Refine card qualification"
    holder = None
    state: dict[str, Any] = {"reason": reason, "kind": None, "holding": False,
                             "acquired": False, "released": False, "failure": None, "scope_error": None}
    try:
        holder = inhibitor if inhibitor is not None else keep_awake.platform_inhibitor(reason)
        with keep_awake.hold_while(reason, inhibitor=holder) as active:
            state.update(kind=active.kind, holding=active.held, acquired=active.held)
            atomic_json(output / "keep-awake.json", state)
            if active.kind != "power-request" or not active.held:
                state["failure"] = "required Windows power request was not held; see worker.log for holder refusal"
                raise RuntimeError(state["failure"])
            yield active
    except BaseException as exc:
        state["scope_error"] = f"{type(exc).__name__}: {exc}"
        if holder is None:
            state["failure"] = state["scope_error"]
        raise
    finally:
        # hold_while has already requested release, including when the body
        # raised. A forced child exit also closes Windows' process handle.
        if holder is not None:
            state.update(holding=holder.held, released=not holder.held)
            if holder.held:
                state["failure"] = "power request remained held after release"
        atomic_json(output / "keep-awake.json", state)
        if state["holding"] and sys.exc_info()[0] is None:
            raise RuntimeError(state["failure"])


def _worker_session(output: Path, include_upscale: bool, hand: Handshake) -> int:
    backend = text = runner = None
    try:
        # First import follows successful mandatory-counter and resource admission.
        from backend import config as config  # noqa: F401
        from backend import refine
        from backend.backends import get_backend
        from backend.backends import anima_loading
        from backend.backends.anima_pipeline import AnimaTurboPipeline
        from backend.backends.anima_refine import AnimaTurboImg2ImgPipeline
        from backend.backends.base import Cancelled
        import importlib.metadata
        import torch
        if sys.platform != "win32" or not torch.cuda.is_available():
            raise RuntimeError("the frozen session requires Windows CUDA")
        gpu = torch.cuda.get_device_name(0)
        if "RTX 5080" not in gpu or importlib.metadata.version("diffusers") != "0.42.1":
            raise RuntimeError("the frozen session requires RTX 5080 and Diffusers 0.42.1")
        cached = anima_loading.converted_tree(pack.CHECKPOINT)
        if cached is None:
            raise RuntimeError("pinned converted Turbo tree is not cached; offline attempt refuses conversion/download")
        backend = get_backend("diffusers")
        utc_started = datetime.now(timezone.utc).isoformat()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        backend.load()
        torch.cuda.synchronize()
        text = getattr(backend, "_pipe", None)
        if type(text) is not AnimaTurboPipeline:
            raise RuntimeError("load did not yield the exact Turbo pipeline")
        runner = refine.build(text, AnimaTurboImg2ImgPipeline)
        text.set_progress_bar_config(disable=True)
        runner.set_progress_bar_config(disable=True)
        session = {"git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "utc_started": utc_started, "checkpoint_revision": anima_loading.SOURCE["revision"],
                   "loader_recipe": "Windows fp8/model, CUDA, Eager/native, CPU generators; offline cached tree",
                   "gpu_name": gpu, "driver": subprocess.check_output(
                       ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=10).strip(),
                   "torch_version": torch.__version__, "diffusers_version": importlib.metadata.version("diffusers"),
                   "torchao_version": importlib.metadata.version("torchao")}
        hand.finish(ok=True, session=session, elapsed_seconds=time.monotonic() - started, cached_tree=str(cached),
                    allocated_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                    reserved_peak_mib=torch.cuda.max_memory_reserved() / 2**20,
                    resident_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    source_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (Path(__file__).resolve(), ROOT / "backend/backends/anima_refine.py",
                                                ROOT / "backend/backends/anima_pipeline.py", ROOT / "tools/card_guard.py")})
        source = None
        hashes: dict[str, str] = {}
        for spec in pack.cases():
            if spec.get("optional") and not include_upscale:
                # The parent creates this recorded skip after successful cleanup.
                continue
            if not hand.begin("case", spec["id"]):
                continue
            result = run_case(text, runner, spec, source, torch=torch, cancelled=Cancelled,
                              fitted=refine.fitted, output=output)
            case = result["case"]
            if spec["id"] == "baseline":
                source = result["image"].copy()
            if case["pixel_sha256"]:
                hashes[spec["id"]] = case["pixel_sha256"]
            compare = spec.get("compare_to")
            if compare and hashes.get(compare) != case["pixel_sha256"]:
                raise RuntimeError(f"decoded pixels differ from {compare}")
            expected = 1 if "cancel_after_completed_steps" in spec else (
                8 if spec["strength"] is None else {0.125: 1, 0.25: 2, 0.4: 4}[spec["strength"]])
            if case["completed_steps"] != expected:
                raise RuntimeError("actual callback count differs from the frozen sequence")
            if spec["strength"] is not None and not math.isclose(
                    case["start_sigma"], {0.125: .3, 0.25: .5, 0.4: .75}[spec["strength"]], abs_tol=1e-6):
                raise RuntimeError("actual first-step sigma differs from the frozen sequence")
            hand.finish(ok=True, case=case)
        hand.begin("cleanup", "cleanup")
        runner = text = None
        backend.unload()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        hand.finish(ok=True, resident_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                    resident_reserved_mib=torch.cuda.memory_reserved() / 2**20)
        return 0
    except BaseException:
        # The outer worker scope releases its power request before reporting
        # failure to the parent, whose next action may kill the process.
        raise


def worker(output: Path, include_upscale: bool) -> int:
    hand = Handshake(output)
    hand.begin("load", "load")
    try:
        with worker_hold(output):
            return _worker_session(output, include_upscale, hand)
    except BaseException as exc:
        hand.finish(ok=False, error=f"{type(exc).__name__}: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory; refuses an existing path")
    parser.add_argument("--need-ram-gib", type=float, required=True, help="explicit conservative load-RAM admission")
    parser.add_argument("--busy-card-mib", type=float, required=True, help="maximum admitted idle-card baseline")
    parser.add_argument("--include-upscale", action="store_true", help="try only the frozen optional 1280-square tail")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.worker:
        return worker(output, args.include_upscale)
    if sys.platform != "win32":
        parser.error("this private frozen protocol is Windows only")
    limits = guard.resolve_limits(None, {"need_ram_gib": args.need_ram_gib, "busy_card_mib": args.busy_card_mib})
    if limits.need_ram_gib is None or limits.need_ram_gib <= limits.ram_floor_gib:
        parser.error("load RAM admission must exceed the unchanged free-RAM floor")
    if os.getenv("DIFFUSERS_ATTN_BACKEND", "native") != "native":
        parser.error("clear the unrecorded DIFFUSERS_ATTN_BACKEND override first")
    if any(name in os.environ for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")):
        parser.error("clear CUDA visibility/order overrides so torch cuda:0 is the guarded physical GPU 0")
    output.mkdir(parents=True, exist_ok=False)
    job = guard.KillOnCloseJob.open()
    if job is None:
        raise RuntimeError("a kill-on-close Windows job is required before starting the worker")
    host = guard.LocalHost(PERIOD)
    try:
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output),
                       "--need-ram-gib", str(args.need_ram_gib), "--busy-card-mib", str(args.busy_card_mib)]
            if args.include_upscale:
                command.append("--include-upscale")
            proc = subprocess.Popen(command, cwd=ROOT, env=environment(output), stdout=log,
                                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            if not job.assign(proc.pid):
                host.kill(proc.pid)
                raise RuntimeError("could not bind waiting worker to the kill-on-close job")
            watchdog = Watchdog(lambda: host.kill(proc.pid))
            watchdog.thread.start()
            try:
                code = supervise(output, limits, host=host, proc=proc, watchdog=watchdog)
            finally:
                watchdog.close()
        if code == 0 and not args.include_upscale:
            receipt = read_json(output / "receipt.json")
            assert receipt is not None
            receipt["cases"][-1].update(status="skipped", reason="optional enlargement not selected for this attempt")
            atomic_json(output / "receipt.json", receipt)
        print(json.dumps({"exit": code, "receipt": str(output / "receipt.json"),
                          "qualification": "not decided; review images and raw receipts"}))
        return code
    finally:
        host.close()
        job.close()


if __name__ == "__main__":
    raise SystemExit(main())
